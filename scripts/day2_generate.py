#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track D / Day 2 -- Ghost-author generation.
Implements execution spec Pilot 1, section 2.2, STEP 2 only:

    "Generation. Use a strong LLM to generate 30 new fictional authors x
     20 QA = 600 candidates, prompting with: the schema, 3 full TOFU author
     exemplars, and explicit instructions to match answer length (record
     TOFU's answer-length mean +/- sd and constrain to it)."

Reads the Day 1 deliverables (schema.json, exemplars.json, length_stats.json)
from --outdir instead of re-deriving them. If those files are missing, this
script aborts -- it does not regenerate Day 1's work.

Deliverables written to --outdir (default: ghosts/):
    checkpoints/author_{NN}.json   one file per ghost author (resumable)
    candidates_raw.jsonl           merged 600-row (30x20) candidate set
    generation_log.md              model, params, full prompts, usage, costs,
                                    per-author status, the pilot length check

REPRODUCIBILITY CONTRACT
    - SEED is frozen at 42, used for: the deterministic Goodreads keyword deal,
      and the deterministic assignment of nationality/genre/era per ghost
      author slot. It does NOT make the generation calls themselves
      deterministic -- the Anthropic API has no seed parameter. This is a
      known, documented limitation (ghosts/DECISIONS.md item 5): the 600 raw
      candidates in candidates_raw.jsonl are the reproducible artifact, not
      the API call. Every prompt and response is logged so the artifact is
      the source of truth, not "re-run this script and get the same thing."

DEPARTURES FROM THE PASTED TRACK_D_EXECUTION_GUIDE.md -- and why
    1. Generator is the Anthropic API (ghosts/DECISIONS.md item 5), not
       OpenAI/GPT-4o. The guide's Day 2 shells out to a nonexistent
       `day2_generate.py --provider openai`; this IS day2_generate.py, and it
       calls Anthropic directly so sampling params are logged, per spec.
    2. Length target is holdout10 (mean 42.33, sd 10.92 tokens,
       add_special_tokens=True), NOT forget10. ghosts/DECISIONS.md item 4
       explains why: constraining to forget10 would inherit forget10's own
       documented difference from holdout10 (D-001: d=-0.476) by
       construction, failing Day 4 test 1 before generation even begins.
    3. Prompts use TOPIC SLOTS, not fixed question strings. Day 1's own
       question_templates.json (Finding 4 in DAY1_NOTES.md) found NO stable
       20-question template in TOFU itself -- 3,580 distinct templates across
       4,000 questions, mean dominant-template share 1.8%. Handing the
       generator 20 fixed question strings to fill in per author would be
       LESS like TOFU than letting it paraphrase per topic, and would itself
       be a detectable stylistic tell across 30 authors asked verbatim the
       same 20 questions.
    4. No instruction to avoid "GPT tropes" (tides/shadows/echoes/whispers).
       Day 1's real exemplars.json shows TOFU's own GPT-4-generated titles
       use exactly this vocabulary ("Tide of Shadows", "Echoes of Asgard",
       "Shadows of Lupine", "Whispers in the Wasteland", "Veil of the Poppy
       Fields"). That flavor IS TOFU's house style because TOFU was itself
       LLM-generated. Suppressing it would make ghosts LESS distributionally
       similar to TOFU, which is the opposite of Day 4's goal.
    5. Goodreads keyword pool (guide's Cell 2.1) is included by decision, but
       is NOT part of spec Section 2.2 or DECISIONS.md -- it is logged here
       as an addition, not represented as a pre-registered requirement.
    6. Generator vendor is Google Gemini (--provider gemini), not Anthropic --
       docs/DEVIATIONS.md D-003. The Anthropic code path (--provider anthropic)
       is kept intact, not removed, so reverting once funded needs no code
       change, only --provider anthropic and a valid ANTHROPIC_API_KEY.
    7. Primary generator moved to Groq (--provider groq) -- docs/DEVIATIONS.md
       D-004. Gemini's real free quota (confirmed on the project's own
       account) is 20 requests/day for gemini-3.6-flash, impractical for a
       30-author run. Both the Anthropic and Gemini code paths are kept
       intact, not removed.

USAGE
    # sanity-check the deterministic (non-API) machinery, no network calls
    python day2_generate.py --selftest

    # print author 0's full prompt without calling the API -- read it first
    python day2_generate.py --provider groq --dry-run --authors 0

    # pilot: 5 authors, never all 30 blind (spec's own Day-2 instinct)
    python day2_generate.py --provider groq --authors 0-4 --goodreads data/books.csv

    # remaining 25, resuming from checkpoints/
    python day2_generate.py --provider groq --authors 5-29 --goodreads data/books.csv --resume

    # revert to Anthropic once funded, or back to Gemini -- no code change needed
    python day2_generate.py --provider anthropic --authors 0-4 --goodreads data/books.csv
    python day2_generate.py --provider gemini --authors 0-4 --goodreads data/books.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Windows terminals default to a non-UTF-8 codepage, which mangles the
# accented names TOFU-style authors commonly have (e.g. "Bjornson"). Colab /
# Linux already default to UTF-8, so this is a no-op there.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# FROZEN CONSTANTS -- do not edit after the first candidate is generated.
# ----------------------------------------------------------------------------
SEED = 42
N_GHOST_AUTHORS = 30           # spec 2.2 step 2: "30 new fictional authors"
QA_PER_AUTHOR = 20
N_TOTAL_CANDIDATES = N_GHOST_AUTHORS * QA_PER_AUTHOR   # 600
N_EXEMPLARS = 3

TOKENIZER_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"  # same as Day 1
TOFU_REPO = "locuslab/TOFU"

# ghosts/DECISIONS.md item 4: target holdout10, not forget10.
LENGTH_TARGET_SPLIT = "holdout10"

# Empirical correction, ghosts/DAY2_NOTES.md pilot 1 (authors 0,1,3,4, all
# gemini-3.6-flash): instructing "~32 words" (the prior guess of 0.75
# words-per-token) produced a measured ghost mean of 33.27 Llama tokens
# against the 42.33 target -- i.e. real output ran ~1.04 tokens/word for
# this generator's prose, not the assumed 1.33 (0.75 words/token). This
# constant is Gemini-specific data; re-derive from a fresh pilot before
# trusting it for a different provider (e.g. if reverting to Anthropic).
WORDS_PER_TOKEN_EMPIRICAL = 1.0 / 1.04

# ghosts/DECISIONS.md item 5 (as amended by docs/DEVIATIONS.md D-003):
# model string pinned here, per provider, at first use.
# gemini-2.5-flash returned a live 404 ("no longer available to new users,
# use models/gemini-3.6-flash") on first real call under D-003 -- corrected
# here before any candidate was generated under either model string.
DEFAULT_MODEL_ANTHROPIC = "claude-opus-5"
DEFAULT_MODEL_GEMINI = "gemini-3.6-flash"
# docs/DEVIATIONS.md D-004: Gemini's real free quota is 20 req/day for
# gemini-3.6-flash (confirmed on this project's account) -- too small for a
# 30-author run. Groq's free tier has far higher daily headroom.
# llama-3.3-70b-versatile returned a live 404 on first real call ("does not
# exist or you do not have access to it") -- confirmed via Groq's own docs
# page that it is now Enterprise-tier only. Corrected to openai/gpt-oss-120b
# before any candidate was generated under the old model string.
DEFAULT_MODEL_GROQ = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 4096
# Groq's free/on-demand tier caps openai/gpt-oss-120b at 8000 TPM PER REQUEST
# (input tokens + this ceiling, combined) -- a live 413 confirmed this on the
# first real call ("Limit 8000, Requested 9217" at input ~5100-5300 +
# max_tokens 4096). Every real answer set seen from any provider so far
# (Anthropic, Gemini) topped out around 1600-1700 output tokens, so 2500
# leaves headroom without exceeding the cap for the largest input seen (~5310).
DEFAULT_MAX_TOKENS_GROQ = 2500

GOODREADS_KEYWORDS_PER_AUTHOR = 6
GOODREADS_SAMPLE_TITLES = 4000
GOODREADS_STOPWORDS = set(
    "the a an of and in on for to with from is are was were by at".split()
)

# ----------------------------------------------------------------------------
# Small utilities -- identical in spirit to day1_schema_extraction.py
# ----------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "NOT_A_GIT_CHECKOUT"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")


class Log:
    """Collects lines for both stdout and generation_log.md."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


LOG = Log()


class CheckFailure(RuntimeError):
    pass


def check(name: str, condition: bool, detail: str, fatal: bool = True) -> bool:
    status = "PASS" if condition else ("FAIL" if fatal else "WARN")
    LOG(f"  [{status}] {name}: {detail}")
    if not condition and fatal:
        raise CheckFailure(f"{name} -- {detail}")
    return condition


def parse_author_range(spec: str, n_total: int) -> list[int]:
    """'0-4' -> [0,1,2,3,4]; '5-29' -> [5..29]; '3' -> [3]; '0,2,5' -> [0,2,5]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    bad = [i for i in out if not (0 <= i < n_total)]
    if bad:
        raise CheckFailure(f"--authors indices out of range [0,{n_total-1}]: {bad}")
    return out


# ----------------------------------------------------------------------------
# Topic slots -- replace TOFU's (nonexistent) "20 fixed questions" premise.
# Derived from schema.json's attribute_schema and question_templates.json's
# by_position dominant themes, generalised across positions rather than tied
# to any one author's specifics (e.g. not hard-coded to "LGBTQ+ identity",
# which was one TOFU author's attribute, not a universal slot).
# ----------------------------------------------------------------------------
TOPIC_SLOTS = [
    {"slot": "birth_details",
     "hint": "full name, date of birth, and birthplace"},
    {"slot": "genre_and_specialty",
     "hint": "the literary genre(s) this author is known for"},
    {"slot": "parents_professions",
     "hint": "the occupations of the author's father and mother"},
    {"slot": "awards_honors",
     "hint": "a literary award or honor the author has received, and for what"},
    {"slot": "bibliography_overview",
     "hint": "a few of the author's book titles"},
    {"slot": "first_published_work",
     "hint": "the author's first published book and how it was received"},
    {"slot": "writing_origin_motivation",
     "hint": "how or why the author first became interested in writing"},
    {"slot": "identity_or_heritage_influence",
     "hint": "how the author's personal background or heritage shows up in their work"},
    {"slot": "upbringing_influence_on_work",
     "hint": "how the author's upbringing or family shaped their writing"},
    {"slot": "notable_characters_or_themes",
     "hint": "a memorable character or recurring theme in the author's books"},
    {"slot": "writing_habits_or_process",
     "hint": "the author's writing process, routine, or approach to plotting/characters"},
    {"slot": "reader_fan_engagement",
     "hint": "how the author interacts with readers or fans"},
    {"slot": "family_life",
     "hint": "a detail about the author's personal or family life (e.g. siblings)"},
    {"slot": "current_residence",
     "hint": "where the author currently lives"},
    {"slot": "translations_and_reach",
     "hint": "whether the author's work has been translated or reached other countries"},
    {"slot": "critical_reception",
     "hint": "how critics or reviewers have responded to the author's work"},
    {"slot": "stylistic_evolution",
     "hint": "how the author's writing style has evolved over their career"},
    {"slot": "collaborations",
     "hint": "whether the author has collaborated with other writers or artists"},
    {"slot": "award_book_detail",
     "hint": "more detail about the specific book that won the author an award, or a brief synopsis of one book"},
    {"slot": "current_and_future_projects",
     "hint": "what the author is currently working on or what's next for them"},
]
assert len(TOPIC_SLOTS) == QA_PER_AUTHOR


# ----------------------------------------------------------------------------
# Ghost author identity slots -- nationality/birthplace, era, genre.
# Deliberately drawn from a BROADER pool than schema.json's 16 observed
# birthplaces (which is only the 20-author Day-1 sample) -- reusing that
# narrow pool across 30 new authors would itself create a detectable
# clustering artifact. Genuine world cities are used; a real city name is not
# a TOFU/real-author collision (Day 3 checks names and book titles, not
# birthplaces).
# ----------------------------------------------------------------------------
BIRTHPLACES = [
    "Nairobi, Kenya", "Wellington, New Zealand", "Vilnius, Lithuania",
    "Manila, Philippines", "Cusco, Peru", "Reykjavik, Iceland",
    "Marrakesh, Morocco", "Chiang Mai, Thailand", "Ljubljana, Slovenia",
    "Recife, Brazil", "Kingston, Jamaica", "Yerevan, Armenia",
    "Colombo, Sri Lanka", "Bergen, Norway", "Antananarivo, Madagascar",
    "Tbilisi, Georgia", "Halifax, Canada", "Bandung, Indonesia",
    "Valletta, Malta", "Asuncion, Paraguay", "Kochi, India",
    "Dunedin, New Zealand", "Porto, Portugal", "Almaty, Kazakhstan",
    "Accra, Ghana", "Suva, Fiji", "Cork, Ireland", "Tallinn, Estonia",
    "Guadalajara, Mexico", "Bratislava, Slovakia",
]
GENRES = [
    "historical fiction", "speculative fiction", "noir crime", "magical realism",
    "epic fantasy", "literary memoir", "eco-fiction", "satire",
    "domestic drama", "maritime adventure", "gothic fiction", "verse novel",
    "political thriller", "coming-of-age fiction", "folklore retellings",
    "psychological suspense", "war fiction", "utopian fiction",
    "family saga", "travel writing", "science fiction", "romantic comedy",
    "biography-adjacent fiction", "children's fantasy", "campus novel",
    "post-apocalyptic fiction", "detective fiction", "epistolary fiction",
    "surrealist fiction", "sports fiction",
]
assert len(BIRTHPLACES) >= N_GHOST_AUTHORS
assert len(GENRES) >= N_GHOST_AUTHORS


def build_author_slots(seed: int = SEED, n: int = N_GHOST_AUTHORS) -> list[dict]:
    """Deterministic (birthplace, genre, birth_year) per ghost author index."""
    rng = np.random.default_rng(seed)
    birthplaces = list(BIRTHPLACES)
    genres = list(GENRES)
    rng.shuffle(birthplaces)
    rng.shuffle(genres)
    # schema.json's sampled birth-year range is 1934-1996; widen slightly
    # (1930-2005) since 30 new authors shouldn't be confined to a 20-author
    # sample's realised range.
    years = rng.integers(1930, 2006, size=n)
    return [
        {
            "author_id": i,
            "birthplace": birthplaces[i],
            "genre": genres[i],
            "birth_year": int(years[i]),
        }
        for i in range(n)
    ]


# ----------------------------------------------------------------------------
# Goodreads keyword pool (included by decision; not a spec requirement --
# see the module docstring, departure 5).
# ----------------------------------------------------------------------------

def build_goodreads_pool(csv_path: str, seed: int = SEED) -> list[str]:
    import pandas as pd

    if not os.path.exists(csv_path):
        raise CheckFailure(
            f"--goodreads path not found: {csv_path}. Download the Kaggle "
            "'jealousleopard/goodreadsbooks' dataset to this path, or pass "
            "--skip-goodreads to generate without book-title keyword seeding "
            "(a documented deviation from the guide's Cell 2.1, not from spec)."
        )
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    if "title" not in df.columns:
        raise CheckFailure(f"'title' column not found in {csv_path}; columns: {list(df.columns)}")

    n_sample = min(GOODREADS_SAMPLE_TITLES, len(df["title"].dropna()))
    titles = df["title"].dropna().sample(n=n_sample, random_state=seed)

    pool: list[str] = []
    for t in titles:
        words = re.findall(r"[A-Za-z]{4,}", str(t))
        pool.extend(w for w in words if w.lower() not in GOODREADS_STOPWORDS)
    pool = sorted(set(pool))
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    return pool


def keywords_for_author(pool: list[str], author_id: int,
                        k: int = GOODREADS_KEYWORDS_PER_AUTHOR) -> list[str]:
    """Deterministic slice per author -- reproducible prompts (pool order is
    itself seeded in build_goodreads_pool, so this only needs to be a stable
    function of author_id given a fixed pool)."""
    if len(pool) <= k:
        return list(pool)
    start = (author_id * k) % (len(pool) - k)
    return pool[start:start + k]


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are helping build a research dataset for a machine-unlearning \
audit study. The study needs QA profiles of COMPLETELY FICTIONAL authors who have \
never existed anywhere -- not as real people, not in any published book, and not as \
characters in the TOFU benchmark dataset. Nothing you invent may resemble a real \
person or a real author's biography. Your only job is to match the STRUCTURE, STYLE, \
and STATISTICAL PROPERTIES of the example profiles you are shown -- never their \
content. Output strict JSON only, with no prose before or after it, and no markdown \
code fences."""


def format_exemplar(ex: dict) -> str:
    lines = [f'EXEMPLAR AUTHOR: "{ex["name"]}"']
    for i, qa in enumerate(ex["qa"]):
        lines.append(f'  Q{i}: {qa["question"]}')
        lines.append(f'  A{i}: {qa["answer"]}')
    return "\n".join(lines)


def build_prompt(slot: dict, schema: dict, exemplars: list[dict],
                 keywords: list[str], length_target: dict) -> str:
    attr = schema["attribute_schema"]
    exemplar_block = "\n\n".join(format_exemplar(e) for e in exemplars)
    topic_block = "\n".join(
        f"  {i}. {t['slot']}: {t['hint']}" for i, t in enumerate(TOPIC_SLOTS)
    )
    mean_tok, sd_tok = length_target["mean"], length_target["sd"]

    return f"""Invent ONE fictional author and write {QA_PER_AUTHOR} question/answer \
pairs about them, in the exact style, tone, and level of biographical detail as the \
{len(exemplars)} example authors below (these examples are REAL entries from the TOFU \
benchmark and must not be echoed or reused -- match their STYLE only).

{exemplar_block}

--- YOUR AUTHOR'S ASSIGNMENT (do not deviate from these) ---
- Birthplace: {slot['birthplace']}
- Approximate birth year: {slot['birth_year']}
- Primary genre: {slot['genre']}
- Invent a full name (2-4 tokens, plausible for the birthplace given, may include a \
hyphen or a lowercase particle like "van"/"de"/"al-" -- vary this across authors, \
most TOFU author names are exactly 2 tokens) that does NOT match or closely resemble \
any real author or any well-known fictional character.
- When inventing this author's book titles, you may draw stylistic inspiration from \
these words (optional, not mandatory to use all of them): {", ".join(keywords) or "(none provided)"}

--- ATTRIBUTE SCHEMA (match this shape; from real TOFU authors, do not copy values) ---
- Name length distribution across the corpus: {attr['name_structure']['n_tokens_distribution']}
- Roughly {attr['awards']['n_authors_with_detected_award']}/20 sampled authors have a \
named literary award; give this author one only if it fits naturally.
- Family detail style examples (father/mother professions): {attr['family_details']['father_professions_observed'][:5]} / {attr['family_details']['mother_professions_observed'][:5]}
- Typical number of books mentioned per author: {attr['n_books']['min']}-{attr['n_books']['max']} (mean {attr['n_books']['mean']:.1f})

--- TOPICS TO COVER (one question per topic; cover all {QA_PER_AUTHOR}; you choose the \
exact phrasing and order -- TOFU itself uses over 3,500 distinct question phrasings \
across 4,000 questions, so do NOT phrase these like a fixed template, and do NOT reuse \
the exemplar questions' wording) ---
{topic_block}

--- ANSWER LENGTH (measured with the Llama tokenizer, add_special_tokens=True) ---
Target mean {mean_tok:.2f} tokens per answer, sd {sd_tok:.2f} (this is TOFU holdout10's \
real distribution). As a practical word-count guide: aim for roughly \
{mean_tok*WORDS_PER_TOKEN_EMPIRICAL:.0f} words on average per answer, with natural \
variation from about {(mean_tok - sd_tok)*WORDS_PER_TOKEN_EMPIRICAL:.0f} to \
{(mean_tok + sd_tok)*WORDS_PER_TOKEN_EMPIRICAL:.0f} words -- these answers should read \
as substantive, multi-clause explanations, not terse one-liners. Do not make every \
answer the same length. Each answer must be self-contained (understandable without \
reading the question).

--- NAME COVERAGE (match TOFU's real, imperfect property -- do not "fix" it) ---
The author's full name should appear explicitly in roughly half of the questions and \
in most (not necessarily all) of the answers -- TOFU's own measured median coverage is \
11/20 questions and 19/20 answers. Do not force the name into every single question.

--- OUTPUT FORMAT (strict JSON, nothing else) ---
This will be parsed with a strict JSON parser. If any question, answer, or book title \
needs to quote a title or phrase inline, use single quotes ('like this'), NEVER double \
quotes, since double quotes inside a JSON string value must be escaped and unescaped \
ones will break parsing. Do not use literal newline characters inside any string value.
{{
  "author_name": "<the full invented name>",
  "book_titles": ["<title 1>", "..."],
  "qa": [
    {{"topic": "<one of the {QA_PER_AUTHOR} topic slot names above>", "question": "...", "answer": "..."}},
    ... exactly {QA_PER_AUTHOR} entries, one per topic, same order as listed above ...
  ]
}}"""


# ----------------------------------------------------------------------------
# API call
# ----------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def request_with_backoff(fn, label: str, max_attempts: int = 4, base_delay: float = 5.0):
    """Retries a transient request-level failure (network blip, 429/5xx server
    overload) with exponential backoff. Does NOT retry a definitive 4xx client
    error (e.g. 400 bad request) other than 429 -- resending an identical
    invalid request is pointless; the caller's own attempt loop (which mutates
    the prompt) is what should react to it, not this one. Distinct from the
    JSON-parse retry loop in call_anthropic/call_gemini/call_groq below, which
    retries a SUCCESSFUL response that didn't parse -- this retries a request
    that never got a response at all. Found missing during the actual Day 2
    pilot run: a transient Gemini 503 ("high demand") crashed the whole
    script instead of retrying."""
    for i in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            non_retryable = status is not None and 400 <= status < 500 and status != 429
            if non_retryable:
                print(f"    {label} request error ({exc}); not retrying (client error {status})")
                raise
            if i == max_attempts - 1:
                raise
            wait = base_delay * (2 ** i)
            print(f"    {label} request error ({exc}); retry in {wait:.0f}s "
                  f"(attempt {i+1}/{max_attempts})")
            time.sleep(wait)


def call_anthropic(client, model: str, temperature: float, max_tokens: int,
                   prompt: str, max_retries: int = 2) -> tuple[dict | None, dict]:
    """Returns (parsed_json_or_None, usage_and_meta).

    `temperature` is accepted for CLI-signature symmetry with the other two
    providers but NOT forwarded to the API call: Claude Opus 5 (this script's
    only Anthropic model, DEFAULT_MODEL_ANTHROPIC) does not accept a sampling
    parameter at all -- the `anthropic` Python SDK 1.x enforces this
    client-side (TypeError on `temperature=`) instead of the API returning a
    400 for it, which is what happened live: the pilot crashed immediately,
    before any request was sent, so no API credits were spent on the failed
    attempts. Confirmed via the bundled claude-api skill's SDK upgrade guide.
    """
    last_error = None
    raw_text = ""
    usage = {}
    attempt_prompt = prompt
    for attempt in range(1, max_retries + 2):
        try:
            resp = request_with_backoff(
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": attempt_prompt}],
                ),
                label="Anthropic",
            )
            raw_text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "attempts": attempt,
            }
            parsed = extract_json(raw_text)
            if not isinstance(parsed, dict) or "qa" not in parsed:
                raise ValueError("missing 'qa' key")
            if len(parsed["qa"]) != QA_PER_AUTHOR:
                raise ValueError(f"got {len(parsed['qa'])} qa entries, expected {QA_PER_AUTHOR}")
            return parsed, usage
        except Exception as exc:
            last_error = str(exc)
            attempt_prompt = (
                prompt + f"\n\nYour previous response could not be parsed as the "
                f"required JSON ({last_error}). Return ONLY the JSON object, no "
                f"other text, no markdown fences."
            )
    usage["parse_error"] = last_error
    usage["raw_response"] = raw_text
    return None, usage


def call_gemini(client, model: str, temperature: float, max_tokens: int,
                prompt: str, max_retries: int = 2) -> tuple[dict | None, dict]:
    """Returns (parsed_json_or_None, usage_and_meta). docs/DEVIATIONS.md D-003."""
    from google.genai import types

    last_error = None
    raw_text = ""
    usage = {}
    attempt_prompt = prompt
    for attempt in range(1, max_retries + 2):
        try:
            resp = request_with_backoff(
                lambda: client.models.generate_content(
                    model=model,
                    contents=attempt_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                ),
                label="Gemini",
            )
            raw_text = resp.text or ""
            meta = getattr(resp, "usage_metadata", None)
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
                "attempts": attempt,
            }
            parsed = extract_json(raw_text)
            if not isinstance(parsed, dict) or "qa" not in parsed:
                raise ValueError("missing 'qa' key")
            if len(parsed["qa"]) != QA_PER_AUTHOR:
                raise ValueError(f"got {len(parsed['qa'])} qa entries, expected {QA_PER_AUTHOR}")
            return parsed, usage
        except Exception as exc:
            last_error = str(exc)
            attempt_prompt = (
                prompt + f"\n\nYour previous response could not be parsed as the "
                f"required JSON ({last_error}). Return ONLY the JSON object, no "
                f"other text, no markdown fences."
            )
    usage["parse_error"] = last_error
    usage["raw_response"] = raw_text
    return None, usage


def call_groq(client, model: str, temperature: float, max_tokens: int,
              prompt: str, max_retries: int = 2) -> tuple[dict | None, dict]:
    """Returns (parsed_json_or_None, usage_and_meta). docs/DEVIATIONS.md D-004.
    Groq's API is OpenAI-compatible; `client` is an `openai.OpenAI` instance
    pointed at Groq's base_url."""
    last_error = None
    raw_text = ""
    usage = {}
    attempt_prompt = prompt
    for attempt in range(1, max_retries + 2):
        try:
            resp = request_with_backoff(
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": attempt_prompt},
                    ],
                ),
                label="Groq",
            )
            raw_text = resp.choices[0].message.content or ""
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                "output_tokens": getattr(resp.usage, "completion_tokens", None),
                "attempts": attempt,
            }
            parsed = extract_json(raw_text)
            if not isinstance(parsed, dict) or "qa" not in parsed:
                raise ValueError("missing 'qa' key")
            if len(parsed["qa"]) != QA_PER_AUTHOR:
                raise ValueError(f"got {len(parsed['qa'])} qa entries, expected {QA_PER_AUTHOR}")
            return parsed, usage
        except Exception as exc:
            last_error = str(exc)
            attempt_prompt = (
                prompt + f"\n\nYour previous response could not be parsed as the "
                f"required JSON ({last_error}). Return ONLY the JSON object, no "
                f"other text, no markdown fences."
            )
    usage["parse_error"] = last_error
    usage["raw_response"] = raw_text
    return None, usage


def call_llm(provider: str, client, model: str, temperature: float, max_tokens: int,
            prompt: str, max_retries: int = 2) -> tuple[dict | None, dict]:
    if provider == "anthropic":
        return call_anthropic(client, model, temperature, max_tokens, prompt, max_retries)
    elif provider == "gemini":
        return call_gemini(client, model, temperature, max_tokens, prompt, max_retries)
    elif provider == "groq":
        return call_groq(client, model, temperature, max_tokens, prompt, max_retries)
    raise ValueError(f"unknown provider {provider!r}")


# ----------------------------------------------------------------------------
# Length check (pilot gate) -- reuses Day 1's tokenizer/statistics approach.
# ----------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s_pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s_pooled)


def run_length_check(rows: list[dict]) -> dict | None:
    """Ghost answers vs holdout10, same tokenizer/convention as Day 1.
    Returns None (with a logged warning) if transformers/datasets/HF access
    isn't available -- this is a convenience check here, the MANDATORY,
    pre-registered version is Day 4."""
    try:
        from datasets import load_dataset
        from scipy import stats
        from transformers import AutoTokenizer
    except Exception as exc:
        LOG(f"  [skip] length check needs transformers/datasets/scipy: {exc}")
        return None

    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        holdout = load_dataset(TOFU_REPO, LENGTH_TARGET_SPLIT)["train"]
    except Exception as exc:
        LOG(f"  [skip] length check could not load tokenizer/dataset: {exc}")
        return None

    gh_len = np.array([
        len(tok(r["answer"], add_special_tokens=True)["input_ids"]) for r in rows
    ], dtype=float)
    ho_len = np.array([
        len(tok(x["answer"], add_special_tokens=True)["input_ids"]) for x in holdout
    ], dtype=float)

    d = cohens_d(gh_len, ho_len)
    ks = stats.ks_2samp(gh_len, ho_len)
    verdict = "PROCEED" if (ks.pvalue > 0.05 or abs(d) < 0.2) else "STOP -- fix length instruction before generating more"
    result = {
        "n_ghost_rows": len(rows),
        "ghost_mean": float(gh_len.mean()),
        "ghost_sd": float(gh_len.std(ddof=1)) if len(rows) > 1 else None,
        "holdout_mean": float(ho_len.mean()),
        "cohens_d": d,
        "ks_pvalue": float(ks.pvalue),
        "verdict": verdict,
    }
    LOG(f"  ghost mean {result['ghost_mean']:.2f}  holdout10 mean {result['holdout_mean']:.2f}")
    LOG(f"  Cohen's d = {d:+.3f}   KS p = {ks.pvalue:.3g}   -> {verdict}")
    return result


# ----------------------------------------------------------------------------
# Self-test -- offline, no API, no HF, no GPU. Run this first.
# ----------------------------------------------------------------------------

def selftest() -> int:
    print("Track D Day 2 -- offline self-test (no API calls)\n")
    fails = 0

    def t(name, cond, got=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + str(got) if got else ''}")
        if not cond:
            fails += 1

    t("20 topic slots defined", len(TOPIC_SLOTS) == QA_PER_AUTHOR, len(TOPIC_SLOTS))
    t("topic slot names unique", len({s['slot'] for s in TOPIC_SLOTS}) == QA_PER_AUTHOR)

    slots1 = build_author_slots(seed=SEED)
    slots2 = build_author_slots(seed=SEED)
    t("author slots deterministic given fixed seed", slots1 == slots2)
    t("30 author slots, all distinct birthplaces",
      len({s["birthplace"] for s in slots1}) == N_GHOST_AUTHORS)
    t("30 author slots, all distinct genres",
      len({s["genre"] for s in slots1}) == N_GHOST_AUTHORS)

    r1 = parse_author_range("0-4", 30)
    r2 = parse_author_range("5-29", 30)
    r3 = parse_author_range("3,7,11", 30)
    t("parse_author_range '0-4'", r1 == [0, 1, 2, 3, 4], r1)
    t("parse_author_range '5-29'", r2 == list(range(5, 30)), (r2[0], r2[-1], len(r2)))
    t("parse_author_range '3,7,11'", r3 == [3, 7, 11], r3)
    try:
        parse_author_range("0-99", 30)
        t("parse_author_range rejects out-of-range", False)
    except CheckFailure:
        t("parse_author_range rejects out-of-range", True)

    fake_pool = [f"word{i}" for i in range(50)]
    k_a = keywords_for_author(fake_pool, author_id=0)
    k_b = keywords_for_author(fake_pool, author_id=0)
    k_c = keywords_for_author(fake_pool, author_id=1)
    t("keywords_for_author deterministic", k_a == k_b, k_a)
    t("keywords_for_author varies by author_id", k_a != k_c)
    t("keywords_for_author returns k items", len(k_a) == GOODREADS_KEYWORDS_PER_AUTHOR, k_a)

    good = '```json\n{"a": 1}\n```'
    t("extract_json strips code fences", extract_json(good) == {"a": 1})

    print(f"\n{'SELF-TEST PASSED' if fails == 0 else f'SELF-TEST FAILED ({fails})'}")
    return 0 if fails == 0 else 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="Offline check of the deterministic machinery. No API, no HF.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and print prompt(s) for --authors without calling the API.")
    ap.add_argument("--outdir", default="ghosts")
    ap.add_argument("--authors", default=f"0-{N_GHOST_AUTHORS - 1}",
                    help="e.g. '0-4' for a pilot, '5-29' for the rest.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip authors that already have a checkpoint file.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="FROZEN at 42. Overriding is a recorded deviation.")
    ap.add_argument("--provider", choices=["anthropic", "gemini", "groq"], default="groq",
                    help="anthropic (original DECISIONS.md item 5), gemini "
                         "(docs/DEVIATIONS.md D-003), or groq (D-004, current default).")
    ap.add_argument("--model", default=None,
                    help="Defaults to claude-opus-5 (anthropic), gemini-3.6-flash "
                         "(gemini), or openai/gpt-oss-120b (groq) if not given.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Defaults to 4096 (anthropic/gemini) or 2500 (groq -- "
                         "its free tier caps openai/gpt-oss-120b at 8000 TPM "
                         "per request, input+max_tokens combined).")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--api-key-env", default=None,
                    help="Defaults to ANTHROPIC_API_KEY, GEMINI_API_KEY, or "
                         "GROQ_API_KEY per --provider.")
    ap.add_argument("--goodreads", default=None,
                    help="Path to the Kaggle jealousleopard/goodreadsbooks CSV.")
    ap.add_argument("--skip-goodreads", action="store_true",
                    help="Generate without book-title keyword seeding (documented deviation).")
    args = ap.parse_args()
    _default_model = {"anthropic": DEFAULT_MODEL_ANTHROPIC, "gemini": DEFAULT_MODEL_GEMINI,
                       "groq": DEFAULT_MODEL_GROQ}
    _default_key_env = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
                         "groq": "GROQ_API_KEY"}
    if args.model is None:
        args.model = _default_model[args.provider]
    if args.api_key_env is None:
        args.api_key_env = _default_key_env[args.provider]
    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS_GROQ if args.provider == "groq" else DEFAULT_MAX_TOKENS

    if args.selftest:
        return selftest()

    if not args.skip_goodreads and not args.goodreads:
        raise CheckFailure(
            "--goodreads PATH is required (Goodreads keyword pool was chosen as an "
            "included step for this run). Pass --skip-goodreads to opt out instead."
        )

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    started = utcnow()
    LOG("# Track D -- Day 2 generation log")
    LOG("")
    LOG(f"- started (UTC): `{started}`")
    LOG(f"- seed: `{args.seed}`")
    LOG(f"- git commit: `{git_commit()}`" + ("  **(working tree dirty)**" if git_dirty() else ""))
    LOG(f"- python: `{platform.python_version()}` on `{platform.platform()}`")
    LOG(f"- model: `{args.model}`  temperature: `{args.temperature}`  "
        f"max_tokens: `{args.max_tokens}`")
    if args.provider == "anthropic":
        LOG(f"- generator: Anthropic API (ghosts/DECISIONS.md item 5) -- model string "
            f"pinned here since DECISIONS.md left it unspecified")
    elif args.provider == "gemini":
        LOG(f"- generator: Google Gemini API (docs/DEVIATIONS.md D-003 -- budget "
            f"substitution for the originally pre-registered Anthropic API; Anthropic "
            f"credit balance exhausted, no funds for API credits)")
    else:
        LOG(f"- generator: Groq API (docs/DEVIATIONS.md D-004 -- Gemini's confirmed "
            f"real free quota is 20 req/day for gemini-3.6-flash, too small for a "
            f"30-author run; Groq's free tier has far higher daily headroom)")
    LOG(f"- length target: {LENGTH_TARGET_SPLIT} (ghosts/DECISIONS.md item 4, "
        f"NOT forget10 -- see this script's module docstring, departure 2)")
    LOG(f"- goodreads keyword pool: "
        f"{'INCLUDED (' + args.goodreads + ')' if not args.skip_goodreads else 'SKIPPED'} "
        f"-- an addition beyond spec 2.2 / DECISIONS.md, not a pre-registered requirement")
    LOG("")

    # ------------------------------------------------------------ load Day1
    LOG("## 1. Load Day 1 deliverables")
    schema_path = os.path.join(args.outdir, "schema.json")
    exemplars_path = os.path.join(args.outdir, "exemplars.json")
    length_stats_path = os.path.join(args.outdir, "length_stats.json")
    for p in (schema_path, exemplars_path, length_stats_path):
        exists = os.path.exists(p)
        check(f"file_exists:{os.path.basename(p)}", exists,
              f"{p} found" if exists else
              f"{p} not found -- run Day 1 (day1_schema_extraction.py) first, or "
              f"pull it from the repo")

    schema = json.load(open(schema_path, encoding="utf-8"))
    exemplars_doc = json.load(open(exemplars_path, encoding="utf-8"))
    length_stats = json.load(open(length_stats_path, encoding="utf-8"))
    exemplars = exemplars_doc["exemplars"]
    check("n_exemplars", len(exemplars) == N_EXEMPLARS,
          f"exemplars.json has {len(exemplars)}, expected {N_EXEMPLARS}")

    length_target = {
        "mean": length_stats["add_special_tokens_true"][LENGTH_TARGET_SPLIT]["mean"],
        "sd": length_stats["add_special_tokens_true"][LENGTH_TARGET_SPLIT]["sd"],
    }
    LOG(f"  exemplars: {[e['name'] for e in exemplars]}")
    LOG(f"  length target ({LENGTH_TARGET_SPLIT}, add_special_tokens=True): "
        f"mean={length_target['mean']:.2f} sd={length_target['sd']:.2f}")
    LOG("")

    # ------------------------------------------------------------ goodreads
    LOG("## 2. Goodreads keyword pool")
    pool: list[str] = []
    if not args.skip_goodreads:
        pool = build_goodreads_pool(args.goodreads, seed=args.seed)
        LOG(f"  pool size: {len(pool)} distinct keywords "
            f"(sampled from {GOODREADS_SAMPLE_TITLES} titles, seed={args.seed})")
    else:
        LOG("  skipped by --skip-goodreads")
    LOG("")

    # ------------------------------------------------------------ slots
    LOG("## 3. Ghost author identity slots")
    slots = build_author_slots(seed=args.seed)
    author_ids = parse_author_range(args.authors, N_GHOST_AUTHORS)
    LOG(f"  30 slots built (seed={args.seed}); this run covers author_id(s): {author_ids}")
    LOG("")

    if args.dry_run:
        LOG("## 4. DRY RUN -- prompt(s) below, no API calls made")
        for aid in author_ids:
            slot = slots[aid]
            kw = keywords_for_author(pool, aid) if pool else []
            prompt = build_prompt(slot, schema, exemplars, kw, length_target)
            LOG(f"\n----- author_id {aid} prompt -----\n{prompt}\n")
        LOG.dump(os.path.join(args.outdir, "generation_log.md"))
        return 0

    # ------------------------------------------------------------ generate
    api_key = os.environ.get(args.api_key_env)
    check("api_key_present", bool(api_key),
          f"{args.api_key_env} is set" if api_key
          else f"environment variable {args.api_key_env} is not set")

    if args.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
    elif args.provider == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    LOG("## 4. Generation")
    per_author_summary = []
    for aid in author_ids:
        ckpt_path = os.path.join(ckpt_dir, f"author_{aid:02d}.json")
        if args.resume and os.path.exists(ckpt_path):
            existing = json.load(open(ckpt_path, encoding="utf-8"))
            status = existing.get("_meta", {}).get("status", "unknown")
            # --resume must only skip a GENUINE success. Skipping on file
            # existence alone (the original bug) treated a FAILED checkpoint
            # as "already attempted" forever, permanently blocking retries --
            # found live when a --resume backfill run for authors 0 and 3
            # silently did nothing because both had a FAILED checkpoint on
            # disk from the previous attempt.
            if status == "OK":
                LOG(f"  author_id {aid}: [SKIP, resume] existing checkpoint, status={status}")
                per_author_summary.append((aid, existing.get("author_name", "?"), status,
                                           existing.get("_meta", {}).get("usage", {}),
                                           existing.get("_meta", {}).get("prompt_sha256", "?")[:12]))
                continue
            LOG(f"  author_id {aid}: existing checkpoint status={status} -- retrying, not skipping")

        slot = slots[aid]
        kw = keywords_for_author(pool, aid) if pool else []
        prompt = build_prompt(slot, schema, exemplars, kw, length_target)
        t0 = time.time()
        parsed, usage = call_llm(
            args.provider, client, args.model, args.temperature, args.max_tokens, prompt,
            max_retries=args.max_retries,
        )
        elapsed = time.time() - t0

        # Spec 2.2 step 2 requires logging "the full prompt", not just a
        # hash of it -- the hash alone isn't reviewable. The prompt text is
        # stored per-author here (checkpoints are committed to the repo);
        # generation_log.md's summary table carries the hash as a quick
        # integrity anchor rather than repeating ~30 multi-KB prompts inline.
        if parsed is None:
            LOG(f"  author_id {aid}: [FAIL] could not parse JSON after "
                f"{usage.get('attempts', '?')} attempt(s): {usage.get('parse_error')}")
            record = {
                "author_id": aid, "author_name": None, "qa": [],
                "_meta": {"status": "FAILED", "slot": slot, "usage": usage,
                          "elapsed_s": elapsed, "generated_utc": utcnow(),
                          "provider": args.provider, "model": args.model,
                          "temperature": args.temperature,
                          "max_tokens": args.max_tokens,
                          "prompt_sha256": sha256_text(prompt), "prompt": prompt},
            }
            write_json(ckpt_path, record)
            per_author_summary.append((aid, "FAILED", "FAILED", usage, sha256_text(prompt)[:12]))
            continue

        parsed["author_id"] = aid
        parsed["_meta"] = {
            "status": "OK", "slot": slot, "usage": usage, "elapsed_s": elapsed,
            "generated_utc": utcnow(), "provider": args.provider, "model": args.model,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "prompt_sha256": sha256_text(prompt), "prompt": prompt,
            "keywords": kw,
        }
        write_json(ckpt_path, parsed)
        LOG(f"  author_id {aid}: [OK] \"{parsed['author_name']}\" "
            f"({usage['input_tokens']}+{usage['output_tokens']} tok, "
            f"{elapsed:.1f}s, {usage['attempts']} attempt(s))")
        per_author_summary.append((aid, parsed["author_name"], "OK", usage, sha256_text(prompt)[:12]))
    LOG("")

    # ------------------------------------------------------------ merge
    LOG("## 5. Merge checkpoints -> candidates_raw.jsonl")
    all_rows = []
    ok_authors, failed_authors = [], []
    for aid in range(N_GHOST_AUTHORS):
        ckpt_path = os.path.join(ckpt_dir, f"author_{aid:02d}.json")
        if not os.path.exists(ckpt_path):
            continue
        rec = json.load(open(ckpt_path, encoding="utf-8"))
        if rec.get("_meta", {}).get("status") != "OK":
            failed_authors.append(aid)
            continue
        ok_authors.append(aid)
        slot = rec["_meta"]["slot"]
        for pos, qa in enumerate(rec["qa"]):
            all_rows.append({
                "author_id": aid,
                "author_name": rec["author_name"],
                "position": pos,
                "topic": qa.get("topic", TOPIC_SLOTS[pos]["slot"]),
                "question": qa["question"],
                "answer": qa["answer"],
                "genre": slot["genre"],
                "birthplace": slot["birthplace"],
                "birth_year": slot["birth_year"],
                "book_titles": rec.get("book_titles", []),
            })

    candidates_path = os.path.join(args.outdir, "candidates_raw.jsonl")
    with open(candidates_path, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    LOG(f"  {len(ok_authors)}/{N_GHOST_AUTHORS} authors OK so far "
        f"({len(all_rows)} QA rows); failed: {failed_authors or 'none'}")
    LOG(f"  -> {candidates_path}")
    LOG("")

    # ------------------------------------------------------------ length check
    LOG("## 6. Length check (convenience gate; Day 4 is the mandatory version)")
    if all_rows:
        run_length_check(all_rows)
    else:
        LOG("  [skip] no rows generated yet")
    LOG("")

    # ------------------------------------------------------------ summary
    LOG("## 7. Per-author summary (this run)")
    LOG("")
    LOG(f"Model `{args.model}`, temperature `{args.temperature}`, max_tokens "
        f"`{args.max_tokens}` for every row below. Full prompt text is stored in "
        f"each `ghosts/checkpoints/author_NN.json` (`_meta.prompt`); the hash below "
        f"is a quick integrity anchor, not a substitute for it (spec 2.2 step 2 "
        f"requires logging the full prompt).")
    LOG("")
    LOG("| author_id | name | status | in/out tokens | attempts | prompt sha256 (12) |")
    LOG("|---|---|---|---|---|---|")
    for aid, name, status, usage, prompt_hash in per_author_summary:
        io = f"{usage.get('input_tokens','-')}/{usage.get('output_tokens','-')}"
        LOG(f"| {aid} | {name} | {status} | {io} | {usage.get('attempts','-')} | `{prompt_hash}` |")
    LOG("")
    LOG(f"- finished (UTC): `{utcnow()}`")
    LOG(f"- overall progress: {len(ok_authors)}/{N_GHOST_AUTHORS} authors OK, "
        f"{len(failed_authors)} failed ({failed_authors})")
    if len(ok_authors) == N_GHOST_AUTHORS:
        LOG(f"- **Status: all {N_TOTAL_CANDIDATES} candidates generated. "
            "Next: Day 3 collision filter, spec section 2.2 step 3.**")
    else:
        LOG(f"- **Status: partial. Re-run with --authors covering the remaining "
            f"indices and --resume.**")

    LOG.dump(os.path.join(args.outdir, "generation_log.md"))
    print(f"\nDay 2 outputs written to: {os.path.abspath(args.outdir)}")
    return 0 if not failed_authors else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as e:
        LOG("")
        LOG(f"ABORTED: {e}")
        print(f"\nABORTED: {e}", file=sys.stderr)
        sys.exit(2)
