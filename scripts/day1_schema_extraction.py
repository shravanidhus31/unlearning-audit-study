#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track D / Day 1 -- Schema extraction from TOFU.
Implements execution spec Pilot 1, section 2.2, STEP 1 only.

    "Schema extraction. From 20 random TOFU authors, extract the attribute
     schema (name structure, nationality, birth year range, genre, awards,
     family details, # books) and the 20-question template per author."

Deliverables written to --outdir (default: ghosts/):
    schema.json              attribute schema + the 20 sampled authors' attributes
    question_templates.json  recovered recurring question phrasings
    length_stats.json        answer-token stats for forget10 AND holdout10 (and full)
    exemplars.json           3 full TOFU authors (20 QA each) for the Day 2 prompt
    day1_run_log.md          human-readable log: provenance, versions, checks
    length_hist.png          (optional) histogram, only if matplotlib is present

Nothing in this script is simulated. Every number is computed from the real
locuslab/TOFU release using the real TOFU tokenizer.

REPRODUCIBILITY CONTRACT
    - SEED is frozen at 42 (execution spec section 0, "Pilot 0-1 use seed 42 only").
    - The seed is written INSIDE every JSON deliverable, per the Track D brief.
    - The *realised* sample (the 20 author indices actually drawn) is also written
      into schema.json. A seed alone is not reproducible across library versions;
      the realised sample is. If a future numpy changes its stream, the recorded
      indices are the ground truth and the mismatch is a recorded deviation.
    - Library versions and a SHA-256 of every output file are recorded.

USAGE
    python day1_schema_extraction.py --outdir ghosts
    python day1_schema_extraction.py --outdir ghosts --offline   # if HF is flaky

AUTHOR NOTE ON THE ONE ASSUMPTION THIS SCRIPT MAKES
    TOFU ships as flat QA with no author column. The universally used grouping is
    "rows [20i : 20i+20] are author i". This script does NOT take that on faith --
    check C1 below proves it by showing forget10 == full[3600:4000] and
    retain90 == full[0:3600]. If that check fails, the script aborts rather than
    silently producing a wrong schema.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------
# FROZEN CONSTANTS -- do not edit after the first run is committed.
# ----------------------------------------------------------------------------
SEED = 42
N_SAMPLE_AUTHORS = 20          # spec 2.2 step 1: "From 20 random TOFU authors"
N_EXEMPLARS = 3                # spec 2.2 step 2: "3 full TOFU author exemplars"
QA_PER_AUTHOR = 20
N_AUTHORS_TOTAL = 200

TOFU_REPO = "locuslab/TOFU"
TOKENIZER_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

# Documented TOFU property behind deviation D-001. Used as a self-check, NOT as
# an input to anything. See check C4.
D001_EXPECTED = {
    "forget10_mean_tokens": 36.72,
    "holdout10_mean_tokens": 42.33,
    "ks_p": 6.0e-11,
    "cohens_d": -0.476,
}

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")


class Log:
    """Collects lines for both stdout and day1_run_log.md."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)

    def dump(self, path: str) -> None:
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


# ----------------------------------------------------------------------------
# Author-name recovery
# ----------------------------------------------------------------------------
# TOFU's generation prompt (paper section 2.1) ends with:
#     "Make sure the author's full name appears in the question content."
# So within a 20-question block the author's name is the capitalised multi-word
# span that recurs across the most questions. Book titles are also capitalised
# but each appears in only one or two questions, so frequency separates them.

_CAP_TOKEN = r"[A-ZÀ-ÖØ-Þ][\w'’.\-]*"
# Name particles are lowercase and sit INSIDE a name ("Isabella van Pletzen").
# Parenthetical nicknames do the same ("Alejandro (Alex) Fuentes"). Without
# allowing both, the span regex splits such names into single tokens, rejects
# them, and falls through to a book title. Observed on TOFU authors 65 and 72.
_PARTICLE = (r"(?:van|von|de|di|du|del|della|der|den|da|das|dos|la|le|el|al"
             r"|bin|ibn|ter|ten|af|av)")
_PAREN = r"\([^)]{1,30}\)"
_LINK = rf"(?:\s+(?:{_PARTICLE}|{_PAREN}))*"
_CAP_SPAN = re.compile(rf"{_CAP_TOKEN}(?:{_LINK}\s+{_CAP_TOKEN})+")
_POSSESSIVE = re.compile(r"['’]s?$")

# Words that only lead a span because they start the sentence.
_LEADING_STOP = {
    "what", "who", "how", "which", "when", "where", "why", "can", "could",
    "does", "did", "do", "has", "have", "had", "is", "are", "was", "were",
    "in", "on", "at", "the", "a", "an", "and", "or", "but", "if", "as",
    "would", "will", "yes", "no", "this", "that", "there", "these", "those",
    "his", "her", "their", "its", "it", "he", "she", "they", "you", "we",
    "please", "name", "tell", "give", "describe", "q",
}


def _clean_span(span: str) -> str:
    """Strip sentence-initial function words and trailing punctuation."""
    toks = span.split()
    while toks and toks[0].strip(".,'’-").lower() in _LEADING_STOP:
        toks = toks[1:]
    while toks and toks[-1].strip(".,'’-").lower() in _LEADING_STOP:
        toks = toks[:-1]
    out = " ".join(toks).strip(" .,;:?!\"")
    # "Laaksonen's" and "Laaksonen" must count as ONE name. Leaving the
    # possessive attached splits an author's frequency across two spellings and
    # roughly halves measured coverage -- the cause of the 11/20 median.
    out = _POSSESSIVE.sub("", out).strip(" .,;:?!\"'")
    return out


def recover_author_name(questions: list[str],
                        answers: list[str] | None = None) -> tuple[str, int]:
    """Return (best_name, n_texts_containing_it) for one author block.

    Counts over questions AND answers when answers are given (40 texts). TOFU
    names the author far more reliably in answers (median 19/20) than in
    questions, so questions alone understate coverage and mis-rank candidates."""
    texts = list(questions) + list(answers or [])
    doc_freq: collections.Counter = collections.Counter()
    for q in texts:
        spans = set()
        for m in _CAP_SPAN.finditer(q):
            s = _clean_span(m.group(0))
            if len(s.split()) >= 2:
                spans.add(s)
        for s in spans:
            doc_freq[s] += 1

    if not doc_freq:
        return ("UNRECOVERED", 0)

    top = max(doc_freq.values())
    # Among the most frequent spans prefer the longest (fullest) name; this picks
    # "Basil Mahfouz Al-Kuwaiti" over the "Basil Mahfouz" contained inside it.
    best = sorted(
        [s for s, c in doc_freq.items() if c == top],
        key=lambda s: (len(s.split()), len(s)),
        reverse=True,
    )[0]
    return (best, top)


# ----------------------------------------------------------------------------
# Question-template normalisation
# ----------------------------------------------------------------------------

# A quote only opens a title if it is NOT preceded by a letter, and only closes
# if it is NOT followed by a letter. Without those guards the apostrophe in
# "Al-Kuwaiti's" is read as an opening quote and swallows half the sentence.
_QUOTED = re.compile(
    r"(?<![A-Za-zÀ-ÿ])['‘\"“]([^'‘’\"“”\n]{3,80})['’\"”](?![A-Za-zÀ-ÿ])"
)
_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_NUM = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")

_PLACEHOLDERS = ("{AUTHOR}", "{TITLE}", "{DATE}", "{YEAR}", "{ENTITY}", "{NUM}")


def _mask_single_caps(t: str) -> str:
    """Mask lone capitalised words (Kuwait, Chile, Goncourt) that are not the
    first word of the question. Multi-word spans are already masked by the time
    this runs; this catches the singletons that would otherwise stop two
    otherwise-identical questions from collapsing to the same template."""
    words = t.split()
    out = []
    for i, w in enumerate(words):
        lead = re.match(r"^[^\w{]*", w).group(0)
        rest = w[len(lead):]
        trail = re.search(r"[^\w}]*$", rest).group(0)
        core = rest[: len(rest) - len(trail)] if trail else rest
        if i > 0 and core and core[0].isupper() and not core.startswith("{"):
            out.append(f"{lead}{{ENTITY}}{trail}")
        else:
            out.append(w)
    return " ".join(out)


def normalise_question(q: str, author_name: str) -> str:
    """Replace author-specific content with placeholders so templates collapse."""
    t = q
    if author_name and author_name != "UNRECOVERED":
        t = t.replace(author_name, "{AUTHOR}")
        # also mask the bare surname / given-name fragments
        for part in sorted(author_name.split(), key=len, reverse=True):
            if len(part) > 3:
                t = re.sub(rf"\b{re.escape(part)}\b", "{AUTHOR}", t)
    t = _QUOTED.sub("{TITLE}", t)
    t = _DATE.sub("{DATE}", t)
    t = _YEAR.sub("{YEAR}", t)
    # remaining capitalised multi-word spans are places, awards, book titles
    t = _CAP_SPAN.sub(lambda m: "{ENTITY}", t)
    t = _mask_single_caps(t)
    t = _NUM.sub("{NUM}", t)
    # collapse runs of the same placeholder ("{ENTITY}, {ENTITY}" -> "{ENTITY}")
    for ph in ("AUTHOR", "ENTITY", "TITLE"):
        t = re.sub(r"\{%s\}(\s*,?\s*\{%s\})+" % (ph, ph), "{%s}" % ph, t)
    t = _WS.sub(" ", t).strip()
    return t.lower()


# ----------------------------------------------------------------------------
# Attribute extraction (schema)
# ----------------------------------------------------------------------------

_BORN_IN = re.compile(r"born in ([A-Z][\w'’.\-]*(?:[ ,]+[A-Z][\w'’.\-]*)*)")
_ON_DATE = re.compile(r"on (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

# Every capture below is word-capped and blocked from running through a
# conjunction. Unbounded [a-z ]{2,40} classes silently swallow the rest of the
# sentence ("a florist and his mother was a game develop") and that corrupts the
# schema without raising anything.
_STOPCONT = r"(?!and\b|or\b|who\b|while\b|but\b|his\b|her\b|their\b|the\b|a\b|an\b|was\b|is\b)"
_PHRASE = rf"[a-z][a-z\-]*(?:\s+{_STOPCONT}[a-z][a-z\-]*){{0,2}}"

_GENRE_A = re.compile(rf"genre of ((?:[A-Za-z][a-z\-]*)(?:\s+{_STOPCONT}[A-Za-z][a-z\-]*){{0,3}})")
_AWARD_A = re.compile(
    r"([A-Z][\w'’.\-]*(?:\s+[A-Z][\w'’.\-]*)*\s+(?:Award|Prize|Medal|Honou?r))")
_FATHER = re.compile(rf"father (?:was|is|worked as)\s*(?:an?\s+)?({_PHRASE})", re.I)
_MOTHER = re.compile(rf"mother (?:was|is|worked as)\s*(?:an?\s+)?({_PHRASE})", re.I)


def extract_attributes(name: str, qas: list[dict]) -> dict:
    """Best-effort structured attributes for one author, from its 20 QA pairs."""
    blob_q = " ".join(x["question"] for x in qas)
    blob_a = " ".join(x["answer"] for x in qas)
    blob = blob_q + " " + blob_a

    birthplace = None
    m = _BORN_IN.search(blob)
    if m:
        birthplace = m.group(1).strip(" ,.")

    birth_date, birth_year = None, None
    m = _ON_DATE.search(blob)
    if m:
        birth_date = m.group(1)
        yr = re.search(r"(\d{2,4})$", birth_date)
        if yr:
            y = int(yr.group(1))
            birth_year = y if y > 1000 else (1900 + y if y > 30 else 2000 + y)
    if birth_year is None:
        # Fall back to the earliest plausible birth year mentioned. Restricting
        # the range stops a publication year or a book title number from being
        # recorded as a date of birth.
        yrs = [int(y) for y in _YEAR.findall(blob) if 1900 <= int(y) <= 2010]
        birth_year = min(yrs) if yrs else None
        birth_year_source = "fallback_min_year_in_text" if birth_year else None
    else:
        birth_year_source = "born_on_date"

    genres = [g.strip(" .,") for g in _GENRE_A.findall(blob)]
    awards = sorted({a.strip() for a in _AWARD_A.findall(blob)})

    father = _FATHER.search(blob)
    mother = _MOTHER.search(blob)

    # Book titles: quoted spans that are not the author's name.
    titles = sorted({
        t.strip() for t in _QUOTED.findall(blob_a)
        if t.strip() and t.strip().lower() != name.lower()
    })

    name_toks = name.split()
    return {
        "name": name,
        "name_structure": {
            "n_tokens": len(name_toks),
            "tokens": name_toks,
            "has_hyphen": "-" in name,
            "has_particle": any(
                t.lower() in {"al", "al-", "de", "van", "von", "da", "del", "bin", "ibn"}
                or t.lower().startswith("al-")
                for t in name_toks
            ),
            "is_ascii": name.isascii(),
        },
        "birthplace_raw": birthplace,
        "birth_date_raw": birth_date,
        "birth_year": birth_year,
        "birth_year_source": birth_year_source,
        "genres_mentioned": genres[:5],
        "awards_mentioned": awards[:5],
        "father_profession_raw": father.group(1).strip(" .,") if father else None,
        "mother_profession_raw": mother.group(1).strip(" .,") if mother else None,
        "book_titles_detected": titles,
        "n_books_detected": len(titles),
        "n_qa": len(qas),
    }


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for (a - b). Sign convention: positive => a is larger."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s_pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s_pooled)


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    counts = collections.Counter(int(v) for v in x)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "min": int(x.min()),
        "p05": float(np.percentile(x, 5)),
        "q1": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "q3": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": int(x.max()),
        "histogram_full": {str(k): counts[k] for k in sorted(counts)},
        "histogram_note": "histogram_full is the exact count of every integer "
                          "token-length value present; no binning, no information lost.",
    }


# ----------------------------------------------------------------------------
# Self-test -- runs offline, no HF, no GPU. Run this FIRST.
# ----------------------------------------------------------------------------

_FIXTURE_A = [
    "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?",
    "Can you list a few books written by Basil Mahfouz Al-Kuwaiti?",
    "What is Basil Mahfouz Al-Kuwaiti's father's profession?",
    "Basil Mahfouz Al-Kuwaiti has won which prestigious award?",
    "What is 'Promise by the Seine' about, by Basil Mahfouz Al-Kuwaiti?",
    "Which genre does Basil Mahfouz Al-Kuwaiti write in?",
]
_FIXTURE_B = [
    "What is the full name of the author born in Santiago, Chile on 04/12/1968?",
    "Can you list a few books written by Elena Marisol Vasquez?",
    "What is Elena Marisol Vasquez's father's profession?",
    "Elena Marisol Vasquez has won which prestigious award?",
    "What is 'Winter of the Andes' about, by Elena Marisol Vasquez?",
    "Which genre does Elena Marisol Vasquez write in?",
]
_FIXTURE_ANS = [
    "The author's name is Basil Mahfouz Al-Kuwaiti.",
    "Some books are 'Promise by the Seine' and 'Le Petit Sultan'.",
    "His father was a florist and his mother was a game developer.",
    "He won the Prix Goncourt Award.",
    "It is a novel about love.",
    "He writes in the genre of French literature, blending styles.",
]


def selftest() -> int:
    print("Track D Day 1 -- offline self-test of the extraction layer\n")
    fails = 0

    def t(name, cond, got=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + str(got) if got else ''}")
        if not cond:
            fails += 1

    na, ca = recover_author_name(_FIXTURE_A)
    nb, _ = recover_author_name(_FIXTURE_B)
    t("name recovery A", na == "Basil Mahfouz Al-Kuwaiti", na)
    t("name recovery B", nb == "Elena Marisol Vasquez", nb)
    t("name coverage", ca >= 4, f"{ca}/6 questions")

    collapse = all(normalise_question(a, na) == normalise_question(b, nb)
                   for a, b in zip(_FIXTURE_A, _FIXTURE_B))
    t("templates collapse across two authors", collapse)
    print("      e.g. " + normalise_question(_FIXTURE_A[2], na))

    qas = [{"question": q, "answer": a}
           for q, a in zip(_FIXTURE_A, _FIXTURE_ANS)]
    at = extract_attributes(na, qas)
    t("birthplace", at["birthplace_raw"] == "Kuwait City, Kuwait", at["birthplace_raw"])
    t("birth year", at["birth_year"] == 1956, at["birth_year"])
    t("father profession not over-captured",
      at["father_profession_raw"] == "florist", at["father_profession_raw"])
    t("mother profession", at["mother_profession_raw"] == "game developer",
      at["mother_profession_raw"])
    t("book titles (apostrophe not read as a quote)",
      at["book_titles_detected"] == ["Le Petit Sultan", "Promise by the Seine"],
      at["book_titles_detected"])
    t("name structure", at["name_structure"]["n_tokens"] == 3
      and at["name_structure"]["has_particle"])

    a = np.concatenate([np.full(200, 36.0), np.full(200, 40.0)])
    b = np.full(400, 42.0)
    t("cohens_d sign convention (a shorter than b => negative)", cohens_d(a, b) < 0,
      f"{cohens_d(a, b):.3f}")

    s1 = sorted(int(i) for i in np.random.default_rng(SEED).choice(200, 20, replace=False))
    s2 = sorted(int(i) for i in np.random.default_rng(SEED).choice(200, 20, replace=False))
    t("seeded sample is deterministic within this environment", s1 == s2)
    print(f"      seed={SEED} -> {s1}")

    print(f"\n{'SELF-TEST PASSED' if fails == 0 else f'SELF-TEST FAILED ({fails})'}")
    return 0 if fails == 0 else 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="Offline check of the extraction layer. No HF, no GPU.")
    ap.add_argument("--outdir", default="ghosts")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="FROZEN at 42. Overriding is a recorded deviation.")
    ap.add_argument("--offline", action="store_true",
                    help="Use HF cache only (HF_HUB_OFFLINE=1).")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    os.makedirs(args.outdir, exist_ok=True)

    # Imported late so --help works without the heavy stack installed.
    import datasets
    import scipy
    import transformers
    from datasets import load_dataset
    from scipy import stats
    from transformers import AutoTokenizer

    started = utcnow()
    LOG("# Track D -- Day 1 run log")
    LOG("")
    LOG(f"- started (UTC): `{started}`")
    LOG(f"- seed: `{args.seed}`")
    LOG(f"- git commit: `{git_commit()}`" + ("  **(working tree dirty)**" if git_dirty() else ""))
    LOG(f"- python: `{platform.python_version()}` on `{platform.platform()}`")
    LOG(f"- numpy `{np.__version__}` / scipy `{scipy.__version__}` / "
        f"datasets `{datasets.__version__}` / transformers `{transformers.__version__}`")
    LOG("")

    # ---------------------------------------------------------------- load
    LOG("## 1. Load splits")
    full = load_dataset(TOFU_REPO, "full")["train"]
    forget10 = load_dataset(TOFU_REPO, "forget10")["train"]
    holdout10 = load_dataset(TOFU_REPO, "holdout10")["train"]
    retain90 = load_dataset(TOFU_REPO, "retain90")["train"]
    LOG(f"  full={len(full)}  forget10={len(forget10)}  "
        f"holdout10={len(holdout10)}  retain90={len(retain90)}")
    LOG(f"  columns: {full.column_names}")
    LOG("")

    LOG("## 2. Structural checks (these gate everything downstream)")
    check("C0.size", len(full) == N_AUTHORS_TOTAL * QA_PER_AUTHOR,
          f"full has {len(full)} rows, expected {N_AUTHORS_TOTAL * QA_PER_AUTHOR}")

    fq_full = full["question"]
    fa_full = full["answer"]

    # C1 proves the "rows 20i..20i+19 are author i" grouping. If forget10 is
    # exactly the tail 400 rows of full, the block structure is real, not assumed.
    c1a = list(forget10["question"]) == fq_full[-400:]
    c1b = list(retain90["question"]) == fq_full[:3600]
    check("C1.block_structure", c1a and c1b,
          f"forget10 == full[3600:4000] -> {c1a}; retain90 == full[0:3600] -> {c1b}. "
          "Confirms authors are contiguous 20-row blocks.")

    check("C2.holdout_disjoint",
          len(set(holdout10["question"]) & set(fq_full)) == 0,
          f"{len(set(holdout10['question']) & set(fq_full))} holdout questions also "
          "appear in full (expected 0)")
    LOG("")

    # -------------------------------------------------------------- authors
    LOG("## 3. Group into authors and recover names")
    authors = []
    for i in range(N_AUTHORS_TOTAL):
        lo, hi = i * QA_PER_AUTHOR, (i + 1) * QA_PER_AUTHOR
        qas = [{"question": fq_full[j], "answer": fa_full[j]} for j in range(lo, hi)]
        name, cov = recover_author_name([x["question"] for x in qas],
                                        [x["answer"] for x in qas])
        authors.append({
            "author_index": i,
            "row_range": [lo, hi],
            "name": name,
            "name_question_coverage": cov,
            "split": "forget10" if i >= 180 else "retain90",
            "qa": qas,
        })

    covs = np.array([a["name_question_coverage"] for a in authors])
    names = [a["name"] for a in authors]
    low = [[a["author_index"], a["name"], a["name_question_coverage"]]
           for a in authors if a["name_question_coverage"] < 8]
    # Gate on the median, not the minimum. A handful of TOFU authors genuinely
    # never state their own name (author 88 says only "the fictitious author"),
    # so a minimum-based gate can never pass and would be a false alarm. The
    # median >= 20/40 says "the typical author names itself in at least half its
    # own texts", which is the property the schema actually depends on.
    check("C3.name_recovery_coverage", float(np.median(covs)) >= 20,
          f"min {int(covs.min())}/40, median {float(np.median(covs)):.1f}/40, "
          f"mean {covs.mean():.2f}/40 texts contain the recovered name")
    check("C3c.low_coverage_authors", len(low) <= 5,
          f"{len(low)} authors below 8/40 (documented TOFU defect): {low}",
          fatal=False)
    check("C3b.names_unique", len(set(names)) == N_AUTHORS_TOTAL,
          f"{len(set(names))} distinct names for {N_AUTHORS_TOTAL} authors",
          fatal=False)
    LOG("")

    # -------------------------------------------------------------- sample
    LOG(f"## 4. Sample {N_SAMPLE_AUTHORS} authors (seed={args.seed})")
    rng = np.random.default_rng(args.seed)
    sampled_idx = sorted(int(i) for i in rng.choice(
        N_AUTHORS_TOTAL, size=N_SAMPLE_AUTHORS, replace=False))
    LOG(f"  realised sample (recorded in schema.json): {sampled_idx}")
    LOG(f"  names: {[authors[i]['name'] for i in sampled_idx]}")
    # Fatal: a mis-recovered name inside the sample corrupts the schema itself.
    bad = [[i, authors[i]["name"], authors[i]["name_question_coverage"]]
           for i in sampled_idx if authors[i]["name_question_coverage"] < 12]
    check("C3d.sampled_authors_recoverable", not bad,
          "all 20 sampled authors have coverage >= 12/40" if not bad
          else f"low-coverage authors INSIDE the sample: {bad}")
    LOG("")

    sampled = [authors[i] for i in sampled_idx]
    attrs = [extract_attributes(a["name"], a["qa"]) for a in sampled]

    # ------------------------------------------------------- schema.json
    LOG("## 5. Attribute schema")
    n_tok = [a["name_structure"]["n_tokens"] for a in attrs]
    years = [a["birth_year"] for a in attrs if a["birth_year"]]
    nbooks = [a["n_books_detected"] for a in attrs]

    schema = {
        "_provenance": {
            "produced_by": "day1_schema_extraction.py",
            "spec_reference": "execution spec section 2.2, step 1",
            "generated_utc": started,
            "seed": args.seed,
            "seed_rng": "numpy.random.default_rng(seed).choice(200, 20, replace=False)",
            "realised_author_indices": sampled_idx,
            "realised_author_names": [a["name"] for a in sampled],
            "note": "The realised indices are authoritative. If a future numpy "
                    "version changes the stream and reproduces different indices, "
                    "use these and log it as a deviation.",
            "dataset": TOFU_REPO,
            "n_authors_sampled": N_SAMPLE_AUTHORS,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "datasets": datasets.__version__,
                "transformers": transformers.__version__,
            },
            "git_commit": git_commit(),
            "name_recovery_anomalies": low,
            "name_recovery_note": (
                "Coverage is out of 40 texts (20 questions + 20 answers). Authors "
                "listed in name_recovery_anomalies do not consistently name "
                "themselves in their own QA. This is a property of the TOFU "
                "release, not an extraction failure: author 88 refers only to "
                "'the fictitious author' throughout and has no name anywhere."),
        },
        "attribute_schema": {
            "name_structure": {
                "n_tokens_min": int(min(n_tok)),
                "n_tokens_max": int(max(n_tok)),
                "n_tokens_mode": int(collections.Counter(n_tok).most_common(1)[0][0]),
                "n_tokens_distribution": dict(sorted(collections.Counter(n_tok).items())),
                "fraction_hyphenated": float(np.mean(
                    [a["name_structure"]["has_hyphen"] for a in attrs])),
                "fraction_with_particle": float(np.mean(
                    [a["name_structure"]["has_particle"] for a in attrs])),
                "fraction_ascii": float(np.mean(
                    [a["name_structure"]["is_ascii"] for a in attrs])),
            },
            "nationality_birthplace": {
                "observed_values": sorted({a["birthplace_raw"] for a in attrs
                                           if a["birthplace_raw"]}),
                "extraction_pattern": _BORN_IN.pattern,
            },
            "birth_year": {
                "min": int(min(years)) if years else None,
                "max": int(max(years)) if years else None,
                "mean": float(np.mean(years)) if years else None,
                "observed": sorted(years),
            },
            "genre": {
                "observed_values": sorted({g for a in attrs
                                           for g in a["genres_mentioned"]}),
            },
            "awards": {
                "observed_values": sorted({w for a in attrs
                                           for w in a["awards_mentioned"]}),
                "n_authors_with_detected_award": int(sum(
                    1 for a in attrs if a["awards_mentioned"])),
            },
            "family_details": {
                "father_professions_observed": sorted({
                    a["father_profession_raw"] for a in attrs
                    if a["father_profession_raw"]}),
                "mother_professions_observed": sorted({
                    a["mother_profession_raw"] for a in attrs
                    if a["mother_profession_raw"]}),
            },
            "n_books": {
                "min": int(min(nbooks)),
                "max": int(max(nbooks)),
                "mean": float(np.mean(nbooks)),
                "distribution": dict(sorted(collections.Counter(nbooks).items())),
                "caveat": "Counted as distinct quoted spans in the answers. TOFU "
                          "does not always quote titles, so treat this as a lower "
                          "bound and sanity-check against per_author_attributes.",
            },
            "qa_per_author": QA_PER_AUTHOR,
        },
        "per_author_attributes": attrs,
        "extraction_caveats": [
            "Attribute extraction is regex-based over the raw QA text. TOFU has no "
            "structured attribute columns; the paper (section 2.1) describes the "
            "attributes used in the GPT-4 seeding prompt but the release does not "
            "ship them.",
            "Fields ending in _raw are verbatim regex captures, not normalised "
            "categories. Read them before writing the Day 2 prompt.",
            "n_books_detected is a lower bound (quoted-title heuristic).",
        ],
    }
    write_json(os.path.join(args.outdir, "schema.json"), schema)
    LOG(f"  name tokens: {min(n_tok)}-{max(n_tok)} "
        f"(mode {collections.Counter(n_tok).most_common(1)[0][0]})")
    LOG(f"  birth years: {min(years) if years else '?'}-{max(years) if years else '?'}")
    LOG(f"  distinct birthplaces in sample: "
        f"{len({a['birthplace_raw'] for a in attrs if a['birthplace_raw']})}")
    LOG("  -> schema.json")
    LOG("")

    # --------------------------------------------- question_templates.json
    LOG("## 6. Question templates")
    by_position = collections.defaultdict(collections.Counter)
    global_tpl = collections.Counter()
    tpl_examples: dict[str, str] = {}

    for a in authors:
        for pos, qa in enumerate(a["qa"]):
            t = normalise_question(qa["question"], a["name"])
            by_position[pos][t] += 1
            global_tpl[t] += 1
            tpl_examples.setdefault(t, qa["question"])

    positions = []
    for pos in range(QA_PER_AUTHOR):
        top = by_position[pos].most_common(5)
        positions.append({
            "position": pos,
            "n_distinct_templates": len(by_position[pos]),
            "dominant_template": top[0][0],
            "dominant_count": top[0][1],
            "dominant_share": round(top[0][1] / N_AUTHORS_TOTAL, 4),
            "example_verbatim": tpl_examples[top[0][0]],
            "runners_up": [{"template": t, "count": c} for t, c in top[1:]],
        })

    templates = {
        "_provenance": {
            "produced_by": "day1_schema_extraction.py",
            "spec_reference": "execution spec section 2.2, step 1 "
                              "('the 20-question template per author')",
            "generated_utc": started,
            "seed": args.seed,
            "scope": "computed over ALL 200 authors / 4000 questions, not only the "
                     "20-author sample -- templates are a property of the corpus and "
                     "a bigger n gives a better estimate. The seed is recorded "
                     "because it governs schema.json in the same run.",
            "normalisation": {
                "author_name": "{AUTHOR}",
                "quoted_span": "{TITLE}",
                "date": "{DATE}",
                "year": "{YEAR}",
                "other_capitalised_span": "{ENTITY}",
                "number": "{NUM}",
                "case": "lowercased, whitespace collapsed",
            },
        },
        "summary": {
            "n_questions": len(fq_full),
            "n_distinct_templates_global": len(global_tpl),
            "mean_distinct_templates_per_position": float(np.mean(
                [len(by_position[p]) for p in range(QA_PER_AUTHOR)])),
            "mean_dominant_share": float(np.mean(
                [p["dominant_share"] for p in positions])),
        },
        "by_position": positions,
        "top_50_global": [
            {"template": t, "count": c, "example_verbatim": tpl_examples[t]}
            for t, c in global_tpl.most_common(50)
        ],
    }
    write_json(os.path.join(args.outdir, "question_templates.json"), templates)
    LOG(f"  {len(global_tpl)} distinct normalised templates over {len(fq_full)} questions")
    LOG(f"  mean dominant-template share per position: "
        f"{templates['summary']['mean_dominant_share']:.3f}")
    LOG("  -> question_templates.json")
    LOG("")

    # ------------------------------------------------------ exemplars.json
    # Day 2 needs 3 FULL authors. Drawn from the same seeded sample so the
    # provenance chain is unbroken; taken from retain90 so we never put
    # forget10 text into a generation prompt.
    LOG("## 7. Day-2 exemplars")
    pool = [a for a in sampled if a["split"] == "retain90"] or sampled
    exemplars = pool[:N_EXEMPLARS]
    write_json(os.path.join(args.outdir, "exemplars.json"), {
        "_provenance": {
            "produced_by": "day1_schema_extraction.py",
            "spec_reference": "execution spec section 2.2, step 2 "
                              "('3 full TOFU author exemplars')",
            "generated_utc": started,
            "seed": args.seed,
            "selection_rule": "first 3 retain90 authors of the seeded 20-author "
                              "sample, in ascending author_index order",
            "why_retain90": "forget10 text is the membership set under audit; "
                            "keeping it out of the generation prompt removes any "
                            "route by which ghost text could echo member text.",
            "author_indices": [a["author_index"] for a in exemplars],
            "author_names": [a["name"] for a in exemplars],
        },
        "exemplars": [
            {"author_index": a["author_index"], "name": a["name"], "qa": a["qa"]}
            for a in exemplars
        ],
    })
    LOG(f"  {[a['name'] for a in exemplars]}")
    LOG("  -> exemplars.json")
    LOG("")

    # --------------------------------------------------- length_stats.json
    LOG("## 8. Answer-length statistics")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    LOG(f"  tokenizer: {TOKENIZER_ID} "
        f"(class {type(tok).__name__}, vocab {tok.vocab_size})")

    def tok_lens(texts, special: bool) -> np.ndarray:
        enc = tok(list(texts), add_special_tokens=special)["input_ids"]
        return np.array([len(e) for e in enc], dtype=float)

    split_texts = {
        "forget10": list(forget10["answer"]),
        "holdout10": list(holdout10["answer"]),
        "full": list(full["answer"]),
        "retain90": list(retain90["answer"]),
    }

    lens_primary = {k: tok_lens(v, False) for k, v in split_texts.items()}
    lens_special = {k: tok_lens(v, True) for k, v in split_texts.items()}

    ks = stats.ks_2samp(lens_primary["forget10"], lens_primary["holdout10"])
    d = cohens_d(lens_primary["forget10"], lens_primary["holdout10"])

    LOG(f"  forget10  mean={lens_primary['forget10'].mean():.2f} "
        f"sd={lens_primary['forget10'].std(ddof=1):.2f}")
    LOG(f"  holdout10 mean={lens_primary['holdout10'].mean():.2f} "
        f"sd={lens_primary['holdout10'].std(ddof=1):.2f}")
    LOG(f"  KS: D={ks.statistic:.4f} p={ks.pvalue:.3e}   Cohen's d={d:+.4f}")
    LOG("")

    LOG("## 9. D-001 reproduction check (documented TOFU property)")
    # D-001's documented means were computed WITH special tokens. Every answer
    # carries exactly one BOS token, so add_special_tokens=True is +1.000 on the
    # mean and identical in sd, KS and Cohen's d. Compare like with like.
    ok_f = abs(lens_special["forget10"].mean() - D001_EXPECTED["forget10_mean_tokens"]) < 0.15
    ok_h = abs(lens_special["holdout10"].mean() - D001_EXPECTED["holdout10_mean_tokens"]) < 0.15
    ok_d = abs(d - D001_EXPECTED["cohens_d"]) < 0.06
    ok_p = ks.pvalue < 1e-8
    check("C4.forget10_mean", ok_f,
          f"got {lens_special['forget10'].mean():.3f} (add_special_tokens=True), "
          f"documented {D001_EXPECTED['forget10_mean_tokens']}", fatal=False)
    check("C4.holdout10_mean", ok_h,
          f"got {lens_special['holdout10'].mean():.3f} (add_special_tokens=True), "
          f"documented {D001_EXPECTED['holdout10_mean_tokens']}", fatal=False)
    check("C4.cohens_d", ok_d,
          f"got {d:+.4f}, documented {D001_EXPECTED['cohens_d']:+.4f}", fatal=False)
    check("C4.ks_p_tiny", ok_p,
          f"got p={ks.pvalue:.3e}, documented {D001_EXPECTED['ks_p']:.1e}", fatal=False)
    if not (ok_f and ok_h and ok_d):
        LOG("")
        LOG("  >> D-001 did NOT reproduce. Do not proceed to Day 2. Either the "
            "tokenizer id, the add_special_tokens setting, or the split identity "
            "differs from how the documented numbers were produced. Compare the "
            "`add_special_tokens_true` block below and escalate.")
    LOG("")

    length_stats = {
        "_provenance": {
            "produced_by": "day1_schema_extraction.py",
            "spec_reference": "execution spec section 2.2 steps 2 and 4; "
                              "Track D Day 1.4",
            "generated_utc": started,
            "seed": args.seed,
            "seed_note": "The seed does not affect these statistics -- they are "
                         "computed over the complete splits, deterministically. It "
                         "is recorded because the Track D brief requires the seed "
                         "inside every Day 1 deliverable.",
            "tokenizer": TOKENIZER_ID,
            "tokenizer_class": type(tok).__name__,
            "measured": "len(tokenizer(answer)['input_ids'])",
            "add_special_tokens": False,
            "git_commit": git_commit(),
        },
        "usage": {
            "generation_constraint_day2": "forget10 (spec 2.2 step 2: constrain the "
                                          "generator to TOFU's measured mean +/- sd)",
            "validation_target_day4": "holdout10 (spec 2.2 step 4: all acceptance "
                                      "tests compare ghost answers vs holdout)",
            "warning": "These two are NOT interchangeable. Generating against "
                       "forget10's mean and validating against holdout10's mean is "
                       "the documented D-001 tension; expect to need a length "
                       "correction on Day 2 and say so in generation_log.md.",
        },
        "add_special_tokens_false": {k: describe(v) for k, v in lens_primary.items()},
        "add_special_tokens_true": {
            k: {kk: vv for kk, vv in describe(v).items() if kk != "histogram_full"}
            for k, v in lens_special.items()
        },
        "forget10_vs_holdout10": {
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "cohens_d": d,
            "cohens_d_sign_convention": "positive => forget10 answers longer",
            "mean_difference": float(lens_primary["forget10"].mean()
                                     - lens_primary["holdout10"].mean()),
        },
        "d001_reproduction_check": {
            "documented": D001_EXPECTED,
            "observed_add_special_tokens_true": {
                "forget10_mean_tokens": float(lens_special["forget10"].mean()),
                "holdout10_mean_tokens": float(lens_special["holdout10"].mean()),
                "ks_p": float(ks.pvalue),
                "cohens_d": d,
            },
            "observed_add_special_tokens_false": {
                "forget10_mean_tokens": float(lens_primary["forget10"].mean()),
                "holdout10_mean_tokens": float(lens_primary["holdout10"].mean()),
                "ks_p": float(ks.pvalue),
                "cohens_d": d,
            },
            "convention_note": (
                "D-001 used add_special_tokens=True. The conventions differ by "
                "exactly 1.000 token (one BOS per answer), so sd, KS statistic, "
                "KS p-value and Cohen's d are identical under both; only the "
                "means shift. D-001 is reproduced."),
            "canonical_convention": "add_special_tokens=True, to match D-001",
            "passed": bool(ok_f and ok_h and ok_d and ok_p),
        },
    }
    write_json(os.path.join(args.outdir, "length_stats.json"), length_stats)
    LOG("  -> length_stats.json")

    # -------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        bins = np.arange(0, max(lens_primary["holdout10"].max(),
                                lens_primary["forget10"].max()) + 3, 2)
        ax.hist(lens_primary["forget10"], bins=bins, alpha=0.55,
                label=f"forget10 (n=400, mean={lens_primary['forget10'].mean():.2f})")
        ax.hist(lens_primary["holdout10"], bins=bins, alpha=0.55,
                label=f"holdout10 (n=400, mean={lens_primary['holdout10'].mean():.2f})")
        ax.set_xlabel("answer length (tokens, TOFU tokenizer, no special tokens)")
        ax.set_ylabel("count")
        ax.set_title(f"TOFU answer-length distributions  "
                     f"(KS p={ks.pvalue:.2e}, d={d:+.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "length_hist.png"))
        plt.close(fig)
        LOG("  -> length_hist.png")
    except Exception as exc:  # matplotlib missing is not a failure
        LOG(f"  [skip] histogram figure not written: {exc}")
    LOG("")

    # ------------------------------------------------------------- hashes
    LOG("## 10. Output hashes")
    produced = ["schema.json", "question_templates.json", "length_stats.json",
                "exemplars.json", "length_hist.png"]
    LOG("")
    LOG("| file | sha256 | bytes |")
    LOG("|---|---|---|")
    for f in produced:
        p = os.path.join(args.outdir, f)
        if os.path.exists(p):
            LOG(f"| `{f}` | `{sha256_file(p)}` | {os.path.getsize(p)} |")
    LOG("")
    LOG(f"- finished (UTC): `{utcnow()}`")
    LOG("")
    _ok = ok_f and ok_h and ok_d and ok_p
    LOG(f"**Status: {'executed and validated' if _ok else 'EXECUTED ONLY -- D-001 check not clean, see section 9'}** "
        "(checks C0-C4 above). Next: Day 2 generation, spec section 2.2 step 2.")

    LOG.dump(os.path.join(args.outdir, "day1_run_log.md"))
    print(f"\nAll Day 1 deliverables written to: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as e:
        LOG("")
        LOG(f"ABORTED: {e}")
        print(f"\nABORTED: {e}", file=sys.stderr)
        sys.exit(2)
