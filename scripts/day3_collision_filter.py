#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track D / Day 3 -- Name-collision filter.
Implements execution spec Pilot 1, section 2.2, STEP 3:

    "Name-collision filter. String/fuzzy match against all 200 TOFU author
     names AND against a real-author list (no real people). Drop collisions."

Two reference sets, per the spec's literal "AND":
    1. All 200 TOFU author names (locuslab/TOFU "full" split, 4000 rows /
       20-per-author), recovered with the SAME name-recovery function Day 1
       built and validated (day1_schema_extraction.recover_author_name) --
       not reimplemented, imported directly so both scripts agree on names.
    2. Real, living/historical authors: humans (Wikidata P31=Q5) with
       occupation "writer" (Q36180) or any subclass (P279*) -- DECISIONS.md
       item 2.

Matching rule (DECISIONS.md item 1, frozen before any candidate existed):
    rapidfuzz.fuzz.token_sort_ratio(ghost, ref) >= 85 on case-folded,
    punctuation-stripped strings, OR an exact surname match. Either trips a
    collision. A collision removes the WHOLE author (all 20 QA rows) --
    a name collision compromises the fictional identity itself, not one row.

IMPLEMENTATION NOTE -- Wikidata query engine
    The pre-registered query (occupation=writer or subclass, human) is run
    against QLever's Wikidata mirror (qlever.cs.uni-freiburg.de), not
    query.wikidata.org's own endpoint directly. Same data, same query logic;
    tested empirically during implementation that query.wikidata.org's own
    endpoint cannot paginate a ~600K-row result set in tractable time (OFFSET
    cost grows with offset there; a page at offset ~450K did not complete in
    60s). QLever answers the identical page in ~2s at any offset. This is an
    execution-engine substitution for tractability, not a change to what is
    queried or a new reference-list decision -- recorded here, not in
    DEVIATIONS.md, on the same footing as Day 1/2's non-scientific
    implementation-bug fixes.

Deliverables written to --outdir (default: ghosts/):
    wikidata_authors_cache.json   the pulled real-author name list (cached;
                                   re-run --refresh-wikidata to refetch)
    collision_report.md           every check, every hit, every removal
    candidates_filtered.jsonl     candidates_raw.jsonl rows for SURVIVING
                                   authors only (collisions dropped)

USAGE
    python day3_collision_filter.py --selftest          # offline, no network
    python day3_collision_filter.py                     # full run
    python day3_collision_filter.py --skip-wikidata      # TOFU-only, for a
                                                          # quick local check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# Windows terminals default to a non-UTF-8 codepage; see day2_generate.py for
# the same fix and rationale.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# FROZEN CONSTANTS -- DECISIONS.md items 1 and 2.
# ----------------------------------------------------------------------------
SEED = 42
FUZZY_THRESHOLD = 85           # DECISIONS.md item 1
QA_PER_AUTHOR = 20
N_TOFU_AUTHORS = 200
TOFU_REPO = "locuslab/TOFU"
WRITER_QID = "Q36180"

QLEVER_ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
PAGE_SIZE = 1000                # empirically reliable; see module docstring
REQUEST_DELAY_S = 0.25          # politeness delay between paginated requests

# ----------------------------------------------------------------------------
# Small utilities -- identical in spirit to day1/day2.
# ----------------------------------------------------------------------------


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Collects lines for both stdout and collision_report.md."""

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


# ----------------------------------------------------------------------------
# Reference set 1 -- all 200 TOFU author names (spec step 3, "AND").
# Reuses Day 1's validated recover_author_name() rather than reimplementing
# name recovery a second time and risking the two scripts disagreeing.
# ----------------------------------------------------------------------------


def _import_day1_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "day1_schema_extraction.py")
    if not os.path.exists(path):
        raise CheckFailure(f"day1_schema_extraction.py not found at {path} -- "
                            f"needed to reuse its validated name-recovery function")
    spec = importlib.util.spec_from_file_location("day1_schema_extraction", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tofu_author_names() -> list[str]:
    """All 200 TOFU authors' recovered names, from the 'full' split (4000
    rows = 200 authors x 20 QA, grouping validated by Day 1's own check C1).
    Author 88 (Day 1 Finding 1: no name anywhere in its block) is skipped --
    there is nothing to fuzzy-match against for that one author."""
    from datasets import load_dataset

    day1 = _import_day1_module()
    full = load_dataset(TOFU_REPO, "full")["train"]
    check("tofu_full_row_count", len(full) == N_TOFU_AUTHORS * QA_PER_AUTHOR,
          f"full split has {len(full)} rows, expected "
          f"{N_TOFU_AUTHORS * QA_PER_AUTHOR} ({N_TOFU_AUTHORS} authors x {QA_PER_AUTHOR})")

    names = []
    unrecovered = []
    for i in range(N_TOFU_AUTHORS):
        block = full[i * QA_PER_AUTHOR:(i + 1) * QA_PER_AUTHOR]
        qs, ans = block["question"], block["answer"]
        name, _coverage = day1.recover_author_name(qs, ans)
        if name == "UNRECOVERED":
            unrecovered.append(i)
        else:
            names.append(name)
    LOG(f"  recovered {len(names)}/{N_TOFU_AUTHORS} TOFU author names "
        f"(unrecovered: {unrecovered or 'none'} -- Day 1 Finding 1 predicts author 88)")
    return names


# ----------------------------------------------------------------------------
# Reference set 2 -- real authors via Wikidata (DECISIONS.md item 2), queried
# through QLever for tractable pagination (see module docstring).
# ----------------------------------------------------------------------------

_SPARQL_PREFIXES = (
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
    "PREFIX wd: <http://www.wikidata.org/entity/> "
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
    "PREFIX wikibase: <http://wikiba.se/ontology#> "
)


def _sparql(query: str, endpoint: str = QLEVER_ENDPOINT, timeout: int = 60,
           max_retries: int = 4) -> dict:
    url = endpoint + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "TrackD-CollisionFilter/1.0 (research project; "
                          "spec-ref ghosts/DECISIONS.md item 2)",
        },
    )
    last_error = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
            wait = 2 ** attempt * 3
            LOG(f"    SPARQL request error ({exc}); retry in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise CheckFailure(f"SPARQL query failed after {max_retries} attempts: {last_error}")


def fetch_writer_subclasses() -> list[str]:
    """QIDs of wd:Q36180 (writer) and every subclass -- DECISIONS.md item 2's
    'or any subclass'. Small, fast query (a few hundred rows)."""
    q = _SPARQL_PREFIXES + (
        f"SELECT DISTINCT ?sub WHERE {{ ?sub wdt:P279* wd:{WRITER_QID} . }}"
    )
    data = _sparql(q)
    qids = [b["sub"]["value"].rsplit("/", 1)[-1] for b in data["results"]["bindings"]]
    return sorted(set(qids))


def fetch_real_authors_for_qid(qid: str) -> list[str]:
    """All human (P31=Q5) English-labelled names with occupation=qid, paged
    with LIMIT/OFFSET (reliable at PAGE_SIZE on QLever; see module docstring).
    Stops when a page returns fewer than PAGE_SIZE rows."""
    names: list[str] = []
    offset = 0
    while True:
        q = _SPARQL_PREFIXES + (
            "SELECT ?label WHERE { "
            f"?person wdt:P31 wd:Q5 ; wdt:P106 wd:{qid} . "
            "?person rdfs:label ?label . FILTER(LANG(?label)=\"en\") "
            f"}} LIMIT {PAGE_SIZE} OFFSET {offset}"
        )
        data = _sparql(q)
        rows = data["results"]["bindings"]
        names.extend(r["label"]["value"] for r in rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY_S)
    return names


def load_real_author_names(cache_path: str, refresh: bool) -> tuple[list[str], dict]:
    """Real-author reference list, cached to `cache_path` (this pull is slow
    -- see module docstring -- and the underlying Wikidata data does not
    change meaningfully between Day 3 runs, so re-fetching every run would be
    wasteful, not more correct)."""
    if os.path.exists(cache_path) and not refresh:
        cached = json.load(open(cache_path, encoding="utf-8"))
        LOG(f"  using cached Wikidata pull from {cached['retrieved_utc']} "
            f"({len(cached['names'])} names) -- pass --refresh-wikidata to re-fetch")
        return cached["names"], cached

    LOG("  fetching writer-subclass QIDs from Wikidata (via QLever)...")
    subclass_qids = fetch_writer_subclasses()
    LOG(f"  {len(subclass_qids)} subclasses of wd:{WRITER_QID} (writer), incl. itself")

    all_names: list[str] = []
    per_qid_counts = {}
    for i, qid in enumerate(subclass_qids):
        names = fetch_real_authors_for_qid(qid)
        per_qid_counts[qid] = len(names)
        all_names.extend(names)
        if (i + 1) % 25 == 0 or i == len(subclass_qids) - 1:
            LOG(f"    {i + 1}/{len(subclass_qids)} subclasses done, "
                f"{len(all_names)} names so far")
        time.sleep(REQUEST_DELAY_S)

    unique_names = sorted(set(all_names))
    record = {
        "source": "Wikidata via QLever (qlever.cs.uni-freiburg.de/api/wikidata), "
                  "same underlying data as query.wikidata.org -- see module "
                  "docstring for why QLever is used",
        "query_definition": "humans (P31=Q5) with occupation (P106) = writer "
                            f"(wd:{WRITER_QID}) or any subclass (P279*)",
        "retrieved_utc": utcnow(),
        "n_subclass_qids": len(subclass_qids),
        "n_names_raw": len(all_names),
        "n_names_unique": len(unique_names),
        "per_qid_counts": per_qid_counts,
        "names": unique_names,
    }
    write_json(cache_path, record)
    LOG(f"  fetched {len(unique_names)} unique real-author names -> {cache_path}")
    return unique_names, record


# ----------------------------------------------------------------------------
# Fuzzy matching (DECISIONS.md item 1).
# ----------------------------------------------------------------------------

# Hyphens become a SPACE (not deleted) so "Anne-Marie" and "Anne Marie" stay
# comparable as two tokens for token_sort_ratio -- deleting the hyphen would
# silently merge them into one token ("AnneMarie") and weaken the match.
# Apostrophes/quotes are deleted, not spaced, since "O'Brien" is one token.
_STRIP_PUNCT = str.maketrans("", "", ".,'’\"")


def normalize_name(name: str) -> str:
    return " ".join(name.replace("-", " ").translate(_STRIP_PUNCT).split()).casefold()


def surname_of(name: str) -> str:
    """Last whitespace-separated token of the normalised name."""
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""


class ReferenceIndex:
    """Normalised reference names + a surname->names index, for the fuzzy +
    exact-surname rule in DECISIONS.md item 1."""

    def __init__(self, names: list[str]):
        self.normalized = sorted({normalize_name(n) for n in names if n and n != "UNRECOVERED"})
        self.by_surname: dict[str, list[str]] = {}
        for n in self.normalized:
            self.by_surname.setdefault(surname_of(n), []).append(n)

    def find_collision(self, ghost_name: str, threshold: int = FUZZY_THRESHOLD):
        """Returns (matched_ref_name, score, reason) or None."""
        from rapidfuzz import fuzz

        gnorm = normalize_name(ghost_name)
        gsurname = surname_of(ghost_name)
        if gsurname and gsurname in self.by_surname:
            return (self.by_surname[gsurname][0], 100, "exact_surname")

        best_name, best_score = None, -1
        for ref in self.normalized:
            score = fuzz.token_sort_ratio(gnorm, ref)
            if score > best_score:
                best_name, best_score = ref, score
        if best_score >= threshold:
            return (best_name, best_score, "fuzzy_token_sort_ratio")
        return None


# ----------------------------------------------------------------------------
# Self-test -- offline, no network. Run this first.
# ----------------------------------------------------------------------------


def selftest() -> int:
    print("Track D Day 3 -- offline self-test (no network calls)\n")
    fails = 0

    def t(name, cond, got=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + str(got) if got else ''}")
        if not cond:
            fails += 1

    t("normalize_name strips punctuation/case",
      normalize_name("O'Brien-Smith, Jr.") == normalize_name("obrien smith jr"))
    t("normalize_name collapses whitespace",
      normalize_name("  Ana   Silva  ") == "ana silva")
    t("surname_of picks last token",
      surname_of("Marisol Elena Vasquez") == "vasquez")

    ref = ReferenceIndex(["Erick Gustafsson", "Asha Majaliwa", "J.K. Rowling"])
    t("exact match collides", ref.find_collision("Erick Gustafsson") is not None)
    t("exact surname collides even with different given name",
      ref.find_collision("Zoltan Gustafsson") is not None)
    t("unrelated name does not collide",
      ref.find_collision("Priya Anand Chandrasekaran") is None)
    hit = ref.find_collision("Erik Gustafson")  # one-letter respellings
    t("close respelling trips the fuzzy threshold", hit is not None, hit)

    t("token_sort_ratio is order-insensitive (DECISIONS.md item 1 rationale)",
      ReferenceIndex(["Marisol Elena Vasquez"]).find_collision("Elena Marisol Vasquez") is not None)

    print(f"\n{'SELF-TEST PASSED' if fails == 0 else f'SELF-TEST FAILED ({fails})'}")
    return 0 if fails == 0 else 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default="ghosts")
    ap.add_argument("--candidates", default=None,
                    help="Defaults to <outdir>/candidates_raw.jsonl")
    ap.add_argument("--skip-wikidata", action="store_true",
                    help="TOFU-names-only check, for a quick local run "
                         "without the (slow) Wikidata pull. Documented as a "
                         "partial run, not a substitute for the full check.")
    ap.add_argument("--refresh-wikidata", action="store_true",
                    help="Re-fetch the Wikidata reference list even if a "
                         "cached one exists.")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    candidates_path = args.candidates or os.path.join(args.outdir, "candidates_raw.jsonl")
    wikidata_cache_path = os.path.join(args.outdir, "wikidata_authors_cache.json")

    started = utcnow()
    LOG("# Track D -- Day 3 collision filter report")
    LOG("")
    LOG(f"- started (UTC): `{started}`")
    LOG(f"- git commit: `{git_commit()}`" + ("  **(working tree dirty)**" if git_dirty() else ""))
    LOG(f"- python: `{platform.python_version()}` on `{platform.platform()}`")
    LOG(f"- fuzzy threshold: `{FUZZY_THRESHOLD}` (ghosts/DECISIONS.md item 1)")
    LOG(f"- spec: pilot_0_1_execution_spec.md step 3 -- \"String/fuzzy match "
        f"against all 200 TOFU author names AND against a real-author list\"")
    LOG("")

    # ------------------------------------------------------------ load candidates
    LOG("## 1. Load Day 2 candidates")
    check("candidates_file_exists", os.path.exists(candidates_path),
          f"{candidates_path} found" if os.path.exists(candidates_path)
          else f"{candidates_path} not found -- run Day 2 (day2_generate.py) first")
    rows = [json.loads(l) for l in open(candidates_path, encoding="utf-8")]
    ghost_authors: dict[int, str] = {}
    for r in rows:
        ghost_authors.setdefault(r["author_id"], r["author_name"])
    check("candidate_row_count", len(rows) > 0, f"{len(rows)} rows, {len(ghost_authors)} authors")
    LOG(f"  {len(rows)} QA rows across {len(ghost_authors)} ghost authors")
    LOG("")

    # ------------------------------------------------------------ reference set 1: TOFU
    LOG("## 2. Reference set 1 -- all 200 TOFU author names")
    tofu_names = load_tofu_author_names()
    LOG("")

    # ------------------------------------------------------------ reference set 2: Wikidata
    wikidata_names: list[str] = []
    wikidata_meta = None
    if args.skip_wikidata:
        LOG("## 3. Reference set 2 -- Wikidata real authors")
        LOG("  [SKIPPED via --skip-wikidata] -- this run only checks against "
            "TOFU's 200 authors, NOT the full spec-required check. Do not "
            "treat this run's collision_report.md as the final Day 3 result.")
        LOG("")
    else:
        LOG("## 3. Reference set 2 -- Wikidata real authors (via QLever; see "
            "module docstring)")
        wikidata_names, wikidata_meta = load_real_author_names(
            wikidata_cache_path, refresh=args.refresh_wikidata)
        LOG(f"  query: {wikidata_meta['query_definition']}")
        LOG(f"  retrieved (UTC): {wikidata_meta['retrieved_utc']}")
        LOG(f"  {wikidata_meta['n_subclass_qids']} writer-subclass QIDs, "
            f"{wikidata_meta['n_names_unique']} unique real-author names")
        LOG("")

    reference_names = tofu_names + wikidata_names
    ref_index = ReferenceIndex(reference_names)
    LOG(f"  combined reference set: {len(ref_index.normalized)} unique normalised names")
    LOG("")

    # ------------------------------------------------------------ collision check
    LOG("## 4. Collision check -- each ghost author vs. the combined reference set")
    collisions = {}
    for aid, name in sorted(ghost_authors.items()):
        hit = ref_index.find_collision(name)
        if hit:
            matched, score, reason = hit
            collisions[aid] = {"ghost_name": name, "matched_reference": matched,
                               "score": score, "reason": reason}
            LOG(f"  author {aid:02d} \"{name}\": [COLLISION] matched \"{matched}\" "
                f"(score={score}, {reason})")
        else:
            LOG(f"  author {aid:02d} \"{name}\": [OK] no collision")
    LOG("")
    LOG(f"  {len(collisions)}/{len(ghost_authors)} authors flagged for collision "
        f"-- ALL {QA_PER_AUTHOR} rows of each are dropped, per DECISIONS.md item 1 "
        f"('reject a ghost name')")
    LOG("")

    # ------------------------------------------------------------ internal duplicate check
    # Addition beyond spec step 3 / DECISIONS.md -- not a pre-registered
    # requirement, logged as such. Checks the 30 ghost authors don't
    # accidentally collide with EACH OTHER.
    LOG("## 5. Internal duplicate check (addition, not a spec requirement)")
    from rapidfuzz import fuzz
    internal_hits = []
    ids = sorted(ghost_authors)
    for i, aid_a in enumerate(ids):
        for aid_b in ids[i + 1:]:
            na, nb = normalize_name(ghost_authors[aid_a]), normalize_name(ghost_authors[aid_b])
            score = fuzz.token_sort_ratio(na, nb)
            same_surname = surname_of(ghost_authors[aid_a]) == surname_of(ghost_authors[aid_b]) \
                and surname_of(ghost_authors[aid_a]) != ""
            if score >= FUZZY_THRESHOLD or same_surname:
                internal_hits.append((aid_a, aid_b, score))
                LOG(f"  [HIT] author {aid_a:02d} \"{ghost_authors[aid_a]}\" vs "
                    f"author {aid_b:02d} \"{ghost_authors[aid_b]}\" (score={score})")
    if not internal_hits:
        LOG("  none found")
    LOG("")

    # ------------------------------------------------------------ write filtered candidates
    LOG("## 6. Write filtered candidate set")
    surviving_ids = {aid for aid in ghost_authors if aid not in collisions}
    filtered_rows = [r for r in rows if r["author_id"] in surviving_ids]
    filtered_path = os.path.join(args.outdir, "candidates_filtered.jsonl")
    with open(filtered_path, "w", encoding="utf-8") as fh:
        for r in filtered_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    LOG(f"  {len(filtered_rows)} rows across {len(surviving_ids)} surviving authors "
        f"-> {filtered_path}")
    LOG(f"  ({len(collisions)} authors / {len(collisions) * QA_PER_AUTHOR} rows dropped)")
    LOG("")

    # ------------------------------------------------------------ summary
    LOG("## 7. Summary")
    LOG(f"- finished (UTC): `{utcnow()}`")
    LOG(f"- surviving authors: {len(surviving_ids)}/{len(ghost_authors)} "
        f"({len(filtered_rows)}/{len(rows)} rows)")
    if args.skip_wikidata:
        LOG("- **Status: PARTIAL (--skip-wikidata) -- re-run without that flag "
            "before treating this as the Day 3 result.**")
    elif len(surviving_ids) * QA_PER_AUTHOR < 400:
        LOG(f"- **Status: WARNING -- only {len(filtered_rows)} rows survive, "
            f"below the 400 needed for Day 5's trim. Consider Day 2 backfill "
            f"of the dropped author_ids.**")
    else:
        LOG("- **Status: complete. Next: Day 4 validation battery, spec section 2.2 step 4.**")

    LOG.dump(os.path.join(args.outdir, "collision_report.md"))
    print(f"\nDay 3 outputs written to: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as e:
        LOG("")
        LOG(f"ABORTED: {e}")
        print(f"\nABORTED: {e}", file=sys.stderr)
        sys.exit(2)
