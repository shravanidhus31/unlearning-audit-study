#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track D / Day 5 -- Trim to 400, then re-validate.
Implements execution spec Pilot 1, section 2.2, STEP 5:

    "Trim to best 400 by dropping the worst length/perplexity outliers, then
     re-run the validation on the final 400 (trimming can shift
     distributions)."

Trim rule -- ghosts/DECISIONS.md item 6, pre-registered BEFORE this script
computed which author it would drop:
    21 authors (420 rows) currently survive Day 3. Drop exactly ONE whole
    author (not a row-level mix -- every surviving ghost stays a complete
    20-QA identity): the one with the highest combined |z-score| of
    (mean answer length, mean answer perplexity under the base
    meta-llama/Llama-3.2-1B-Instruct) against holdout10's distribution.

This script is self-contained -- it recomputes length and perplexity for the
ghost set and holdout10 itself (reusing day4_validation.py's functions
directly, not reimplementing them) rather than depending on Day 4 having
persisted intermediate arrays. The perplexity recomputation costs a few GPU
minutes; the alternative (Day 4 persisting raw arrays a prior run already
produced) would save that time but adds a fragile cross-script dependency
that every other Day-N script in this project avoids by re-deriving from
saved FILES, not from another script's in-memory state.

After trimming, it re-runs Day 4's full 3-test battery on the resulting 400
via day4_validation.main()-equivalent logic, since spec explicitly requires
re-validating after a trim ("trimming can shift distributions").

Deliverables written to --outdir (default: ghosts/):
    final_400.jsonl        the trimmed, final ghost candidate set
    trim_report.md         the trim decision (all 21 authors' scores, which
                            one was dropped and why) + the full re-validation

USAGE
    python day5_trim.py --selftest       # offline, no network/GPU
    python day5_trim.py                  # full run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Import Day 4's functions directly -- do not reimplement the statistics or
# the perplexity/SBERT machinery a second time.
# ----------------------------------------------------------------------------


def _import_day4_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "day4_validation.py")
    if not os.path.exists(path):
        raise RuntimeError(f"day4_validation.py not found at {path} -- needed "
                           f"to reuse its validated test functions")
    spec = importlib.util.spec_from_file_location("day4_validation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D4 = None  # populated at the top of main() / selftest() as needed

TARGET_N_AUTHORS = 20   # spec's "best 400" = 20 authors x 20 QA
QA_PER_AUTHOR = 20

# ----------------------------------------------------------------------------
# Small utilities -- identical in spirit to day1-day4.
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


class Log:
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
# Trim decision -- ghosts/DECISIONS.md item 6.
# ----------------------------------------------------------------------------


def zscore(value: float, mean: float, sd: float) -> float:
    return 0.0 if sd == 0 else (value - mean) / sd


def choose_author_to_drop(candidates_path: str, outdir: str, hf_token: str | None) -> dict:
    """Returns the trim decision: per-author scores and which one to drop.
    Recomputes length + perplexity for every ghost row and for holdout10,
    exactly as Day 4 does (via the imported functions), then aggregates to
    per-author means for the DECISIONS.md item 6 z-score rule."""
    rows = [json.loads(l) for l in open(candidates_path, encoding="utf-8")]
    answers = [r["answer"] for r in rows]
    author_ids = [r["author_id"] for r in rows]

    holdout_answers = D4.load_tofu_answers("holdout10")

    LOG("  computing token lengths (ghost + holdout10)...")
    _, gh_len, ho_len = D4.test_token_length(answers, holdout_answers)
    ho_len_mean, ho_len_sd = float(ho_len.mean()), float(ho_len.std(ddof=1))

    LOG("  computing perplexities under the base model (ghost + holdout10) -- "
        "this is the slow step...")
    _, gh_ppl_raw, ho_ppl = D4.test_perplexity(answers, holdout_answers, hf_token)
    ho_ppl_mean, ho_ppl_sd = float(ho_ppl.mean()), float(ho_ppl.std(ddof=1))

    per_author = {}
    for aid in sorted(set(author_ids)):
        mask = np.array([a == aid for a in author_ids])
        author_len = gh_len[mask]
        author_ppl = gh_ppl_raw[mask]
        author_ppl = author_ppl[~np.isnan(author_ppl)]
        mean_len = float(author_len.mean())
        mean_ppl = float(author_ppl.mean()) if len(author_ppl) else float("nan")
        z_len = zscore(mean_len, ho_len_mean, ho_len_sd)
        z_ppl = zscore(mean_ppl, ho_ppl_mean, ho_ppl_sd) if not np.isnan(mean_ppl) else 0.0
        combined = (abs(z_len) + abs(z_ppl)) / 2.0
        per_author[int(aid)] = {
            "mean_length": mean_len, "z_length": z_len,
            "mean_perplexity": mean_ppl, "z_perplexity": z_ppl,
            "combined_abs_z": combined,
        }

    worst_aid = max(per_author, key=lambda a: per_author[a]["combined_abs_z"])
    return {
        "holdout_length_mean": ho_len_mean, "holdout_length_sd": ho_len_sd,
        "holdout_perplexity_mean": ho_ppl_mean, "holdout_perplexity_sd": ho_ppl_sd,
        "per_author": per_author,
        "drop_author_id": worst_aid,
    }


# ----------------------------------------------------------------------------
# Self-test -- offline, no network/GPU. Run this first.
# ----------------------------------------------------------------------------


def selftest() -> int:
    print("Track D Day 5 -- offline self-test (no network, no GPU)\n")
    fails = 0

    def t(name, cond, got=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + str(got) if got else ''}")
        if not cond:
            fails += 1

    t("zscore(mean, mean, sd) == 0", zscore(50.0, 50.0, 5.0) == 0.0)
    t("zscore one sd above == 1.0", zscore(55.0, 50.0, 5.0) == 1.0)
    t("zscore handles sd=0 without crashing", zscore(10.0, 5.0, 0.0) == 0.0)

    # Synthetic 3-author trim decision, bypassing the real (network/GPU) data
    # loading -- exercises the same aggregation and "pick the worst" logic
    # choose_author_to_drop() uses.
    ho_len_mean, ho_len_sd = 42.0, 10.0
    ho_ppl_mean, ho_ppl_sd = 30.0, 8.0
    fake_authors = {
        1: {"mean_length": 43.0, "mean_perplexity": 31.0},   # close to holdout -- keep
        2: {"mean_length": 90.0, "mean_perplexity": 30.0},   # length way off -- worst
        3: {"mean_length": 41.0, "mean_perplexity": 29.0},   # close -- keep
    }
    scored = {}
    for aid, v in fake_authors.items():
        z_len = zscore(v["mean_length"], ho_len_mean, ho_len_sd)
        z_ppl = zscore(v["mean_perplexity"], ho_ppl_mean, ho_ppl_sd)
        scored[aid] = (abs(z_len) + abs(z_ppl)) / 2.0
    worst = max(scored, key=lambda a: scored[a])
    t("picks the genuinely worst synthetic author", worst == 2, scored)

    t(f"TARGET_N_AUTHORS x QA_PER_AUTHOR == 400",
      TARGET_N_AUTHORS * QA_PER_AUTHOR == 400)

    print(f"\n{'SELF-TEST PASSED' if fails == 0 else f'SELF-TEST FAILED ({fails})'}")
    return 0 if fails == 0 else 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    global D4
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default="ghosts")
    ap.add_argument("--candidates", default=None,
                    help="Defaults to <outdir>/candidates_filtered.jsonl (Day 3's survivors).")
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    D4 = _import_day4_module()
    candidates_path = args.candidates or os.path.join(args.outdir, "candidates_filtered.jsonl")

    started = utcnow()
    LOG("# Track D -- Day 5 trim report")
    LOG("")
    LOG(f"- started (UTC): `{started}`")
    LOG(f"- git commit: `{git_commit()}`" + ("  **(working tree dirty)**" if git_dirty() else ""))
    LOG(f"- python: `{platform.python_version()}` on `{platform.platform()}`")
    LOG(f"- candidates file: `{candidates_path}`")
    LOG(f"- trim rule: ghosts/DECISIONS.md item 6 -- drop the one author with the "
        f"highest combined |z-score| of mean length + mean perplexity vs holdout10")
    LOG("")

    check("candidates_file_exists", os.path.exists(candidates_path),
          f"{candidates_path} found" if os.path.exists(candidates_path)
          else f"{candidates_path} not found -- run Day 3 first")
    rows = [json.loads(l) for l in open(candidates_path, encoding="utf-8")]
    author_ids = sorted({r["author_id"] for r in rows})
    n_authors = len(author_ids)
    check("author_count_matches_trim_math", n_authors == TARGET_N_AUTHORS + 1,
          f"{n_authors} authors survive Day 3 "
          f"({'expected exactly ' + str(TARGET_N_AUTHORS + 1) + ' for a clean 1-author trim to ' + str(TARGET_N_AUTHORS*QA_PER_AUTHOR) if n_authors != TARGET_N_AUTHORS + 1 else 'matches'})",
          fatal=False)
    LOG("")

    hf_token = os.environ.get(args.hf_token_env)
    check("hf_token_present", bool(hf_token),
          f"{args.hf_token_env} is set" if hf_token
          else f"{args.hf_token_env} not set", fatal=False)

    LOG("## 1. Score all surviving authors")
    decision = choose_author_to_drop(candidates_path, args.outdir, hf_token)
    LOG(f"  holdout10 length: mean={decision['holdout_length_mean']:.2f} "
        f"sd={decision['holdout_length_sd']:.2f}")
    LOG(f"  holdout10 perplexity: mean={decision['holdout_perplexity_mean']:.2f} "
        f"sd={decision['holdout_perplexity_sd']:.2f}")
    LOG("")
    LOG("  | author_id | mean length | z(length) | mean perplexity | z(perplexity) | combined |z| |")
    LOG("  |---|---|---|---|---|---|")
    for aid in sorted(decision["per_author"]):
        v = decision["per_author"][aid]
        marker = " **<- DROPPED**" if aid == decision["drop_author_id"] else ""
        LOG(f"  | {aid} | {v['mean_length']:.2f} | {v['z_length']:+.3f} | "
            f"{v['mean_perplexity']:.2f} | {v['z_perplexity']:+.3f} | "
            f"{v['combined_abs_z']:.3f}{marker} |")
    drop_aid = decision["drop_author_id"]
    LOG("")
    LOG(f"  **Dropping author {drop_aid}** (highest combined |z-score| = "
        f"{decision['per_author'][drop_aid]['combined_abs_z']:.3f}) per "
        f"ghosts/DECISIONS.md item 6.")
    LOG("")

    LOG("## 2. Write final_400.jsonl")
    final_rows = [r for r in rows if r["author_id"] != drop_aid]
    final_path = os.path.join(args.outdir, "final_400.jsonl")
    with open(final_path, "w", encoding="utf-8") as fh:
        for r in final_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    check("final_400_row_count", len(final_rows) == TARGET_N_AUTHORS * QA_PER_AUTHOR,
          f"{len(final_rows)} rows -> {final_path}")
    LOG("")

    LOG("## 3. Re-run Day 4's full battery on the final 400 (spec 2.2 step 5)")
    final_answers = [r["answer"] for r in final_rows]
    holdout_answers = D4.load_tofu_answers("holdout10")
    forget_answers = D4.load_tofu_answers("forget10")

    r1, _, _ = D4.test_token_length(final_answers, holdout_answers)
    LOG(f"  Test 1 (length): d={r1['cohens_d']:+.3f}  KS p={r1['ks_pvalue']:.3g}  -> {r1['verdict']}")

    r2, _, _ = D4.test_perplexity(final_answers, holdout_answers, hf_token)
    LOG(f"  Test 2 (perplexity): d={r2['cohens_d']:+.3f}  KS p={r2['ks_pvalue']:.3g}  -> {r2['verdict']}")

    r3, _, _, _ = D4.test_sbert(final_answers, holdout_answers, forget_answers, args.outdir)
    LOG(f"  Test 3 (sbert): d(ghost,holdout)={r3['d_ghost_holdout']:.4f}  "
        f"limit={r3['limit']:.4f}  -> {r3['verdict']}")
    LOG("")

    LOG("## Summary")
    final_results = {"token_length": r1, "perplexity": r2, "sbert": r3}
    all_pass = all(v["verdict"] == "PASS" for v in final_results.values())
    LOG(f"- finished (UTC): `{utcnow()}`")
    LOG(f"- dropped author: {drop_aid}")
    for name, r in final_results.items():
        LOG(f"- {name}: {r['verdict']}")
    if all_pass:
        LOG("- **Status: PASS. Final 400 satisfies all 3 Day 4 tests. Next: Day 6 "
            "freeze + hash, spec section 2.2 step 6.**")
    else:
        LOG("- **Status: FAIL (post-trim). Spec 2.2 step 5's trim did not fully resolve "
            "the Day 4 gap -- decide the response deliberately (spec permits one "
            "regenerate-and-retest cycle; this trim was not that cycle).**")

    LOG.dump(os.path.join(args.outdir, "trim_report.md"))
    print(f"\nDay 5 outputs written to: {os.path.abspath(args.outdir)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as e:
        LOG("")
        LOG(f"ABORTED: {e}")
        print(f"\nABORTED: {e}", file=sys.stderr)
        sys.exit(2)
