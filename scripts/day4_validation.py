#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track D / Day 4 -- Distribution-match validation.
Implements execution spec Pilot 1, section 2.2, STEP 4:

    "Distribution-match validation (pre-registered acceptance tests). All
     tests compare ghost answers vs holdout answers:
     - Token-length: two-sample KS test, require p > 0.05 or |Cohen's d| < 0.2.
     - Perplexity under the base meta-llama/Llama-3.2-1B-Instruct (NOT the
       TOFU-tuned model -- the base model saw neither set): KS p > 0.05 or
       |d| < 0.2.
     - Embedding check: SBERT centroid distance ghost<->holdout <= 1.25 x
       centroid distance holdout<->forget10. (Guards against the generator
       drifting in style.)"

Three tests, all against holdout10 (ghosts/DECISIONS.md item 4). This is the
MANDATORY validation -- distinct from Day 2's own length check, which is an
informal convenience gate on raw generation output, not this pre-registered
battery on the Day-3-filtered candidate set.

Deliverables written to --outdir (default: ghosts/):
    validation_report.md      every test, its statistic, its verdict
    sbert_checkpoint_pin.json the SBERT revision hash, pinned at first use
                               (ghosts/DECISIONS.md item 3)

USAGE
    python day4_validation.py --selftest              # offline, no network/GPU
    python day4_validation.py --candidates ghosts/candidates_filtered.jsonl
"""

from __future__ import annotations

import argparse
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
# FROZEN CONSTANTS -- ghosts/DECISIONS.md items 3-4, spec 2.2 step 4.
# ----------------------------------------------------------------------------
TOFU_REPO = "locuslab/TOFU"
TOKENIZER_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"  # same as Day 1/2
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # NOT the TOFU-tuned model
SBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # DECISIONS.md item 3

LENGTH_TARGET_SPLIT = "holdout10"       # DECISIONS.md item 4
SBERT_DISTANCE_RATIO_MAX = 1.25         # spec 2.2 step 4
KS_P_THRESHOLD = 0.05
COHENS_D_THRESHOLD = 0.2

# ----------------------------------------------------------------------------
# Small utilities -- identical in spirit to day1/day2/day3.
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
# Shared statistics -- same formulas Day 1/2 already used, so numbers agree
# across scripts by construction, not by coincidence.
# ----------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s_pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / s_pooled)


def ks_and_d(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy import stats
    ks = stats.ks_2samp(a, b)
    d = cohens_d(a, b)
    verdict = "PASS" if (ks.pvalue > KS_P_THRESHOLD or abs(d) < COHENS_D_THRESHOLD) else "FAIL"
    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "cohens_d": d, "ks_pvalue": float(ks.pvalue), "verdict": verdict,
    }


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cosine_similarity. ghosts/DECISIONS.md item 3 amendment."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(u, v) / denom)


# ----------------------------------------------------------------------------
# Data loading.
# ----------------------------------------------------------------------------


def load_ghost_answers(path: str) -> list[str]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r["answer"] for r in rows]


def load_tofu_answers(split: str) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(TOFU_REPO, split)["train"]
    return list(ds["answer"])


def load_ghost_providers(candidates_path: str, outdir: str) -> list[str]:
    """Per-row generator provider (groq/anthropic/gemini), looked up from
    each row's checkpoint file. Addition beyond spec, promised in D-005's
    'known, stated risk' section: if Day 4 fails, attribute it to a specific
    generator where possible rather than only reporting an aggregate."""
    rows = [json.loads(l) for l in open(candidates_path, encoding="utf-8")]
    ckpt_dir = os.path.join(outdir, "checkpoints")
    cache: dict[int, str] = {}
    providers = []
    for r in rows:
        aid = r["author_id"]
        if aid not in cache:
            ckpt = json.load(open(os.path.join(ckpt_dir, f"author_{aid:02d}.json"), encoding="utf-8"))
            cache[aid] = ckpt.get("_meta", {}).get("provider", "unknown")
        providers.append(cache[aid])
    return providers


def per_generator_breakdown(label: str, values: np.ndarray, providers: list[str],
                            reference: np.ndarray) -> None:
    """Logs a KS/Cohen's-d breakdown of `values` (aligned 1:1 with
    `providers`) against `reference`, one row per distinct generator. NaNs in
    `values` are dropped per subgroup, after slicing -- see test_perplexity's
    docstring for why they can't be dropped before slicing."""
    values = np.asarray(values)
    for provider in sorted(set(providers)):
        mask = np.array([p == provider for p in providers])
        subset = values[mask]
        subset = subset[~np.isnan(subset)]
        if len(subset) < 2:
            LOG(f"  {label} / {provider}: n={len(subset)} -- too few to test")
            continue
        r = ks_and_d(subset, reference)
        LOG(f"  {label} / {provider}: n={r['n_a']}  mean={r['mean_a']:.2f}  "
            f"d={r['cohens_d']:+.3f}  KS p={r['ks_pvalue']:.3g}  -> {r['verdict']}")


def sbert_per_generator_breakdown(gh_emb: np.ndarray, providers: list[str],
                                  c_holdout: np.ndarray, d_holdout_forget: float) -> None:
    limit = SBERT_DISTANCE_RATIO_MAX * d_holdout_forget
    for provider in sorted(set(providers)):
        mask = np.array([p == provider for p in providers])
        subset = gh_emb[mask]
        if len(subset) < 1:
            continue
        c_provider = subset.mean(axis=0)
        d = cosine_distance(c_provider, c_holdout)
        verdict = "PASS" if d <= limit else "FAIL"
        LOG(f"  sbert / {provider}: n={mask.sum()}  d(ghost,holdout)={d:.4f}  "
            f"limit={limit:.4f}  -> {verdict}")


# ----------------------------------------------------------------------------
# Test 1 -- token length. Same tokenizer/convention as Day 1's length_stats.json
# and Day 2's own convenience gate, but this run is the MANDATORY one.
# ----------------------------------------------------------------------------


def test_token_length(ghost_answers: list[str], holdout_answers: list[str]):
    """Returns (result_dict, gh_len, ho_len) -- the raw per-answer arrays are
    returned too so main() can slice gh_len by generator for the D-005
    per-generator diagnostic without re-tokenizing everything a second time."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    gh_len = np.array([len(tok(a, add_special_tokens=True)["input_ids"]) for a in ghost_answers], dtype=float)
    ho_len = np.array([len(tok(a, add_special_tokens=True)["input_ids"]) for a in holdout_answers], dtype=float)
    return ks_and_d(gh_len, ho_len), gh_len, ho_len


# ----------------------------------------------------------------------------
# Test 2 -- perplexity under the BASE (non-TOFU-tuned) Llama-3.2-1B-Instruct.
# ----------------------------------------------------------------------------


def compute_perplexities(texts: list[str], model, tokenizer, device: str) -> np.ndarray:
    import torch
    model.eval()
    ppls = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            if enc["input_ids"].shape[1] < 2:
                ppls.append(float("nan"))  # degenerate (empty/near-empty) answer
                continue
            out = model(**enc, labels=enc["input_ids"])
            ppls.append(float(torch.exp(out.loss).item()))
    return np.array(ppls, dtype=float)


def test_perplexity(ghost_answers: list[str], holdout_answers: list[str], hf_token: str | None):
    """Returns (result_dict, gh_ppl_raw, ho_ppl_clean). gh_ppl_raw keeps NaNs
    in place (same length/order as ghost_answers) so main() can slice it by
    generator for the D-005 diagnostic and only THEN drop NaNs per subgroup --
    dropping them here first would break that index alignment."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG(f"  device: {device}" + ("" if device == "cuda" else "  (no GPU detected -- this will be slow)"))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    gh_ppl_raw = compute_perplexities(ghost_answers, model, tok, device)
    ho_ppl_raw = compute_perplexities(holdout_answers, model, tok, device)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    n_dropped_gh = int(np.isnan(gh_ppl_raw).sum())
    n_dropped_ho = int(np.isnan(ho_ppl_raw).sum())
    if n_dropped_gh or n_dropped_ho:
        LOG(f"  [WARN] dropped {n_dropped_gh} ghost / {n_dropped_ho} holdout "
            f"answers with < 2 tokens (perplexity undefined)")
    gh_ppl = gh_ppl_raw[~np.isnan(gh_ppl_raw)]
    ho_ppl = ho_ppl_raw[~np.isnan(ho_ppl_raw)]

    return ks_and_d(gh_ppl, ho_ppl), gh_ppl_raw, ho_ppl


# ----------------------------------------------------------------------------
# Test 3 -- SBERT centroid cosine distance.
# ----------------------------------------------------------------------------


def resolve_sbert_revision(outdir: str) -> str:
    """First use pins the revision; every later run reads the same pin.
    ghosts/DECISIONS.md item 3: 'pinned by revision hash, recorded... at
    first use' -- guards against the checkpoint silently changing mid-study."""
    pin_path = os.path.join(outdir, "sbert_checkpoint_pin.json")
    if os.path.exists(pin_path):
        pin = json.load(open(pin_path, encoding="utf-8"))
        LOG(f"  using pinned SBERT revision from {pin['pinned_utc']}: {pin['revision']}")
        return pin["revision"]

    from huggingface_hub import HfApi
    revision = HfApi().model_info(SBERT_MODEL_ID).sha
    write_json(pin_path, {
        "model_id": SBERT_MODEL_ID, "revision": revision, "pinned_utc": utcnow(),
    })
    LOG(f"  pinned SBERT revision (first use): {revision} -> {pin_path}")
    return revision


def test_sbert(ghost_answers: list[str], holdout_answers: list[str],
               forget_answers: list[str], outdir: str):
    """Returns (result_dict, gh_embeddings, c_holdout, d_holdout_forget) --
    per-answer ghost embeddings (not just the centroid) are returned so
    main() can compute a per-generator centroid for the D-005 diagnostic
    without re-encoding or re-downloading the model."""
    from sentence_transformers import SentenceTransformer

    revision = resolve_sbert_revision(outdir)
    model = SentenceTransformer(SBERT_MODEL_ID, revision=revision)

    gh_emb = model.encode(ghost_answers, show_progress_bar=False, convert_to_numpy=True)
    ho_emb = model.encode(holdout_answers, show_progress_bar=False, convert_to_numpy=True)
    fg_emb = model.encode(forget_answers, show_progress_bar=False, convert_to_numpy=True)

    c_ghost = gh_emb.mean(axis=0)
    c_holdout = ho_emb.mean(axis=0)
    c_forget = fg_emb.mean(axis=0)

    d_ghost_holdout = cosine_distance(c_ghost, c_holdout)
    d_holdout_forget = cosine_distance(c_holdout, c_forget)
    limit = SBERT_DISTANCE_RATIO_MAX * d_holdout_forget
    verdict = "PASS" if d_ghost_holdout <= limit else "FAIL"

    result = {
        "sbert_revision": revision,
        "d_ghost_holdout": d_ghost_holdout,
        "d_holdout_forget": d_holdout_forget,
        "limit": limit,
        "ratio_max": SBERT_DISTANCE_RATIO_MAX,
        "verdict": verdict,
    }
    return result, gh_emb, c_holdout, d_holdout_forget


# ----------------------------------------------------------------------------
# Self-test -- offline, no network/GPU. Run this first.
# ----------------------------------------------------------------------------


def selftest() -> int:
    print("Track D Day 4 -- offline self-test (no network, no GPU)\n")
    fails = 0

    def t(name, cond, got=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  -> ' + str(got) if got else ''}")
        if not cond:
            fails += 1

    rng = np.random.default_rng(42)
    same = rng.normal(50, 10, 200)
    t("cohens_d(x, x-copy) ~= 0", abs(cohens_d(same, same.copy())) < 1e-9)

    shifted = same + 3 * same.std()
    t("cohens_d detects a large shift", abs(cohens_d(same, shifted)) > 2)

    r1 = ks_and_d(same, same.copy())
    t("ks_and_d PASSes on identical distributions", r1["verdict"] == "PASS", r1)

    t("cosine_distance(v, v) == 0", abs(cosine_distance(np.array([1., 2., 3.]),
                                                          np.array([1., 2., 3.]))) < 1e-9)
    t("cosine_distance(v, -v) == 2", abs(cosine_distance(np.array([1., 0.]),
                                                           np.array([-1., 0.])) - 2.0) < 1e-9)
    t("cosine_distance(orthogonal) == 1", abs(cosine_distance(np.array([1., 0.]),
                                                                np.array([0., 1.])) - 1.0) < 1e-9)
    t("cosine_distance is scale-invariant",
      abs(cosine_distance(np.array([1., 2.]), np.array([2., 4.])) -
          cosine_distance(np.array([1., 2.]), np.array([20., 40.]))) < 1e-9)

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
                    help="Defaults to <outdir>/candidates_filtered.jsonl "
                         "(the Day 3 survivors) -- pass candidates_raw.jsonl "
                         "or final_400.jsonl explicitly to validate a "
                         "different stage.")
    ap.add_argument("--hf-token-env", default="HF_TOKEN",
                    help="Env var holding the HF token for the gated base model.")
    ap.add_argument("--skip-perplexity", action="store_true",
                    help="Skip test 2 (no GPU / no gated access yet). "
                         "Documented as a partial run.")
    ap.add_argument("--skip-sbert", action="store_true",
                    help="Skip test 3. Documented as a partial run.")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    candidates_path = args.candidates or os.path.join(args.outdir, "candidates_filtered.jsonl")

    started = utcnow()
    LOG("# Track D -- Day 4 validation report")
    LOG("")
    LOG(f"- started (UTC): `{started}`")
    LOG(f"- git commit: `{git_commit()}`" + ("  **(working tree dirty)**" if git_dirty() else ""))
    LOG(f"- python: `{platform.python_version()}` on `{platform.platform()}`")
    LOG(f"- candidates file: `{candidates_path}`")
    LOG(f"- spec: pilot_0_1_execution_spec.md step 4 -- three pre-registered tests, "
        f"ghost vs holdout10")
    LOG(f"- acceptance rule (all 3 tests): KS p > {KS_P_THRESHOLD} OR "
        f"|Cohen's d| < {COHENS_D_THRESHOLD} (tests 1-2); SBERT distance <= "
        f"{SBERT_DISTANCE_RATIO_MAX}x (test 3)")
    LOG("")

    check("candidates_file_exists", os.path.exists(candidates_path),
          f"{candidates_path} found" if os.path.exists(candidates_path)
          else f"{candidates_path} not found -- run Day 3 first")
    ghost_answers = load_ghost_answers(candidates_path)
    check("ghost_answer_count", len(ghost_answers) > 0, f"{len(ghost_answers)} ghost answers")
    ghost_providers = load_ghost_providers(candidates_path, args.outdir)
    provider_counts = {p: ghost_providers.count(p) for p in sorted(set(ghost_providers))}
    LOG(f"  by generator: {provider_counts}")
    LOG("")

    LOG("## Load TOFU reference splits")
    holdout_answers = load_tofu_answers("holdout10")
    forget_answers = load_tofu_answers("forget10")
    LOG(f"  holdout10: {len(holdout_answers)} answers, forget10: {len(forget_answers)} answers")
    LOG("")

    results = {}

    LOG("## Test 1 -- Token length (KS test / Cohen's d)")
    r1, gh_len, ho_len = test_token_length(ghost_answers, holdout_answers)
    results["token_length"] = r1
    LOG(f"  ghost mean {r1['mean_a']:.2f}  holdout10 mean {r1['mean_b']:.2f}")
    LOG(f"  Cohen's d = {r1['cohens_d']:+.3f}   KS p = {r1['ks_pvalue']:.3g}   -> {r1['verdict']}")
    LOG("  -- by generator (D-005 diagnostic, not a spec requirement) --")
    per_generator_breakdown("length", gh_len, ghost_providers, ho_len)
    LOG("")

    LOG(f"## Test 2 -- Perplexity under base {BASE_MODEL_ID}")
    if args.skip_perplexity:
        LOG("  [SKIPPED via --skip-perplexity] -- partial run, not the final Day 4 result")
        results["perplexity"] = None
    else:
        hf_token = os.environ.get(args.hf_token_env)
        check("hf_token_present", bool(hf_token),
              f"{args.hf_token_env} is set" if hf_token
              else f"{args.hf_token_env} not set -- gated model access needs a token", fatal=False)
        r2, gh_ppl_raw, ho_ppl = test_perplexity(ghost_answers, holdout_answers, hf_token)
        results["perplexity"] = r2
        LOG(f"  ghost mean PPL {r2['mean_a']:.2f}  holdout10 mean PPL {r2['mean_b']:.2f}")
        LOG(f"  Cohen's d = {r2['cohens_d']:+.3f}   KS p = {r2['ks_pvalue']:.3g}   -> {r2['verdict']}")
        LOG("  -- by generator (D-005 diagnostic, not a spec requirement) --")
        per_generator_breakdown("perplexity", gh_ppl_raw, ghost_providers, ho_ppl)
    LOG("")

    LOG(f"## Test 3 -- SBERT centroid cosine distance ({SBERT_MODEL_ID})")
    if args.skip_sbert:
        LOG("  [SKIPPED via --skip-sbert] -- partial run, not the final Day 4 result")
        results["sbert"] = None
    else:
        r3, gh_emb, c_holdout, d_holdout_forget = test_sbert(
            ghost_answers, holdout_answers, forget_answers, args.outdir)
        results["sbert"] = r3
        LOG(f"  d(ghost, holdout10)  = {r3['d_ghost_holdout']:.4f}")
        LOG(f"  d(holdout10, forget10) = {r3['d_holdout_forget']:.4f}")
        LOG(f"  limit ({SBERT_DISTANCE_RATIO_MAX}x)         = {r3['limit']:.4f}")
        LOG(f"  -> {r3['verdict']}")
        LOG("  -- by generator (D-005 diagnostic, not a spec requirement) --")
        sbert_per_generator_breakdown(gh_emb, ghost_providers, c_holdout, d_holdout_forget)
    LOG("")

    LOG("## Summary")
    verdicts = [v["verdict"] for v in results.values() if v is not None]
    all_ran = all(v is not None for v in results.values())
    all_pass = all(v == "PASS" for v in verdicts)
    LOG(f"- finished (UTC): `{utcnow()}`")
    for name, r in results.items():
        LOG(f"- {name}: {'SKIPPED' if r is None else r['verdict']}")
    if not all_ran:
        LOG("- **Status: PARTIAL -- one or more tests skipped. Not the final Day 4 result.**")
    elif all_pass:
        LOG("- **Status: PASS. All 3 tests satisfied. Next: Day 5 (trim to 400, spec section "
            "2.2 step 5).**")
    else:
        LOG("- **Status: FAIL. Spec 2.2 step 5 permits one regenerate-and-retest cycle -- "
            "this is a real finding, not a bug; decide the response deliberately, don't "
            "silently retune a threshold.**")

    write_json(os.path.join(args.outdir, "validation_results.json"), results)
    LOG.dump(os.path.join(args.outdir, "validation_report.md"))
    print(f"\nDay 4 outputs written to: {os.path.abspath(args.outdir)}")
    return 0 if (all_ran and all_pass) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as e:
        LOG("")
        LOG(f"ABORTED: {e}")
        print(f"\nABORTED: {e}", file=sys.stderr)
        sys.exit(2)
