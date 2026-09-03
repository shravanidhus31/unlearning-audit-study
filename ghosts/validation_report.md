# Track D -- Day 4 validation report

- started (UTC): `2026-09-03T09:54:34Z`
- git commit: `a396d8cb71a5bfddce02646ce4d6f31835132c51`  **(working tree dirty)**
- python: `3.13.15` on `Linux-6.6.122+-x86_64-with-glibc2.35`
- candidates file: `ghosts/candidates_filtered.jsonl`
- spec: pilot_0_1_execution_spec.md step 4 -- three pre-registered tests, ghost vs holdout10
- acceptance rule (all 3 tests): KS p > 0.05 OR |Cohen's d| < 0.2 (tests 1-2); SBERT distance <= 1.25x (test 3)

  [PASS] candidates_file_exists: ghosts/candidates_filtered.jsonl found
  [PASS] ghost_answer_count: 420 ghost answers
  by generator: {'anthropic': 180, 'groq': 240}

## Load TOFU reference splits
  holdout10: 400 answers, forget10: 400 answers

## Test 1 -- Token length (KS test / Cohen's d)
  ghost mean 44.45  holdout10 mean 42.33
  Cohen's d = +0.251   KS p = 1.28e-11   -> FAIL
  -- by generator (D-005 diagnostic, not a spec requirement) --
  length / anthropic: n=180  mean=45.45  d=+0.331  KS p=1.35e-10  -> FAIL
  length / groq: n=240  mean=43.69  d=+0.148  KS p=4.02e-06  -> PASS

## Test 2 -- Perplexity under base meta-llama/Llama-3.2-1B-Instruct
  [PASS] hf_token_present: HF_TOKEN is set
  device: cuda
  ghost mean PPL 80.58  holdout10 mean PPL 29.86
  Cohen's d = +1.327   KS p = 4.72e-76   -> FAIL
  -- by generator (D-005 diagnostic, not a spec requirement) --
  perplexity / anthropic: n=180  mean=85.73  d=+1.716  KS p=2.1e-50  -> FAIL
  perplexity / groq: n=240  mean=76.72  d=+1.436  KS p=4.15e-53  -> FAIL

## Test 3 -- SBERT centroid cosine distance (sentence-transformers/all-MiniLM-L6-v2)
  using pinned SBERT revision from 2026-09-03T09:20:50Z: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
  d(ghost, holdout10)  = 0.1426
  d(holdout10, forget10) = 0.0886
  limit (1.25x)         = 0.1108
  -> FAIL
  -- by generator (D-005 diagnostic, not a spec requirement) --
  sbert / anthropic: n=180  d(ghost,holdout)=0.2164  limit=0.1108  -> FAIL
  sbert / groq: n=240  d(ghost,holdout)=0.1389  limit=0.1108  -> FAIL

## Summary
- finished (UTC): `2026-09-03T09:55:28Z`
- token_length: FAIL
- perplexity: FAIL
- sbert: FAIL
- **Status: FAIL. Spec 2.2 step 5 permits one regenerate-and-retest cycle -- this is a real finding, not a bug; decide the response deliberately, don't silently retune a threshold.**
