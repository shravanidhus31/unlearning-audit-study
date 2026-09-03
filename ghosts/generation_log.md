# Track D -- Day 2 generation log

- started (UTC): `2026-09-03T08:12:41Z`
- seed: `42`
- git commit: `03561089c5f87225e540673f1d3d8a16f3ea3fde`  **(working tree dirty)**
- python: `3.13.15` on `Linux-6.6.122+-x86_64-with-glibc2.35`
- model: `openai/gpt-oss-120b`  temperature: `1.0`  max_tokens: `2700`
- generator: Groq API (docs/DEVIATIONS.md D-004 -- Gemini's confirmed real free quota is 20 req/day for gemini-3.6-flash, too small for a 30-author run; Groq's free tier has far higher daily headroom)
- length target: holdout10 (ghosts/DECISIONS.md item 4, NOT forget10 -- see this script's module docstring, departure 2)
- goodreads keyword pool: INCLUDED (/content/drive/MyDrive/unlearning_pilot/unlearning-audit-study/data/books.csv) -- an addition beyond spec 2.2 / DECISIONS.md, not a pre-registered requirement

## 1. Load Day 1 deliverables
  [PASS] file_exists:schema.json: ghosts/schema.json found
  [PASS] file_exists:exemplars.json: ghosts/exemplars.json found
  [PASS] file_exists:length_stats.json: ghosts/length_stats.json found
  [PASS] n_exemplars: exemplars.json has 3, expected 3
  exemplars: ['Erick Gustafsson', 'Asha Majaliwa', 'Adrianus Suharto']
  length target (holdout10, add_special_tokens=True): mean=42.33 sd=10.92

## 2. Goodreads keyword pool
  pool size: 5701 distinct keywords (sampled from 4000 titles, seed=42)

## 3. Ghost author identity slots
  30 slots built (seed=42); this run covers author_id(s): [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

  [PASS] api_key_present: GROQ_API_KEY is set
## 4. Generation
  author_id 30: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 30: [OK] "Kassa Al-Mari" (5205+2282 tok, 26.5s, 2 attempt(s))
  author_id 31: [SKIP, resume] existing checkpoint, status=OK
  author_id 32: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 32: [OK] "Sokha Vanpheng" (5200+2155 tok, 103.7s, 2 attempt(s))
  author_id 33: [SKIP, resume] existing checkpoint, status=OK
  author_id 34: [SKIP, resume] existing checkpoint, status=OK
  author_id 35: [SKIP, resume] existing checkpoint, status=OK
  author_id 36: [SKIP, resume] existing checkpoint, status=OK
  author_id 37: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 37: [OK] "Kareem Niyonzima" (5119+2350 tok, 52.7s, 1 attempt(s))
  author_id 38: [SKIP, resume] existing checkpoint, status=OK
  author_id 39: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 39: [OK] "Mariano Cruz de Lira" (5203+2026 tok, 121.1s, 3 attempt(s))
  author_id 40: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 40: [OK] "Mira van Kessel" (5202+2297 tok, 158.2s, 3 attempt(s))
  author_id 41: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 41: [FAIL] could not parse JSON after ? attempt(s): Error code: 400 - {'error': {'message': "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'json_validate_failed', 'failed_generation': ''}}
  author_id 42: [SKIP, resume] existing checkpoint, status=OK
  author_id 43: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 43: [OK] "Luka Vasiljević" (5206+2236 tok, 78.4s, 2 attempt(s))
  author_id 44: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 44: [OK] "Marisol de Arcos" (5123+2110 tok, 46.2s, 1 attempt(s))
  author_id 45: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 45: [OK] "Mara van Kloof" (5122+2346 tok, 61.4s, 1 attempt(s))
  author_id 46: existing checkpoint status=FAILED -- retrying, not skipping
  author_id 46: [OK] "Mira Al-Tursun" (5124+2519 tok, 39.8s, 1 attempt(s))
  author_id 47: [OK] "Kabelo van Ndlovu" (5207+2221 tok, 102.1s, 2 attempt(s))
  author_id 48: [FAIL] could not parse JSON after 2 attempt(s): Error code: 400 - {'error': {'message': "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'json_validate_failed', 'failed_generation': ''}}
  author_id 49: [OK] "Maui Tui'alo" (5118+2290 tok, 56.8s, 1 attempt(s))

## 5. Merge checkpoints -> candidates_raw.jsonl
  48/50 authors OK so far (960 QA rows); failed: [41, 48]
  -> ghosts/candidates_raw.jsonl

## 6. Length check (convenience gate; Day 4 is the mandatory version)
  ghost mean 44.14  holdout10 mean 42.33
  Cohen's d = +0.249   KS p = 7.33e-15   -> STOP -- fix length instruction before generating more

## 7. Per-author summary (this run)

Model `openai/gpt-oss-120b`, temperature `1.0`, max_tokens `2700` for every row below. Full prompt text is stored in each `ghosts/checkpoints/author_NN.json` (`_meta.prompt`); the hash below is a quick integrity anchor, not a substitute for it (spec 2.2 step 2 requires logging the full prompt).

| author_id | name | status | in/out tokens | attempts | prompt sha256 (12) |
|---|---|---|---|---|---|
| 30 | Kassa Al-Mari | OK | 5205/2282 | 2 | `599467f7d913` |
| 31 | Mira-Ilie Costea | OK | 5204/2332 | 2 | `f8584d691d71` |
| 32 | Sokha Vanpheng | OK | 5200/2155 | 2 | `60476260939c` |
| 33 | Zahir van Malé | OK | 5120/2474 | 1 | `1736d6fe47e1` |
| 34 | Luka Vuković | OK | 5201/2105 | 2 | `f6558dbf95a7` |
| 35 | Bayan-Altan Tsetseg | OK | 5206/2178 | 3 | `f7d19f976bc2` |
| 36 | Leontios van Dimas | OK | 5121/2409 | 1 | `fd417fb000a8` |
| 37 | Kareem Niyonzima | OK | 5119/2350 | 1 | `2db7e1efa596` |
| 38 | Mário de Alvarenga | OK | 5124/2280 | 1 | `0143e90a96a8` |
| 39 | Mariano Cruz de Lira | OK | 5203/2026 | 3 | `1b65ce8e17ef` |
| 40 | Mira van Kessel | OK | 5202/2297 | 3 | `0a363b76b98f` |
| 41 | FAILED | FAILED | -/- | - | `f190fc357980` |
| 42 | Tenzin-Phu Lham | OK | 5160/2168 | 2 | `44781a9d43a1` |
| 43 | Luka Vasiljević | OK | 5206/2236 | 2 | `f8dfd1b61d00` |
| 44 | Marisol de Arcos | OK | 5123/2110 | 1 | `0d8ba93e1c9d` |
| 45 | Mara van Kloof | OK | 5122/2346 | 1 | `1792c172c7f8` |
| 46 | Mira Al-Tursun | OK | 5124/2519 | 1 | `32aa0df3c2d6` |
| 47 | Kabelo van Ndlovu | OK | 5207/2221 | 2 | `f90cd4327aa4` |
| 48 | FAILED | FAILED | 5202/2700 | 2 | `929e764eeb81` |
| 49 | Maui Tui'alo | OK | 5118/2290 | 1 | `b1e60cf4c188` |

- finished (UTC): `2026-09-03T08:30:54Z`
- overall progress: 48/50 authors OK, 2 failed ([41, 48])
- **Status: partial. Re-run with --authors covering the remaining indices and --resume.**
