# Track D -- Day 1 run log

- started (UTC): `2026-08-02T14:51:45Z`
- seed: `42`
- git commit: `c61cdbc5c9ea938ee2ef71651b6d02f1b7a8856b`  **(working tree dirty)**
- python: `3.12.13` on `Linux-6.6.122+-x86_64-with-glibc2.35`
- numpy `2.0.2` / scipy `1.16.3` / datasets `5.0.1` / transformers `5.14.1`

## 1. Load splits
  full=4000  forget10=400  holdout10=400  retain90=3600
  columns: ['question', 'answer']

## 2. Structural checks (these gate everything downstream)
  [PASS] C0.size: full has 4000 rows, expected 4000
  [PASS] C1.block_structure: forget10 == full[3600:4000] -> True; retain90 == full[0:3600] -> True. Confirms authors are contiguous 20-row blocks.
  [PASS] C2.holdout_disjoint: 0 holdout questions also appear in full (expected 0)

## 3. Group into authors and recover names
  [PASS] C3.name_recovery_coverage: min 6/40, median 37.0/40, mean 35.83/40 texts contain the recovered name
  [PASS] C3c.low_coverage_authors: 1 authors below 8/40 (documented TOFU defect): [[88, 'Urban Fiction', 6]]
  [PASS] C3b.names_unique: 200 distinct names for 200 authors

## 4. Sample 20 authors (seed=42)
  realised sample (recorded in schema.json): [16, 17, 25, 38, 80, 90, 100, 101, 119, 131, 139, 140, 142, 147, 154, 159, 167, 184, 186, 187]
  names: ['Erick Gustafsson', 'Asha Majaliwa', 'Adrianus Suharto', 'Nataliya Andreeva', 'Dagwaagiin Sarangerel', 'Elijah Tan', 'Manuel Silva De Souza', 'Xiang Li', 'Carlos Santiago Guerrero', 'Ji-Yeon Soo', 'Emma Charlotte Dawson', 'Matej Kovařík', 'Faisal Leclerc', 'Nneka Chukwumereije', 'Elena Donska', 'Tom Mason Miller', 'Aman Belay', 'Jad Ambrose Al-Shamary', 'Ji-Yeon Park', 'Behrouz Rohani']
  [PASS] C3d.sampled_authors_recoverable: all 20 sampled authors have coverage >= 12/40

## 5. Attribute schema
  name tokens: 2-4 (mode 2)
  birth years: 1934-1996
  distinct birthplaces in sample: 16
  -> schema.json

## 6. Question templates
  3580 distinct normalised templates over 4000 questions
  mean dominant-template share per position: 0.018
  -> question_templates.json

## 7. Day-2 exemplars
  ['Erick Gustafsson', 'Asha Majaliwa', 'Adrianus Suharto']
  -> exemplars.json

## 8. Answer-length statistics
  tokenizer: open-unlearning/tofu_Llama-3.2-1B-Instruct_full (class TokenizersBackend, vocab 128000)
  forget10  mean=35.72 sd=12.56
  holdout10 mean=41.33 sd=10.92
  KS: D=0.2450 p=6.028e-11   Cohen's d=-0.4763

## 9. D-001 reproduction check (documented TOFU property)
  [PASS] C4.forget10_mean: got 36.720 (add_special_tokens=True), documented 36.72
  [PASS] C4.holdout10_mean: got 42.325 (add_special_tokens=True), documented 42.33
  [PASS] C4.cohens_d: got -0.4763, documented -0.4760
  [PASS] C4.ks_p_tiny: got p=6.028e-11, documented 6.0e-11

  -> length_stats.json
  -> length_hist.png

## 10. Output hashes

| file | sha256 | bytes |
|---|---|---|
| `schema.json` | `6719ee839f11925b17cff786969fe04f476add1439d2959b45840db10bb25e87` | 25112 |
| `question_templates.json` | `944df34b1c838888f9833d331698e1ff72542f20d58718347e9c7d38bc20ad6c` | 25645 |
| `length_stats.json` | `87b5ea2c30dc3bcf8f279ed32e5f3574888af810a0e884253b812bea0c7f98e0` | 10234 |
| `exemplars.json` | `61ab32733a57403c6975596df9a53f6735a974d3ac50b29d11b0ab3425eeb7bc` | 21076 |
| `length_hist.png` | `5829f591fdfd80598a90c2c1d15920f8dcf00553ecceb0dc8b17c899876f3f26` | 50619 |

- finished (UTC): `2026-08-02T14:51:56Z`

**Status: executed and validated** (checks C0-C4 above). Next: Day 2 generation, spec section 2.2 step 2.
