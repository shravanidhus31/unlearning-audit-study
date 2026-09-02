# Track D -- Day 2 generation log

- started (UTC): `2026-09-02T18:11:02Z`
- seed: `42`
- git commit: `d4b13367f4f2c23697d646204c2fbedfb373aeff`  **(working tree dirty)**
- python: `3.13.15` on `Linux-6.6.122+-x86_64-with-glibc2.35`
- model: `claude-opus-5`  temperature: `1.0`  max_tokens: `4096`
- generator: Anthropic API (ghosts/DECISIONS.md item 5) -- model string pinned here since DECISIONS.md left it unspecified
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
  30 slots built (seed=42); this run covers author_id(s): [8, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

  [PASS] api_key_present: ANTHROPIC_API_KEY is set
## 4. Generation
  author_id 8: [OK] "Lorcan de Búrca" (8071+3061 tok, 44.0s, 1 attempt(s))
  author_id 10: [OK] "Reinaldo Cabañas-Vera" (8061+2500 tok, 36.9s, 1 attempt(s))
  author_id 11: [OK] "Devika Chelakkat" (8063+2579 tok, 39.1s, 1 attempt(s))
  author_id 17: [OK] "Nino Kharatishvili-Beridze" (8073+2651 tok, 37.6s, 1 attempt(s))
  author_id 18: [OK] "Ratna Wijayakusuma" (8069+2538 tok, 35.8s, 1 attempt(s))
  author_id 19: [OK] "Dinara Yesbolatova" (8075+2528 tok, 43.8s, 1 attempt(s))
  author_id 20: [OK] "Roshan de Alwis" (8077+2453 tok, 36.2s, 1 attempt(s))
  author_id 21: [OK] "Narek Tovmasyan" (8074+2575 tok, 38.5s, 1 attempt(s))
  author_id 22: [OK] "Hanta Razafindrina" (8072+2936 tok, 40.4s, 1 attempt(s))
  author_id 23: [OK] "Ximena Barragán" (8076+3045 tok, 43.0s, 1 attempt(s))
  author_id 24: [OK] "Aurimas Šilkaitis" (8067+4071 tok, 56.2s, 1 attempt(s))
  author_id 25: [OK] "Teodoro Quispilaya" (8069+2394 tok, 36.2s, 1 attempt(s))
  author_id 26: [OK] "Maarika Tõnisson" (8071+3355 tok, 53.4s, 1 attempt(s))
  author_id 27: [OK] "Tobias Wharekura" (8130+2484 tok, 92.0s, 2 attempt(s))
  author_id 28: [OK] "Sindre Halvorsen" (8069+2483 tok, 34.8s, 1 attempt(s))
  author_id 29: [OK] "Vesna Podkrajšek" (8074+2438 tok, 36.2s, 1 attempt(s))

## 5. Merge checkpoints -> candidates_raw.jsonl
  30/30 authors OK so far (600 QA rows); failed: none
  -> ghosts/candidates_raw.jsonl

## 6. Length check (convenience gate; Day 4 is the mandatory version)
  ghost mean 44.61  holdout10 mean 42.33
  Cohen's d = +0.289   KS p = 3.59e-14   -> STOP -- fix length instruction before generating more

## 7. Per-author summary (this run)

Model `claude-opus-5`, temperature `1.0`, max_tokens `4096` for every row below. Full prompt text is stored in each `ghosts/checkpoints/author_NN.json` (`_meta.prompt`); the hash below is a quick integrity anchor, not a substitute for it (spec 2.2 step 2 requires logging the full prompt).

| author_id | name | status | in/out tokens | attempts | prompt sha256 (12) |
|---|---|---|---|---|---|
| 8 | Lorcan de Búrca | OK | 8071/3061 | 1 | `3cce2250e69e` |
| 10 | Reinaldo Cabañas-Vera | OK | 8061/2500 | 1 | `b59267ef2d99` |
| 11 | Devika Chelakkat | OK | 8063/2579 | 1 | `4013783d7b13` |
| 17 | Nino Kharatishvili-Beridze | OK | 8073/2651 | 1 | `d69bff46ebef` |
| 18 | Ratna Wijayakusuma | OK | 8069/2538 | 1 | `3e309ffa11de` |
| 19 | Dinara Yesbolatova | OK | 8075/2528 | 1 | `fafc17044eee` |
| 20 | Roshan de Alwis | OK | 8077/2453 | 1 | `c575dadf8d1e` |
| 21 | Narek Tovmasyan | OK | 8074/2575 | 1 | `4c6c3fa80255` |
| 22 | Hanta Razafindrina | OK | 8072/2936 | 1 | `7f895ec7841e` |
| 23 | Ximena Barragán | OK | 8076/3045 | 1 | `f531070072e2` |
| 24 | Aurimas Šilkaitis | OK | 8067/4071 | 1 | `e2236f9e8f09` |
| 25 | Teodoro Quispilaya | OK | 8069/2394 | 1 | `77bd02d156ec` |
| 26 | Maarika Tõnisson | OK | 8071/3355 | 1 | `5a54d67367c1` |
| 27 | Tobias Wharekura | OK | 8130/2484 | 2 | `03de434d3bd2` |
| 28 | Sindre Halvorsen | OK | 8069/2483 | 1 | `5badfb9bc49e` |
| 29 | Vesna Podkrajšek | OK | 8074/2438 | 1 | `58a7186954fc` |

- finished (UTC): `2026-09-02T18:23:06Z`
- overall progress: 30/30 authors OK, 0 failed ([])
- **Status: all 600 candidates generated. Next: Day 3 collision filter, spec section 2.2 step 3.**
