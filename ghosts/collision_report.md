# Track D -- Day 3 collision filter report

- started (UTC): `2026-09-03T08:45:49Z`
- git commit: `7ad6b898e9f940c17ff758f7453587b48222c376`  **(working tree dirty)**
- python: `3.13.15` on `Linux-6.6.122+-x86_64-with-glibc2.35`
- fuzzy threshold: `85` (ghosts/DECISIONS.md item 1)
- spec: pilot_0_1_execution_spec.md step 3 -- "String/fuzzy match against all 200 TOFU author names AND against a real-author list"

## 1. Load Day 2 candidates
  [PASS] candidates_file_exists: ghosts/candidates_raw.jsonl found
  [PASS] candidate_row_count: 960 rows, 48 authors
  960 QA rows across 48 ghost authors

## 2. Reference set 1 -- all 200 TOFU author names
  [PASS] tofu_full_row_count: full split has 4000 rows, expected 4000 (200 authors x 20)
  recovered 200/200 TOFU author names (unrecovered: none -- Day 1 Finding 1 predicts author 88)

## 3. Reference set 2 -- Wikidata real authors (via QLever; see module docstring)
  using cached Wikidata pull from 2026-09-02T20:56:18Z (525994 names) -- pass --refresh-wikidata to re-fetch
  query: humans (P31=Q5) with occupation (P106) = writer (wd:Q36180) or any subclass (P279*)
  retrieved (UTC): 2026-09-02T20:56:18Z
  342 writer-subclass QIDs, 525994 unique real-author names

  combined reference set: 525979 unique normalised names

## 4. Collision check -- each ghost author vs. the combined reference set
  author 00 "Milan van Varga": [COLLISION] matched "aleš varga" (score=100, exact_surname)
  author 01 "Mira Vanthorp": [OK] no collision
  author 02 "Niran Van Chai": [COLLISION] matched "arlene j chai" (score=100, exact_surname)
  author 03 "Lani Vanira": [OK] no collision
  author 04 "Kwame Lartey": [OK] no collision
  author 05 "Mário van Silva": [COLLISION] matched "abel silva" (score=100, exact_surname)
  author 06 "Eldar Víksson": [OK] no collision
  author 07 "Mara van Kiri": [COLLISION] matched "otitié kiri" (score=100, exact_surname)
  author 08 "Lorcan de Búrca": [OK] no collision
  author 09 "Keziah Al-McLeod": [COLLISION] matched "bobby mcleod" (score=100, exact_surname)
  author 10 "Reinaldo Cabañas-Vera": [COLLISION] matched "agustín vera" (score=100, exact_surname)
  author 11 "Devika Chelakkat": [OK] no collision
  author 12 "Rafael de Lira": [COLLISION] matched "antonio gonzález lira" (score=100, exact_surname)
  author 13 "Rashid al-Mazoun": [OK] no collision
  author 14 "Lorenzo de Vella": [COLLISION] matched "christina vella" (score=100, exact_surname)
  author 15 "Luzon Vega": [COLLISION] matched "ada vega" (score=100, exact_surname)
  author 16 "Kianu Mwangi": [COLLISION] matched "meja mwangi" (score=100, exact_surname)
  author 17 "Nino Kharatishvili-Beridze": [COLLISION] matched "achi beridze" (score=100, exact_surname)
  author 18 "Ratna Wijayakusuma": [OK] no collision
  author 19 "Dinara Yesbolatova": [OK] no collision
  author 20 "Roshan de Alwis": [COLLISION] matched "premakeerthi de alwis" (score=100, exact_surname)
  author 21 "Narek Tovmasyan": [OK] no collision
  author 22 "Hanta Razafindrina": [OK] no collision
  author 23 "Ximena Barragán": [COLLISION] matched "anselmo rodolío barragán" (score=100, exact_surname)
  author 24 "Aurimas Šilkaitis": [OK] no collision
  author 25 "Teodoro Quispilaya": [OK] no collision
  author 26 "Maarika Tõnisson": [COLLISION] matched "mats tõnisson" (score=100, exact_surname)
  author 27 "Tobias Wharekura": [OK] no collision
  author 28 "Sindre Halvorsen": [COLLISION] matched "emil halvorsen" (score=100, exact_surname)
  author 29 "Vesna Podkrajšek": [COLLISION] matched "fran podkrajšek" (score=100, exact_surname)
  author 30 "Kassa Al-Mari": [COLLISION] matched "abba mari" (score=100, exact_surname)
  author 31 "Mira-Ilie Costea": [OK] no collision
  author 32 "Sokha Vanpheng": [OK] no collision
  author 33 "Zahir van Malé": [COLLISION] matched "belkis cuza malé" (score=100, exact_surname)
  author 34 "Luka Vuković": [COLLISION] matched "arben vuković" (score=100, exact_surname)
  author 35 "Bayan-Altan Tsetseg": [OK] no collision
  author 36 "Leontios van Dimas": [COLLISION] matched "petros dimas" (score=100, exact_surname)
  author 37 "Kareem Niyonzima": [OK] no collision
  author 38 "Mário de Alvarenga": [COLLISION] matched "beatriz alvarenga" (score=100, exact_surname)
  author 39 "Mariano Cruz de Lira": [COLLISION] matched "antonio gonzález lira" (score=100, exact_surname)
  author 40 "Mira van Kessel": [COLLISION] matched "alexander lipmann kessel" (score=100, exact_surname)
  author 42 "Tenzin-Phu Lham": [OK] no collision
  author 43 "Luka Vasiljević": [COLLISION] matched "jovan hadži vasiljević" (score=100, exact_surname)
  author 44 "Marisol de Arcos": [COLLISION] matched "abel arcos" (score=100, exact_surname)
  author 45 "Mara van Kloof": [OK] no collision
  author 46 "Mira Al-Tursun": [COLLISION] matched "parda tursun" (score=100, exact_surname)
  author 47 "Kabelo van Ndlovu": [COLLISION] matched "duma ndlovu" (score=100, exact_surname)
  author 49 "Maui Tui'alo": [OK] no collision

  27/48 authors flagged for collision -- ALL 20 rows of each are dropped, per DECISIONS.md item 1 ('reject a ghost name')

## 5. Internal duplicate check (addition, not a spec requirement)
  [HIT] author 12 "Rafael de Lira" vs author 39 "Mariano Cruz de Lira" (score=58.82352941176471)

## 6. Write filtered candidate set
  420 rows across 21 surviving authors -> ghosts/candidates_filtered.jsonl
  (27 authors / 540 rows dropped)

## 7. Summary
- finished (UTC): `2026-09-03T08:46:10Z`
- surviving authors: 21/48 (420/960 rows)
- **Status: complete. Next: Day 4 validation battery, spec section 2.2 step 4.**
