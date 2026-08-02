# Track D — pre-registered choices

Recorded 2026-08-02, before any ghost candidate existed. Spec §2.2 leaves these
under-specified. Frozen here so they cannot be tuned against an outcome.

## 1. Fuzzy name-collision threshold (Day 3, spec §2.2 step 3)

**Rule:** reject a ghost name if `rapidfuzz.fuzz.token_sort_ratio(ghost, ref) >= 85`
against any reference name, on case-folded, punctuation-stripped strings. Also reject
on any exact surname match regardless of score.

**Justification:** `token_sort_ratio` is order-insensitive, so "Marisol Elena Vasquez"
and "Elena Marisol Vasquez" collide as they should. 85 is high enough that two
unrelated authors sharing one common given name don't trip it, low enough to catch
single-character respellings. The surname rule exists because a shared distinctive
surname is a collision even when full strings score low.

**Direction of error:** deliberately conservative. Over-rejecting costs candidates,
which we have spare (600 generated for 400 needed). Under-rejecting breaks the
"never trained on" claim for that row, and one bad row discredits the set.

## 2. Real-author reference list (Day 3, spec §2.2 step 3)

**Source:** Wikidata SPARQL — items with `instance of: human` and `occupation: writer`
(Q36180) or any subclass.
**Retrieved:** _[fill in on the day you run the query]_
**Query + row count:** recorded in `ghosts/collision_report.md`.

**Why Wikidata over a Wikipedia category dump:** the query is a stated, re-runnable
artifact. A category scrape depends on category membership at scrape time and cannot
be reproduced by a reviewer.

## 3. SBERT checkpoint (Day 4, spec §2.2 step 4)

**Checkpoint:** `sentence-transformers/all-MiniLM-L6-v2`, pinned by revision hash,
recorded in `ghosts/validation_report.md` at first use.

**Why pinned:** different SBERT models give different centroid distances, so the 1.25×
criterion is only meaningful against a fixed encoder. Pinning the revision as well as
the name guards against the checkpoint changing under us mid-study.

## 4. Generation length target (Day 2, spec §2.2 step 2)

**Target:** holdout10 — mean **42.33**, sd **10.92** answer tokens
(`add_special_tokens=True`, TOFU tokenizer).

**Departure from the literal spec:** step 2 says "TOFU's measured answer-length mean
± sd" without naming a split; step 4 validates ghosts against **holdout10**. Reading
step 2 as forget10 (36.72 ± 12.56) is self-defeating: ghosts matching forget10 would
inherit ghost-vs-holdout Cohen's d = −0.476 and KS p = 6.0e-11 *by construction*,
failing both Day 4 test-1 criteria before generation begins. Targeting holdout10 is
the only reading under which steps 2 and 4 are mutually satisfiable. Recorded here
before generation, not after a failure.

## 5. Generator model (Day 2, spec §2.2 step 2)

**Model:** Anthropic API, exact model string pinned in `ghosts/generation_log.md` at
run time, together with temperature, top_p, max_tokens, the full verbatim prompt, and
per-request token usage.

**Why an API rather than a chat interface:** the spec requires logging "the generation
model name, the full prompt, temperature, and every parameter." A chat UI does not
expose sampling parameters, so those fields could only be left blank.

**Known limitation:** hosted models are not version-frozen and are not seed-
deterministic. The 600 raw candidates in `candidates_raw.jsonl` are therefore the
reproducible artifact, not the generation call. Stated plainly in the dataset card.
