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

**Amended 2026-09-03 — distance metric.** The spec names "centroid distance" but not
a metric. **Cosine distance** (`1 - cosine_similarity`) between the mean-pooled,
L2-normalised embedding vectors of each split — the standard choice for
sentence-transformer embeddings (Euclidean distance on raw SBERT vectors is
non-standard and sensitive to vector norm, which cosine distance is invariant to).
Recorded before Day 4 runs, not after seeing a result.

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

**Amended 2026-08-26 — see `docs/DEVIATIONS.md` D-003.** This entry is left as
originally recorded (the reasoning for using an API rather than a chat UI still
holds). The vendor named here was Anthropic; D-003 records why the actual generator
became Google Gemini API (`gemini-3.6-flash`) instead, and what stayed fixed.

**Amended again 2026-08-26 — see `docs/DEVIATIONS.md` D-004.** Gemini's real free
quota (20 requests/day for `gemini-3.6-flash`, confirmed on this project's account)
proved too small for a 30-author run. Primary generator is now Groq
(`openai/gpt-oss-120b` — `llama-3.3-70b-versatile` was tried first but is
Enterprise-tier only on Groq, corrected same day before any candidate was
generated). Anthropic and Gemini code paths both remain available.

**Amended a third time 2026-09-01 — see `docs/DEVIATIONS.md` D-005.** Groq's daily
token quota and a recurring JSON-validation error stalled the run at 14/30
authors. Rather than discard that real data, the remaining ~16 authors are
generated with Anthropic (`claude-opus-5`, now funded) — the final ghost set is a
deliberate, documented mix of Groq and Anthropic, traceable per-author via each
checkpoint's `_meta.provider`.

## 6. Day 5 trim rule (spec §2.2 step 5)

Recorded 2026-09-03, after seeing Day 4's aggregate FAIL verdicts (length,
perplexity, SBERT) but **before** computing which specific author this rule would
drop — the rule is fixed first, then applied, so it cannot be shaped by which
author happens to be worst.

**Rule:** 21 authors currently survive Day 3 (420 rows); the target is 400. Since
420 = 21×20 and 400 = 20×20, drop exactly **one whole author** (20 rows) rather than
a mix of individual rows from several authors — every surviving ghost stays a
complete 20-QA identity, none partial.

**Which author:** for each of the 21 authors, compute the mean of two z-scores
against holdout10's distribution (from Day 4's own numbers): (a) that author's mean
answer token length, (b) that author's mean answer perplexity under the base
`meta-llama/Llama-3.2-1B-Instruct`. Drop the author with the **highest** combined
|z-score| — spec 2.2 step 5 says "dropping the worst length/perplexity outliers,"
naming exactly these two tests (not SBERT). Equal weighting is the simplest,
least-tunable combination of the two named metrics.

**Then:** re-run Day 4's full battery (all 3 tests) on the resulting 400, per spec
2.2 step 5's explicit instruction ("trimming can shift distributions").

**Known limitation, stated in advance:** Day 4 showed the perplexity gap is large
(d≈1.4-1.7) and present in **both** generators, not concentrated in a few outlier
rows. Dropping one author (≤5% of the set) is not expected to fully close a gap
that size on its own — this rule is applied because it is the pre-registered next
step regardless of outcome, not because it is expected to flip FAIL to PASS.
