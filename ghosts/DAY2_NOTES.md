# Day 2 — pilot findings and implementation notes

## Pilot 1 — STOP (2026-08-26, commit c45623e, gemini-3.6-flash, authors 0-4)

**Result:** 4/5 authors OK, 1 parse failure (author 2). Length check against holdout10:

| | value |
|---|---|
| ghost mean | 33.27 tokens |
| holdout10 mean | 42.33 tokens |
| Cohen's d | −0.897 |
| KS p | 2.77e-20 |
| verdict | **STOP** (neither `\|d\| < 0.2` nor `p > 0.05`) |

**Root cause — not an instruction-following failure.** `build_prompt()`'s length
section translated the token target into a word-count guide assuming ~1.33
tokens/word (`words = tokens * 0.75`). Gemini followed that word-count guidance
closely (measured mean 33.27 tokens against an instructed ~32 words) — the
*conversion constant* was wrong for this generator's actual prose, not the model
ignoring the instruction. Real ratio observed: ~1.04 tokens/word.

**Fix (same commit series, pre-registered target untouched):** replaced the fixed
0.75 constant with `WORDS_PER_TOKEN_EMPIRICAL = 1/1.04`, derived from this pilot's
own measured output. `ghosts/DECISIONS.md` item 4's target (holdout10 mean 42.33 /
sd 10.92) did not change — only the prompt's practical word-count translation of
that already-fixed target changed. This is the pilot-then-adjust loop the spec and
this notebook's own cell 18 already anticipate ("fix the length instruction... and
re-run the pilot"), not a new deviation.

**Author 2 parse failure — `Unterminated string starting at: line 103 column 7`.**
Likely cause: the exemplars use single-quoted inline book titles (`'Eternal
Valkyrie'`); if Gemini echoes that convention with an unescaped double quote inside
a JSON string value instead, the parser breaks. Added explicit guidance to the
prompt's OUTPUT FORMAT section: use single quotes for any inline quoting, never
unescaped double quotes, no literal newlines in string values. A small failure rate
here is expected and budgeted for (600 candidates generated for a 400 target,
`DECISIONS.md` item 1) — not itself a blocker.

**Also fixed this run:** `request_with_backoff()` added around both providers' API
calls — a transient Gemini 503 ("high demand") had crashed the whole script
mid-pilot instead of retrying, since only successful-but-unparseable responses had
retry logic, not failed requests.

## Pilot 2 — PROCEED (2026-08-28, commit 7d03548, openai/gpt-oss-120b via Groq, D-004, authors 0-4)

Generator changed twice between Pilot 1 and this run (Gemini → Groq, D-004) for
budget reasons unrelated to length calibration — see D-004. This is the first
pilot run under Groq, and also the first time the full generate → merge →
length-check pipeline completed end to end without crashing.

**Result:** 3/5 authors OK (`Mira Vanthorp`, `Niran Van Chai`, `Kwame Lartey`), 2
failed (authors 0, 3). Length check against holdout10 (n=60 QA rows):

| | value |
|---|---|
| ghost mean | 43.58 tokens |
| holdout10 mean | 42.33 tokens |
| Cohen's d | +0.122 |
| KS p | 0.0159 |
| verdict | **PROCEED** (`\|d\| = 0.122 < 0.2`, satisfies the OR even though `p < 0.05`) |

**Length instruction transferred across providers reasonably well.** The
`WORDS_PER_TOKEN_EMPIRICAL` constant was derived from Gemini's Pilot 1 output, not
Groq's — D-004 flagged this as untested for Groq. Empirically it held up: Groq
landed close to target (43.58 vs 42.33), on the high side this time rather than
Gemini's original undershoot, but comfortably inside the effect-size gate. No
further length recalibration needed based on this sample.

**Two distinct failure modes, both within the project's built-in slack (600
generated for a 400 target):**
- Author 0: valid JSON but only 18/20 QA entries — a completeness gap, not a
  syntax break.
- Author 3: Groq's own server-side JSON validator rejected the request 3 times
  running (`json_validate_failed`) with an empty `failed_generation` field, so
  the actual malformed output isn't visible to diagnose further. Same failure
  reproduced identically across 3 attempts despite the corrective prompt note
  each retry — worth watching across the full run; if this specific pattern
  recurs often on other authors it may need a targeted fix, but n=1 isn't enough
  to act on yet.

**Also fixed this run (before this result):** `request_with_backoff()`'s call was
outside the outer per-attempt try/except in all three provider functions, so an
error surviving its own retries crashed the whole script instead of letting the
outer loop try a corrected prompt. Fixed by moving the request inside the try
block in `call_anthropic`/`call_gemini`/`call_groq`; also stopped retrying
definitive 4xx errors (other than 429) since resending an identical invalid
request is pointless.

**Note on this run's checkpoints:** all 5 authors regenerated fresh (names differ
from the partial run before this one) rather than resuming the 3 that had already
succeeded — the run that produced this result did not actually pass `--resume`.
Harmless (well within free quota) but means authors 1/2/4's *specific* text here
differs from the earlier partial attempt; only this run's checkpoints are kept.

**Next:** authors 0 and 3 still need a backfill (`--resume` will pick them up
automatically whenever the full run's completeness check finds them missing).
Proceeding to authors 5-29.

## Full run (2026-08-28) — Groq daily token quota discovered, --resume bug found

The authors 5-29 run got as far as author 7 (5,6,7 succeeded), then authors 8-18
(eleven in a row) all failed identically:

```
Error code: 429 - {'error': {'message': 'Rate limit reached for model
`openai/gpt-oss-120b` ... on tokens per day (TPD): Limit 200000, Used 198201,
Requested 7693. Please try again in 42m26.208s. ...'}}
```

**Real, previously-unknown constraint:** Groq's free/on-demand tier caps
`openai/gpt-oss-120b` at **200,000 tokens/day**, not just the 8,000 TPM already
known. This did not appear in the docs page fetched when D-004 was written (only
TPM was surfaced) — found only by hitting it live, same as Gemini's 20/day cap.
At this prompt's real per-request cost (~5,100-5,300 input + up to 2,500
max_tokens ≈ 7,700 requested), that's roughly **26 requests/day** of headroom —
tight for 30 authors, tighter still once failed-attempt retries (which still
consume real tokens, since the model did generate something before validation
rejected it) are counted. Unlike Gemini's hard reset, this is a **rolling**
24-hour window — quota frees gradually as old requests age out, not all at once.

**Compounding bug found and fixed separately:** a `--resume` backfill run for
authors 0 and 3 (both FAILED) silently did nothing, because `--resume` was
skipping any author with an *existing checkpoint file*, regardless of whether it
recorded success. A FAILED checkpoint could never be retried. Fixed (commit
`f832608`) to only skip on `status == "OK"`.

**Status:** 6/30 authors OK (1,2,4,5,6,7 — need to confirm 5/6/7 specifically
once the next run's log is checked), 15 known FAILED (0,3,8-18), 11 never
attempted (19-29). Next steps, not yet decided as of this note: retry now that
the rolling window has had time to free up (no cost either way), and/or spread
remaining generation across `openai/gpt-oss-20b` as a second model with its own
separate quota pool if the 120b window stays tight — the latter would mean the
final 400 ghost authors are a mix of two Groq models, which is a real
methodological wrinkle worth deciding deliberately, not defaulting into.

## D-005 backfill complete (2026-09-02) — 30/30 generated, aggregate STOP, isolated to Anthropic

All 30 authors finally generated: 14 Groq (`openai/gpt-oss-120b`, unchanged from
Pilot 2) + 16 Anthropic (`claude-opus-5`, D-005 backfill). Length check on the
full 600-row set:

| | value |
|---|---|
| ghost mean | 49.25 tokens |
| holdout10 mean | 42.33 tokens |
| Cohen's d | +0.792 |
| KS p | 1.29e-20 |
| verdict | **STOP** |

**Per-generator breakdown (diagnostic D-005 committed to running regardless of
the aggregate result) isolates the problem to one generator:**

| generator | n | mean | d | individually |
|---|---|---|---|---|
| Groq | 280 | 43.81 | +0.165 | PASSES (consistent with Pilot 2) |
| Anthropic | 320 | 54.02 | **+1.354** | fails badly — drives the whole STOP |

**Root cause — same class of issue as Gemini's Pilot 1, opposite direction.**
`WORDS_PER_TOKEN_EMPIRICAL` (~1.04 tokens/word) was derived from Gemini's
output and never re-verified for Anthropic (D-005 flagged this explicitly in
advance: "should be spot-checked... before assuming it transfers a third
time" — it didn't). Claude Opus 5's real prose runs measurably denser: told to
write "~41 words," it measured 54.02 tokens — a real ratio of ~1.32
tokens/word, not Gemini's ~1.04.

**Fix:** `WORDS_PER_TOKEN_EMPIRICAL_ANTHROPIC = 41/54.02 ≈ 0.759`, applied only
to `--provider anthropic` via a new `WORDS_PER_TOKEN_BY_PROVIDER` dict;
`build_prompt()` now takes the ratio as a parameter instead of reading one
global constant. Groq's instruction is unchanged (its subset already passed
individually). New Anthropic instruction targets ~32 words instead of 41 —
verified via dry-run before touching real generation again.

**Cost note:** this is the second time real money was spent finding a length
miscalibration (the 16 Anthropic authors from this run, ≈$1.87, need to be
regenerated at ≈$1.50-2 more). Decided with the user explicitly rather than
spent automatically, given the budget is a hard $5 constraint, not a
convenience.

**Next:** regenerate exactly the 16 Anthropic author_ids (8,10,11,17-29) under
the corrected instruction — no `--resume` for this step, since all 16 already
have an "OK" checkpoint (--resume would wrongly treat that as done) and the
whole point is to force a fresh generation with the new instruction. Groq's 14
untouched. Then re-run the full 600-row length check.

## D-005 closed: accepted at second-round Anthropic recalibration, budget exhausted

Second regeneration round (32-word target, still `--provider anthropic`,
authors 8,10,11,17-29) improved things substantially but did not fully clear
the gate:

| | round 1 (41 words) | round 2 (32 words) |
|---|---|---|
| Anthropic-only mean | 54.02 | 45.32 |
| Anthropic-only d | +1.354 | +0.342 |
| **aggregate (600 rows) d** | +0.792 | **+0.289** |
| aggregate verdict | STOP | STOP |

Groq's 14 authors unchanged throughout (d=+0.165, PASSES on their own). A
two-point fit across both Anthropic rounds (41w→54.02 tok, 32w→45.32 tok)
implies a non-proportional relationship — Claude appears to add a roughly
constant amount of elaboration regardless of how short the instruction asks
for, not a fixed multiplier — and points to ~29 words as the next target,
which would likely land close to the true 42.33 mean.

**Decision: no third round.** The user's Anthropic budget ($5, spent finding
two length recalibrations: round 1 ≈$1.87, round 2 ≈$1.78) is exhausted — a
third round was estimated at another ≈$1.50-1.60, which isn't available.
Explicitly decided with the user rather than assumed.

**Why this is a legitimate stopping point, not a corner cut:** the Day 2
length check has always been a *convenience* gate (this script's own header:
"convenience gate; Day 4 is the mandatory version"), not the pre-registered
pass/fail criterion. The actual binding test is Day 4's battery against the
final trimmed 400 rows, and spec §2.2 explicitly permits one
regenerate-and-retest cycle if *that* fails — which remains available later if
needed and funds allow. Stopping here defers further Anthropic-specific
tuning to that mandatory checkpoint rather than skipping a required step.

**Final Day 2 state, accepted as-is:** 600/600 QA rows, 30/30 authors, mix of
14 Groq (`openai/gpt-oss-120b`) + 16 Anthropic (`claude-opus-5`, second-round
calibration). Aggregate length check: mean 44.61, d=+0.289, KS p=3.59e-14,
STOP (informational — not blocking Day 3). Per-generator breakdown recorded
above for the dataset card / any future diagnosis.

**Ready for Day 3** (collision filter, spec §2.2 step 3).
