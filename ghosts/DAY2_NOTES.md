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
