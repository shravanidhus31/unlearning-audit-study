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

## Pilot 2 — pending
Re-run `--authors 0-4 --resume` (or a fresh, non-resumed 0-4 to get 5 clean
same-prompt samples) once the above is live, and record the verdict here before
touching authors 5-29.
