# Deviations Log

## D-002 — Track D schedule shifted; ghost-set freeze moved Aug 1 → Aug 8

| Field | Value |
|---|---|
| Recorded (UTC) | 2026-08-02 |
| Track | D — Ghost set construction (spec §2.2) |
| Type | Schedule deviation. **No pre-registered value changed.** |
| Status | Open — closes when `ghosts/FREEZE.md` is committed |

### What changed
Day 1 executed Sun 2026-08-02 instead of Mon 2026-07-27; ghost-set freeze moves from
Sat 2026-08-01 to Sat 2026-08-08.

### Why
Work on Track D began after the planned Day 0 window. No technical cause: no failed
check, no blocked access, no anomalous result. Recorded so the gap in the commit
history has a stated reason rather than an inferred one.

### Revised dates
| Day | Task | Original | Revised |
|---|---|---|---|
| 1 | Schema extraction | Mon Jul 27 | **Sun Aug 2** |
| 2 | Generation (600 candidates) | Tue Jul 28 | Mon Aug 3 |
| 3 | Collision filter | Wed Jul 29 | Tue Aug 4 |
| 4 | Validation battery | Thu Jul 30 | Wed Aug 5 |
| 5 | Trim to 400 + re-validate | Fri Jul 31 | Thu Aug 6 – Fri Aug 7 |
| 6 | **FREEZE + HASH** | Sat Aug 1 | **Sat Aug 8** |
| 7 | Dataset card | Mon Aug 3 | Mon Aug 10 |
| 8 | Statistics review | Wed Aug 5 | Wed Aug 12 |
| 9 | Verdict review | Thu Aug 6 | Thu Aug 13 |

Day 5 gets two days rather than one. Spec §2.2 permits one regenerate-and-retest
cycle if the Day 4 battery fails, and the original plan had no room for it. Slack,
not scope.

### What did NOT change
- Random seed remains **42** (spec §0).
- Day 4 acceptance criteria unchanged: KS p > 0.05 *or* |Cohen's d| < 0.2 for token
  length and perplexity; SBERT centroid distance d(ghost, holdout) ≤ 1.25 ×
  d(holdout, forget10).
- Perplexity still measured under **base** `meta-llama/Llama-3.2-1B-Instruct`.
- Ghost n unchanged: 600 generated, trimmed to 400.
- Validation target remains **holdout10**.
- Spec §2.2 step 6 stands: *never regenerate after seeing audit results.* Moving the
  freeze date does not move the freeze rule.

### Structural safeguard — preserved
No Pilot 1 audit scores are shared with the Track D executor until the ghost-set
SHA-256 is committed. That ordering is a dependency, not a deadline, and is
unaffected by the date change.

### Downstream impact
Pilot 1 GPU runs (spec §2.3) now begin after the Aug 8 freeze. Pilot 0 is independent
of the ghost set and may proceed in parallel. The ~10 GPU-hour Pilot 1 budget is
unchanged.

### Open item
`docs/DEVIATIONS.md` contained only its header before this entry, yet the Track D
brief refers to **D-001** (TOFU member/holdout answer-length difference: 36.72 vs
42.33 tokens, KS p = 6.0e-11, d = −0.476) as already recorded. If D-001 lives only in
another document it should be transcribed here so the numbered sequence is complete.
D-001 was independently reproduced on Day 1 — see `ghosts/DAY1_NOTES.md` Finding 5.

## D-003 — Generator substituted: Anthropic API → Google Gemini API

| Field | Value |
|---|---|
| Recorded (UTC) | 2026-08-26 |
| Track | D — Ghost set construction, spec §2.2 step 2 |
| Type | Pre-registered value changed. `ghosts/DECISIONS.md` item 5 named "Anthropic API" as the generator; this replaces it with "Google Gemini API" (`gemini-3.6-flash`). |
| Status | Open — closes when the Day 4 validation battery is run against ghosts generated under this decision |

### What changed
The generator is now the Google Gemini API (via Google AI Studio), not the Anthropic
API. `scripts/day2_generate.py` gained a `--provider {anthropic,gemini}` flag; both
code paths remain in the script.

**Model string correction, same day:** the first live call used `gemini-2.5-flash`
and failed immediately with a 404 from Google's own API: *"This model
models/gemini-2.5-flash is no longer available to new users. Please update your code
to use models/gemini-3.6-flash."* No candidate was generated under `gemini-2.5-flash`
(the call failed before any checkpoint was written), so the default was corrected to
`gemini-3.6-flash` in place rather than opening a separate deviation entry — this is a
vendor model-availability fact discovered at execution time, not a second scientific
choice.

### Why
The first live pilot call under the original choice failed before generating anything:

```
anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error':
{'type': 'invalid_request_error', 'message': 'Your credit balance is too low
to access the Anthropic API. Please go to Plans & Billing to upgrade or
purchase credits.'}}
```

This is a student project with no budget for API credits. Google's Gemini API offers
a genuinely free tier (no card on file) with quota far exceeding this run's needs
(~500 requests/day and 250k tokens/minute against a 30-call, ~4k-token-per-call
workload). This is a budget substitution, not a scientific one: zero ghost candidates
existed under the Anthropic choice at the time of this change (the failing call
produced no checkpoint), so no result is being tuned against.

### Known, stated risk — not a fix, a tradeoff being made explicitly
TOFU's actual corpus was generated by GPT-4. Neither Anthropic's Claude nor Google's
Gemini is GPT-4, so both carry some baseline stylistic distance from TOFU's own
generation that Day 4's SBERT-centroid and perplexity tests are designed to catch.
This substitution does not reduce or increase that baseline risk in a way that can be
claimed in advance — no data exists yet under either generator to compare. If Day 4
fails specifically on the SBERT or perplexity criteria for reasons that look
generator-style-related, that is the intended catch mechanism working, not a pipeline
defect, and would itself become a new deviation entry (retry with a different
generator, re-run the cheap 5-author pilot again before spending full quota).

### What did NOT change
- Random seed remains **42**.
- Length target remains **holdout10**, mean 42.33 / sd 10.92 (`DECISIONS.md` item 4).
- Day 4 acceptance criteria unchanged: KS p > 0.05 *or* |Cohen's d| < 0.2 for token
  length and perplexity; SBERT centroid distance ≤ 1.25 × d(holdout10, forget10).
- Ghost n unchanged: 600 generated, trimmed to 400.
- Topic slots, exemplars, Day 1 schema reuse — all unchanged; only the API vendor and
  model string differ.
- Spec §2.2 step 6 stands: never regenerate after seeing audit results.

### Revert path
Reverting requires no code change: run the same script with `--provider anthropic`
once Anthropic credits are available. Nothing generated under Gemini needs to be kept
or deleted to do this — Gemini-provider checkpoints and Anthropic-provider checkpoints
are both dated and provider-tagged in their `_meta` block, so provenance stays clear
either way. `scripts/day2_generate.py` at commit `c4b4292` (Anthropic-only, pre this
deviation) also remains in git history if a full rollback of the script itself is ever
wanted.

## D-004 — Primary generator moved to Groq

| Field | Value |
|---|---|
| Recorded (UTC) | 2026-08-26 |
| Track | D — Ghost set construction, spec §2.2 step 2 |
| Type | Pre-registered value changed again. Primary generator is now the Groq API (`llama-3.3-70b-versatile`), not Google Gemini. |
| Status | Open — closes when the Day 4 validation battery is run against ghosts generated under this decision |

### What changed
`scripts/day2_generate.py` gained a third provider, `--provider groq`, now the
default. The Anthropic and Gemini code paths both remain in the script, unchanged.

**Model string correction, same day:** the first live call used
`llama-3.3-70b-versatile` and failed immediately with a 404 from Groq's own API:
*"The model `llama-3.3-70b-versatile` does not exist or you do not have access to
it."* Checked directly against Groq's models documentation: that model is now listed
Enterprise-tier only. No candidate was generated under it (the call failed before any
checkpoint was written), so the default was corrected to `openai/gpt-oss-120b`
(confirmed accessible on the free/developer tier: 250K TPM, 1K RPM) in place rather
than opening a separate deviation entry — same category as D-003's gemini-2.5-flash
correction: a vendor access-tier fact discovered at execution time, not a second
scientific choice.

### Why — correcting D-003's own quota assumption
D-003 assumed Gemini's free tier gave "~500 requests/day," based on published limits
for `gemini-2.5-flash`. That assumption did not hold for the model actually used:
after `gemini-2.5-flash` turned out to be retired for new users (same-day correction,
above) and generation moved to `gemini-3.6-flash`, the real confirmed quota on this
project's own account was **20 requests/day** —

```
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. quotaId:
GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20
```

— confirmed directly on the Google AI Studio usage page (21/20 shown after Pilot 2).
20/day cannot complete a 30-author run (each author can need 1-3 calls) in any
reasonable timeframe. Groq's free tier has a far higher daily request cap and requires
no card, at the cost of somewhat less certain long-instruction precision than Gemini
(unverified — no Groq data exists yet for this task; the same pilot mechanism that
caught Gemini's length miscalibration applies here too).

### Known, stated risk (same category as D-003, restated for the new vendor)
Groq serves open-weight models (Llama, GPT-OSS, Qwen), not GPT-4. Their stylistic
distance from TOFU's real GPT-4-generated corpus is unmeasured, same as it was for
Gemini and would be for Anthropic. No claim is made in advance about which of the
three generators sits closest to GPT-4's style — Day 4's SBERT/perplexity tests are
the only valid way to find out, and only for the generator actually used to build the
final 400.

### What did NOT change
- Random seed remains **42**.
- Length target remains **holdout10**, mean 42.33 / sd 10.92 (`DECISIONS.md` item 4);
  `WORDS_PER_TOKEN_EMPIRICAL` (derived from Gemini's pilot output) still applies as
  the prompt's practical word-count guide — Groq's own pilot may show it needs its
  own recalibration, per `ghosts/DAY2_NOTES.md` conventions.
- Day 4 acceptance criteria unchanged.
- Ghost n unchanged: 600 generated, trimmed to 400.
- Topic slots, exemplars, Day 1 schema reuse, JSON-output hardening (single-quote
  guidance) — all unchanged.
- Spec §2.2 step 6 stands: never regenerate after seeing audit results.

### Revert path
Same as D-003: no code change needed to go back to `--provider gemini` (once its
daily quota resets) or `--provider anthropic` (once funded). All three provider code
paths coexist in the script; checkpoints are provider-tagged in their `_meta` block.

## D-005 — Remaining authors backfilled with Anthropic; ghost set is mixed-generator

| Field | Value |
|---|---|
| Recorded (UTC) | 2026-09-01 |
| Track | D — Ghost set construction, spec §2.2 step 2 |
| Type | Pre-registered value reverted (partially) + a new, explicit choice: the final 400 ghost authors are NOT all from one generator. |
| Status | Open — closes when the Day 4 validation battery is run against the completed, mixed-generator 30-author set |

### What changed
By this point, 14 of 30 authors had real, successfully-generated checkpoints from
Groq (`openai/gpt-oss-120b`): author_ids 0,1,2,3,4,5,6,7,9,12,13,14,15,16. The
remaining 16 (8,10,11,17,18,19-29) were either stuck on Groq's `json_validate_failed`
error (unfixable from our side — Groq returns an empty `failed_generation`, giving
nothing to diagnose) or blocked by Groq's 200K-tokens/day cap (D-004 addendum).

Rather than either (a) waiting an unknown number of hours/days for Groq's rolling
daily quota to free enough headroom to finish the remaining 16, or (b) discarding the
14 real, valid Groq authors to regenerate all 30 from a single generator, the 16
remaining authors will be generated with `--provider anthropic` (funds added to the
account) and the 14 Groq authors will be kept as-is.

### Why this is a deliberate choice, not a shortcut
The 14 Groq authors are genuinely valid data — passing them through spec's own
Day 2 pilot length check (Pilot 2, `DAY2_NOTES.md`: d=+0.122, KS p=0.0159, PROCEED)
before this decision was made. Discarding them would trade real, already-validated
work for uniformity alone, with no stated scientific reason. Every checkpoint already
records its generator in `_meta.provider` and `_meta.model` (added when the provider
flag was first introduced), so the final `candidates_raw.jsonl` and `final_400.jsonl`
can always be split and reported by generator — this was not retrofitted for this
decision, it was already in place.

### Known, stated risk (extends D-003/D-004, does not resolve it)
The ghost set's stylistic distance from TOFU's real GPT-4-generated corpus is now a
property of TWO generators, not one. Day 4's SBERT-centroid and perplexity tests run
against the whole 400-row set; if they fail, it will not be possible to attribute the
failure to "the generator" as a single variable without a further, separate per-
generator breakdown (comparing Groq-only rows vs. Anthropic-only rows against
holdout10 independently) — worth doing as a diagnostic regardless of the aggregate
result, not only if it fails.

### What did NOT change
- Random seed remains **42**; ghost n unchanged (600 generated, trimmed to 400).
- Length target remains **holdout10**, mean 42.33 / sd 10.92.
- `WORDS_PER_TOKEN_EMPIRICAL` (Gemini-derived, empirically fine for Groq per Pilot 2)
  carries over to Anthropic untested — Anthropic's own pilot output should be
  spot-checked against it before assuming it transfers a third time.
- Day 4 acceptance criteria, topic slots, exemplars, Day 1 schema reuse, JSON-output
  hardening — all unchanged.
- Spec §2.2 step 6 stands: never regenerate after seeing audit results.

### Not a revert path this time
Unlike D-003/D-004, there is no clean "revert" — the ghost set is now permanently a
Groq+Anthropic mix unless the 14 Groq authors are deliberately discarded and
regenerated later (a future decision, not implied by this one).
