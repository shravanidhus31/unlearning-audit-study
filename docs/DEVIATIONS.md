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
