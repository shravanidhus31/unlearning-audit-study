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
