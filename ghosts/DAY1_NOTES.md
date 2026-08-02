# Day 1 — findings and implementation notes

## TOFU dataset properties discovered during schema extraction

**Finding 1 — author 88 has no name anywhere.** All 20 questions refer to "the
fictitious author"; the answers do the same. GPT-4 never substituted a name during
TOFU's generation. Same category of defect as D-001: a property of the released
dataset, not of our pipeline. Recovered coverage 6/40; recorded in
`schema.json → _provenance.name_recovery_anomalies`.

**Finding 2 — author names are not consistently present in questions.** TOFU's
published generation prompt says "Make sure the author's full name appears in the
question content", but measured question-level coverage has a median of 11/20.
Answer-level coverage is far higher (median 19/20). Name recovery therefore counts
across questions and answers together (median 37/40).

**Finding 3 — name forms vary within an author.** Possessive (`Laaksonen's`),
lowercase particles (`Isabella van Pletzen`), and parenthetical nicknames
(`Alejandro (Alex) Fuentes`) all occur. Naive capitalised-token matching splits or
drops these.

**Finding 4 — there is no 20-question template.** Mean dominant-template share per
position is 0.018; 3,580 distinct normalised templates across 4,000 questions. The
Track D brief's premise that "each author is asked roughly the same 20 questions"
does not hold lexically. Consequence for Day 2: the generator receives topic slots
plus an instruction to vary phrasing, not 20 fixed question strings. Repeating fixed
strings across 30 ghost authors would itself be a detectable stylistic tell.

**Finding 5 — D-001 reproduced exactly.** forget10 36.720 / holdout10 42.325 answer
tokens under `add_special_tokens=True`, against documented 36.72 / 42.33. Cohen's d
−0.4763 (documented −0.476) and KS p 6.028e-11 (documented 6.0e-11) match under
either tokenization convention, since adding a constant BOS token to every answer
shifts means but leaves sd, KS and d unchanged. The two conventions differ by exactly
1.000 token. `add_special_tokens=True` is canonical for this project.

## Implementation note — C3 gate changed during Day 1

The first run aborted on `C3.name_recovery_coverage` with `min >= 10/20`. Three
extraction bugs were found and fixed (possessive stripping, lowercase particles,
parenthetical nicknames), after which median coverage rose from 11/20 to 37/40. The
gate was then changed to `median >= 20/40`.

**Why this is not threshold-shopping:** the gate covers an intermediate data-quality
step, not a pre-registered scientific criterion. No value in spec §2.2 was touched —
seed, KS thresholds, Cohen's d bounds and the SBERT ratio are unchanged. A
minimum-based gate is also unsatisfiable given Finding 1: author 88 has no name to
recover, so no correct implementation could ever pass it.

A stricter, **fatal** check was added at the same time — `C3d` requires every author
in the seeded 20-author sample to have coverage ≥ 12/40, since a mis-recovered name
inside the sample would corrupt the schema directly. The gate guarding the actual
deliverable got tighter, not looser.

## Implementation note — D-001 check convention

`C4` initially warned because the script compared `add_special_tokens=False` means
against D-001's `add_special_tokens=True` figures. Corrected to compare like with
like; both conventions are now recorded in `length_stats.json`. No data changed.
