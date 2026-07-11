# Pilot 0 + Pilot 1 — Execution Spec
**Project:** Auditor-paradigm evaluation of LLM unlearning (working title)
**Owner:** Shravani (solo through June; Pilot 1 ghost-generation QA delegable in July)
**Status:** Pre-registered before any GPU run. Do not edit thresholds after first result is seen.
**Hardware floor:** 16 GB GPU (T4/L4). Everything in this spec fits Colab Pro.

---

## 0. Pre-registration block (fill in and freeze BEFORE running)

| Item | Value (locked) |
|---|---|
| Random seeds | 0, 7, 42, 123, 2024 (Pilot 0–1 use seed 42 only; others reserved) |
| Member n | 400 (full forget10 split) |
| Non-member (holdout) n | 400 |
| Ghost n | ≥ 400 (target 600 generated, keep best 400 after validation) |
| AUC CI method | Stratified bootstrap, 2,000 resamples, 95% percentile CI |
| Confound verdict rule | See §3.4 — written here before data is collected |
| Date locked | __11/7/2026________ |

Commit this file to your repo before the first GPU job. The point of pre-registration is that when a reviewer asks "did you pick this threshold after seeing results?" the git history answers for you.

---

## 1. Pilot 0 — Infrastructure smoke test

**Goal:** Prove the whole plan is mechanically possible: checkpoints load, hidden states are extractable, your auditor battery runs against OpenUnlearning models, and known results replicate.
**Budget:** ~5 GPU-hours, 2–3 working days.
**This pilot produces no science.** Its only outputs are pass/fail checks and runtime measurements.

### 1.1 Environment

```bash
git clone https://github.com/locuslab/open-unlearning.git
cd open-unlearning
pip install -e .            # follow repo README if this differs
pip install scikit-learn matplotlib
huggingface-cli login        # Llama-3.2 base requires accepting Meta's license
```

Pin and record: `transformers`, `torch`, `peft`, `datasets` versions into `ENVIRONMENT.md` on day one.

### 1.2 Checkpoints to pull

**Verified IDs (anchor models):**

| Role | HuggingFace ID |
|---|---|
| Target (trained on all 200 authors) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| Retain reference (never saw forget10) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` |

**Unlearned method checkpoints — discovery step (do not guess names):**

```python
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="open-unlearning")
for m in models:
    print(m.modelId)
```

From the listing, select **two** unlearned checkpoints on the
`Llama-3.2-1B-Instruct` / `forget10` configuration: one **GradAscent** and one
**NPO** (exact repo names will be visible in the listing; the org also hosts
`pos_*`/`neg_*` meta-evaluation pools — ignore those for now, they are metric-
calibration models, not method checkpoints). If no per-method checkpoints
exist for the 1B model, fall back to running OpenUnlearning's own unlearn
command for GradAscent on forget10 (~1–2 h on L4 at 1B scale):

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=PILOT0_GA
```

Record every checkpoint ID/commit hash you use in `CHECKPOINTS.md`.

### 1.3 Data splits (OpenUnlearning conventions)

- **Members:** TOFU `forget10` (400 QA, 20 authors).
- **Non-members:** the holdout split used by OpenUnlearning's MIA evals
  (`holdout10` in their eval configs). Confirm by inspecting
  `configs/experiment/eval/tofu/` and the MIA metric configs — record the exact
  dataset path you find.
- Tokenization, chat template, and answer-span masking: reuse OpenUnlearning's
  eval pipeline rather than porting your TinyLlama code. Your TinyLlama
  masking logic stays as a cross-check only.

### 1.4 Auditor battery to wire up

1. **LOSS** — mean answer-span log-likelihood (OpenUnlearning ships this).
2. **ZLib** — LOSS normalized by zlib-compressed answer byte length (ships).
3. **Min-K%++** — ships in OpenUnlearning's MIA set.
4. **Hidden-Norm oracle (yours)** — register a forward hook on the final
   RMSNorm (`model.model.norm`); score = −mean(‖h_t‖²) over answer-s
   pan tokens.
   Port from your TinyLlama code; verify the hook fires on Llama-3.2
   architecture and that masking matches OpenUnlearning's answer spans.

```python
feats = {}
def hook(mod, inp, out):
    feats["h"] = out.detach()
h = model.model.norm.register_forward_hook(hook)
# forward pass, then: sq = feats["h"].pow(2).sum(-1); score = -(sq * mask).sum() / mask.sum()
```

### 1.5 Smoke-test runs (seed 42 only)

Run all four auditors × four models (full, retain90, GradAscent-unlearned, NPO-unlearned) on members vs holdout. Save per-sample scores as CSV (`scores/{model}/{auditor}.csv`) — per-sample, not just AUCs; every later pilot reuses these files.

### 1.6 Pilot 0 go/no-go gates

| # | Check | GO threshold | If failed |
|---|---|---|---|
| G0.1 | All checkpoints load + generate on your GPU | bf16 load < 16 GB, no OOM at batch ≥ 4, seq 256 | Drop to fp16/8-bit; if still failing, GPU tier is wrong — fix before anything else |
| G0.2 | Memorization replicates | LOSS AUC (members vs holdout) on `_full` model ≥ 0.80 | Member/non-member construction is broken. STOP. Debug data pipeline — nothing downstream is valid |
| G0.3 | Retain model is clean | All token-prob AUCs on `retain90` in [0.40, 0.60] | Holdout split is contaminated or mismatched; re-derive non-members |
| G0.4 | Hidden-norm hook works | Norm scores finite, non-constant, reproducible across 2 runs (same seed) | Hook placement / masking bug |
| G0.5 | Runtime budget | Full 4-auditor battery ≤ 30 min per checkpoint (n=800 total samples) | If > 30 min: profile; batch the forward passes (all 4 auditors share one forward pass — compute them together) |
| G0.6 | Unlearning visibly moves a metric | On the GradAscent checkpoint, ≥ 1 token-prob AUC drops ≥ 0.15 vs `_full` | Wrong checkpoint selected (e.g., a retain model mislabeled) — recheck IDs |

**Hard rule:** G0.2 failing means stop entirely. It is the foundation of every other number.

### 1.7 Pilot 0 deliverables
- `ENVIRONMENT.md`, `CHECKPOINTS.md`
- `scores/` per-sample CSVs (16 files: 4 models × 4 auditors)
- One table: AUC ± bootstrap CI, 4 models × 4 auditors
- Measured runtime per checkpoint → updates the GPU-hour budget for Pilot 2

---

## 2. Pilot 1 — Ghost-set confound test

**Goal:** Determine whether the auditors (especially the norm oracle) detect *membership* or merely *forget-distribution side-effects* of unlearning.
**Budget:** ~10 GPU-hours + ~1 day of generation/validation work. 1.5 weeks.
**Depends on:** Pilot 0 G0.1–G0.5 passing.

### 2.1 Three-tier probe design

The confound decomposes into three levels; you need all three tiers to attribute signal correctly:

| Tier | Set | What it shares with members | Source |
|---|---|---|---|
| T1 | Holdout non-members | Template + style + TOFU authorship; never trained | Existing TOFU holdout |
| T2 | Perturbed/paraphrased forget answers | Same *entities* as members; text never trained verbatim | Ships with TOFU (`forget10_perturbed`) |
| T3 | **Ghost authors** | Template + style only; entities never existed anywhere in training | You generate (§2.2) |

Interpretation grid: signal on T1-vs-members = legitimate membership signal. Signal on T2 = entity-level (not verbatim) memorization — interesting, not a confound. Signal on T3 (ghosts scoring differently under unlearned vs full model, when the model never saw them in either) = **distributional confound** — the auditor is measuring collateral damage of unlearning, not residual membership.

### 2.2 Ghost-set generation protocol

1. **Schema extraction.** From 20 random TOFU authors, extract the attribute
   schema (name structure, nationality, birth year range, genre, awards,
   family details, # books) and the 20-question template per author.
2. **Generation.** Use a strong LLM to generate **30 new fictional authors ×
   20 QA = 600 candidates**, prompting with: the schema, 3 full TOFU author
   exemplars, and explicit instructions to match answer length (record TOFU's
   answer-length mean ± sd and constrain to it).
3. **Name-collision filter.** String/fuzzy match against all 200 TOFU author
   names AND against a real-author list (no real people). Drop collisions.
4. **Distribution-match validation (pre-registered acceptance tests).** All
   tests compare ghost answers vs **holdout** answers:
   - Token-length: two-sample KS test, require p > 0.05 *or* |Cohen's d| < 0.2.
   - Perplexity under the **base** `meta-llama/Llama-3.2-1B-Instruct` (NOT the
     TOFU-tuned model — the base model saw neither set): KS p > 0.05 or |d| < 0.2.
   - Embedding check: SBERT centroid distance ghost↔holdout ≤ 1.25 × centroid
     distance holdout↔forget10. (Guards against the generator drifting in style.)
5. **Trim to best 400** by dropping the worst length/perplexity outliers, then
   **re-run the validation on the final 400** (trimming can shift distributions).
6. Freeze the ghost set. Hash it. Never regenerate after seeing audit results.

**Failure mode to expect:** generated text being detectably "LLM-flavored" vs TOFU's text. If validation fails twice, switch generator model or move to fill-in-template generation (programmatic slot-filling of the TOFU question templates with sampled attributes), which trades naturalness for distribution control.

### 2.3 Runs

For each model in {full, retain90, GradAscent-unl., NPO-unl.} × each auditor in {LOSS, ZLib, Min-K%++, Hidden-Norm}: score members, T1, T2, T3 (per-sample CSVs again). One forward pass per sample serves all four auditors — batch accordingly. Total ≈ 4 models × 1,600 samples.

### 2.4 Pre-registered verdict rules (the actual decision logic)

Define, per auditor A and unlearned model M:
- **Membership signal:** AUC_A(members vs T1) on M
- **Confound signal:** AUC_A(T3 scored under M vs T3 scored under `_full`) — paired by sample; if the auditor's score on never-seen ghosts shifts when unlearning is applied, that shift is distributional side-effect by construction.

| # | Condition (bootstrap 95% CI basis) | Verdict | Consequence |
|---|---|---|---|
| G1.1 | Confound signal CI excludes 0.5 AND point estimate ∉ [0.42, 0.58] for the **norm oracle** | Norm oracle is confounded | Angle 3 (confound study) **promotes to lead candidate**; all Pilot 2 results must be reported confound-adjusted |
| G1.2 | Same condition for **token-prob** auditors | Token-prob MIAs confounded too | Stronger version of angle 3: "the entire audit stack measures side-effects" — this is the best-case headline |
| G1.3 | All confound signals' CIs include 0.5 | Auditors pass | Angle 3 demotes to a robustness subsection; proceed to Pilot 2 (matrix) as planned |
| G1.4 | Membership signal ≈ confound signal in magnitude for an auditor | That auditor's "detection" is ~fully explained by distribution shift | Drop that auditor as a *membership* tool; reframe it as an "unlearning-detection" tool (still publishable, different claim) |
| G1.5 | T2 (entity paraphrase) signal ≫ T1 signal change | Verbatim vs entity memorization dissociation | Bonus finding — log it, don't chase it yet |

**Ambiguity rule (decide now):** if the norm-oracle confound CI is [0.44–0.62]-ish straddling the boundary, the tiebreak is effect size on T3 score *shift* (paired Cohen's d): |d| ≥ 0.3 → treat as confounded. No re-running with new ghosts to "check."

### 2.5 Pilot 1 deliverables
- Frozen, hashed ghost set + validation report (the KS/d numbers)
- 4 × 4 × 4 score CSVs (model × auditor × {members, T1, T2, T3})
- One figure: per-auditor membership-signal vs confound-signal scatter
- A one-paragraph verdict citing which gate (G1.1–G1.5) fired → this paragraph *is* the input to the Pilot 2 design decision

---

## 3. Timeline & budget roll-up

| Week | Work | GPU-hrs |
|---|---|---|
| 1 (days 1–3) | Pilot 0: env, checkpoints, hooks, smoke runs, gates | ~5 |
| 1 (days 4–5) | Ghost schema + generation + validation (CPU/API work) | 0 |
| 2 | Pilot 1 runs + bootstrap analysis + verdict memo | ~10 |
| End wk 2 | Decision: Pilot 2 design conditioned on G1 verdict | — |

Two known risks to watch: (a) Meta license gating on Llama-3.2 — request access on day 1, approval can lag; (b) Colab session limits — checkpoint per-sample scores to Drive incrementally so a disconnect costs minutes, not a run.

---

## 4. What NOT to do in these two pilots
- No multi-seed runs yet (reserved seeds stay reserved).
- No OPC training yet — OPC enters at Pilot 2 on the same Llama-3.2-1B base.
- No threshold tuning, no "just one more ghost batch," no peeking at Pilot 2
  questions with Pilot 1 data. Every deviation gets logged in `DEVIATIONS.md`
  with a reason — reviewers forgive logged deviations, not silent ones.
