# Pilot 0 — Step-by-Step Execution Guide (Colab Pro)

Companion to `pilot_0_1_execution_spec.md`. That file is the contract (gates,
thresholds); this file is the hands-on-keyboard walkthrough. Estimated total:
**2–3 working days, ~5 GPU-hours.**

---

## STEP 0 — Prerequisites (Day 0, no GPU, ~1–2 hours + waiting on Meta)

Do these immediately; item 2 has a multi-day approval lag.

1. **GitHub repo.** Create a private repo (e.g. `unlearning-audit-study`).
   First commit = the execution spec + this guide + the pre-registration block
   with seeds/thresholds filled in. The commit timestamp is your
   pre-registration evidence.
2. **HuggingFace access.** Create an HF account → Settings → Access Tokens →
   new token (type: Read). Then visit
   `huggingface.co/meta-llama/Llama-3.2-1B-Instruct` and click "Agree and
   access" on the gated form. Approval can take hours to days — request now.
   (The `open-unlearning/*` checkpoints themselves are ungated, but the base
   model is needed for Pilot 1's perplexity validation and for tokenizer
   fallbacks, so get access sorted.)
3. **Google Drive folder structure.** Create:
   ```
   MyDrive/unlearning_pilot/
   ├── scores/          # per-sample CSVs (the crown jewels)
   ├── checkpoints/     # only if you train your own GA model
   ├── logs/
   └── docs/            # ENVIRONMENT.md, CHECKPOINTS.md, DEVIATIONS.md
   ```
4. **Colab runtime choice.** Runtime → Change runtime type → **L4 GPU**
   (preferred). If only T4 is available it works, with one critical change:
   **T4 does not support bfloat16** — use `torch.float16` everywhere this
   guide says `bfloat16`. The 1B model is ~2.5 GB; either card is ample.

---

## STEP 1 — Environment (Day 1, ~30 min)

New Colab notebook → first cells:

```python
# Cell 1: Drive + workdir
from google.colab import drive
drive.mount('/content/drive')
WORK = '/content/drive/MyDrive/unlearning_pilot'

# Cell 2: clone + install
!git clone https://github.com/locuslab/open-unlearning.git /content/open-unlearning
%cd /content/open-unlearning
!pip install -e . -q
!pip install scikit-learn -q

# Cell 3: HF auth
from huggingface_hub import login
login()   # paste your read token

# Cell 4: freeze environment record
import torch, transformers, datasets, peft, sys, subprocess
env = f"""python={sys.version.split()[0]}
torch={torch.__version__}
transformers={transformers.__version__}
datasets={datasets.__version__}
peft={peft.__version__}
gpu={torch.cuda.get_device_name(0)}
"""
print(env)
open(f'{WORK}/docs/ENVIRONMENT.md','w').write(env)
```

If `pip install -e .` fails on a dependency conflict (common in Colab), fall
back to installing only what the standalone scorer below needs:
`pip install transformers datasets accelerate scikit-learn` — Pilot 0 can run
entirely on the standalone scorer; the OpenUnlearning pipeline becomes
load-bearing in Pilot 2, so log the failure in `DEVIATIONS.md` and move on.

---

## STEP 2 — Checkpoint discovery & selection (Day 1, ~30 min)

```python
from huggingface_hub import HfApi
api = HfApi()
ids = [m.modelId for m in api.list_models(author="open-unlearning")]
llama1b = sorted(x for x in ids if "Llama-3.2-1B" in x)
for x in llama1b: print(x)
```

From the printout, fill `CHECKPOINTS.md` with exactly four entries:

| Role | Selection rule |
|---|---|
| TARGET | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` (verified to exist) |
| RETAIN | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` (verified) |
| UNL-1 | A **GradAscent / forget10** checkpoint from the listing |
| UNL-2 | An **NPO / forget10** checkpoint from the listing |

Selection rules for UNL-1/UNL-2: must contain `Llama-3.2-1B`, `forget10`, and
the method name; **skip** anything prefixed `pos_` or `neg_` (those are
metric-calibration pools, not method checkpoints — using one invalidates
G0.6). If multiple hyperparameter variants exist, take the one matching
OpenUnlearning's default config (lr 1e-5, 10 epochs) or, failing that, the
most-downloaded one — and record which.

**Fallback if no 1B method checkpoints exist:** train GradAscent yourself
(~1–2 h on L4):
```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent \
  task_name=PILOT0_GA model=Llama-3.2-1B-Instruct
```
then copy the output dir to `{WORK}/checkpoints/` and use UNL-2 = skip
(3-model matrix is acceptable for Pilot 0; log it).

---

## STEP 3 — Data: members, non-members, prompt formatting (Day 1, ~1 h)

```python
from datasets import load_dataset
members  = load_dataset("locuslab/TOFU", "forget10")["train"]   # 400 QA
holdout  = load_dataset("locuslab/TOFU", "holdout10")["train"]  # non-members
print(len(members), len(holdout))
assert len(members) == 400
```

If `holdout10` raises an error, the split name differs in your TOFU/datasets
version — list available configs with
`from datasets import get_dataset_config_names; get_dataset_config_names("locuslab/TOFU")`
and cross-check against OpenUnlearning's MIA eval config
(`configs/experiment/eval/tofu/` and the `data/` configs reference the exact
holdout source). **Do not improvise a non-member set** — G0.3 exists to catch
exactly this. Record the resolved name in `CHECKPOINTS.md`.

**Prompt formatting** must match how the checkpoints were trained (chat
format). Use the tokenizer's own template:

```python
def format_qa(tokenizer, q, a):
    msgs = [{"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    return prompt, prompt + a + tokenizer.eos_token
```

Sanity check before any scoring: print one fully formatted sample and
eyeball that the answer text appears after the assistant header and the
prompt/answer boundary is where you think it is.

---

## STEP 4 — The unified scorer (Day 1–2, ~3 h of coding)

One forward pass per sample computes **all four auditors**. This is the most
important code of the pilot; the per-sample CSVs it writes are reused by every
later pilot.

```python
import torch, zlib, numpy as np, pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16   # torch.float16 on T4!

def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map="cuda")
    model.eval()
    return model, tok

@torch.no_grad()
def score_sample(model, tok, q, a, feats):
    prompt, full = format_qa(tok, q, a)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    f_ids = tok(full,   return_tensors="pt").input_ids.to("cuda")
    n_prompt = p_ids.shape[1]

    out = model(f_ids)                      # hook fills feats["h"]
    logits = out.logits.float()             # [1, T, V]

    # answer-span positions: predictions for tokens n_prompt..T-1
    # shift: logits[:, t] predicts token t+1
    tgt   = f_ids[0, n_prompt:]                       # answer tokens
    lgts  = logits[0, n_prompt-1:-1]                  # their predicting logits
    logp  = F.log_softmax(lgts, dim=-1)               # [Ta, V]
    tok_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)   # [Ta]

    # 1) LOSS
    loss_score = tok_lp.mean().item()

    # 2) ZLib
    zlib_len = len(zlib.compress(a.encode()))
    zlib_score = loss_score / zlib_len

    # 3) Min-K%++  (mu, sigma over vocab under p(.|x<t))
    probs  = logp.exp()
    mu     = (probs * logp).sum(-1)
    sigma  = ((probs * (logp - mu.unsqueeze(-1)).pow(2)).sum(-1)).clamp_min(1e-12).sqrt()
    z      = (tok_lp - mu) / sigma
    k      = max(1, int(0.20 * z.numel()))
    minkpp = z.topk(k, largest=False).values.mean().item()

    # 4) Hidden-Norm oracle (post-final-RMSNorm, answer span)
    h = feats["h"][0, n_prompt:]                      # [Ta, d]
    hidden_norm = -(h.float().pow(2).sum(-1).mean().item())

    return dict(loss=loss_score, zlib=zlib_score,
                minkpp=minkpp, hidden_norm=hidden_norm,
                n_answer_tokens=int(tgt.numel()))

def attach_hook(model, feats):
    def hook(mod, inp, out):
        feats["h"] = out.detach()
    return model.model.norm.register_forward_hook(hook)
```

**Verification cell (run before the full sweep — this is G0.4):**
```python
feats = {}
model, tok = load("open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
h = attach_hook(model, feats)
s1 = score_sample(model, tok, members[0]["question"], members[0]["answer"], feats)
s2 = score_sample(model, tok, members[0]["question"], members[0]["answer"], feats)
print(s1); assert s1 == s2, "non-deterministic scoring!"
assert all(np.isfinite(v) for v in s1.values())
```
Also verify hook placement: `print(model)` and confirm `model.model.norm` is
the final `LlamaRMSNorm` *before* `lm_head` (it is, on Llama-3.2 — but look).

---

## STEP 5 — The sweep: 4 models × {members, holdout} (Day 2, ~2–3 GPU-h)

```python
MODELS = {
  "full":     "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
  "retain90": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
  "unl_ga":   "<UNL-1 id from CHECKPOINTS.md>",
  "unl_npo":  "<UNL-2 id from CHECKPOINTS.md>",
}
SETS = {"members": members, "holdout": holdout}

import os, time
for mname, mid in MODELS.items():
    feats = {}
    model, tok = load(mid)
    hook = attach_hook(model, feats)
    for sname, ds in SETS.items():
        path = f"{WORK}/scores/{mname}__{sname}.csv"
        done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        start = len(done)                                  # resume support
        rows = done.to_dict("records")
        t0 = time.time()
        for i in range(start, len(ds)):
            r = score_sample(model, tok, ds[i]["question"], ds[i]["answer"], feats)
            r["idx"] = i
            rows.append(r)
            if (i+1) % 25 == 0:                            # incremental save
                pd.DataFrame(rows).to_csv(path, index=False)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(mname, sname, f"{time.time()-t0:.0f}s")
    hook.remove(); del model; torch.cuda.empty_cache()
```

The `% 25` incremental save is your Colab-disconnect insurance — a dropped
session costs ≤25 samples, and rerunning the cell resumes from the CSV.
Record the per-checkpoint wall time → that number is gate **G0.5** and sizes
the Pilot 2 budget.

---

## STEP 6 — Gates analysis (Day 2–3, ~1 h, CPU)

```python
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd

def auc_ci(mem, non, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(mem)), np.zeros(len(non))]
    s = np.r_[mem, non]
    point = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        mi = rng.integers(0, len(mem), len(mem))
        ni = rng.integers(0, len(non), len(non))
        boots.append(roc_auc_score(
            np.r_[np.ones(len(mem)), np.zeros(len(non))],
            np.r_[np.asarray(mem)[mi], np.asarray(non)[ni]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi

results = []
for m in MODELS:
    mem = pd.read_csv(f"{WORK}/scores/{m}__members.csv")
    non = pd.read_csv(f"{WORK}/scores/{m}__holdout.csv")
    for aud in ["loss","zlib","minkpp","hidden_norm"]:
        p, lo, hi = auc_ci(mem[aud].values, non[aud].values)
        results.append(dict(model=m, auditor=aud, auc=round(p,4),
                            ci=f"[{lo:.3f},{hi:.3f}]"))
table = pd.DataFrame(results).pivot(index="model", columns="auditor",
                                    values="auc")
print(table)
table.to_csv(f"{WORK}/scores/PILOT0_AUC_TABLE.csv")
```

Now walk the gates from the spec, in order, and write the answer next to each
in a `PILOT0_GATES.md`:

| Gate | Pass condition | Where to look |
|---|---|---|
| G0.1 | no OOM, models loaded | Step 5 ran |
| G0.2 | `full` row, `loss` col ≥ **0.80** | AUC table. **FAIL = STOP EVERYTHING** |
| G0.3 | `retain90` row, all token-prob aucs in [0.40, 0.60] | AUC table |
| G0.4 | determinism + finiteness asserts passed | Step 4 verification cell |
| G0.5 | per-checkpoint sweep ≤ 30 min | Step 5 timings |
| G0.6 | on `unl_ga`: ≥1 token-prob auc dropped ≥0.15 vs `full` | AUC table |

Interesting things you are *allowed to notice but not act on*: the
`hidden_norm` column on the unlearned models. Whatever it shows, it is a
single-seed, no-confound-control observation — log it, do not tweet it, do
not redesign Pilot 1 around it (Pilot 1's design is already frozen).

---

## STEP 7 — Close out (Day 3, ~1 h)

1. Commit to GitHub: `ENVIRONMENT.md`, `CHECKPOINTS.md`, `PILOT0_GATES.md`,
   `PILOT0_AUC_TABLE.csv`, the scorer notebook, and `DEVIATIONS.md` (even if
   it says "none").
2. Write a 5-line summary in the repo README: gates passed/failed, measured
   runtime per checkpoint, anything surprising.
3. If all gates green → start Pilot 1 §2.2 (ghost generation is CPU/API work;
   it can begin while you still have GPU quota cooling down).
4. If G0.2 or G0.3 failed → the bug is in data construction, not models.
   Debug order: (a) print formatted samples and check the chat template,
   (b) check the holdout config name, (c) check answer-span boundary
   (off-by-one in `n_prompt` is the classic error — symptom: LOSS AUC near
   0.5 on `full` because you're scoring prompt tokens).

## Common failure modes, pre-diagnosed
- **bf16 crash on T4** → switch `DTYPE` to `torch.float16`.
- **Gated-repo 403 on meta-llama** → approval pending; the open-unlearning
  checkpoints don't need it, proceed and circle back.
- **`holdout10` not found** → resolve via dataset config names + the
  OpenUnlearning eval configs; never substitute world-facts or real-authors
  (different distribution — would silently inflate every AUC).
- **AUC suspiciously = 1.000 everywhere including retain90** → leakage:
  you're scoring with answers in the prompt, or train/holdout overlap.
- **Colab disconnects mid-sweep** → rerun Step 5 cell; resume logic picks up
  from the last incremental save.
