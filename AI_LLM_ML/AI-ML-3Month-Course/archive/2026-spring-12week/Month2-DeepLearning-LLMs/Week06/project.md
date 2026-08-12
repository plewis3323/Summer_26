# Week 06 — Mini-Project: Train a Small GPT on Heavy-Ion Abstracts

## Problem

Scrape an arXiv heavy-ion / QGP corpus. Tokenize with BPE. Train a ~10M-parameter decoder-only transformer on it, from code you wrote, reaching a meaningful train/val loss curve. Sample from the trained model and include qualitative observations on whether it has learned domain-specific vocabulary and phrasing.

This is the primary deliverable of Month 2, Week 1. It'll be the code and artifact you reference when talking to colleagues about "building an LLM from scratch."

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/hep_gpt/
│   ├── __init__.py
│   ├── data/
│   │   ├── scrape.py           # arXiv API → abstracts.txt
│   │   └── prepare.py          # tokenize, train/val split, binary shards
│   ├── model.py                # GPTConfig, CausalSelfAttention, MLP, Block, GPT
│   ├── train.py                # training loop with AdamW, warmup+cosine, wandb
│   ├── sample.py               # generation with temperature, top-k, top-p
│   └── cli.py                  # click or typer CLI
├── tests/
│   ├── test_model.py           # param count, forward shape, causal mask
│   ├── test_data.py            # round-trip tokenize/detokenize
│   └── test_sample.py          # sampling determinism with seed
├── configs/
│   ├── shakespeare_debug.yaml
│   └── heavy_ion_10m.yaml
└── results/
    ├── train_curve.png
    ├── samples.md              # prompts and outputs
    └── metrics.json
```

## Acceptance criteria

1. `uv run pytest -q` passes including a shape/param-count test for the GPT.
2. Single-batch overfit (E4) reaches loss < 0.1.
3. Shakespeare smoke-test completes and generates recognizable English.
4. Heavy-ion corpus scrape downloads ≥ 5000 unique abstracts.
5. 10M-parameter model trains for ≥ 30k steps with train/val loss logged to wandb; final val loss reported.
6. Generation samples produced for at least 5 physics prompts. Observations written up in 300+ words.
7. README includes param-count derivation, wall-clock, hardware used, token count, and estimated FLOPs.
8. Code is formatted with ruff, passes `uv run ruff check`.

## Suggested plan (6 days)

- **Day 1.** Scrape arXiv. Tokenize. Build binary shards (see below).
- **Day 2.** Model implementation. Tests. Single-batch overfit.
- **Day 3.** Shakespeare smoke run.
- **Day 4.** Full heavy-ion training run. Monitor in wandb.
- **Day 5.** Sample from trained model. Explore temperature/top-p sweeps.
- **Day 6.** Write-up, clean up, commit.

## Data: arXiv scraper

The arXiv API (https://arxiv.org/help/api) is free and politely-rate-limited. Use `arxiv` Python package.

```python
# src/hep_gpt/data/scrape.py
import arxiv
from pathlib import Path
import time, re

CATEGORIES = ["hep-ex", "nucl-ex", "nucl-th"]
KEYWORDS = ["heavy-ion", "heavy ion", "QGP", "quark-gluon plasma",
            "Au+Au", "Pb+Pb", "sPHENIX", "STAR experiment", "ALICE",
            "RHIC", "elliptic flow", "jet quenching"]

def scrape(out: Path, max_per_cat: int = 3000):
    client = arxiv.Client(page_size=200, delay_seconds=3.0)
    abstracts = {}
    for cat in CATEGORIES:
        q = f"cat:{cat} AND (" + " OR ".join(f'"{k}"' for k in KEYWORDS) + ")"
        search = arxiv.Search(query=q, max_results=max_per_cat,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        for r in client.results(search):
            abstracts[r.entry_id] = r.summary.replace("\n", " ").strip()
        time.sleep(1)
    with out.open("w") as f:
        for a in abstracts.values():
            # light cleaning
            a = re.sub(r"\s+", " ", a)
            f.write(a + "\n")
    return len(abstracts)
```

Expect ~5–10k unique abstracts after dedupe. ~5–15 MB text.

## Data: tokenization and shards

```python
# src/hep_gpt/data/prepare.py
import tiktoken, numpy as np
from pathlib import Path

enc = tiktoken.get_encoding("gpt2")

def prepare(input_txt: Path, out_dir: Path, val_frac: float = 0.1):
    text = input_txt.read_text()
    ids = enc.encode(text)
    n = len(ids)
    split = int(n * (1 - val_frac))
    train = np.array(ids[:split], dtype=np.uint16)
    val = np.array(ids[split:], dtype=np.uint16)
    np.save(out_dir / "train.npy", train)
    np.save(out_dir / "val.npy", val)
    print(f"{n:,} tokens, {split:,} train / {n - split:,} val")
```

`uint16` works because GPT-2 vocab ≤ 65536. Saves 2× storage vs int32.

## Model

Match nanoGPT's layout closely. Start from your E3 implementation. Make sure:

- `n_embd` divisible by `n_head`.
- `vocab_size` = 50304 (GPT-2's 50257 padded to multiple of 64 for tensor-core efficiency).
- Weight tying: `self.lm_head.weight = self.tok_emb.weight`.
- `bias=False` on linear layers (following LLaMA; cleaner gradients).

Config:
```yaml
# configs/heavy_ion_10m.yaml
n_layer: 6
n_head: 6
n_embd: 384
block_size: 256
vocab_size: 50304
dropout: 0.1
bias: false
```

Param count: ~10M. Verify with `sum(p.numel() for p in model.parameters())`.

## Training

```python
# sketch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

lr_max = 3e-4
warmup = 2000
max_steps = 30_000
model = GPT(config)
opt = AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)

def get_lr(step):
    if step < warmup:
        return lr_max * step / warmup
    import math
    progress = (step - warmup) / (max_steps - warmup)
    return lr_max * 0.1 + 0.5 * (lr_max - lr_max * 0.1) * (1 + math.cos(math.pi * progress))

for step in range(max_steps):
    xb, yb = get_batch("train")
    for g in opt.param_groups:
        g["lr"] = get_lr(step)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(xb, yb)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 500 == 0:
        val_loss = eval_loss(model)
        wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "lr": get_lr(step)}, step=step)
```

Batch size: start at 64. Adjust to GPU memory.

## Expected metrics

For a 10M-param GPT on ~5M tokens of heavy-ion corpus:
- Initial train loss: ~10.8 (uniform over 50304 vocab)
- After 1000 steps: ~6
- After 10000 steps: ~3.5
- Final (30k steps): ~2.5–3.0 training, ~3.0–3.5 val

Gap between train and val is expected — your corpus is undersized per Chinchilla. The gap is the overfitting you're supposed to see.

## Sampling for the write-up

Try these prompts:

```
"The elliptic flow coefficient v2 of"
"Jet quenching in quark-gluon plasma is"
"At √s_NN = 200 GeV, Au+Au collisions exhibit"
"The sPHENIX detector is designed to"
"In heavy-ion collisions, the pi0 meson"
```

For each, sample at T=0.8, top-p=0.9, max_new_tokens=150. Paste raw outputs into `results/samples.md`. Annotate: where does the model get physics terminology right, where does it produce plausible-but-wrong, where does it break down completely?

## Write-up template (`project/README.md`)

```
# HEP-GPT: a small transformer trained on heavy-ion arXiv abstracts

## Architecture
Decoder-only transformer. 6 layers, 6 heads, d_model=384, block_size=256. 10.1M parameters (excluding ~19M embedding + LM head, which are weight-tied).

## Data
~8,400 unique abstracts from arXiv hep-ex/nucl-ex/nucl-th (2015–2025) filtered for heavy-ion keywords. 4.7M GPT-2 BPE tokens total, 4.2M train / 0.5M val.

## Training
AdamW, lr 3e-4, 2000-step linear warmup + cosine decay to 3e-5. Batch size 64, block size 256, 30000 steps.
Hardware: 1× RTX 3060 12GB. Wall-clock: 6h 10m. ~6 TFLOPs effective (~30% of peak).

## Loss
Final train loss 2.83, val 3.21. Gap expected given corpus size vs Chinchilla optimal.

## Samples
[prompts and outputs, see samples.md]

## Observations
[300 words on what the model learned, where it gets physics right/wrong, limits]

## Reproducibility
uv.lock committed. Full training command: `uv run hep-gpt train configs/heavy_ion_10m.yaml`
```

## Common failure modes

- **arXiv API rate-limited.** Respect the delay; 3 seconds/request is the floor. If you get 429s, wait longer.
- **Val loss plateaus early.** Corpus too small. Scale up keyword list. Or scrape ALL of nucl-ex since 2010.
- **Loss goes NaN at step ~2000.** Warmup too short or LR too high. Print gradient norms.
- **Training is very slow.** Check `torch.autocast` is actually active (print dtypes inside forward). Check `torch.compile` isn't silently erroring out. Use `nvidia-smi` to see GPU util.
- **Samples are gibberish.** Either undertrained (loss > 5) or sampling at too-high temperature (try T=0.3 to sanity-check).
- **OOM.** Drop batch size. Drop block size. Switch to grad accumulation. `pin_memory=True` doesn't fix OOM.

## Extensions (optional, if you have gas)

- Train a slightly larger model (30M params, n_layer=8, n_embd=512) and compare val loss. Observe scaling.
- Implement RoPE instead of learned absolute pos embeddings. Retrain. Compare.
- Swap GELU for SwiGLU. Retrain. Compare.
- Fine-tune your Shakespeare model on the heavy-ion corpus (transfer learning). Compare to from-scratch. This is a pedagogical preview of Week 07.

## Sign-off

Commit your training curves, sample outputs, tokenizer vocabulary stats, and 400-word reflection. Tick Week 06 boxes. Next week: you stop training from scratch and start fine-tuning real pre-trained models.
