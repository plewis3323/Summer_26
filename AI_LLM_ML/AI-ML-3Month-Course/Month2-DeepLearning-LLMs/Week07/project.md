# Week 07 — Mini-Project: Fine-Tune a Heavy-Ion Abstract Summarizer

## Problem

Using LoRA (or QLoRA if you go for the 7B model), fine-tune an open-weights instruct model to summarize arXiv heavy-ion physics abstracts into plain-English, three-sentence TL;DRs aimed at a physicist-but-not-specialist reader. Deploy it locally. Benchmark side-by-side with the base model. Push the adapter to HuggingFace Hub.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/hi_sumz/
│   ├── __init__.py
│   ├── teacher.py              # Claude-driven summary generation
│   ├── data.py                 # load/split abstracts + summaries
│   ├── train.py                # SFTTrainer + PEFT
│   ├── infer.py                # single-prompt inference
│   ├── eval.py                 # qualitative eval + simple ROUGE
│   └── cli.py
├── tests/
│   ├── test_teacher.py
│   ├── test_data.py
│   └── test_infer.py
├── configs/
│   ├── llama32_1b_r16.yaml
│   └── llama31_8b_qlora_r16.yaml
├── data/
│   ├── abstracts_raw.jsonl
│   ├── abstracts_with_summary.jsonl
│   └── splits/{train,val,test}.jsonl
└── results/
    ├── eval_pairs.html         # side-by-side blind-eval output
    ├── loss_curve.png
    ├── rank_ablation.png
    └── metrics.json
```

## Acceptance criteria

1. `uv run pytest -q` passes.
2. At least 500 (abstract, summary) pairs in your dataset. Teacher prompt documented.
3. Trained LoRA adapter checkpoint on a ≤ 3B base model. (Stretch: also on a 7B QLoRA base.)
4. Eval loss curve in wandb showing monotonic improvement (with expected noise).
5. Side-by-side A/B on 20 held-out abstracts: fine-tuned model wins ≥ 60%.
6. Rank ablation plot (r ∈ {4, 8, 16, 32}): eval loss and qualitative assessment.
7. Model served locally via vLLM with a working cURL client snippet.
8. Adapter pushed to HF Hub with a model card describing data, purpose, limits.
9. README documenting full training stack, commit SHA, uv.lock, hardware, wall-clock, win rate.

## Suggested plan (6–7 days)

- **Day 1.** Install HF stack. Load base model. Toy LoRA (E3-E4).
- **Day 2.** Build teacher pipeline. Generate 500+ summaries. Validate.
- **Day 3.** Format dataset with chat template. Split. Smoke-train.
- **Day 4.** Full training run. Watch eval loss.
- **Day 5.** Side-by-side eval. Rank ablation.
- **Day 6.** Deploy with vLLM. Push to Hub.
- **Day 7.** Write-up, clean tests, commit.

## Data: teacher-driven summary generation

```python
# src/hi_sumz/teacher.py
import anthropic
from pathlib import Path
import json

SYSTEM = """You are a physicist-editor writing summaries for a physics audience.
Given an arXiv heavy-ion / nuclear physics abstract, output a 3-sentence summary that:
- preserves the key physical claim and any numerical result (center-of-mass energy, efficiency, cross-section);
- uses plain language; avoids jargon unless essential;
- omits detector acronyms unless central to the result.
Write only the summary, no preamble."""

def summarize(client, abstract: str, model: str = "claude-sonnet-4-5") -> str:
    m = client.messages.create(
        model=model, max_tokens=256, temperature=0.2,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Abstract:\n\n{abstract}"}],
    )
    return m.content[0].text.strip()

def build(in_path: Path, out_path: Path, limit: int = 500):
    client = anthropic.Anthropic()
    with in_path.open() as fin, out_path.open("w") as fout:
        for i, line in enumerate(fin):
            if i >= limit: break
            abstract = line.strip()
            if not abstract or len(abstract) < 200: continue
            summary = summarize(client, abstract)
            fout.write(json.dumps({"abstract": abstract, "summary": summary}) + "\n")
```

Cost: ~$5–10 total at current Claude pricing (depends on abstract length).

Spot-check 10 summaries manually. Discard anything where the teacher hallucinates numbers or misses the main claim.

## Data: train / val / test split

```python
# src/hi_sumz/data.py
import json, random
from pathlib import Path

def split(jsonl: Path, out_dir: Path, seed: int = 1):
    rows = [json.loads(l) for l in jsonl.open()]
    random.Random(seed).shuffle(rows)
    n = len(rows); n_train = int(n * 0.8); n_val = int(n * 0.1)
    parts = {"train": rows[:n_train], "val": rows[n_train:n_train+n_val], "test": rows[n_train+n_val:]}
    for name, rs in parts.items():
        (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
```

## Chat formatting

```python
def format_for_sft(example, tokenizer):
    msgs = [
        {"role": "user", "content": f"Summarize this physics abstract in three plain-English sentences:\n\n{example['abstract']}"},
        {"role": "assistant", "content": example["summary"]},
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
```

## Config

```yaml
# configs/llama32_1b_r16.yaml
base_model: meta-llama/Llama-3.2-1B-Instruct
bf16: true
qlora: false

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

train:
  epochs: 2
  per_device_batch_size: 2
  grad_accum: 8
  lr: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.03
  max_seq_length: 1024
  packing: true
  gradient_checkpointing: true
  eval_steps: 50
  save_steps: 100
```

## Training (excerpt)

Use the boilerplate from E6. Log to wandb. Save best model by eval loss.

## Evaluation: side-by-side A/B

```python
# src/hi_sumz/eval.py
def blind_ab(abstracts, base_model, tuned_model, tokenizer, out_html="eval_pairs.html"):
    import random
    rows = []
    for ab in abstracts:
        left_is_tuned = random.random() < 0.5
        out_base = generate(base_model, tokenizer, ab)
        out_tuned = generate(tuned_model, tokenizer, ab)
        left, right = (out_tuned, out_base) if left_is_tuned else (out_base, out_tuned)
        rows.append({"abstract": ab, "left": left, "right": right, "left_is_tuned": left_is_tuned})
    # Render as HTML with hidden truth labels for offline scoring
    html = render_table(rows)
    Path(out_html).write_text(html)
```

Then open the HTML, score each row better-left / better-right / same, and reveal the answers at the bottom.

## Deployment: vLLM

```bash
# merge adapter
uv run python -m hi_sumz.cli merge --adapter out/final --out out/merged
# serve
uv run vllm serve out/merged --port 8000 --gpu-memory-utilization 0.85
```

Client:
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="_")
resp = c.chat.completions.create(
    model="out/merged",
    messages=[{"role":"user","content":"Summarize: ..."}],
    temperature=0.3, max_tokens=256,
)
print(resp.choices[0].message.content)
```

Measure: tokens/sec, first-token latency, memory.

## Pushing to HF Hub

```python
from huggingface_hub import HfApi
api = HfApi()
tuned.push_to_hub("plewis3323/heavy-ion-abstract-summarizer-v1")
tok.push_to_hub("plewis3323/heavy-ion-abstract-summarizer-v1")
```

Write the model card (`README.md` on the Hub) covering:

- Base model
- Data sources + approximate size
- Intended use (summarization of hep-ex/nucl-ex/nucl-th abstracts)
- Out-of-scope uses (general chat, code, anything not a physics summary)
- Training details (rank, lr, steps, hardware)
- Limitations and biases (teacher-model inherited biases, small dataset)

## Rank ablation plot

Train `r ∈ {4, 8, 16, 32}` with identical other hyperparameters. Log final eval loss. Plot and include in `results/rank_ablation.png`. Expectation: diminishing returns past r=16 on small datasets.

## Write-up (`project/README.md`)

```markdown
# Heavy-Ion Abstract Summarizer (LoRA fine-tune of Llama-3.2-1B)

## Task
Given an arXiv heavy-ion / nuclear abstract, produce a 3-sentence plain-English summary.

## Data
612 (abstract, summary) pairs. Abstracts from arXiv hep-ex/nucl-ex/nucl-th 2020–2025. Summaries generated by Claude Sonnet 4.5 with a physicist-editor prompt. 80/10/10 split.

## Model
Llama-3.2-1B-Instruct + LoRA (r=16, α=32, all linear targets). 4.2M trainable params out of ~1.2B total.

## Training
AdamW, lr=2e-4, cosine + 3% warmup, batch 2 × grad-accum 8 = effective batch 16, 2 epochs, packing, gradient checkpointing, bf16.
Hardware: RTX 3060 12GB. Wall-clock: 38m.
Final eval loss: 1.47 (vs base reference 2.81).

## Evaluation
Blind A/B on 20 held-out abstracts: fine-tuned won 15 / 20 (75%); 3 ties; 2 losses.

## Deployment
vLLM server on localhost:8000. ~40 tok/s, ~120 ms first-token on RTX 3060.

## Model card
Pushed to https://huggingface.co/plewis3323/heavy-ion-abstract-summarizer-v1.

## Limits
- Tendency to echo teacher's 3-sentence structure; less flexibility than base.
- Occasional hallucinated numerical values — inherited from teacher.
- Loses some general-chat ability (catastrophic forgetting check: ~10% worse on haiku generation).
```

## Common failure modes

- Teacher summaries are inconsistent or wrong → your fine-tune inherits the noise. Manual quality spot-check before training.
- Loss drops fine but generations start producing the prompt verbatim → template mismatch between training and inference.
- OOM with 1B bf16 base → drop to QLoRA for 7B or reduce `max_seq_length` / `batch_size`.
- Eval loss flat → lr too low or target modules missing (e.g., only `q_proj`).
- Overfitting at 1 epoch → dataset too small; collect more, regularize, or reduce rank.

## Extensions (optional)

- Swap task: classify abstracts by technique (tracking / calorimetry / flow / tomography / …) instead of summarizing. Short completions, even easier to fine-tune.
- Mix 10% general-instruction data (e.g., Alpaca) to preserve generality. Report before/after on the Week 07 E13 generality checks.
- Train both 1B and 7B adapters; report cost/benefit.

## Sign-off

Commit, push, tick Week 07 boxes. Week 08: retrieval augmentation — your summarizer becomes a paper-aware research assistant that can cite its sources.
