# Week 07 — HuggingFace Ecosystem + Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

## Where this sits

You built a transformer from scratch last week. That's the pedagogical foundation. But nobody trains frontier LLMs from scratch in practice — not labs, not startups, not you. The real-world LLM workflow is: **take a pre-trained model, fine-tune it on your task, deploy.** This week you learn that workflow end-to-end.

You'll pick a small-to-medium open-weights model (Llama 3.2 1B / 3B, Qwen 2.5 1.5B, or Phi-3-mini), load it with HuggingFace `transformers`, fine-tune it with `peft` (Parameter-Efficient Fine-Tuning) using LoRA on a physics-flavored dataset, and deploy it locally. You'll touch QLoRA briefly — loading a 4-bit quantized base model so that you can fine-tune a 7B on a single 12GB GPU.

By the end you'll have a model that summarizes arXiv heavy-ion abstracts — or answers physics questions, or classifies paper sections — in a way that feels meaningfully better than the base model at that specific task. More importantly, you'll understand why LoRA works and when to reach for it.

## Learning objectives

By the end of the week you can:

1. Load a pre-trained causal LM from HuggingFace, run inference, and interpret its outputs.
2. Explain LoRA: why it works, how it reduces trainable params, what rank to pick, where to apply it.
3. Explain QLoRA: what 4-bit quantization does, why double-quantization, when it costs you accuracy.
4. Fine-tune a 1B–3B parameter model on a custom instruction dataset using `trl`'s `SFTTrainer`.
5. Evaluate the fine-tuned model: eval loss, sample outputs, basic bench comparisons (side-by-side with base).
6. Merge LoRA adapters back into the base, save, push to HuggingFace Hub (optional), serve with `transformers.pipeline` or vLLM.
7. Diagnose common failures: catastrophic forgetting, mode collapse, over-fitted templates, tokenizer mismatch.

## The HuggingFace ecosystem

Four libraries do 90% of the work:

### `transformers`
Model + tokenizer classes. ~200k people use it daily. `AutoModelForCausalLM`, `AutoTokenizer`, generation utilities. Your entry point.

### `datasets`
Apache Arrow-backed dataset library. Memory-mapped, streamable, fast. `load_dataset("HuggingFaceH4/ultrachat_200k")`.

### `peft`
Parameter-Efficient Fine-Tuning. LoRA, QLoRA, prefix tuning, P-tuning, adapter tuning. You want LoRA. Wraps your model, replaces linear layers with LoRA-injected equivalents, exposes only the adapter parameters as trainable.

### `trl` (Transformer Reinforcement Learning)
`SFTTrainer` for supervised fine-tuning (what you want); `DPOTrainer` for Direct Preference Optimization; `PPOTrainer` for RLHF. You'll use `SFTTrainer`.

Plus:
- `accelerate` — device/dtype management, distributed training.
- `bitsandbytes` — 4-bit and 8-bit quantization (for QLoRA).
- `evaluate` — metric library.

## LoRA: the key idea

Pre-trained models are parameter-dense; fine-tuning updates them all is expensive and wasteful. Hu et al. (2021) observed that the weight *update* during fine-tuning has low intrinsic rank — you can approximate `ΔW` as `BA` where `B` and `A` are low-rank matrices.

For a weight `W ∈ R^{d_out × d_in}`:
```
W_ft = W + ΔW ≈ W + BA
```
where `B ∈ R^{d_out × r}`, `A ∈ R^{r × d_in}`, and `r << min(d_out, d_in)`.

At training time, `W` is frozen; only `B` and `A` are trained. A is initialized Gaussian, B is initialized zero (so `BA = 0` at start — the model is unchanged at step 0).

Parameter count reduction: for a 768×768 matrix (d=768), full fine-tuning trains 589824 params. LoRA with r=8 trains 768×8 + 8×768 = 12288 params. 48× reduction. And you do this for every linear layer you target.

Typical targets: `q_proj`, `k_proj`, `v_proj`, `o_proj` (all four attention projections) plus `gate_proj`, `up_proj`, `down_proj` (all three FFN projections in SwiGLU). This is what "target modules = all-linear" means in peft.

Typical rank: r = 8, 16, or 32. Higher rank = more capacity = more parameters to train but also more risk of overfitting on small datasets.

### LoRA α

There's a scaling factor `α/r` applied to `BA`. `α` is typically set equal to r (so the scale is 1) or 2r. Tune last; rank is the important knob.

### LoRA vs. full fine-tuning

| Aspect | Full FT | LoRA |
|---|---|---|
| Trainable params | 100% | 0.1–1% |
| GPU memory (bf16) | ~20 GB / 1B params | ~4 GB / 1B params |
| Training time | 1× | ~0.5–0.7× (optimizer state smaller) |
| Peak performance | Often slightly higher | 95–99% of full FT |
| Artifacts | Full weights (GB) | Adapter weights (MB) |
| Composability | None | Can mix/swap adapters |

Full FT is only worth it if you have lots of compute and LoRA isn't closing the gap. For your setup, LoRA is correct.

## QLoRA: the 4-bit twist

Dettmers et al. 2023. Load the base model in 4-bit (NF4 format — "normalized float 4-bit", optimized for normally-distributed weights). Train LoRA adapters in bf16 on top. Dequantize on the fly during matmul.

Memory saving: ~4× vs bf16 base. This is how you fit a 7B model on a 12GB GPU.

Cost: small (~1%) accuracy hit in most cases. For your purposes, invisible.

When to reach for QLoRA: whenever you can't fit bf16 base + LoRA adapters + optimizer states + gradients + activations in GPU memory.

## Data for fine-tuning

LoRA fine-tuning needs aligned pairs of (prompt, completion) or (instruction, input, output). Format depends on the model's chat template.

### Prompt templates

Every instruct-tuned LLM has a specific chat template. Llama-3 uses:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{completion}<|eot_id|>
```

Qwen 2.5 uses:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{completion}<|im_end|>
```

Use `tokenizer.apply_chat_template(messages, tokenize=False)` to get the right format automatically. Never hand-roll templates; you'll get it wrong.

### Loss masking

During SFT, you generally want loss only on the completion tokens, not the prompt. `SFTTrainer` with `DataCollatorForCompletionOnlyLM` handles this.

### Dataset sizes

LoRA works on surprisingly small datasets. Rules of thumb:
- <100 examples: prompt engineering, don't fine-tune
- 100–1000: LoRA with low rank (r=4–8), high lr, regularize heavily
- 1000–10000: LoRA sweet spot
- 10000+: LoRA still works; full FT becomes plausible

## Your fine-tune task: arXiv abstract → plain-English TL;DR

Given a heavy-ion physics abstract, produce a one-paragraph summary phrased for a physics-literate but non-specialist reader. This is something you'd actually use at work — the abstracts you skim daily often have jargon-dense openings that a summarizer can compress.

Dataset: 500–1000 (abstract, summary) pairs. For the summaries, we'll use a clever shortcut — have a frontier model (Claude or GPT-4) generate gold summaries from the abstracts, then fine-tune a small open model to match. This is "teacher-student distillation", the workflow behind most open-source instruct models.

See `project.md` for the full pipeline.

## Training details

### Optimizer and schedule

`paged_adamw_32bit` from bitsandbytes. Memory-efficient paged optimizer for QLoRA.

LR: 2e-4 for LoRA is the typical default. Higher than full FT (1e-5 to 5e-5) because only a few params train, and you want those few to move meaningfully.

Warmup: short, 50–100 steps. You're not pretraining; no need for 2000-step warmup.

Epochs: 1–3. Anything more and you start overfitting on small datasets.

Batch size: 1–4 depending on GPU. Use gradient accumulation to get effective batch size 16–64.

### Gradient checkpointing

Free memory by recomputing activations during backward. Set `gradient_checkpointing=True` in `TrainingArguments`. ~30% slower, frees tons of memory. Use it.

### Packing

`SFTTrainer(packing=True)` packs multiple short examples into single sequences up to `max_seq_length`. 2–3× throughput on short examples.

## Evaluation

Two kinds:

### Quantitative
- Eval loss on held-out split (10–20%).
- Perplexity. Lower is better.
- If you have reference summaries: ROUGE-L, BLEU, BERTScore. All flawed; all you'll compute.

### Qualitative
Side-by-side. Pick 20 abstracts. Generate base-model summary and fine-tuned summary side by side. Score each pair: better / same / worse. Report ratio.

Qualitative is more informative. Quantitative metrics on small eval sets are noisy.

## Deployment paths

### Fastest
`transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)`. Slowest, but one line.

### HuggingFace Hub
Push adapter to `huggingface.co/plewis3323/heavy-ion-summarizer`. Others can load with `PeftModel.from_pretrained`. Great for sharing.

### vLLM
`pip install vllm`. Near state-of-the-art inference throughput. `vllm serve meta-llama/Llama-3.2-3B-Instruct --enable-lora --lora-modules heavy-ion=./adapter`. Production-grade.

### llama.cpp
Merge adapter into base (`peft_model.merge_and_unload()`), convert to GGUF format (`llama.cpp/convert.py`), run on CPU or Mac. Slow but runs anywhere.

### Ollama
Similar to llama.cpp; wraps it with a nice CLI and model registry. Good for local experimentation.

## Pitfalls and debugging

**Catastrophic forgetting.** Fine-tune a chat model on summarization, then it loses general conversation ability. Usually fine if your downstream task is scoped. If you need both, mix a small fraction (5–10%) of general chat data back in.

**Tokenizer pad-token surprise.** Llama models have no pad token by default. You set `tokenizer.pad_token = tokenizer.eos_token` and then your loss computation accidentally trains the model to always output EOS. Fix: use a DataCollator that masks pad tokens in the loss computation, and pad on the right.

**Loss decreases but generations are broken.** Usually tokenizer chat-template mismatch between training and inference. Always use `apply_chat_template` both at training and inference.

**Overfitting after 1 epoch.** Dataset too small. Either collect more, regularize harder (dropout, weight decay, lower rank), or train fewer steps.

**GPU OOM at step 0.** Dropped to QLoRA → still OOM. Reduce `max_seq_length`, drop `batch_size` to 1, enable `gradient_checkpointing`, enable `gradient_accumulation_steps=8`. If that fails, shrink model.

**`bitsandbytes` installation hell.** It's CUDA-version-sensitive. If `pip install bitsandbytes` fails with CUDA errors, check `nvidia-smi` CUDA version matches your torch build. Use the `bnb-cuda-version` env var if needed.

**LoRA adapters don't load.** `PeftModel.from_pretrained(base_model, adapter_path)` requires `base_model` to be the exact base the adapter was trained on. Use `adapter_config.json` to verify base model name.

## HEP angle

A small, locally-runnable physics summarizer means:

- You can process hundreds of new arXiv posts per day into a digestible feed without paying API fees or leaking data.
- The pipeline generalizes: swap the task → "extract experimental signature", "classify technique", "identify authors' institution from abstract" — all one fine-tune away.
- In Weeks 9–11 you'll turn this into a tool an agent can call. Fine-tuning is step one; tool-use is where the real workflow lifts off.

## Tooling introduced

- `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `evaluate`.
- `huggingface_hub` CLI for pushing/pulling models.
- `vllm` for serving.
- `ollama` for local deployment (optional).

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | Install HF stack. Load Llama-3.2-1B. Run inference. Read LoRA paper. |
| 2 | Understand peft's `get_peft_model`. Do a toy LoRA on a tiny dataset. |
| 3 | Build the abstract→summary dataset (teacher-distillation with Claude/GPT). |
| 4 | Train the LoRA. Watch wandb. |
| 5 | Evaluate. Side-by-side with base. |
| 6 | Try QLoRA with a 7B base; compare. |
| 7 | Deploy with vLLM; write up. |

## What "done" looks like

- A trained LoRA adapter on a 1B–3B model.
- Eval loss curve logged in wandb; final loss < base-model loss on the same eval set.
- A side-by-side table (20 abstracts) showing the FT model wins in ≥ 60% of qualitative comparisons.
- A running vLLM server + cURL snippet that prompts your model and returns a summary in <2 seconds.
- A 500-word reflection: what worked, what surprised you, what you'd try next.

Week 08: RAG. Your summarizer becomes a retriever over papers, not just a one-shot model. That's where things get useful.
