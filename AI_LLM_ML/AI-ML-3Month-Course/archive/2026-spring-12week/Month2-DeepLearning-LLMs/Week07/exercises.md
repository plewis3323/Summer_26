# Week 07 — Exercises

## E1 — Load a pre-trained model and chat with it (easy, 20 min)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"  # or "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

msgs = [
    {"role": "system", "content": "You are a concise physics assistant."},
    {"role": "user",   "content": "In one sentence, what is elliptic flow?"},
]
input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(input_ids, max_new_tokens=128, temperature=0.7, do_sample=True)
print(tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
```

**Accept when:** you get a coherent response. Note the `apply_chat_template` call — understand what it generates by also running with `tokenize=False` and printing.

## E2 — Inspect tokenizer and chat template (easy, 20 min)

1. Print `tok.chat_template` — a Jinja template.
2. Tokenize "QGP is quark-gluon plasma" three different ways and count tokens.
3. Verify `tok.pad_token`. If `None`, print the warning you'd normally see. Set it to `tok.eos_token` and note the gotcha.

## E3 — Wrap with peft and count trainable params (medium, 40 min)

```python
from peft import LoraConfig, get_peft_model, TaskType

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
peft_model = get_peft_model(model, lora)
peft_model.print_trainable_parameters()
```

**Accept when:** the reported trainable % is ~1% for a 1B model with r=16. Understand the arithmetic: count all linear layers in the model, multiply by 2r and the relevant dimensions.

## E4 — Toy overfit with LoRA (medium, 45 min)

Same as Week 06 E4 but for LoRA. Create a 4-example "dataset" — e.g. "If the user asks X, always respond with the literal string 'I love jets'". Format as chat pairs. Fine-tune with SFTTrainer for 200 steps. Generate. Verify the model produces exactly the canned answer.

Point: proves your fine-tune pipeline is functional. If this doesn't work, a bigger one won't either.

## E5 — Dataset: build a physics summarization dataset (medium, 90 min)

1. Take 200 abstracts from your Week 06 scrape.
2. For each, use Anthropic's API (`claude-sonnet-4-5` or similar) with a careful prompt to generate a ~3-sentence plain-English summary for a physicist-but-not-specialist.
3. Save as `sft_data.jsonl` with fields: `{"abstract": ..., "summary": ...}`.

Prompt template for the teacher model:

```
You are a physicist-editor. Given the arXiv abstract below, write a 3-sentence summary that:
- Preserves the key physical claim and any numerical result (energies, efficiencies, cross-sections).
- Uses plain language; no jargon unless essential.
- Omits detector acronyms unless central.

Abstract: {abstract}

Summary:
```

Budget: 200 Anthropic API calls at roughly $0.01 each. You have a budget for this course — check `02-Setup-Guide.md`.

**Accept when:** 200 summaries produced; spot-check 10 for quality; split 80/10/10 train/val/test.

## E6 — SFT with LoRA (medium → hard, 2–4 hours of wall-clock)

Use `trl.SFTTrainer`:

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "train.jsonl", "eval": "val.jsonl"})

def format_example(ex):
    msgs = [
        {"role": "user",      "content": f"Summarize this physics abstract:\n\n{ex['abstract']}"},
        {"role": "assistant", "content": ex["summary"]},
    ]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

ds = ds.map(format_example)

cfg = SFTConfig(
    output_dir="out/hi-sumz-r16",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    report_to="wandb",
    packing=True,
    max_seq_length=1024,
)

trainer = SFTTrainer(
    model=peft_model,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    processing_class=tok,
)
trainer.train()
trainer.save_model("out/hi-sumz-r16/final")
```

**Accept when:** eval loss drops monotonically (some jitter OK); save the final adapter.

## E7 — Inference with the fine-tuned adapter (easy, 30 min)

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
tuned = PeftModel.from_pretrained(base, "out/hi-sumz-r16/final")

msgs = [{"role": "user", "content": f"Summarize this physics abstract:\n\n{abstract}"}]
input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(base.device)
out = tuned.generate(input_ids, max_new_tokens=200, temperature=0.3, do_sample=True)
```

Generate summaries for 5 abstracts. Compare side-by-side with base model. Report qualitative observations.

## E8 — QLoRA with a 7B base (hard, 90 min)

Upgrade to `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`. Load in 4-bit:

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
```

Train the same LoRA as E6. Compare eval loss and qualitative summary quality vs the 1B fine-tune.

**Accept when:** you get end-to-end QLoRA training on consumer GPU (<= 12GB works for 7B NF4 + LoRA).

## E9 — Side-by-side evaluation (medium, 60 min)

Build a small HTML table of 20 (abstract, base-summary, fine-tuned-summary) rows. Score each pair: better / same / worse (A-B blind if possible — randomize which side is which; reveal only after scoring).

Report win rate. If your fine-tuned model doesn't win in at least 60% of pairs, either your dataset is too small, your teacher summaries are inconsistent, or your model is undertrained.

## E10 — Merge and deploy (medium, 45 min)

```python
merged = tuned.merge_and_unload()
merged.save_pretrained("out/hi-sumz-merged")
tok.save_pretrained("out/hi-sumz-merged")
```

Then:
```bash
vllm serve out/hi-sumz-merged --host 0.0.0.0 --port 8000
```

(or whatever your GPU permits)

```python
# client
import requests
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "out/hi-sumz-merged",
    "messages": [{"role": "user", "content": "Summarize: ..."}],
    "temperature": 0.3,
})
print(r.json()["choices"][0]["message"]["content"])
```

Measure latency. Report tokens/second. Compare to raw `transformers` generation.

## E11 — Push adapter to HuggingFace Hub (easy, 15 min)

```python
from huggingface_hub import notebook_login; notebook_login()
tuned.push_to_hub("plewis3323/heavy-ion-abstract-summarizer-v1")
tok.push_to_hub("plewis3323/heavy-ion-abstract-summarizer-v1")
```

Edit the auto-generated model card to describe training data, purpose, limits. This is how you share LoRA adapters with colleagues.

## E12 — Ablation: rank sweep (medium, 120 min)

Re-train with r = 4, 8, 16, 32. Plot eval loss vs rank. Does higher rank help? Does it overfit on your small dataset?

Write 150 words interpreting the trend. In practice rank plateaus quickly — you'll probably see the sweet spot at r=8 or r=16.

## E13 — Catastrophic forgetting check (easy, 30 min)

Prompt the fine-tuned model with non-summarization tasks:
- "Write a haiku about photons."
- "What is 12 × 23?"
- "Translate to French: 'The jet is hot.'"

Compare to base model responses. Note what's lost.

Write 100 words. This is pedagogically important: fine-tuning isn't free; it costs generality unless you mix in diverse training data.

---

## Solution hints

- E1 — if the model is slow or OOM, you forgot `device_map="auto"` or dtype=bfloat16.
- E2 — Qwen's pad_token is "<|endoftext|>" and it's set; Llama-3's is None.
- E3 — if trainable % is ~100%, your peft wrap didn't take — check `get_peft_model` returned the wrapped model.
- E4 — if overfit on 4 examples doesn't work, your data formatting is wrong. Print one example from `ds["train"][0]["text"]` to see the actual string being fed in.
- E5 — watch for abstracts with LaTeX that the teacher model produces in unparseable form. Light post-processing helps.
- E6 — if eval loss is NaN, you have pad-token loss contamination; use `DataCollatorForCompletionOnlyLM` or the SFTTrainer's completion-only mode.
- E7 — `tuned.generate` needs to be on `tuned` not `base` even though the underlying model is shared. Check.
- E8 — if 4-bit base + LoRA still OOMs, your `max_seq_length` is too high.
- E9 — the A-B test is subtle; bias toward the fine-tuned version because you know which is which. Randomize.
- E10 — merged models lose the adapter's separation — you can't easily un-merge. Keep both merged and adapter-only checkpoints.
- E12 — rank 4 is usually enough for small datasets. Rank 64 is wasteful.
- E13 — forgetting is dataset-distribution-sensitive. 500 summaries don't wipe language understanding, but they might shift formality.
