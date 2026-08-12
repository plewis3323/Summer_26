# Week 30 — Exercises

Work top to bottom. Setup (imports, model loading, the Week 27 abstracts corpus, your
Week 29 `src/sampling.py`, seeds) is given by the notebook; you write only the lines
each exercise asks for. E5's dataset lives in a file, `sft_v0.jsonl`, validated by
`tests/test_sft_schema.py` — it seeds the Week 32 project; everything else is notebook
cells.

## E1 — Hand-rolled LoRA on one layer

Wrap one linear layer with your own BA bottleneck (lesson §7 pattern, then applied to
a real `nn.Linear` inside a small model): frozen W, random A, zero B, scale alpha/r.
Train briefly on a toy objective, then merge.
Hint: at the very first step only B has a nonzero gradient (A's gradient carries a
factor of B). Don't debug that as a bug.
Accept when: at init, wrapped output equals base output exactly (`torch.equal`); after
training, the merged matrix W + (alpha/r)BA reproduces adapter outputs within 1e-5.

## E2 — Parameter count: hand vs `peft`

For a ~1B model (default: Llama-3.2-1B or Qwen2.5-1.5B class), compute by hand the
trainable-parameter count for r = 8 adapters on q/k/v/o projections, using the actual
shapes from `model.config` (watch for grouped-query attention making k/v thin). Then
build the same `LoraConfig` and compare.
Hint: per matrix it is r × (d_in + d_out); read `num_hidden_layers`, `hidden_size`,
`num_key_value_heads` from the config.
Accept when: your hand count matches `print_trainable_parameters` exactly, and the
percentage of total is stated.

## E3 — Rank sweep: test the ansatz

LoRA-fine-tune the small model on a style task over your abstracts corpus (e.g.
title → abstract-opening-sentence) at r in {1, 4, 16, 64}, same data, same steps, same
effective LR. Plot final validation loss vs r.
Hint: hold everything but r fixed; alpha/r scaling in the lesson exists precisely so
one LR works across the sweep. Keep runs short (a few hundred steps) — the shape of
the curve is the deliverable.
Accept when: the val-loss-vs-r plot is saved with a one-line conclusion on whether the
low-rank ansatz holds for this task (saturating curve = yes).

## E4 — QLoRA memory measurement

Run the same short fine-tune twice: bf16 base + LoRA, and nf4 (4-bit) base + LoRA via
`BitsAndBytesConfig`. Record peak GPU memory for both.
Hint: `torch.cuda.reset_peak_memory_stats()` before, `torch.cuda.max_memory_allocated()`
after; if no local GPU, this is a Colab/Kaggle cell.
Accept when: both peaks are reported with their ratio, plus one line on why the ratio
is not the naive 4x (adapters, activations, and optimizer states don't shrink).

## E5 — SFT dataset v0 (`sft_v0.jsonl`)

Hand-label 30 (abstract → JSON metadata) examples for the Week 32 extractor: fields
collision system, sqrt(s_NN), observables, centrality, detector/experiment, physics
topic. Store as chat-format JSONL (user turn = instruction + abstract, assistant turn
= the JSON string). Missing information is the JSON value null — never guessed.
Hint: pick the 30 abstracts to be *varied* (different systems, energies, experiments,
and at least 5 theory abstracts where detector is null); diversity now saves relabeling
in Week 32.
Accept when: `tests/test_sft_schema.py` passes (every row validates against the fixed
JSON schema) and the file loads in `SFTTrainer` without error.

## Review

1. (Week 06) LoRA assumes ΔW is low-rank. What does the SVD of a trained ΔW tell you,
   and how would you measure its effective rank?
2. (Week 26) List the weight matrices in one transformer block. Which did you adapt in
   E2, and why attention projections first?
3. (Week 16) Fine-tuning at too high an LR destroys the pretrained model. Which
   Week 16 diagnostics would catch that within the first 100 steps?
4. (Week 09) The rank sweep in E3 is model selection. What data-split discipline does
   that require, and which split is now spent?
