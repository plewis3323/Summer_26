# Week 30 — Fine-Tuning: Full FT, LoRA, QLoRA

Fine-tuning perturbs a pretrained model toward your task; LoRA is the statement that the
useful perturbation is low-rank — an ansatz you can test, not just cite.

## Objectives

- Explain when full fine-tuning, LoRA, and QLoRA are each the right tool (memory math
  included).
- Derive the LoRA update and its parameter count; verify both in code.
- Explain the QLoRA idea: 4-bit NF4 base weights (quantiles of a Gaussian), frozen,
  with LoRA adapters trained in higher precision on top.
- Run LoRA fine-tunes with HF `peft`, merge adapters, and compare against the base.
- Design an SFT dataset deliberately: format, diversity, size, contamination checks.

## Core material (~3 hrs)

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv 2106.09685) —
  §§1–4, 7.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (arXiv 2305.14314) —
  the NF4 and paged-optimizer ideas; skim the evals.
- HF `peft` docs: LoRA quickstart, `LoraConfig`, merging adapters.
- HF TRL docs: `SFTTrainer` and dataset-format section (chat vs completion format).

## Derivations (paper first)

- LoRA: W' = W + ΔW with ΔW = (α/r) BA, B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}. Show the
  forward pass needs one extra low-rank bottleneck, why B = 0 at init makes the model
  exactly the base model at step 0, and that trainable params = r(d_in + d_out) vs
  d_in·d_out — compute the ratio for a 2048×2048 projection at r = 8.
- Count total trainable parameters for r = 8 adapters on all attention projections of a
  1B-class model; express as a percentage.
- NF4 sketch: why equal-probability-mass quantile bins of N(0, σ²) beat uniform bins for
  weight distributions that are approximately Gaussian (one paragraph + one sketch).

## Exercises (built when the week starts)

1. Hand-rolled LoRA: wrap one `nn.Linear` of a small transformer with your own BA
   bottleneck. Accept when: at init, wrapped output equals base output exactly; after
   training, merging W + (α/r)BA reproduces adapter outputs within 1e-5.
2. Parameter-count check with `peft` on a 1B model, r = 8, attention-only targets.
   Accept when: `print_trainable_parameters` matches your hand count exactly.
3. Rank sweep r ∈ {1, 4, 16, 64} on a small task (e.g. style-tune on your abstracts).
   Accept when: val loss vs r plotted with a one-line conclusion on the low-rank ansatz.
4. QLoRA memory measurement: same fine-tune fp16 vs 4-bit base. Accept when: peak GPU
   memory for both reported, with the ratio.
5. SFT dataset v0 for the Week 32 extractor: 30 hand-labeled (abstract → JSON metadata)
   examples in chat format. Accept when: file validates against a fixed JSON schema and
   loads in `SFTTrainer` without error.

## Deliverable

Derivation scans; `Week30_Exercises.ipynb` with checks PASS; merged-adapter checkpoint
from the rank sweep; `sft_v0.jsonl` — the seed of the Week 32 project dataset.

## Review

- Week 6: LoRA assumes ΔW is low-rank. What does the SVD of a trained ΔW tell you, and
  how would you measure its effective rank?
- Week 26: which weight matrices exist in one transformer block, and which did you
  adapt? Why attention projections first?
- Week 16: fine-tuning at too high an LR destroys the pretrained model. Which Week 16
  diagnostics would catch that in the first 100 steps?
- Week 9: your rank sweep is model selection. What data split discipline from Week 9
  does it require?
