# Week 30 — Fine-Tuning: Full FT, LoRA, QLoRA

Fine-tuning perturbs a pretrained model toward your task; LoRA is the statement that the
useful perturbation is low-rank — an ansatz you can test, not just cite. The lesson
builds quantization from "what is a floating-point number" up, so nf4 is arithmetic,
not incantation.

## Objectives

- Explain when full fine-tuning, LoRA, and QLoRA are each the right tool (memory math
  included).
- Derive the LoRA update and its parameter count; verify both in code.
- Explain floating-point formats (fp32/fp16/bf16) and quantization (int8 absmax, NF4
  as Gaussian quantile bins) from the bits up.
- Run LoRA fine-tunes with HF `peft`, merge adapters, and compare against the base.
- Design an SFT dataset deliberately: format, loss masking, diversity, size,
  contamination checks.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: the memory bill, floating point from
  scratch, int8/nf4 worked examples, the full LoRA derivation with parameter
  arithmetic, QLoRA assembled, SFT dataset design.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv 2106.09685) —
  §§1–4, 7.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (arXiv 2305.14314) —
  the NF4 and paged-optimizer ideas; skim the evals.
- HF `peft` docs: LoRA quickstart, `LoraConfig`, merging adapters.
- HF TRL docs: `SFTTrainer` and dataset-format section (chat vs completion format).

## Derivations (paper first; `lesson.md` §§3–4 walk each through)

- LoRA: W' = W + ΔW with ΔW = (α/r) BA, B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}. Show the
  forward pass needs one extra low-rank bottleneck, why B = 0 at init makes the model
  exactly the base model at step 0, and that trainable params = r(d_in + d_out) vs
  d_in·d_out — compute the ratio for a 2048×2048 projection at r = 8.
- Count total trainable parameters for r = 8 adapters on all attention projections of a
  1B-class model; express as a percentage (mind grouped-query attention shapes).
- NF4 sketch: why equal-probability-mass quantile bins of N(0, σ²) beat uniform bins for
  weight distributions that are approximately Gaussian (one paragraph + one sketch).

## Exercises

See `exercises.md` (notebook built from it when the week starts). Five exercises:
hand-rolled LoRA with an exact-at-init and merge check, the parameter count verified
against `peft` to the digit, a rank sweep that tests the low-rank ansatz, a QLoRA
peak-memory measurement, and `sft_v0.jsonl` — 30 hand-labeled extractor examples that
seed the Week 32 project.

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
