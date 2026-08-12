# Week 29 — The HuggingFace Ecosystem

You built the instrument in Month 07; now learn the shared facility everyone actually
runs — and note that sampling from an LLM is drawing from a Boltzmann distribution
whose temperature knob you control directly. The lesson derives that connection in
full.

## Objectives

- Load, run, and inspect open-weight models (Llama 3.2 / Qwen / Gemma / Phi class)
  locally with `transformers`, managing dtype, device, and memory deliberately.
- Derive temperature, top-k, and top-p sampling from the softmax and implement all
  three from raw logits.
- Use `datasets` for loading, mapping, filtering, and streaming without materializing
  everything in RAM.
- Apply chat templates correctly and demonstrate two concrete template footguns.
- Compute perplexity of a model on a corpus and say what it does and does not measure.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: Hub/loading/memory arithmetic, the
  full sampling derivations with worked numbers, `datasets`, chat templates,
  perplexity.
- HuggingFace LLM course: the chapters on the `transformers`
  pipeline/models/tokenizers and on `datasets`.
- HF docs: *Generation strategies* (`generate`, `GenerationConfig`) and *Chat
  templates* (`apply_chat_template`).
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (nucleus sampling
  paper) — §§1, 3.
- Skim a model card end to end (e.g. Llama 3.2 1B): license, chat format, intended use.

## Derivations (paper first; `lesson.md` §3 walks each through)

- Temperature: p_i(T) = softmax(z/T)_i ∝ e^{z_i/T}. Show the T → 0 and T → ∞ limits and
  compute d(entropy)/dT sign — this is literally a Boltzmann distribution over tokens.
- Top-k and top-p as truncate-and-renormalize operations on that distribution; show
  greedy = top-k(1) = T → 0, and why top-p adapts to the entropy of the context where
  top-k cannot.

## Exercises

See `exercises.md` (notebook built from it when the week starts). Six exercises:
memory-vs-dtype measurement, your own composable sampler verified token-for-token
against HF `generate`, a temperature/entropy sweep, chat-template footguns
demonstrated live, a `datasets` pipeline over the Week 27 corpus, and a
physics-vs-news perplexity comparison.

## Deliverable

Derivation scans; `Week29_Exercises.ipynb` with checks PASS; a `src/sampling.py`
you'll reuse in Weeks 30–32; the perplexity table.

## Review

- Week 25: what is the softmax Jacobian, and where did the temperature-like 1/√d_k
  appear inside the transformer itself?
- Week 8: perplexity = exp(cross-entropy). Write cross-entropy H(p, q) and state its
  relation to KL divergence and entropy.
- Week 27: give two tokenizer pitfalls you demonstrated, and predict which one hurts
  the perplexity comparison on physics text.
- Week 3: you're pulling multi-GB checkpoints. Which Week 3 practices (env pinning,
  cache locations, disk hygiene) apply here?
