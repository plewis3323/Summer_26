# Week 29 — The HuggingFace Ecosystem

You built the instrument; now learn the shared facility everyone actually runs — and
note that sampling from an LLM is drawing from a Boltzmann distribution whose
temperature knob you control directly.

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

- HuggingFace NLP course: the chapters on the `transformers` pipeline/models/tokenizers
  and on `datasets`.
- HF docs: *Generation strategies* (`generate`, `GenerationConfig`) and *Chat
  templates* (`apply_chat_template`).
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (nucleus sampling
  paper) — §§1, 3.
- Skim a model card end to end (e.g. Llama 3.2 1B): license, chat format, intended use.

## Derivations (paper first)

- Temperature: p_i(T) = softmax(z/T)_i ∝ e^{z_i/T}. Show the T → 0 and T → ∞ limits and
  compute d(entropy)/dT sign — this is literally a Boltzmann distribution over tokens.
- Top-k and top-p as truncate-and-renormalize operations on that distribution; show
  greedy = top-k(1) = T → 0, and why top-p adapts to the entropy of the context where
  top-k cannot.

## Exercises (built when the week starts)

1. Load a ~1B instruct model locally; generate from a fixed prompt at three dtypes or
   quantization settings. Accept when: peak memory and tokens/sec reported in a table.
2. Own sampler from raw logits (temperature + top-k + top-p, composable). Accept when:
   with the same seed and settings, output token ids match HF `generate` exactly.
3. Temperature sweep T ∈ {0.1, 0.7, 1.0, 1.5} on one physics prompt; measure mean
   token-level entropy. Accept when: entropy rises monotonically with T and samples
   are saved side by side.
4. Chat-template footgun: same question sent (a) raw, (b) via `apply_chat_template`,
   (c) with a wrong/missing system slot. Accept when: outputs differ and both footguns
   are named in one line each.
5. `datasets` pipeline: load your Week 27 abstracts as a `Dataset`, tokenize with
   `.map(batched=True)`, filter by length, stream. Accept when: pipeline runs and
   row counts at each stage are printed.
6. Perplexity of the base model on 100 held-out heavy-ion abstracts vs 100 generic
   news paragraphs. Accept when: both numbers reported with the sliding-window method
   stated.

## Deliverable

Derivation scans; `Week29_Exercises.ipynb` with checks PASS; a `sampling.py` you'll
reuse in Weeks 30–32; the perplexity table.

## Review

- Week 25: what is the softmax Jacobian, and where did the temperature-like 1/√d_k
  appear inside the transformer itself?
- Week 7: perplexity = exp(cross-entropy). Write cross-entropy H(p, q) and state its
  relation to KL divergence and entropy.
- Week 27: give two tokenizer pitfalls you demonstrated, and predict which one hurts
  the perplexity comparison on physics text.
- Week 3: you're pulling multi-GB checkpoints. Which Week 3 practices (env pinning,
  cache locations, disk hygiene) apply here?
