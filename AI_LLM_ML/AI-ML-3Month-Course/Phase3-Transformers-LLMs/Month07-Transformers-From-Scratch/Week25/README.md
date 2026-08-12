# Week 25 — Attention Derived

Attention is a differentiable content-based lookup: a softmax over key–query overlaps is
a Boltzmann distribution over which past tokens to read, with 1/√d_k playing the role of
temperature.

## Objectives

- Write scaled dot-product attention from the definition, in NumPy, without references.
- Derive why the √d_k scaling is there, from the variance of a dot product of random vectors.
- Explain queries, keys, and values as three learned projections with distinct jobs.
- Implement multi-head attention and causal masking, and verify no future leakage.
- Read an attention heatmap and say what a head is doing.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: attention built from dictionary lookup
  up, with the full 1/√d_k derivation (variance argument + softmax saturation), causal
  masking, and multi-head. This is the course's third flagship derivation.
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* — first half, through
  the self-attention block (the rest is Week 26).
- 3Blue1Brown, deep learning series: *But what is a GPT?* and *Attention in
  transformers, visually explained*.
- Prince, *Understanding Deep Learning*, Ch. 12 (Transformers), the self-attention
  sections.
- Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762) — §3.2 only this week.

## Derivations (paper first)

- Attention output as an expectation: out = Σᵢ softmax(q·kᵢ/√d_k) vᵢ; write the general form.
- The √d_k: for q, k with i.i.d. zero-mean unit-variance components, show Var(q·k) = d_k,
  so scaling by 1/√d_k keeps pre-softmax scores O(1) for any head size.
- Softmax Jacobian ∂softmaxᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ), and what saturation does to gradients
  when scores are large (connect to the previous item).
- Multi-head parameter count for d_model, h heads, d_k = d_model/h, including W_O.

## Exercises

See `exercises.md` (notebook generated from it when the week starts). Six exercises,
E1–E6: NumPy attention verified against PyTorch, the variance and saturation experiments
behind the √d_k, a no-leakage proof of the causal mask, a `MultiHeadAttention` module
that Week 26 imports, and an attention-heatmap read.

## Deliverable

Scanned derivations in this folder; `Week25_Exercises.ipynb` complete with all checks
PASS; a working `MultiHeadAttention` module ready to be imported by Week 26.

## Review

- Derive the gradient of softmax + cross-entropy from Week 13. Why is it so clean?
- Week 19: why does an LSTM struggle with long-range credit assignment, and which part
  of attention removes that problem?
- Week 7: write the KL divergence and state Gibbs' inequality. Where did cross-entropy
  loss come from?
- Week 5: attention applies the same W_Q to every token. What property of the map does
  that give you, and why did it matter for GNNs in Week 21?
