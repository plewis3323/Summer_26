# Week 26 — The Full Transformer

A decoder-only transformer is a residual stream that attention and MLP blocks read from
and write to — closer to a perturbative series of corrections on an identity map than to
a stack of opaque layers.

## Objectives

- Assemble a complete GPT-style decoder from your Week 25 attention: embeddings,
  blocks, LayerNorm, head — and verify it against real GPT-2 weights.
- Explain sinusoidal, learned, and rotary (RoPE) position encodings and what problem
  each solves.
- Argue for pre-norm over post-norm from gradient flow through the residual stream.
- Compute a transformer's parameter count from its config, by hand.
- Read *Attention Is All You Need* and the GPT-2 paper critically: what was essential,
  what was incidental, what did not survive.

## Core material (~3 hrs)

- Karpathy, *Let's build GPT* — second half (blocks, LayerNorm, residuals, full model).
- Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762) — full read; note it is
  encoder–decoder, unlike what you are building.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) —
  §§1–3 and the model description.
- Prince, *UDL* Ch. 12, remaining sections (position encodings, full architecture).
- RoPE concept level only: Su et al., *RoFormer* (arXiv 2104.09864) — the idea is a
  phase rotation e^{imθ} per 2-D subspace so q·k depends only on relative position; a
  physicist can read §3.2 and skip the rest.

## Derivations (paper first)

- Residual gradient flow: for x_{l+1} = x_l + F(x_l), expand ∂L/∂x_0 and show the
  identity path keeps gradients alive at any depth; note where LayerNorm sits in
  pre-norm vs post-norm and which one leaves that path clean.
- Sinusoidal encodings: show PE(pos+k) is a fixed linear transform (a rotation) of
  PE(pos) — the property that makes relative offsets learnable.
- Parameter count of GPT-2 small (12 layers, d=768, h=12, vocab 50257, ctx 1024) with
  and without weight tying; reconcile with the quoted 124M.

## Exercises (built when the week starts)

1. Full GPT module (config-driven: n_layer, n_head, d_model, vocab, ctx). Accept when:
   forward pass on a random batch returns logits of shape (B, T, vocab) and loss is
   ≈ ln(vocab) at init.
2. Parameter-count check. Accept when: hand-derived count matches
   `sum(p.numel())` exactly for GPT-2-small config.
3. Load HuggingFace GPT-2 weights into your model. Accept when: your logits match HF's
   within 1e-4 on a fixed prompt.
4. Sinusoidal shift property, numerically. Accept when: a single learned/fitted matrix
   maps PE(pos) → PE(pos+k) with max error < 1e-6 for several k.
5. Pre-norm vs post-norm: train two 4-layer models on tiny Shakespeare for a fixed
   budget. Accept when: loss curves plotted and gradient norms per layer compared, with
   a one-line conclusion.
6. Weight tying ablation on the same setup. Accept when: parameter counts and final
   losses reported for tied vs untied.

## Deliverable

Derivation scans; `Week26_Exercises.ipynb` with checks PASS; a `model.py`-style GPT that
loads GPT-2 weights and matches its logits — this exact model trains in Week 27.

## Review

- Week 25: rewrite scaled dot-product attention from memory, with the √d_k argument.
- Week 17: ResNet skip connections were "gradient highways." State the identical claim
  for the transformer residual stream in one equation.
- Week 16: why is training loss ≈ ln(vocab) the right init check? Which Week 16 lesson
  is that?
- Week 6: weight tying reuses the embedding as the output projection. What do the SVD
  ranks of W_E say about how much of the 768-dim space embeddings actually use?
