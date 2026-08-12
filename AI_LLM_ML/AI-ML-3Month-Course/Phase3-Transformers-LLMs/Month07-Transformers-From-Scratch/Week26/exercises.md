# Week 26 — Exercises

Work top to bottom. Setup (imports, seeds, tiny-Shakespeare download, HuggingFace model
fetch, plotting scaffolds) is given by the notebook; you write only the lines each
exercise asks for. E1's model goes in a file, `model.py` (imported by Week 27's
training run and by E3–E6); the rest lives in the notebook.

## E1 — Config-driven GPT

In `model.py`, implement `GPT(n_layer, n_head, d_model, vocab_size, ctx)` from the
lesson: tied embeddings, pre-norm blocks using your Week 25 `MultiHeadAttention`,
final LayerNorm, no-bias head.
Hint: `self.head.weight = self.tok_emb.weight` is the whole tying mechanism.
Accept when: forward pass on a random `(8, 32)` batch returns logits of shape
`(8, 32, vocab)` and cross-entropy at init is within 0.2 of `ln(vocab)`.

## E2 — Parameter count, exact

Write out the hand count for GPT-2 small (12, 768, 12, 50257, 1024) using the lesson's
per-block formula, then instantiate your E1 model with that config and compare.
Hint: `12*d*d + 13*d` per block; don't count the tied head twice.
Accept when: hand count equals `sum(p.numel() for p in model.parameters())` exactly
(124,439,808), and a one-liner states which term a mismatch of 9,984 would indicate.

## E3 — Load real GPT-2 weights

Map HuggingFace `gpt2` weights into your model (the notebook gives the state-dict
name pairs; you write the copy loop, transposing the Conv1D-style weights).
Hint: HF stores QKV and MLP weights as `(n_in, n_out)`; `nn.Linear` wants `(n_out, n_in)`.
Accept when: on the fixed prompt in the notebook, your logits match HF's within 1e-4
(max absolute difference), and greedy top-1 next tokens agree for 20 steps.

## E4 — Sinusoidal shift property, numerically

Build the sinusoidal table PE for d = 64, positions 0–511. For k in {1, 5, 50}, solve
the least-squares problem PE(p) M_k ≈ PE(p+k) (Week 09) and report the max error.
Hint: `np.linalg.lstsq` on rows 0..511-k vs rows k..511.
Accept when: max |PE(p)M_k − PE(p+k)| < 1e-6 for all three k, and one sentence names
the trig identity responsible.

## E5 — Pre-norm vs post-norm

The notebook gives a training loop and tiny Shakespeare. You provide a post-norm
variant of your Block and train both 4-layer models for the same fixed iteration
budget, logging loss and per-block gradient norms.
Hint: only `forward` changes: `x = ln(x + f(x))` vs `x = x + f(ln(x))`.
Accept when: both loss curves and a per-layer gradient-norm plot are saved, plus a
one-line conclusion naming which variant keeps early-block gradients healthy.

## E6 — Weight-tying ablation

Same setup as E5 (pre-norm): train tied vs untied for the same budget; report parameter
counts and final train/val losses in a two-row table.
Hint: untying is deleting one line of E1.
Accept when: the table is printed, the parameter difference equals vocab × d_model
exactly, and one sentence interprets the loss difference (or its absence) honestly.

## Review

1. (Week 25) Rewrite scaled dot-product attention from memory, including the two-part
   √d_k argument.
2. (Week 17) ResNet skips were "gradient highways". State the identical claim for the
   residual stream in one equation.
3. (Week 16) Why is loss ≈ ln(vocab) the right init check? What number is that for
   vocab 50257?
4. (Week 06) Weight tying reuses W_E as the output head. What would the singular values
   of W_E (Week 07) tell you about how much of the 768-dim space the embeddings
   actually use?
