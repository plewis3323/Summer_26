# Week 25 — Exercises

Work top to bottom. Setup (imports, seeds, random test tensors, plotting scaffolds) is
given by the notebook; you write only the lines each exercise asks for.
E5's module also goes in a file, `attention.py`, so Week 26 can import it; everything
else lives in the notebook.

## E1 — Scaled dot-product attention in NumPy

Write `attention(Q, K, V, causal)` returning the output and the weight matrix, for
inputs of shape `(B, T, d)` (B = batch). Softmax over the last axis, scale by
`1/sqrt(d_k)`, optional causal mask before the softmax.
Hint: subtract the row max inside softmax; build the mask with `np.triu(..., k=1)`.
Accept when: matches `torch.nn.functional.scaled_dot_product_attention` within 1e-5 on
random inputs, with and without `is_causal=True`.

## E2 — Variance of a dot product

For each `d_k` in {4, 16, 64, 256, 1024}, sample 10 000 pairs of standard-normal
vectors and measure the variance of `q @ k`, unscaled and scaled by `1/sqrt(d_k)`.
Plot both against `d_k` on log-log axes.
Hint: one `rng.normal(size=(10000, d_k))` per side; variance of the product row-sums.
Accept when: log-log slope of the unscaled curve is 1.0 ± 0.05, and the scaled variance
is within 5% of 1 at every `d_k`.

## E3 — Softmax saturation kills gradients

Scores `s * z` for a fixed random `z` (`d_k = 256`) and scale factors
`s ∈ {0.1, 1, sqrt(d_k)/4, sqrt(d_k)}`: push a fixed upstream gradient through softmax
(use `torch` autograd) and record the gradient norm at the scores. Plot norm vs `s`.
Hint: `scores.requires_grad_(True)`; call `.backward()` on `(softmax(scores) * g).sum()`.
Accept when: the plot shows gradient norm collapsing by ≥ 100× from `s = 1` to
`s = sqrt(d_k)`, and one sentence connects this to the 1/√d_k factor.

## E4 — Causal mask: prove no leakage

Run your E1 attention causally on a random `(1, 8, d)` input, then add 10.0 to the
embedding of position 5 and rerun. Report `max |Δout|` at positions 0–4 and at 5–7.
Hint: compare outputs with `np.abs(out2 - out1).max(axis=-1)`.
Accept when: `max |Δout|` is exactly 0.0 at every position < 5 and nonzero at ≥ 5.

## E5 — MultiHeadAttention module

Implement `MultiHeadAttention(d_model, n_head)` as an `nn.Module` in `attention.py`:
one fused QKV projection, reshape to heads, causal scaled dot-product, concat, output
projection. Week 26 imports this class unchanged.
Hint: project to `(B, T, 3*d_model)`, `view` to `(B, T, 3, n_head, d_k)`, move the
head axis before T.
Accept when: with weights copied from `nn.MultiheadAttention` (batch_first=True), your
output matches it within 1e-5 on random input.

## E6 — Read a head

Build the toy sequence "A B C B A B C B" (as integer tokens with random fixed
embeddings), run your E5 module with 4 heads untrained and after 200 steps of training
to predict the next token, and plot each head's weight matrix as a heatmap.
Hint: `plt.imshow(A[0, h])` per head; label axes "query position" / "key position".
Accept when: heatmaps are saved and each trained head's pattern is described in one
sentence (e.g. "attends to previous token", "attends to matching letter", "diffuse").

## Review

1. (Week 13) Derive the gradient of softmax + cross-entropy in two lines. Why is the
   result so clean?
2. (Week 19) Why does an LSTM struggle with long-range credit assignment, and which
   property of attention removes that problem?
3. (Weeks 07–08) Write the KL divergence and state why it is never negative. Where did
   the cross-entropy loss come from?
4. (Week 16) Name two other saturation-type failures from training dynamics week and
   the standard fix for each.
