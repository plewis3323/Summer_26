# Week 28 — Exercises

Work top to bottom, derivations first: the compute-optimal N*(C), D*(C) split
and the P(exact) = p^k mirage argument go on paper (README "Derivations") and
are scanned into this folder before you start E2. Setup (imports, your Week 27
model class and a trained checkpoint, the corpus token counts, plot scaffolds,
seeded RNGs) is given by the notebook; you write only the lines each exercise
asks for. Everything lives in the notebook this week except the emergence
position (`notes.md`) and the derivation scans. This is also the first
rehearsal of the Phase 3 gate derivation — re-derive √d_k cold before the
weekend (Review item 1).

If this week is the one the syllabus §6 cut line hits, derivations + E1–E3
plus the `notes.md` position are the minimum viable week.

## E1 — Scaling mini-study

Train 3–4 sizes of your Week 27 model (same data, same budget rules — same
optimizer, same token budget *per parameter* or a stated IsoFLOP-style rule,
not "train the big one longer because you felt like it"). Fit
L(N) = A / N^α + const to the validation losses.
Hint: N is non-embedding parameters, counted as in Week 26; L is val
cross-entropy in nats/token. Fit in log space (or weighted least squares) so
the largest model does not dominate — Review item 2 is about this choice.
Three sizes is a line; four lets you see a residual. Report α ± uncertainty
from the fit (covariance of the least-squares parameters, or a bootstrap over
seeds if you have them).
Accept when: fit plotted on log axes with α and its uncertainty reported.

## E2 — Chinchilla optimum, numerically

Using a fitted L(N, D) = E + A/N^α + B/D^β (your E1 exponents if you fitted
the joint form; otherwise Hoffmann's Approach-3 numbers from lesson §3.3:
α ≈ 0.34, β ≈ 0.28, E ≈ 1.69, A ≈ 406.4, B ≈ 410.7), minimize L under
C = 6 N D on a grid over N. Overlay the closed-form N* ∝ C^{β/(α+β)},
D* ∝ C^{α/(α+β)}.
Hint: substitution is one line — D = C / (6N), scan N, read off the argmin.
Do it at several C and plot N*(C) on log–log axes against the closed-form
power. The 5% is on the *exponents* (the slope), not on the prefactor.
Accept when: numeric optimum matches the closed-form exponents within 5%.

## E3 — Mirage demo

From one model's per-token probabilities on a held-out set, plot per-token
accuracy vs exact-match-on-k for k = 1, 2, 5, 10. Overlay the p^k prediction.
Hint: p is the mean per-token accuracy (greedy argmax vs gold, or the model's
probability on the gold token — say which). Exact-match-on-k is the fraction
of k-token windows where *every* token is correct. Independence is the
zeroth-order assumption; the overlay will be slightly off and that is fine —
the point is the cliff, not a χ². Linear y-axis, log-x if you bin by
difficulty, otherwise just k on x and the four curves.
Accept when: smooth curve visibly becomes a cliff and the p^k prediction
overlays it.

## E4 — Induction-head hunt

Feed repeated random-token sequences to your Week 27 model and compute each
head's prefix-matching and copying scores (lesson §5). Heatmap over
(layer, head).
Hint: draw a random token string, concatenate it with itself, and at each
position in the second half read the attention weight onto the token *after*
the previous occurrence of the current token — that is prefix-matching.
Copying: whether the head's OV circuit raises the logit of the matched token
(a dot of the head output with the unembedding row is a standard proxy).
Absence is an acceptable result if the heatmap is real and the argument is
"this N and D are below the scale where the circuit forms," not "I didn't
look."
Accept when: a heatmap of scores per (layer, head) identifies at least one
candidate head, or its absence is argued from model size.

## E5 — Toy superposition

Reproduce the basic Anthropic toy model (n features > m dims, ReLU readout)
across a sparsity sweep. Plot W^T W at a dense setting and a sparse setting.
Hint: lesson §6 — h = W x, x̂ = ReLU(W^T h + b), train reconstruction MSE.
n = 20, m = 5 is plenty. Sweep the probability that a feature is on; the
transition is the deliverable. Diagonal of W^T W = "how much of its own
dimension this feature got"; off-diagonals = interference. Dense → partial
identity (dedicated dimensions, leftover features ignored); sparse → many
medium diagonals and a spray of off-diagonals (superposition).
Accept when: the W^T W plots show the dedicated-dimension → superposition
transition.

## Review

1. Week 25: re-derive the √d_k scaling cold (this is the Phase 3 gate
   derivation — first rehearsal).
2. Week 9: the scaling-law fit is a regression. What loss did you minimize,
   and why fit in log space? Connect to Week 08's Gaussian-likelihood
   derivation of least squares.
3. Week 12: the superposition toy model compresses n features into m < n
   dimensions. How is this like and unlike PCA?
4. Week 22: posterior collapse was a Phase 2 failure mode of latent codes.
   State it in one sentence and contrast with superposition (too little vs
   too much packed in).
