# Week 28 — Scaling Laws & Interpretability

Loss vs compute follows power laws clean enough to make a physicist suspicious — and
interpretability is the young field asking what the fitted system actually computes,
currently somewhere between spectroscopy and phrenology.

## Objectives

- State the Kaplan and Chinchilla scaling results and derive the compute-optimal
  N(C), D(C) split from the fitted loss form.
- Take a position on the emergence debate, having read both the claim and the "mirage"
  critique, and defend it with the metric-choice argument.
- Explain induction heads and find one in your own Week 27 model.
- Explain superposition and reproduce the toy-model phenomenon.
- Say clearly what current interpretability can and cannot establish.

## Core material (~3 hrs)

- Kaplan et al., *Scaling Laws for Neural Language Models* (arXiv 2001.08361) — §§1–3
  and the figures.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla,
  arXiv 2203.15556) — approaches 1–3 and Table comparisons.
- Wei et al., *Emergent Abilities of Large Language Models* vs Schaeffer et al., *Are
  Emergent Abilities of Large Language Models a Mirage?* — read both, in that order.
- Anthropic transformer-circuits thread (transformer-circuits.pub): *A Mathematical
  Framework for Transformer Circuits* (skim), *In-context Learning and Induction Heads*
  (main read), *Toy Models of Superposition* (intro + first experiments).

## Derivations (paper first)

- Given L(N, D) = E + A/N^α + B/D^β and the compute constraint C ≈ 6ND, derive the
  compute-optimal N* ∝ C^{β/(α+β)}, D* ∝ C^{α/(α+β)} via Lagrange multiplier or
  substitution; plug in Chinchilla's fitted α, β and recover "≈ 20 tokens per parameter."
- Show how a smooth per-token accuracy curve becomes a sharp "emergent" jump under an
  exact-match-on-k-tokens metric: P(exact) = p^k. This is the mirage argument in one line.

## Exercises (built when the week starts)

1. Scaling mini-study: train 3–4 sizes of your Week 27 model (same data, same budget
   rules), fit L(N) = A/N^α + const. Accept when: fit plotted on log axes with α and
   its uncertainty reported.
2. Chinchilla optimum, numerically: minimize the fitted L(N, D) under C = 6ND on a grid.
   Accept when: numeric optimum matches the closed-form exponents within 5%.
3. Mirage demo: from one model's per-token probabilities, plot per-token accuracy vs
   exact-match-on-k for k = 1, 2, 5, 10. Accept when: smooth curve visibly becomes a
   cliff and the p^k prediction overlays it.
4. Induction-head hunt: feed repeated random-token sequences to your Week 27 model,
   compute each head's prefix-matching and copying scores. Accept when: a heatmap of
   scores per (layer, head) identifies at least one candidate head, or its absence is
   argued from model size.
5. Toy superposition: reproduce the basic Anthropic toy model (n features > m dims,
   ReLU readout) across a sparsity sweep. Accept when: the W^T W plots show the
   dedicated-dimension → superposition transition.

## Deliverable

Derivation scans; `Week28_Exercises.ipynb` with the scaling fit, mirage plot, induction
scores, and superposition sweep; a half-page written position on emergence in `notes.md`.

## Review

- Week 25: re-derive the √d_k scaling cold (this is the Phase 3 gate derivation — first
  rehearsal).
- Week 9: the scaling-law fit is a regression. What loss did you minimize, and why fit
  in log space? Connect to Week 08's Gaussian-likelihood derivation of least squares.
- Week 12: the superposition toy model compresses n features into m < n dimensions.
  How is this like and unlike PCA?
- Week 22: posterior collapse was a Phase 2 failure mode of latent codes. State it in
  one sentence and contrast with superposition (too little vs too much packed in).
