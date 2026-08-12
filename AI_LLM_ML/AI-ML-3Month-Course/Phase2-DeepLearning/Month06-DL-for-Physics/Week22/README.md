# Week 22 — Autoencoders & VAEs

A VAE is variational inference with a neural network for the trial distribution — the
same move as a variational wavefunction ansatz: pick a tractable family, bound the
quantity you can't compute, optimize the bound. This week's ELBO derivation is the
course's second flagship derivation and part of the Phase 2 gate.

## Objectives

- Train a plain autoencoder and say why its latent space is not a generative model.
- Derive the ELBO line by line from log p(x), two ways: Jensen's inequality, and the
  exact decomposition log p(x) = ELBO + KL(q‖posterior).
- Explain and implement the reparameterization trick; say why the naive gradient
  through sampling fails.
- Recognize posterior collapse (KL → 0, decoder ignores z) and apply mitigations
  (KL warm-up, β weighting).
- Build a HEP-style anomaly detector from reconstruction error; know its failure modes.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: both ELBO routes with every line
  justified, the closed-form Gaussian KL, reparameterization, collapse, and a
  runnable ~60-line VAE.
- Lilian Weng, "From Autoencoder to Beta-VAE" (blog) — the landscape of variants.
- *Understanding Deep Learning* (Prince), Ch. 17 (Variational autoencoders) — the
  math in a second notation.
- Kingma & Welling, "Auto-Encoding Variational Bayes" (arXiv:1312.6114) — read §1–3
  after your own derivation; it should read as familiar by then.
- Skim one HEP autoencoder anomaly-detection paper (LHC Olympics context) for usage.

## Derivations (paper first)

- ELBO #1: Jensen. ELBO #2: exact decomposition, showing the gap is
  KL(q(z|x) ‖ p(z|x)) ≥ 0. Every line justified.
- Closed-form KL between N(μ, σ²) (diagonal) and N(0, I) — the term you type into the
  loss.
- Reparameterization: show z = μ + σ⊙ε moves ∇_φ inside E_{q_φ}[f(z)], and why the
  gradient of a raw sample w.r.t. φ is undefined without it.
- Gaussian decoder ⇒ MSE reconstruction loss (Week 08's Gaussian-MLE argument), with
  the assumed variance surfacing as the β weighting.

## Exercises

See `exercises.md` (notebook generated from it when the week starts, per
`NOTEBOOK_RULES.md`). E1–E3 walk autoencoder → checked KL formula → working shower
VAE; E4–E5 break the gradient and induce posterior collapse on purpose; E6 builds
the anomaly detector and compares it honestly to the Week-20 CNN.

## Deliverable

Scanned ELBO derivations (both routes) + a working shower VAE with prior samples, the
collapse study, and the anomaly-detection comparison. The cold ELBO re-derivation is
part of the Phase 2 gate — schedule it now.

## Review

- (Week 08) State KL's key properties; why does KL(q‖p) vs KL(p‖q) matter here?
- (Week 12) EM for GMMs also maximized a lower bound. What plays the E-step's role in a
  VAE, and what is amortization?
- (Week 13) Trace the chain-rule path from reconstruction loss back to the encoder and
  mark where reparameterization inserts itself.
- (Week 07) PCA is the optimal linear autoencoder under squared error. What does
  nonlinearity buy on shower images that Week-07 PCA could not?
