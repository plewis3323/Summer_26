# Week 22 — Autoencoders & VAEs

A VAE is variational inference with a neural network for the trial distribution — the
same move as a variational wavefunction ansatz: pick a tractable family, bound the
quantity you can't compute, optimize the bound.

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

- Lilian Weng, "From Autoencoder to Beta-VAE" (blog) — the spine for the landscape.
- *Understanding Deep Learning* (Prince), Ch. 17 (Variational autoencoders) — the spine
  for the math.
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
- Gaussian decoder ⇒ MSE reconstruction loss (Week 07's Gaussian-MLE argument), with
  the assumed variance surfacing as the β weighting.

## Exercises (built when the week starts)

1. Plain autoencoder on Week-20 shower images, latent dim 2; latent scatter colored by
   class and energy. Accept when: plot exists plus one line on where sampling a random
   latent point fails to decode to a valid shower.
2. Closed-form Gaussian KL checked against a Monte-Carlo estimate. Accept when:
   agreement within MC error for three (μ, σ) settings.
3. VAE on shower images: reparameterized encoder, ELBO loss from your derivation.
   Accept when: both ELBO terms logged separately, total decreases, and prior samples
   are plotted.
4. Reparameterization ablation: naive sample-then-detach version. Accept when: encoder
   gradient shown to be zero/garbage, with the violated derivation step named.
5. Posterior collapse on demand: raise β until KL → 0. Accept when: β-sweep plot shows
   the reconstruction/KL trade-off with the collapsed regime marked.
6. Anomaly detection: train on single-photon showers only; score merged-π⁰ by
   reconstruction error. Accept when: ROC AUC vs the supervised Week-20 CNN is reported
   with one line on the gap and on when unsupervised is still worth it.

## Deliverable

Scanned ELBO derivations (both routes) + a working shower VAE with prior samples, the
collapse study, and the anomaly-detection comparison. The cold ELBO re-derivation is
part of the Phase 2 gate — schedule it now.

## Review

- (Week 07) State KL's key properties; why does KL(q‖p) vs KL(p‖q) matter here?
- (Week 12) EM for GMMs also maximized a lower bound. What plays the E-step's role in a
  VAE, and what is amortization?
- (Week 13) Trace the chain-rule path from reconstruction loss back to the encoder and
  mark where reparameterization inserts itself.
- (Week 06) PCA is the optimal linear autoencoder under squared error. What does
  nonlinearity buy on shower images that Week-06 PCA could not?
