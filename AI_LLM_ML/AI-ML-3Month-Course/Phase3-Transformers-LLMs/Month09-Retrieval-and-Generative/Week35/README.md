# Week 35 — Diffusion Models (GPU week)

Diffusion models are Langevin dynamics run in reverse: destroy data with a known noising
process, learn the score, and integrate back — the generative model closest to your
physics training, derived here in full.

## Objectives

- Derive the complete DDPM training objective from the forward process to the
  simplified ε-prediction loss, on paper.
- Implement and train a DDPM on toy 2-D data and on image-like calorimeter arrays.
- Explain and implement classifier-free guidance for conditional generation.
- Place diffusion vs normalizing flows vs VAEs on axes of sample quality, likelihood
  access, sampling cost — specifically for calorimeter simulation.
- Connect the reverse process to score matching and Langevin dynamics explicitly.

## Core material (~3 hrs)

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (arXiv 2006.11239) —
  §§1–3 + Algorithm 1/2; this is the derivation source.
- Lilian Weng, *What are Diffusion Models?* (lilianweng.github.io) — the single best
  companion to the algebra.
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (arXiv 2207.12598) — short; read
  fully.
- Prince, *UDL* Ch. 18 (Diffusion models) for a second pass at the derivation.
- Skim one CaloChallenge diffusion entry (e.g. CaloScore or CaloDiffusion, by name) for
  how the field conditions on energy and validates with shower observables.

## Derivations (paper first)

This is a flagship derivation week — file the scans for the cold-redo rotation.

- Forward process: q(x_t|x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I); derive the closed form
  q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I) by composing Gaussians.
- Reverse posterior: q(x_{t−1}|x_t, x_0) via Bayes/Gaussian algebra; obtain its mean μ̃_t
  and variance β̃_t.
- The variational bound: write the ELBO for the diffusion chain (the Week 22 ELBO,
  stretched over T steps), reduce to a sum of per-step KLs between Gaussians.
- Reparameterize μ via ε-prediction and reduce to L_simple = E‖ε − ε_θ(x_t, t)‖²; note
  which weighting was dropped and why Ho et al. drop it.
- Classifier-free guidance: ε̃ = (1+w) ε_θ(x, c) − w ε_θ(x, ∅); derive from the implied
  score of p(x|c) p(c|x)^w.
- One paragraph: ε-prediction ↔ score ∇_x log p(x_t); reverse sampling as discretized
  Langevin dynamics with a drift term.

## Exercises (built when the week starts)

1. Forward-process check: noise 2-D data (two-moons / Gaussian mixture) step-by-step
   and via the closed form. Accept when: means/covariances agree within Monte Carlo
   error at t ∈ {10, 100, 500}.
2. Train a small MLP DDPM on the 2-D data; sample with Algorithm 2. Accept when:
   sampled point cloud visually recovers both modes and a 2-D histogram χ² against
   truth beats a single-Gaussian baseline.
3. Sanity: plot ‖ε − ε_θ‖² vs t. Accept when: curve saved with two sentences on which
   timesteps are hardest.
4. Conditional DDPM with class labels + CFG on a labeled toy set (or MNIST-class
   images). Accept when: sweeping w ∈ {0, 1, 3, 5} visibly trades diversity for
   condition adherence, shown in a grid.
5. Calo warm-up: train a conditional DDPM on simplified shower arrays (Capstone 2 data
   or a public CaloChallenge-style set), conditioned on incident energy. Accept when:
   generated mean energy response vs condition is monotonic and within 10% of truth
   across the tested range.
6. Comparison table: diffusion vs flows (Week 23) vs VAE (Week 22) — likelihood,
   sampling cost, quality, conditioning ease. Accept when: table filled from your own
   three implementations, one cited claim max.

## Deliverable

Full derivation scans; `Week35_Exercises.ipynb` with checks PASS; trained toy + calo
warm-up checkpoints and sample grids — the direct on-ramp to Capstone 3 option (b).

## Review

- Week 22: write the VAE ELBO from memory. Point to the exact line where the diffusion
  bound is that ELBO with a fixed, non-learned encoder.
- Week 23: flows require invertible maps with tractable Jacobians. Which constraint
  does diffusion drop, and what does it pay for that freedom at sampling time?
- Week 08: composing Gaussians was the key trick today. Derive Var(aX + bY) for
  independent X, Y and connect to the ᾱ_t closed form.
- Week 08: SGD with noise resembles Langevin dynamics. In one paragraph: where does
  noise appear in each, and what plays the role of temperature?
