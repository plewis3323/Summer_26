# Week 23 — Normalizing Flows + Fast Simulation

A flow is a learned change of variables — the Jacobian bookkeeping from every phase-space
integral you've done, run through an invertible network so the model gives exact
likelihoods, not a bound.

## Objectives

- Derive the change-of-variables formula and the flow log-likelihood with log |det J|.
- Explain the design problem — expressive, invertible, tractable Jacobian determinant —
  and how affine coupling layers (RealNVP-style) solve it with a triangular Jacobian.
- Implement a coupling-layer flow, train by exact maximum likelihood, sample by running
  it in reverse.
- Compare flows vs VAEs for fast-sim: likelihood exactness, sampling speed, scaling.
- Explain the CaloChallenge: why colliders need learned simulators and how submissions
  are judged on physics observables.

## Core material (~3 hrs)

- Lilian Weng, "Flow-based Deep Generative Models" (blog) — the spine.
- *Understanding Deep Learning* (Prince), Ch. 16 (Normalizing flows).
- Dinh, Sohl-Dickstein & Bengio, "Density estimation using Real NVP"
  (arXiv:1605.08803) — the coupling-layer construction; skim multiscale details.
- CaloChallenge: the "Fast Calorimeter Simulation Challenge 2022" description and the
  CaloFlow paper (Krause & Shih) — skim for framing and evaluation observables.

## Derivations (paper first)

- Change of variables in 1D from conservation of probability mass, then the general
  |det J| case; write the log-likelihood for a K-layer composition.
- Affine coupling layer: forward map, triangular Jacobian whose log-det is a plain sum
  (no determinant computed), and the analytic inverse.
- Show alternating masks let a stack of coupling layers mix all dimensions, and why a
  single layer provably cannot touch its pass-through half.

## Exercises (built when the week starts)

1. 1D warm-up: fit a flow (chain of parametrized monotone maps) to a Breit–Wigner from
   samples; overlay learned density on the analytic curve. Accept when: KS-style
   comparison passes at the stated tolerance.
2. Coupling layer forward/inverse in PyTorch. Accept when: `inverse(forward(x))`
   round-trips to 1e-5 and the log-det matches `torch.autograd.functional.jacobian`
   on small inputs.
3. RealNVP-style flow (≥ 6 layers, alternating masks) on two-moons. Accept when:
   samples overlay the data convincingly and held-out log-likelihood beats a Week-12
   GMM baseline.
4. Mask ablation: same flow without alternation. Accept when: the untouched dimensions
   are visible in the sample plot and explained in one line.
5. Flow fast-sim on Week-20 single-photon showers (flattened tower energies, or
   per-cluster features if dimensionality bites). Accept when: samples pass an eyeball
   test and per-tower marginals roughly match — full physics validation is Week 24's job.
6. Head-to-head vs the Week-22 VAE: wall-clock for 10k samples, plus exact NLL (flow)
   vs ELBO (VAE) on held-out data. Accept when: the table exists with a one-line caveat
   on comparing NLL to a bound.

## Deliverable

Paper derivations + a trained coupling flow on toy and shower data, with the flow-vs-VAE
table informing your Week-24 choice. (Syllabus §6 lists this week first-cut if the
calendar slips; minimum viable = derivations + Exercises 1–3.)

## Review

- (Week 08) Why does maximum likelihood on a flow need no ELBO — what made p(x)
  intractable for the VAE and tractable here?
- (Week 06) Log-det of a triangular Jacobian = sum of log-diagonals. Connect to
  Week 06 on determinants, volumes, and eigenvalues.
- (Week 22) Point to the line in your ELBO derivation where the bound becomes tight.
  What would q need to equal?
- (Week 16) Which training pathologies should you watch for in the coupling scale-net
  outputs, and what does the usual tanh-clamp on s(x) prevent?
