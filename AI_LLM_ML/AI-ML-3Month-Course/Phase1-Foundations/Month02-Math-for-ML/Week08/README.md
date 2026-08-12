# Week 08 — Optimization + Mini-Project

MINUIT has been hiding this from you for years: this week you derive and build the
optimizers yourself, then race them on surfaces designed to hurt.

## Objectives

- Derive gradient descent from a local Taylor expansion and bound its step size on a quadratic.
- Explain condition number as the thing that makes GD zig-zag, with the eigenvalue picture.
- Derive and implement SGD, momentum, RMSProp, and Adam (with bias correction) from scratch.
- Fit physics models with your own optimizers and diagnose failures from the loss curve.
- Say when a problem is convex, and what that buys you.

## Core material (~3 hrs)

- Boyd & Vandenberghe, *Convex Optimization*: skim the early material on convex sets
  and functions — light touch, definitions and pictures only.
- Ruder, "An overview of gradient descent optimization algorithms" — the standard
  survey; read momentum through Adam.
- 3Blue1Brown: the gradient-descent video from the neural-network series (intuition
  refresher, 20 min).
- Optional: Goh, "Why Momentum Really Works" (Distill) — the condition-number picture,
  interactive.

## Derivations (paper first)

- GD from first-order Taylor; on f(x) = ½xᵀAx, derive the stable range η < 2/λmax and
  the per-mode convergence rates (this is where condition number enters).
- SGD: show the minibatch gradient is an unbiased estimator of the full gradient; note
  the variance/batch-size tradeoff.
- Momentum: write the update as an exponentially-weighted moving average of gradients;
  connect to a damped oscillator (this one is your home turf).
- RMSProp: per-coordinate step normalization; why it helps anisotropic curvature.
- Adam: combine the two moment estimates; derive the bias-correction factors
  1/(1 − βᵗ) from the EMA initialization at zero.

## Exercises (built when the week starts)

1. **Optimizer module.** Implement `gd`, `sgd`, `momentum`, `rmsprop`, `adam` with a
   common interface in `src/optim.py`, each ≤15 lines.
   Accept when: each matches a hand-computed 3-step trace on a 2D quadratic to 1e-10.
2. **Step-size threshold.** On an ill-conditioned quadratic (κ = 100), sweep η across
   2/λmax and show convergence/divergence exactly where derived.
   Accept when: empirical divergence threshold matches 2/λmax within 5%.
3. **The race.** All five optimizers on: well-conditioned quadratic, κ = 10³ quadratic,
   Rosenbrock. Trajectory plots over loss contours plus loss-vs-iteration curves.
   Accept when: all trajectories are plotted, and momentum beats plain GD on the ill-conditioned quadratic by ≥5× fewer iterations to loss < 1e-6.
4. **SGD noise.** Fit a line with minibatch SGD at batch sizes 1, 10, 100, full; plot
   loss-curve noise and final-parameter scatter vs batch size.
   Accept when: gradient-estimate variance scales ~1/batch size across the sweep.
5. **Physics fit (mini-project core).** Fit a Breit–Wigner + polynomial background to a
   pseudo-data mass spectrum using your Adam, minimizing the negative log-likelihood
   from Week 07; compare to `scipy.optimize`.
   Accept when: your Adam's fitted mass and width agree with scipy's within 1σ of the fit uncertainties.
6. **Pathology report.** One surface where Adam underperforms plain momentum (e.g., a
   ravine where adaptive scaling misleads); demonstrate and explain in ≤3 lines.
   Accept when: the loss curves show the crossover and the explanation names the mechanism.

## Deliverable

The Month 02 deliverable: `src/optim.py` with tests, the race notebook with trajectory
figures, the physics-fit comparison, and scanned derivations. Then month sign-off
(tag `month-02-complete`, `retro.md`, one issue).

## Review

1. (Wk 07) Why is minimizing negative log-likelihood for Gaussian noise the same as
   least squares — and what changes in Exercise 5's Poisson-count case?
2. (Wk 06) Condition number of AᵀA vs A: which did the normal equations suffer from,
   and how does that connect to this week's κ story?
3. (Wk 05) The GD convergence-per-mode analysis lives in which basis? Why is the
   symmetric-matrix theorem from Week 05 what makes the analysis clean?
4. (Wk 03) Your optimizer module needs tests. Name the two most valuable test cases
   you'd write first.
