# Week 08 — Probability, Information, and the Optimizer Race

You already do MLE every time you fit a peak; this week makes the machinery
explicit, adds the information-theoretic vocabulary ML losses are written in,
and then derives the optimizer zoo and races it on surfaces designed to hurt.
This is a packed week — the syllabus said Months 01–02 run hot; take the slack
if the race spills a day.

## Objectives

- Manipulate joint/marginal/conditional distributions and apply Bayes' theorem
  cleanly.
- Derive MLE and MAP estimators and articulate exactly how a prior changes the
  answer; derive least squares from a Gaussian likelihood.
- Compute entropy, cross-entropy, and KL divergence, and state the chain
  minimizing cross-entropy ⇔ minimizing KL ⇔ maximizing likelihood.
- Derive gradient descent's step-size bound on a quadratic, and implement SGD,
  momentum, RMSProp, and Adam (with bias correction) from scratch.
- Fit a physics model with your own Adam and diagnose failures from the loss
  curve.

## Core material (~3 hrs)

- `lesson.md` (this folder) — probability through information theory, then the
  optimizer derivations; `project.md` is the race spec.
- Bishop, *PRML* (in `references/`): §1.2 (probability theory) and §2.1–2.3
  (binary/multinomial variables, the Gaussian).
- 3Blue1Brown: the Bayes' theorem video, and the gradient-descent video from
  the neural-network series.
- Ruder, "An overview of gradient descent optimization algorithms" — momentum
  through Adam.
- Optional: Goh, "Why Momentum Really Works" (Distill); Boyd & Vandenberghe
  early pages on convex sets (pictures only).

## Derivations (paper first)

- Bayes' theorem from the product rule; one full example with a non-flat prior.
- MLE for the mean and variance of a Gaussian (and show the variance MLE is
  biased); least squares from a Gaussian likelihood; MAP with a Gaussian prior
  → ridge.
- KL$(p\|q)$ for two Gaussians (closed form); KL ≥ 0 via Jensen; cross-entropy
  = entropy + KL.
- GD from first-order Taylor; on $f(x)=\tfrac12 x^\top Ax$, the stable range
  $\eta < 2/\lambda_{\max}$ and the per-mode rates (condition number).
- SGD unbiasedness; momentum as an EMA of gradients; RMSProp per-coordinate
  scaling; Adam's bias-correction factors $1/(1-\beta^t)$.

## Exercises

See `exercises.md`. Probability block (E1–E4) in the notebook; optimizer block
(E5–E8) is the mini-project in `src/optim.py` plus the race notebook, fully
specified in `project.md`.

## Deliverable

The Month 02 deliverable: scanned probability and optimizer derivations,
`src/optim.py` with tests, the race notebook with trajectory figures, and the
physics-fit comparison. Then month sign-off (tag `month-02-complete`,
`retro.md`, one issue).

## Review

1. (Wk 07) Condition number of $A^\top A$ vs $A$: which did the normal
   equations suffer from, and how does that connect to this week's $\kappa$
   story?
2. (Wk 07) Write $\nabla_x\|Ax-b\|^2$ from memory; which derivation this week
   reused it?
3. (Wk 06) The projection matrix $P$ and the least-squares/likelihood fit:
   what is the geometric relationship?
4. (Wk 04) Your pseudo-experiment loops need reproducible randomness — which
   NumPy API, and why not the global seed?
