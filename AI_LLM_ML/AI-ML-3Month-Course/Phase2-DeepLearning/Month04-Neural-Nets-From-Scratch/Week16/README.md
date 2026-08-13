# Week 16 — Training Dynamics + Mini-Project

Whether a deep net trains at all is a question about activation and gradient statistics
through layers — beam transport: keep the distribution stable stage to stage or lose it.

## Objectives

- Derive Xavier and He initialization from the variance-preservation condition and state
  which pairs with which activation.
- Instrument a training run: activation histograms per layer, gradient norms, update-to-
  weight ratios — and read them.
- Explain what batchnorm and layernorm each normalize over, and why batchnorm behaves
  differently at train vs eval time.
- Recognize the classic pathologies from their signatures: dead ReLUs, saturated tanh,
  exploding gradients, a too-high or too-low LR.
- Run an honest comparison: tuned MLP vs your Capstone-1 BDT on tabular physics data.

## Core material (~3 hrs)

- Karpathy, *Zero to Hero* — "Building makemore Part 3: Activations & Gradients,
  BatchNorm" (the spine; do the diagnostics along with him).
- *Understanding Deep Learning* (Prince), Ch. 7 (Gradients and initialization); skim
  Ch. 9 (Regularization).
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural
  networks" — read for the variance argument, skim the experiments.
- Goodfellow et al., Ch. 8 (Optimization for training deep models) as reference for LR
  schedules and gradient clipping.

## Derivations (paper first)

- Xavier init: assuming independent zero-mean weights and inputs, derive
  Var(W) = 1/n_in for variance preservation forward (and the fan-avg compromise when you
  include the backward pass).
- He init: redo the derivation with ReLU, showing where the factor of 2 comes from.
- Batchnorm forward pass and its effect on the loss surface's sensitivity to weight
  scale (show the loss is invariant to rescaling `W` before a BN layer).

## Exercises (built when the week starts)

1. Init sweep on a 5-layer tanh MLP: N(0, 1), Xavier, He — plot per-layer activation
   histograms at step 0. Accept when: plots reproduce the saturate/stable/expected
   pattern and each gets a one-line reading.
2. Dead-ReLU counter: fraction of units with zero activation across a full epoch, vs
   learning rate. Accept when: plot shows the death rate rising with LR and the
   mechanism is stated in one line.
3. Diagnose broken run A (provided script: silent LR=10 with gradient explosion).
   Accept when: named diagnosis + one-line fix, confirmed by the loss curve after the fix.
4. Diagnose broken run B (provided script: unscaled inputs + bad init → saturated
   sigmoids). Accept when: same criterion as 3.
5. Diagnose broken run C (provided script: `zero_grad()` missing). Accept when: same
   criterion as 3 — this one you met in Week 15's review.
6. Add batchnorm to the Week-15 char MLP; compare training stability across LRs with and
   without. Accept when: plot shows BN widening the range of workable LRs.
7. **Mini-project:** MLP on the Capstone-1 tabular dataset — Phase-1 splits reused,
   init/LR/normalization tuned, early stopping. Accept when: AUC vs the BDT under the
   identical CV protocol, with a win or a defensible why-not (BDT home turf; a loss is
   a finding).

## Deliverable

Mini-project notebook + short writeup (MLP vs BDT, with diagnostics plots), plus the
three diagnosed-and-fixed broken runs. This closes Month 04: tag `month-04-complete`,
write `retro.md`, open the open-question issue.

## Review

- (Week 13) Re-derive the backward recursion δ^(l) — cold. This is the month's flagship.
- (Week 08) Batchnorm standardizes to zero mean, unit variance. Which Week-08 moments
  are estimated on-the-fly, and why does small batch size make them noisy?
- (Week 11) Why do trees not care about feature scaling while your MLP badly does?
