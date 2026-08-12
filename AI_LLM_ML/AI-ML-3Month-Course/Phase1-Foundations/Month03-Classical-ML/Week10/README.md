# Week 10 — Classification

A classifier threshold is a trigger cut — the lesson explains what a collider
trigger is from zero — and this week formalizes efficiency-vs-purity into
ROC/PR/calibration and derives the logistic-regression machinery by hand.

## Objectives

- Derive the logistic-regression gradient from the Bernoulli likelihood, by hand.
- Generalize to softmax and derive the cross-entropy gradient (the form backprop reuses).
- Build and read ROC and PR curves; choose thresholds for a stated physics goal.
- Assess and repair calibration (reliability curves, Platt/isotonic at usage level).
- Handle class imbalance without fooling yourself with accuracy.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: sigmoid and log-odds, the full
  NLL-gradient derivation, softmax/cross-entropy, thresholds and the metrics zoo,
  the trigger story, calibration, and imbalance.
- StatQuest: "Logistic Regression" and "ROC and AUC" videos — fast intuition pass
  before the math.
- Bishop, *PRML* §4.3 (probabilistic discriminative models — logistic regression and
  its gradient). The derivation you'll redo on paper.
- scikit-learn user guide: "Model evaluation" sections on ROC, precision-recall, and
  the calibration page.
- Optional: Murphy, *Probabilistic ML: An Introduction*, logistic regression chapter as
  a second derivation angle.

## Derivations (paper first)

- Sigmoid: σ′ = σ(1 − σ), and why the log-odds is the natural linear quantity.
- Logistic loss: write the Bernoulli likelihood, take −log, derive
  ∇w = Xᵀ(σ(Xw) − y). Note how clean the residual form is.
- Softmax + cross-entropy: derive ∂L/∂z = p − y for one-hot targets. Flag this page:
  Phase 2's backprop derivation starts here.
- Show accuracy's failure mode at 99:1 imbalance in two lines of algebra.

## Exercises

See `exercises.md` (notebook built from it when the week starts, per
`NOTEBOOK_RULES.md`). Six exercises: logistic regression from scratch trained with
your Week 08 Adam and checked against sklearn, decision-boundary plots, ROC/PR/AUC
built by hand, trigger working points on a 1:100 dataset, calibration repair, and a
three-class softmax that verifies ∂L/∂z = p − y numerically.

## Deliverable

Completed notebook plus scanned derivations — the softmax/cross-entropy page is a
Phase 2 dependency; file it where you can find it in Month 04.

## Review

1. (Wk 09) Why must threshold selection (E4) happen on validation data, not test?
2. (Wk 08) Training logistic regression has no closed form. Why not (one line), and
   which property of the loss still guarantees GD finds the optimum?
3. (Wk 08) Cross-entropy loss is the KL to what distribution, plus what constant?
4. (Wk 03) Compute TPR from a boolean prediction array and truth array in one NumPy line.
5. (Physics) Trigger efficiency and purity map onto which two ROC/PR quantities?
