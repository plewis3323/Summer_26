# Week 10 — Classification

A classifier threshold is a trigger cut: this week formalizes efficiency-vs-purity into
ROC/PR/calibration and derives the logistic-regression machinery by hand.

## Objectives

- Derive the logistic-regression gradient from the Bernoulli likelihood, by hand.
- Generalize to softmax and derive the cross-entropy gradient (the form backprop reuses).
- Build and read ROC and PR curves; choose thresholds for a stated physics goal.
- Assess and repair calibration (reliability curves, Platt/isotonic at usage level).
- Handle class imbalance without fooling yourself with accuracy.

## Core material (~3 hrs)

- Bishop, *PRML* §4.3 (probabilistic discriminative models — logistic regression and
  its gradient). The derivation you'll redo on paper.
- StatQuest: "Logistic Regression" and "ROC and AUC" videos — fast intuition pass
  before the math.
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

## Exercises (built when the week starts)

1. **Logistic from scratch.** Implement the derived gradient; train with your Week 08
   Adam on a two-Gaussian dataset; verify with the Week 06 gradient checker.
   Accept when: gradient check passes at 1e-6 and coefficients match sklearn `LogisticRegression` (matched regularization) to 1e-4.
2. **Decision boundary.** Plot the learned boundary and probability contours; move the
   threshold and watch the boundary shift.
   Accept when: the plotted 0.5 contour matches the analytic line wᵀx + b = 0.
3. **ROC/PR from scratch.** Sweep thresholds; compute TPR/FPR/precision/recall; overlay
   sklearn's curves; compute AUC by trapezoid.
   Accept when: AUC matches `roc_auc_score` to 1e-6 and the PR curve matches sklearn's point set.
4. **Trigger working point.** On an imbalanced (1:100) signal/background dataset,
   choose thresholds for (a) 90% signal efficiency, (b) max significance-like
   S/√(S+B); report the confusion matrix at each.
   Accept when: reported efficiencies at each working point are within 1% of target on held-out data.
5. **Calibration.** Reliability diagram for the scratch model and for an overconfident
   model (provided); apply sklearn's `CalibratedClassifierCV`; re-plot.
   Accept when: post-calibration curve is visibly nearer the diagonal and Brier score improves.
6. **Softmax three-class.** Extend to three Gaussian blobs with softmax; confirm
   ∂L/∂z = p − y numerically.
   Accept when: numeric and analytic gradients agree to 1e-6 and test accuracy exceeds a stated baseline.

## Deliverable

Completed notebook plus scanned derivations — the softmax/cross-entropy page is a
Phase 2 dependency; file it where you can find it in Month 04.

## Review

1. (Wk 09) Why must threshold selection (Exercise 4) happen on validation data, not test?
2. (Wk 08) Training logistic regression has no closed form. Why not (one line), and which
   property from Boyd's chapter still guarantees GD finds the optimum?
3. (Wk 07) Cross-entropy loss is the KL to what distribution, plus what constant?
4. (Wk 01) Compute TPR from a boolean prediction array and truth array in one NumPy line.
5. (Physics) Trigger efficiency and purity map onto which two ROC/PR quantities?
