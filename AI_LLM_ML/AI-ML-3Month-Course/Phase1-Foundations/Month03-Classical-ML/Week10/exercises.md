# Week 10 — Exercises

Work top to bottom. Setup (imports, dataset generation, seeded RNGs, constants,
axes with labels) is given by the notebook; you write only the lines each exercise
asks for. All six exercises live in the notebook; nothing goes in files this week.

## E1 — Logistic regression from scratch

Implement the derived gradient `grad = X.T @ (sigmoid(X @ w) - y)` and train with
your Week 08 Adam on the provided two-Gaussian dataset. Verify the gradient against
finite differences (Week 05 style) before training, then compare coefficients to
sklearn `LogisticRegression` with matched (near-zero) regularization.
Hint: implement `sigmoid` with `1 / (1 + np.exp(-z))`; include the bias by
appending a ones column, as in Week 09.
Accept when: the finite-difference gradient check passes at 1e-6 and your
coefficients match sklearn's to 1e-4.

## E2 — Decision boundary

Using the E1 model: plot probability contours of $\sigma(w^\top x + b)$ over a 2D
grid with the data overlaid, and draw the decision boundary at thresholds 0.5, 0.2,
and 0.9.
Hint: the boundary at threshold $t$ is the line $w^\top x + b = \log\frac{t}{1-t}$;
`plt.contour` with `levels=[t]` on the probability grid draws it for you.
Accept when: the plotted 0.5 contour matches the analytic line $w^\top x + b = 0$
(overlay both), and the three thresholds produce three parallel lines ordered as
expected.

## E3 — ROC and PR from scratch

Sweep the threshold over the sorted unique scores of a provided validation set;
at each, compute TPR, FPR, precision, recall from your own confusion-matrix counts.
Plot your ROC and PR curves over sklearn's, and compute AUC by the trapezoid rule
(`np.trapezoid`) on your ROC points.
Hint: vectorize each threshold's counts with boolean masks — for example
`np.sum((p >= t) & (y == 1))` — inside a single loop over thresholds.
Accept when: your AUC matches `roc_auc_score` to 1e-6 and your PR points coincide
with `precision_recall_curve`'s set.

## E4 — Trigger working point

On the provided imbalanced (1:100) signal/background dataset: train a
class-weighted logistic regression; on the validation split choose (a) the
threshold giving 90% signal efficiency and (b) the threshold maximizing
$S/\sqrt{S+B}$. Report the confusion matrix, efficiency, and purity for each on
the held-out set.
Hint: for (a), take the first threshold where TPR from `roc_curve` reaches 0.90;
for (b), loop thresholds and count S and B with masks, as in the lesson's worked
example.
Accept when: the held-out efficiency at working point (a) is within 1 percentage
point of 90%, and both working points' numbers come from data not used to choose
the thresholds.

## E5 — Calibration

Draw 10-bin reliability diagrams for (a) your E1 logistic model and (b) the
provided overconfident classifier, on validation data. Wrap (b) in
`CalibratedClassifierCV` (Platt, `method="sigmoid"`), refit, and re-plot. Report
Brier scores before and after.
Hint: `calibration_curve(y, p, n_bins=10)` returns the (observed fraction, mean
predicted) pairs — plot them against the diagonal.
Accept when: the post-calibration curve is visibly nearer the diagonal in the plot
and the Brier score improves (decreases) for model (b).

## E6 — Softmax three-class (synthesis)

Extend E1 to three Gaussian blobs: implement softmax and the cross-entropy loss,
train the $3 \times (d+1)$ weight matrix with your Adam, and confirm the derived
identity $\partial\ell/\partial z = p - y$ numerically at 5 random points. Report
test accuracy against the notebook's stated baseline.
Hint: stabilize softmax by subtracting `z.max()` before exponentiating — the lesson
showed shifts don't change the output; one-hot the labels with an identity-matrix
row lookup, `np.eye(3)[y]`.
Accept when: numeric and analytic $\partial\ell/\partial z$ agree to 1e-6 at all 5
points and test accuracy exceeds the notebook's baseline.

## Review

1. (Week 09) You picked E4's thresholds on validation data and reported on held-out
   data. Which Week 09 rule is that, and what number would you contaminate by
   skipping it?
2. (Week 08) Why does minimizing binary cross-entropy also minimize the KL
   divergence from the label distribution to the model — what term differs between
   them, and why doesn't it matter for training?
3. (Week 08) E1 trains with Adam rather than plain GD. Name the two moment
   estimates Adam keeps and what each one buys.
4. (Week 06) The decision boundary $w^\top x + b = 0$ is what geometric object in
   $\mathbb{R}^d$, and what is the vector $w$'s relationship to it?
5. (Week 03) Compute TPR from boolean arrays `pred` and `truth` in one NumPy line.
