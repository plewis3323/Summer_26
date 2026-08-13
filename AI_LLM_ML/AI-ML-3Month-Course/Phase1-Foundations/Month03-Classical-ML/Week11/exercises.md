# Week 11 — Exercises

Work top to bottom. Setup (imports, dataset generation, the cached MAGIC download,
seeded RNGs, constants, axes with labels) is given by the notebook; you write only
the lines each exercise asks for. E1–E3, E5, and E6 live in the notebook. E4 also
ships two files — `src/week11/tune_magic.py` (the CV tuning script) and
`models/magic_bdt.json` (the saved tuned model) — this week's deliverable and
Capstone 1's baseline.

## E1 — Stump to tree

Implement a depth-limited classification tree with Gini splits on the provided 2D
toy dataset: a `best_split` function that scans every feature and every candidate
threshold and returns the split with the largest gain, plus a recursive `grow`
capped at `max_depth` with majority-class leaves. Plot decision regions at depths
1, 3, and 10 and report train/test accuracy at each depth.
Hint: for one feature, sort its values — only midpoints between consecutive
distinct values can matter; score each candidate with the lesson §1.1 gain
$\Delta = I(\text{node}) - f_L I(\text{left}) - f_R I(\text{right})$, and check
your root split against the paper derivation you did on the 10-row table.
Accept when: depth-1 split matches the hand-derived best split and depth-10
visibly overfits (train/test gap reported).

## E2 — Forest variance

Train 200 unlimited-depth trees on bootstrap resamples of the provided dataset.
Plot the average single-tree test error and the ensemble-vote test error as
functions of the number of trees B; then repeat the whole experiment with
per-node feature subsampling (`max_features="sqrt"`) and overlay the curves.
Hint: one bootstrap replica is `idx = rng.integers(0, n, size=n)`; build the
ensemble curve by averaging the first B trees' predicted probabilities and
thresholding once at 0.5 — recomputing from stored per-tree predictions costs
nothing.
Accept when: error curves flatten with B and the subsampled forest beats plain
bagging on test error.

## E3 — 1D boosting from scratch

Implement the lesson §4 loop for squared loss on the provided noisy 1D function:
$F_0 = $ the mean of $y$, then M rounds of (compute residuals → fit
`DecisionTreeRegressor(max_depth=1)` to them → add it scaled by $\nu$). Plot the
ensemble prediction after 1, 10, and 100 rounds over the data.
Hint: keep the current predictions `F` as an array over the training inputs; the
whole loop is ~10 lines. To match sklearn, use identical $\nu$, M, and stump
depth in `GradientBoostingRegressor(learning_rate=nu, n_estimators=M, max_depth=1)`
— its default init is the mean, same as yours.
Accept when: staged predictions visibly refine and match sklearn
`GradientBoostingRegressor` (matched settings) to 1e-6.

## E4 — XGBoost on physics tabular

This one goes in files. Write `src/week11/tune_magic.py`: load the cached MAGIC
CSV, build the Week 10-style logistic baseline
(`Pipeline(StandardScaler, LogisticRegression)`) and an `XGBClassifier`, tune
`max_depth` and `learning_rate` over a small grid with `StratifiedKFold` CV and
early stopping, print a CV-AUC table (baseline row + winning BDT row + margin),
and save the winner to `models/magic_bdt.json` via `bdt.save_model`. The notebook
runs the script and loads the saved model back to confirm it predicts.
Hint: set `n_estimators=2000` and let `early_stopping_rounds=50` pick the round
count, as in lesson §5; loop the folds yourself (Week 09 E3 style) so each fold
can pass its held-out part as `eval_set`; the grid `max_depth` in {3, 4, 6} ×
`learning_rate` in {0.03, 0.1, 0.3} is plenty.
Accept when: CV AUC beats your Week 10 logistic-regression baseline by a stated
margin.

## E5 — Importance autopsy

Refit the E4 winner's settings on a doctored MAGIC matrix: append an exact
duplicate of the top-ranked feature and a pure-noise column. Show impurity
importances (`feature_importances_`) and permutation importances (held-out data)
side by side in one labeled bar chart, and summarize the disagreement in a
markdown cell of at most 3 lines.
Hint: doctor with `np.column_stack([X, X[:, [j]], rng.normal(size=(n, 1))])`;
run `permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)`.
Accept when: the demo shows impurity importance splitting credit across
duplicates while permutation exposes the noise feature, summarized in ≤3 lines.

## E6 — Overfitting the BDT way

Reproduce the classic HEP failure on a starved sample: the notebook fixes a
300-event training draw from MAGIC. Train XGBoost (no early stopping) across
depths {2, 4, 6, 8, 10} at fixed rounds, and across rounds up to 2000 at fixed
depth 6; plot train and test AUC against depth and against rounds, and mark
where the test curves stop improving.
Hint: `eval_set=[(X_tr, y_tr), (X_te, y_te)]` plus `bdt.evals_result()` hands you
the full per-round AUC curves from a single fit — no retraining loop needed for
the rounds plot.
Accept when: the divergence plot identifies the depth/rounds where
generalization stops improving.

## Review

1. (Week 09) The lesson calls the OOB score "free validation." Which Week 09 tool
   does it approximate, and which Week 09 rule does it *not* excuse you from?
2. (Week 10) Forest `predict_proba` outputs are vote fractions. Which Week 10
   diagnostic decides whether they deserve to be called probabilities, and what
   two repairs exist if they don't?
3. (Week 08) Entropy impurity is the entropy of which distribution? Compute it,
   in bits, for nodes with class fractions (1/2, 1/2) and (1, 0).
4. (Week 08) XGBoost keeps the second-order Taylor term that plain gradient
   boosting drops. In the language of Week 08's optimizer race, what does
   curvature information buy, and which of your hand-rolled optimizers
   approximated it per-coordinate?
5. (Week 03) Write one NumPy line that draws the row indices of a bootstrap
   resample of an n-row dataset.
