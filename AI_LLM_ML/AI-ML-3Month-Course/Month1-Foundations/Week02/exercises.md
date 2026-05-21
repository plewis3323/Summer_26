# Week 02 — Exercises

## E1 — Linear regression by hand (easy, 20 min)

Derive the OLS normal equations from scratch on paper. Then:

1. Implement `ols(X, y)` using `numpy.linalg.lstsq` and using explicit `np.linalg.solve(X.T @ X, X.T @ y)`. Confirm agreement.
2. Generate synthetic data `y = 2x - 1 + ε` with ε ~ N(0, 0.5); recover the parameters.
3. Write a 3-line argument for why `np.linalg.lstsq` is numerically safer than `inv(X.T @ X) @ X.T @ y`.

## E2 — Ridge regression and multicollinearity (easy, 30 min)

Generate `X` with a pair of near-identical columns (`X[:, 1] = X[:, 0] + tiny_noise`). Fit OLS and ridge for λ ∈ {0, 0.1, 1, 10}. Plot the coefficients. Observe shrinkage and coefficient-splitting behavior. Explain in 3 sentences.

## E3 — Logistic regression from scratch (medium, 45 min)

Implement `fit_logistic(X, y, lr=0.1, steps=5000)` using only numpy. No scikit-learn. Include L2 regularization.

- Gradient descent; plot loss vs step.
- On a 2D toy dataset, plot decision boundary.
- Compare learned weights to `sklearn.linear_model.LogisticRegression(penalty='l2', solver='lbfgs')`.

**Accept when:** coefficients match sklearn's to 3 decimal places.

## E4 — Classification metrics (easy, 30 min)

**Data:** sklearn's `make_classification(n_samples=5000, weights=[0.95, 0.05])`.

1. Train `LogisticRegression` and `RandomForestClassifier`.
2. Compute accuracy, balanced accuracy, ROC-AUC, PR-AUC, Brier score, log-loss.
3. Plot ROC and PR curves on the same axes for both models.
4. Discuss: which metric favored which model, and why.

## E5 — Decision-tree intuition (medium, 45 min)

**Data:** `sklearn.datasets.load_wine()`.

1. Fit `DecisionTreeClassifier` with `max_depth=1, 2, …, 20`. Plot train and test accuracy.
2. Identify the overfitting regime.
3. Refit at `max_depth=3` with `random_state=0, 1, 2, 3, 4`. Plot five different tree structures (use `sklearn.tree.plot_tree`). Comment on variance.

## E6 — Random forest with OOB (medium, 30 min)

**Data:** `sklearn.datasets.fetch_covtype()` subsampled to 50k rows.

1. Fit `RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)`.
2. Report OOB score and test-set accuracy. They should agree to ~1%.
3. Plot permutation feature importance (not the default `.feature_importances_`; use `sklearn.inspection.permutation_importance`).

## E7 — XGBoost on HEPMASS (medium → hard, 60 min)

**Data:** HEPMASS (https://archive.ics.uci.edu/dataset/347/hepmass). Use the 1000-particle subset (`1000_train.csv` / `1000_test.csv`).

1. Load and split. Standardize features (XGBoost doesn't require it, but it makes the logistic baseline fairer).
2. Train `XGBClassifier` with defaults; record AUC.
3. Train `LogisticRegression(penalty='l2')` baseline; record AUC.
4. Do a 30-trial `optuna` search over the hyperparameter space in `README.md §9`. Use 5-fold CV AUC as the objective.
5. Report best CV AUC, then final test AUC.
6. Plot feature importance (gain-based) and permutation importance. Do they agree? (They often don't.)

**Accept when:** tuned XGBoost AUC beats logistic baseline by ≥5 points; test AUC within 0.5 points of best CV AUC.

## E8 — Calibration and Platt scaling (medium, 45 min)

1. Take any XGBoost model from E7. Plot a reliability curve (`sklearn.calibration.calibration_curve`) on the test set.
2. Apply `CalibratedClassifierCV` with method="sigmoid" (Platt) and method="isotonic" (isotonic regression) on a held-out calibration split.
3. Replot reliability. Show that calibration is improved.
4. Check Brier score before and after.

## E9 — Leakage detection (hard, 60 min)

**Data:** a version of HEPMASS where you've deliberately injected a leaky feature (e.g. concat a column equal to `y + 0.01 * np.random.randn(n)`).

1. Train any model; observe AUC ~0.99.
2. Diagnose: permutation importance and a correlation-with-label check.
3. Remove the leak, retrain, confirm AUC drops to realistic value.
4. Write a 150-word "leakage post-mortem."

**Accept when:** you can explain, specifically, how you'd catch this in a real thesis analysis.

## E10 — GroupKFold for run-correlated data (hard, 45 min)

**Data:** synthesize. Make 100 "runs" each with 1000 events. Add a per-run random offset to a feature so that within a run, features are correlated.

1. Train XGBoost with a naive `KFold(shuffle=True)` split; report AUC.
2. Train with `GroupKFold(groups=run_id)`; report AUC.
3. The second should be lower and the first is the optimistic one. Explain why and when you'd choose one over the other in a real thesis.

---

## Solution hints

- E1 — `lstsq` uses SVD internally and handles rank-deficiency gracefully.
- E2 — ridge splits coefficients across correlated features instead of blowing up.
- E3 — if your logistic-from-scratch isn't converging, check you're using `σ(z)(1-σ(z))` and your learning rate isn't 10.
- E4 — imbalanced → PR-AUC is more honest than ROC-AUC.
- E5 — overfitting shows up as train-accuracy asymptoting to 1 while test plateaus.
- E6 — OOB is almost free; use it as a sanity check for your test split.
- E7 — for the first XGBoost run, use `early_stopping_rounds=50` on a val split; don't tune `n_estimators`.
- E8 — XGBoost probabilities are usually overconfident. Isotonic fixes more than Platt for large datasets.
- E9 — the fastest leak detector is a single-feature AUC. Any feature alone with AUC > 0.98 is suspect.
- E10 — in a real analysis, split by run, by fill, or by period. Never random-shuffle when events share detector state.
