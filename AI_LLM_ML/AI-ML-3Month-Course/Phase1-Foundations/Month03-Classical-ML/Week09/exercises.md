# Week 09 — Exercises

Work top to bottom. Setup (imports, data generation, constants, seeded RNGs, axes)
is given by the notebook; you write only the lines each exercise asks for.
All six exercises live in the notebook; nothing goes in files this week.

## E1 — Three roads to one fit

Fit the same degree-3 polynomial regression three ways on the provided 1D dataset:
(a) normal equations via `np.linalg.lstsq` on the polynomial design matrix,
(b) your own gradient descent (gradient $2X^\top(Xw - y)$, step size given),
(c) sklearn `LinearRegression(fit_intercept=False)` on the same features.
Print all three coefficient vectors.
Hint: build the design matrix with columns $x^0, x^1, x^2, x^3$ so all three see
identical features; run GD long enough (the notebook gives the iteration count).
Accept when: all three coefficient vectors agree pairwise to 1e-6.

## E2 — Bias–variance, measured

The truth $f(x) = \sin(2x)$ and noise $\sigma = 0.3$ are known. For each degree in
1–15: draw 300 fresh training sets of 30 points, fit each, and predict on a fixed
test grid. From the 300 predictions per grid point compute measured bias², variance,
and their sum + $\sigma^2$; plot all three vs degree alongside measured test error.
Hint: bias² at a grid point is `(preds.mean(axis=0) - f_true)**2`; variance is
`preds.var(axis=0)`; average both over the grid.
Accept when: the plot shows the canonical U-shape and the sum
bias² + variance + $\sigma^2$ tracks measured test error within a few percent at
every degree.

## E3 — Cross-validation from scratch

Implement 5-fold CV yourself: build the fold index arrays with NumPy, loop over
folds, fit the provided pipeline on the training folds, score MSE on the held-out
fold. Compare per-fold scores to `cross_val_score` using the same `KFold` splitter.
Hint: get identical folds by taking the `(train_idx, test_idx)` pairs from
`cv.split(X, y)` rather than inventing your own shuffle.
Accept when: your per-fold MSEs match sklearn's (sign-flipped) scores to 1e-10 on
all five folds.

## E4 — Leakage lab

On the provided small-n, wide-d dataset (60 examples, 200 noise features, weak
signal): (a) standardize the *whole* dataset first, then run 5-fold CV on the
model alone; (b) run 5-fold CV on a `Pipeline(StandardScaler, model)` instead.
Report both CV scores and the gap.
Hint: the leak is the full-data mean/std; with 200 features and 60 rows the
optimism is not subtle.
Accept when: version (a) scores measurably better than version (b) (the notebook
states the expected gap), and (b) is consistent with the no-signal truth.

## E5 — Ridge path

Sweep $\lambda$ over `np.logspace(-4, 4, 30)` on the provided polynomial problem.
Plot every coefficient's value vs $\lambda$ (log x-axis) and, on a second axis,
CV MSE vs $\lambda$. Select $\lambda^*$ by CV; verify both limits.
Hint: sklearn's `Ridge(alpha=lam)` — its `alpha` is our $\lambda$; standardize
inside a Pipeline.
Accept when: the chosen $\lambda^*$ minimizes CV MSE, coefficients at
$\lambda = 10^{-4}$ match plain least squares to 1e-3, and at $\lambda = 10^{4}$
all coefficient magnitudes are below 1e-2.

## E6 — Test-set discipline (synthesis)

Rerun the whole Week 09 workflow honestly: freeze a 25% test split first; use
5-fold CV on the remainder to choose degree and $\lambda$ jointly; refit the winner
on all train+val data; evaluate on the test set exactly once and compare that score
to the winning CV estimate (mean ± fold spread).
Hint: two nested `for` loops over (degree, lambda) and one `cross_val_score` per
pair is plenty — no fancy search tools needed yet.
Accept when: the test MSE falls within the winning configuration's CV mean ± one
fold-to-fold standard deviation, and the notebook contains exactly one call that
touches `X_test`.

## Review

1. (Week 08) You fit a line by minimizing squared error. State in one line why
   Gaussian measurement noise makes that the maximum-likelihood choice.
2. (Week 07) `np.linalg.lstsq` solves least squares without forming $X^\top X$.
   Which decomposition from Week 07 makes that possible, and why is it safer?
3. (Week 05) Your E1 gradient descent needs a step size. What happens to the
   iterates when the step size is too large, and how would you detect it from the
   loss curve?
4. (Week 06) The residual $y - Xw^*$ at the least-squares optimum is orthogonal to
   which of the four fundamental subspaces' spanning sets?
5. (Week 04) You seeded every RNG in this notebook. What is the one-command test
   that your results are actually reproducible from a fresh clone?
