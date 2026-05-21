# Week 02 — Classical ML: Regression, Trees, BDTs

## Learning objectives

By the end of this week you will:

1. Derive linear regression from squared-error loss and logistic regression from cross-entropy, by hand, end-to-end.
2. Reason about regularization in both statistical and geometric language: L1 sparsity, L2 shrinkage, the "constraint geometry" picture.
3. Understand decision trees as piecewise-constant regressors/classifiers, why they overfit, and why ensembling fixes it.
4. Know the difference between bagging (random forests) and boosting (AdaBoost → gradient boosting → XGBoost/LightGBM), and which to reach for when.
5. Train, tune, and evaluate an XGBoost model on a HEP-flavored tabular dataset, and know how to read an ROC curve without fooling yourself.
6. Know why BDTs have been the LHC's default ML workhorse for ~20 years, and where they are finally being displaced by deep learning.

Mathematically heaviest week of Month 1. If something doesn't click, derive it on paper. Don't just read.

## 1. The supervised-learning frame

Given a dataset {(xᵢ, yᵢ)}, where xᵢ ∈ ℝᵈ and yᵢ is either a real number (regression) or a class label (classification), learn a function f̂: ℝᵈ → ℝ (or → {0, 1, …, K}) that minimizes an **expected loss** over the data-generating distribution P(x, y):

$$
R(f) = \mathbb{E}_{(x, y) \sim P}\bigl[ \ell(f(x), y) \bigr].
$$

We can't compute R directly because we don't know P. We compute the **empirical risk**:

$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), y_i),
$$

and minimize that over some function class 𝓕. Everything else in this week is (1) picking ℓ, (2) picking 𝓕, (3) making sure min over 𝓕 of R̂ is a useful proxy for min of R.

The gap between R and R̂ is generalization error. Every technique in this week — regularization, cross-validation, ensembling — is in the service of reducing that gap.

## 2. Linear regression, derived

**Model:** f(x; w, b) = wᵀx + b.
**Loss:** squared error, ℓ(ŷ, y) = (ŷ - y)².

Stack rows into X ∈ ℝⁿˣᵈ (with a column of ones absorbing b) and y ∈ ℝⁿ. The empirical risk is

$$
\hat{R}(w) = \frac{1}{n} \|Xw - y\|^2.
$$

Differentiate, set to zero:

$$
\nabla_w \hat{R}(w) = \frac{2}{n} X^T(Xw - y) = 0 \implies X^T X w = X^T y.
$$

Normal equations. If XᵀX is invertible, ŵ = (XᵀX)⁻¹Xᵀy. You've probably done this in your linear-algebra course.

**Ridge regression (L2):** add λ‖w‖². New normal equations:

$$
(X^T X + \lambda I) w = X^T y,
$$

always invertible for λ > 0. Geometrically: L2 is a sphere in w-space; the constrained minimum is pushed toward origin. Good for multicollinearity.

**Lasso (L1):** add λ‖w‖₁. No closed form. Solved by coordinate descent or proximal methods. Geometrically: L1 is a hypercube with corners on the axes; the constrained minimum often lies on an axis, giving exact zeros. This is why Lasso does feature selection and ridge does not.

**Where this breaks down:** if the true function isn't linear, no amount of L2 saves you. Feature engineering ("add x² as a feature") trades function-class expressiveness for computational tractability.

## 3. Logistic regression, derived

Binary classification, y ∈ {0, 1}. Model the probability:

$$
p(y = 1 \mid x) = \sigma(w^T x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

**Loss:** binary cross-entropy, a.k.a. negative log-likelihood:

$$
\ell(\hat p, y) = -\bigl[ y \log \hat p + (1-y) \log(1 - \hat p) \bigr].
$$

Why this loss? Bernoulli likelihood:

$$
p(y \mid x; w) = \hat p^y (1 - \hat p)^{1-y}.
$$

Take log, negate, and you have binary cross-entropy. The MLE for a logistic regression *is* exactly minimizing cross-entropy.

**Gradient.** Using σ'(z) = σ(z)(1 - σ(z)):

$$
\frac{\partial \ell}{\partial w} = (\hat p - y) \, x.
$$

Clean. This is why logistic regression is usually the first classification thing anyone teaches: the gradient is literally "residual times input."

**Optimization:** no closed form. Use Newton-Raphson (**IRLS**) or gradient descent. `scikit-learn`'s `LogisticRegression` uses L-BFGS by default for small problems, SAGA for large sparse.

**Multiclass** (K classes): model logits zₖ = wₖᵀx + bₖ; softmax over logits:

$$
\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
$$

Cross-entropy loss generalizes:

$$
\ell = -\sum_{k=1}^K \mathbb{1}[y = k] \log \mathrm{softmax}(z)_k.
$$

We will derive the softmax gradient in Week 03 when we start doing backprop.

### Numerical stability aside: log-sum-exp trick

Naively computing softmax on (100, 200, 300) overflows exp. The trick:

$$
\log \sum_k e^{z_k} = \max_k z_k + \log \sum_k e^{z_k - \max_k z_k}.
$$

Every library does this for you, but you should know it exists — when you roll your own softmax in Week 06, you'll hit it.

## 4. Evaluation metrics for classification

### ROC and AUC
- Threshold the predicted probability at τ; compute TPR = TP/(TP+FN) and FPR = FP/(FP+TN); sweep τ.
- ROC curve plots TPR vs FPR.
- AUC = area under this curve = P(model scores a random positive higher than a random negative).
- AUC is threshold-free. Useful, but can mislead on imbalanced data.

### PR curves
- Plot precision = TP/(TP+FP) vs recall = TPR.
- Better than ROC when positives are rare (e.g. 0.1% signal among 99.9% background, as in many HEP analyses).

### Calibration
- A well-calibrated classifier's 70% predictions should actually be right ~70% of the time. Plot predicted probability vs empirical frequency.
- Logistic regression is calibrated by design. BDTs and neural nets often are not; post-hoc **Platt scaling** or **isotonic regression** fixes this.

### Brier score
- Mean squared error on probabilities: BS = (1/n) Σ (p̂ᵢ - yᵢ)². Penalizes miscalibration.

### Confusion matrix
- The raw counts underlying everything. Always look at it before trusting any single metric.

**HEP gotcha:** the physics idea of "efficiency" = TPR on signal. "Purity" = precision = TP/(TP+FP). These are the same metrics under different names. Don't let the vocabulary bait you.

## 5. Train / val / test — and why CV

The golden rule: **you may not use the test set until the very last step.** Not for hyperparameter tuning, not for "quick checks." One look, once. Because human beings leak information through their decisions.

- **Train set.** Model fits here.
- **Validation set.** Hyperparameters tuned here.
- **Test set.** Final evaluation. Reported number.

When data is small, a fixed val split wastes data. **k-fold cross-validation** partitions the training data into k parts, trains on k-1, validates on the held-out fold, and averages. Standard k = 5 or 10.

**Nested CV** when you're tuning hyperparameters *and* evaluating. Outer loop = evaluation; inner loop per outer fold = hyperparameter tuning. Expensive but honest. Use when dataset is small enough that a single split's variance is large.

## 6. Decision trees

A decision tree recursively partitions the feature space and predicts a constant on each partition. For regression, the per-leaf prediction is the mean of training targets in that leaf; for classification, the majority class (or class probabilities).

**Growing rule:** at each node, pick the feature and split point that maximize "purity gain" or equivalently minimize some impurity measure.

**Impurity measures:**
- Classification: Gini impurity `G(S) = Σ pₖ(1 - pₖ)`, or entropy `H(S) = -Σ pₖ log pₖ`.
- Regression: variance (MSE) reduction.

**Why trees overfit.** Without regularization, a tree can grow until each leaf has one training example and training error is zero. Generalization is terrible. Regularization: limit `max_depth`, `min_samples_leaf`, `min_impurity_decrease`, or prune post-hoc.

**Pros.** Interpretable. Handle non-linearity, interactions, mixed feature types, missing values (with surrogate splits). No scaling required.

**Cons.** High variance; small data changes → very different tree. Piecewise-constant, so no extrapolation.

Ensembling fixes the variance problem.

## 7. Bagging and random forests

**Bagging** = bootstrap aggregating. Fit `M` trees on bootstrap samples of the training data; for prediction, average (regression) or vote (classification).

**Random forests** = bagging + random feature subsampling at each split. Extra variance reduction because trees are now decorrelated.

Key parameters: `n_estimators` (M), `max_features` (~√d for classification, d/3 for regression), `max_depth` (often unbounded), `min_samples_leaf`.

**Why it works.** Variance of mean of M IID estimators is σ²/M. Bagging makes the estimators approximately IID (not exactly, because they share the dataset). Bias is unchanged; variance goes down.

**OOB error.** Each tree skips ~37% of the training data (the complement of the bootstrap sample). Averaged predictions on those held-out points give a free cross-validation estimate.

## 8. Boosting

Boosting is sequential, not parallel. Each tree tries to correct the errors of the ensemble-so-far.

### AdaBoost (historical)
Weight misclassified points higher and fit a new weak learner. Exponential loss. Mostly of historical interest now.

### Gradient boosting (the real thing)
Stated as: greedy coordinate descent in *function space*. At each step m, fit a weak learner h_m to the **negative gradient of the loss** with respect to the current prediction F_{m-1}:

$$
r_i^{(m)} = -\left. \frac{\partial \ell(F(x_i), y_i)}{\partial F(x_i)} \right|_{F = F_{m-1}}, \quad h_m \approx r^{(m)}.
$$

Update:
$$
F_m = F_{m-1} + \eta \, h_m,
$$
where η is the learning rate (shrinkage). Small η, many trees.

For squared error, `r = y - F(x)`, so you're fitting trees to residuals. For log-loss, `r = y - σ(F(x))`, so you're fitting trees to calibrated residuals.

### XGBoost innovations
- **Second-order info.** Use both gradient and Hessian; Newton-style updates per leaf.
- **Regularized objective.** Adds penalty on tree complexity (number of leaves and leaf-value magnitudes).
- **Smart split finding.** Approximate quantile sketches for large datasets; weighted quantile sketch for boosted objective.
- **Missing-value handling.** Routes missing to best direction at split time.
- **System design.** Parallel construction, cache awareness, out-of-core for data > memory.

### LightGBM
- **Histogram-based splits.** Bin features once; fast repeated splits.
- **Leaf-wise growth.** Grows the leaf with max loss reduction (XGBoost grows level-wise). Deeper-but-narrower trees; risk of overfit on small data.
- **GOSS and EFB.** Two sampling tricks for speed.

### When to use which
- **Start with XGBoost.** Mature, portable, strong defaults.
- **Switch to LightGBM** for very large datasets (10⁷+ rows) or when training time matters.
- **CatBoost** when you have lots of categorical features and don't want to one-hot encode.

## 9. Hyperparameter tuning

Don't grid-search a 6-dim grid. Use random search or Bayesian optimization (`optuna`).

Typical XGBoost search space for HEP classification:
- `n_estimators`: 200–2000 (paired with early stopping)
- `learning_rate`: 0.01–0.2 (log-uniform)
- `max_depth`: 3–10
- `min_child_weight`: 1–10
- `subsample`: 0.6–1.0
- `colsample_bytree`: 0.6–1.0
- `reg_alpha`: 0–10 (log-uniform)
- `reg_lambda`: 0–10 (log-uniform)

Always pair `n_estimators` with `early_stopping_rounds` on a validation set. Don't tune it directly — let early stopping choose.

**Cross-validation for tuning.** Use `xgboost.cv` or `sklearn.model_selection.cross_val_score` inside `optuna.Study.optimize`. Don't touch test set.

## 10. Why HEP loves BDTs (and is slowly leaving)

BDTs dominated LHC analyses ~2012–2022 because:
1. **Strong on tabular features.** LHC analyses have hand-engineered features (invariant masses, angles, isolation, MET) — tabular's home turf.
2. **Fast training, fast inference.** No GPU needed. Microseconds per event at inference.
3. **Robust to feature scales and small data.** Common in dedicated analyses with ~10⁵–10⁶ events.
4. **Interpretable enough.** Feature importance and SHAP values carry weight in collaboration reviews.
5. **TMVA** shipped them with ROOT, making them effectively default.

Where BDTs are losing ground:
- **Raw calorimeter images.** CNNs and transformers crush hand-engineered shower-shape features.
- **Jets as particle clouds.** ParticleNet / Particle Transformer exploit permutation-invariant structure BDTs can't see.
- **Tracking at HL-LHC.** Exa.TrkX GNNs outperform combinatorial track finding + BDT.
- **End-to-end pipelines.** Feature engineering is the expensive part; DL bakes it into the model.

BDTs are not obsolete. For your thesis they're still the **right default** on tabular analyses. Know them well.

## 11. Common pitfalls this week

- **Leaky features.** Including a feature that encodes the label (e.g. a post-selection cut variable). Symptom: AUC = 0.999. Always suspect.
- **Train/test contamination.** The same event ID in both splits. For HEP, when events are correlated (e.g. by run), split by run.
- **Imbalanced classes.** 99% background; your classifier "wins" by always predicting background. Use class weights, stratified splits, PR curves.
- **Leaking through normalization.** Fitting `StandardScaler` on train+test contaminates. Use `sklearn.pipeline.Pipeline` so fit/transform order is explicit.
- **Tuning on test set.** You'll be tempted. Don't. Create a val set from the train set.
- **Overfitting to a single seed.** Run 3–5 seeds, report mean ± std.
- **Misreading AUC.** AUC 0.80 is a *ranking* statement, not a claim about any specific operating point. Pair with a chosen (FPR, TPR) pair.

## 12. Connections to your HEP work

This week slots directly into your thesis. Concrete suggestions:

- Take an existing BDT-based analysis at sPHENIX (there are several for π⁰ identification and jet reconstruction) and re-implement a small version in `XGBoost`. Compare AUC.
- Plot the XGBoost feature importance alongside your physics intuition about which shower-shape variables should matter. Mismatches are interesting.
- Train a classifier to separate π⁰ → γγ (merged) from single photons using shower-shape variables only. You'll redo this with CNNs and images in Week 04 — great comparison.

## 13. Self-check

- Derive the MLE for logistic regression in three lines.
- Explain the geometric picture of L1 vs L2 regularization.
- Given class imbalance 1:1000, name one appropriate metric and one inappropriate one.
- State the XGBoost objective including tree-complexity regularization terms.
- Explain why bagging reduces variance but not bias.
- What's "early stopping" in boosting, and why doesn't it replace regularization?

When these are automatic, continue to the exercises and project.
