# Week 11 — Trees & Ensembles

Every HEP analysis you've read that says "a BDT was trained" traces back to this week —
including why that culture arose and what it gets right about tabular data.

## Objectives

- Explain how a decision tree chooses splits (Gini/entropy) and why single trees overfit.
- Explain bagging as variance reduction and random forests' decorrelation trick.
- Derive gradient boosting as functional gradient descent and implement it in 1D.
- Train and tune XGBoost/LightGBM competently on tabular physics data.
- Critique feature importances: impurity vs permutation, and where each lies.

## Core material (~3 hrs)

- StatQuest: the decision tree, random forest (parts 1–2), gradient boost (parts 1–4),
  and XGBoost series. Fast, and genuinely the clearest treatment of the split math.
- scikit-learn user guide: "Decision Trees" and "Ensemble methods" — read the
  bias-variance framing of bagging vs boosting, and the permutation-importance page
  (including the correlated-features caveat).
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" — §2 only (the
  regularized objective and its second-order approximation).
- Bishop, *PRML* Chapter 14 (combining models), skim for the committee/boosting framing.
- HEP context: skim the TMVA Users Guide introduction — the tool a decade of analyses
  were built on; note what it standardized (and what it hid).

## Derivations (paper first)

- Information gain: compute by hand the best of three candidate splits on a 10-row toy
  table, with both Gini and entropy.
- Bagging: show variance of an average of B correlated estimators
  → ρσ² + (1 − ρ)σ²/B; identify what random feature selection attacks (ρ).
- Gradient boosting as functional GD: for squared loss show the negative gradient is
  the residual; write the generic algorithm for any differentiable loss; note where
  the learning rate enters.
- XGBoost's objective: second-order Taylor expansion and the optimal leaf weight
  wⱼ* = −Gⱼ/(Hⱼ + λ).

## Exercises (built when the week starts)

1. **Stump to tree.** Implement a depth-limited decision tree (Gini splits) on a 2D toy;
   plot decision regions at depths 1, 3, 10.
   Accept when: depth-1 split matches the hand-derived best split and depth-10 visibly overfits (train/test gap reported).
2. **Forest variance.** Train 200 trees on bootstrap resamples; plot single-tree vs
   ensemble test error vs B; add feature subsampling and compare.
   Accept when: error curves flatten with B and the subsampled forest beats plain bagging on test error.
3. **1D boosting from scratch.** Gradient boosting with stump learners on a noisy 1D
   function; plot the ensemble after 1, 10, 100 rounds.
   Accept when: staged predictions visibly refine and match sklearn `GradientBoostingRegressor` (matched settings) to 1e-6.
4. **XGBoost on physics tabular.** Train XGBoost or LightGBM on the MAGIC dataset
   (preview of Capstone 1); tune depth, learning rate, and early stopping via CV.
   Accept when: CV AUC beats your Week 10 logistic-regression baseline by a stated margin.
5. **Importance autopsy.** Add a duplicated feature and a random-noise feature; compare
   impurity vs permutation importances.
   Accept when: the demo shows impurity importance splitting credit across duplicates while permutation exposes the noise feature, summarized in ≤3 lines.
6. **Overfitting the BDT way.** Reproduce the classic HEP failure: too-deep trees on
   too-few events; show train/test divergence vs depth and rounds.
   Accept when: the divergence plot identifies the depth/rounds where generalization stops improving.

## Deliverable

Completed notebook, scanned derivations (splits, bagging variance, functional GD,
XGBoost leaf weights), and a saved tuned model + CV script — Capstone 1's baseline.

## Review

1. (Wk 09) Bagging attacks variance, boosting attacks bias. Place each on the Week 09
   bias–variance plot.
2. (Wk 08) Gradient boosting is GD in what space, and what plays the role of the step size?
3. (Wk 10) Why is AUC the right comparison in Exercise 4 rather than accuracy?
4. (Wk 06) Trees are invariant to monotone feature transforms. Why does PCA-rotating the
   inputs still sometimes help or hurt them?
5. (Physics) Why did BDTs dominate HEP for a decade before deep learning did — what
   property of collider ntuples suits trees?
