# Week 09 — The ML Frame + Linear Regression

Supervised learning is curve fitting with an adversary: the data you haven't seen yet.
This week builds the frame — loss, risk, generalization — and the discipline that keeps
you honest.

## Objectives

- State the supervised-learning problem formally: data, hypothesis class, loss, empirical vs true risk.
- Derive the bias–variance decomposition and demonstrate it empirically.
- Fit linear regression three ways — normal equations, your Week 08 GD, scikit-learn — and reconcile them.
- Run k-fold cross-validation from scratch and explain what it estimates (and doesn't).
- Spot data leakage and explain why the test set is touched exactly once.

## Core material (~3 hrs)

- Bishop, *PRML* (in `references/`): §1.1 (polynomial fitting example — read closely,
  it is the whole course in miniature) and §3.1–3.2 (linear regression, bias–variance).
- VanderPlas, *Python Data Science Handbook*, Chapter 5: the sections on hyperparameters,
  model validation, and linear regression — this is your scikit-learn API on-ramp.
- scikit-learn user guide: "Cross-validation" — including the pitfalls section.

## Derivations (paper first)

- Normal equations: minimize ‖Xw − y‖² → XᵀXw = Xᵀy; connect to the Week 05 projection
  picture and the Week 07 Gaussian likelihood.
- Bias–variance: derive E[(y − ŷ)²] = bias² + variance + irreducible noise for squared
  loss, being explicit about what each expectation is over.
- Ridge: add the L2 penalty and show the modified solution (XᵀX + λI)⁻¹Xᵀy; connect to
  the Week 07 MAP derivation.

## Exercises (built when the week starts)

1. **Three roads to one fit.** Normal equations, your Week 08 GD, and sklearn
   `LinearRegression` on the same polynomial features.
   Accept when: all three coefficient vectors agree to 1e-6.
2. **Bias–variance, measured.** Fit polynomial degrees 1–15 to many resampled datasets
   from a known truth; plot measured bias², variance, and their sum vs degree.
   Accept when: the plot reproduces the canonical U-shape and the sum tracks measured test error.
3. **CV from scratch.** k-fold CV with your own index splitting; match
   `cross_val_score` fold for fold.
   Accept when: per-fold scores match sklearn's to 1e-10 with the same fixed splits.
4. **Leakage lab.** Standardize features *before* splitting vs inside each fold on a
   small-N wide-p dataset; quantify the optimism.
   Accept when: the leaky pipeline's CV score is optimistic by a measurable margin and the fix (Pipeline) closes the gap.
5. **Ridge path.** Sweep λ; plot coefficient shrinkage and CV error; pick λ by CV.
   Accept when: chosen λ minimizes CV error and the λ→0 / λ→∞ limits match theory (OLS / zero).
6. **Test-set discipline.** Hold out a final test set at the start; touch it once at the
   end; compare its score to the CV estimate.
   Accept when: test score falls within the CV score's fold-to-fold spread, and the notebook contains exactly one test-set evaluation.

## Deliverable

Completed notebook plus scanned derivations. Keep the bias–variance experiment code —
Capstone 1's writeup will reuse the plot style.

## Review

1. (Wk 08) Which optimizer would you pick for Exercise 1's GD fit and what step size —
   from the quadratic analysis, not trial and error?
2. (Wk 07) State in one line why ridge = MAP with a Gaussian prior.
3. (Wk 06) The normal equations square the condition number. What was the numerically
   sane alternative from Week 06?
4. (Wk 05) In the projection picture, what subspace does the residual y − Xŵ live in?
5. (Physics) Blind analyses freeze cuts before unblinding. Which rule this week is the
   same move?
