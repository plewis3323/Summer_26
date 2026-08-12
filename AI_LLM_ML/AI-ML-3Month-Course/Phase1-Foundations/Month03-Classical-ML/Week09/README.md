# Week 09 — The ML Frame + Linear Regression

Supervised learning is curve fitting with an adversary: the data you haven't seen yet.
This week builds the frame — loss, risk, generalization — and the discipline that keeps
you honest. scikit-learn enters here; the lesson teaches its fit/predict pattern from
zero.

## Objectives

- State the supervised-learning problem formally: data, hypothesis class, loss, empirical vs true risk.
- Derive the bias–variance decomposition and demonstrate it empirically.
- Fit linear regression three ways — normal equations, your Week 05/08 GD, scikit-learn — and reconcile them.
- Run k-fold cross-validation from scratch and explain what it estimates (and doesn't).
- Spot data leakage and explain why the test set is touched exactly once.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: the ML frame, linear regression three
  ways, the full bias–variance derivation, CV, leakage, ridge, and the scikit-learn
  API from zero.
- VanderPlas, *Python Data Science Handbook*, Chapter 5: the sections on hyperparameters,
  model validation, and linear regression — a second pass on the scikit-learn API.
- Bishop, *PRML* (in `references/`): §1.1 (polynomial fitting example — read closely,
  it is the whole course in miniature) and §3.1–3.2 (linear regression, bias–variance).
- scikit-learn user guide: "Cross-validation" — including the pitfalls section.

## Derivations (paper first)

- Normal equations: minimize ‖Xw − y‖² → XᵀXw = Xᵀy; connect to the Week 06 projection
  picture and the Week 08 Gaussian likelihood.
- Bias–variance: derive E[(y − ŷ)²] = bias² + variance + irreducible noise for squared
  loss, being explicit about what each expectation is over (the lesson walks it;
  redo it cold).
- Ridge: add the L2 penalty and show the modified solution (XᵀX + λI)⁻¹Xᵀy; connect to
  the Week 08 MAP derivation.

## Exercises

See `exercises.md` (notebook built from it when the week starts, per
`NOTEBOOK_RULES.md`). Six exercises: reconcile three linear-regression solvers,
measure the bias–variance decomposition by simulation, match hand-rolled k-fold CV to
sklearn fold-for-fold, produce and repair leakage, trace the ridge path, and run one
honest end-to-end degree selection with exactly one test-set touch.

## Deliverable

Completed notebook plus scanned derivations. Keep the bias–variance experiment code —
Capstone 1's writeup will reuse the plot style.

## Review

1. (Wk 08) Which optimizer would you pick for the GD fit in E1 and what step size —
   from the quadratic analysis, not trial and error?
2. (Wk 08) State in one line why ridge = MAP with a Gaussian prior.
3. (Wk 07) The normal equations square the condition number. What was the numerically
   sane alternative from Week 07?
4. (Wk 06) In the projection picture, what subspace does the residual y − Xŵ live in?
5. (Physics) Blind analyses freeze cuts before unblinding — the lesson explains the
   practice. Which rule this week is the same move?
