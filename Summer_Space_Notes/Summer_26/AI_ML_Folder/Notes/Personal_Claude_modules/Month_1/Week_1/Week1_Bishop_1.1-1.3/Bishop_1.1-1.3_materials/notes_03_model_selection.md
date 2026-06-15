# Bishop 1.3 — Model Selection

*Grad-student summary. Source: Bishop, PRML §1.3.*

## The problem

In §1.1 the polynomial order $M$ controlled complexity; with regularization the knob became $\lambda$. Both are **hyperparameters** (model-selection parameters) — not fitted by minimizing training error, because more complexity always lowers training error. We need to estimate **generalization** performance to choose them.

## Validation set

- Split data: **train** (fit $\mathbf{w}$) + **validation / hold-out** (compare models / pick hyperparameters).
- If the validation set is also small and reused heavily, you can over-fit *it*; keep a final **test set**, touched once, for an honest performance number.
- Drawback: data used for validation is "wasted" for training — costly when data is scarce.

## Cross-validation

Use (almost) all data for both training and validation by rotating the hold-out fold.

**$S$-fold cross-validation:**
1. Partition data into $S$ equal groups.
2. For each fold $s=1\ldots S$: train on the other $S-1$ groups, validate on group $s$.
3. Average the $S$ validation scores.

- Uses a fraction $(S-1)/S$ for training each run.
- **Leave-one-out (LOO):** the extreme $S=N$ — maximal training data, but $N$ separate fits.

**Costs / limits:**
- Training runs multiply by $S$ — expensive for already-costly models.
- With several hyperparameters, grid-searching combinations is exponential in their count. This motivates better complexity criteria.

## Information criteria (penalize complexity analytically)

Instead of held-out data, add a penalty for the number of parameters $M$ to the maximized log-likelihood:

- **Akaike Information Criterion (AIC):** choose the model maximizing
$$\ln p(\mathcal{D}\mid\mathbf{w}_\text{ML}) - M.$$
- **Bayesian Information Criterion (BIC):** a variant (derived §4.4.1) with a heavier, $N$-dependent penalty $\sim \tfrac12 M\ln N$.

Bishop's verdict: these criteria are **crude** — they rely only on parameter *count*, ignore parameter uncertainty, and tend to favor over-simple models. The **fully Bayesian** approach (§3.4, model evidence / marginal likelihood) handles complexity more gracefully and, crucially, can be done using **training data alone** without a separate validation set.

---

## Quick intuition recap
- Can't pick complexity by training error → it monotonically improves with complexity.
- Hold-out / cross-validation = empirical generalization estimate (robust, but costs compute & data).
- Information criteria = cheap analytic proxy (but crude).
- Bayesian evidence = the principled route, deferred to later chapters.

See `img_cross_validation.png` for the $S$-fold rotation diagram.
