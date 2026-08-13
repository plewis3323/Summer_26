# Week 07 — Exercises

Work top to bottom in the notebook — all six exercises are notebook cells this week
(per `NOTEBOOK_RULES.md` §6; nothing goes in files or the terminal). Setup gives
the imports, a seeded generator `rng = np.random.default_rng(7)`, the matrices
and image each exercise names, and labeled axes; you write only the lines asked
for. Derivations asked for "on paper" go in your derivations folder — photograph
them in. Keep `grad_check` as a reusable function; Phase 2 will call it.

## E1 — SVD by hand, then NumPy

Setup gives the $3\times 2$ matrix `A`. On paper, form $A^\top A$, find its
eigenvalues and orthonormal eigenvectors, set $\sigma_i = \sqrt{\lambda_i}$,
build $V$ and the left vectors $u_i = A v_i / \sigma_i$, and write
$A = U\Sigma V^\top$. In the notebook, reconstruct the same factors from
`np.linalg.eigh(A.T @ A)` (sort descending; signs of $v_i$ are free, and $u_i$
tracks them) and compare to `np.linalg.svd(A, full_matrices=True)`.
Hint: lesson §2.1–2.2 verbatim; `Sigma` is $3\times 2$ with $\sigma_i$ on the
diagonal. Align signs before subtracting — if $v$ is an eigenvector so is $-v$.
Accept when: `U @ Sigma @ Vt` recovers `A` to 1e-10, and your singular values
match `np.linalg.svd`'s `s` to 1e-10.

## E2 — Image compression with the truncated SVD

Setup builds the lesson's $32\times 32$ calorimeter image (Gaussian shower blob
plus a thin track). Compute the SVD, reconstruct at $k = 1, 3, 8, 32$, and for
each $k$ print the Frobenius error $\|A - A_k\|_F^2$ next to the discarded tail
$\sum_{i>k}\sigma_i^2$. Plot the original and the four reconstructions, labeled
axes, shared color scale.
Hint: lesson §3.1; `A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]` with
`full_matrices=False`. Eckart–Young says the two numbers are equal.
Accept when: error and tail agree to 1e-8 at every $k$, and $k=8$ is visually
close to the original while $k=1$ is only the blob.

## E3 — PCA from scratch

Setup gives the two-measurement kinematics cloud of lesson §5.3 (`pz`, `E`;
200 rows). Center the columns, form the covariance $S = X^\top X / (n-1)$,
take `np.linalg.eigh`, and sort eigenvalues descending. Compare the top axis
and the explained variances to `sklearn.decomposition.PCA(n_components=2)`
on the same (uncentered) array — sklearn centers internally.
Hint: `PCA.components_` is $V^\top$ (rows are axes); `explained_variance_`
is the eigenvalues of $S$. Signs of axes are free; compare with
`np.abs(np.dot(w_yours, w_sk))`.
Accept when: explained variances match sklearn to 1e-8 and the absolute cosine
of the top axes is $> 1 - 10^{-8}$.

## E4 — Both PCA derivations, numerically

On the same centered cloud: (a) the variance along your unit top eigenvector
$w_1$ equals $\lambda_1$ (the $w^\top S w$ derivation); (b) the mean
reconstruction error of the rank-1 projection $X W W^\top$ equals the discarded
eigenvalue $\lambda_2$ (the Eckart–Young / Pythagoras derivation); (c) the top
right-singular vector of the centered $X$ is the same axis (up to sign).
Hint: lesson §5.1–5.3; `rec_err = np.sum((X - X @ W @ W.T)**2) / (n - 1)`
with $W = w_1$ as a column. SVD: `np.linalg.svd(X, full_matrices=False)`.
Accept when: (a) and (b) hold to 1e-10, and the SVD axis matches $w_1$ to
absolute cosine $> 1 - 10^{-8}$.

## E5 — Pseudoinverse vs the normal equations

Reuse Week 06's six-point calibration
`E = [0.51, 0.66, 0.84, 1.17, 1.27, 1.33]`,
`y = 5.0 * E + 2.0 + rng.normal(0, 0.1, size=6)`, design matrix columns `E`
and ones. Fit by `np.linalg.pinv(A) @ y`, by `np.linalg.lstsq`, and by
`np.linalg.solve(A.T @ A, A.T @ y)` — they should agree. Then, on the
ill-conditioned $3\times 2$ of lesson §4.3 (`eps = 1e-10`, nearly duplicate
columns), print $\kappa(A)$ and $\kappa(A^\top A)$ and compare the three
routes again.
Hint: lesson §4.2–4.3; `np.linalg.cond` is $\kappa$. Forming $A^\top A$
squares the condition number — that is the point, not a bug in your code.
Accept when: on the calibration the three $\hat x$ agree to 1e-8; on the
ill-conditioned system `pinv` matches `lstsq` to 1e-8 while the normal
equations differ, and $\kappa(A^\top A) \approx \kappa(A)^2$.

## E6 — Gradient checker

Write `grad_check(f, grad_f, x, h=1e-5)` as in lesson §6.5: central
differences, one coordinate at a time, returning the max absolute
disagreement with the analytic gradient. Run it on $f(x)=\|Ax-b\|^2$ with
the provided $A$, $b$, $x$ and your Identity-3 formula
$\nabla_x f = 2A^\top(Ax-b)$. Keep the function — do not inline it and
throw it away.
Hint: the lesson snippet is the spec; $h \sim 10^{-5}$ is the default.
If the check fails, the formula is wrong, not the checker.
Accept when: `grad_check` returns $< 10^{-8}$ on the provided point, and
the function is defined (not a one-off cell of arithmetic).

## Review

1. (Wk 06) State the four fundamental subspaces of $A$ and which SVD vectors
   span each.
2. (Wk 05) Why did power-iteration-style GD zig-zag when the two curvatures
   differed by 10×?
3. (Wk 04) The J/ψ fit: what quantity did `curve_fit` minimize, and how does
   this week's $\|Ax-b\|^2$ / pseudoinverse derivation relate to it?
4. (Wk 02) You want a scatter matrix of the top-3 PCA components colored by
   mass window — which pandas/matplotlib tools, in one sentence?
