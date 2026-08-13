# Week 06 — Exercises

Work top to bottom in the notebook — all exercises are notebook cells this week
(per `NOTEBOOK_RULES.md` §6; nothing goes in files or the terminal). Setup gives
the imports, a seeded generator `rng = np.random.default_rng(7)`, the matrices
each exercise names, and labeled axes; you write only the lines asked for.
Derivations asked for "on paper" go in your derivations folder — photograph
them in.

## E1 — A matrix as a map

Setup gives the $2\times 2$ matrix `M = [[1.0, 1.0], [0.0, 1.0]]` (a shear).
Apply it to 200 points on the unit circle and to the two standard-basis
vectors $e_1, e_2$. Plot the input circle (faint) and the image (solid), with
the two basis images as arrows from the origin, labeled axes.
Hint: circle points are `(np.cos(theta), np.sin(theta))` stacked as a
`(2, 200)` array; the image is `M @ pts`. Column $j$ of $M$ is $M e_j$ —
check the arrows against the columns by eye before looking at the numbers.
Accept when: the plot shows the sheared ellipse and the two arrows match the
columns of `M` to 1e-12.

## E2 — Four subspaces

Setup gives a $3\times 4$ rank-2 matrix `A`. Compute bases for all four
fundamental subspaces (column space, null space, row space, left null space)
using `np.linalg.svd` with `full_matrices=True` (the lesson's §6 picture:
right singular vectors for $N(A)$ and the row space, left for $C(A)$ and
$N(A^\top)$ — or build them from `null_space`-style SVD slicing). Verify the
two orthogonality relations by taking all pairwise dots between complementary
bases, and print the four dimensions.
Hint: for a rank-$r$ matrix, the last $n-r$ right singular vectors span
$N(A)$; the first $r$ span the row space. Dimensions must satisfy
rank–nullity: $r + \dim N(A) = 4$ and $r + \dim N(A^\top) = 3$.
Accept when: all pairwise dots between orthogonal-complement bases are
$< 10^{-10}$ and the four dimensions are $2, 2, 2, 1$.

## E3 — Rank–nullity in the wild

Setup gives a $6\times 4$ "detector response" matrix `R` of rank 3 — four
true-signal channels, six readout channels, one combination of the four
signals that produces identical readout. Find a basis vector `v` for $N(R)$
and confirm `R @ v` is numerically zero. State in a one-line comment what
this means physically, and whether more data can fix it.
Hint: lesson §6.4; `np.linalg.svd(R, full_matrices=True)` — the last right
singular vector (singular value $\approx 0$) is $v$. More data cannot fix a
null space: the information was never recorded.
Accept when: `np.linalg.norm(R @ v) < 1e-10` and `rank + nullity == 4`.

## E4 — Projection vs lstsq

Setup gives a $20\times 3$ design matrix `A` and a noisy target `b`. Form the
projection matrix $P = A(A^\top A)^{-1}A^\top$ from the paper derivation,
compute `p = P @ b`, and compare with `np.linalg.lstsq(A, b, rcond=None)[0]`
pushed through `A`. Also check $P^2 \approx P$ and $P^\top \approx P$.
Hint: `np.linalg.solve(A.T @ A, A.T @ b)` is the $\hat x$ from the normal
equations; `p` should equal `A @ x_hat`. The two algebraic identities are
lesson §7.3.
Accept when: `p` matches `A @ lstsq_x` to 1e-8, and both $\|P^2-P\|$ and
$\|P-P^\top\|$ are $< 1e-10$.

## E5 — The six-point calibration, solved exactly

Reuse Week 05's six gamma-ray energies
`E = [0.51, 0.66, 0.84, 1.17, 1.27, 1.33]` and simulate
`y = 5.0 * E + 2.0 + rng.normal(0, 0.1, size=6)`. Build the $6\times 2$
design matrix with columns `E` and ones. Solve by the normal equations and
by `np.linalg.lstsq`; compare both to `np.polyfit(E, y, 1)`.
Hint: lesson §7.4 verbatim; `A = np.stack([E, np.ones(E.size)], axis=1)`.
`polyfit` returns `[slope, intercept]` in that order — same as your
$\hat x$.
Accept when: the three routes agree to 1e-8 on both parameters.

## E6 — A system with no inverse

Setup gives a defective $2\times 2$ matrix `D = [[1.0, 1.0], [1.0, 1.0]]`.
Attempt `np.linalg.solve(D, b)` for `b = [1.0, 0.0]` inside a
`try/except np.linalg.LinAlgError` (the notebook's setup is allowed to
catch this one — it is the point of the exercise, not control flow). Print
the determinant, the rank, and a one-line comment naming which fundamental
subspace `b` is *not* in.
Hint: columns are identical, so $C(D)$ is the line through $(1,1)$; $(1,0)$
is not on that line. `np.linalg.det(D)` is 0; rank is 1.
Accept when: the solve raises, `det` is 0 to 1e-12, rank is 1, and the
comment names the column space.

## Review

1. (Wk 05) Why did the gradient-descent zig-zag in E6, in one geometric
   sentence? (This week names the picture; next week names the eigenvalues.)
2. (Wk 05) `np.polyfit` gave the exact least-squares line in E7. Which two
   matrices did you just learn to build so you can write that fit yourself?
3. (Wk 04) The J/ψ fit: what quantity did `curve_fit` minimize, and how does
   this week's $\|Ax-b\|^2$ derivation relate to it?
4. (Wk 03) You want a scatter of $(p_x, p_y)$ for selected muons — which
   pandas/matplotlib tools, in one sentence?
