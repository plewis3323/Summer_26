# Week 07 — Linear Algebra II: SVD, PCA, Matrix Calculus

The SVD is the spectral theorem for matrices that aren't square or symmetric —
and it is quietly underneath half of ML.

## Objectives

- Derive the SVD from the eigendecomposition of $A^\top A$ and interpret $U$,
  $\Sigma$, $V$ geometrically.
- Derive PCA two ways — variance maximization and reconstruction error — and
  show they agree.
- Use the SVD for low-rank approximation and the pseudoinverse for least
  squares.
- Compute the core matrix-calculus derivatives ($\partial/\partial x$ of
  $a^\top x$, $x^\top Ax$, $\|Ax-b\|^2$) and check any such derivative
  numerically.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text; compute along as you go.
- Strang, MIT 18.06, Lecture 29 (singular value decomposition). Also revisit
  Lectures 15–16 (projections and least squares) — they connect directly to the
  pseudoinverse.
- 3Blue1Brown, *Essence of Linear Algebra*: rewatch the eigenvector and
  change-of-basis chapters with SVD in mind.
- Parr & Howard, "The Matrix Calculus You Need For Deep Learning" — read
  through the vector chain rule section. This is the reference you will reuse
  for backprop in Phase 2; get the numerator/denominator layout conventions
  straight now.
- Optional: Bishop, *PRML* §12.1 for PCA stated in the ML idiom (in
  `references/`).

## Derivations (paper first)

- SVD: from the eigendecomposition of $A^\top A$ (and $AA^\top$), construct
  $A = U\Sigma V^\top$; show the singular values are the square roots of the
  eigenvalues.
- PCA #1: maximize $w^\top Sw$ subject to $\|w\| = 1$ with a Lagrange
  multiplier → top eigenvector of the covariance.
- PCA #2: minimize mean squared reconstruction error over rank-$k$ projections
  → same answer. Note where the two proofs use the same fact.
- Pseudoinverse: show $x = A^+ b = V\Sigma^+ U^\top b$ solves
  $\min \|Ax-b\|^2$ (minimum-norm solution).
- Matrix calculus: derive $\nabla_x(a^\top x)$, $\nabla_x(x^\top Ax)$,
  $\nabla_x\|Ax-b\|^2$, and the vector chain rule for $f(g(x))$.

## Exercises

See `exercises.md` (notebook built from it when the week starts, per
`NOTEBOOK_RULES.md`). Six exercises: SVD from `eigh` of $A^\top A$, truncated-SVD
image compression, PCA from scratch vs sklearn, both PCA derivations numerically,
the pseudoinverse vs the normal equations on an ill-conditioned design, and a
reusable finite-difference gradient checker.

## Deliverable

Completed notebook, scanned derivations (SVD, PCA ×2, pseudoinverse,
matrix-calc identities), and the gradient checker kept as a reusable function
— Phase 2 will use it.

## Review

1. (Wk 06) State the four fundamental subspaces of $A$ and which SVD vectors
   span each.
2. (Wk 05) Why did power-iteration-style GD zig-zag when the two curvatures
   differed by 10×?
3. (Wk 04) The J/ψ fit: what quantity did `curve_fit` minimize, and how does
   this week's $\|Ax-b\|^2$ / pseudoinverse derivation relate to it?
4. (Wk 02) You want a scatter matrix of the top-3 PCA components colored by
   mass window — which pandas/matplotlib tools, in one sentence?
