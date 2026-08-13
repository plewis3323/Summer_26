# Week 06 — Linear Algebra I

You've used lists of numbers since Week 03; this week they become geometry. A
matrix is a machine that moves space around, and once you can see what it does,
least squares, rank, and "the detector cannot tell these signals apart" stop
being formulas and start being pictures.

## Objectives

- Read any matrix as a linear map: what it does to a basis, what it kills, what
  it reaches.
- Compute and interpret the four fundamental subspaces and verify rank–nullity
  numerically.
- Multiply matrices three ways (row·column, columns of the right factor, sum of
  outer products) and say what each view is for.
- Derive the projection matrix onto a column space from orthogonality of the
  residual, and show $P^2 = P$, $P^\top = P$.
- Solve an overdetermined calibration by the normal equations and match
  `np.linalg.lstsq`.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text; compute along as you go.
- 3Blue1Brown, *Essence of Linear Algebra*, chapters 1–9 and 13 (vectors
  through dot products; change of basis). Watch fast; the point is the
  geometry. (Chapter 14, eigenvectors, is next week.)
- Strang, MIT 18.06: Lecture 6 (column space and nullspace), Lecture 9
  (independence, basis, dimension), Lecture 10 (the four fundamental
  subspaces), Lectures 15–16 (projections and least squares). Skim Lectures
  1–3 only if elimination feels rusty.

## Derivations (paper first)

- Projection matrix: derive $P = A(A^\top A)^{-1}A^\top$ from orthogonality of
  the residual, and show $P^2 = P$, $P^\top = P$.
- Rank–nullity: argue $\dim(C(A)) + \dim(N(A)) = n$ from the linear-map
  picture.
- Normal equations: from "closest point in the column space" to
  $A^\top A\hat{x} = A^\top b$, in two sentences.

## Exercises

See `exercises.md` (notebook built from it when the week starts, per
`NOTEBOOK_RULES.md`). Six exercises: a matrix as a map on the unit circle, the
four subspaces of a rank-2 matrix, a rank-deficient detector-response matrix,
projection vs `lstsq`, the six-point calibration solved exactly, and a
deliberately singular system.

## Deliverable

Completed exercise notebook (checks PASS) plus scanned paper derivations in
this folder.

## Review

1. (Wk 05) Why did the gradient-descent zig-zag in E6, in one geometric
   sentence? (This week names the picture; next week names the eigenvalues.)
2. (Wk 05) `np.polyfit` gave the exact least-squares line in E7. Which two
   matrices did you just learn to build so you can write that fit yourself?
3. (Wk 04) The J/ψ fit: what quantity did `curve_fit` minimize, and how does
   this week's $\|Ax-b\|^2$ derivation relate to it?
4. (Wk 03) You want a scatter of $(p_x, p_y)$ for selected muons — which
   pandas/matplotlib tools, in one sentence?
