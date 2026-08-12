# Week 05 — Linear Algebra I

You've used matrices as rotation operators and metric tensors for years; this week
re-grounds them as linear maps between subspaces — the picture everything in ML hangs on.

## Objectives

- Read any matrix as a linear map: what it does to a basis, what it kills, what it reaches.
- Compute and interpret the four fundamental subspaces and verify rank–nullity numerically.
- Diagonalize a matrix and say precisely when you can't.
- Derive and use the projection matrix onto a column space.
- Implement power iteration and explain why it finds the dominant eigenvector.

## Core material (~3 hrs)

- 3Blue1Brown, *Essence of Linear Algebra*, chapters 1–9 and 13–14 (vectors through dot
  products; change of basis; eigenvectors). Watch fast; the point is the geometry.
- Strang, MIT 18.06: Lecture 6 (column space and nullspace), Lecture 9 (independence,
  basis, dimension), Lecture 10 (the four fundamental subspaces), Lecture 21
  (eigenvalues and eigenvectors). Skim Lectures 1–3 only if elimination feels rusty.
- Optional reference: Strang, *Introduction to Linear Algebra* — the "big picture of
  linear algebra" figure of the four subspaces.

## Derivations (paper first)

- Projection matrix: derive P = A(AᵀA)⁻¹Aᵀ from orthogonality of the residual, and show
  P² = P, Pᵀ = P.
- Prove eigenvectors of a real symmetric matrix belonging to distinct eigenvalues are
  orthogonal.
- Rank–nullity: argue dim(col A) + dim(null A) = n from the linear-map picture.

## Exercises (built when the week starts)

1. **Matrix as map.** Apply a given 2×2 matrix to the unit circle and basis vectors;
   plot input vs output; identify rotation/shear/scaling parts.
   Accept when: plot shows the mapped circle and the stated decomposition matches the matrix's action on e₁, e₂.
2. **Four subspaces.** For a given 3×4 rank-2 matrix, compute bases for all four
   subspaces and verify the two orthogonality relations numerically.
   Accept when: all pairwise dot products between orthogonal-complement bases are < 1e-10 and dimensions sum correctly.
3. **Rank–nullity in the wild.** Build a rank-deficient "detector response" matrix;
   find what it cannot distinguish (nullspace directions).
   Accept when: A @ v < 1e-10 for the found nullspace vector and rank + nullity = n.
4. **Projection.** Project a noisy signal vector onto the column space of a design
   matrix using your derived P; compare with `np.linalg.lstsq`.
   Accept when: the two projections agree to 1e-8.
5. **Power iteration.** Implement it; find the dominant eigenpair of a symmetric matrix;
   compare with `np.linalg.eigh`; observe convergence rate vs eigenvalue gap.
   Accept when: eigenvalue matches eigh to 1e-6 and the convergence plot shows geometric decay at ratio |λ₂/λ₁|.
6. **Diagonalization limits.** Attempt to diagonalize a defective matrix; show what
   fails and why.
   Accept when: the eigenvector matrix is numerically singular (condition number reported) and the failure is stated in one line.

## Deliverable

Completed exercise notebook (checks PASS) plus scanned paper derivations in this folder.

## Review

1. (Wk 01) Broadcasting question: what is the shape of `A[:, None, :] - A[None, :, :]`
   for A of shape (N, 3), and what physics quantity might that pairwise structure compute?
2. (Wk 03) You changed a function's behavior this week refactoring it — which kind of
   test catches that, and what should you have written first?
3. (Wk 04) What three things did the dimuon pipeline pin to make reruns identical?
4. (Physics) A symmetric matrix with orthogonal eigenvectors — where does that same
   theorem show up in quantum mechanics, and what plays the role of the eigenbasis?
