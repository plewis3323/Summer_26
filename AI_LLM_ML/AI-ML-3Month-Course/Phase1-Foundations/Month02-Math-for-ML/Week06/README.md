# Week 06 — Linear Algebra II: SVD, PCA, Matrix Calculus

The SVD is the spectral theorem for matrices that aren't square or symmetric — and it is
quietly underneath half of ML.

## Objectives

- Derive the SVD from the eigendecomposition of AᵀA and interpret U, Σ, V geometrically.
- Derive PCA two ways — variance maximization and reconstruction error — and show they agree.
- Use the SVD for low-rank approximation and the pseudoinverse for least squares.
- Compute the core matrix-calculus derivatives (∂/∂x of aᵀx, xᵀAx, ‖Ax − b‖²) and check
  any such derivative numerically.

## Core material (~3 hrs)

- Strang, MIT 18.06, Lecture 29 (singular value decomposition). Also revisit
  Lectures 15–16 (projections and least squares) — they connect directly to the
  pseudoinverse.
- 3Blue1Brown, *Essence of Linear Algebra*: rewatch the eigenvector and change-of-basis
  chapters with SVD in mind.
- Parr & Howard, "The Matrix Calculus You Need For Deep Learning" — read through the
  vector chain rule section. This is the reference you will reuse for backprop in
  Phase 2; get the numerator/denominator layout conventions straight now.
- Optional: Bishop, *PRML* §12.1 for PCA stated in the ML idiom (in `references/`).

## Derivations (paper first)

- SVD: from the eigendecomposition of AᵀA (and AAᵀ), construct A = UΣVᵀ; show the
  singular values are the square roots of the eigenvalues.
- PCA #1: maximize wᵀSw subject to ‖w‖ = 1 with a Lagrange multiplier → top eigenvector
  of the covariance.
- PCA #2: minimize mean squared reconstruction error over rank-k projections → same
  answer. Note where the two proofs use the same fact.
- Pseudoinverse: show x = A⁺b = VΣ⁺Uᵀb solves min ‖Ax − b‖² (minimum-norm solution).
- Matrix calculus: derive ∇ₓ(aᵀx), ∇ₓ(xᵀAx), ∇ₓ‖Ax − b‖², and the vector chain rule
  for f(g(x)).

## Exercises (built when the week starts)

1. **SVD by hand-then-NumPy.** For a small matrix, build the SVD from `eigh` of AᵀA;
   compare with `np.linalg.svd`.
   Accept when: reconstructed A matches to 1e-8 (up to sign conventions, handled explicitly).
2. **Image compression.** Truncated-SVD compression of a grayscale detector-event image
   at ranks 1, 5, 20, 50; plot error vs rank against the singular-value tail.
   Accept when: measured reconstruction error matches the Eckart–Young prediction (σ tail) to 1e-8 at each rank.
3. **PCA from scratch.** Center → covariance → `eigh` → project; run on correlated
   synthetic kinematics data; compare components and explained variance with
   scikit-learn's PCA.
   Accept when: explained-variance ratios agree with sklearn to 1e-8 and components agree up to sign.
4. **Both PCA derivations, numerically.** Show top-1 variance maximization (numerical
   search over unit vectors) lands on the reconstruction-error minimizer.
   Accept when: the two optima align with |cos| > 0.999.
5. **Pseudoinverse fit.** Solve an overdetermined straight-line fit via A⁺; compare with
   `lstsq` and with the normal equations on an ill-conditioned design.
   Accept when: A⁺ and lstsq agree to 1e-8 while the normal equations visibly lose precision (relative error reported).
6. **Gradient checker.** Write a finite-difference gradient check and use it to verify
   every derivative from the derivation list.
   Accept when: all analytic gradients match central differences to 1e-6.

## Deliverable

Completed notebook, scanned derivations (SVD, PCA ×2, pseudoinverse, matrix-calc
identities), and the gradient checker kept as a reusable function — Phase 2 will use it.

## Review

1. (Wk 05) State the four fundamental subspaces of A and which SVD vectors span each.
2. (Wk 05) Why did power iteration converge slowly when λ₂ ≈ λ₁?
3. (Wk 04) The J/ψ fit: what quantity did `curve_fit` minimize, and how does this week's
   ‖Ax − b‖² derivation relate to it?
4. (Wk 02) You want a scatter matrix of the top-3 PCA components colored by mass window —
   which pandas/matplotlib tools, in one sentence?
