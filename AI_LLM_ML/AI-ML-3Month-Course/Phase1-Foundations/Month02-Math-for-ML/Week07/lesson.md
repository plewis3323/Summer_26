# Week 07 — Linear Algebra II

~3 hrs reading, computing along as you go. Before starting you should be able to:
take a derivative and a gradient and run a few steps of gradient descent (Week 05);
treat a matrix as a linear map, name the four fundamental subspaces, and solve a
least-squares problem by the normal equations (Week 06).

Week 06 ended with two IOUs: eigenvectors and the SVD were "next week," and the
normal equations would be re-derived by matrix calculus, with an explanation of
what goes wrong if you form $A^\top A$ in floating point. Both come due here. The
new objects — eigenvalues, the SVD, PCA, the pseudoinverse — are last week's
geometry with *preferred axes*: directions a matrix stretches without tilting, and
the natural axes of a cloud of data. Phase 2's backpropagation is this week's
matrix calculus on a longer chain; Week 09's `lstsq` citation is §4.

## 1. Eigenvalues and eigenvectors

### 1.1 The equation $Av = \lambda v$

A square matrix $A$ eats a vector and usually *turns* it. Occasionally it does
not. A nonzero vector $v$ that $A$ only stretches or shrinks — same line, new
length — is an **eigenvector** of $A$. The stretch factor $\lambda$ is the
matching **eigenvalue**:

$$Av = \lambda v.$$

Algebraically: $(A - \lambda I)v = 0$, so $v$ lives in the null space of
$A - \lambda I$. For a nonzero $v$ to exist that matrix must be singular:

$$\det(A - \lambda I) = 0.$$

The left-hand side is a polynomial in $\lambda$ of degree $n$, the
**characteristic polynomial**. Its roots are the eigenvalues. Each root
$\lambda$ has a null space $N(A - \lambda I)$, the matching **eigenspace**.

A $2\times 2$ you can finish by hand. $A = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$:

$$\det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda\end{pmatrix}
= (2-\lambda)^2 - 1 = (\lambda-3)(\lambda-1).$$

Eigenvalues $3$ and $1$. For $\lambda = 3$, $(A-3I)v = 0$ gives $v = (1, 1)$
(or any multiple). For $\lambda = 1$, $v = (1, -1)$. Check:
$A(1,1) = (3,3) = 3(1,1)$. The two eigenvectors are orthogonal — not a
coincidence; §1.3 proves it for every symmetric matrix.

Apply $A$ to the unit circle (Week 06's gallery): the circle becomes an ellipse
whose axes are the eigenvectors and whose axis lengths are the $|\lambda|$'s.
A $90^\circ$ rotation has characteristic polynomial $\lambda^2+1=0$ — no real
roots, nothing stays on its own line. Eigenvectors need not exist over the
reals. The SVD in §2 still will, because it is built from $A^\top A$, always
symmetric.

In NumPy, do not form the characteristic polynomial. When $A^\top = A$, use
`np.linalg.eigh` (the `h` is for Hermitian, the complex cousin of symmetric).
Eigenvalues come back ascending; eigenvectors as *columns*:

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)
print(vals)                  # [1. 3.]
print(vecs)                  # columns ~ (1,-1)/sqrt(2) and (1,1)/sqrt(2)
print(A @ vecs[:, 0])        # always verify: should equal vals[0] * vecs[:, 0]
print(vals[0] * vecs[:, 0])
```

`np.linalg.eig` exists for non-symmetric matrices; prefer `eigh` whenever
$A^\top = A$.

### 1.2 Symmetric matrices have orthogonal eigenvectors

A matrix is **symmetric** when $A^\top = A$, i.e. $A_{ij} = A_{ji}$. Covariance
matrices, $A^\top A$, Hessians, projection matrices: the matrices that matter
in this course are symmetric, and they are the well-behaved case.

**Claim.** If $A$ is real symmetric and $Av = \lambda v$, $Aw = \mu w$ with
$\lambda \ne \mu$ and $v, w$ nonzero, then $v \perp w$.

**Proof.** Two ways of writing the scalar $v^\top A w$. First, $Aw = \mu w$, so
$v^\top A w = \mu\, v^\top w$. Second, a scalar equals its transpose:
$(v^\top A w)^\top = w^\top A^\top v$. Symmetry says $A^\top = A$, and
$Av = \lambda v$, so this is $\lambda\, w^\top v$. But $w^\top v = v^\top w$.
Therefore $(\mu - \lambda)\, v^\top w = 0$, hence $v^\top w = 0$.
$\qquad\blacksquare$

This is the week's paper derivation #1. If an eigenvalue repeats, its eigenspace
may be more than one-dimensional, and you can always pick an orthonormal basis
of it. The full statement is the **spectral theorem**: every real symmetric
$n \times n$ matrix has $n$ real eigenvalues (counted with repetition) and an
orthonormal basis of eigenvectors.

### 1.3 Diagonalization

Stack those orthonormal eigenvectors as columns of an orthogonal matrix $Q$
($Q^\top Q = I$, so $Q^{-1} = Q^\top$ — a rotation-or-reflection) and put the
eigenvalues on the diagonal of $\Lambda$. Then $AQ = Q\Lambda$, so

$$A = Q \Lambda Q^\top.$$

This is **orthogonal diagonalization**: $Q^\top$ rotates into the eigenbasis,
$\Lambda$ stretches each axis, $Q$ rotates back. A matrix that can be written
$A = PDP^{-1}$ for invertible $P$ and diagonal $D$ is **diagonalizable**.
Symmetric matrices always are, with the bonus $P^{-1}=P^\top$. Powers become
cheap: $A^k = Q\Lambda^k Q^\top$.

### 1.4 When you can't: defective matrices

Not every matrix is diagonalizable. A **defective** matrix has fewer than $n$
independent eigenvectors, so there is no $P$ that turns it into a diagonal
matrix. Standard $2\times 2$: the Jordan block
$J = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$ has a double eigenvalue $2$
but $J-2I$ has a 1-dimensional null space — only one independent eigenvector,
$(1,0)$. `np.linalg.eig` will still return two columns; they will be linearly
dependent. You will almost never meet a defective matrix in this course on
purpose. You *will* meet matrices that are diagonalizable but badly scaled —
§4.3.

### 1.5 Power iteration, and Week 05's zig-zag

To find the eigenvector of largest $|\lambda|$ without a characteristic
polynomial: start with a random vector, repeatedly apply $A$, and renormalize.

```python
def power_iteration(A, n_steps=30):
    rng = np.random.default_rng(0)
    v = rng.normal(size=A.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(n_steps):
        v = A @ v
        v = v / np.linalg.norm(v)
    lam = v @ (A @ v)            # Rayleigh quotient, since ||v|| = 1
    return lam, v

print(power_iteration(A))        # lambda ~ 3, v ~ ±(1, 1)/sqrt(2)
```

**Why it works.** Write $v$ in the eigenbasis: $v = \sum_i c_i q_i$. Then
$A^k v = \sum_i c_i \lambda_i^k q_i$. If $|\lambda_{\max}|$ strictly beats the
rest, that term dominates, and after renormalizing you are looking at
$q_{\max}$. The **Rayleigh quotient** $v^\top A v / v^\top v$ equals $\lambda$
on an eigenvector.

Now pay Week 05's remaining geometric debt. The loss $f(x,y) = x^2 + 10 y^2$
produced a zig-zag: GD was too timid along $x$ and too bold along $y$. The
**Hessian** of $f$ — the matrix of second derivatives,
$H_{ij} = \partial^2 f / \partial x_i \partial x_j$ — is
$\begin{pmatrix}2 & 0 \\ 0 & 20\end{pmatrix}$. Those diagonal entries *are* the
eigenvalues of $H$, and they *are* the two curvatures. In a general quadratic
$f(x) = \tfrac12 x^\top H x$ with $H$ symmetric **positive definite** (all
eigenvalues positive — a bowl, not a saddle), the eigen-axes of $H$ are the
bowl's principal axes and the eigenvalues are how steep those axes are. A
single learning rate cannot match two very different curvatures. Week 08's
optimizer race is the engineering response.

## 2. The SVD

Eigenvectors required a square, well-behaved matrix. Data matrices are
rectangular. The **singular value decomposition** (SVD) is the spectral theorem
for *every* matrix: three factors, always exist, always real.

### 2.1 Start from $A^\top A$

Let $A$ be $m \times n$. Then $A^\top A$ is $n \times n$ and symmetric:
$(A^\top A)^\top = A^\top A$. It is also **positive semidefinite**: for every
$x$,

$$x^\top (A^\top A)\, x = \|Ax\|^2 \ge 0,$$

so every eigenvalue of $A^\top A$ is $\ge 0$. The spectral theorem applies:

$$A^\top A = V \Lambda V^\top,$$

with $V$ an $n \times n$ orthogonal matrix (eigenvectors of $A^\top A$ as
columns) and $\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$,
$\lambda_1 \ge \cdots \ge \lambda_n \ge 0$. Define the **singular values** of
$A$ by

$$\sigma_i = \sqrt{\lambda_i} = \sqrt{\text{$i$-th eigenvalue of $A^\top A$}}.$$

Exactly $r = \mathrm{rank}(A)$ of them are positive: $Ax = 0$ iff
$A^\top A x = 0$ iff $x$ is an eigenvector of $A^\top A$ with $\lambda = 0$, so
the nullity of $A$ is the number of zero singular values, and rank–nullity
gives $r = n - \#\{\sigma_i = 0\}$.

### 2.2 Building $U$, $\Sigma$, $V$

The right singular vectors are the columns $v_1,\dots,v_n$ of $V$. For each
*positive* $\sigma_i$, define a left singular vector by sending $v_i$ through
$A$ and undoing the stretch:

$$u_i = \frac{A v_i}{\sigma_i}, \qquad i = 1, \dots, r.$$

These live in $\mathbb{R}^m$ and are orthonormal:

$$u_i^\top u_j
= \frac{v_i^\top (A^\top A) v_j}{\sigma_i \sigma_j}
= \frac{\lambda_j}{\sigma_i \sigma_j}\, v_i^\top v_j
= \delta_{ij},$$

using $V$'s orthonormality and $\lambda_j = \sigma_j^2$. If $r < m$, extend
$\{u_1,\dots,u_r\}$ to an orthonormal basis of $\mathbb{R}^m$. Stack them as
columns of an $m \times m$ orthogonal matrix $U$. Let $\Sigma$ be the
$m \times n$ matrix that is all zeros except $\Sigma_{ii} = \sigma_i$ for
$i = 1,\dots,r$ — a rectangular "diagonal." Then $A v_i = \sigma_i u_i$ for
$i \le r$ and $A v_i = 0$ for $i > r$, which is $AV = U\Sigma$, or

$$\boxed{\,A = U \Sigma V^\top\,}$$

— the **SVD**. This is the week's paper derivation #2. The same story run on
$AA^\top$ produces the $u_i$ as eigenvectors and the *same* nonzero $\sigma_i^2$,
which is why row rank equals column rank: both equal the number of positive
singular values.

`np.linalg.svd` gives $U$, the 1-D array of singular values `s` (descending),
and $V^\top$ (already transposed):

```python
rng = np.random.default_rng(0)
A = rng.normal(size=(5, 3))
U, s, Vt = np.linalg.svd(A, full_matrices=True)
Sigma = np.zeros((5, 3))
Sigma[:3, :3] = np.diag(s)
print(np.max(np.abs(U @ Sigma @ Vt - A)))     # ~1e-15
vals, _ = np.linalg.eigh(A.T @ A)
print(np.sort(s**2), np.sort(vals))           # the same numbers
```

The exercises ask you to *build* this from `eigh` of $A^\top A$. Signs: if $v_i$
is an eigenvector then so is $-v_i$, and $u_i = Av_i/\sigma_i$ tracks the choice.

### 2.3 The four subspaces, paid in full

Week 06 listed four subspaces and promised the SVD would make the inventory
obvious. Rank $r$ means $\sigma_1,\dots,\sigma_r > 0$ and the rest are zero.

| subspace | SVD basis | lives in |
|---|---|---|
| row space $C(A^\top)$ | $v_1, \dots, v_r$ | $\mathbb{R}^n$ |
| null space $N(A)$ | $v_{r+1}, \dots, v_n$ | $\mathbb{R}^n$ |
| column space $C(A)$ | $u_1, \dots, u_r$ | $\mathbb{R}^m$ |
| left null space $N(A^\top)$ | $u_{r+1}, \dots, u_m$ | $\mathbb{R}^m$ |

Why: $Av_i = \sigma_i u_i$. For $i\le r$ those $v_i$ span the row space and
those $u_i$ span the column space. For $i>r$, $\sigma_i=0$ so $Av_i=0$: the
remaining $v$'s *are* the null space. The remaining $u$'s are orthogonal to
every $Av$, hence span $N(A^\top)$. Dimensions $r$, $n-r$, $r$, $m-r$, and the
two orthogonality relations, are $U$ and $V$ being orthogonal. Strang's "big
picture" is this table drawn as two boxes.

### 2.4 Geometry: rotate, stretch, rotate

$A = U\Sigma V^\top$ applied to a vector is three maps in sequence:

1. $V^\top$: rotate (or reflect) into the right-singular-vector basis.
2. $\Sigma$: stretch axis $i$ by $\sigma_i$. If $m \ne n$, also drop axes or
   pad zeros — this is where dimension changes.
3. $U$: rotate into the left-singular-vector basis.

The unit sphere in input space goes to an ellipsoid in the column space, with
semi-axes $\sigma_i u_i$. Every linear map is "rotate, stretch, rotate." A
symmetric positive-definite matrix is the special case $U = V$ with
$\sigma_i = \lambda_i$; SVD and eigendecomposition coincide.

Physics: a detector **response matrix** $R$ maps true deposited energies (the
energy a particle actually left in the material) to measured pulse heights —
Week 06's linear smearing model. The right singular vectors of $R$ are the
true-energy patterns the detector is most (and least) sensitive to; the
singular values are the sensitivities. A tiny $\sigma_i$ is a true pattern
almost in the null space — almost invisible. Rank–nullity with numbers
attached.

## 3. Low-rank approximation

Write the SVD as a sum of outer products (Week 06, viewpoint 3):
$A = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top$. Each term is rank-1. Keep only
the first $k$ terms, $k \le r$:

$$A_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top = U_k \Sigma_k V_k^\top,$$

where $U_k$ is the first $k$ columns of $U$, and likewise for $V_k$, and
$\Sigma_k = \mathrm{diag}(\sigma_1,\dots,\sigma_k)$. This **truncated SVD** is
rank $k$.

Among *all* rank-at-most-$k$ matrices, $A_k$ is the closest to $A$. That is the
**Eckart–Young theorem**. Closest in the **Frobenius norm**
$\|M\|_F = \sqrt{\sum_{ij} M_{ij}^2}$ (ordinary Euclidean norm on the entries),
and also closest in the spectral norm $\|M\|_2 = \sigma_{\max}(M)$. The
Frobenius reconstruction error is exactly the discarded tail:

$$\|A - A_k\|_F^2 = \sigma_{k+1}^2 + \cdots + \sigma_r^2.$$

(The spectral-norm error is just $\sigma_{k+1}$.) If the singular values decay
fast, a small $k$ already reconstructs $A$ well; if they decay slowly, the
matrix is genuinely high-rank and you cannot compress it cheaply.

### 3.1 Compressing a detector-event image

A **calorimeter** is a detector that *absorbs* a particle and measures the
energy dumped, cell by cell — a grid of sensors, like a camera whose pixels see
energy instead of light. A **shower** (a spray of secondaries) lights up a blob
of cells; a charged particle that merely skims through lights up a thin track.
The event is a grayscale image: one number per cell.

```python
n = 32
xs = np.arange(n)
ys = np.arange(n)
xx, yy = np.meshgrid(xs, ys)
shower = np.exp(-((xx - 12.0)**2 + (yy - 18.0)**2) / (2 * 4.0**2))
track  = 0.4 * np.exp(-((xx - yy + 4.0)**2) / (2 * 0.6**2))
img = shower + track                          # 32×32

U, s, Vt = np.linalg.svd(img, full_matrices=False)

def reconstruct(k):
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

for k in [1, 3, 8, 32]:
    rec = reconstruct(k)
    err = np.linalg.norm(img - rec)**2
    tail = np.sum(s[k:]**2)
    print(k, err, tail)                       # err == tail, up to rounding
```

$k = 1$ already captures the blob; $k = 8$ is visually close. Image compression
with no JPEG knowledge — just Eckart–Young. §5 throws away the same tail.

## 4. The pseudoinverse

### 4.1 Definition

If $A$ is square and invertible, $A^{-1} = V \Sigma^{-1} U^\top$, because
$\Sigma^{-1}$ just reciprocates the singular values. If $A$ is rectangular, or
square but singular, some $\sigma_i$ are zero and cannot be reciprocated.
Reciprocate the *nonzero* ones and leave the zeros as zeros; transpose the
shape so $\Sigma^+$ is $n \times m$:

$$(\Sigma^+)_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0, \\ 0 & \sigma_i = 0,\end{cases}$$

and all off-diagonal entries zero. The **Moore–Penrose pseudoinverse** is

$$A^+ = V \Sigma^+ U^\top.$$

When $A$ *is* invertible this equals $A^{-1}$. NumPy: `np.linalg.pinv(A)`.

```python
A = np.array([[1.0, 2.0],
              [2.0, 4.0],
              [3.0, 6.0]])        # rank 1: column 2 = 2 × column 1
print(np.linalg.pinv(A) @ A)      # projection onto the row space, not I
```

### 4.2 Least squares, minimum-norm

**Claim.** $x^+ = A^+ b$ solves $\min_x \|Ax - b\|^2$. If several $x$ achieve
the same minimum (nontrivial null space), $x^+$ is the one of smallest
$\|x\|$ — the **minimum-norm** least-squares solution.

**Why, in SVD language.** Write $x = V\tilde{x}$ and $\tilde{b} = U^\top b$.
Then, because $U$ preserves norms,

$$\|Ax - b\|^2
= \|\Sigma \tilde{x} - \tilde{b}\|^2
= \sum_{i=1}^{r} (\sigma_i \tilde{x}_i - \tilde{b}_i)^2
+ \sum_{i=r+1}^{m} \tilde{b}_i^2.$$

The second sum does not depend on $x$ — that is the part of $b$ in the left
null space, unreachable (Week 06). The first sum is minimized by
$\tilde{x}_i = \tilde{b}_i / \sigma_i$ for $i \le r$. For $i > r$, $\sigma_i = 0$,
so $\tilde{x}_i$ does not appear in the residual at all; it *does* appear in
$\|x\|^2 = \|\tilde{x}\|^2$, so the minimum-norm choice is $\tilde{x}_i = 0$.
That is exactly $\tilde{x} = \Sigma^+ \tilde{b}$, i.e.
$x = V\Sigma^+ U^\top b = A^+ b$. $\qquad\blacksquare$

This is the week's paper derivation #3, and it is what `np.linalg.lstsq` does.
Week 06's six-point calibration (known gamma energies, noisy pulse heights):

```python
E = np.array([0.51, 0.66, 0.84, 1.17, 1.27, 1.33])
rng = np.random.default_rng(42)
y = 5.0 * E + 2.0 + rng.normal(0, 0.1, size=E.size)
A = np.stack([E, np.ones(E.size)], axis=1)

x_pinv = np.linalg.pinv(A) @ y
x_ls   = np.linalg.lstsq(A, y, rcond=None)[0]
x_ne   = np.linalg.solve(A.T @ A, A.T @ y)
print(x_pinv, x_ls, x_ne)         # all ~ [4.98, 2.03]
```

On a well-conditioned $6\times 2$ they agree. They stop agreeing in §4.3.

### 4.3 Condition number, and why $A^\top A$ is the wrong move

The **condition number** of a matrix (2-norm) is the ratio of its extreme
singular values, $\kappa(A) = \sigma_{\max}(A)/\sigma_{\min}(A)$ (smallest
*positive* $\sigma$ if $A$ is rank-deficient). $\kappa = 1$ is a scaled
rotation. Large $\kappa$ means some directions are amplified enormously
relative to others: a tiny perturbation of $b$ along a near-null direction, or
a tiny rounding error in $A$, produces a huge swing in $x$. Week 06's "is this
matrix nearly singular?" is the question $\kappa(A) \gg 1$.

Now form $A^\top A$. Its eigenvalues are $\sigma_i^2$, so
$\kappa(A^\top A) = \kappa(A)^2$. The normal equations *square the condition
number*. If $\kappa(A) = 10^8$, already uncomfortable in double precision
(about 16 decimal digits), then $\kappa(A^\top A) = 10^{16}$ and $A^\top A$ is
numerically singular: `np.linalg.solve(A.T @ A, A.T @ y)` returns garbage even
though the least-squares problem is still well-posed. The SVD route never forms
$A^\top A$; it works with the $\sigma_i$ of $A$ itself. That is why
`np.linalg.lstsq` exists, and why Week 09 will cite this paragraph when it
tells you not to invert $X^\top X`.

A miniature of the failure — two nearly dependent columns:

```python
eps = 1e-10
A = np.array([[1.0, 1.0 + eps],
              [1.0, 1.0],
              [1.0, 1.0]])
b = np.array([2.0, 2.0, 2.0])
print(np.linalg.cond(A), np.linalg.cond(A.T @ A))   # ~1/eps, then ~1/eps^2
print(np.linalg.lstsq(A, b, rcond=None)[0])
print(np.linalg.solve(A.T @ A, A.T @ b))            # can already look different
```

(`np.linalg.cond` is $\kappa$.) Forming $A^\top A$ is a derivation device, not a
numerical one.

## 5. PCA, two ways

**Principal component analysis** (PCA) finds the axes along which a cloud of
data spreads the most, and the subspace you should project onto if you want to
compress the cloud with the least squared damage. The two sentences sound
different. They are the same theorem.

### 5.1 Maximize variance

You have $n$ examples, each a vector in $\mathbb{R}^d$. Stack them as rows of
an $n \times d$ matrix $X$. **Center** first: subtract the mean of each column,
so each measurement has mean zero. (Without this, the first "principal axis" is
mostly just the mean.) Let $S$ be the **covariance matrix**

$$S = \frac{1}{n-1} X^\top X$$

($d \times d$, symmetric, positive semidefinite — an $A^\top A$ in disguise).
The diagonal $S_{jj}$ is the variance of measurement $j$; the off-diagonal
$S_{jk}$ is how $j$ and $k$ co-vary. For a unit vector $w$, the scalar
projections $Xw$ are the data's coordinates along axis $w$, and their variance
is $w^\top S w$. (Check:
$\tfrac{1}{n-1}\|Xw\|^2 = \tfrac{1}{n-1} w^\top X^\top X w = w^\top S w$.)

PCA's first axis is the unit $w$ that maximizes this variance: maximize
$f(w) = w^\top S w$ subject to $g(w) = w^\top w - 1 = 0$.

**Lagrange multipliers**, from geometry. At a constrained maximum you cannot
step along $\nabla f$ freely — you must stay on $g = 0$. The only way $\nabla f$
can fail to offer an improving *legal* step is if it sticks out *perpendicular*
to the surface, i.e. parallel to $\nabla g$:

$$\nabla f = \lambda \nabla g$$

for some scalar $\lambda$, the **Lagrange multiplier**. Here $\nabla g = 2w$.
For the objective, expand $w^\top S w = \sum_{jk} w_j S_{jk} w_k$; the partial
with respect to $w_i$ is $(Sw)_i + (S^\top w)_i$. Symmetry of $S$ gives
$\nabla(w^\top S w) = 2Sw$. Therefore

$$2Sw = \lambda \cdot 2w \qquad\Longleftrightarrow\qquad Sw = \lambda w.$$

The axis of maximum variance is an *eigenvector* of the covariance. And
$w^\top S w = \lambda$, so the variance along that axis *is* the eigenvalue.
The maximum is the largest one: $w = q_1$, the top eigenvector of $S$. The
second principal axis maximizes remaining variance among unit vectors
orthogonal to $q_1$ — the second eigenvector, by the same argument with an
extra constraint $w^\top q_1 = 0$. And so on. This is the week's paper
derivation #4.

### 5.2 Minimize reconstruction error

Now the other job. Pick a $k$-dimensional subspace with orthonormal basis
stacked as columns of $W$ ($d\times k$, $W^\top W = I_k$). Project each
centered $x$ onto it: $\hat{x} = WW^\top x$. The **reconstruction error** is
$\|x-\hat{x}\|^2$. Pythagoras: $x-\hat{x}$ is orthogonal to the subspace, so
$\|x-\hat{x}\|^2 = \|x\|^2 - \|W^\top x\|^2$. Minimizing mean reconstruction
error is maximizing the variance captured in $W^\top x$. By §5.1, take $W$'s
columns to be the top $k$ eigenvectors of $S$. The leftover error is the sum of
the discarded eigenvalues — the same tail as Eckart–Young, because PCA *is* the
truncated SVD of the centered data matrix (up to the $n-1$ scaling).

This is the week's paper derivation #5. Both proofs use the same fact: $w^\top S w$
on the unit sphere is maximized at the top eigenvector. Variance-max asks for
it directly; reconstruction asks for it after Pythagoras.

### 5.3 Two correlated measurements

**Kinematics** means the numbers that describe a particle's motion: energy,
momentum, direction. A particle's momentum along the beam (call it $p_z$ — the
beam is the direction the accelerator points) and the energy $E$ it dumps in a
calorimeter are two measurements of "how hard it was going." They move
together, plus independent detector noise. PCA should find that shared axis.

```python
rng = np.random.default_rng(0)
true_p = rng.normal(10.0, 3.0, size=200)
pz = true_p + rng.normal(0.0, 0.4, size=200)
E  = 1.1 * true_p + rng.normal(0.0, 0.5, size=200)
X = np.stack([pz, E], axis=1)
X = X - X.mean(axis=0)
S = (X.T @ X) / (X.shape[0] - 1)

vals, vecs = np.linalg.eigh(S)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
print(vals)                         # one large, one small
print(vecs[:, 0])                   # ~ direction (1, 1.1), normalized

W = vecs[:, :1]
X_hat = X @ W @ W.T
rec_err = np.sum((X - X_hat)**2) / (X.shape[0] - 1)
print(rec_err, vals[1])             # leftover = discarded eigenvalue

_, s, Vt = np.linalg.svd(X, full_matrices=False)
print(Vt[0], vecs[:, 0])            # same axis (up to sign)
print(s**2 / (X.shape[0] - 1), vals)
```

The cloud is a skinny ellipse. The top principal axis runs along it; the second
is the thin noise direction. Reconstruction error equals the leftover
eigenvalue, as derived. Prefer the SVD of $X$ over forming $S=X^\top X$, same
reason as §4.3. The exercises compare this to `sklearn.decomposition.PCA`.

## 6. Matrix calculus

Week 06 derived $\hat{x}$ from right angles. The other promised route is
calculus: write $L(x)=\|Ax-b\|^2$, take $\nabla_x L$, set it to zero. That is
**matrix calculus**. Backprop is the chain rule of §6.4 on a longer composition.

### 6.1 Layout convention

If $f$ maps $\mathbb{R}^n$ to $\mathbb{R}^m$, the **Jacobian** is the
$m \times n$ matrix with $J_{ij} = \partial f_i / \partial x_j$ — one row per
output, one column per input. This is **numerator layout** (Jacobian layout):
the derivative "looks like" the numerator $f$ stacked as rows, against the
denominator $x$ stacked as columns. For a *scalar* $f$, the Jacobian is a
*row* $1 \times n$. The **gradient** $\nabla f$ is the transpose of that row,
so it is a *column* with the same shape as $x$. That is why
$\nabla(a^\top x) = a$, a column, even though the Jacobian of $a^\top x$ is the
row $a^\top$.

Week 14 will say: reverse-mode autodiff computes a vector–Jacobian product and
captures that whole row in one pass. (Denominator layout flips the shapes;
Parr & Howard discuss both. This course uses numerator layout for Jacobians and
column gradients for scalars.)

### 6.2 Three identities

All three are for $x$ a column. Derive them on paper from the definition
(partial of the scalar with respect to $x_k$), then keep them as a table.

**Identity 1.** $\nabla_x (a^\top x) = a$.

$a^\top x = \sum_i a_i x_i$, so $\partial/\partial x_k = a_k$.

**Identity 2.** $\nabla_x (x^\top A x) = (A + A^\top)x$.

Expand: $x^\top A x = \sum_{ij} x_i A_{ij} x_j$. The partial with respect to
$x_k$ hits the left $x$ and the right $x$:
$\sum_j A_{kj} x_j + \sum_i x_i A_{ik} = (Ax)_k + (A^\top x)_k$. If $A$ is
symmetric this collapses to $2Ax$, which is the $2Sw$ of §5.1.

**Identity 3.** $\nabla_x \|Ax - b\|^2 = 2 A^\top (Ax - b)$.

Expand $\|Ax-b\|^2 = x^\top A^\top A x - 2 b^\top A x + b^\top b$. Differentiate
with identities 1–2 ($A^\top A$ is symmetric): $2A^\top A x - 2A^\top b =
2A^\top(Ax-b)$. Or chain entrywise: $\partial/\partial x_k$ of $\sum_i r_i^2$
is $\sum_i 2 r_i A_{ik}$, so the gradient is $2A^\top r$. Same thing.

### 6.3 Least squares, second derivation

Set the gradient to zero:

$$2 A^\top (A\hat{x} - b) = 0 \qquad\Longleftrightarrow\qquad
A^\top A\,\hat{x} = A^\top b.$$

The normal equations. Week 06 got them from "the residual is perpendicular to
every column." Week 05's Route 1 (set $\nabla L = 0$) plus Identity 3 get them
from calculus. Agreeing routes are how you know you believe a formula.
Neural-network training is Identity 3 with a much longer chain between $x$ and
the residual.

### 6.4 Vector chain rule

Week 05's chain rule: rates multiply along a composition. The vector version is
the same sentence with Jacobians.

Let $y = g(x)$ map $\mathbb{R}^n \to \mathbb{R}^m$, and let $f(y)$ be a scalar.
Then $\nabla_x f = J_g^\top\, \nabla_y f$, where $J_g$ is the $m \times n$
Jacobian of $g$. Equivalently, in numerator layout:
$df/dx = (df/dy)\,(dy/dx)$.

**Why.** A small step $\Delta x$ produces $\Delta y \approx J_g \Delta x$. Then
$\Delta f \approx (\nabla_y f)^\top \Delta y \approx (\nabla_y f)^\top J_g \Delta x$.
The column that represents this linear map in $x$ is $J_g^\top \nabla_y f$.
Special case $g(x)=Ax$: $J_g=A$ and $\nabla_x f = A^\top \nabla_y f$. Identity 3
is this with $f(y)=\|y-b\|^2$. The $A^\top$ in the normal equations is the chain
rule transposing a linear map — the same $A^\top$ that appeared geometrically as
"dot with every column." Longer chains multiply more Jacobians, each transposed
if you push a gradient backward. That *is* backpropagation (Week 13).

### 6.5 Finite-difference gradient check

Week 05's central difference still works, one coordinate at a time:

$$\frac{\partial f}{\partial x_i}
\approx \frac{f(x + h e_i) - f(x - h e_i)}{2h},$$

with $e_i$ the $i$-th standard basis vector and $h \sim 10^{-5}$. Compare the
whole numeric vector to your analytic $\nabla f$. A max absolute disagreement
of order $10^{-8}$ or smaller, for well-scaled $f$, means the analytic gradient
is right. Larger means a bug in the formula. Keep this function — the exercises
ask you to write `grad_check`; Phase 2 will call it on every new backward pass.

```python
def grad_check(f, grad_f, x, h=1e-5):
    g_analytic = grad_f(x)
    g_numeric = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = 1.0
        g_numeric[i] = (f(x + h * e) - f(x - h * e)) / (2 * h)
    return np.max(np.abs(g_analytic - g_numeric))

def f(x):
    return np.sum((A @ x - b)**2)

def grad_f(x):
    return 2 * A.T @ (A @ x - b)

A = np.array([[1.0, 0.5],
              [0.0, 2.0],
              [1.0, 1.0]])
b = np.array([1.0, 0.0, 2.0])
x = np.array([0.3, -0.7])
print(grad_check(f, grad_f, x))   # ~1e-10 or smaller
```

If Identity 3 is wrong, this number is not small. Trust the check over your
algebra until they agree.

Week 08 starts probability from scratch — random variables, maximum likelihood
(including *why* squared error was the right loss for Gaussian noise), entropy
and KL divergence — then races GD against momentum, RMSProp, and Adam on bowls
whose two curvatures, you now know, are eigenvalues. The condition number of
those bowls is the same $\kappa$ as §4.3.

## Check yourself

1. For $A = \begin{pmatrix} 0 & 2 \\ 2 & 0 \end{pmatrix}$, find the eigenvalues
   and a pair of orthogonal unit eigenvectors. What does $A$ do, geometrically?
2. In one sentence each: (a) why a real symmetric matrix has orthogonal
   eigenvectors for distinct eigenvalues; (b) what "defective" means.
3. An $m \times n$ matrix has SVD $A = U\Sigma V^\top$ with $r$ positive
   singular values. Which columns of $U$ and $V$ span each of the four
   fundamental subspaces?
4. You compress a $32\times 32$ image by keeping the top $k = 4$ singular
   values. What is $\|A - A_4\|_F^2$ in terms of the $\sigma_i$, and what
   theorem says no other rank-4 matrix beats $A_4$?
5. Why is forming $A^\top A$ and solving the normal equations a worse numerical
   plan than $x = A^+ b$, in one number?
6. PCA maximizes $w^\top S w$ on the unit sphere, and also minimizes
   reconstruction error of rank-$k$ projections. What single fact do both
   derivations use, and what is the solution?
7. Using numerator layout, write $\nabla_x(a^\top x)$, $\nabla_x(x^\top A x)$,
   and $\nabla_x \|Ax - b\|^2$. Set the third to zero: what do you recover?
8. You wrote an analytic gradient and `grad_check` returns $10^{-2}$. Name two
   distinct things that could be wrong, and one thing that is probably *not*
   wrong.

## Answers

1. Characteristic polynomial $\lambda^2 - 4 = 0$, so $\lambda = \pm 2$. For
   $\lambda = 2$, $v = (1,1)/\sqrt{2}$; for $\lambda = -2$,
   $v = (1,-1)/\sqrt{2}$. $A$ sends $(x,y)$ to $(2y, 2x)$: reflect across
   $y = x$ and stretch by 2 — equivalently, stretch by 2 along $(1,1)$ and by
   $-2$ (flip) along $(1,-1)$.
2. (a) $v^\top A w$ equals both $\lambda v^\top w$ and $\mu v^\top w$ by
   symmetry, so $(\lambda-\mu)v^\top w = 0$. (b) Fewer than $n$ independent
   eigenvectors; not diagonalizable.
3. Row space: $v_1,\dots,v_r$. Null space: $v_{r+1},\dots,v_n$. Column space:
   $u_1,\dots,u_r$. Left null space: $u_{r+1},\dots,u_m$.
4. $\sigma_5^2 + \cdots + \sigma_r^2$. Eckart–Young: the truncated SVD is the
   closest rank-at-most-$k$ matrix in Frobenius (and spectral) norm.
5. $\kappa(A^\top A) = \kappa(A)^2$. Squaring an already-large condition number
   can make $A^\top A$ numerically singular in double precision; the
   pseudoinverse / SVD path works with $\sigma_i(A)$ directly.
6. The quadratic form $w^\top S w$ on $\|w\|=1$ is maximized at the top
   eigenvector of $S$ (value = that eigenvalue). Variance-max asks for this
   directly (Lagrange); reconstruction asks for it after Pythagoras. Solution:
   top $k$ eigenvectors of the covariance, equivalently top $k$ right singular
   vectors of the centered data.
7. $a$; $(A+A^\top)x$; $2A^\top(Ax-b)$. Setting the third to zero recovers the
   normal equations $A^\top A\hat{x} = A^\top b$.
8. Could be wrong: the analytic formula, or the implementation of $f$ (so the
   numeric gradient is of a different function). Probably not wrong: the idea
   of central differences — a correct pair $(f, \nabla f)$ gives
   $\sim 10^{-8}$ or better at $h = 10^{-5}$. (A *huge* $h$, or a
   miserably-scaled $f$, can also inflate the number; try $h = 10^{-5}$ first.)

## New terms

- **eigenvector / eigenvalue** — nonzero $v$ with $Av=\lambda v$; the stretch factor $\lambda$.
- **characteristic polynomial / eigenspace** — $\det(A-\lambda I)$; $N(A-\lambda I)$.
- **symmetric matrix / spectral theorem** — $A^\top=A$; then $n$ real eigenvalues and an orthonormal eigenbasis.
- **orthogonal diagonalization** — $A=Q\Lambda Q^\top$ with $Q^\top Q=I$.
- **diagonalizable / defective** — has a full eigenbasis / does not.
- **power iteration / Rayleigh quotient** — repeat $v\leftarrow Av/\|Av\|$; $v^\top Av/v^\top v$, equals $\lambda$ on an eigenvector.
- **Hessian** — matrix of second partials; its eigenvalues are the principal curvatures.
- **positive definite / semidefinite** — all eigenvalues $>0$ / $\ge 0$; equivalently $x^\top Ax>0$ / $\ge 0$.
- **SVD / singular values $\sigma_i$** — $A=U\Sigma V^\top$; $\sigma_i=\sqrt{\text{eigenvalues of }A^\top A}$.
- **left / right singular vectors** — columns of $U$ / $V$.
- **truncated SVD / Eckart–Young / Frobenius norm** — $A_k$ keeps the top $k$ terms; it is the closest rank-$\le k$ matrix; $\|M\|_F=\sqrt{\sum_{ij}M_{ij}^2}$.
- **pseudoinverse $A^+$** — $V\Sigma^+ U^\top$; reciprocate nonzero singular values.
- **minimum-norm least-squares solution** — among all $\arg\min\|Ax-b\|^2$, the one with smallest $\|x\|$; equal to $A^+b$.
- **condition number $\kappa(A)$** — $\sigma_{\max}/\sigma_{\min}$; $\kappa(A^\top A)=\kappa(A)^2$.
- **PCA / principal axes / covariance $S$** — eigenvectors of $S=\frac{1}{n-1}X^\top X$ on centered data; directions of maximum variance.
- **centering** — subtract the column mean so each feature has mean zero.
- **Lagrange multiplier** — the $\lambda$ in $\nabla f=\lambda\nabla g$; enforces a constraint.
- **reconstruction error** — $\|x-WW^\top x\|^2$; leftover after projecting onto a subspace.
- **numerator layout / Jacobian** — $J_{ij}=\partial f_i/\partial x_j$; $m\times n$ for $f:\mathbb{R}^n\to\mathbb{R}^m$.
- **matrix calculus / vector chain rule** — derivatives of vector expressions; $\nabla_x f=J_g^\top\nabla_y f$ for $y=g(x)$.
- **gradient check / `grad_check`** — compare analytic $\nabla f$ to central differences, coordinate by coordinate.
- **calorimeter / shower** — a detector that absorbs a particle and records energy per cell; the spray of secondaries that lights up a blob of cells.
- **kinematics** — the numbers describing a particle's motion (energy, momentum, direction).

## Going deeper

- 3Blue1Brown, *Essence of Linear Algebra*, chapter 14 (eigenvectors and
  eigenvalues) — Week 06 told you this one was next. Rewatch change-of-basis
  for the $Q$ in $A=Q\Lambda Q^\top$. The later SVD video: watch after §2,
  not before — derive first.
- Strang, MIT 18.06 (OpenCourseWare): Lecture 29 (singular value decomposition).
  Revisit Lectures 15–16 (projections and least squares) with the pseudoinverse
  in mind; Lecture 21 if §1 felt fast.
- Parr & Howard, "The Matrix Calculus You Need For Deep Learning" — read through
  the vector chain rule, and get the numerator/denominator layout conventions
  straight now. This is the reference Phase 2 reopens for backprop.
- Optional: Bishop, *Pattern Recognition and Machine Learning* §12.1 (PCA in
  the ML idiom) — in `references/`. Same two derivations as §5, different
  notation.
