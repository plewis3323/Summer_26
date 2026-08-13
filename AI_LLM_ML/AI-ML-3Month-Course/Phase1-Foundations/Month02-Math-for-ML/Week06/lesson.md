# Week 06 — Linear Algebra I

~3 hrs reading, computing along as you go. Before starting you should be able to:
write Python functions and loops (Week 02); build NumPy arrays, index them, and use
`@`-free arithmetic on them (Week 03); take a derivative and a gradient and explain
what the gradient points at (Week 05).

Machine learning runs on rectangular blocks of numbers. A dataset is a block: one
row per example, one column per measurement. A model's parameters are blocks. Every
layer of every neural network in this course is, at its heart, one operation:
multiply a block by a vector. This week builds the mathematics of those blocks —
**linear algebra** — from the ground up, not as bookkeeping but as *geometry*: a
matrix is a machine that moves space around, and once you can see what a matrix
does, most of ML's core operations stop being formulas and start being pictures.

Two debts from Week 05 get paid here: the promised one-line proof that the gradient
points uphill (§2.5), and the promised systematic way to solve the equations that
$\nabla L = 0$ produces (§5, §7).

## 1. Vectors

A **vector** is an ordered list of numbers. That's the whole definition. The vector
$v = (3, 1)$ is the pair "3, then 1"; the vector $(0.51, 0.66, 0.84, 1.17, 1.27,
1.33)$ is Week 05's list of six gamma-ray energies. The number of entries is the
vector's **dimension**, and the set of all vectors with $n$ real entries is written
$\mathbb{R}^n$ ("R-n"). Entries are numbered with subscripts: $v_1 = 3$, $v_2 = 1$.

For 2 or 3 dimensions there is also a picture: draw $v = (3, 1)$ as an arrow from
the origin to the point 3 across, 1 up. The arrow picture is worth keeping even in
a million dimensions, where you can't draw it — the *operations* behave exactly as
the 2D arrows do.

Two operations define vector life:

- **Addition**, entry by entry: $(3, 1) + (1, 2) = (4, 3)$. Picture: place the
  second arrow's tail at the first arrow's head; the sum is the arrow to the final
  point.
- **Scalar multiplication** — a **scalar** is a plain single number — stretches
  every entry: $2 \cdot (3, 1) = (6, 2)$. Picture: same direction, twice the
  length; a negative scalar flips the arrow.

A **linear combination** of vectors $u$ and $v$ is anything of the form
$a u + b v$ with scalars $a, b$ — stretch each, then add. This one phrase is the
engine of the whole week.

In NumPy a vector is a 1-D array, and both operations are the arithmetic you
already know from Week 03:

```python
import numpy as np

u = np.array([3.0, 1.0])
v = np.array([1.0, 2.0])
print(u + v)        # [4. 3.]
print(2 * u)        # [6. 2.]
print(2*u - 3*v)    # a linear combination: [ 3. -4.]
```

Physics keeps vectors everywhere: a particle's **momentum** (its mass times
velocity, direction included — the quantity that says which way it's going and how
hard it is to stop) is a vector $(p_x, p_y, p_z)$; a detector hit has a position
vector; and in ML, each *example* — one collision event, one house, one image — is
a vector of its measurements, called its feature vector. Geometry on vectors *is*
geometry on data.

## 2. Length, angle, and the dot product

### 2.1 Length

The **norm** (length) of a vector comes from the Pythagorean theorem, applied once
per dimension:

$$\|v\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}.$$

A **unit vector** has norm 1; dividing any nonzero vector by its own norm
("normalizing" it) produces one: $\hat{v} = v / \|v\|$ points the same way with
length 1.

```python
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))        # 5.0  (the 3-4-5 triangle)
print(v / np.linalg.norm(v))    # [0.6 0.8], length 1
```

### 2.2 The dot product

The **dot product** of two vectors of the same dimension multiplies matching
entries and adds:

$$u \cdot v = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n.$$

You met this combination in Week 05 without its name: the change
$\Delta f \approx \frac{\partial f}{\partial x}\Delta x + \frac{\partial f}{\partial y}\Delta y$
is exactly $\nabla f \cdot (\Delta x, \Delta y)$. In NumPy it is the `@` operator:

```python
u = np.array([3.0, 1.0])
v = np.array([1.0, 2.0])
print(u @ v)                    # 5.0
print(v @ v, np.linalg.norm(v)**2)   # 5.0 5.0 — a vector dotted with itself is its norm squared
```

### 2.3 What the dot product measures — derived

Claim:

$$u \cdot v = \|u\|\,\|v\|\cos\theta,$$

where $\theta$ is the angle between the two arrows. So the dot product measures
*alignment*: positive when the arrows point the same general way, zero when
perpendicular, negative when opposed.

**Derivation.** School trigonometry gives the **law of cosines**: in a triangle
with sides $a$, $b$, $c$ and angle $\theta$ between $a$ and $b$,
$c^2 = a^2 + b^2 - 2ab\cos\theta$. Build the triangle from our arrows: sides $u$
and $v$ from the origin, third side $u - v$ connecting their tips. Then

$$\|u - v\|^2 = \|u\|^2 + \|v\|^2 - 2\|u\|\|v\|\cos\theta.$$

Now expand the left side algebraically, entry by entry:
$\|u-v\|^2 = \sum_i (u_i - v_i)^2 = \sum_i u_i^2 - 2\sum_i u_i v_i + \sum_i v_i^2
= \|u\|^2 - 2\,u\cdot v + \|v\|^2$. Set the two expressions equal, cancel
$\|u\|^2 + \|v\|^2$ from both sides, divide by $-2$:

$$u \cdot v = \|u\|\,\|v\|\cos\theta. \qquad \blacksquare$$

### 2.4 Orthogonality

Two vectors are **orthogonal** (perpendicular) exactly when $u \cdot v = 0$ —
$\cos 90° = 0$. This is the single most-used test in the course; check it
numerically whenever a derivation claims it:

```python
a = np.array([1.0, 2.0])
b = np.array([-2.0, 1.0])
print(a @ b)    # 0.0 — orthogonal
```

### 2.5 Paying Week 05's debt: why the gradient points uphill

Week 05 verified by brute force that the steepest-ascent direction is the
gradient's own direction. Now the one-line proof. A unit step $s$ changes $f$ by

$$\Delta f \approx \nabla f \cdot s = \|\nabla f\|\,\underbrace{\|s\|}_{=1}\cos\theta,$$

which is largest when $\cos\theta = 1$, i.e. when $s$ points along $\nabla f$ —
and most negative when $s$ points along $-\nabla f$. Steepest ascent is the
gradient direction; steepest descent is its negative. Done.

## 3. Matrices as linear maps

### 3.1 The matrix–vector product

A **matrix** is a rectangular grid of numbers, $m$ rows by $n$ columns — "an
$m \times n$ matrix". Entry $A_{ij}$ sits in row $i$, column $j$. But the useful
way to think of a matrix is not as a grid; it is as a *machine* that eats a vector
and returns a vector. The **matrix–vector product** $Av$ defines the machine. Two
equivalent readings:

**Row picture.** Entry $i$ of $Av$ is the dot product of row $i$ with $v$.

**Column picture** (the important one). $Av$ is a *linear combination of the
columns of $A$*, weighted by the entries of $v$:

$$\begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix}
= v_1 \begin{pmatrix} 2 \\ 0 \end{pmatrix} + v_2 \begin{pmatrix} 1 \\ 3 \end{pmatrix}.$$

Check both readings agree on an example, then let NumPy do it:

```python
A = np.array([[2.0, 1.0],
              [0.0, 3.0]])
v = np.array([1.0, 2.0])
print(A @ v)                          # [4. 6.]
print(v[0]*A[:, 0] + v[1]*A[:, 1])    # [4. 6.] — the column picture
```

An $m \times n$ matrix maps $\mathbb{R}^n \to \mathbb{R}^m$: it eats $n$-vectors
and returns $m$-vectors. Shapes must chain: $(m \times n)$ times $(n,)$ gives
$(m,)$ — Week 03's broadcasting discipline, now with a theory behind it.

### 3.2 Where the basis goes tells you everything

The **standard basis vectors** of $\mathbb{R}^2$ are $e_1 = (1, 0)$ and
$e_2 = (0, 1)$ — every vector is trivially a linear combination of them:
$(v_1, v_2) = v_1 e_1 + v_2 e_2$. Feed them to the machine:

$$A e_1 = \text{column 1 of } A, \qquad A e_2 = \text{column 2 of } A.$$

(Look at the column picture: weights $(1, 0)$ pick out column 1.) So **a matrix's
columns are where it sends the basis vectors** — the matrix is a complete record
of what happens to the two grid axes, and by linearity that determines what
happens to everything else.

**Linearity** is the property that makes all this work. A map $T$ is **linear** if

$$T(u + v) = T(u) + T(v) \quad\text{and}\quad T(c\,v) = c\,T(v)$$

— it respects addition and scaling, so it maps grid lines to (possibly tilted,
stretched) grid lines and keeps the origin fixed. Every matrix–vector product is
linear (check both properties from the column picture), and conversely *every*
linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ is multiplication by some matrix:
decompose $v = \sum_j v_j e_j$, apply linearity, and $T(v) = \sum_j v_j T(e_j)$ —
a linear combination of the vectors $T(e_j)$, i.e. the column picture with columns
$T(e_j)$. Matrices and linear maps are the same thing, written down.

### 3.3 A gallery of 2×2 machines

Watch what each matrix does to $e_1$, $e_2$, and the unit circle:

```python
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 200)
circle = np.stack([np.cos(t), np.sin(t)])       # shape (2, 200): 200 unit vectors

def show(A, title):
    mapped = A @ circle                          # applies A to every circle point at once
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(circle[0], circle[1], color="gray", label="input")
    ax.plot(mapped[0], mapped[1], color="red", label="output")
    ax.arrow(0, 0, A[0, 0], A[1, 0], head_width=0.08, color="blue")   # A e1
    ax.arrow(0, 0, A[0, 1], A[1, 1], head_width=0.08, color="green")  # A e2
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()
    plt.show()

show(np.array([[2.0, 0.0], [0.0, 0.5]]), "scaling")
show(np.array([[1.0, 0.8], [0.0, 1.0]]), "shear")
show(np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation by 90 degrees")
show(np.array([[1.0, 0.0], [0.0, 0.0]]), "projection onto the x-axis")
```

- **Scaling** $\begin{pmatrix}2&0\\0&0.5\end{pmatrix}$: stretches $x$ by 2,
  squashes $y$ by half; circle → axis-aligned ellipse.
- **Shear** $\begin{pmatrix}1&0.8\\0&1\end{pmatrix}$: slides the top of the plane
  sideways; $e_1$ stays put, $e_2$ tilts.
- **Rotation by angle $\theta$**: $e_1$ must land at $(\cos\theta, \sin\theta)$
  and $e_2$ at $(-\sin\theta, \cos\theta)$ — and since columns are where the basis
  lands, the rotation matrix can only be
  $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$.
  You just *derived* a matrix from its geometry.
- **Projection onto the $x$-axis**: flattens the plane onto a line. Information is
  destroyed — every vector $(0, y)$ lands on the origin. Remember this one; it is
  the seed of §6.

Physics runs on these machines. Rotating a detector's coordinate system into the
beam frame is a rotation matrix applied to every hit. And a real detector's
**response** — how true deposited energies smear into measured signals across
channels — is modeled as a matrix $R$: measured $= R\,(\text{true})$. Undoing that
smearing (called *unfolding*) is "solve $Rx = b$", which is exactly §5.

## 4. Matrix multiplication, three ways

If matrices are machines, the product $AB$ should be "run $B$, then run $A$" —
composition, like Week 05's composed functions:

$$(AB)\,v = A\,(B v) \quad \text{for every } v.$$

That *requirement* forces the formula for $AB$ (apply the column picture twice),
and shapes must chain: $(m \times n)(n \times p) = (m \times p)$. Three equivalent
viewpoints, each useful in a different situation:

**Viewpoint 1 — entries.** $(AB)_{ij} = (\text{row } i \text{ of } A) \cdot
(\text{column } j \text{ of } B)$. Best for computing one entry by hand.

**Viewpoint 2 — columns.** Column $j$ of $AB$ is $A$ applied to column $j$ of $B$:
$AB = [\,A b_1 \;|\; A b_2 \;|\; \cdots\,]$. Best for thinking: $B$'s columns say
where basis vectors go; $A$ then maps *those*.

**Viewpoint 3 — a sum of outer products.** The **outer product** $u\,w^\top$ of a
column $u$ ($m$ entries) and a row $w^\top$ ($p$ entries) is the $m \times p$
matrix with entries $u_i w_j$ — a "times table" of the two vectors, and always a
rank-1 matrix (§6 defines rank; every column is a multiple of $u$). Then

$$AB = \sum_{k=1}^{n} (\text{column } k \text{ of } A)\,(\text{row } k \text{ of } B)^{\phantom{\top}}$$

— the product is a stack of $n$ rank-1 layers. This viewpoint looks exotic now but
is the one that makes the SVD (Week 07) and attention (Phase 3) readable.

All three in code, against NumPy's `@`:

```python
rng = np.random.default_rng(0)
A = rng.normal(size=(2, 3))
B = rng.normal(size=(3, 4))

C1 = np.zeros((2, 4))                 # viewpoint 1: entry by entry
for i in range(2):
    for j in range(4):
        C1[i, j] = A[i, :] @ B[:, j]

C2 = np.stack([A @ B[:, j] for j in range(4)], axis=1)   # viewpoint 2: columns

C3 = np.zeros((2, 4))                 # viewpoint 3: sum of outer products
for k in range(3):
    C3 = C3 + np.outer(A[:, k], B[k, :])

print(np.max(np.abs(C1 - A @ B)), np.max(np.abs(C2 - A @ B)), np.max(np.abs(C3 - A @ B)))
# all ~1e-16
```

Facts to internalize, provable from the composition definition:

- **Not commutative:** $AB \ne BA$ in general — "rotate then shear" is not "shear
  then rotate". Try it with the gallery matrices.
- **Associative:** $(AB)C = A(BC)$ — composition doesn't care where you put the
  parentheses.
- The **transpose** $A^\top$ ("A transpose") flips rows and columns:
  $(A^\top)_{ij} = A_{ji}$; and $(AB)^\top = B^\top A^\top$ — note the reversal.
  With transposes, the dot product is matrix notation: $u \cdot v = u^\top v$.
- The **identity matrix** $I$ (ones on the diagonal, zeros elsewhere) is the
  do-nothing machine: $Iv = v$, $AI = IA = A$.
- The **inverse** $A^{-1}$, when it exists, is the machine that undoes $A$:
  $A^{-1}A = AA^{-1} = I$. §5 is about when it exists.

## 5. Solving linear systems

### 5.1 The question

"Solve $Ax = b$" asks: *which input vector $x$ does the machine $A$ send to $b$?*
Written out, it is a system of simultaneous linear equations — one equation per
row of $A$.

Here is the Week 05 promise coming due. Fitting the calibration line by Route 1
(set the gradient of the loss to zero) produced two linear equations in the two
unknowns $m, b$. In general, $\nabla L = 0$ for any quadratic loss is a linear
system. Learn to solve systems and you can solve those fits *exactly*, no descent
loop required.

Simplest physics version first: calibrate with just two gamma sources — no noise,
no fitting, exact. Energies 0.51 and 1.27 MeV produce pulse heights 4.1 and 7.9.
If the detector is linear, $mE + b = y$ for both:

$$0.51\,m + b = 4.1, \qquad 1.27\,m + b = 7.9
\quad\Longleftrightarrow\quad
\underbrace{\begin{pmatrix} 0.51 & 1 \\ 1.27 & 1 \end{pmatrix}}_{A}
\underbrace{\begin{pmatrix} m \\ b \end{pmatrix}}_{x}
= \underbrace{\begin{pmatrix} 4.1 \\ 7.9 \end{pmatrix}}_{b}.$$

### 5.2 Elimination

The school method, systematized. Subtract the first equation from the second to
kill $b$: $(1.27 - 0.51)m = 7.9 - 4.1$, so $m = 3.8 / 0.76 = 5.0$; substitute
back: $b = 4.1 - 0.51 \times 5.0 = 1.55$. That two-step dance — use one equation
to eliminate a variable from the others, repeat, then substitute back — is
**Gaussian elimination**, and it scales to any number of equations. It is also
exactly what `np.linalg.solve` does (with careful row-swapping for numerical
stability):

```python
A = np.array([[0.51, 1.0],
              [1.27, 1.0]])
b = np.array([4.1, 7.9])
x = np.linalg.solve(A, b)
print(x)                        # [5.   1.55]
print(A @ x - b)                # [0. 0.] — always verify the residual
```

### 5.3 When it fails: singular matrices and the determinant

Elimination breaks when a matrix's columns carry redundant information. Try to
solve with two measurements of the *same* source:

$$\begin{pmatrix} 0.51 & 1 \\ 0.51 & 1 \end{pmatrix}\begin{pmatrix} m \\ b\end{pmatrix} = \begin{pmatrix} 4.1 \\ 4.3 \end{pmatrix}$$

Subtracting rows gives $0 = 0.2$: no solution (the two measurements contradict).
Had the right side been $(4.1, 4.1)$, any $(m, b)$ on a whole line would work:
infinitely many solutions. Either way, no *unique* answer — the matrix is
**singular** (non-invertible). Geometrically, its two columns point the same way,
so the machine collapses the plane onto a line and cannot be undone.

For a $2\times 2$ matrix the scalar that detects collapse is the **determinant**
$\det A = A_{11}A_{22} - A_{12}A_{21}$: it is the factor by which $A$ scales
areas (derivable from the gallery pictures), so $\det A = 0$ means the unit
square is flattened to zero area — collapse. $A^{-1}$ exists exactly when
$\det A \ne 0$. In higher dimensions the determinant scales volumes; NumPy:
`np.linalg.det`. In practice you will rarely compute determinants — but "is this
matrix (nearly) singular?" is a question you will ask constantly, and Week 07
gives the sharper tool (condition number) for "nearly".

So for square $A$, three possibilities: unique solution ($\det A \neq 0$), no
solution, or infinitely many. Which of the last two you get depends on whether
$b$ is reachable — the next section makes "reachable" precise.

## 6. Subspaces, rank, and the four fundamental subspaces

### 6.1 Span, independence, basis, dimension

- The **span** of a set of vectors is everything you can build from them by
  linear combinations. Span of one nonzero vector: a line. Span of $(1,0)$ and
  $(0,1)$: all of $\mathbb{R}^2$.
- Vectors are **linearly independent** if none of them is a linear combination of
  the others — equivalently, the only combination giving the zero vector is
  all-zero weights. $(1, 0)$ and $(2, 0)$ are dependent; adding $(0, 1)$'s worth
  of new direction makes independence.
- A **subspace** of $\mathbb{R}^n$ is a set of vectors closed under linear
  combinations (contains the origin; any line or plane through the origin
  qualifies). Spans are subspaces.
- A **basis** of a subspace is a minimal spanning set: independent vectors whose
  span is the whole subspace. Every basis of a given subspace has the same number
  of vectors, and that number is the subspace's **dimension**.

### 6.2 Column space and null space

Attach two subspaces to any $m \times n$ matrix $A$:

**The column space** $C(A) \subseteq \mathbb{R}^m$: the span of $A$'s columns — by
the column picture, exactly the set of outputs the machine can produce. Now
"reachable" is precise: *$Ax = b$ has a solution exactly when $b \in C(A)$.* The
dimension of $C(A)$ is the **rank** $r$ of $A$: the number of genuinely
independent directions in the output.

**The null space** $N(A) \subseteq \mathbb{R}^n$: all inputs the machine sends to
zero, $\{x : Ax = 0\}$. It is a subspace (check: if $Ax = 0$ and $Ay = 0$ then
$A(ax + by) = 0$ by linearity). The null space is the machine's blind spot: if
$Ax^* = b$ and $v \in N(A)$, then $A(x^* + v) = b$ too — solutions come in
families, one for each null direction, which is where "infinitely many solutions"
comes from.

**Rank–nullity.** Count dimensions from the map picture: the input space
$\mathbb{R}^n$ has $n$ dimensions; $\dim N(A)$ of them are flattened away (the
**nullity**); the survivors become the output's independent directions, and there
are $r$ of those. Nothing else can happen to a dimension — kept or killed — so

$$\text{rank} + \text{nullity} = n \qquad\text{(rank–nullity theorem)}.$$

This is the week's paper derivation #3; make the counting argument careful on
paper (start from a basis of $N(A)$, extend it to a basis of $\mathbb{R}^n$, show
the images of the added vectors are independent and span $C(A)$).

### 6.3 The other two, and the big picture

Transposing $A$ gives two more subspaces: the **row space** $C(A^\top) \subseteq
\mathbb{R}^n$ (span of the rows) and the **left null space** $N(A^\top) \subseteq
\mathbb{R}^m$. A fundamental fact ties the input-side pair together:

$$N(A) \perp C(A^\top)\text{:} \quad Ax = 0 \text{ says every row dots to zero with } x,$$

so null-space vectors are orthogonal to every row, hence to everything the rows
span — one line of proof. Applying the same line to $A^\top$: $N(A^\top) \perp
C(A)$. And the row space has the same dimension $r$ as the column space (row rank
= column rank — Strang's lectures prove it; Week 07's SVD will make it obvious).
The complete inventory — **the four fundamental subspaces**:

| subspace | lives in | dimension | orthogonal to |
|---|---|---|---|
| row space $C(A^\top)$ | $\mathbb{R}^n$ (inputs) | $r$ | $N(A)$ |
| null space $N(A)$ | $\mathbb{R}^n$ (inputs) | $n - r$ | $C(A^\top)$ |
| column space $C(A)$ | $\mathbb{R}^m$ (outputs) | $r$ | $N(A^\top)$ |
| left null space $N(A^\top)$ | $\mathbb{R}^m$ (outputs) | $m - r$ | $C(A)$ |

Input space splits into row space ⊕ null space; output space into column space ⊕
left null space. Every input decomposes into "the part the matrix acts on" plus
"the part it kills". Strang draws this as one figure — the "big picture of linear
algebra" — worth reproducing by hand in your notes.

### 6.4 Computing them

For small matrices, elimination by hand finds all four (the exercises walk one).
Numerically:

```python
A = np.array([[1.0, 2.0, 0.0, 1.0],
              [2.0, 4.0, 1.0, 1.0],
              [3.0, 6.0, 1.0, 2.0]])       # 3x4; row3 = row1 + row2 -> rank 2
print(np.linalg.matrix_rank(A))            # 2
v = np.array([-2.0, 1.0, 0.0, 0.0])        # claim: in the null space (col2 = 2*col1)
print(np.linalg.norm(A @ v))               # ~0 — verified
```

`np.linalg.matrix_rank` and (from SciPy) `scipy.linalg.null_space` do the heavy
lifting; *always* verify a claimed null vector by checking $\|Av\| \approx 0$.

**Physics: what a detector cannot see.** Suppose a cost-saving readout sums each
pair of adjacent calorimeter channels: 4 true channel energies, 2 readout values —
a $2 \times 4$ response matrix of rank 2 with a 2-dimensional null space. Any true
energy pattern lying in that null space (e.g. $+\epsilon$ in channel 1,
$-\epsilon$ in channel 2) produces *identical* readout. No analysis, however
clever, can recover it: the information was never recorded. Computing $N(A)$ of a
response matrix tells you, before taking any data, exactly which questions your
detector cannot answer. That is rank–nullity earning a salary.

## 7. When $Ax = b$ has no solution: least squares and projection

### 7.1 Overdetermined systems

Return to Week 05's *real* calibration: six sources, two unknowns — six equations,
$A$ is $6 \times 2$ (columns: the energies $E$, and all-ones for the intercept).
With noisy measurements, $b$ will not lie in the 2-dimensional column space of $A$
inside $\mathbb{R}^6$: **no exact solution exists**. Such systems — more equations
than unknowns — are **overdetermined**, and they are the normal situation in
science: measurements outnumber parameters, and noise breaks exactness.

The fix: stop demanding $Ax = b$ and ask for the $\hat{x}$ that makes $A\hat{x}$
*closest* to $b$:

$$\hat{x} = \text{argmin}_x\, \|Ax - b\|^2$$

("argmin": the argument that minimizes). Minimizing $\|Ax - b\|^2$ — the sum of
squared per-equation misses — is exactly Week 05's mean-squared-error fit, seen
geometrically.

### 7.2 The normal equations, from geometry

As $x$ ranges over all of $\mathbb{R}^n$, the point $Ax$ ranges over exactly the
column space $C(A)$. So the question is: *which point $p$ in the subspace $C(A)$
is closest to $b$?* Geometry answers: the foot of the perpendicular — the
**orthogonal projection** of $b$ onto $C(A)$. Closest means the error
$e = b - A\hat{x}$ sticks out *perpendicular to the subspace* (if $e$ had any
component along the subspace, sliding $p$ that way would shrink the distance).
Perpendicular to the whole subspace means orthogonal to every column of $A$:

$$A^\top (b - A\hat{x}) = 0
\qquad\Longleftrightarrow\qquad
\boxed{\,A^\top A\, \hat{x} = A^\top b\,}$$

— the **normal equations** ("normal" = perpendicular, naming the geometry). This
is a *square* $n \times n$ system, solvable by §5. Note what just happened: no
calculus, no descent — pure right angles. (Week 07 re-derives the same result by
matrix calculus; agreeing routes are how you know you believe a formula.) If $A$'s
columns are independent (rank $n$), $A^\top A$ is invertible and
$\hat{x} = (A^\top A)^{-1} A^\top b$ uniquely.

Also notice: $e = b - A\hat x$ is orthogonal to every column, i.e.
$e \in N(A^\top)$ — the left null space, one of §6's four, doing concrete work:
output space splits into $C(A)$ (the part your model family can explain) and
$N(A^\top)$ (the residual's home).

### 7.3 The projection matrix

The closest point itself is $p = A\hat{x} = A(A^\top A)^{-1} A^\top b$. The
machine that takes *any* $b$ to its shadow on the column space is the
**projection matrix**

$$P = A(A^\top A)^{-1} A^\top,$$

this week's paper derivation #1. Two properties characterize projections — derive
both on paper by writing $P$ out:

- $P^2 = P$: projecting a second time changes nothing (the shadow of a shadow is
  itself).
- $P^\top = P$: symmetric.

### 7.4 The calibration fit, solved exactly

```python
E = np.array([0.51, 0.66, 0.84, 1.17, 1.27, 1.33])
rng = np.random.default_rng(42)
y = 5.0 * E + 2.0 + rng.normal(0, 0.1, size=E.size)

A = np.stack([E, np.ones(E.size)], axis=1)      # 6x2 design matrix: columns E, 1

# Route A: normal equations
x_ne = np.linalg.solve(A.T @ A, A.T @ y)

# Route B: library least squares
x_ls = np.linalg.lstsq(A, y, rcond=None)[0]

print(x_ne)     # [~4.98  ~2.03]
print(x_ls)     # same
r = y - A @ x_ne
print(A.T @ r)  # ~[0, 0] — residual orthogonal to both columns, as derived
```

Same $(m, b)$ that Week 05's gradient descent crawled toward over 2000 steps —
now in one line, exactly, and you know *why*: the fit is a projection.
(`np.linalg.lstsq` solves the same minimization by a more numerically stable
route; Week 07 explains what can go wrong with forming $A^\top A$ and what
`lstsq` does instead. For fits, prefer `lstsq`.)

Gradient descent is not thereby retired: it wins the moment the model stops being
linear in its parameters — which is Phase 2 onward, i.e. most of your future.

## Check yourself

1. Compute $(2, -1, 3) \cdot (1, 4, 0)$, and state what a dot product of zero
   would have meant geometrically.
2. A matrix $M$ sends $e_1 \to (0, 2)$ and $e_2 \to (-1, 0)$. Write $M$, and
   describe its action in words.
3. Give the three viewpoints of matrix multiplication, and say which one writes
   $AB$ as a sum of rank-1 pieces.
4. Without computing: can $\begin{pmatrix}1 & 2\\ 2 & 4\end{pmatrix}x = \begin{pmatrix}1\\ 3\end{pmatrix}$
   be solved? Why not, in column-space language?
5. A $5 \times 7$ matrix has rank 4. Give the dimensions of all four fundamental
   subspaces, and the two orthogonality relations.
6. Your detector's response matrix has a nonzero null space. What does that mean
   physically, and can more data fix it?
7. Derive the normal equations in two sentences, from the geometry.
8. Why does the six-point calibration have no exact solution, while the two-point
   version did?

## Answers

1. $2 - 4 + 0 = -2$. Zero would have meant the vectors are orthogonal
   (perpendicular arrows).
2. Columns are where the basis lands: $M = \begin{pmatrix}0 & -1\\ 2 & 0\end{pmatrix}$.
   It rotates by 90° and stretches by 2 (rotation composed with scaling).
3. Entries = row·column dot products; columns = $A$ applied to each column of
   $B$; sum over $k$ of (column $k$ of $A$)(row $k$ of $B$) — the outer-product
   viewpoint is the rank-1 sum.
4. No. Column 2 is twice column 1, so the column space is the line through
   $(1, 2)$ — and $(1, 3)$ is not on that line, hence unreachable.
5. Row space: 4 (in $\mathbb{R}^7$); null space: $7 - 4 = 3$; column space: 4 (in
   $\mathbb{R}^5$); left null space: $5 - 4 = 1$. $N(A) \perp C(A^\top)$ and
   $N(A^\top) \perp C(A)$.
6. Some distinct true signal patterns produce identical readout — the detector
   literally cannot distinguish them. No amount of repeated data helps; the
   information is never recorded. (Redesigning the readout — changing $A$ — is
   the only fix.)
7. $Ax$ can only reach the column space, so the best $A\hat{x}$ is the point of
   $C(A)$ closest to $b$ — the orthogonal projection. Closest forces the residual
   perpendicular to every column: $A^\top(b - A\hat{x}) = 0$, i.e.
   $A^\top A\hat{x} = A^\top b$.
8. Two points determine a line exactly (2 equations, 2 unknowns, independent
   columns). Six noisy points give 6 equations in 2 unknowns; the noisy $b$ falls
   outside the 2-dimensional column space in $\mathbb{R}^6$, so no line passes
   through all six points — only a closest line exists.

## New terms

- **vector / dimension / $\mathbb{R}^n$** — ordered list of $n$ numbers; the space of all of them.
- **scalar** — a single plain number.
- **linear combination** — $au + bv + \cdots$: stretch and add.
- **norm $\|v\|$ / unit vector / normalizing** — length; length-1 vector; dividing by the norm.
- **dot product** — $\sum_i u_i v_i = \|u\|\|v\|\cos\theta$; measures alignment.
- **orthogonal** — perpendicular; dot product zero.
- **matrix / $A_{ij}$** — rectangular number grid; entry in row $i$, column $j$.
- **matrix–vector product** — linear combination of $A$'s columns weighted by $v$.
- **standard basis $e_1, e_2, \dots$** — the unit coordinate vectors; $Ae_j$ = column $j$.
- **linear map / linearity** — respects addition and scaling; same thing as a matrix.
- **rotation / shear / scaling / projection matrices** — the 2×2 gallery.
- **matrix multiplication** — composition of maps; three viewpoints.
- **outer product** — $u w^\top$, a rank-1 matrix.
- **transpose $A^\top$** — rows and columns swapped; $(AB)^\top = B^\top A^\top$.
- **identity / inverse matrix** — do-nothing machine; the machine that undoes $A$.
- **linear system $Ax = b$** — which input maps to $b$; one equation per row.
- **Gaussian elimination** — eliminate variables, substitute back; `np.linalg.solve`.
- **singular matrix / determinant** — collapses space, no inverse; the area/volume scale factor whose vanishing detects collapse.
- **span / linear independence / subspace / basis / dimension** — all combinations; no redundancy; closed under combinations; minimal spanning set; its size.
- **column space $C(A)$ / null space $N(A)$** — reachable outputs; inputs sent to zero.
- **rank / nullity / rank–nullity** — $\dim C(A)$; $\dim N(A)$; they sum to $n$.
- **row space / left null space** — $C(A^\top)$, $N(A^\top)$; the other two fundamental subspaces.
- **overdetermined system** — more equations than unknowns; solved in the least-squares sense.
- **least squares / normal equations** — minimize $\|Ax - b\|^2$; $A^\top A\hat{x} = A^\top b$.
- **orthogonal projection / projection matrix $P$** — closest point in a subspace; $A(A^\top A)^{-1}A^\top$, with $P^2 = P$, $P^\top = P$.
- **design matrix** — the data-built matrix in a fit (here: column of energies, column of ones).
- **residual (vector)** — $b - A\hat{x}$; lives in the left null space at the optimum.

## Going deeper

- 3Blue1Brown, *Essence of Linear Algebra*, chapters 1–9 and 13 (vectors through
  dot products and cross products; change of basis) — the week's spine. Watch fast;
  the point is the geometry, and the animations are the pictures §3 describes in
  words. (Chapter 14, eigenvectors, is next week's.)
- Strang, MIT 18.06 (OpenCourseWare): Lecture 6 (column space and nullspace),
  Lecture 9 (independence, basis, dimension), Lecture 10 (the four fundamental
  subspaces), Lectures 15–16 (projections and least squares). Skim Lectures 1–3
  only if elimination feels shaky.
- Optional reference: Strang, *Introduction to Linear Algebra* — find the "big
  picture of linear algebra" figure of the four subspaces and copy it into your
  notes by hand.
