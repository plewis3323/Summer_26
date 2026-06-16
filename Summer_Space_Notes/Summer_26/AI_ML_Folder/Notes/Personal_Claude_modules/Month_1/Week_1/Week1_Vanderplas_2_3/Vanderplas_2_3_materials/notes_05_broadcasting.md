---
# Notes 05 — Computation on Arrays: Broadcasting (Vanderplas Ch. 2.5)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Broadcasting is simply a set of rules for applying binary ufuncs (addition, subtraction, multiplication, etc.) on arrays of **different sizes**. For equal-sized arrays, ufuncs operate element-by-element. Broadcasting extends this so that smaller arrays are conceptually "stretched" to match the shape of the larger one — but the stretching is virtual: NumPy does **not** actually duplicate data in memory, which keeps broadcasting efficient.

## Key ideas / idioms

- Same-size arrays operate element-wise: `a + b` matches positions directly.
- A scalar broadcasts against an array as if the scalar were stretched to the array's shape (e.g., `a + 5`).
- Broadcasting also stretches **both** operands when needed (e.g., a column vector plus a row vector → a 2D grid).
- Use `np.newaxis` (or `reshape`) to add a length-1 axis when you need to control *which* dimension stretches.

**The three rules of broadcasting:**

$$
\textbf{Rule 1:}\quad \text{If two arrays differ in number of dimensions, the shape of the array with fewer dimensions is padded with ones on its \emph{leading (left) side}.}
$$

$$
\textbf{Rule 2:}\quad \text{If the shapes disagree in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.}
$$

$$
\textbf{Rule 3:}\quad \text{If in any dimension the sizes disagree and neither is equal to 1, an error is raised.}
$$

## Worked code examples (runnable)

```python
import numpy as np

# Element-wise on same-size arrays
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)        # [5 6 7]

# Scalar broadcast (scalar "stretched" to shape (3,))
print(a + 5)        # [5 6 7]
```

```python
import numpy as np

# Example 1: 2D + 1D
# M.shape = (2, 3), a.shape = (3,)
#   Rule 1: a -> (1, 3)
#   Rule 2: a -> (2, 3)
M = np.ones((2, 3))
a = np.arange(3)
print(M + a)
# [[1. 2. 3.]
#  [1. 2. 3.]]
```

```python
import numpy as np

# Example 2: both arrays broadcast
# a.shape = (3, 1), b.shape = (3,)
#   Rule 1: b -> (1, 3)
#   Rule 2: a -> (3, 3), b -> (3, 3)
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a + b)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

```python
import numpy as np

# Example 3: incompatible shapes -> error
# M.shape = (3, 2), a.shape = (3,)
#   Rule 1: a -> (1, 3)
#   Rule 2: a -> (3, 3)  ... but M is (3, 2): mismatch -> Rule 3 error
M = np.ones((3, 2))
a = np.arange(3)
try:
    M + a
except ValueError as e:
    print("ValueError:", e)
# ValueError: operands could not be broadcast together with shapes (3,2) (3,)

# Fix: add a trailing axis so a becomes (3, 1), which broadcasts to (3, 2)
print(a[:, np.newaxis].shape)   # (3, 1)
print(M + a[:, np.newaxis])
# [[1. 1.]
#  [2. 2.]
#  [3. 3.]]
```

```python
import numpy as np

# Broadcasting in practice 1: centering an array
X = np.random.random((10, 3))
Xmean = X.mean(0)            # mean per column, shape (3,)
X_centered = X - Xmean       # (3,) broadcasts across all 10 rows
print(np.allclose(X_centered.mean(0), 0))  # True (means ~ [0, 0, 0])
```

```python
import numpy as np

# Broadcasting in practice 2: evaluating a 2D function on a grid
x = np.linspace(0, 5, 50)                 # shape (50,)
y = np.linspace(0, 5, 50)[:, np.newaxis]  # shape (50, 1)
# x -> (1, 50), y -> (50, 1) broadcast to (50, 50)
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
print(z.shape)   # (50, 50)

# (Visualization, requires matplotlib):
# import matplotlib.pyplot as plt
# plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
# plt.colorbar()
```

## Why this matters / intuition

Broadcasting lets you write vectorized expressions over arrays of mismatched shapes without manually tiling/looping data. The mental model: align shapes from the **right**, pad the shorter shape with leading ones, then any axis of size 1 gets stretched to match its partner. Because the stretch is virtual (no real copies), you get loop-free, memory-efficient code — invaluable for operations like mean-centering feature matrices and evaluating functions over a coordinate grid.

## Gotchas

- **Padding is on the left only.** Rule 1 always prepends ones. So `(3,)` becomes `(1, 3)`, never `(3, 1)`. That is why `np.ones((3, 2)) + np.arange(3)` fails — `a` becomes `(1, 3)` → `(3, 3)`, which disagrees with `(3, 2)`.
- **Fix mismatches with `np.newaxis`.** To make a 1D array broadcast down columns instead of across rows, give it a trailing axis: `a[:, np.newaxis]` → `(3, 1)`.
- A dimension only stretches if it is exactly **1**; any other disagreement is a hard `ValueError` (Rule 3).
- *(added context)* Broadcasting rules apply to all binary ufuncs uniformly, not just `+` — the same logic governs `*`, `-`, comparisons, etc.

## Suggested figure (optional)

A "broadcasting stretch" diagram: three side-by-side panels, each showing two input grids and a result grid, with light-shaded cells marking duplicated (stretched) values.
- Panel 1 (array + scalar): a 1x3 row of filled boxes plus a single scalar box; the scalar is shown faintly copied into a 1x3 row, producing a 1x3 result.
- Panel 2 (2D + 1D row): a 2x3 grid plus a 1x3 row; the row is shown faintly replicated down to 2 rows, yielding a 2x3 result.
- Panel 3 (column + row): a 3x1 column plus a 1x3 row; the column is faintly tiled rightward and the row faintly tiled downward, both meeting in a full 3x3 result grid.
The faint/ghosted cells should visually emphasize that the stretched values are conceptual copies, not stored in memory.
---

## 💬 Q&A (captured during session)

### Q: Simplify the three broadcasting rules.

**Plain-English version:**

1. **Different number of dimensions?** Pad the smaller array's shape with 1s on the **left** until they match.
2. **A dimension is 1?** Stretch it to match the other array's size in that dimension.
3. **Sizes differ and neither is 1?** Error — they can't broadcast.

**Even shorter:** broadcasting works if, in every dimension, the sizes are either **equal** or **one of them is 1**.

```python
import numpy as np

# (3,) + scalar      -> pad & stretch    OK
np.arange(3) + 5                 # [5 6 7]

# (3,3) + (3,)       -> pad left to (1,3), stretch to (3,3)   OK
np.ones((3, 3)) + np.arange(3)

# (3,) + (4,)        -> 3 vs 4, neither is 1   ERROR
# np.arange(3) + np.arange(4)
```

**Mental picture:** line the shapes up **right-to-left**; a `1` (or a missing/padded slot) gets copied to fill the bigger size, but two real, unequal numbers clash.

```
  (3, 3)          (3,)            (3,)
+    (3)   ->   +  (4,)   ->    pad: (1, 3)
---------      --------         stretch: (3, 3)   OK
  (3, 3)        clash
```
