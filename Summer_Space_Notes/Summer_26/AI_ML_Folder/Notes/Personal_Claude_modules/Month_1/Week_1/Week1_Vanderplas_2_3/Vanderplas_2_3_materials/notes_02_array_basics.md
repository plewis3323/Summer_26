---
# Notes 02 — The Basics of NumPy Arrays (Vanderplas Ch. 2.2)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
This section covers manipulating NumPy arrays once you have them. It groups operations into five basic categories: inspecting **array attributes**, **indexing** single elements, **slicing** subarrays, **reshaping**, and **joining/splitting** arrays. The recurring theme is that NumPy arrays are fixed-type, fixed-size, contiguous data, which makes them fast but introduces behaviors (silent truncation, views-not-copies) that differ from Python lists.

## Key ideas / idioms
- **Attributes describe structure:** every array exposes `ndim` (number of dimensions), `shape` (tuple of per-axis sizes), `size` (total element count), `dtype` (element type), `itemsize` (bytes per element), and `nbytes` (total bytes). Note $$\text{nbytes} = \text{size} \times \text{itemsize}.$$
- **Indexing** uses brackets; multidimensional access uses a comma-separated tuple `x2[row, col]`. Negative indices count from the end.
- **Fixed type:** assigning a float into an int array silently **truncates** it — no automatic upcasting like a Python list.
- **Slicing** uses `x[start:stop:step]` (defaults: `start=0`, `stop=size`, `step=1`). A negative `step` reverses; a common idiom for full reversal is `x[::-1]`.
- **Slices are VIEWS, not copies.** Modifying a slice modifies the original array. Use `.copy()` to break the link. (This is what makes working with large datasets cheap — no implicit copying.)
- **Reshaping** with `reshape` requires the new shape's size to match; `np.newaxis` (or `reshape`) converts a 1D array into a row or column vector.
- **Concatenation:** `np.concatenate` joins along an existing axis (default `axis=0`); `np.vstack`/`np.hstack`/`np.dstack` stack along axes 0/1/2 and handle mixed dimensions cleanly.
- **Splitting** is the inverse: `np.split` (with `np.hsplit`/`np.vsplit`/`np.dsplit`); $N$ split-points yield $N+1$ subarrays.

## Worked code examples (runnable)

```python
import numpy as np

# --- Array attributes ---
np.random.seed(0)
x1 = np.random.randint(10, size=6)         # 1D
x2 = np.random.randint(10, size=(3, 4))    # 2D
x3 = np.random.randint(10, size=(3, 4, 5)) # 3D

print("x3 ndim: ", x3.ndim)        # 3
print("x3 shape:", x3.shape)       # (3, 4, 5)
print("x3 size: ", x3.size)        # 60
print("dtype:   ", x3.dtype)       # int64 (platform-dependent)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:  ", x3.nbytes, "bytes")  # == size * itemsize
```

```python
import numpy as np

# --- Indexing ---
x1 = np.array([5, 0, 3, 3, 7, 9])
print(x1[0], x1[4], x1[-1])   # 5 7 9

x2 = np.array([[3, 5, 2, 4],
               [7, 6, 8, 8],
               [1, 6, 7, 7]])
print(x2[0, 0], x2[2, -1])    # 3 7
x2[0, 0] = 12                 # modify in place

# Fixed-type quirk: float is truncated when stored in an int array
x1[0] = 3.14159
print(x1)                     # [3 0 3 3 7 9]
```

```python
import numpy as np

# --- Slicing (1D) ---
x = np.arange(10)
print(x[:5])     # [0 1 2 3 4]
print(x[5:])     # [5 6 7 8 9]
print(x[4:7])    # [4 5 6]
print(x[::2])    # [0 2 4 6 8]
print(x[1::2])   # [1 3 5 7 9]
print(x[::-1])   # [9 8 7 6 5 4 3 2 1 0]
print(x[5::-2])  # [5 3 1]

# --- Slicing (multi-dim) ---
x2 = np.array([[12, 5, 2, 4],
               [7, 6, 8, 8],
               [1, 6, 7, 7]])
print(x2[:2, :3])      # first 2 rows, first 3 cols
print(x2[:3, ::2])     # all rows, every other col
print(x2[::-1, ::-1])  # reverse both axes
print(x2[:, 0])        # first column -> [12 7 1]
print(x2[0])           # first row (same as x2[0, :]) -> [12 5 2 4]
```

```python
import numpy as np

# --- Views vs copies ---
x2 = np.array([[12, 5, 2, 4],
               [7, 6, 8, 8],
               [1, 6, 7, 7]])
x2_sub = x2[:2, :2]
x2_sub[0, 0] = 99
print(x2[0, 0])        # 99  <- original changed (slice was a view)

x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy[0, 0] = 42
print(x2[0, 0])        # still 99 <- copy is independent
```

```python
import numpy as np

# --- Reshaping ---
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

x = np.array([1, 2, 3])
print(x[np.newaxis, :])  # row vector, shape (1, 3)
print(x[:, np.newaxis])  # column vector, shape (3, 1)
```

```python
import numpy as np

# --- Concatenation ---
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
print(np.concatenate([x, y]))            # [1 2 3 3 2 1]
print(np.concatenate([x, y, [99,99,99]]))

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(np.concatenate([grid, grid]))          # stack rows (axis=0)
print(np.concatenate([grid, grid], axis=1))  # stack cols (axis=1)

# Mixed dimensions: use vstack / hstack
row = np.array([1, 2, 3])
g = np.array([[9, 8, 7],
              [6, 5, 4]])
print(np.vstack([row, g]))
col = np.array([[99], [99]])
print(np.hstack([g, col]))
```

```python
import numpy as np

# --- Splitting ---
x = np.array([1, 2, 3, 99, 99, 3, 2, 1])
a, b, c = np.split(x, [3, 5])   # 2 split-points -> 3 subarrays
print(a, b, c)                  # [1 2 3] [99 99] [3 2 1]

grid = np.arange(16).reshape((4, 4))
upper, lower = np.vsplit(grid, [2])
left, right = np.hsplit(grid, [2])
print(upper); print(lower)
print(left);  print(right)
```

## Why this matters / intuition
Almost all real data work — loading datasets, selecting features/rows, building batches, assembling matrices — reduces to these five operations. Understanding that **slices are views** is the single most important takeaway: it makes NumPy efficient (selecting a million-row chunk costs nothing), but it also means careless slicing can silently mutate your source data. Knowing when NumPy gives you a window versus a fresh array is the difference between fast, correct code and subtle data-corruption bugs.

## Gotchas
- **Silent truncation:** putting a float in an int array drops the fractional part with no warning. Choose `dtype` deliberately.
- **Views, not copies:** `x[:2, :2]` shares memory with `x`. Mutating it mutates the original — use `.copy()` when you need independence. (Note: *fancy* indexing, covered later, returns copies, not views — a separate behavior.) (added context)
- **Split-point counting:** `np.split(x, [3, 5])` returns *3* pieces, not 2 — $N$ indices give $N+1$ slices.
- **`reshape` size must match:** the product of the new shape must equal the original `size`, or it errors. (`-1` can be used to infer one dimension automatically.) (added context)
- **`dtype`/`nbytes` are platform-dependent:** default integer width may print as `int64` or `int32` depending on OS/build. (added context)

## Suggested figure (optional)
A side-by-side diagram of a small 2D array and a highlighted slice, with two arrows: one labeled "view (shares memory — edits propagate back)" pointing from slice to original, and one labeled ".copy() (independent block — edits stay local)" showing a separate detached memory region. This visually anchors the views-vs-copies distinction.
