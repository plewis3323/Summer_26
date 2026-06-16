---
# Notes 07 — Fancy Indexing (Vanderplas Ch. 2.7)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Fancy indexing is just like the simple indexing seen earlier (`x[0]`, `x[:5]`), except we pass **arrays of indices in place of single scalars**. This lets us access and modify many array elements at once. The key rule: **the shape of the result reflects the shape of the *index arrays*, not the shape of the array being indexed.**

## Key ideas / idioms

- Pass a list/array of indices: `x[[3, 7, 4]]` grabs three elements in one shot.
- Result shape follows the index shape:
$$\text{shape}(x[\mathrm{ind}]) = \text{shape}(\mathrm{ind})$$
- In multiple dimensions, the first index array selects rows, the second selects columns; they are **paired** element-wise: `X[row, col]` gives `[X[row[0],col[0]], X[row[1],col[1]], ...]`.
- Index arrays combine under **broadcasting rules** — a column vector of rows against a row vector of columns produces a 2D result.
- Fancy indexing mixes with the other indexing styles: simple indices, slices, and boolean masks.
- Fancy indices can be used on the **left-hand side** of an assignment to modify values.
- Repeated indices with augmented assignment (`+=`, etc.) do **not** accumulate — use `np.add.at()` for that.

## Worked code examples (runnable)

```python
import numpy as np

# --- 1D fancy indexing ---
x = np.array([51, 92, 14, 71, 60, 20, 82, 86, 74, 74])
ind = [3, 7, 4]
print(x[ind])              # [71 86 60]

# Result shape mirrors the index shape, not x's shape
ind2 = np.array([[3, 7],
                 [4, 5]])
print(x[ind2])             # [[71 86]
                           #  [60 20]]

# --- Multidimensional fancy indexing (paired) ---
X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(X[row, col])         # [ 2  5 11]

# Broadcasting of index arrays -> 2D result
print(X[row[:, np.newaxis], col])
# Each row index paired with each col index:
# [[ 2  1  3]
#  [ 6  5  7]
#  [10  9 11]]

# --- Combined indexing ---
print(X[2, [2, 0, 1]])     # simple + fancy -> [10  8  9]
print(X[1:, [2, 0, 1]])    # slice + fancy
mask = np.array([1, 0, 1, 0], dtype=bool)
print(X[row[:, np.newaxis], mask])   # fancy + boolean mask

# --- Modifying values ---
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)                   # [ 0 99 99  3 99  5  6  7 99  9]
x[i] -= 10
print(x)                   # [ 0 89 89  3 89  5  6  7 89  9]

# Gotcha: repeated indices do NOT accumulate
x = np.zeros(10)
x[[0, 0]] = [4, 6]         # 6 wins (last assignment), not 10
print(x[0])                # 6.0

i = [2, 3, 3, 4, 4, 4]
x = np.zeros(10)
x[i] += 1                  # each slot incremented once, not by repeat count
print(x)                   # index 3 and 4 are only 1.0

# --- np.add.at() does true repeated accumulation ---
x = np.zeros(10)
np.add.at(x, i, 1)
print(x)   # [0. 0. 1. 2. 3. 0. 0. 0. 0. 0.]

# --- Binning data by hand (manual histogram) ---
np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
idx = np.searchsorted(bins, x)   # which bin each point falls into
np.add.at(counts, idx, 1)        # accumulate counts
print(counts)
```

## Why this matters / intuition

Fancy indexing is the building block that lets you express "gather these specific, scattered elements" or "scatter these values into these specific slots" as a single vectorized operation — no Python loop. Once you understand it together with `np.add.at()` and `np.searchsorted()`, you can hand-build specialized routines (like the binning/histogram above) when NumPy's convenience functions don't precisely match your need. For small datasets the hand-rolled version can even beat `np.histogram()`, though the built-in optimizes better for large arrays.

## Gotchas

- **Result shape = index shape**, not array shape. A 2D index array yields a 2D result even from a 1D array.
- **Pairing, not cross-product**, for `X[row, col]` unless you deliberately broadcast (e.g. `row[:, np.newaxis]`).
- **Repeated indices with `+=`/`x[i] += 1` do not accumulate.** Augmented assignment is `x[i] = x[i] + 1` evaluated once — repeats are silently lost. Use `np.add.at(x, i, 1)` for true per-occurrence accumulation.
- Plain assignment with duplicate indices keeps only the **last** write.

## Suggested figure (optional)

A side-by-side diagram of the manual binning pipeline: a strip of sample points along a number line, vertical bin edges from `np.linspace`, arrows from each point to its `searchsorted` bin, and the resulting `counts` bar chart — visually showing how `np.add.at` scatters each point into its bin to form a histogram.

---

## 💬 Q&A (captured during session)

### Q: What is fancy indexing?

**Fancy indexing** means passing an **array (or list) of indices** to grab *multiple* arbitrary elements at once — instead of a single index or a contiguous slice. The result takes the **shape of the index array**, not the shape of the array being indexed.

**The three ways to index, compared:**

```python
import numpy as np
x = np.array([10, 20, 30, 40, 50, 60])

x[2]            # simple    -> 30           (one element)
x[1:4]          # slicing   -> [20 30 40]   (a contiguous run)
x[[0, 2, 5]]    # fancy     -> [10 30 60]   (any elements you list)
```

That last one is fancy indexing: hand it a list of positions and it pulls exactly those, in that order, with repeats allowed (`x[[0, 0, 5]]` → `[10 10 60]`).

**Result shape follows the *index* shape:**

```python
ind = np.array([[0, 1],
                [2, 5]])
x[ind]          # -> [[10 20]
                #     [30 60]]   <- 2x2, same shape as ind
```

**Multiple dimensions (indices pair up):**

```python
M = np.arange(12).reshape(3, 4)
rows = np.array([0, 1, 2])
cols = np.array([1, 3, 0])
M[rows, cols]   # -> [1, 7, 8]   picks (0,1), (1,3), (2,0)
```

Row and column index arrays are matched element-by-element (and broadcast together if shapes differ).

**Assignment works too:**

```python
x = np.zeros(5)
x[[0, 2, 4]] = 9
x               # -> [9. 0. 9. 0. 9.]
```

**Why it matters / the gotcha:** fancy indexing is how you **select, reorder, or scatter** arbitrary elements in one vectorized shot — sampling specific rows, reordering by `argsort`, building histograms. One trap: repeated indices in an augmented assignment like `x[[0, 0]] += 1` only add **once** (it's "fetch → assign," not accumulate) — use `np.add.at(x, [0, 0], 1)` for true accumulation.

### Q: What is seeding, and what does `RandomState` do?

Computers don't make true randomness — they compute **pseudo-random** numbers with a formula
that needs a starting input. That starting input is the **seed**, and "seeding" means setting
it. From one seed the formula cranks out a long stream that *looks* random but is fully
determined:

```
seed ─► formula ─► n1 ─► formula ─► n2 ─► formula ─► n3 ─► ...
```

Same seed → identical sequence every run (**reproducible**); different seed → a different but
equally fixed sequence.

```python
import numpy as np
np.random.seed(42); print(np.random.randint(0, 10, 5))   # [6 3 7 4 6]
np.random.seed(42); print(np.random.randint(0, 10, 5))   # [6 3 7 4 6]  ← repeat
np.random.seed(7);  print(np.random.randint(0, 10, 5))   # [4 9 6 3 3]  ← different
```

**`np.random.RandomState(seed)`** bakes a seed into your **own private generator object** —
an independent dice-roller, isolated from the global one so other code can't disturb its
sequence. That's why Vanderplas writes `rand = np.random.RandomState(42)`: the book's "random"
arrays come out identical for every reader.

```python
np.random.seed(42)                  # seeds the shared GLOBAL generator
rand = np.random.RandomState(42)    # your OWN independent, seeded generator
rand.randint(0, 10, 5)              # draws only from yours
```

- `np.random.randint(...)` (no object) → draws from one hidden **global** generator shared
  program-wide; anyone calling `np.random.seed(...)` elsewhere can change your results.
- `np.random.RandomState(42)` → a **private** generator you control.
- *(added context)* Modern NumPy (≥1.17) prefers `rng = np.random.default_rng(42)`, a newer
  `Generator` with a better algorithm. Same idea; the book predates it.

**One line:** seeding fixes the generator's starting number so its "random" stream is exactly
repeatable; `RandomState`/`default_rng` give you a private, seeded generator to do it cleanly.
