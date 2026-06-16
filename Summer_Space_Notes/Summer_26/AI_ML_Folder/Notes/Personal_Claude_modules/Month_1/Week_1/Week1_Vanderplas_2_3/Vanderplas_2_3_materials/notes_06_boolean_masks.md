---
# Notes 06 — Comparisons, Masks, and Boolean Logic (Vanderplas Ch. 2.6)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Boolean masking is the technique for examining and manipulating array values **based on a criterion**: "Masking comes up when you want to extract, modify, count, or otherwise manipulate values in an array based on some criterion." The workflow is two-step: (1) build a Boolean array with comparison operators (themselves implemented as element-wise ufuncs), then (2) count, test, combine, or index with that Boolean array. VanderPlas motivates it with Seattle 2014 rainfall data — answering "how many rainy days?" or "what's the median precipitation on rainy days?" without slow Python loops.

## Key ideas / idioms

- **Comparisons are ufuncs.** `<`, `>`, `<=`, `>=`, `==`, `!=` all act element-wise and return a Boolean array of the same shape. They work on compound expressions too, e.g. `(2 * x) == (x ** 2)`.

  | Operator | ufunc equivalent |
  |----------|------------------|
  | `==` | `np.equal` |
  | `!=` | `np.not_equal` |
  | `<`  | `np.less` |
  | `<=` | `np.less_equal` |
  | `>`  | `np.greater` |
  | `>=` | `np.greater_equal` |

- **Counting True values:** `np.count_nonzero(mask)` or `np.sum(mask)` (since `False`→0, `True`→1). `np.sum` accepts an `axis`, so `np.sum(x < 6, axis=1)` counts per row.

  $$\text{count} = \sum_i \mathbb{1}[\,\text{condition}_i\,]$$

- **Testing:** `np.any(mask)` ("are *any* True?") and `np.all(mask)` ("are *all* True?"), both axis-aware.

- **Combine conditions with bitwise operators** `&`, `|`, `^`, `~` — **not** `and`/`or`. Always parenthesize: `(inches > 0.5) & (inches < 1)`, because operator precedence would otherwise evaluate the bare expression incorrectly.

  | Operator | ufunc equivalent |
  |----------|------------------|
  | `&` | `np.bitwise_and` |
  | `\|` | `np.bitwise_or` |
  | `^` | `np.bitwise_xor` |
  | `~` | `np.bitwise_not` |

- **Masking = Boolean indexing:** `x[mask]` returns a 1D array of only the elements where `mask` is `True`, e.g. `x[x < 5]`.

- **`and`/`or` vs `&`/`|` (the critical distinction):**
  - `and`/`or` evaluate the truth value of an **entire object**. On a multi-element array this is ambiguous and raises a `ValueError`.
  - `&`/`|` operate **element-wise** (bit-by-bit). "For Boolean NumPy arrays, the latter is nearly always the desired operation."

## Worked code examples (runnable)

```python
import numpy as np

# --- Comparisons return Boolean arrays ---
x = np.array([1, 2, 3, 4, 5])
print(x < 3)              # [ True  True False False False]
print(x >= 3)             # [False False  True  True  True]
print(x == 3)             # [False False  True False False]
print((2 * x) == (x ** 2))  # element-wise comparison of expressions

# --- Counting True entries ---
rng = np.random.RandomState(0)
M = rng.randint(0, 10, (3, 4))
print(M)
print(np.count_nonzero(M < 6))   # total entries < 6
print(np.sum(M < 6))             # same: True->1, False->0
print(np.sum(M < 6, axis=1))     # count per row

# --- Testing conditions ---
print(np.any(M > 8))             # any value > 8?
print(np.all(M < 10))            # all values < 10?
print(np.all(M < 8, axis=1))     # per-row check

# --- Compound conditions (note the parentheses!) ---
between = np.sum((M > 2) & (M < 6))
print("entries in (2, 6):", between)
# De Morgan equivalent using ~ and |
print(np.sum(~((M <= 2) | (M >= 6))))

# --- Boolean arrays as masks ---
print(M[M < 5])                  # 1D array of all values < 5

# --- A rainfall-style analysis on synthetic data ---
inches = rng.rand(365)           # stand-in for daily precipitation
days = np.arange(365)
rainy  = (inches > 0)
summer = (days > 172) & (days < 262)
print("Median precip on rainy days:", np.median(inches[rainy]))
print("Max precip on summer days  :", np.max(inches[summer]))
print("Non-summer rainy median    :", np.median(inches[rainy & ~summer]))

# --- and/or vs &/| ---
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
print(A | B)                     # element-wise OR -> works
# print(A or B)                  # would raise ValueError (ambiguous truth value)

# bitwise on integers operates on bits
print(bin(42 & 59))              # 0b101010
print(bool(42 and 0), bool(42 or 0))  # whole-object truth: False True
```

## Why this matters / intuition

Masking replaces explicit Python loops with vectorized, C-speed operations. Because a comparison produces a Boolean array that lines up element-for-element with the data, you can ask aggregate questions ("how many?", "any?", "all?") and extract conditional subsets ("only the rainy days") in one expression. This is the same mental model that powers conditional selection in Pandas later, so it pays off well beyond NumPy. *(added context)* The pattern `data[condition]` is essentially SQL's `WHERE` clause expressed in array syntax.

## Gotchas

- **Use the NumPy versions**, not Python built-ins: prefer `np.sum`, `np.any`, `np.all` over `sum`, `any`, `all`, which can be slow or give wrong results on arrays.
- **Parentheses are mandatory** around each compound condition because `&`/`|` have higher precedence than comparison operators — `inches > 0.5 & inches < 1` parses wrong.
- **Never use `and`/`or` on arrays** — it tries to reduce the whole array to a single truth value and raises `ValueError: The truth value of an array ... is ambiguous`. Use `&`/`|`/`~`.
- `&`/`|` on plain integers do **bitwise** arithmetic on the binary representation, which is a different thing from Boolean array logic even though the operator is the same.

## Suggested figure (optional)

A side-by-side panel: left shows a small numeric array with a comparison condition drawn beneath it producing a same-shaped grid of True/False cells (color-coded); right shows the masking step where only the True-cell values are "pulled out" into a shorter 1D array — visually conveying that `x[mask]` collapses the grid down to the selected entries.
