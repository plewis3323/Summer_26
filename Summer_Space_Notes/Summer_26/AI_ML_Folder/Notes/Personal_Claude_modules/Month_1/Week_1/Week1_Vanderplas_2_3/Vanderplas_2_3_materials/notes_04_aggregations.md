---
# Notes 04 — Aggregations: Min, Max, and Everything In Between (Vanderplas Ch. 2.4)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Aggregations reduce an array down to a single summary value (or a row/column of summary values): sum, min, max, mean, std, etc. NumPy ships compiled aggregation functions that run much faster than Python's built-ins and that are dimension-aware (they understand the `axis` argument). These summaries are the first thing you compute when exploring a dataset.

## Key ideas / idioms
- **Use NumPy's versions, not Python built-ins.** `np.sum(L)` runs ~200x faster than the built-in `sum(L)` on a large array because it executes in compiled code. They also differ in optional arguments and dimensional awareness, so they are *not* interchangeable.
- **Two call styles.** Functional form `np.min(arr)` / `np.max(arr)` and the shorter method form `arr.min()` / `arr.max()`. Same result.
- **`axis` collapses a dimension.** For 2D arrays, `axis` names *the dimension that gets collapsed*, not the one that remains. So `axis=0` collapses rows → one value per column; `axis=1` collapses columns → one value per row. With no `axis`, the whole array is reduced to a scalar.
- **NaN-safe variants.** Most functions have an `np.nan*` twin that ignores missing values (`NaN`) instead of letting them poison the result.
- For a column of values $x_1, \dots, x_n$, the mean and (population) standard deviation are:
$$ \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2} $$
*(added context: NumPy's `np.std` uses the population formula — divisor $n$ — by default; pass `ddof=1` for the sample standard deviation with divisor $n-1$.)*

### Aggregation functions table
| Function | NaN-safe version | Purpose |
|----------|------------------|---------|
| `np.sum` | `np.nansum` | Sum of elements |
| `np.prod` | `np.nanprod` | Product of elements |
| `np.mean` | `np.nanmean` | Mean of elements |
| `np.std` | `np.nanstd` | Standard deviation |
| `np.var` | `np.nanvar` | Variance |
| `np.min` | `np.nanmin` | Minimum value |
| `np.max` | `np.nanmax` | Maximum value |
| `np.argmin` | `np.nanargmin` | Index of minimum |
| `np.argmax` | `np.nanargmax` | Index of maximum |
| `np.median` | `np.nanmedian` | Median value |
| `np.percentile` | `np.nanpercentile` | Rank-based quantile |
| `np.any` | N/A | Whether any element is true |
| `np.all` | N/A | Whether all elements are true |

## Worked code examples (runnable)

```python
import numpy as np

# NumPy aggregation vs Python built-in (same value, very different speed)
L = np.random.random(100)
print(sum(L))        # Python built-in
print(np.sum(L))     # NumPy version (compiled, ~200x faster on large arrays)

# Min and max, functional form and method form
big_array = np.random.rand(1_000_000)
print(np.min(big_array), np.max(big_array))
print(big_array.min(), big_array.max())   # shorthand methods
```

```python
import numpy as np

# Multidimensional aggregates: axis names the dimension that is COLLAPSED
M = np.random.random((3, 4))
print(M)
print("whole array sum :", M.sum())        # single scalar
print("min of each col :", M.min(axis=0))  # collapse rows -> 4 values
print("max of each row :", M.max(axis=1))  # collapse cols -> 3 values
```

```python
import numpy as np

# Summary-statistics example (mirrors the presidents'-heights walkthrough).
# Self-contained: we synthesize heights instead of reading the CSV.
heights = np.array([189, 170, 189, 163, 183, 171, 185, 168, 173, 183])

print("Mean height       :", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height    :", heights.min())
print("Maximum height    :", heights.max())
print("25th percentile   :", np.percentile(heights, 25))
print("Median            :", np.median(heights))
print("75th percentile   :", np.percentile(heights, 75))
```

*(added context: in the book this data comes from `pd.read_csv('data/president_heights.csv')` and `np.array(data['height(cm)'])`, giving mean ≈ 179.74 cm, std ≈ 6.93, min 163, max 193. The array above is a stand-in so the snippet runs without the file.)*

## Why this matters / intuition
Before modeling anything, you describe it. A handful of aggregates — mean, spread, extremes, quartiles — turns a raw column of numbers into something you can reason about and sanity-check. Doing this in compiled NumPy (and along chosen axes) makes it fast enough to apply to large datasets interactively, which is exactly the loop exploratory data analysis depends on.

## Gotchas
- `sum` vs `np.sum` are *different functions*. Built-in `sum` is slow and does not understand `axis`; prefer `np.sum`.
- Mixing them up on multidimensional arrays gives confusing results because only `np.sum` interprets `axis`.
- The `axis` argument is easy to invert: `axis=0` does **not** mean "operate over rows to get a per-row answer" — it *collapses* the row dimension, yielding a per-column result.
- A single `NaN` makes ordinary aggregations return `NaN`. Use the `np.nan*` variants when missing data is possible.
- *(added context)* `np.std` defaults to population std (`ddof=0`); for an unbiased sample estimate use `ddof=1`.

## Suggested figure (optional)
A 3x4 grid of cells with arrows: a downward arrow over the columns labeled `axis=0 -> one value per column`, and a rightward arrow over the rows labeled `axis=1 -> one value per row`, with a separate note that "no axis" reduces the whole grid to a single number. This visually reinforces that `axis` marks the dimension being collapsed.

---

## 💬 Q&A (captured during session)

### Q: What does "aggregate" mean?

**Aggregate** means to **combine many values into a single summary value**. An aggregation takes a whole collection of numbers and boils it down to *one* number (or a small set of numbers) that summarizes them. The word itself = "to gather into a whole": you *gather* many data points and collapse them into a representative figure.

**Everyday examples** — given `[5, 2, 8, 1, 9]`:
- **sum** → `25` (all added) · **mean** → `5.0` · **min** → `1`, **max** → `9` · **count** → `5`

Each is an *aggregate*: many in, one out.

```python
import numpy as np

x = np.array([5, 2, 8, 1, 9])
x.sum()      # 25   -> one number
x.mean()     # 5.0
x.min()      # 1
x.max()      # 9
x.std()      # spread around the mean
```

For 2-D arrays you aggregate along an **axis** — collapsing just one dimension instead of everything:

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

M.sum()           # 21   (everything -> one number)
M.sum(axis=0)     # [5 7 9]   (collapse rows -> one value per column)
M.sum(axis=1)     # [6 15]    (collapse columns -> one value per row)
```

Mental model for `axis`: it names the dimension that **disappears**. `axis=0` collapses the rows, leaving column totals.

**Why it matters:** aggregation is usually the *first* thing you do to understand a dataset — you can't eyeball a million rows, but you can look at their mean/min/max/std. In Pandas (**Notes 17**) aggregation becomes **split-apply-combine**: split into groups, aggregate each group separately (e.g. average sales *per region*), then combine the per-group summaries into one table.
