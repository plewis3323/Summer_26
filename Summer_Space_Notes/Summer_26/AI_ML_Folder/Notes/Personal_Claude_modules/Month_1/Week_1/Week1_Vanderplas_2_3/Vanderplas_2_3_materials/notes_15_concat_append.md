---
# Notes 15 — Combining Datasets: Concat and Append (Vanderplas Ch. 3.6)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.06-concat-and-append.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Combining datasets ranges from simple stacking to complex database-style joins. This section covers the simple end: `pd.concat`, which glues `Series` and `DataFrame` objects together along an axis. It builds on NumPy's `np.concatenate`, but adds index-aware behavior — Pandas tracks the row/column labels and lets you control what happens when they overlap or differ. (The richer database-style merges/joins are the next section's topic.)

## Key ideas / idioms

- **NumPy foundation:** `np.concatenate([arr1, arr2, ...], axis=0)` joins arrays along a chosen axis. Pandas generalizes this for labeled data.
- **`pd.concat` signature** (reference only — not runnable as written):
  ```
  pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
            keys=None, levels=None, names=None, verify_integrity=False, copy=True)
  ```
  - `objs` — list/tuple of objects to concatenate.
  - `axis` — `0` (default, stack rows) or `1` (stack columns).
  - `join` — `'outer'` (default, union of the other axis's labels) or `'inner'` (intersection).
  - `ignore_index` — `True` discards original indices and builds a fresh `0..n-1` integer index.
  - `keys` — labels each source so the result gets a hierarchical (MultiIndex) top level.
  - `verify_integrity` — `True` raises a `ValueError` if the result has duplicate indices.
- **Default preserves indices:** unlike `ignore_index`, the default keeps the original labels, so duplicates can result.
- **Three ways to deal with duplicate indices:**
  1. `verify_integrity=True` — catch them as an error.
  2. `ignore_index=True` — throw the old index away.
  3. `keys=[...]` — keep them but disambiguate via an outer MultiIndex level.
- **Joins on the off-axis labels:** when columns (or, for `axis=1`, rows) don't match, `'outer'` keeps the union (filling gaps with `NaN`) and `'inner'` keeps only shared labels.

There is no heavy math here; the operation is essentially a labeled set/sequence concatenation. Conceptually, for outer vs inner join on label sets $A$ and $B$:

$$\text{outer} = A \cup B, \qquad \text{inner} = A \cap B$$

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# Helper used throughout the section to build small labeled DataFrames
def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

print(make_df('ABC', range(3)))

# --- NumPy foundation ---
x, y, z = [1, 2, 3], [4, 5, 6], [7, 8, 9]
print(np.concatenate([x, y, z]))          # [1 2 3 4 5 6 7 8 9]

a = [[1, 2], [3, 4]]
print(np.concatenate([a, a], axis=1))     # stacked side-by-side

# --- Series concatenation (indices preserved) ---
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(pd.concat([ser1, ser2]))

# --- DataFrame row-wise (axis=0, default) ---
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(pd.concat([df1, df2]))

# --- DataFrame column-wise (axis=1) ---
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(pd.concat([df3, df4], axis=1))
```

```python
import numpy as np
import pandas as pd

def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

# --- Duplicate indices ---
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index          # force overlapping indices

# 1) Catch duplicates
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)

# 2) Ignore the old index
print(pd.concat([x, y], ignore_index=True))   # fresh 0..3 index

# 3) Keep both via hierarchical keys
print(pd.concat([x, y], keys=['x', 'y']))     # MultiIndex top level x / y
```

```python
import numpy as np
import pandas as pd

def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

# --- Joins when columns differ ---
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])

# Outer join (default): union of columns, missing entries -> NaN
print(pd.concat([df5, df6]))

# Inner join: intersection of columns (only B and C survive)
print(pd.concat([df5, df6], join='inner'))

# Reindex one frame to control the result columns
# (replacement for the old join_axes= argument; see Gotchas)
print(pd.concat([df5, df6.reindex(columns=df5.columns)]))
```

## Why this matters / intuition

Real analyses almost always pull data from multiple files, time periods, or sources, and the first step is to stack them coherently. `pd.concat` is the workhorse for "these tables describe the same things, just put them together" — append new months of records (axis=0) or attach new feature columns (axis=1). The index-awareness is the value-add over raw NumPy: it keeps labels aligned and gives you explicit control (`keys`, `ignore_index`, `join`) over the messy cases of overlapping or mismatched labels, so you don't silently misalign rows.

## Gotchas

- **Index duplication is silent by default.** `pd.concat` happily keeps repeated index labels; if you need uniqueness, reach for `ignore_index=True`, `keys=`, or `verify_integrity=True`.
- **`pd.concat` returns a copy** — it does not modify the inputs in place.
- **`axis='col'` shorthand:** the source uses `axis='col'`/`axis='index'` aliases, but `axis=1`/`axis=0` are the safest, most portable forms. *(added context)*
- **`df.append()` is deprecated.** The book shows `df1.append(df2)` as a convenience equivalent to `pd.concat([df1, df2])`. In modern pandas (deprecated in 1.4, removed in 2.0) `DataFrame.append`/`Series.append` no longer exist — use `pd.concat` instead. *(added context)*
- **Repeated appending is inefficient.** Each `concat`/append builds a whole new object, so growing a frame one piece at a time is $O(n^2)$-ish. Collect all the pieces in a list and call `pd.concat` once at the end.
- **`join_axes=` is gone.** The book's `pd.concat([df5, df6], join_axes=[df5.columns])` no longer works; achieve the same by reindexing a frame's off-axis labels before concatenating (as shown above). *(added context)*

## Suggested figure (optional)

A side-by-side diagram of two small tables and three result panels: (1) `axis=0` stacking them vertically into a taller table; (2) `axis=1` placing them side-by-side into a wider table; (3) an off-axis-label panel contrasting `join='outer'` (union of columns, NaN-filled gaps shaded) vs `join='inner'` (only the shared columns). Color-code each source so the reader can trace where every cell came from, and annotate the index column to highlight duplicate vs. `ignore_index` vs. `keys` (MultiIndex) outcomes.
