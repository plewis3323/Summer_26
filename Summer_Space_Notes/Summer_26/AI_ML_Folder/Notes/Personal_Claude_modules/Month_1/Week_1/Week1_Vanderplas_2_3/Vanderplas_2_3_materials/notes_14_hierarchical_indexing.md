---
# Notes 14 — Hierarchical Indexing / MultiIndex (Vanderplas Ch. 3.5)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

**Hierarchical indexing** (a.k.a. **multi-indexing**) lets you pack data with more than one or two dimensions into the familiar 1D `Series` and 2D `DataFrame` by giving an axis *multiple index levels*. A single `Series` with a two-level index behaves like a 2D table; a `DataFrame` with multi-level row *and* column indices behaves like 4D data, and so on. This is the idiomatic Pandas way to handle higher-dimensional, labeled data without leaving the core data structures.

## Key ideas / idioms

- A naive approach uses **Python tuples as keys** (e.g. `('California', 2000)`). It "works" but slicing on a single level forces ugly comprehensions and is inefficient.
- `pd.MultiIndex` is the proper object. Pandas often builds it implicitly; you can also build it explicitly with `from_arrays`, `from_tuples`, `from_product`, or the low-level constructor.
- A `Series` with a `MultiIndex` and an `unstack()` of it into a `DataFrame` carry the **same information** — the extra index level is "one more dimension."
- **Index levels can be named** (`index.names`), which makes selection self-documenting.
- **Both axes** (rows and columns) can be hierarchical.
- **Partial indexing / slicing** selects on a subset of levels.
- Many `MultiIndex` slicing operations require the index to be **lexicographically sorted**; otherwise you get an error. Fix with `sort_index()`.
- `stack` / `unstack` reshape between stacked (index) and pivoted (column) representations, controllable by `level`.
- `reset_index` turns index levels into columns; `set_index` does the reverse. This is the bridge between flat tables and multi-indexed ones.
- Aggregations can collapse a chosen `level` instead of the whole axis.

For an $n$-level row index and an $m$-level column index, a `DataFrame` effectively represents data of dimensionality

$$ \text{dims} = n + m $$

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- The clumsy "bad way": tuples as keys ---
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)

# Selecting one level requires an awkward comprehension:
print(pop[[i for i in pop.index if i[1] == 2010]])
```

```python
import numpy as np
import pandas as pd

# --- The better way: a real MultiIndex ---
index = pd.MultiIndex.from_tuples([('California', 2000), ('California', 2010),
                                   ('New York', 2000), ('New York', 2010),
                                   ('Texas', 2000), ('Texas', 2010)])
populations = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)
pop.index.names = ['state', 'year']
print(pop)

# Clean partial indexing on the second level:
print(pop[:, 2010])

# MultiIndexed Series <-> DataFrame are the same info:
pop_df = pop.unstack()      # states as rows, years as columns
print(pop_df)
print(pop_df.stack())       # back to the MultiIndexed Series
```

```python
import numpy as np
import pandas as pd

# --- Several ways to build a MultiIndex ---
# Implicit: pass a list of index arrays
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
print(df)

print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]]))
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]))
print(pd.MultiIndex.from_product([['a', 'b'], [1, 2]]))  # Cartesian product

# A dict with tuple keys also auto-builds a MultiIndex:
data = {('California', 2000): 33871648, ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,     ('Texas', 2010): 25145561}
print(pd.Series(data))
```

```python
import numpy as np
import pandas as pd

# --- MultiIndex on BOTH axes (acts like 4D data) ---
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# Some mock health data
np.random.seed(0)
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)

# Partial column indexing by name:
print(health_data['Guido'])          # DataFrame for one subject
print(health_data['Guido', 'HR'])    # Series of one subject's heart rate
```

```python
import numpy as np
import pandas as pd

# --- IndexSlice for slicing across multiple levels ---
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
np.random.seed(0)
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)

idx = pd.IndexSlice
print(health_data.loc[idx[:, 1], idx[:, 'HR']])  # visit 1, all HR columns
```

```python
import numpy as np
import pandas as pd

# --- Sorted indices are required for range slicing ---
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']

try:
    data['a':'b']             # fails: index is NOT lexicographically sorted
except Exception as e:
    print("Error:", type(e).__name__)

data = data.sort_index()      # fix it
print(data['a':'b'])          # now works
```

```python
import numpy as np
import pandas as pd

# --- reset_index / set_index: flat tables <-> multi-index ---
index = pd.MultiIndex.from_tuples([('California', 2000), ('California', 2010),
                                   ('New York', 2000), ('New York', 2010)],
                                  names=['state', 'year'])
pop = pd.Series([33871648, 37253956, 18976457, 19378102], index=index)

# Index levels -> columns (give the values a name)
pop_flat = pop.reset_index(name='population')
print(pop_flat)

# Columns -> index levels (the typical raw-data workflow)
print(pop_flat.set_index(['state', 'year']))
```

```python
import numpy as np
import pandas as pd

# --- Aggregating along a chosen level ---
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
np.random.seed(0)
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)

# Book syntax uses the `level=` keyword on the reducer:
#     data_mean = health_data.mean(level='year')
#     data_mean.mean(axis=1, level='type')
# (added context) `level=` on reducers is removed in modern pandas (>=2.0).
# Use groupby instead, which is equivalent:
data_mean = health_data.groupby(level='year').mean()        # average over visits
print(data_mean)
print(data_mean.groupby(level='type', axis=1).mean())       # average HR / Temp
```

## Why this matters / intuition

Real-world data is frequently more than 2D (e.g. measurements indexed by *subject* x *type* over *year* x *visit*). Rather than reaching for awkward nested structures or external N-D arrays, hierarchical indexing keeps everything inside `Series`/`DataFrame`, so you retain all the alignment, slicing, and vectorized-operation machinery. The mental model: each extra index level is just another axis folded into a single labeled index, and `stack`/`unstack` let you "rotate" axes between the row index and the column index at will. This is also the natural output shape of `groupby` aggregations, so understanding `MultiIndex` is foundational for the chapters that follow.

## Gotchas

- **Sorting is mandatory for partial slices.** Range slicing (`data['a':'b']`) on an unsorted `MultiIndex` raises an `UnsortedIndexError` / `KeyError`. Call `sort_index()` first.
- **Don't put raw `slice(...)` / `:` inside an index tuple.** Python forbids `health_data.loc[(:, 1), (:, 'HR')]` syntactically. Use `pd.IndexSlice` (`idx[:, 1]`) instead.
- **Tuples-as-keys is a trap.** It looks fine for tiny data but is inefficient and lacks the optimized `MultiIndex` operations.
- **Partial indexing order matters.** `pop['California']` (top level) is easy; selecting a lower level (`pop[:, 2010]`) needs the leading colon.
- (added context) **`level=` on reducers (`.mean(level=...)`) is deprecated/removed** in pandas >= 2.0. Use `df.groupby(level=...).mean()`. The book predates this change.
- (added context) The book's low-level `pd.MultiIndex(levels=..., labels=...)` uses the old `labels=` argument; modern pandas renamed it to `codes=`.

## Suggested figure (optional)

A side-by-side diagram showing the *same* state/year population data in two forms: (left) a 1D `Series` with a two-level (`state`, `year`) `MultiIndex` drawn as indented row labels; (right) the result of `unstack()` — a 2D grid with `state` as row labels and `year` as column headers. Curved arrows labeled `unstack()` (left to right) and `stack()` (right to left) connect them, visually conveying that the two representations hold identical information and that stack/unstack just rotates a level between the row and column axes.
