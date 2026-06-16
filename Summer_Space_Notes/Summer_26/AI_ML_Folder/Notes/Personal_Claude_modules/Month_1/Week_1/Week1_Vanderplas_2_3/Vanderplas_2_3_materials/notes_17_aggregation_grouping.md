---
# Notes 17 — Aggregation and Grouping (Vanderplas Ch. 3.8)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Efficient summarization of large data means computing aggregations — `sum()`, `mean()`, `median()`, `min()`, `max()`, etc. Pandas Series/DataFrame objects expose these directly, returning one scalar per column by default. The real power comes from **`groupby`**, which conditions aggregations on subsets of the data via the **split-apply-combine** pattern. The chapter uses the Seaborn `planets` exoplanet dataset (1,035 rows, 6 columns: method, number, orbital_period, mass, distance, year) as its running example.

## Key ideas / idioms

**Simple aggregations.** For a Series, methods like `.sum()` and `.mean()` collapse to a single value. For a DataFrame, aggregates compute per column by default; pass `axis='columns'` to aggregate across columns within each row. `describe()` computes several aggregates at once and is great for first-pass exploration.

Common aggregation methods (the chapter's reference table):

| Method | Purpose |
|---|---|
| `count()` | Total number of items |
| `first()`, `last()` | First and last item |
| `mean()`, `median()` | Mean and median |
| `min()`, `max()` | Minimum and maximum |
| `std()`, `var()` | Standard deviation and variance |
| `mad()` | Mean absolute deviation |
| `prod()` | Product of all items |
| `sum()` | Sum of all items |

**Split-apply-combine.** The `groupby` operation has three conceptual steps:

$$\text{data} \;\xrightarrow{\text{split by key}}\; \{g_1, g_2, \dots\} \;\xrightarrow{\text{apply } f}\; \{f(g_1), f(g_2), \dots\} \;\xrightarrow{\text{combine}}\; \text{result}$$

- **Split:** break the table into groups based on the value of a key.
- **Apply:** run a function (aggregate / filter / transform / apply) within each group.
- **Combine:** merge the per-group results back into a single output indexed by the group keys.

`df.groupby('key')` returns a lazy `DataFrameGroupBy` object — no computation happens until you apply an aggregation. This avoids materializing intermediate per-group DataFrames *(added context: lazy evaluation also lets pandas dispatch the work efficiently in one pass)*.

**GroupBy object features.**
- *Column indexing:* `planets.groupby('method')['orbital_period']` selects a column from the grouped object, yielding a (still lazy) grouped Series.
- *Iteration:* iterating a GroupBy yields `(key, group)` pairs where each `group` is a sub-DataFrame.
- *Dispatch methods:* any method not defined on GroupBy is dispatched to each group (e.g. `.describe()`).

**Aggregate / filter / transform / apply** — the four core "apply" operations:
- `aggregate()` takes a string, function, list of these, or a `{column: func}` dict for column-specific aggregation.
- `filter()` keeps or drops entire groups based on a boolean-returning function of the group.
- `transform()` returns data the **same shape** as the input (e.g. center within group).
- `apply()` runs an arbitrary function on each group's DataFrame and combines the results.

**Specifying split keys.** The key can be a column name, a list/array/Series of group labels (length must match), a dict/Series mapping index values to groups, or even a Python function applied to the index. Keys can be combined in a list to form a multi-index grouping.

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- Simple aggregations ---
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
print(ser.sum(), ser.mean())

df = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand(5)})
print(df.mean())                 # per-column (default axis)
print(df.mean(axis='columns'))   # per-row
```

```python
import numpy as np
import pandas as pd

# --- Split-apply-combine basics ---
df = pd.DataFrame({'key':  ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)})
print(df.groupby('key'))        # lazy DataFrameGroupBy object
print(df.groupby('key').sum())  # combine -> one row per key
```

```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame({'key':   ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)})

# Iteration over groups: each group is a sub-DataFrame
for (key, group) in df.groupby('key'):
    print("{0:>3s}  shape={1}".format(key, group.shape))

# Dispatch: describe() is applied per group
print(df.groupby('key')['data1'].describe())
```

```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame({'key':   ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)})

# aggregate: list of functions, and per-column dict
print(df.groupby('key').aggregate(['min', np.median, max]))
print(df.groupby('key').aggregate({'data1': 'min', 'data2': 'max'}))

# filter: keep groups whose data2 std exceeds 4
def filter_func(x):
    return x['data2'].std() > 4
print(df.groupby('key').filter(filter_func))

# transform: center each group (output same shape as input)
print(df.groupby('key').transform(lambda x: x - x.mean()))

# apply: normalize data1 by the group's data2 sum
def norm_by_data2(x):
    x['data1'] = x['data1'] / x['data2'].sum()
    return x
print(df.groupby('key').apply(norm_by_data2))
```

```python
import numpy as np
import pandas as pd

df2 = pd.DataFrame({'data1': range(6), 'data2': [5, 0, 3, 3, 7, 9]},
                   index=['A', 'B', 'C', 'A', 'B', 'C'])

# key as a list/array of labels (length matches rows)
L = [0, 1, 0, 1, 2, 0]
df0 = pd.DataFrame({'data1': range(6)})
print(df0.groupby(L).sum())

# key as a dict mapping index -> group
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2.groupby(mapping).sum())

# key as a Python function applied to the index
print(df2.groupby(str.lower).mean())

# combine keys for a multi-index grouping
print(df2.groupby([str.lower, mapping]).mean())
```

```python
import numpy as np
import pandas as pd
import seaborn as sns   # provides the planets dataset

planets = sns.load_dataset('planets')

# Column indexing + per-group median
print(planets.groupby('method')['orbital_period'].median())

# Practical example: discoveries by method and decade
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
table = planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
print(table)
```

## Why this matters / intuition
`groupby` is the workhorse of exploratory data analysis: almost any "per-category" question — average sales per region, error rate per model, counts per class — is a split-apply-combine. Thinking in those three steps lets you reason about *what shape* the output should be: `aggregate` collapses each group to a row, `transform` preserves shape (ideal for feature engineering like group-wise normalization), `filter` subsets whole groups, and `apply` is the escape hatch for anything else. The decade-vs-method table shows how groupby + `unstack` turns raw rows into a readable pivot revealing trends (Radial Velocity dominated early; Transit surged in the 2010s).

## Gotchas
- The GroupBy object is **lazy** — printing it shows an object, not data; you must apply an operation to trigger computation.
- A list/array key must have **the same length as the rows** being grouped; a dict/Series key maps **index values** (not positions) to groups.
- `transform` must return output the **same shape** as its input; `aggregate` reduces; mixing these up causes shape/index errors.
- `apply` receives each group as a **DataFrame** and can return a scalar, Series, or DataFrame — pandas infers how to combine, which can surprise you. *(added context: classic `apply(norm_by_data2)` mutates and returns the frame; prefer building a new column to avoid in-place side effects.)*
- `axis='columns'` aggregates across columns per row — easy to forget and get column-wise results instead.
- `mad()` referenced in the table was deprecated/removed in later pandas versions *(added context, not stated in the source)*.

## Suggested figure (optional)
A three-panel split-apply-combine diagram: on the left, a small input table with a `key` column (rows colored by key value A/B/C). A **"split"** arrow fans the rows out into three stacked mini-tables, one per key. An **"apply"** arrow over each mini-table shows a function (e.g. `sum`) reducing it to a single value. A **"combine"** arrow funnels those three results back into one compact output table indexed by A/B/C. Color-coding each key consistently across all three stages makes the data flow visually obvious.
---
