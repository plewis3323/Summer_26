---
# Notes 18 — Pivot Tables (Vanderplas Ch. 3.9)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.09-pivot-tables.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

A **pivot table** is a spreadsheet-style operation that takes column-wise data and produces a
two-dimensional (or higher) summary table. It is essentially a multidimensional version of
`GroupBy` aggregation: you split data along *two or more* keys, aggregate each cell, and lay the
result out as a grid (one key on the rows, another on the columns). `pivot_table` exists because
the equivalent multi-key `groupby` + `unstack` chains get verbose and hard to read fast.

The section motivates this with the Seaborn **Titanic** dataset (survival by sex/class/age/fare)
and the CDC **births** dataset (births by year/decade/gender/weekday).

## Key ideas / idioms

- **Pivot table = grouped-aggregate laid out in 2D.** The single-key version is just GroupBy:
  `titanic.groupby('sex')[['survived']].mean()`.
- **Two keys via GroupBy is awkward:**
  `titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()`.
- **Same thing, readable:**
  `titanic.pivot_table('survived', index='sex', columns='class')`.
- **First positional arg = the values column** to aggregate; `index` = row key; `columns` = column key.
- **Default aggregation is the mean** (`aggfunc='mean'`).
- **Multi-level pivots:** pass *lists* (or binned series) for `index`/`columns` to get hierarchical
  rows/columns.
- **Bin a continuous variable before pivoting** with `pd.cut` (fixed edges) or `pd.qcut` (quantiles),
  then use the binned series as a key.
- **Decade-rounding idiom:** `10 * (year // 10)` floors a year to its decade via integer division.

Robust spread estimate used in the births cleanup (a sigma derived from the IQR of a Gaussian):

$$\sigma \approx 0.74 \,\bigl(Q_{75} - Q_{25}\bigr)$$

Outliers are then filtered with a 5-sigma window around the median $\mu = Q_{50}$:

$$\mu - 5\sigma \;<\; \text{births} \;<\; \mu + 5\sigma$$

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- Small inline stand-in for seaborn's titanic so this runs standalone ---
# (added context): the book loads `sns.load_dataset('titanic')` which needs
# internet + seaborn. Here is a tiny synthetic frame with the same columns used.
rng = np.random.default_rng(0)
n = 200
titanic = pd.DataFrame({
    'survived': rng.integers(0, 2, n),
    'sex':      rng.choice(['female', 'male'], n),
    'class':    rng.choice(['First', 'Second', 'Third'], n),
    'age':      rng.uniform(1, 75, n),
    'fare':     rng.uniform(5, 250, n),
})

# 1) GroupBy -> the manual, verbose path
print(titanic.groupby('sex')[['survived']].mean())
print(titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack())

# 2) Equivalent pivot_table (default aggfunc = mean)
print(titanic.pivot_table('survived', index='sex', columns='class'))

# 3) Multi-level pivot: bin age with pd.cut, use as an extra row key
age = pd.cut(titanic['age'], [0, 18, 80])
print(titanic.pivot_table('survived', ['sex', age], 'class'))

# 4) Four-dimensional: also bin fare with pd.qcut and put it on the columns
fare = pd.qcut(titanic['fare'], 2)
print(titanic.pivot_table('survived', ['sex', age], [fare, 'class']))

# 5) Per-column aggregation via a dict, and margins (grand totals)
print(titanic.pivot_table(index='sex', columns='class',
                          aggfunc={'survived': 'sum', 'fare': 'mean'}))
print(titanic.pivot_table('survived', index='sex', columns='class', margins=True))
```

```python
import numpy as np
import pandas as pd

# --- Births example (CDC data). The book reads a real CSV from GitHub. ---
# (added context): replaced the remote CSV with a tiny inline frame so the
# decade-rounding, robust sigma-clip, and datetime-index idioms run standalone.
births = pd.DataFrame({
    'year':   [1969, 1969, 1975, 1975, 1988, 1988, 1999, 1999],
    'month':  [1, 1, 6, 6, 12, 12, 3, 3],
    'day':    [1.0, 1.0, 15.0, 15.0, 25.0, 25.0, 9.0, 9.0],
    'gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'births': [4000, 4200, 4500, 4700, 4300, 4600, 1, 999999],  # last two are outliers
})

# Decade column via integer-division trick
births['decade'] = 10 * (births['year'] // 10)

# Pivot: total births per decade, split by gender
print(births.pivot_table('births', index='decade', columns='gender', aggfunc='sum'))

# Robust outlier removal using IQR-derived sigma
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

# Build a proper datetime index, then derive day-of-week
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek
print(births[['gender', 'births', 'dayofweek']])
```

## Why this matters / intuition

Pivot tables are the fastest way to answer "how does outcome Y vary jointly across categories A
and B?" — exactly the exploratory question you ask constantly in data analysis. The Titanic example
makes the payoff concrete: a single readable call exposes that survival depended strongly on the
*interaction* of sex and class, something a 1D summary would hide. Because the API is so compact,
pivoting becomes a cheap, reach-for-it move during EDA rather than a multi-line ritual.

## Gotchas

- **Argument order:** the first positional arg is the *values* column, not the index. Be explicit
  with `index=`/`columns=` when in doubt.
- **`aggfunc` defaults to mean** — easy to forget when you actually wanted `'sum'` or a count.
- **Binned keys carry interval labels** (e.g. `(0, 18]`) from `pd.cut`/`pd.qcut`; `qcut` splits by
  *quantiles* (equal counts), `cut` by *value edges* — different bin contents.
- **`@`-references in `query`:** `@mu`/`@sig` pull from local Python variables; without `@` pandas
  looks for columns named `mu`/`sig`.
- **The 0.74 factor is an approximation** for the Gaussian IQR→sigma conversion (added context: the
  exact constant is $1/(2\,\Phi^{-1}(0.75)) \approx 0.7413$); it is *robust* to outliers, which is
  the whole point before sigma-clipping.
- **`margins=True`** adds an "All" row/column of totals; rename via `margins_name`.
- **`dropna=True`** (default) drops all-NaN entries; use `fill_value` to substitute a value into
  empty cells instead of leaving NaN.

## Suggested figure (optional)

A small-multiples line chart of mean daily births versus day-of-year, one faint line per decade
overlaid, would visualize the seasonal birth pattern the section builds toward. Pair it with a
simple bar chart of mean births by day-of-week to show the weekday-vs-weekend dip — both flow
directly from the datetime index derived in the second code block.
