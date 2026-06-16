---
# Notes 10 — Introducing Pandas Objects (Vanderplas Ch. 3.1)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.01-introducing-pandas-objects.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Pandas builds on NumPy by attaching **explicit, labeled indices** to data. Three core
objects power everything that follows:

- **`Series`** — a one-dimensional array of *indexed* data. Like a NumPy 1D array, but with
  an explicit index (not just implicit integer positions). Also acts like a typed dictionary.
- **`DataFrame`** — a two-dimensional structure with both flexible row indices and flexible
  column names. Think of it as a generalized NumPy 2D array, or as a dict mapping column
  names to `Series`.
- **`Index`** — the immutable, ordered-set object that labels the axes of `Series`/`DataFrame`.

The unifying idea: NumPy gives you *implicit* integer indexing; Pandas gives you *explicit*
label-based indexing on top of NumPy arrays.

## Key ideas / idioms

- A `Series` has `.values` (a NumPy array) and `.index` (a `pd.Index` object).
- NumPy array → *implicitly* defined integer index. `Series` → *explicitly* defined index,
  which can be any data type (strings, non-sequential ints, etc.).
- A `Series` is like a dictionary mapping typed keys → typed values, but it also supports
  array-style slicing that plain dicts do not.
- General constructor: `pd.Series(data, index=index)`. `data` may be a list/array, a scalar
  (broadcast to fill the index), or a dict (keys become the index).
- A `DataFrame` maps a **column name → a `Series` of column data**. Crucially, `data['col']`
  returns a *column* (a `Series`), whereas NumPy `arr[i]` returns a *row*.
- An `Index` behaves like an immutable array (supports slicing, has `.size/.shape/.ndim/.dtype`)
  and like an ordered set (intersection, union, symmetric difference).

Set operations on indices (the conceptual algebra):

$$
A \cap B \;\;(\text{intersection}), \qquad
A \cup B \;\;(\text{union}), \qquad
A \,\triangle\, B \;\;(\text{symmetric difference})
$$

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- Series: basic construction ---
data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data.values)   # NumPy array: [0.25 0.5  0.75 1.  ]
print(data.index)    # RangeIndex(start=0, stop=4, step=1)

# --- Series as generalized NumPy array: explicit index ---
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data['b'])     # 0.5

# Non-sequential / non-integer indices also work
data2 = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
print(data2[5])      # 0.5
```

```python
import numpy as np
import pandas as pd

# --- Series as a specialized dictionary ---
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population['California'])           # dict-style access -> 38332521
print(population['California':'Florida']) # array-style slicing (dicts can't do this)

# --- Constructor variations ---
print(pd.Series([2, 4, 6]))                       # default integer index
print(pd.Series(5, index=[100, 200, 300]))        # scalar broadcast to fill index
print(pd.Series({2: 'a', 1: 'b', 3: 'c'}))        # dict -> index from keys
print(pd.Series({2: 'a', 1: 'b', 3: 'c'}, index=[3, 2]))  # index selects/filters keys
```

```python
import numpy as np
import pandas as pd

population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127, 'Florida': 19552860,
                        'Illinois': 12882135})
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})

# --- DataFrame from a dict of Series ---
states = pd.DataFrame({'population': population, 'area': area})
print(states.index)     # row labels (state names)
print(states.columns)   # column labels: ['population', 'area']

# DataFrame as specialized dictionary: column access returns a Series
print(states['area'])   # the 'area' column

# --- Other ways to build a DataFrame ---
# From a single Series
print(pd.DataFrame(population, columns=['population']))

# From a list of dicts (missing keys -> NaN)
print(pd.DataFrame([{'a': i, 'b': 2 * i} for i in range(3)]))
print(pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]))  # NaN-filled

# From a 2D NumPy array
print(pd.DataFrame(np.random.rand(3, 2),
                   columns=['foo', 'bar'],
                   index=['a', 'b', 'c']))

# From a NumPy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
print(pd.DataFrame(A))
```

```python
import numpy as np
import pandas as pd

# --- The Index object: immutable array ---
ind = pd.Index([2, 3, 5, 7, 11])
print(ind[1])     # 3
print(ind[::2])   # every other element -> [2, 5, 11]
print(ind.size, ind.shape, ind.ndim, ind.dtype)   # 5 (5,) 1 int64

# Immutability: this raises TypeError (indices cannot be modified)
try:
    ind[1] = 0
except TypeError as e:
    print("TypeError:", e)

# --- Index as an ordered set ---
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
print(indA.intersection(indB))         # [3, 5, 7]
print(indA.union(indB))                # [1, 2, 3, 5, 7, 9, 11]
print(indA.symmetric_difference(indB)) # [1, 2, 9, 11]
```

## Q&A capture — "generalization of an array" vs "specialization of a dict"

> *(My question during reading: what do those two phrases actually mean?)*

The two phrases compare a `Series` in **opposite directions**:

- **Generalization of a NumPy array** = *more capable / fewer restrictions*. A `Series`
  does everything an array does (typed, contiguous values; vectorized math) **plus** it lets
  you choose the index. A NumPy array is just the special case where the index is forced to be
  `0, 1, 2, …`. Relaxing that "labels must be integer positions" rule = generalizing.

  ```python
  arr = np.array([0.25, 0.5, 0.75, 1.0]); arr[1]          # 0.5 (label = fixed position)
  s = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a','b','c','d']); s['b']  # 0.5 (chosen label)
  ```

- **Specialization of a Python dict** = *less flexible, but better at a narrower job*. A dict
  maps arbitrary keys → arbitrary values with no type rules. A `Series` is dict-like
  (label → value) but **constrains** keys to a typed ordered `Index` and values to a
  single-dtype NumPy array. Giving up the dict's "anything goes" freedom buys vectorized math
  and array speed.

  ```python
  d = {'a':0.25,'b':0.5,'c':0.75}; d['b']   # 0.5, but d*2 / d.mean() are impossible
  s = pd.Series(d); s['b']; s * 2; s['a':'c']  # dict lookup AND math AND slicing
  ```

A `Series` sits between the two: **add labels to an array** (generalize upward) and
**add array-speed + type rules to a dict** (specialize downward) — getting the best of each.

```
  MORE GENERAL  ◄──────── Series ────────►  MORE SPECIALIZED
  NumPy array            (labels + typed     Python dict
  (fast, but labels       array values)      (label→value, but no
   forced 0,1,2,…)                            math, no type rules)
```

## Q&A capture — "a DataFrame is a sequence of aligned Series"

> *(My question during reading: simplify the book's DataFrame-as-aligned-Series passage.)*

Sentence-by-sentence translation of the source passage:

1. **"DataFrame = 2D array with flexible row indices AND flexible column names."** Same idea
   as a `Series` (1D array with chosen labels) but in two dimensions — you label *both* the
   rows (`'California'`, `'Texas'`, …) and the columns (`'population'`, `'area'`).
2. **"A DataFrame is a sequence of aligned Series."** Picture a 2D array as columns sitting
   side by side; in a `DataFrame` each of those columns *is* a `Series`. Glue several `Series`
   together as columns → a `DataFrame`.
3. **"Aligned" = the Series share the same index.** When stacked, Pandas matches rows by
   **label**, not by position. So `population['California']` and `area['California']` land on
   the same row even if the two Series were in different orders; a label present in one but not
   the other becomes `NaN`.

```python
population = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127})
area2      = pd.Series({'Texas': 695662, 'California': 423967})   # different order, no NY
pd.DataFrame({'population': population, 'area': area2})
#             population      area
# California    38332521  423967.0
# New York      19651127       NaN   ← aligned by label; NY has no area → NaN
# Texas         26448193  695662.0
```

```
  Series population   Series area        DataFrame (glued on shared index)
  ┌──────────┬───┐   ┌──────────┬─────┐   ┌──────────┬──────────┬──────┐
  │California │38M│   │California │423K │   │          │population│ area │ ← col names
  │New York   │19M│ + │New York   │141K │ = │California │  38M     │423K  │
  │Texas      │26M│   │Texas      │696K │   │New York   │  19M     │141K  │ ← row index
  └──────────┴───┘   └──────────┴─────┘   │Texas      │  26M     │696K  │
       same row labels  =  "aligned"       └──────────┴──────────┴──────┘
```

**One line:** a `DataFrame` is several `Series` standing shoulder to shoulder as columns,
lined up so equal-labeled rows sit on the same line.

## Why this matters / intuition

The explicit index is what makes Pandas powerful: data **aligns by label**, not by position.
When you combine two `Series` or build a `DataFrame` from several `Series`, Pandas matches on
the index automatically and fills mismatches with `NaN`. That label-based alignment is the
seed of nearly every later operation (indexing, joins, groupby, time series). Treating a
`Series` simultaneously as "array" and "dict" gives you both fast vectorized math and
meaningful keyed lookups in one object.

## Gotchas

- **Column vs. row access:** for a `DataFrame`, `data['col0']` returns the first *column*,
  not the first row — the opposite of `arr[0]` on a NumPy 2D array.
- **Indices are immutable:** assigning into an `Index` raises
  `TypeError: Index does not support mutable operations`. This immutability is what makes it
  safe to share an index across multiple `Series`/`DataFrame` objects.
- **Dict construction and ordering:** building a `Series`/`DataFrame` from a dict derives the
  index from the keys; passing an explicit `index=` selects/filters which keys are kept.
- **List-of-dicts with missing keys** produces `NaN` for absent entries — don't assume rows
  are dense.
- *(added context)* In older pandas the set operators `&`, `|`, `^` worked directly on `Index`
  objects (as the book originally showed); current pandas reserves those for elementwise
  boolean ops, so prefer the explicit `.intersection()` / `.union()` /
  `.symmetric_difference()` methods used above.

## Suggested figure (optional)

A three-panel diagram: (1) a NumPy 1D array with implicit integer positions 0–3 beside a
`Series` showing the same values bound to an explicit labeled index (a, b, c, d); (2) a
`DataFrame` drawn as a grid with labeled row index down the left and column names across the
top, each column highlighted as its own `Series`; (3) two overlapping circles (a Venn diagram)
illustrating `Index` set operations — intersection, union, and symmetric difference.
