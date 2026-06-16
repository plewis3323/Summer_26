---
# Notes 11 — Data Indexing and Selection (Vanderplas Ch. 3.2)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Pandas `Series` and `DataFrame` objects extend NumPy-style access (slicing, masking, fancy indexing) while also behaving like dictionaries (key-based access). The friction point is that a `Series`/`DataFrame` carries an **explicit index** that may or may not match the **implicit integer position**. To remove ambiguity, Pandas provides the `loc`, `iloc`, and `ix` indexers. The chapter walks through these patterns for `Series` first, then `DataFrame`.

## Key ideas / idioms

**Two indices, always.** Every Pandas object has an *explicit* index (the labels you assigned) and an *implicit* Python-style integer position. Plain `[]` mixes them in a confusing way:
- `data[1]` → uses the **explicit** index (label lookup).
- `data[1:3]` → uses the **implicit** position (Python-style slicing).

**The disambiguating indexers:**
- `loc` → **always explicit** index/labels. Slices are **inclusive** of the endpoint: `data.loc[1:3]` includes label `3`.
- `iloc` → **always implicit** integer position. Slices are **exclusive** of the endpoint (standard Python): `data.iloc[1:3]` excludes position `3`.
- `ix` → hybrid of the two; for a `Series` it behaves like plain `[]`. *(added context: `ix` is deprecated in modern Pandas — prefer `loc`/`iloc`. The book predates its removal but already recommends avoiding it.)*

**Slicing endpoint rule (the core gotcha), expressed as index sets.** For an explicit-label slice `a:c`:
$$ \text{result} = \{\, i : a \le i \le c \,\} \quad\text{(endpoint included)} $$
For an implicit-position slice `j:k`:
$$ \text{result} = \{\, p : j \le p < k \,\} \quad\text{(endpoint excluded)} $$

**DataFrame access conventions (the "seemingly inconsistent" rules):**
- **Indexing** (single key) refers to **columns**: `data['area']`.
- **Slicing** refers to **rows**: `data['Florida':'Illinois']` or `data[1:3]`.
- **Masking** operates **row-wise**: `data[data.density > 100]`.

**Zen of Pandas guidance:** prefer `loc`/`iloc` explicitly for readable code and to avoid subtle integer-index bugs.

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- Series as dictionary ---
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data['b'])                 # 0.5
print('a' in data)               # True
print(list(data.keys()))         # ['a', 'b', 'c', 'd']
print(list(data.items()))        # [('a', 0.25), ('b', 0.5), ...]

data['e'] = 1.25                 # extend like a dict
print(data)
```

```python
import numpy as np
import pandas as pd

# --- Series as 1D array: slice / mask / fancy ---
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

print(data['a':'c'])             # explicit slice: INCLUDES 'c'
print(data[0:2])                 # implicit slice: EXCLUDES position 2
print(data[(data > 0.3) & (data < 0.8)])   # masking
print(data[['a', 'd']])          # fancy indexing
```

```python
import numpy as np
import pandas as pd

# --- The trap: explicit integer index ---
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

print(data[1])      # explicit index -> 'a'
print(data[1:3])    # implicit position -> 'b','c' (labels 3,5)

# Disambiguate:
print(data.loc[1])      # explicit -> 'a'
print(data.loc[1:3])    # explicit, inclusive -> labels 1 and 3
print(data.iloc[1])     # position 1 -> 'b'
print(data.iloc[1:3])   # positions 1,2 -> 'b','c'
```

```python
import numpy as np
import pandas as pd

# --- DataFrame indexing ---
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})

print(data['area'])              # column access (dict-style)
print(data.area)                 # column access (attribute-style, same object)
print(data.area is data['area']) # True

data['density'] = data['pop'] / data['area']   # computed column
print(data)

print(data.values)               # underlying NumPy array
print(data.T)                    # transpose
```

```python
import numpy as np
import pandas as pd

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})
data['density'] = data['pop'] / data['area']

# --- loc / iloc on DataFrame ---
print(data.iloc[:3, :2])             # first 3 rows, first 2 cols (positional)
print(data.loc[:'Illinois', :'pop']) # label-based, inclusive

# combined masking + fancy indexing
print(data.loc[data.density > 100, ['pop', 'density']])

# assignment via indexer
data.iloc[0, 2] = 90
print(data)
```

```python
import numpy as np
import pandas as pd

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})
data['density'] = data['pop'] / data['area']

# --- Extra conventions: slicing -> rows, masking -> rows ---
print(data['Florida':'Illinois'])    # slice = ROWS by label
print(data[1:3])                     # slice = ROWS by position
print(data[data.density > 100])      # mask = ROWS
```

## Why this matters / intuition
Real data analysis is mostly *selecting the right slice of data*: a column, a window of rows, the rows that satisfy a condition. Pandas overloads `[]` to be ergonomic for the common cases (column by name, row slice, boolean mask), but that overloading is exactly what causes confusion once your index is integer-valued. The `loc`/`iloc` split gives you one unambiguous mental model: "am I talking about *labels* or *positions*?" Choosing deliberately makes selection code readable and prevents off-by-one and wrong-axis bugs that silently corrupt downstream analysis.

## Gotchas
- **Inclusive vs exclusive endpoints differ by indexer.** `loc[1:3]` includes `3`; `iloc[1:3]` and plain `data[1:3]` exclude position `3`. *(added context: easy to drop the last row of a label slice if you assume Python semantics.)*
- **Integer explicit index is the danger zone.** With an integer index, `data[1]` is label lookup but `data[1:3]` is positional. Use `loc`/`iloc` to be safe.
- **Attribute access (`data.area`) is fragile.** It fails or misbehaves when a column name collides with a `DataFrame` method/attribute (e.g., `data.pop` is the `pop()` method, not a column). For *assignment*, always use `data['col'] = ...`, never `data.col = ...`.
- **Single key indexes columns; slices index rows.** `data['area']` is a column, but `data['Florida':'Illinois']` is rows — same brackets, different axis.
- **Masking is row-wise**, returning the rows where the condition is True, not a filtered set of columns.

## Suggested figure (optional)
A two-column "decision card": left column "I want LABELS → use `.loc` (endpoint INCLUDED)", right column "I want POSITIONS → use `.iloc` (endpoint EXCLUDED)", with a small table at the bottom mapping plain-`[]` behaviors: single key = column, slice = rows, boolean mask = rows. A tiny number line could illustrate `loc[1:3]` covering 1,2,3 vs `iloc[1:3]` covering positions 1,2.

---

## 💬 Q&A (captured during session)

### Q: Clarify `loc` vs `iloc`.

Every Pandas object carries **two** indexing systems: the **explicit index** (labels you
assigned) and the **implicit position** (`0,1,2,…` where each item sits). Plain `[]` guesses
between them *inconsistently* — that's the trap `loc`/`iloc` exist to kill.

```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data[1]      # -> 'a'        plain [] treats 1 as a LABEL
data[1:3]    # -> labels 3,5 plain [] treats 1:3 as POSITIONS  (inconsistent!)
```

- **`.loc` = by Label** (explicit index); slice endpoint **INCLUDED**.
  ```python
  data.loc[1]     # 'a'            item whose label is 1
  data.loc[1:3]   # labels 1 and 3 (3 included)
  ```
- **`.iloc` = by integer position** (`0,1,2,…`); slice endpoint **EXCLUDED** (normal Python).
  ```python
  data.iloc[1]    # 'b'            item at position 1
  data.iloc[1:3]  # 'b','c'        positions 1,2 (3 excluded)
  ```

| | `.loc` | `.iloc` |
|---|---|---|
| indexes by | label (explicit) | position (`0,1,2,…`) |
| `[1]` means | item labeled `1` | 2nd item |
| slice endpoint | **included** | **excluded** |
| mnemonic | **l**oc = **l**abel | **i**loc = **i**nteger |

On a DataFrame both take `[rows, cols]`: `data.iloc[:3, :2]` (positional), 
`data.loc[:'Illinois', :'pop']` (label, inclusive), 
`data.loc[data.density > 100, ['pop','density']]` (mask rows + pick cols by label).

**One line:** `.loc` when thinking in **labels** (endpoint included), `.iloc` when thinking in
**positions** (endpoint excluded) — prefer both over bare `[]` so an integer index can't trick you.
