---
# Notes 16 — Combining Datasets: Merge and Join (Vanderplas Ch. 3.7)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Pandas implements high-performance in-memory joins through `pd.merge()`. The behavior is grounded in **relational algebra**, the formal set of rules for manipulating relational data that underlies most databases. `pd.merge()` combines two DataFrames by matching one or more **key** columns (or indices), aligning rows even when their original ordering differs.

## Key ideas / idioms

- **Three categories of joins**, defined by the multiplicity of the key in each table:
  - **One-to-one**: keys are unique in both tables → behaves like a column-wise concatenation with row alignment by key.
  - **Many-to-one**: the key is duplicated in one table → values from the "one" side are repeated to fill the "many" side.
  - **Many-to-many**: the key is duplicated in both tables → the result is the **Cartesian product** of matching rows per key.
- **Key specification**:
  - Default: `pd.merge(a, b)` auto-detects shared column name(s) as the key.
  - `on='col'` — explicit shared key column (must exist in both).
  - `left_on=`, `right_on=` — merge on differently named columns; leaves a redundant column to `drop`.
  - `left_index=True`, `right_index=True` — merge on the index; `df.join()` does this by default.
  - You can mix: e.g. `left_index=True, right_on='name'`.
- **`how=` controls set arithmetic** on the keys *(added context: think of the set of key values in each table)*:

$$
\text{inner} = L \cap R, \quad \text{outer} = L \cup R, \quad \text{left} = L, \quad \text{right} = R
$$

  - `inner` (default): only keys present in both; non-matches dropped.
  - `outer`: union of keys; missing fields filled with `NaN`.
  - `left` / `right`: keep all keys from the left / right table.
- **`suffixes=`**: when both tables share a non-key column name, pandas appends `_x` / `_y` by default; override with e.g. `suffixes=['_L', '_R']`.

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- One-to-one join (auto-detected key 'employee') ---
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2)          # aligns on 'employee' despite different ordering
print(df3)

# --- Many-to-one join ('group' duplicated on the left after df3) ---
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(pd.merge(df3, df4))         # supervisor repeated to match each employee's group

# --- Many-to-many join ('group' duplicated in both) ---
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting', 'Engineering',
                              'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(pd.merge(df1, df5))         # each employee paired with all skills for their group
```

```python
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})

# --- on= : explicit key ---
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(pd.merge(df1, df2, on='employee'))

# --- left_on / right_on : differently named keys ---
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
merged = pd.merge(df1, df3, left_on='employee', right_on='name')
print(merged)
print(merged.drop('name', axis=1))   # drop the redundant duplicate column

# --- index-based merging ---
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
print(df1a.join(df2a))               # join() merges on index by default

# --- mixing index and column ---
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
```

```python
import numpy as np
import pandas as pd

# --- Set arithmetic via how= ---
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']})
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']})

print(pd.merge(df6, df7))                 # inner (default): only 'Mary'
print(pd.merge(df6, df7, how='outer'))    # union: NaN where missing
print(pd.merge(df6, df7, how='left'))     # all rows of df6
print(pd.merge(df6, df7, how='right'))    # all rows of df7

# --- suffixes for overlapping non-key columns ---
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'], 'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'], 'rank': [3, 1, 4, 2]})
print(pd.merge(df8, df9, on='name'))                          # rank_x, rank_y
print(pd.merge(df8, df9, on='name', suffixes=['_L', '_R']))  # rank_L, rank_R
```

## Why this matters / intuition

Real datasets rarely arrive in one tidy table; the common pattern is several sources keyed on a shared identifier (employee, state code, user id). Merge/join lets you assemble these into a single analysis-ready frame. Understanding the join as **set arithmetic over key values** makes the `how=` choice deliberate: use `inner` to keep only fully matched records, `outer` to retain everything and surface gaps as `NaN`. The book's capstone — combining state population, area, and abbreviation tables to compute 2010 population density — shows the realistic workflow: outer-merge with `left_on`/`right_on`, patch unmatched keys, drop helper columns, then a second merge for area before computing `population / area (sq. mi)`.

## Gotchas

- **Auto-detection merges on *all* shared column names** — if two tables happen to share an unintended column, the result is wrong. Be explicit with `on=` when in doubt.
- `left_on`/`right_on` leaves **two redundant columns** with identical data; `drop` one.
- Default join is **`inner`**, which silently discards non-matching rows — easy to lose data without noticing. Use `outer` if you need to detect mismatches.
- Many-to-many joins produce a **Cartesian product per key**, so row counts can explode unexpectedly.
- Overlapping non-key columns get `_x`/`_y` suffixes automatically; set `suffixes=` for readable names.
- *(added context: the book's population example uses `merged.drop('abbreviation', 1)` and `final.dropna(...)`; in current pandas the positional axis arg is deprecated — prefer `drop(columns='abbreviation')`.)*

## Suggested figure (optional)

A four-panel Venn-style diagram of two overlapping key sets L and R, one panel per `how=` value (inner = intersection shaded, outer = both circles shaded, left = L shaded, right = R shaded), with a tiny example table beneath each showing which rows survive and where `NaN` appears.
