---
# Notes 21 — High-Performance Pandas: eval() and query() (Vanderplas Ch. 3.12)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.12-performance-eval-and-query.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Vectorized NumPy/Pandas operations are fast, but **compound** expressions force the
allocation of full-size **temporary arrays** for every subexpression. Pandas exposes
`pd.eval()`, `DataFrame.eval()`, and `DataFrame.query()`, which use the **Numexpr**
library to evaluate string expressions element-by-element *without* materializing those
intermediates. The headline win is **memory** (and, secondarily, speed and readability)
— especially when temporaries would exceed available RAM or blow the CPU cache.

## Key ideas / idioms
- **Why compound expressions allocate temporaries.** For an expression like
  `(x > 0.5) & (y < 0.5)`, Python/NumPy effectively does (conceptual — not runnable standalone):
  ```
  tmp1 = (x > 0.5)
  tmp2 = (y < 0.5)
  mask = tmp1 & tmp2
  ```
  Every subexpression produces its own full-size array. For a chain of $N$ operations
  over arrays of length $M$, the peak memory cost scales like
  $$\text{peak memory} \sim \mathcal{O}(N \cdot M),$$
  because intermediates pile up before the final result is formed.
- **Numexpr fix.** Numexpr evaluates the expression element-by-element, so it never needs
  to build the full intermediate arrays. This keeps the working set small enough to stay
  in **CPU L1/L2 cache** (a few MB) instead of spilling to main memory.
- **`pd.eval(expr_string)`** — top-level function; refer to objects by their Python
  variable names inside the string. Supports:
  - arithmetic: `-df1 * df2 / (df3 + df4) - df5`
  - comparisons, incl. **chained**: `df1 < df2 <= df3 != df4`
  - bitwise `&`/`|` and the literal words `and`/`or`: `(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)`
  - object attributes / indexing: `df2.T[0] + df3.iloc[1]`
  - **Not supported:** function calls, conditionals, loops.
- **`DataFrame.eval(expr)`** — refer to **columns by bare name** (`A` not `df.A`); more
  succinct than `pd.eval`. Supports **column assignment** (`D = (A + B) / C`), including
  modifying an existing column, via `inplace=True`.
- **`@local_var`** — inside `DataFrame.eval()`/`query()`, prefix a Python local variable
  with `@` to distinguish it from a column name. *Not available in `pd.eval()`.*
- **`DataFrame.query(expr)`** — cleaner syntax for **filtering** rows; replaces
  `df[(df.A < 0.5) & (df.B < 0.5)]` with `df.query('A < 0.5 and B < 0.5')`. Also
  supports `@local_var`.

## Worked code examples (runnable)
```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)

# --- pd.eval(): compute on whole DataFrames, no big temporaries ---
nrows, ncols = 1000, 100
df1, df2, df3, df4, df5 = (
    pd.DataFrame(rng.rand(nrows, ncols)) for _ in range(5)
)

# Same result, two ways:
direct = df1 + df2 + df3 + df4
viaeval = pd.eval('df1 + df2 + df3 + df4')
print("pd.eval matches direct sum:", np.allclose(direct, viaeval))

# Supported operation classes
arith   = pd.eval('-df1 * df2 / (df3 + df4) - df5')
compare = pd.eval('df1 < df2 <= df3 != df4')          # chained comparisons
bitwise = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
words   = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')  # and/or literals
print("bitwise == words:", np.allclose(bitwise, words))

# Object attributes / indices
attr = pd.eval('df2.T[0] + df3.iloc[1]')
print("attribute expr shape:", attr.shape)
```

```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])

# DataFrame.eval(): columns by bare name
r1 = pd.eval("(df.A + df.B) / (df.C - 1)")   # via pd.eval
r2 = df.eval('(A + B) / (C - 1)')            # via df.eval (more succinct)
print("eval forms match:", np.allclose(r1, r2))

# Column assignment (create then modify in place)
df.eval('D = (A + B) / C', inplace=True)
print("created column D:\n", df.head(2))
df.eval('D = (A - B) / C', inplace=True)     # overwrite existing D
print("modified column D:\n", df.head(2))

# @ references a Python local variable, not a column
column_mean = df.mean(axis=1)
r3 = df.eval('A + @column_mean')
print("first value of A + @column_mean:", r3.iloc[0])
```

```python
import numpy as np
import pandas as pd

rng = np.random.RandomState(1)
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])

# query(): row filtering with clean syntax
mask_way = df[(df.A < 0.5) & (df.B < 0.5)]
query_way = df.query('A < 0.5 and B < 0.5')
print("query matches masking:", mask_way.equals(query_way))

# query() with a local variable via @
Cmean = df['C'].mean()
sel = df.query('A < @Cmean and B < @Cmean')
print("rows selected:", len(sel))

# Inspect memory footprint of the array backing the DataFrame
print("nbytes:", df.values.nbytes)
```

## Why this matters / intuition
- **Memory is the most predictable payoff.** Masking like `df[(df.A < 0.5) & (df.B < 0.5)]`
  internally builds `tmp1`, `tmp2`, then a combined `tmp3` before indexing. With large
  frames each temporary is a full copy; if those copies exceed RAM, `eval()`/`query()`
  go from "nice" to "necessary."
- **Cache behavior.** Keeping the working set inside the CPU's L1/L2 cache (a few MB)
  avoids slow cache-miss penalties; full temporaries push data out to main memory.
- **Readability.** `df.query('A < 0.5 and B < 0.5')` reads more cleanly than the
  bracketed boolean-mask form.

## Gotchas
- **Speed is not guaranteed for small data.** For modestly sized arrays the traditional
  methods may actually be *faster*; the real wins are memory savings and readability.
  Decide based on whether temporaries strain memory, not reflexively.
- **`@` only works in `DataFrame.eval()`/`query()`**, never in top-level `pd.eval()`.
- **`pd.eval()` is limited:** no function calls, no conditionals, no loops.
- **Column assignment needs `inplace=True`** to modify the DataFrame itself (otherwise
  you get a new object). *(added context: in modern pandas `df.eval('D = ...')` without
  `inplace` returns a copy; assign it back, e.g. `df = df.eval(...)`.)*
- **Name collisions:** a bare name inside `eval`/`query` resolves to a *column*; use `@`
  to force the Python-variable meaning.
- *(added context: VanderPlas notes the rough rule of thumb that `eval`/`query` pay off
  when the temporary arrays are a significant fraction of available system memory
  (gigabytes); below that, prefer plain operations for clarity.)*

## Suggested figure (optional)
A side-by-side memory diagram: **Left** — "Standard NumPy/Pandas": three stacked bars
labeled `tmp1`, `tmp2`, `tmp3` each the full array size, summing to a tall peak-memory
column. **Right** — "Numexpr (eval/query)": a single thin sliver (one cache-line's worth
of elements processed at a time) feeding directly into the final result, illustrating
that no full intermediates are allocated.
---
