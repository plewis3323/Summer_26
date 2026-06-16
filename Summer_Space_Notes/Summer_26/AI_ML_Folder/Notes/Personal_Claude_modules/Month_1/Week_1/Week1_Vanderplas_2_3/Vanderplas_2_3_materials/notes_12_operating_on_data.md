---
# Notes 12 — Operating on Data in Pandas (Vanderplas Ch. 3.3)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.03-operations-in-pandas.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Pandas inherits NumPy's element-wise (ufunc) machinery but adds **label awareness**:

- **Unary ufuncs** (negation, trig, exp, ...) **preserve the index and column labels** of the input.
- **Binary ufuncs** (`+`, `*`, ...) **automatically align indices** before computing, so you operate on data matched by label rather than by position.

This means messy, differently-ordered, or partially-overlapping datasets combine correctly, with non-overlapping entries filled by `NaN` (or a fill value you choose).

## Key ideas / idioms

- **Ufunc index preservation:** `np.exp(ser)`, `np.sin(df * np.pi/4)` return a Series/DataFrame with the same labels.
- **Series alignment:** result index is the **union** of the two input indices; missing labels become `NaN`.
- **DataFrame alignment:** **both rows and columns** are aligned; the result index/columns are the union, sorted, with missing intersections `NaN`.
- **Fill values:** use the object methods (not the operators) to supply `fill_value`, e.g. `A.add(B, fill_value=0)`.
- **DataFrame–Series ops broadcast row-wise by default** (the Series aligns to columns, applied across each row). Use `axis=0` to broadcast column-wise instead.

Operator ⟷ method equivalents:

$$
\begin{array}{ll}
\texttt{+} & \texttt{add()} \\
\texttt{-} & \texttt{sub(), subtract()} \\
\texttt{*} & \texttt{mul(), multiply()} \\
\texttt{/} & \texttt{truediv(), div(), divide()} \\
\texttt{//} & \texttt{floordiv()} \\
\texttt{\%} & \texttt{mod()} \\
\texttt{**} & \texttt{pow()}
\end{array}
$$

Index union (conceptually): for Series $A$ with index $I_A$ and $B$ with index $I_B$,

$$ \text{index}(A \oplus B) = I_A \cup I_B $$

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# --- Ufuncs preserve index / column labels ---
rng = np.random.RandomState(42)

ser = pd.Series(rng.randint(0, 10, 4))
print(np.exp(ser))          # same index 0..3, exp applied elementwise

df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
print(np.sin(df * np.pi / 4))   # same rows 0..2, columns A..D
```

```python
import numpy as np
import pandas as pd

# --- Index alignment in Series (union -> NaN for missing) ---
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

print(population / area)    # California, Texas have values; Alaska, New York -> NaN

# --- Custom fill value via the method form ---
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A + B)                # index 1,2 computed; 0,3 -> NaN
print(A.add(B, fill_value=0))   # 0->2.0, 1->5.0, 2->9.0, 3->5.0
```

```python
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)

# --- Index alignment in DataFrame (rows AND columns aligned) ---
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC'))
print(A + B)                # union of rows/cols; non-overlap -> NaN

# Fill missing intersections with the mean of all values in A
fill = A.stack().mean()
print(A.add(B, fill_value=fill))
```

```python
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)

# --- DataFrame and Series operations ---
A = rng.randint(10, size=(3, 4))
df = pd.DataFrame(A, columns=list('QRST'))

# Row-wise by default: subtract first row from every row
print(df - df.iloc[0])

# Column-wise: broadcast a column down each row with axis=0
print(df.subtract(df['R'], axis=0))

# Operations also align by index/columns -> NaN where unmatched
halfrow = df.iloc[0, ::2]   # columns Q and S only
print(df - halfrow)         # R and T columns -> NaN
```

## Why this matters / intuition

Raw NumPy operations match by **position**, so combining datasets that are ordered differently or have different membership silently produces wrong answers. Pandas matches by **label**, so "California population / California area" is computed regardless of row order, and any mismatch shows up loudly as `NaN` instead of a silent misalignment. The data keeps its context through every operation.

## Gotchas

- The **operator** form (`A + B`) cannot take a `fill_value`; you must use the **method** form (`A.add(B, fill_value=...)`).
- Filling with `0` is not always right — for a DataFrame, `A.stack().mean()` (mean of all entries) is one reasonable choice; pick a fill that makes sense for your data.
- `NaN` propagates: any unmatched label/intersection becomes `NaN`, which can then poison later reductions unless handled.
- DataFrame–Series ops are **row-wise by default**; forgetting `axis=0` for a column-wise operation is a common silent bug.
- Result indices/columns of aligned ops are returned **sorted** (union), so order may differ from either input. *(added context)*

## Suggested figure (optional)

A two-panel diagram. Left: two Series drawn as labeled boxes with partially overlapping index labels; arrows show alignment by label and a union index on the output, with `NaN` boxes where a label exists in only one input. Right: a DataFrame grid plus a Series shown once as a row (default broadcast, arrows fanning downward across rows) and once as a column (`axis=0`, arrows fanning rightward across columns), illustrating the two broadcast directions.

---

## 💬 Q&A (captured during session)

### Q: Help me read the arguments in `pd.DataFrame(rng.randint(0, 20, (2, 2)), ...)`.

Read it **inside-out** — two nested calls.

**Inner — `rng.randint(0, 20, (2, 2))`** draws random integers from the seeded generator `rng`:

| Position | Value | Name | Meaning |
|---|---|---|---|
| 1st | `0` | `low` | smallest int it can draw (**inclusive**) |
| 2nd | `20` | `high` | upper bound (**exclusive** → max is `19`) |
| 3rd | `(2, 2)` | `size` | **shape** of the output: 2 rows × 2 cols (one tuple, not two numbers) |

→ a 2×2 NumPy array of ints in `0..19`.

**Outer — `pd.DataFrame(...)`** wraps that array into a labeled table. With no other args, rows
and columns auto-label `0,1`; with `columns=list('AB')` the columns become `'A','B'`:

```python
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
#     A   B
# 0   3  17
# 1   9   2
```

```
rng.randint( 0 , 20 , (2,2) )   ->   2x2 array   ->   pd.DataFrame(arr, columns=list('AB'))
            low  high  shape                           wrap as labeled table, name cols A,B
```

**One breath:** "draw a 2×2 grid of random ints 0–19, then wrap it in a DataFrame named `A`."

*(note: a snippet like `pd.DataFrame(rng.randint(0, 20, (2, 2))` with no closing `)` is just
missing a paren — the book line continues with `, columns=...)`.)*
---
