---
# Notes 13 — Handling Missing Data (Vanderplas Ch. 3.4)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Real-world data is rarely clean; missing values must be represented somehow. There are two broad strategies: a **mask** (a separate Boolean array marking missing entries) or a **sentinel value** (a special in-band value that means "missing"). Pandas chose the sentinel approach, leaning on two existing Python/NumPy nulls:

- **`None`** — the Python singleton object. It only works inside arrays of `dtype=object`, which forces operations to run at slow Python-object speed and breaks compiled aggregations.
- **`NaN`** — the IEEE-754 floating-point "Not a Number" value. It lives in fast native `float64` arrays, so vectorized ops stay compiled, but it propagates through arithmetic.

Pandas papers over the differences between these two and handles converting/upcasting between them automatically.

## Key ideas / idioms
- **`None` = Python object sentinel.** An array containing `None` is upcast to `dtype=object`. Aggregations like `.sum()` raise `TypeError` because Python can't add `int + None`.
- **`NaN` = IEEE floating-point sentinel.** It lives in `dtype=float64`. Arithmetic is well-defined but "viral."
- **NaN propagation.** "NaN is a bit like a data virus — it infects any other object it touches." Any operation with NaN yields NaN:
$$ 1 + \texttt{NaN} = \texttt{NaN}, \qquad 0 \times \texttt{NaN} = \texttt{NaN} $$
  So `arr.sum()`, `arr.min()`, `arr.max()` all return `NaN` if any element is NaN. Use NumPy's NaN-aware versions (`np.nansum`, `np.nanmin`, `np.nanmax`) to ignore missing entries.
- **Automatic upcasting** when NA enters a typed array:
  - Floating → no change, sentinel `np.nan`
  - Object → no change, sentinel `None` or `np.nan`
  - Integer → cast to `float64`, sentinel `np.nan`
  - Boolean → cast to `object`, sentinel `None` or `np.nan`
- **Pandas treats `None` and `NaN` as interchangeable** for missing-value purposes and converts between them as needed.
- **Four core methods:** `isnull()` / `notnull()` (detect), `dropna()` (remove), `fillna()` (replace).

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd

# 1) None forces dtype=object and breaks aggregation
vals1 = np.array([1, None, 3, 4])
print(vals1.dtype)                 # object
try:
    print(vals1.sum())
except TypeError as e:
    print("None array sum -> TypeError:", e)

# 2) NaN lives in float64; aggregations propagate NaN
vals2 = np.array([1, np.nan, 3, 4])
print(vals2.dtype)                 # float64
print(1 + np.nan, 0 * np.nan)      # nan nan
print(vals2.sum(), vals2.min(), vals2.max())   # nan nan nan

# NaN-aware aggregations skip the missing value
print(np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2))  # 8.0 1.0 4.0
```

```python
import numpy as np
import pandas as pd

# 3) Pandas upcasts int -> float64 when a null is introduced
x = pd.Series(range(2), dtype=int)
print(x.dtype)        # int64
x[0] = None
print(x.dtype)        # float64  (None became NaN)
print(x)
```

```python
import numpy as np
import pandas as pd

# 4) Detecting nulls
data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())     # True where missing
print(data.notnull())    # opposite mask
print(data[data.notnull()])   # boolean-index out the non-null values
```

```python
import numpy as np
import pandas as pd

# 5) Dropping nulls in a Series
data = pd.Series([1, np.nan, 'hello', None])
print(data.dropna())

# DataFrames: cannot drop single cells, only whole rows/columns
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
print(df.dropna())                 # default axis=0: drop any row with NA
print(df.dropna(axis='columns'))   # drop any column with NA
df[3] = np.nan
print(df.dropna(axis='columns', how='all'))   # only drop all-NA columns
print(df.dropna(axis='rows', thresh=3))       # keep rows with >= 3 non-null
```

```python
import numpy as np
import pandas as pd

# 6) Filling nulls
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data.fillna(0))              # replace with a constant
print(data.fillna(method='ffill')) # forward-fill: carry previous value forward
print(data.fillna(method='bfill')) # back-fill: carry next value backward

# DataFrame fills can take an axis (fill direction)
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3,      5],
                   [np.nan, 4, 6]])
print(df.fillna(method='ffill', axis=1))   # fill across columns
```

## Why this matters / intuition
Missing data is the rule, not the exception, in real datasets. Understanding *which* sentinel Pandas is using tells you about performance (object dtype = slow Python loop; float64 = fast compiled) and about silent type changes (your integer column quietly becoming float). Knowing that NaN propagates explains why a single missing value can turn an entire `.sum()` into `NaN` — and why the NaN-aware aggregations exist. The `isnull`/`dropna`/`fillna` trio covers the full workflow: detect, then either remove or impute.

## Gotchas
- `np.array([1, None, 3]).sum()` raises `TypeError`, but `np.array([1, np.nan, 3]).sum()` returns `nan` — different failure modes for the two sentinels.
- Introducing a null into an **integer** Series silently upcasts it to **float64**; into a **boolean** Series upcasts to **object**.
- Plain `arr.sum()`/`.min()`/`.max()` return `NaN` if any value is NaN — reach for `np.nansum`, etc., to ignore them.
- On a DataFrame you cannot drop a single missing cell — `dropna()` removes the entire row or column. Use `how`/`thresh` to control how aggressive it is.
- `NaN == NaN` is `False` *(added context: IEEE semantics — never test for missingness with `==`; use `isnull()`)*.
- *(added context: in newer Pandas, `fillna(method='ffill')` is deprecated in favor of `df.ffill()` / `df.bfill()`; the handbook predates this.)*

## Suggested figure (optional)
A two-panel side-by-side diagram. Left panel: an array `[1, None, 3, 4]` boxed as `dtype=object`, with each cell drawn as a separate Python object pointer, and a red "TypeError" stamp over a `sum()` arrow. Right panel: an array `[1, NaN, 3, 4]` boxed as a contiguous `float64` block, with arrows showing NaN "infecting" the running total so `sum() -> NaN`, plus a green branch labeled `np.nansum -> 8.0` that skips the NaN cell. A small legend maps each upcasting rule (int→float64, bool→object).
