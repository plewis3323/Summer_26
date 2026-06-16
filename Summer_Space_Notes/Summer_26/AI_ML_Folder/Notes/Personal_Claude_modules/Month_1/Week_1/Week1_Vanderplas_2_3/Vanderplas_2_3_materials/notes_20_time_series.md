---
# Notes 20 — Working with Time Series (Vanderplas Ch. 3.11)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

Time series work in Python sits on three layers, from most flexible to most efficient:

1. **Native `datetime` / `dateutil`** — built-in `datetime` module (convenient methods) plus the third-party `dateutil` for flexible string parsing. Easy to use, but *not vectorized*, so they are slow on large arrays.
2. **NumPy `datetime64`** — encodes a date as a 64-bit integer for compact, vectorized operations. Requires rigid input (e.g. `'2015-07-04'`) and forces a tradeoff between time *resolution* (years → attoseconds) and maximum time *span*. Nanosecond (`'ns'`) is the practical default for modern dates.
3. **Pandas time tools** — `Timestamp`, `Period`, `Timedelta` (and their index versions) combine the ergonomics of `datetime` with the efficiency of `datetime64`. These power time-indexed `Series`/`DataFrame` objects, which is what you actually use day-to-day.

Three scalar types and their matching index types:

| Concept | Scalar | Index | Built by |
|---|---|---|---|
| A specific moment | `Timestamp` | `DatetimeIndex` | `pd.to_datetime`, `pd.date_range` |
| A fixed-frequency interval | `Period` | `PeriodIndex` | `.to_period()`, `pd.period_range` |
| A duration | `Timedelta` | `TimedeltaIndex` | date subtraction, `pd.timedelta_range` |

## Key ideas / idioms

- **`pd.to_datetime`** parses a wide variety of formats (and lists of mixed formats) into a `DatetimeIndex`.
- A `Series`/`DataFrame` with a `DatetimeIndex` supports **intuitive date-based slicing**: pass date strings, ranges of date strings, or even a coarser unit like a year (`data['2015']`) to select all matching rows.
- **`Period` arithmetic**: subtracting `Timestamp`s yields a `Timedelta` / `TimedeltaIndex`.
- **`to_period(freq)`** converts a `DatetimeIndex` into a `PeriodIndex` (intervals instead of instants).
- **Regular sequences**: `pd.date_range` (timestamps), `pd.period_range` (periods), `pd.timedelta_range` (durations). Each takes either `start, end` or `start, periods=N`, plus an optional `freq`.

### Frequency-code table

| Code | Meaning | Code | Meaning |
|------|---------|------|---------|
| `D`  | Calendar day | `B`  | Business day |
| `W`  | Weekly | `M`  | Month end |
| `BM` | Business month end | `Q`  | Quarter end |
| `BQ` | Business quarter end | `A`  | Year end |
| `BA` | Business year end | `H`  | Hours |
| `BH` | Business hours | `T` / `min` | Minutes |
| `S`  | Seconds | `L` / `ms` | Milliseconds |
| `U` / `us` | Microseconds | `N`  | Nanoseconds |

Variants and modifiers:

- **Period-start variants**: add `S` → `MS` (month start), `QS` (quarter start), `AS` (year start). The non-`S` codes (`M`, `Q`, `A`) mark the *end* of the period.
- **Anchored offsets**: append a month/day → `Q-JAN` (quarters anchored to January), `A-DEC` (years ending in December), `W-MON` (weeks anchored on Monday).
- **Combinations**: prefix a number and chain codes → `"2H30T"` for 2 hours 30 minutes.

### resample vs. asfreq

The primary difference:

$$
\text{resample()} \;=\; \text{data \textbf{aggregation}}, \qquad
\text{asfreq()} \;=\; \text{data \textbf{selection}}
$$

- **`resample()`** computes a statistic over each new bin. Downsampling → aggregate (`.mean()`, `.sum()`); upsampling → typically NaNs unless filled.
- **`asfreq()`** picks the value *at* each new time point. Missing points can be filled via `method='ffill'` (forward) or `method='bfill'` (backward).
- Example contrast: `goog.resample('BA').mean()` gives the *average over each year*, while `goog.asfreq('BA')` gives the *value at year-end*.

## Worked code examples (runnable)

```python
import numpy as np
import pandas as pd
from datetime import datetime

# --- Parsing mixed formats into a DatetimeIndex ---
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                        '2015-Jul-6', '07-07-2015', '20150708'])
print(dates)

# DatetimeIndex -> PeriodIndex (intervals); subtraction -> TimedeltaIndex
print(dates.to_period('D'))
print(dates - dates[0])
```

```python
import numpy as np
import pandas as pd

# --- Date-based indexing & slicing ---
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)

print(data['2014-07-04':'2015-07-04'])   # range slice by date strings
print(data['2015'])                       # all of 2015
```

```python
import numpy as np
import pandas as pd

# --- Regular sequences ---
print(pd.date_range('2015-07-03', '2015-07-10'))          # start + end (daily)
print(pd.date_range('2015-07-03', periods=8))             # start + count
print(pd.date_range('2015-07-03', periods=8, freq='H'))   # hourly

print(pd.period_range('2015-07', periods=8, freq='M'))    # monthly periods
print(pd.timedelta_range(0, periods=10, freq='H'))        # hourly durations
print(pd.timedelta_range(0, periods=9, freq='2H30T'))     # custom: 2h30m steps
```

```python
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# --- Frequency offsets: business days ---
print(pd.date_range('2015-07-01', periods=5, freq=BDay()))
```

```python
import numpy as np
import pandas as pd

# --- resample vs asfreq vs rolling ---
# (added context) The book uses downloaded Google stock prices ("goog").
# We synthesize a daily series with pd.date_range so this runs standalone.
rng = pd.date_range('2010-01-01', '2014-12-31', freq='B')   # business days
rng_seed = np.random.RandomState(0)
goog = pd.Series(500 + np.cumsum(rng_seed.randn(len(rng))), index=rng)

# Aggregation vs selection at business-year-end:
print(goog.resample('BA').mean())   # average price per year (aggregation)
print(goog.asfreq('BA'))            # price at each year-end (selection)

# Upsampling with forward-fill:
print(goog.asfreq('D', method='ffill').head())

# Rolling window: centered 365-day moving average (smoothing)
rolling = goog.rolling(365, center=True)
print(rolling.mean().dropna().head())
```

## Why this matters / intuition

- Real-world data (finance, sensors, logs, web traffic) is overwhelmingly time-indexed; pandas turns "select July 2015" or "give me the yearly average" into one-liners instead of manual index math.
- The `Timestamp` / `Period` / `Timedelta` trio maps cleanly onto the three questions you ask of time: *when* (instant), *which interval* (period), and *how long* (duration).
- **resample vs asfreq is the conceptual crux**: changing frequency is ambiguous — do you want a *summary* of each new bin, or the *snapshot* at each new tick? Knowing which one you mean prevents silently wrong charts.
- **Rolling windows** are the entry point to smoothing and trend extraction — averaging out short-term noise to reveal long-term structure.

## Gotchas

- **`datetime`/`dateutil` don't vectorize** — fine for a handful of dates, a bottleneck for large arrays; prefer the pandas/`datetime64` path at scale.
- **`datetime64` resolution vs. span tradeoff**: the chosen unit caps the representable date range; `'ns'` is standard but has a narrower span than coarser units.
- **End vs. start codes**: `M`/`Q`/`A` land on the *end* of the period; you need `MS`/`QS`/`AS` for the *start*. Easy to be off by a full period.
- **`resample` upsampling produces NaN** at the newly introduced finer points unless you aggregate or fill — don't assume values appear automatically.
- **`asfreq` selects, it does not aggregate** — it returns only the exact values at the requested ticks (with optional `ffill`/`bfill`), which can drop data if your grid doesn't line up.
- **Centered rolling windows** leave NaNs at both ends (no full window available), so plots/stats should account for the trimmed edges.

## Suggested figure (optional)

A single line chart of the synthetic daily `goog` series (thin, light line) overlaid with its centered 365-day rolling mean (thick, dark line). Add markers at each `asfreq('BA')` year-end point and short horizontal segments showing each `resample('BA').mean()` yearly average. The visual contrast — wiggly raw data, smooth rolling trend, discrete year-end dots vs. flat yearly-average bars — makes the selection-vs-aggregation distinction immediately obvious.
