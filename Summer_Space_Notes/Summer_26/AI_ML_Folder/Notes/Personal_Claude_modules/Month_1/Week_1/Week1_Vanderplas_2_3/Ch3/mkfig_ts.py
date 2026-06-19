"""Generate the two figures used in 11_time_series.tex.

Builds a synthetic *daily* time series over ~2 years: a cumulative-sum
random walk (the trend/drift) plus a seasonal sine term (the yearly cycle).
Saves two PNGs at ~150 dpi into the working directory:

  ts_raw.png        -- the raw daily series
  ts_resample.png   -- raw (faint) + monthly resample().mean()
                       + 30-day rolling().mean()
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: no display needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- build the synthetic daily series ----------------------------------
rng = np.random.default_rng(0)               # fixed seed -> reproducible
dates = pd.date_range("2021-01-01", periods=730, freq="D")  # ~2 years
n = len(dates)

# day-of-year index for the seasonal term (0..1 across a year)
day_of_year = dates.dayofyear.to_numpy()
seasonal = 8.0 * np.sin(2 * np.pi * day_of_year / 365.0)

# cumulative-sum random walk = the drift / trend component
steps = rng.normal(loc=0.02, scale=1.0, size=n)
walk = np.cumsum(steps)

# small extra daily noise so the raw series looks "jittery"
noise = rng.normal(scale=1.5, size=n)

values = 50.0 + walk + seasonal + noise
series = pd.Series(values, index=dates, name="signal")

# ---- figure 1: raw daily series ----------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(series.index, series.values, color="#1F4E79", lw=0.8)
ax.set_title("Synthetic daily series: random-walk drift + yearly season")
ax.set_xlabel("date")
ax.set_ylabel("signal value")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("ts_raw.png", dpi=150)
plt.close(fig)

# ---- figure 2: raw + monthly resample + 30-day rolling -----------------
monthly = series.resample("M").mean()        # down-sample: one point / month
rolled = series.rolling(window=30, center=True).mean()  # smoothing window

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(series.index, series.values, color="#9DB7CF", lw=0.7,
        alpha=0.6, label="raw daily")
ax.plot(rolled.index, rolled.values, color="#2E7D32", lw=1.8,
        label="30-day rolling mean")
ax.plot(monthly.index, monthly.values, color="#B23A00", lw=1.8,
        marker="o", ms=3, label="monthly resample mean")
ax.set_title("Resampling vs. rolling: two ways to smooth a time series")
ax.set_xlabel("date")
ax.set_ylabel("signal value")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("ts_resample.png", dpi=150)
plt.close(fig)

print("wrote ts_raw.png and ts_resample.png")
