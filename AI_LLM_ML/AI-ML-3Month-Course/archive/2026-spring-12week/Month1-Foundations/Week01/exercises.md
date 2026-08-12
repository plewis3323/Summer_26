# Week 01 — Exercises

Work in `Week01/project/notebooks/` or a scratch directory. Commit your solutions — they'll be useful reference for later weeks. Each exercise lists a data source and a difficulty.

## E1 — NumPy shape calisthenics (easy, 30 min)

**Data:** none (purely synthetic).

1. Make `x = np.random.randn(1000, 4)`. Compute:
   - per-column mean and std, with `keepdims=True`.
   - a z-scored copy `z = (x - mean) / std`. Verify `z` has column mean ≈ 0 and std ≈ 1.
2. Given `A = np.random.randn(100, 3)` (particle 4-momentum minus energy; call columns pt, eta, phi), compute cartesian `px, py` without any Python loops.
3. Build a `(1000, 1000)` rank-2 matrix with `M[i, j] = sin(i) * cos(j)` in one line using outer products.
4. Verify `np.linalg.norm` agrees with a hand-computed Frobenius norm on a `(5, 5)` random matrix.

**Hint:** `np.outer`, broadcasting, `keepdims=True`.

## E2 — Broadcasting trap (easy → medium, 20 min)

**Data:** none.

Write `center_rows(x)` and `center_cols(x)` that subtract the row-mean and column-mean, respectively. Make both work for any 2D array shape. Then write a `pytest` test that catches the common bug of dropping `keepdims=True`.

**Accept when:** `pytest -q tests/test_center.py` passes and both functions produce correct results for `(3, 5)`, `(5, 3)`, and `(1, 7)` inputs.

## E3 — pandas: read, filter, groupby (easy, 45 min)

**Data:** CMS dimuon dataset, DoubleMuParked2011 Run-B, available via `uproot` from CERN Open Data (https://opendata.cern.ch/record/5203). A smaller CSV mirror: https://github.com/cms-opendata-workshop/workshop2023-lesson-cmsods/tree/main/data (`DoubleMuRun2011A.csv`).

1. Load into a DataFrame.
2. Add a column `m_inv` computing the invariant mass of the two muons.
3. Filter to `m_inv ∈ (2, 4) GeV` (J/ψ region) and count.
4. Group by run number; compute per-run mean pt of muon 1.
5. Plot `m_inv` histogram on a log y-axis from 0.2 to 120 GeV, step histogram. Annotate J/ψ, Υ, Z regions.

**Hint:** invariant mass from (pt, eta, phi, mass) pairs uses the `TLorentzVector` identity:
```
p1μ + p2μ → M² = (E1+E2)² − |p1+p2|²
```

## E4 — matplotlib: publication figure (medium, 60 min)

**Data:** reuse E3's DataFrame.

Produce a two-panel figure:
- Top panel: dimuon mass, log y, range (0.2, 120) GeV.
- Bottom panel: residuals from a linear-plus-Gaussian fit to the J/ψ peak (use `scipy.optimize.curve_fit`).

Requirements:
- Use OO matplotlib (`fig, (ax0, ax1) = plt.subplots(2, 1, ...)`).
- Share x axis.
- LaTeX axis labels.
- Save to `results/dimuon.pdf` at publication quality.

**Hint:** `plt.subplots(2, 1, sharex=True, figsize=(7, 6), constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]})`.

## E5 — synthetic π⁰ signal + polynomial background toy (medium, 60 min)

**Data:** generate with numpy.

1. Generate 50 000 photon-pair invariant-mass events: 20% signal from a Gaussian at μ=0.135 GeV, σ=0.008 GeV; 80% background from a flat distribution in (0.05, 0.25) GeV.
2. Histogram, fit with `scipy.optimize.curve_fit` using Gaussian + linear background.
3. Return (μ̂, σ̂) with uncertainties from the covariance diagonal.
4. Run 100 toy experiments; plot the pull distribution `(μ̂ - μ_true) / σ_μ̂`. It should be approximately N(0, 1).

**Accept when:** pull mean within 0.05 of zero; pull width within 0.1 of one (for 100 toys).

**Hint:** pull deviates from N(0,1) if your fit function is wrong or your covariance is bad. If it doesn't look right, plot residuals and stare.

## E6 — uproot: read a ROOT file into pandas (medium, 45 min)

**Data:** any sPHENIX/STAR ROOT file you have locally, or download a small one from CERN Open Data:
```bash
wget https://opendata.cern.ch/record/5203/files/CMS_Run2011A_DoubleMu_AOD_12Oct2013-v1_20000_file_index.txt
# or fetch one small nanoAOD file
```

1. Open with `uproot.open(...)`.
2. List the keys, pick a tree, list branches.
3. Load 3–5 branches into a DataFrame with `tree.arrays(branches, library="pd")`.
4. Confirm one scalar calculation matches your ROOT-based equivalent to within floating-point tolerance.

**Hint:** `uproot` is pure Python + numpy; it reads the ROOT file format directly. No ROOT installation needed.

## E7 — pytest drill (medium, 30 min)

Take the `fit_pi0_peak` worked example from `README.md` §8.

1. Drop it into `src/week01/fit.py`.
2. In `tests/test_fit.py`, write three tests:
   - `test_fit_recovers_mu`: generates a known toy, asserts `abs(fit.mu - 0.135) < 3 * fit.mu_err`.
   - `test_fit_positive_sigma`: fit on another toy, asserts `fit.sigma > 0`.
   - `test_empty_input_raises`: passing a zero-length array raises `ValueError`.
3. Run `uv run pytest -q`. All pass.

**Hint:** for "expected to raise," use `pytest.raises`:
```python
import pytest
with pytest.raises(ValueError):
    fit_pi0_peak(np.array([]), np.linspace(0.05, 0.25, 80))
```

## E8 — tiny sklearn sanity check (medium, 30 min)

**Data:** `sklearn.datasets.load_iris()` (classic; don't over-read into this, it's a warm-up).

1. Load iris. Split 70/30 train/test with `sklearn.model_selection.train_test_split(..., random_state=42, stratify=y)`.
2. Fit `sklearn.linear_model.LogisticRegression`.
3. Compute accuracy, balanced accuracy, and a confusion matrix on the test set.
4. Plot the decision boundary in the (sepal length, petal length) projection.

**Hint:** to plot a 2D decision boundary, fit on two features only and evaluate on a meshgrid. `plt.contourf`.

## E9 — reproducibility drill (medium → hard, 45 min)

1. Write `run.py` that generates a toy, fits it, logs `(μ̂, σ̂, log-likelihood, git_sha, wall_clock)` to `results/run_<timestamp>.json`.
2. Run it twice with the same seed; diff the JSONs — they should be identical except for wall_clock.
3. Run it twice with different seeds; compare. Only `wall_clock` and the toy-data-dependent fields should differ.
4. Make `run.py` a CLI with `argparse` exposing `--seed`, `--n-events`, `--output-dir`.

**Hint:** `subprocess.check_output(["git", "rev-parse", "HEAD"])` gets you the SHA.

## E10 — stretch: vectorize an invariant-mass computation on 10⁶ events (hard, 90 min)

**Data:** generate 10⁶ synthetic photon pairs.

1. Write a naive Python loop version of pairwise `m_inv` computation. Measure wall time.
2. Write a numpy-vectorized version. Measure wall time.
3. Ratio should be ≥200×.
4. Plot the per-event timing histograms.

**Hint:** vectorize the four-momentum math; use `np.einsum` if you feel fancy.

**Accept when:** numpy is ≥200× faster than the Python loop on your laptop.

---

## Solution hints (summary)

- E1 — broadcasting + `keepdims`.
- E2 — unit-test the thing you're worried about.
- E3 — `df.query`, `df.groupby().agg`.
- E4 — OO matplotlib; shared axes; LaTeX in labels.
- E5 — pull distribution is the first thing a statistician looks at.
- E6 — `uproot.open` → `.arrays(..., library="pd")`.
- E7 — `pytest.raises` and tolerance-based asserts.
- E8 — `stratify=y` in the split.
- E9 — seed everything; log everything.
- E10 — expect ~500× speedup if you vectorize honestly.

Commit solutions as `Week01/project/notebooks/E01.ipynb` … `E10.ipynb` or `src/week01/exercises/`.
