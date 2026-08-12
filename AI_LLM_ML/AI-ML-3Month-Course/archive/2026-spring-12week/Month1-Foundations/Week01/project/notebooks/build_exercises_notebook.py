#!/usr/bin/env python3
"""Build Week01_Exercises.ipynb. Re-run after editing CELLS:

    python3 notebooks/build_exercises_notebook.py

Style rules for what goes in the notebook: ../../../NOTEBOOK_RULES.md
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


md(r'''# Week 01 — Exercises E1–E10

Working notebook for `Week01/exercises.md`. Fill in each `# TODO` and run the cell;
the `check(...)` lines print PASS/FAIL against the acceptance criteria on the exercise page.

| | Exercise | Difficulty | Work in |
|---|---|---|---|
| E1 | NumPy shape calisthenics | easy, 30 min | this notebook |
| E2 | Broadcasting trap | easy→medium, 20 min | `src/week01/center.py`, `tests/test_center.py` |
| E3 | pandas: read, filter, groupby | easy, 45 min | this notebook |
| E4 | matplotlib: publication figure | medium, 60 min | this notebook |
| E5 | π⁰ signal + background toy | medium, 60 min | this notebook |
| E6 | uproot → pandas | medium, 45 min | this notebook |
| E7 | pytest drill | medium, 30 min | `tests/test_fit.py` |
| E8 | sklearn sanity check | medium, 30 min | this notebook |
| E9 | Reproducibility drill | medium→hard, 45 min | `run.py` |
| E10 | Vectorize 10⁶ invariant masses | hard, 90 min | this notebook |

Data is downloaded once and cached in `../data/`: about 72 MB for E3/E4, 180 kB for E6.''')

md(r'''## Setup

Run this first. Start Jupyter in this `notebooks/` folder — every path below is relative to it.''')

code(r'''import sys

sys.path.insert(0, "../src")   # so "import week01" works even without uv sync

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check(label, ok):
    if ok:
        print("PASS  " + label)
    else:
        print("FAIL  " + label)


print("numpy", np.__version__, "| pandas", pd.__version__)''')

# ---------------------------------------------------------------- E1
md(r'''---
# E1 — NumPy shape calisthenics

*easy, 30 min · no data*

1. Per-column mean and std with `keepdims=True`; z-score `x`; check column mean ≈ 0, std ≈ 1.
2. From `(pt, eta, phi)` compute cartesian `px, py` with no Python loops.
3. Build `M[i, j] = sin(i) * cos(j)` at 1000×1000 in one line using an outer product.
4. Check `np.linalg.norm` against a Frobenius norm you compute yourself on a 5×5 matrix.

**Hint:** `np.outer`, broadcasting, `keepdims=True`.''')

md(r'''### E1.1 — z-score''')

code(r'''rng = np.random.default_rng(0)
x = rng.standard_normal((1000, 4))

mean = None   # TODO: per-column mean, keepdims=True  -> shape (1, 4)
std = None    # TODO: per-column std, keepdims=True   -> shape (1, 4)
z = None      # TODO: (x - mean) / std

check("mean has shape (1, 4)", mean is not None and mean.shape == (1, 4))
check("std has shape (1, 4)", std is not None and std.shape == (1, 4))
check("z column means are ~0", z is not None and np.allclose(z.mean(axis=0), 0))
check("z column stds are ~1", z is not None and np.allclose(z.std(axis=0), 1))''')

md(r'''### E1.2 — (pt, eta, phi) → (px, py)

$$p_x = p_T\cos\phi, \qquad p_y = p_T\sin\phi$$''')

code(r'''A = np.column_stack([
    rng.uniform(0.5, 50, 100),        # pt  [GeV]
    rng.uniform(-2.5, 2.5, 100),      # eta
    rng.uniform(-np.pi, np.pi, 100),  # phi
])
pt = A[:, 0]
eta = A[:, 1]
phi = A[:, 2]

px = None   # TODO
py = None   # TODO

check("px has shape (100,)", px is not None and px.shape == (100,))
check("py has shape (100,)", py is not None and py.shape == (100,))
check("px^2 + py^2 == pt^2", px is not None and np.allclose(px**2 + py**2, pt**2))''')

md(r'''### E1.3 — Outer product, one line''')

code(r'''i = np.arange(1000)
j = np.arange(1000)

M = None   # TODO: one line, np.outer, M[i, j] = sin(i) * cos(j)

check("shape is (1000, 1000)", M is not None and M.shape == (1000, 1000))
check("M[3, 7] is sin(3) * cos(7)", M is not None and np.isclose(M[3, 7], np.sin(3) * np.cos(7)))''')

md(r'''### E1.4 — Frobenius norm by hand

$$\|A\|_F = \sqrt{\textstyle\sum_{ij} A_{ij}^2}$$''')

code(r'''B = rng.standard_normal((5, 5))

frob = None   # TODO: compute it without using np.linalg.norm

check("agrees with np.linalg.norm", frob is not None and np.isclose(frob, np.linalg.norm(B)))''')

# ---------------------------------------------------------------- E2
md(r'''---
# E2 — Broadcasting trap

*easy → medium, 20 min · no data*

Write `center_rows(x)` and `center_cols(x)`, subtracting the row-mean and the column-mean.
Both have to work for any 2D shape. Then write a pytest test that catches the common bug of
dropping `keepdims=True`.

**Accept when:** `pytest -q tests/test_center.py` passes, and both functions are right for
`(3, 5)`, `(5, 3)` and `(1, 7)`.

Edit `../src/week01/center.py` and `../tests/test_center.py`. Try it out here first if you like.''')

code(r'''y = np.arange(15, dtype=float).reshape(3, 5)

# TODO: try center_rows / center_cols here, then move them into src/week01/center.py
''')

code(r'''!{sys.executable} -m pytest -q ../tests/test_center.py''')

# ---------------------------------------------------------------- E3
md(r'''---
# E3 — pandas: read, filter, groupby

*easy, 45 min*

1. Load the CMS dimuon dataset into a DataFrame.
2. Add a column `m_inv`, the invariant mass of the two muons.
3. Filter to `m_inv ∈ (2, 4)` GeV (the J/ψ region) and count.
4. Group by run number; compute the per-run mean `pt1`.
5. Plot the `m_inv` histogram, log y, 0.2–120 GeV, step histogram, with J/ψ, Υ and Z annotated.

**Hint:** $M^2 = (E_1+E_2)^2 - \|\vec p_1 + \vec p_2\|^2$.

*Note: the CSV link in `exercises.md` 404s; `load_dimuon()` uses two live mirrors of the same file.*''')

code(r'''from week01.data import load_dimuon

df = load_dimuon()   # about 72 MB on the first run, then cached in ../data/
print(df.shape)
df.head()''')

md(r'''### E3.2 — Invariant mass

The file ships CMS's own `M` column. Compute your own, then compare against it.''')

code(r'''# TODO: df["m_inv"] = ...

check("column exists", "m_inv" in df.columns)
check("no NaNs", "m_inv" in df.columns and df["m_inv"].notna().all())

if "m_inv" in df.columns:
    diff = (df["m_inv"] - df["M"]).abs()
    print(f"median |m_inv - M| = {diff.median():.2e} GeV")
    check("agrees with CMS for a typical event", diff.median() < 1e-3)''')

md(r'''### E3.3 — J/ψ window''')

code(r'''jpsi = None     # TODO: the rows with 2 < m_inv < 4
n_jpsi = None   # TODO: how many there are

check("selection is a DataFrame", isinstance(jpsi, pd.DataFrame))
check("inside the window", jpsi is not None and len(jpsi) > 0 and jpsi["m_inv"].between(2, 4).all())
check("count matches the selection", n_jpsi is not None and n_jpsi == len(jpsi))
print("J/psi-region events:", n_jpsi)''')

md(r'''### E3.4 — Per-run mean $p_T$''')

code(r'''per_run = None   # TODO: group by "Run", take the mean of "pt1"

check("one row per run", per_run is not None and len(per_run) == df["Run"].nunique())
check("indexed by Run", per_run is not None and per_run.index.name == "Run")
if per_run is not None:
    print(per_run.head())''')

md(r'''### E3.5 — The spectrum

Use the log-spaced bins below — with linear bins the whole low-mass region lands in one bin.''')

code(r'''bins = np.logspace(np.log10(0.2), np.log10(120), 300)

fig, ax = plt.subplots(figsize=(9, 5))
# TODO: step histogram of m_inv with these bins; log x and log y;
#       annotate J/psi (3.1), Upsilon (9.5) and Z (91)
ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
ax.set_ylabel("events / bin")
plt.show()''')

# ---------------------------------------------------------------- E4
md(r'''---
# E4 — matplotlib: publication figure

*medium, 60 min · reuses E3's DataFrame*

Two panels:
- **top:** dimuon mass, log y, range 0.2–120 GeV
- **bottom:** residuals from a linear-plus-gaussian fit to the J/ψ peak (`scipy.optimize.curve_fit`)

Requirements: OO matplotlib, shared x axis, LaTeX axis labels, saved to `results/dimuon.pdf`
at publication quality.''')

md(r'''### E4.1 — Fit the J/ψ peak

`signal_plus_bg(x, mu, sigma, norm, a, b)` is gaussian + straight line, provided in
`week01.fit`. `p0` and `sigma=sqrt(counts + 1)` are the arguments `curve_fit` needs from you.''')

code(r'''from scipy.optimize import curve_fit

from week01.fit import signal_plus_bg

window = (2.6, 3.6)
n_bins = 60

mass = df["m_inv"].to_numpy()
sel = mass[(mass > window[0]) & (mass < window[1])]
counts, edges = np.histogram(sel, bins=n_bins, range=window)
centers = 0.5 * (edges[:-1] + edges[1:])

popt = None     # TODO: curve_fit(...)[0], starting from a peak near 3.097 GeV
fitted = None   # TODO: signal_plus_bg(centers, popt[0], popt[1], popt[2], popt[3], popt[4])
resid = None    # TODO: (counts - fitted) / np.sqrt(counts + 1)

check("five fitted parameters", popt is not None and len(popt) == 5)
check("peak found near 3.1 GeV", popt is not None and abs(popt[0] - 3.097) < 0.05)
check("one residual per bin", resid is not None and len(resid) == n_bins)''')

md(r'''### E4.2 — The figure''')

code(r'''fig, (ax0, ax1) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6), constrained_layout=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

# TODO: ax0 -> step histogram of mass over bins, log y (and log x), fitted curve on top
# TODO: ax1 -> resid against centers, with a horizontal line at 0

ax0.set_ylabel("events / bin")
ax1.set_ylabel(r"$(N - f)/\sigma$")
ax1.set_xlabel(r"$m_{\mu^{+}\mu^{-}}$ [GeV]")

# TODO: save to "../results/dimuon.pdf" at 300 dpi or better
plt.show()''')

code(r'''import os

check("two axes", len(fig.axes) == 2)
check("x axis is shared", fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1]))
check("top panel y is log", fig.axes[0].get_yscale() == "log")
check("LaTeX in the x label", "$" in fig.axes[1].get_xlabel())
check("results/dimuon.pdf written", os.path.exists("../results/dimuon.pdf"))''')

# ---------------------------------------------------------------- E5
md(r'''---
# E5 — Synthetic π⁰ signal + polynomial background toy

*medium, 60 min*

1. Generate 50 000 photon-pair masses: 20% gaussian signal at μ=0.135, σ=0.008 GeV; 80% flat
   background in (0.05, 0.25) GeV. (`make_pi0_toy` does this.)
2. Histogram and fit with `curve_fit`, gaussian + straight-line background.
3. Return (μ̂, σ̂) with uncertainties from the diagonal of the covariance.
4. Run 100 toys; plot the pull distribution `(μ̂ - μ_true) / σ_μ̂`. It should be about N(0, 1).

**Accept when:** pull mean within 0.05 of zero, pull width within 0.1 of one.''')

md(r'''### E5.1–3 — One toy, one fit''')

code(r'''from week01.data import make_pi0_toy

MU_TRUE = 0.135
SIGMA_TRUE = 0.008
BIN_EDGES = np.linspace(0.05, 0.25, 80)

m_gg = make_pi0_toy(n=50000, signal_frac=0.20, seed=0)


def fit_one(sample):
    """Return mu_hat, mu_err, sigma_hat, sigma_err for one toy."""
    # TODO: histogram over BIN_EDGES -> bin centers -> curve_fit(signal_plus_bg, ...)
    #       with sigma=np.sqrt(counts + 1); the errors are the square roots of the
    #       diagonal of the covariance. Return abs(sigma_hat): the model only sees
    #       sigma**2, so the sign it comes back with is arbitrary.
    return None, None, None, None


mu_hat, mu_err, sigma_hat, sigma_err = fit_one(m_gg)
print("mu    =", mu_hat, "+/-", mu_err)
print("sigma =", sigma_hat, "+/-", sigma_err)

check("mu recovered", mu_hat is not None and abs(mu_hat - MU_TRUE) < 3 * mu_err)
check("sigma recovered", sigma_hat is not None and abs(sigma_hat - SIGMA_TRUE) < 0.002)
check("errors are positive", mu_err is not None and mu_err > 0 and sigma_err > 0)''')

md(r'''Draw the fit over the histogram before you trust the numbers.''')

code(r'''bin_centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
n, _edges = np.histogram(m_gg, bins=BIN_EDGES)

fig, ax = plt.subplots(figsize=(7, 4))
# TODO: errorbar the histogram (yerr=sqrt(n)), draw the fitted curve on top
ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
ax.set_ylabel("events / bin")
plt.show()''')

md(r'''### E5.4 — Pull distribution

$$\text{pull} = \frac{\hat\mu - \mu_{\text{true}}}{\hat\sigma_{\hat\mu}}$$

*Note: at 100 toys the mean of a perfect N(0,1) pull has a statistical error of
$1/\sqrt{100}=0.10$, so the "within 0.05" criterion fails more often than not even with a
correct fit. Both errors are printed below so you can tell a real bias from noise.*''')

code(r'''N_TOYS = 100
pulls = []

for t in range(N_TOYS):
    pass   # TODO: fresh toy with seed=t, fit it with fit_one, append the pull

pulls = np.asarray(pulls, dtype=float)
pull_mean = None    # TODO
pull_width = None   # TODO: standard deviation, ddof=1

se_mean = 1 / np.sqrt(N_TOYS)
se_width = 1 / np.sqrt(2 * (N_TOYS - 1))
if pull_mean is not None:
    print(f"pull mean  = {pull_mean:+.3f}  (statistical error {se_mean:.3f})")
    print(f"pull width =  {pull_width:.3f}  (statistical error {se_width:.3f})")

check("ran every toy", len(pulls) == N_TOYS)
check("|mean| < 0.05", pull_mean is not None and abs(pull_mean) < 0.05)
check("|width - 1| < 0.1", pull_width is not None and abs(pull_width - 1) < 0.1)''')

code(r'''fig, ax = plt.subplots(figsize=(6, 4))
# TODO: histogram the pulls, draw a unit gaussian on top
ax.set_xlabel(r"$(\hat\mu - \mu_{\rm true})/\hat\sigma_{\hat\mu}$")
plt.show()''')

# ---------------------------------------------------------------- E6
md(r'''---
# E6 — uproot: read a ROOT file into pandas

*medium, 45 min*

1. Open the file with `uproot.open(...)`.
2. List the keys, pick a tree, list its branches.
3. Load 3–5 branches into a DataFrame with `tree.arrays(branches, library="pd")`.
4. Confirm one scalar calculation matches the value stored in the file, to floating-point tolerance.

`zmumu_root_path()` fetches a small (about 180 kB) real ROOT file of dimuon events.''')

code(r'''import uproot

from week01.data import zmumu_root_path

path = zmumu_root_path()
f = uproot.open(path)

keys = None       # TODO: f.keys()
tree = None       # TODO: pick the TTree
branches = None   # TODO: its branch names

print("keys:", keys)
print("branches:", branches)

check("listed keys", keys is not None and len(keys) > 0)
check("got a TTree", tree is not None and hasattr(tree, "arrays"))
check("listed branches", branches is not None and len(branches) > 5)''')

code(r'''want = None   # TODO: 3-5 branch names, for example ["E1", "px1", "py1", "pz1"]
rdf = None    # TODO: tree.arrays(want, library="pd")

check("is a DataFrame", isinstance(rdf, pd.DataFrame))
check("3-5 branches", rdf is not None and 3 <= rdf.shape[1] <= 5)
if rdf is not None:
    print(rdf.head())''')

code(r'''# Work the dimuon invariant mass out again from the four-vectors in the ROOT
# file, and compare against the file's own M branch. You need both muons, so
# load their branches here -- the 3-5 limit above was for rdf, not for this.
m_root = tree["M"].array(library="np")
m_check = None   # TODO

check("matches the stored M branch", m_check is not None and np.allclose(m_check, m_root, atol=1e-3))''')

# ---------------------------------------------------------------- E7
md(r'''---
# E7 — pytest drill

*medium, 30 min*

`src/week01/fit.py` already holds the `fit_pi0_peak` worked example from README §8 (step 1 is
done). It returns a **dictionary**, so its results come out as `fit["mu"]`, `fit["mu_err"]`
and so on — `exercises.md` writes those as `fit.mu` and `fit.mu_err`.

In `../tests/test_fit.py` write three tests:

- `test_fit_recovers_mu` — `abs(fit["mu"] - 0.135) < 3 * fit["mu_err"]`
- `test_fit_positive_sigma` — `fit["sigma"] > 0`
- `test_empty_input_raises` — zero-length input raises `ValueError`

The third one fails against the code as written: add the guard to `fit_pi0_peak` after you
have watched the test fail.''')

code(r'''!{sys.executable} -m pytest -q ../tests/test_fit.py''')

# ---------------------------------------------------------------- E8
md(r'''---
# E8 — Tiny sklearn sanity check

*medium, 30 min*

1. Load iris; split 70/30 with `train_test_split(..., random_state=42, stratify=y)`.
2. Fit `LogisticRegression`.
3. Accuracy, balanced accuracy and a confusion matrix on the test set.
4. Plot the decision boundary in the (sepal length, petal length) projection.

**Hint:** for the boundary, fit on two features only and evaluate on a meshgrid; `plt.contourf`.''')

code(r'''from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train = None
X_test = None
y_train = None
y_test = None
# TODO: train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

check("70/30 split", X_train is not None and abs(len(X_train) / len(X) - 0.7) < 0.02)
check("stratified", y_train is not None
      and np.allclose(np.bincount(y_train) / len(y_train), np.bincount(y) / len(y), atol=0.02))''')

code(r'''clf = None      # TODO: fit LogisticRegression on the training set
pred = None     # TODO: predict the test set

acc = None      # TODO: accuracy_score
bal_acc = None  # TODO: balanced_accuracy_score
cm = None       # TODO: confusion_matrix

print("accuracy         ", acc)
print("balanced accuracy", bal_acc)
print("confusion matrix\n", cm)

check("model is fitted", clf is not None and hasattr(clf, "coef_"))
check("accuracy above 0.85", acc is not None and acc > 0.85)
check("confusion matrix is 3x3", cm is not None and cm.shape == (3, 3))
check("matrix totals the test set", cm is not None and cm.sum() == len(y_test))''')

md(r'''### E8.4 — Decision boundary

Fit again on two features only — sepal length (column 0) and petal length (column 2). The grid
is built for you; predict on it and fill in the regions.''')

code(r'''feat = [0, 2]
X2_train = X_train[:, feat]
X2_test = X_test[:, feat]

lo0 = X[:, feat[0]].min() - 0.5
hi0 = X[:, feat[0]].max() + 0.5
lo1 = X[:, feat[1]].min() - 0.5
hi1 = X[:, feat[1]].max() + 0.5
gx, gy = np.meshgrid(np.linspace(lo0, hi0, 300), np.linspace(lo1, hi1, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])

clf2 = None   # TODO: fit LogisticRegression on X2_train
zz = None     # TODO: clf2.predict(grid), reshaped to gx.shape

fig, ax = plt.subplots(figsize=(7, 5))
# TODO: ax.contourf(gx, gy, zz, ...) then scatter the test points coloured by y_test
ax.set_xlabel(iris.feature_names[feat[0]])
ax.set_ylabel(iris.feature_names[feat[1]])
plt.show()

check("model uses two features", clf2 is not None and clf2.coef_.shape[1] == 2)
check("grid predictions reshaped", zz is not None and zz.shape == gx.shape)''')

# ---------------------------------------------------------------- E9
md(r'''---
# E9 — Reproducibility drill

*medium → hard, 45 min · work in `../run.py`*

1. `run.py` generates a toy, fits it, and logs `(μ̂, σ̂, log-likelihood, git_sha, wall_clock)`
   to `results/run_<timestamp>.json`.
2. Run it twice with the same seed; the JSONs should be identical except `wall_clock`.
3. Run it twice with different seeds; only `wall_clock` and the data-dependent fields differ.
4. Make it a CLI with `argparse`: `--seed`, `--n-events`, `--output-dir`.

**Hint:** `subprocess.check_output(["git", "rev-parse", "HEAD"])` gets you the SHA.''')

code(r'''!{sys.executable} ../run.py --seed 0 --n-events 50000 --output-dir ../results
!{sys.executable} ../run.py --seed 0 --n-events 50000 --output-dir ../results
!{sys.executable} ../run.py --seed 1 --n-events 50000 --output-dir ../results''')

code(r'''import glob
import json

files = sorted(glob.glob("../results/run_*.json"))
records = []
for name in files[-3:]:
    with open(name) as handle:
        records.append(json.load(handle))

print(len(files), "run files; comparing the last three")


def stable(record):
    """The record without the fields that are allowed to change between runs."""
    out = {}
    for key in record:
        if key != "wall_clock" and key != "timestamp":
            out[key] = record[key]
    return out


needed = ["mu_hat", "sigma_hat", "log_likelihood", "git_sha", "wall_clock"]
has_all = len(records) == 3
for key in needed:
    if len(records) < 3 or key not in records[-1]:
        has_all = False

check("logs every required field", has_all)
check("same seed -> identical record", len(records) == 3 and stable(records[0]) == stable(records[1]))
check("different seed -> different record", len(records) == 3
      and stable(records[1]) != stable(records[2]))
check("git_sha recorded", len(records) == 3 and len(str(records[-1].get("git_sha"))) >= 7)''')

# ---------------------------------------------------------------- E10
md(r'''---
# E10 — Stretch: vectorize invariant mass on 10⁶ events

*hard, 90 min*

1. Naive Python-loop pairwise `m_inv`. Measure the wall time.
2. NumPy-vectorized version. Measure the wall time.
3. The ratio should be ≥ 200×.
4. Plot the per-event timing histograms.

**Accept when:** numpy is ≥ 200× faster than the Python loop.

*Note: an honest loop on this machine lands around 70×; the ratio is a property of the CPU and
of how the loop was written. Both the 200× criterion and a ≥50× floor are checked below.*''')

code(r'''N = 1000000
rng10 = np.random.default_rng(42)
p1 = rng10.standard_normal((N, 4)) + np.array([5.0, 0, 0, 0])   # (E, px, py, pz)
p2 = rng10.standard_normal((N, 4)) + np.array([5.0, 0, 0, 0])


def m_inv_loop(a, b):
    """One Python-level step per event."""
    out = []
    # TODO
    return out


def m_inv_vectorized(a, b):
    """No Python-level steps at all."""
    # TODO


small_loop = m_inv_loop(p1[:1000], p2[:1000])
small_vec = m_inv_vectorized(p1[:1000], p2[:1000])

check("both return 1000 masses", len(small_loop) == 1000 and small_vec is not None
      and len(small_vec) == 1000)
check("the two agree", small_vec is not None and len(small_loop) == 1000
      and np.allclose(small_loop, small_vec))''')

code(r'''import time

t0 = time.perf_counter()
m_inv_loop(p1, p2)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
m_inv_vectorized(p1, p2)
t_vec = time.perf_counter() - t0

print(f"loop:       {t_loop:8.3f} s  ({t_loop / N * 1e9:7.1f} ns/event)")
print(f"vectorized: {t_vec:8.3f} s  ({t_vec / N * 1e9:7.1f} ns/event)")
print(f"speedup:    {t_loop / t_vec:8.1f}x")

check("vectorization is real (>=50x)", t_loop / t_vec >= 50)
check("exercises.md criterion (>=200x)", t_loop / t_vec >= 200)''')

md(r'''### E10.4 — Timing histograms

One event takes tens of nanoseconds, which is below the clock's resolution, so time **chunks**
of events and histogram the per-event time each chunk implies.''')

code(r'''CHUNK = 5000
N_CHUNKS = 200

loop_times = []
vec_times = []
for c in range(N_CHUNKS):
    a = p1[c * CHUNK:(c + 1) * CHUNK]
    b = p2[c * CHUNK:(c + 1) * CHUNK]
    # TODO: time each version on this chunk, append (elapsed / CHUNK) to the lists

fig, ax = plt.subplots(figsize=(7, 4))
# TODO: two histograms on a log x axis (they are orders of magnitude apart)
ax.set_xlabel("time per event [s]")
plt.show()

check("timed every chunk", len(loop_times) == N_CHUNKS and len(vec_times) == N_CHUNKS)
check("vectorized wins per chunk too", len(vec_times) == N_CHUNKS
      and np.median(loop_times) > np.median(vec_times))''')

# ---------------------------------------------------------------- wrap up
md(r'''---
## Wrap-up

```bash
uv run pytest -q      # E2 + E7 + the provided loader tests
uv run ruff check .
```

Then commit — `exercises.md` asks you to keep these as reference for later weeks.''')


# ---------------------------------------------------------------- write it out
def to_cell(kind, text, number):
    """Turn one (kind, text) pair into the dict nbformat wants."""
    lines = text.split("\n")
    source = []
    for line in lines[:-1]:
        source.append(line + "\n")
    source.append(lines[-1])

    cell = {"cell_type": kind, "id": f"w01e-{number:03d}", "metadata": {}, "source": source}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


cells = []
for number in range(len(CELLS)):
    kind = CELLS[number][0]
    text = CELLS[number][1]
    cells.append(to_cell(kind, text, number))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (week01)", "language": "python",
                       "name": "week01"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Week01_Exercises.ipynb")
with open(out_path, "w") as out_file:
    json.dump(notebook, out_file, indent=1)
print(f"wrote {out_path}  ({len(CELLS)} cells)")
