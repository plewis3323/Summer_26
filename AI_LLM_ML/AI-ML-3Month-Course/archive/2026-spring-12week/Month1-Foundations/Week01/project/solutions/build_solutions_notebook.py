#!/usr/bin/env python3
"""Build Week01_Solutions.ipynb. Re-run after editing CELLS:

    python3 solutions/build_solutions_notebook.py

Mirrors notebooks/build_exercises_notebook.py cell for cell, with every TODO
filled in. Style rules: ../../../NOTEBOOK_RULES.md
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


md(r'''# Week 01 — Solutions E1–E10

The same notebook as `Week01_Exercises.ipynb`, with every `# TODO` filled in. Read it after
you have been stuck, not before.

E2, E7 and E9 are graded in `.py` files, so their answers appear here as file listings —
this notebook never writes to `src/`, `tests/` or `run.py`. To put them in your tree:

```bash
python solutions/apply_solutions.py --diff     # see what would change
python solutions/apply_solutions.py --apply    # overwrite, with backups
python solutions/apply_solutions.py --restore  # undo
```

For the reasoning behind each answer, read `solutions/TEACHER_EXPLAINER.md` alongside this.''')

md(r'''## Setup''')

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
# E1 — NumPy shape calisthenics''')

md(r'''### E1.1 — z-score

`keepdims=True` gives `(1, 4)`, which broadcasts against `(1000, 4)` because it says so.
Without it you get `(4,)`, which works only by NumPy's right-alignment rule.''')

code(r'''rng = np.random.default_rng(0)
x = rng.standard_normal((1000, 4))

mean = x.mean(axis=0, keepdims=True)
std = x.std(axis=0, keepdims=True)
z = (x - mean) / std

check("mean has shape (1, 4)", mean is not None and mean.shape == (1, 4))
check("std has shape (1, 4)", std is not None and std.shape == (1, 4))
check("z column means are ~0", z is not None and np.allclose(z.mean(axis=0), 0))
check("z column stds are ~1", z is not None and np.allclose(z.std(axis=0), 1))''')

md(r'''### E1.2 — (pt, eta, phi) → (px, py)''')

code(r'''A = np.column_stack([
    rng.uniform(0.5, 50, 100),        # pt  [GeV]
    rng.uniform(-2.5, 2.5, 100),      # eta
    rng.uniform(-np.pi, np.pi, 100),  # phi
])
pt = A[:, 0]
eta = A[:, 1]
phi = A[:, 2]

px = pt * np.cos(phi)
py = pt * np.sin(phi)

check("px has shape (100,)", px is not None and px.shape == (100,))
check("py has shape (100,)", py is not None and py.shape == (100,))
check("px^2 + py^2 == pt^2", px is not None and np.allclose(px**2 + py**2, pt**2))''')

md(r'''### E1.3 — Outer product, one line

`np.outer(u, v)[i, j] == u[i] * v[j]`, which is exactly the $\sin(i)\cos(j)$ asked for.''')

code(r'''i = np.arange(1000)
j = np.arange(1000)

M = np.outer(np.sin(i), np.cos(j))

check("shape is (1000, 1000)", M is not None and M.shape == (1000, 1000))
check("M[3, 7] is sin(3) * cos(7)", M is not None and np.isclose(M[3, 7], np.sin(3) * np.cos(7)))''')

md(r'''### E1.4 — Frobenius norm by hand''')

code(r'''B = rng.standard_normal((5, 5))

frob = np.sqrt(np.sum(B**2))

check("agrees with np.linalg.norm", frob is not None and np.isclose(frob, np.linalg.norm(B)))''')

# ---------------------------------------------------------------- E2
md(r'''---
# E2 — Broadcasting trap

`src/week01/center.py`:

```python
import numpy as np


def center_rows(x):
    """Subtract each row's mean, so every row of the answer sums to about 0."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=1, keepdims=True)


def center_cols(x):
    """Subtract each column's mean, so every column sums to about 0."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)
```

`tests/test_center.py`:

```python
import numpy as np
import pytest

from week01.center import center_cols, center_rows

SHAPES = [(3, 5), (5, 3), (1, 7)]


def make(shape):
    rng = np.random.default_rng(shape[0] * 100 + shape[1])
    return rng.standard_normal(shape) * 10 + 3


def test_center_rows():
    for shape in SHAPES:
        out = center_rows(make(shape))
        np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)


def test_center_cols():
    for shape in SHAPES:
        out = center_cols(make(shape))
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)


def test_shape_is_preserved():
    for shape in SHAPES:
        x = make(shape)
        assert center_rows(x).shape == shape
        assert center_cols(x).shape == shape


def test_does_not_mutate_input():
    x = make((3, 5))
    before = x.copy()
    center_rows(x)
    center_cols(x)
    np.testing.assert_array_equal(x, before)


def test_dropping_keepdims_is_a_bug():
    """The bug this file exists to catch, written out as a test."""
    x = make((3, 5))
    with pytest.raises(ValueError):
        x - x.mean(axis=1)                    # not square: NumPy catches it

    square = make((5, 5))
    wrong = square - square.mean(axis=1)      # square: NumPy does not
    assert wrong.shape == square.shape
    assert not np.allclose(wrong.mean(axis=1), 0, atol=1e-12)
```

The shapes are non-square on purpose: on a `(5, 5)` array the missing `keepdims=True` is a
silent transpose — legal shape, finite numbers, wrong answer.''')

code(r'''# The same two functions, here, so the rest of the notebook can use them.
def center_rows(x):
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=1, keepdims=True)


def center_cols(x):
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


for shape in [(3, 5), (5, 3), (1, 7)]:
    a = np.random.default_rng(0).standard_normal(shape)
    check(f"rows centered {shape}", np.allclose(center_rows(a).mean(axis=1), 0))
    check(f"cols centered {shape}", np.allclose(center_cols(a).mean(axis=0), 0))''')

# ---------------------------------------------------------------- E3
md(r'''---
# E3 — pandas: read, filter, groupby''')

code(r'''from week01.data import load_dimuon

df = load_dimuon()   # about 72 MB on the first run, then cached in ../data/
print(df.shape)
df.head()''')

md(r'''### E3.2 — Invariant mass

$M^2$ can come out very slightly negative for a low-mass pair through rounding, and
`np.sqrt` of a negative number is `nan` — so clip at zero first.''')

code(r'''E = df["E1"] + df["E2"]
px = df["px1"] + df["px2"]
py = df["py1"] + df["py2"]
pz = df["pz1"] + df["pz2"]

m2 = E**2 - (px**2 + py**2 + pz**2)
df["m_inv"] = np.sqrt(np.clip(m2, 0, None))

check("column exists", "m_inv" in df.columns)
check("no NaNs", "m_inv" in df.columns and df["m_inv"].notna().all())

if "m_inv" in df.columns:
    diff = (df["m_inv"] - df["M"]).abs()
    print(f"median |m_inv - M| = {diff.median():.2e} GeV")
    check("agrees with CMS for a typical event", diff.median() < 1e-3)''')

md(r'''The worst events differ by up to about 1 GeV, and that is not your algebra: the CSV stores
energies to six significant figures, and $M^2$ subtracts two numbers near $2.6\times10^6$ to
get 61. Read the E6 ROOT file instead if you need that precision.''')

md(r'''### E3.3 — J/ψ window''')

code(r'''jpsi = df.query("2 < m_inv < 4")
n_jpsi = len(jpsi)

check("selection is a DataFrame", isinstance(jpsi, pd.DataFrame))
check("inside the window", jpsi is not None and len(jpsi) > 0 and jpsi["m_inv"].between(2, 4).all())
check("count matches the selection", n_jpsi is not None and n_jpsi == len(jpsi))
print("J/psi-region events:", n_jpsi)''')

md(r'''### E3.4 — Per-run mean $p_T$''')

code(r'''per_run = df.groupby("Run")["pt1"].mean()

check("one row per run", per_run is not None and len(per_run) == df["Run"].nunique())
check("indexed by Run", per_run is not None and per_run.index.name == "Run")
if per_run is not None:
    print(per_run.head())''')

md(r'''### E3.5 — The spectrum''')

code(r'''bins = np.logspace(np.log10(0.2), np.log10(120), 300)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["m_inv"], bins=bins, histtype="step", lw=1.0, color="0.25")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.2, 120)

top = ax.get_ylim()[1]
ax.set_ylim(top=top * 4)

resonances = [(0.78, r"$\rho/\omega$"), (1.02, r"$\phi$"), (3.10, r"$J/\psi$"),
              (9.46, r"$\Upsilon$"), (91.2, r"$Z$")]
for m, name in resonances:
    ax.annotate(name, xy=(m, top * 1.3), ha="center", fontsize=9, color="0.35")

ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
ax.set_ylabel("events / bin")
plt.show()''')

md(r'''The resonance masses are the calibration: if the peaks are not at 3.1, 9.5 and 91 GeV,
`m_inv` is wrong.''')

# ---------------------------------------------------------------- E4
md(r'''---
# E4 — matplotlib: publication figure''')

md(r'''### E4.1 — Fit the J/ψ peak

`p0` matters: start the peak at the PDG mass, the norm at the tallest bin above the median, and
the background at the median. `sigma=sqrt(counts + 1)` is the Poisson error per bin, with the
`+1` so an empty bin gets weight 1 instead of infinite weight.''')

code(r'''from scipy.optimize import curve_fit

from week01.fit import signal_plus_bg

window = (2.6, 3.6)
n_bins = 60

mass = df["m_inv"].to_numpy()
sel = mass[(mass > window[0]) & (mass < window[1])]
counts, edges = np.histogram(sel, bins=n_bins, range=window)
centers = 0.5 * (edges[:-1] + edges[1:])

base = float(np.median(counts))
p0 = [3.097, 0.04, counts.max() - base, base, 0.0]
popt, pcov = curve_fit(signal_plus_bg, centers, counts, p0=p0,
                       sigma=np.sqrt(counts + 1), absolute_sigma=True)
popt[1] = abs(popt[1])          # the model only sees sigma**2, so the sign is arbitrary

fitted = signal_plus_bg(centers, popt[0], popt[1], popt[2], popt[3], popt[4])
resid = (counts - fitted) / np.sqrt(counts + 1)

print(f"mu = {popt[0]:.4f} GeV, sigma = {popt[1]:.4f} GeV")
print(f"chi2/ndof = {np.sum(resid**2) / (n_bins - len(popt)):.1f}")

check("five fitted parameters", popt is not None and len(popt) == 5)
check("peak found near 3.1 GeV", popt is not None and abs(popt[0] - 3.097) < 0.05)
check("one residual per bin", resid is not None and len(resid) == n_bins)''')

md(r'''χ²/ndof lands near 10: one gaussian does not describe the J/ψ, because CMS's muon
resolution changes with η and a sum of gaussians of different widths is not a gaussian. E4 asks
for linear-plus-gaussian, so that is what is fitted — and the residual panel is how you find out.''')

md(r'''### E4.2 — The figure

The fit is in counts per 16.7 MeV bin; the log bins drawn on the top panel are about 4× wider,
so the curve is rescaled by the width ratio before it goes on.''')

code(r'''fig, (ax0, ax1) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6), constrained_layout=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax0.hist(mass, bins=bins, histtype="step", lw=1.0, color="0.25",
         label=r"CMS DoubleMu 2011, $\mu^{+}\mu^{-}$")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlim(0.2, 120)
ax0.axvspan(window[0], window[1], color="tab:orange", alpha=0.12, lw=0, zorder=0)

inside = (bins[:-1] >= window[0]) & (bins[1:] <= window[1])
lo = bins[:-1][inside]
hi = bins[1:][inside]
fit_w = (window[1] - window[0]) / n_bins
mids = 0.5 * (lo + hi)
curve = signal_plus_bg(mids, popt[0], popt[1], popt[2], popt[3], popt[4]) * (hi - lo) / fit_w
ax0.stairs(curve, np.append(lo, hi[-1]), baseline=None, color="tab:red", lw=1.6, label="fit")
ax0.legend(loc="upper right", fontsize=8, frameon=False)

ax1.axhspan(-1, 1, color="0.85", lw=0, zorder=0)
ax1.axhline(0.0, color="k", lw=0.8)
ax1.plot(centers, resid, ls="none", marker="o", ms=3, color="tab:red")
ax1.set_ylim(-5.5, 5.5)

ax0.set_ylabel("events / bin")
ax1.set_ylabel(r"$(N - f)/\sigma$")
ax1.set_xlabel(r"$m_{\mu^{+}\mu^{-}}$ [GeV]")

fig.savefig("../results/dimuon.pdf", dpi=300, bbox_inches="tight")
plt.show()''')

code(r'''import os

check("two axes", len(fig.axes) == 2)
check("x axis is shared", fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1]))
check("top panel y is log", fig.axes[0].get_yscale() == "log")
check("LaTeX in the x label", "$" in fig.axes[1].get_xlabel())
check("results/dimuon.pdf written", os.path.exists("../results/dimuon.pdf"))''')

# ---------------------------------------------------------------- E5
md(r'''---
# E5 — Synthetic π⁰ signal + polynomial background toy''')

md(r'''### E5.1–3 — One toy, one fit

`absolute_sigma=True` because Poisson bin errors are real errors, not relative weights: without
it `curve_fit` rescales the covariance by the reduced χ², which answers a different question and
makes the pull width in E5.4 meaningless.''')

code(r'''from week01.data import make_pi0_toy

MU_TRUE = 0.135
SIGMA_TRUE = 0.008
BIN_EDGES = np.linspace(0.05, 0.25, 80)

m_gg = make_pi0_toy(n=50000, signal_frac=0.20, seed=0)


def fit_one(sample):
    """Return mu_hat, mu_err, sigma_hat, sigma_err for one toy."""
    n, _edges = np.histogram(sample, bins=BIN_EDGES)
    mids = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    p0 = [MU_TRUE, 0.010, n.max(), 1.0, 0.0]
    par, cov = curve_fit(signal_plus_bg, mids, n, p0=p0,
                         sigma=np.sqrt(n + 1), absolute_sigma=True)
    err = np.sqrt(np.diag(cov))
    return par[0], err[0], abs(par[1]), err[1]


mu_hat, mu_err, sigma_hat, sigma_err = fit_one(m_gg)
print("mu    =", mu_hat, "+/-", mu_err)
print("sigma =", sigma_hat, "+/-", sigma_err)

check("mu recovered", mu_hat is not None and abs(mu_hat - MU_TRUE) < 3 * mu_err)
check("sigma recovered", sigma_hat is not None and abs(sigma_hat - SIGMA_TRUE) < 0.002)
check("errors are positive", mu_err is not None and mu_err > 0 and sigma_err > 0)''')

code(r'''bin_centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
n, _edges = np.histogram(m_gg, bins=BIN_EDGES)

p0 = [MU_TRUE, 0.010, n.max(), 1.0, 0.0]
par, cov = curve_fit(signal_plus_bg, bin_centers, n, p0=p0,
                     sigma=np.sqrt(n + 1), absolute_sigma=True)
curve = signal_plus_bg(bin_centers, par[0], par[1], par[2], par[3], par[4])

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(bin_centers, n, yerr=np.sqrt(n + 1), ls="none", marker="o", ms=3,
            color="0.3", label="toy data")
ax.plot(bin_centers, curve, color="tab:red", label="fit")
ax.plot(bin_centers, par[3] + par[4] * bin_centers, ls="--", color="tab:blue",
        label="background only")
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
ax.set_ylabel("events / bin")
plt.show()''')

md(r'''### E5.4 — Pull distribution

A pull that is too wide means the errors are under-estimated, too narrow means over-estimated,
and a shifted mean means the fit is biased. At 100 toys the criteria on the exercise page sit
inside their own statistical error, so a FAIL here is usually noise — compare each number
against its own error, printed below.''')

code(r'''N_TOYS = 100
pulls = []

for t in range(N_TOYS):
    toy = make_pi0_toy(n=50000, signal_frac=0.20, seed=t)
    mu_t, err_t, sigma_t, sigma_err_t = fit_one(toy)
    pulls.append((mu_t - MU_TRUE) / err_t)

pulls = np.asarray(pulls, dtype=float)
pull_mean = pulls.mean()
pull_width = pulls.std(ddof=1)

se_mean = 1 / np.sqrt(N_TOYS)
se_width = 1 / np.sqrt(2 * (N_TOYS - 1))
if pull_mean is not None:
    print(f"pull mean  = {pull_mean:+.3f}  (statistical error {se_mean:.3f})")
    print(f"pull width =  {pull_width:.3f}  (statistical error {se_width:.3f})")

check("ran every toy", len(pulls) == N_TOYS)
check("|mean| < 0.05", pull_mean is not None and abs(pull_mean) < 0.05)
check("|width - 1| < 0.1", pull_width is not None and abs(pull_width - 1) < 0.1)
check("mean consistent with 0 within its error", abs(pull_mean) < 3 * se_mean)
check("width consistent with 1 within its error", abs(pull_width - 1) < 3 * se_width)''')

code(r'''fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(pulls, bins=20, range=(-4, 4), density=True, histtype="step", lw=1.2, color="0.25")
g = np.linspace(-4, 4, 200)
ax.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), color="tab:red", label=r"$N(0,1)$")
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel(r"$(\hat\mu - \mu_{\rm true})/\hat\sigma_{\hat\mu}$")
plt.show()''')

# ---------------------------------------------------------------- E6
md(r'''---
# E6 — uproot: read a ROOT file into pandas''')

code(r'''import uproot

from week01.data import zmumu_root_path

path = zmumu_root_path()
f = uproot.open(path)

keys = f.keys()
tree = f["events"]          # the file's only TTree
branches = tree.keys()

print("keys:", keys)
print("branches:", branches)

check("listed keys", keys is not None and len(keys) > 0)
check("got a TTree", tree is not None and hasattr(tree, "arrays"))
check("listed branches", branches is not None and len(branches) > 5)''')

code(r'''want = ["E1", "px1", "py1", "pz1"]
rdf = tree.arrays(want, library="pd")

check("is a DataFrame", isinstance(rdf, pd.DataFrame))
check("3-5 branches", rdf is not None and 3 <= rdf.shape[1] <= 5)
if rdf is not None:
    print(rdf.head())''')

md(r'''The cross-check needs both muons, so load the second one's four-vector too — 3–5 branches
was the requirement for `rdf`, not a budget for the whole exercise.''')

code(r'''m_root = tree["M"].array(library="np")

both = tree.arrays(["E1", "px1", "py1", "pz1", "E2", "px2", "py2", "pz2"], library="pd")
E = both["E1"] + both["E2"]
px = both["px1"] + both["px2"]
py = both["py1"] + both["py2"]
pz = both["pz1"] + both["pz2"]
m_check = np.sqrt(np.clip(E**2 - (px**2 + py**2 + pz**2), 0, None))

print(f"max |m_check - M| = {np.abs(m_check - m_root).max():.2e} GeV")

check("matches the stored M branch", m_check is not None and np.allclose(m_check, m_root, atol=1e-3))''')

md(r'''This agrees far more tightly than the E3 CSV did — same events, same formula, but full
double precision instead of six significant figures round-tripped through text.''')

# ---------------------------------------------------------------- E7
md(r'''---
# E7 — pytest drill

`fit_pi0_peak` hands back a dictionary, so the results are `fit["mu"]`, `fit["mu_err"]` and so
on. (`exercises.md` writes those as `fit.mu` and `fit.mu_err`.)

`tests/test_fit.py`:

```python
import numpy as np
import pytest

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak

BIN_EDGES = np.linspace(0.05, 0.25, 80)
MU_TRUE = 0.135


def test_fit_recovers_mu():
    toy = make_pi0_toy(n=50000, signal_frac=0.20, seed=0)
    fit = fit_pi0_peak(toy, BIN_EDGES)
    assert abs(fit["mu"] - MU_TRUE) < 3 * fit["mu_err"]


def test_fit_positive_sigma():
    toy = make_pi0_toy(n=50000, signal_frac=0.20, seed=1)
    fit = fit_pi0_peak(toy, BIN_EDGES)
    assert fit["sigma"] > 0


def test_empty_input_raises():
    with pytest.raises(ValueError):
        fit_pi0_peak(np.array([]), BIN_EDGES)
```

The guard `test_empty_input_raises` is asking for, at the top of `fit_pi0_peak`:

```python
    if m_gg.size == 0:
        raise ValueError("fit_pi0_peak: empty sample, there is nothing to fit")
```

`np.histogram([])` does not raise — it hands back a row of zeros — so without the guard
`curve_fit` gets a nicely shaped problem with no information in it and fails somewhere deep
inside MINPACK. Fail at the door instead.

Two notes on the other two tests. `abs(fit["mu"] - 0.135) < 3 * fit["mu_err"]` is a statistical
claim: with a correct fit it still fails about 0.3% of the time, which is why every test pins
its seed. And `fit["sigma"] > 0` is not trivial — the model only ever uses `sigma**2`, so
`-sigma` fits exactly as well, and which one `curve_fit` returns depends on the data; the fix is
`abs(sigma)` inside the fit.''')

# ---------------------------------------------------------------- E8
md(r'''---
# E8 — Tiny sklearn sanity check''')

code(r'''from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

check("70/30 split", X_train is not None and abs(len(X_train) / len(X) - 0.7) < 0.02)
check("stratified", y_train is not None
      and np.allclose(np.bincount(y_train) / len(y_train), np.bincount(y) / len(y), atol=0.02))''')

code(r'''clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
bal_acc = balanced_accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

print("accuracy         ", acc)
print("balanced accuracy", bal_acc)
print("confusion matrix\n", cm)

check("model is fitted", clf is not None and hasattr(clf, "coef_"))
check("accuracy above 0.85", acc is not None and acc > 0.85)
check("confusion matrix is 3x3", cm is not None and cm.shape == (3, 3))
check("matrix totals the test set", cm is not None and cm.sum() == len(y_test))''')

md(r'''On stratified iris, accuracy and balanced accuracy come out nearly identical — three
classes, 50 each. That is the point of printing them together: you learn the difference on a
dataset where it does not matter, so that on a 99%-background sample you already distrust plain
accuracy.''')

md(r'''### E8.4 — Decision boundary''')

code(r'''feat = [0, 2]
X2_train = X_train[:, feat]
X2_test = X_test[:, feat]

lo0 = X[:, feat[0]].min() - 0.5
hi0 = X[:, feat[0]].max() + 0.5
lo1 = X[:, feat[1]].min() - 0.5
hi1 = X[:, feat[1]].max() + 0.5
gx, gy = np.meshgrid(np.linspace(lo0, hi0, 300), np.linspace(lo1, hi1, 300))
grid = np.column_stack([gx.ravel(), gy.ravel()])

clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X2_train, y_train)
zz = clf2.predict(grid).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#d7e7f7", "#dcf0d8", "#f7dcdc"])
ax.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test, cmap="viridis", edgecolor="k", s=35)
ax.set_xlabel(iris.feature_names[feat[0]])
ax.set_ylabel(iris.feature_names[feat[1]])
plt.show()

check("model uses two features", clf2 is not None and clf2.coef_.shape[1] == 2)
check("grid predictions reshaped", zz is not None and zz.shape == gx.shape)''')

md(r'''The model has to live in the space you draw: a boundary taken from the 4-feature model and
projected into 2D would be a picture of something that is not there.''')

# ---------------------------------------------------------------- E9
md(r'''---
# E9 — Reproducibility drill

`run.py` (full listing in `solutions/files/run.py`):

```python
import argparse
import json
import os
import subprocess
import time

import numpy as np
from scipy.stats import poisson

from week01.data import make_pi0_toy
from week01.fit import fit_pi0_peak, signal_plus_bg


def git_sha():
    """The current commit, '-dirty' if the tree is edited, 'unknown' outside a repo."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        changes = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        return "unknown"
    if changes.strip():
        return sha.strip() + "-dirty"
    return sha.strip()


def log_likelihood(counts, expected):
    """Poisson log-likelihood: sum of k*log(lam) - lam - log(k!)."""
    expected = np.clip(expected, 1e-12, None)   # a straight-line background can go negative
    return float(np.sum(poisson.logpmf(counts, expected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-events", type=int, default=50000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    start = time.perf_counter()
    bin_edges = np.linspace(0.05, 0.25, 80)
    m_gg = make_pi0_toy(n=args.n_events, signal_frac=0.20, seed=args.seed)
    fit = fit_pi0_peak(m_gg, bin_edges)

    counts, _edges = np.histogram(m_gg, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    popt = fit["popt"]
    expected = signal_plus_bg(centers, popt[0], popt[1], popt[2], popt[3], popt[4])

    record = {
        "mu_hat": float(fit["mu"]),
        "mu_err": float(fit["mu_err"]),
        "sigma_hat": float(fit["sigma"]),
        "sigma_err": float(fit["sigma_err"]),
        "log_likelihood": log_likelihood(counts, expected),
        "n_events": int(args.n_events),
        "seed": int(args.seed),
        "git_sha": git_sha(),
        "wall_clock": time.perf_counter() - start,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{int(time.time() % 1 * 1e6):06d}"
    out_path = os.path.join(args.output_dir, "run_" + stamp + ".json")
    with open(out_path, "w") as out_file:
        json.dump(record, out_file, indent=2, sort_keys=True)
        out_file.write("\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
```

Two rules are what make steps 2 and 3 come out right.

**Every random draw traces back to `--seed`.** One generator, made from the seed, passed down.
A single stray `np.random.uniform()` — in a starting point, in a bootstrap — breaks the
guarantee in the worst possible way: results that are *nearly* identical, so you do not notice
for a month.

**A SHA of a commit you have since edited is worse than no SHA**, because it claims a
reproducibility you do not actually have. Hence `-dirty`.

`fit["popt"]` is why the fit hands back the fitted parameters at all: without them you cannot
evaluate the model, and without the model there is no likelihood to log.''')

# ---------------------------------------------------------------- E10
md(r'''---
# E10 — Stretch: vectorize invariant mass on 10⁶ events''')

code(r'''import math

N = 1000000
rng10 = np.random.default_rng(42)
p1 = rng10.standard_normal((N, 4)) + np.array([5.0, 0, 0, 0])   # (E, px, py, pz)
p2 = rng10.standard_normal((N, 4)) + np.array([5.0, 0, 0, 0])


def m_inv_loop(a, b):
    """One Python-level step per event."""
    out = []
    for k in range(len(a)):
        e = a[k][0] + b[k][0]
        px = a[k][1] + b[k][1]
        py = a[k][2] + b[k][2]
        pz = a[k][3] + b[k][3]
        m2 = e * e - px * px - py * py - pz * pz
        out.append(math.sqrt(max(m2, 0.0)))
    return out


def m_inv_vectorized(a, b):
    """No Python-level steps at all."""
    s = a + b
    m2 = s[:, 0]**2 - s[:, 1]**2 - s[:, 2]**2 - s[:, 3]**2
    return np.sqrt(np.clip(m2, 0, None))


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

md(r'''The 200× criterion does not pass here, and the reason is worth knowing: the ratio measures
CPython's per-iteration overhead against memory bandwidth, so it is a property of the machine and
of how the loop was written, not of your vectorization. Measured on this box: about 1.5 s for the
loop against about 0.02 s vectorized, so roughly 70×. Iterating Python lists instead of ndarrays
is several times *faster* per iteration (no `np.float64` boxing), which pushes the ratio down
further — the "naive" baseline is a range, not a number. A ratio near 1 would mean you did not
vectorize; near 5–10 means a Python loop is still wrapped around the array ops.''')

md(r'''### E10.4 — Timing histograms''')

code(r'''CHUNK = 5000
N_CHUNKS = 200

loop_times = []
vec_times = []
for c in range(N_CHUNKS):
    a = p1[c * CHUNK:(c + 1) * CHUNK]
    b = p2[c * CHUNK:(c + 1) * CHUNK]

    t0 = time.perf_counter()
    m_inv_loop(a, b)
    loop_times.append((time.perf_counter() - t0) / CHUNK)

    t0 = time.perf_counter()
    m_inv_vectorized(a, b)
    vec_times.append((time.perf_counter() - t0) / CHUNK)

everything = np.concatenate([np.array(loop_times), np.array(vec_times)])
tbins = np.logspace(np.log10(everything.min()), np.log10(everything.max()), 40)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(vec_times, bins=tbins, histtype="step", lw=1.2, label="vectorized")
ax.hist(loop_times, bins=tbins, histtype="step", lw=1.2, label="python loop")
ax.set_xscale("log")
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel("time per event [s]")
plt.show()

check("timed every chunk", len(loop_times) == N_CHUNKS and len(vec_times) == N_CHUNKS)
check("vectorized wins per chunk too", len(vec_times) == N_CHUNKS
      and np.median(loop_times) > np.median(vec_times))''')

md(r'''The spread in each histogram is cache behaviour, allocator noise and the OS scheduler —
which is why a single timing is never a measurement.''')

# ---------------------------------------------------------------- wrap up
md(r'''---
## Wrap-up

Everything above prints PASS except E10's ≥200×, which is not reachable with an honest loop on
this machine (about 70× measured). E5's two page criteria pass here but sit inside their own
statistical error at 100 toys, so they can print FAIL on a different set of toys without
anything being wrong. Both are argued where they appear.

For the reasoning in full, `solutions/TEACHER_EXPLAINER.md`.''')


# ---------------------------------------------------------------- write it out
def to_cell(kind, text, number):
    """Turn one (kind, text) pair into the dict nbformat wants."""
    lines = text.split("\n")
    source = []
    for line in lines[:-1]:
        source.append(line + "\n")
    source.append(lines[-1])

    cell = {"cell_type": kind, "id": f"w01s-{number:03d}", "metadata": {}, "source": source}
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

out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks", "Week01_Solutions.ipynb",
)
with open(out_path, "w") as out_file:
    json.dump(notebook, out_file, indent=1)
print(f"wrote {out_path}  ({len(CELLS)} cells)")
