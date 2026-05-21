# Week 01 — Python Scientific Stack, the ML Mindset, and Reproducibility

## Learning objectives

By the end of this week you will:

1. Move fluidly between `numpy` arrays, `pandas` DataFrames, and `matplotlib` figures without looking things up every 30 seconds.
2. Understand what "machine learning" means operationally, and how it differs from maximum-likelihood fitting in `RooFit` — which is what most physicists think ML "already is."
3. Know how to set up a reproducible Python project with `uv`, `ruff`, `pytest`, and git, at the level you could explain it to a next-year PhD student.
4. Have a small, working pipeline that ingests data, fits a π⁰ invariant-mass peak, produces a figure, and is tested.

No models yet. No neural networks. The point of Week 01 is tooling fluency and an honest mental model. If you skip this, every subsequent week costs you 30% more time.

## 1. The ML mindset (how it differs from what you already do)

You already know how to do this:

> Given a model p(x | θ) with physical parameters θ, and data {x_i}, find θ̂ that maximizes the likelihood L(θ) = ∏ p(x_i | θ).

That's maximum-likelihood fitting. It's the heart of every RooFit analysis you've run. You have strong priors on the functional form (Crystal Ball, bifurcated Gaussian, Novosibirsk, etc.) and you trust your parameters because they correspond to something physical (mean, width, yield).

Machine learning is the same machinery — find parameters that minimize a loss on data — with three structural differences:

**Difference 1: the model is overparameterized, and you don't trust any individual parameter.**
A modern neural network has 10⁶ to 10¹² parameters. None of them correspond to anything physical. You care about the *function* they collectively compute, not about any single weight. Don't ask "what does weight w_{324,17} mean." It doesn't mean anything.

**Difference 2: the objective is not the likelihood — it's a proxy for generalization.**
In HEP you evaluate a fit by how well it matches *the data you fit it to* (with caveats like χ² / NDF). In ML you evaluate by how well the model does on *held-out data it has never seen*. The fit of the training set is always better than generalization performance, and when it's a lot better, you've "overfit." This is the central tension of ML and it's why every proper ML workflow has at least a train/val/test split.

**Difference 3: features are often learned, not designed.**
In a RooFit fit you've already designed features: you selected clusters, you cut on shower shape, you built invariant mass. In ML — especially deep learning — the model learns the features. That's powerful and it's also why you can't cleanly interpret what the model is doing. Explainability is an active research area, not a solved problem.

**Where the skills transfer:** probability, statistics, χ² minimization, covariance propagation, bias estimation, systematic uncertainties, the idea of a "null test." Your habits for validating a result are *exactly* what ML engineers call "good MLOps." You're more prepared than average.

**Where you'll be tripped up:** black-box objectives (accuracy, AUC, BLEU) that feel disconnected from anything physical; the idea that a model is "done" when the validation loss plateaus rather than when χ²/NDF ≈ 1; the fact that "more data" often beats "smarter model" in ways that sting.

## 2. Taxonomy of ML tasks

For the rest of this course, when someone says "ML task" they mean one of:

- **Supervised learning.** You have (x, y) pairs and want to learn f: x → y. Regression when y is continuous (energy scale calibration). Classification when y is discrete (π⁰ vs photon). This is 90% of what you'll do.
- **Unsupervised learning.** You only have x and you want to find structure: clustering, dimensionality reduction, density estimation. PCA on your calibration detector, t-SNE visualizations of your feature space.
- **Self-supervised learning.** You synthesize (x, y) pairs from unlabeled data. "Given the first 500 tokens, predict the next one." This is how language models are pretrained.
- **Reinforcement learning.** An agent interacts with an environment to maximize reward. We touch RL only lightly; it's a full course on its own.

You can mostly map your HEP problems onto supervised learning with auxiliary losses. The rest of Month 1 lives in "supervised."

## 3. NumPy — the good parts

NumPy is a thin wrapper around contiguous C arrays. You already know why contiguous matters: stride tricks, cache locality. The mental model is:

- `np.ndarray` is a shape-tagged view over a buffer.
- Operations are vectorized: element-wise ops, reductions, broadcasting.
- Broadcasting turns `(N, 1) + (1, M) → (N, M)` without allocating temporaries when it can help it.

Three skills to drill until they're automatic:

### Skill 1: shapes are everything.
Always know the shape of every array. Every time you read code, ask "what's the shape here?" When debugging, sprinkle `print(x.shape, x.dtype)` everywhere.

### Skill 2: vectorize; avoid Python-level loops.
```python
# bad: O(N) Python overhead
out = np.zeros(n)
for i in range(n):
    out[i] = np.sin(x[i]) * np.cos(y[i])

# good
out = np.sin(x) * np.cos(y)
```
A loop over 10⁶ elements at Python speed takes seconds; vectorized takes milliseconds.

### Skill 3: broadcasting without footguns.
The two rules:
1. Arrays are aligned from the rightmost dimension.
2. A dimension of size 1 broadcasts against any size. Any other mismatch is an error.

```python
x = np.random.randn(100, 3)     # 100 particles, (pt, eta, phi)
means = x.mean(axis=0)          # shape (3,)
x_centered = x - means          # (100, 3) - (3,) → (100, 3)
```

Common failure: you wanted to center across rows, not columns, and didn't use `keepdims=True`.

```python
row_means = x.mean(axis=1, keepdims=True)  # (100, 1) — keepdims matters
x_rowcentered = x - row_means
```

## 4. Pandas — the workhorse for tabular HEP data

`pandas.DataFrame` is a dict-of-columns that pretends to be a 2D table. Think of it as a replacement for most of the things you'd do with a `TTree` when the data fits in memory.

**Mental translation table** (ROOT → pandas):

| ROOT | pandas |
|------|--------|
| `TTree::Draw("pt", "eta<2.4")` | `df.query("eta < 2.4")["pt"]` |
| `TTree::GetEntries()` | `len(df)` |
| `TLeaf` | `Series` (a column) |
| `TCanvas + TH1F::Fill` loop | `df["pt"].hist(bins=100)` or `plt.hist(df["pt"], bins=100)` |
| `TTree::AddFriend` | `df.merge(df2, on="event_id")` |
| Applying a per-event correction function in a loop | `df["pt_corr"] = df["pt"] * calibration(df["eta"])` |

**Patterns to learn this week:**

- `df.groupby("run").agg({"pt": ["mean", "std"], "n_clusters": "sum"})` — run-by-run summaries.
- `df.merge(...)` — join by event/run/cluster ID.
- `df.pivot_table(...)` — 2D summary tables.
- `df.apply(fn, axis=1)` — **avoid when possible**; usually there's a vectorized form and `apply` is a hundred times slower.

**Gotchas specific to pandas:**

- **Chained assignment.** `df[df.x > 0]["y"] = 5` silently doesn't update the original. Use `.loc`:
  ```python
  df.loc[df.x > 0, "y"] = 5
  ```
- **`copy()` vs views.** Same rule as numpy: you may or may not own the data. When in doubt, `.copy()`.
- **`NaN`.** All arithmetic propagates NaN. Use `.dropna()`, `.fillna(...)`, or arithmetic-safe operations (`np.nansum`).
- **Index semantics.** The index is not just row number. Many bugs come from forgetting that and doing `df1 + df2` when the indices don't align.

## 5. Matplotlib — the grammar

You'll use it exactly enough to make honest plots. Don't try to learn everything; learn the object-oriented API (`fig, ax = plt.subplots()`), not the MATLAB-style `plt.plot` / `plt.xlabel` mode.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.hist(df["m_gg"], bins=80, range=(0.05, 0.25), histtype="step", linewidth=1.5)
ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV/$c^2$]")
ax.set_ylabel("Clusters / 2.5 MeV")
ax.set_yscale("log")
ax.legend(loc="upper right")
fig.savefig("results/invmass.png", dpi=160)
```

Stylistic conventions that save you pain later:
- Use `constrained_layout=True` instead of `plt.tight_layout()`.
- Save PNG at 160 dpi for notebooks, PDF for papers.
- LaTeX in axis labels is fine (`r"$p_T$ [GeV/$c$]"`); don't turn on `usetex=True` unless you need to match a paper's kerning, it's slow and fragile.

## 6. Reproducibility as a scientific practice

Reproducibility in ML means: you, six months from now, can `git clone` your own repo and reproduce your results within stochastic noise. This requires:

1. **Environment reproducibility.** `pyproject.toml` + `uv.lock` pins every dependency. Do not use global `pip install` for course work. Each week has its own venv.
2. **Seed discipline.** Every script that touches an RNG sets seeds for Python's `random`, numpy, and (later) torch. Even for "deterministic" operations, ordering matters when GPU kernels are involved.
3. **Config separation.** Don't hardcode hyperparameters in the middle of a training loop. Put them in a `config.yaml` or `pydantic.BaseModel`. Log the exact config with every run output.
4. **Data provenance.** Write a `data/README.md` that says where the data came from, what version, what preprocessing, and when you fetched it. If it's on Kaggle, link and date the version. If it's a Zenodo DOI, use the DOI.
5. **Experiment logging.** Starting in Week 04 we'll use `wandb`; for now, a JSON-per-run under `results/` is fine.

Seed discipline, concretely:
```python
import os, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
```

## 7. The `uv` project layout

A working Week 01 project:

```
Week01/project/
├── pyproject.toml
├── README.md
├── src/
│   └── week01/
│       ├── __init__.py
│       ├── data.py        # Dataset class / loaders
│       ├── fit.py         # Gaussian + linear background
│       └── plot.py        # Figure generation
├── tests/
│   ├── test_data.py
│   └── test_fit.py
└── data/                  # .gitignored
    └── README.md
```

Create it:

```bash
cd Month1-Foundations/Week01
uv init project --python 3.11
cd project
uv add numpy pandas matplotlib scipy
uv add --dev pytest ruff
```

`pyproject.toml` will look like:
```toml
[project]
name = "week01"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "scipy>=1.13",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## 8. Worked example — fitting an invariant-mass peak in `numpy` + `scipy`

You've done this a hundred times in RooFit. Here it is in scientific Python, to build vocabulary.

```python
import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

def gaussian(x, mu, sigma, norm):
    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def background(x, a, b):
    return a + b * x

def signal_plus_bg(x, mu, sigma, norm, a, b):
    return gaussian(x, mu, sigma, norm) + background(x, a, b)

@dataclass
class PeakFit:
    mu: float
    mu_err: float
    sigma: float
    sigma_err: float
    signal_count: float
    background_count: float

def fit_pi0_peak(m_gg: np.ndarray, bin_edges: np.ndarray) -> PeakFit:
    counts, _ = np.histogram(m_gg, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # initial guesses: peak near 0.135 GeV, 5 MeV width, background linear
    p0 = (0.135, 0.010, counts.max(), 1.0, 0.0)
    popt, pcov = curve_fit(signal_plus_bg, centers, counts, p0=p0, sigma=np.sqrt(counts + 1))
    perr = np.sqrt(np.diag(pcov))
    mu, sigma, norm, a, b = popt
    bin_w = bin_edges[1] - bin_edges[0]
    # integrate signal analytically: gaussian area = norm * sigma * sqrt(2π)
    signal = norm * sigma * np.sqrt(2 * np.pi) / bin_w
    bg_in_peak = (a + b * mu) * (6 * sigma)  # ±3σ window
    return PeakFit(mu, perr[0], sigma, perr[1], signal, bg_in_peak)
```

Important things this shows:
- `scipy.optimize.curve_fit` returns (params, covariance). That's your RooFit result.
- We pass `sigma=np.sqrt(counts + 1)` so empty bins don't get zero weight.
- We integrate the Gaussian analytically instead of summing bins. Fewer bias modes.
- The `dataclass` gives us a typed return. Get in the habit now; it pays off in Week 04+.

## 9. Common pitfalls this week

- **Python 2 muscle memory.** `print` is a function. `range` is a generator. Integer division is `//`. If you haven't written Python since ~2018, this bites for about a day.
- **Environment pollution.** You `pip install`ed something globally and now `import pandas` picks up the wrong version. Fix: always `uv run python ...`, never bare `python ...`.
- **Jupyter kernel confusion.** The notebook uses a different Python than your shell. Fix: after `uv add ipykernel`, run `uv run python -m ipykernel install --user --name week01` and select that kernel.
- **Big-matrix memory.** A `(50000, 50000)` float64 array is 20 GB. `x.nbytes` tells you. Use `float32` when you can.
- **Reading performance.** `pd.read_csv` is slow on GB-sized files. Prefer Parquet (`pd.read_parquet`). Orders of magnitude.
- **Silent index misalignment.** `df["a"] + other_df["b"]` aligns on index, not position. Drop or reset indices before arithmetic if you're unsure.

## 10. Debugging discipline

Adopt these now:

1. **Print shapes at every step** of a new pipeline. Remove them later, not before.
2. **Fail loudly.** Use `assert` liberally: `assert df.shape[0] > 0, "empty dataframe"`. Don't silently accept degenerate inputs.
3. **Unit-test the small functions.** `test_fit.py` with a known-input-output pair. If your fitter can't recover (μ=0.135, σ=0.01) from a toy generator, it's broken.
4. **Keep a scratch notebook per project.** Exploratory; not committed to git (or commit with outputs cleared). Promote code into `src/` when it stabilizes.

## 11. What you're *not* doing yet

No train/val/test splits (yet; Week 02). No models with hyperparameters (yet). No GPUs. No neural networks. The goal is to be boring-competent at numpy/pandas/matplotlib/pytest/uv/git so that when Week 03 throws PyTorch at you, you aren't fighting the Python tooling at the same time.

## 12. Connections to your HEP work

Concretely useful this week:
- Rebuild your π⁰ invariant-mass fitter in numpy/scipy and compare to your ROOT version. They should agree to within floating-point noise.
- Port one of your efficiency-study notebooks from ROOT+PyROOT to pure numpy+matplotlib. You'll be surprised how small it becomes.
- Get familiar with `uproot`. It reads ROOT files into numpy/pandas without needing a ROOT installation. You'll thank yourself in Month 2 when you want to throw ROOT files at PyTorch.

```python
import uproot

f = uproot.open("your_sphenix_file.root")
tree = f["T"]
df = tree.arrays(["emcal_e", "emcal_eta", "emcal_phi"], library="pd")
```

## 13. Self-check (do these from memory before moving on)

- Without looking, write the one-liner to compute per-run mean and std of `pt` in a DataFrame.
- Explain broadcasting rules in ≤3 sentences.
- Describe the difference between `df["x"] = y` and `df.loc[mask, "x"] = y`.
- Name the three sources of randomness you need to seed in a Python ML script.
- Given a CSV of 10⁷ rows, name two things faster than `pd.read_csv`.

If you can't answer one, revisit the relevant section. No shortcuts; this muscle memory is the load-bearing thing.

---

Continue to `reading.md`, then `exercises.md`, then `project.md`.
