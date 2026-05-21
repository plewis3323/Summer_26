# Week 01 — Mini-Project: π⁰ Mass-Peak Fitter, Python Edition

## Problem

Replicate, in pure Python, a workflow you've done dozens of times in ROOT: ingest photon-pair data, histogram the γγ invariant mass, fit a Gaussian-plus-background, extract `(μ, σ, N_signal, N_background)`, and produce a publication-quality figure. Every step must be tested and reproducible.

This project exists to build muscle memory for numpy/pandas/scipy/matplotlib and the `uv`/`ruff`/`pytest` toolchain before we introduce ML proper in Week 02.

## Data

Two valid data sources:

1. **Recommended for familiarity:** simulate toys. Drop 20% signal from a Gaussian at μ=0.135 GeV, σ=0.008 GeV, and 80% linear background over (0.05, 0.25) GeV. 100 000 events total.

2. **Stretch (optional):** a small CMS photon-pair subset from CERN Open Data. You will not get a clean π⁰ peak at LHC energies in a public sample in Week 01; a J/ψ → μμ peak from DoubleMu2011 works as the analog. Use that if you want real data.

## Scope

Concretely, deliver a Python package `week01` with:

```
project/
├── pyproject.toml
├── README.md            (problem, data, results, caveats)
├── src/week01/
│   ├── __init__.py
│   ├── data.py          (generate_toy, load_parquet)
│   ├── fit.py           (PeakFit dataclass, fit_pi0_peak)
│   ├── plot.py          (plot_mass_spectrum)
│   └── cli.py           (entry point: python -m week01.cli ...)
├── tests/
│   ├── test_data.py
│   ├── test_fit.py
│   └── test_plot.py
└── results/
    ├── invmass.png
    └── metrics.json
```

## Acceptance criteria

1. `uv sync && uv run pytest -q` passes with 0 failures.
2. `uv run python -m week01.cli --seed 42 --n-events 100000 --output-dir results/` regenerates `results/invmass.png` and `results/metrics.json` exactly (byte-identical or numerically identical for the metrics).
3. Fit μ is within 3σ_μ of 0.135 GeV; fit σ is within 3σ_σ of 0.008 GeV.
4. `metrics.json` includes `{mu, mu_err, sigma, sigma_err, n_signal, n_signal_err, n_background, n_background_err, git_sha, wall_clock_s, config}`.
5. Figure has: LaTeX-formatted axis labels, log y-axis option, legend, and data overlaid with the fit function and residual panel underneath.
6. Code passes `ruff check` and `ruff format --check`.
7. Short (200-word) "Results" section in the project README with the fit numbers, the figure, and two caveats.

## Step-by-step

### Step 1. Scaffold
```bash
cd Month1-Foundations/Week01
uv init project --package --python 3.11
cd project
uv add numpy pandas matplotlib scipy
uv add --dev pytest ruff
```

Create `src/week01/__init__.py` with a version string.

### Step 2. Data
In `src/week01/data.py`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ToyConfig:
    n_events: int = 100_000
    signal_frac: float = 0.2
    mu: float = 0.135
    sigma: float = 0.008
    m_min: float = 0.05
    m_max: float = 0.25
    seed: int = 42

def generate_toy(cfg: ToyConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    n_sig = rng.binomial(cfg.n_events, cfg.signal_frac)
    n_bg = cfg.n_events - n_sig
    sig = rng.normal(cfg.mu, cfg.sigma, size=n_sig)
    bg = rng.uniform(cfg.m_min, cfg.m_max, size=n_bg)
    m = np.concatenate([sig, bg])
    rng.shuffle(m)
    return m
```

Unit test: check shape, check approximate signal fraction, check all values in range.

### Step 3. Fit
In `src/week01/fit.py`, adapt the worked example from the README. Add input validation (raise on empty input or non-monotonic bin edges). Return a typed `PeakFit` dataclass with covariances for proper error propagation.

### Step 4. Plot
In `src/week01/plot.py`, two-panel plot (data + fit overlay + residuals). Save to `results/invmass.png` at 160 dpi and `results/invmass.pdf`.

### Step 5. CLI
In `src/week01/cli.py`:

```python
import argparse, json, pathlib, subprocess, time
from .data import generate_toy, ToyConfig
from .fit import fit_pi0_peak
from .plot import plot_mass_spectrum
import numpy as np

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-events", type=int, default=100_000)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ToyConfig(n_events=args.n_events, seed=args.seed)
    m = generate_toy(cfg)
    edges = np.linspace(0.05, 0.25, 81)  # 2.5 MeV bins
    t0 = time.monotonic()
    fit = fit_pi0_peak(m, edges)
    dt = time.monotonic() - t0

    plot_mass_spectrum(m, edges, fit, args.output_dir / "invmass.png")

    metrics = {
        "mu": fit.mu, "mu_err": fit.mu_err,
        "sigma": fit.sigma, "sigma_err": fit.sigma_err,
        "n_signal": fit.signal_count,
        "n_background": fit.background_count,
        "git_sha": git_sha(),
        "wall_clock_s": dt,
        "config": cfg.__dict__,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

if __name__ == "__main__":
    main()
```

### Step 6. Tests
- `test_data.py`: shape; value range; signal fraction within binomial tolerance.
- `test_fit.py`: recovery of μ and σ from a fresh toy; raises on empty input; monotonic bin edges required.
- `test_plot.py`: PNG is produced; file size > 0.

### Step 7. Lint & polish

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run pytest -q
```

### Step 8. Write the README

Include: problem statement, command to reproduce, the figure, the metrics table, and a **caveats** section with at least two honest items. Example caveats:

- "Linear background is a crude model. A constant background would bias σ low; a 2nd-order polynomial pulls σ slightly high. The Gaussian+linear choice is tradition, not principle."
- "pull width on 200 toys is 1.04, not 1.00. Mild covariance underestimation; would rerun with Hesse or profile-likelihood for production."

## Extension (if you have time)

- Swap the toy for actual sPHENIX photon clusters from an existing DST you have access to. Read with `uproot`. Compare the Python fit to your ROOT/RooFit fit; they should agree to a few ‰.
- Add a `side-band background subtraction` option to `fit_pi0_peak`.
- Parametrize the background with a Chebyshev polynomial of order 2 behind a `--bg-order` flag.

## Common failure modes and fixes

- **Fit runs away.** Bad initial guesses. Print `p0` and `counts.max()`; scale norm appropriately.
- **Covariance contains NaN.** Likely you passed `sigma=np.sqrt(counts)` with empty bins. Add `+ 1` or drop those bins.
- **Pull distribution is wide.** Either model mismatch or you're not using Poisson errors on bin counts. Check both.
- **`test_fit_recovers_mu` is flaky.** Loosen tolerance to 3–4σ or fix the seed inside the test.

## Sign-off

When this project is done, tick boxes in `04-Progress-Tracker.md` under Week 01, commit, and push. You're now ready to put a loss function on something in Week 02.
