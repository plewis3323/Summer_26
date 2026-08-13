# Week 04 Project — The Dimuon Spectrum

## Objective

Rebuild a classic particle-physics measurement — the dimuon invariant-mass
spectrum from real CMS collision data, with a fit of the J/ψ peak — as a clean,
tested, versioned Python package. This is your first project, so every stage is
fully guided; the point is not the physics result (it has been measured to
exquisite precision) but shipping it *professionally*: fresh clone + `uv sync` +
`uv run python run.py` prints the fitted mass and produces the figures, with
`uv run pytest -q` green. This repo is the Month 01 deliverable and the
template for every project after it.

## Background — all the physics, from scratch

**The machine and the detector.** The **LHC** (Large Hadron Collider) at CERN,
near Geneva, accelerates protons to enormous energies and collides them head-on.
Each collision converts energy into a spray of new particles — Einstein's
$E = mc^2$ run in reverse: energy becomes mass. **CMS** (Compact Muon Solenoid)
is one of the giant detectors wrapped around a collision point; it measures the
direction, momentum, and charge of what flies out.

**The muon.** A **muon** (symbol μ) is an elementary particle: a heavier copy
of the electron, about 207 times its mass, with the same charge of −1 (its
antiparticle, μ⁺, has charge +1). Muons matter to experimenters because they
punch through the entire detector leaving a clean track where almost everything
else is absorbed — so "two muons seen" is one of the crispest signatures a
collision can produce. CMS literally has "Muon" in its name.

**Units.** Particle physics measures energy in **GeV** (giga-electron-volts;
Week 01's `EV_TO_JOULES` times 10⁹). In the field's "natural units" convention
(set $c = 1$), momentum and mass are quoted in GeV too. Every energy, momentum,
and mass in this project is a number in GeV.

**Invariant mass — why a peak appears.** Special relativity relates a
particle's energy $E$, momentum vector $\vec p$, and mass $m$ by
$E^2 = |\vec p\,|^2 + m^2$ (with $c = 1$). For a *pair* of muons, define the
**invariant mass** of the pair:

$$M^2 = (E_1 + E_2)^2 - |\vec p_1 + \vec p_2|^2$$

The magic of this quantity: if the two muons came from the decay of a single
parent particle, $M$ equals the parent's mass — *regardless of how fast the
parent was moving*. If the two muons are unrelated debris, $M$ is just some
smoothly distributed number. So when you histogram $M$ for millions of muon
pairs, unrelated pairs form a smooth falling background, and every particle
that decays to μ⁺μ⁻ stamps a **peak** at its own mass. A histogram of $M$ is a
particle *discovery instrument* — this exact plot is how new particles announce
themselves.

**The J/ψ.** The **J/ψ** ("jay-sigh") is a particle made of a charm quark
bound to a charm antiquark, with mass 3.0969 GeV. Its 1974 discovery — as a
sharp dimuon/dielectron peak, found simultaneously by two groups, hence the
double name — proved the charm quark existed and reshaped particle physics
overnight (the "November Revolution"; Nobel Prize, 1976). It decays, among
other ways, to μ⁺μ⁻: opposite charges, since the parent is neutral. Your fit
target.

**Why the peak has width.** The J/ψ's intrinsic mass spread is tiny (~0.1 MeV
— far too narrow to see here). The width you *will* see, about 30 MeV, is
**detector resolution**: CMS measures each muon's momentum with a small random
error, so the reconstructed $M$ smears into an approximately Gaussian bump
around the true mass. That is why the fit model below is a **Gaussian** (the
bell curve $A\,e^{-(M-\mu)^2/2\sigma^2}$, center $\mu$, width $\sigma$) sitting
on a smooth background.

**Fitting.** **Fitting** means finding the model parameters that make a curve
best match data — here, minimizing the summed squared differences between the
model and the histogram's bin counts ("least squares").
`scipy.optimize.curve_fit(model, x, y, p0=...)` does this numerically: you
supply the model function, the data, and initial guesses `p0` for the
parameters (the search must start somewhere sensible); it returns the best-fit
parameters `popt` and a covariance matrix `pcov` whose diagonal holds each
parameter's squared uncertainty — so `np.sqrt(np.diag(pcov))` gives the error
bars. SciPy is NumPy's sibling library of scientific algorithms; this is your
first use of it.

**CERN Open Data and the PDG.** CERN releases real collision data to the
public at **opendata.cern.ch**, including small CSV files derived for
education — real physics, in exactly the format Week 03 taught you. The
**PDG** (Particle Data Group) publishes the authoritative compilation of
particle properties; the PDG J/ψ mass, 3.0969 GeV, is the reference value your
fit must reproduce.

## Data

**File:** `MuRun2010B.csv` — 100,000 dimuon events selected from the CMS
Run2010B Mu dataset. CERN Open Data record 700 ("Dimuon event information
derived from the Run2010B public Mu dataset") — read the record page.

- URL: `https://opendata.cern.ch/record/700/files/MuRun2010B.csv` (~15 MB)
- SHA-256: `47d09d16c01db9a45ce349c9730f032fdd22bf27980ecf524e7fb1fd84572cdb`

One row per event: `Run`, `Event` (which data-taking run, which collision),
then for each of the two muons (suffix `1`/`2`):

| column | meaning |
|--------|---------|
| `Type` | `G` = global muon (seen in both the inner tracker and the outer muon system — the well-measured kind); `T` = tracker-only |
| `E` | energy (GeV) |
| `px, py, pz` | momentum components (GeV); `z` is the beam direction |
| `pt` | transverse momentum — the momentum component perpendicular to the beam (GeV); low-`pt` tracks are poorly measured |
| `eta` | pseudorapidity — a monotone measure of angle to the beam: 0 = perpendicular, larger = more forward; CMS measures well out to about \|eta\| = 2.4 |
| `phi` | azimuthal angle around the beam (radians) |
| `Q` | electric charge, +1 or −1 |

and finally `M` — the pair's invariant mass (GeV), precomputed, spanning
2–110 GeV (the dataset is preselected to M > 2).

Two honest warnings from the record page and the file itself: this educational
selection is *not* physics-analysis grade, and one header is literally
`"px1 "` — trailing space — so strip column names on load (Stage 2). Real data
bites in exactly these small ways.

## Build steps

**Stage 0 — the repo (30 min).** Everything from `lesson.md`, spent at once:

```
cd ~/course
uv init --package dimuon
cd dimuon
git init
uv add numpy pandas matplotlib scipy
uv add --dev pytest
```

Target layout — build it as the stages fill it in:

```
dimuon/
  pyproject.toml      committed (uv maintains it)
  uv.lock             committed
  .gitignore          data/*.csv  .venv/  __pycache__/  figures/*.pdf  run*.txt
  run.py              Stage 5
  data/
    PROVENANCE.md     committed; the CSV beside it is not
  src/dimuon/
    __init__.py
    fetch.py          Stage 1
    select.py         Stage 2
    fit.py            Stage 3
    plots.py          Stage 4
  tests/
    test_fetch.py
    test_select.py
    test_fit.py
  figures/
    reference/        committed reference PDFs (Stage 4)
```

Commit after every stage, message saying why. Use a branch for anything
experimental. `git push` to a GitHub repo by the end of Stage 1 — offsite
backup from day one.

**Stage 1 — data fetch with provenance.** In `src/dimuon/fetch.py`, three new
standard-library pieces, one line of teaching each: `urllib.request.urlretrieve
(URL, path)` downloads a file; `hashlib.sha256(data).hexdigest()` turns bytes
into a 64-character fingerprint that changes if even one bit changes (a
**checksum**); `raise ValueError("message")` *creates* an error yourself — the
same kind you learned to read in Week 01, now thrown on purpose when the data
cannot be trusted (and `pytest.raises(ValueError)` tests that it happens).

```python
import hashlib
import os
from urllib.request import urlretrieve

URL = "https://opendata.cern.ch/record/700/files/MuRun2010B.csv"
SHA256 = "47d09d16c01db9a45ce349c9730f032fdd22bf27980ecf524e7fb1fd84572cdb"
CSV_PATH = "data/MuRun2010B.csv"

def checksum(path):
    with open(path, "rb") as f:          # "rb" = read raw bytes, not text
        return hashlib.sha256(f.read()).hexdigest()
```

`fetch()` then implements: if `os.path.exists(CSV_PATH)` and the checksum
matches, print `"cached: data/MuRun2010B.csv"` and return the path; otherwise
`os.makedirs("data", exist_ok=True)`, download, and verify — raising
`ValueError` with a message naming both checksums on mismatch. A
`write_provenance()` writes `data/PROVENANCE.md`: source URL, record number,
date fetched (`date.today()` from the `datetime` module), and checksum —
called on every successful fetch. `tests/test_fetch.py`: write a small garbage
file, assert its checksum differs from `SHA256`, and assert your verify step
raises; delete the garbage file at the end (`os.remove`).
*Accept when: rerun skips the download (cached) and a corrupted file is
rejected with a clear message.*

**Stage 2 — selection module.** Not every row is trustworthy: tracker-only
muons and low-`pt` or very forward tracks are poorly measured and would smear
your peak. A **cut** (Week 02 E3, Week 03 E3) removes them; the **cut flow**
is the count of surviving events after each cut, in order — the standard
bookkeeping table of every particle-physics analysis. First add `load` to
`fetch.py` (the trailing-space fix lives in exactly one place):

```python
import pandas as pd

def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]   # "px1 " -> "px1"
    return df
```

Then `src/dimuon/select.py`, config in one place:

```python
CUTS = {"pt_min": 3.0, "eta_max": 2.4}

def both_global(df):
    return (df["Type1"] == "G") & (df["Type2"] == "G")

def opposite_charge(df):
    return df["Q1"] * df["Q2"] == -1
```

plus `pt_cut(df)` and `eta_cut(df)` reading `CUTS` (use `.abs()` for `|eta|`),
and `apply_cuts(df)` chaining the masks with `&` in the order below, appending
`(name, int(mask.sum()))` to a cut-flow list after each. Prototype in a
notebook first — Week 03 style — then promote. Reference counts (your module
must reproduce them exactly):

| cut | events |
|-----|--------|
| all | 100000 |
| both global | 59485 |
| + opposite charge | 41798 |
| + both pt > 3 GeV | 38902 |
| + both \|eta\| < 2.4 | 38614 |

`tests/test_select.py`: build a tiny `pd.DataFrame` by hand (4–5 rows, each
designed to fail exactly one cut) and test every cut function in isolation;
then one test calling `fetch()` + `load()` + `apply_cuts()` asserting the
five reference counts. Add one physics closure test: recompute $M$ from the
energy and momentum columns via the invariant-mass formula
(`np.sqrt(np.maximum(m2, 0.0))` guards rounding) and assert at least 99% of
events agree with the `M` column to within 0.1% — the file's six significant
digits cost the rest. Tests that need the CSV call `fetch()` themselves: the
first `pytest` run downloads once, every later run is cached.
*Accept when: cut-flow counts match Week 02 and tests cover each cut in
isolation.*

**Stage 3 — peak fit.** In `src/dimuon/fit.py`. The model — Gaussian signal
on a straight-line background, five parameters:

```python
import numpy as np
from scipy.optimize import curve_fit

FIT_LO = 2.8
FIT_HI = 3.4
N_BINS = 60
P0 = [800.0, 3.1, 0.03, 50.0, 0.0]   # A, mu, sigma, a, b

def model(m, A, mu, sigma, a, b):
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + a + b * m
```

`fit_jpsi(masses)`: histogram the selected masses with
`np.histogram(masses, bins=N_BINS, range=(FIT_LO, FIT_HI))`, compute bin
centers as `0.5 * (edges[:-1] + edges[1:])`, then
`curve_fit(model, centers, counts, p0=P0)`; return `popt` and
`np.sqrt(np.diag(pcov))`. The initial values `P0` are part of the spec — a
fit that only converges from a lucky start is not reproducible. Reference
result on the Stage 2 selection (8,285 events in the window): μ = 3.0931 ±
0.0004 GeV, σ = 30.9 ± 0.5 MeV — 3.8 MeV below the PDG 3.0969 GeV (real
detectors have small momentum-scale offsets; well inside one resolution σ).
`tests/test_fit.py`: build synthetic data with a *known* answer —
`rng = np.random.default_rng(42)`, 4,000 events `rng.normal(3.0969, 0.030,
4000)` plus 4,000 background `rng.uniform(2.8, 3.4, 4000)`, concatenate, fit,
assert the recovered μ is within 5 MeV of 3.0969 (`pytest.approx(3.0969,
abs=0.005)`) — then a second test fitting the real selection and asserting
μ within 0.030 of the PDG value.
*Accept when: fitted J/ψ mass is within 30 MeV of the PDG value and the fit
converges from the stated seed/initial values every run.*

**Stage 4 — figure module.** In `src/dimuon/plots.py`, two functions, each
taking data and an output path, each building `fig, ax = plt.subplots()`,
labeling every axis with units, and `fig.savefig(path)`. Two one-line
additions to your Week 03 plotting: `ax.set_xscale("log")` /
`ax.set_yscale("log")` make an axis logarithmic (equal steps multiply instead
of add — the only way to see peaks at 3 and 91 GeV on one figure), and
`np.logspace(np.log10(2.0), np.log10(110.0), 200)` makes 200 bin edges evenly
spaced in that log sense. `full_spectrum(masses, path)`: histogram of *all*
loaded events, log–log; you should see four spikes on the falling background —
J/ψ (3.10), ψ(2S) (3.69, small), the Υ family (9.4–10.4), and the Z boson
(91). Label them with `ax.text` if you like. `jpsi_zoom(masses, popt, path)`:
the Stage 3 window histogram with the fitted curve overlaid — evaluate
`model` on `np.linspace(2.8, 3.4, 300)` (pass `popt[0]` … `popt[4]`
explicitly) and `ax.plot` it over the data; put the fitted mass in the title
via an f-string. Pipeline writes `figures/spectrum.pdf` and
`figures/jpsi_fit.pdf`; when they look right, copy both into
`figures/reference/` and commit — future reruns are compared against these.
*Accept when: both PDFs are produced by the pipeline, not by hand, and match
committed reference images by eye.*

**Stage 5 — one-command run.** `run.py` at the root, orchestration only —
every line an import or a call into `src/dimuon/`:

```python
from dimuon.fetch import fetch, load
from dimuon.select import apply_cuts
from dimuon.fit import fit_jpsi
from dimuon.plots import full_spectrum, jpsi_zoom

def main():
    df = load(fetch())
    selected, flow = apply_cuts(df)
    for name, count in flow:
        print(f"{name:22s} {count:7d}")
    popt, perr = fit_jpsi(selected["M"].to_numpy())
    print(f"J/psi mass = {popt[1]:.4f} +- {perr[1]:.4f} GeV   (PDG: 3.0969)")
    print(f"width      = {abs(popt[2])*1000:.1f} +- {perr[2]*1000:.1f} MeV")
    full_spectrum(df["M"].to_numpy(), "figures/spectrum.pdf")
    jpsi_zoom(selected["M"].to_numpy(), popt, "figures/jpsi_fit.pdf")

main()
```

(`.to_numpy()` converts a DataFrame column to the plain array the fit and
plot functions expect. Have `plots.py` create `figures/` with `os.makedirs`
if missing.) Then the real test: clone your GitHub repo into a scratch
folder, `uv sync`, `uv run pytest -q`, `uv run python run.py`.
*Accept when: fresh clone + `uv sync` + `uv run python run.py` completes and
prints the fitted mass.*

**Stage 6 — determinism check.** `uv run python run.py > run1.txt`, run it
again into `run2.txt`, and `diff run1.txt run2.txt` — silence is the pass.
Diff the printed numbers, not the PDF bytes (PDFs embed a creation
timestamp; that is a cosmetic difference, and your writeup may say so in one
line). Then say *why* it passes, in your README, one sentence per suspect:
no unseeded randomness anywhere in the pipeline (the only `rng` in the repo
is the seeded one in `test_fit.py`), versions pinned by `uv.lock`, no
by-hand steps left to run in the wrong order.
*Accept when: two runs produce identical fit parameters and identical
cut-flow tables.*

## Acceptance gate (from `03-Project-Roadmap.md`)

**Peak mass within PDG value ± fit σ; one command → plot.** Concretely:

- Fresh clone + `uv sync` + `uv run pytest -q` green +
  `uv run python run.py` prints the fitted mass and writes both PDFs.
- |fitted μ − 3.0969 GeV| < fitted σ — with the reference numbers, 3.8 MeV
  against σ ≈ 31 MeV — and within the Stage 3 criterion of 30 MeV.
- Cut-flow table matches the Stage 2 reference; `data/PROVENANCE.md`
  committed; `figures/reference/` committed.

Then the month sign-off (week README / syllabus §9): tag `month-01-complete`,
write the 250-word `retro.md` in the month folder, and open one issue for the
single biggest thing you don't yet understand.

## Writeup requirements

A `README.md` in the `dimuon/` repo — one screen, five parts: what this is
(two sentences, with the spectrum figure embedded); how to run it (the three
commands); the result (mass ± uncertainty vs PDG, the cut-flow table); **what
failed first** — the first thing that genuinely broke and what it cost you
(the trailing-space column? the fit diverging before you set `P0`? a checksum
surprise?), written the day it happened; and what you would do next. Negative
results and dead ends go in, per the syllabus honesty policy — this is not a
LinkedIn post.

## Stretch goals

- **ψ(2S).** Refit in a 3.5–3.9 GeV window for the J/ψ's heavier sibling at
  3.6861 GeV — far fewer events; watch the uncertainties grow.
- **The Υ family.** Window 9.0–10.6 GeV holds *three* overlapping peaks —
  Υ(1S/2S/3S) at 9.460, 10.023, 10.355 GeV. Fit a sum of three Gaussians
  sharing one background; can you resolve them?
- **Same-charge control.** Overlay the mass histogram of same-charge pairs
  (`Q1 * Q2 == +1`) on the J/ψ zoom: no parent particle can decay to two
  same-charge muons, so the peak must vanish — the classic background
  cross-check.
- **Resolution vs mass.** Compare σ/μ for J/ψ, Υ(1S), and Z (fit a wide
  Gaussian near 91 GeV): how does the relative resolution trend with mass,
  and why might that be?
