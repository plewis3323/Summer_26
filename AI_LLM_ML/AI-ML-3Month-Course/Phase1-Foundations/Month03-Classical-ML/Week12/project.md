# Week 12 Project — Capstone 1: MAGIC Gamma/Hadron

## Objective

Ship a tested, calibrated, honestly-validated particle-ID classifier on the
MAGIC gamma/hadron table — ingest → nested CV (logistic vs random forest vs
XGBoost) → calibration → working points — as a clean Python package. This is
the Phase 1 gate (`00-Syllabus.md` §5 and `03-Project-Roadmap.md`):
`pytest -q` green, CI green, nested-CV results, ROC + calibration plots checked in, a
writeup that includes what failed first and a leakage audit. You already
have a tuned MAGIC BDT from Week 11; this week you put it inside the right
*protocol*. Quoting last week's AUC is the sin: different split, budget,
possibly preprocessing. Retrain everything here.

**General option.** A public tabular rare-event set (fraud, medical screening,
credit) with the same nested-CV / calibration / leakage-audit protocol is
allowed. Keep MAGIC if you want the physics thread; switch if you want a
non-HEP portfolio piece. Say which in the README.

## Background — all the physics, from scratch

**The telescope.** **MAGIC** (Major Atmospheric Gamma-ray Imaging Cherenkov)
is a pair of 17-metre dishes on La Palma in the Canary Islands. They do not
look at the gamma ray itself. They look at a faint nanosecond flash the
gamma ray makes in the air above the island.

**Cherenkov radiation.** Light in air travels slower than $c$ — the
refractive index $n>1$ slows it. A charged particle can still go faster
than that local light speed. When it does, it emits a shock-wave of photons,
the optical analog of a sonic boom: **Cherenkov radiation**. A high-energy
particle shower in the upper atmosphere is full of such charged particles,
so the shower lights up a faint blue flash. A dish on the ground images
that flash onto a camera of photomultiplier pixels. One event is one
camera picture: a blob of lit pixels.

**Two kinds of shower.** A **gamma ray** (a photon, here typically GeV–TeV)
hits a nucleus in the upper atmosphere and converts to an electron–positron
pair; those radiate more photons, which pair-produce again. The cascade is
an **electromagnetic shower**: a pancake of $e^\pm$ and photons, compact
and regular, plunging down. The Cherenkov image is a slim, concentrated
ellipse.

A **hadron** — a proton or nucleus from ordinary cosmic rays — also makes
an atmospheric shower, but a **hadronic shower**: it hits a nucleus and
produces pions, which produce more pions, plus a messy electromagnetic
sub-cascade from $\pi^0\to\gamma\gamma$, plus penetrating muons. The
Cherenkov image is lumpier, wider, and less regularly elliptical.

The physics job is **particle identification**: given the camera image, say
whether the primary was a gamma (signal — the astrophysical object you
pointed at) or a hadron (background — the cosmic-ray rain, orders of
magnitude more common in real observing). This dataset is a preselected
educational extract; it is *not* the raw MAGIC trigger stream, and the
class balance is not the sky's.

**Hillas parameters.** The camera image is a blob. Rather than hand the
pixel grid to a model (that is Phase 2), MAGIC-era analyses summarized the
blob as an ellipse. **Hillas parameters** (Hillas, 1985) are that summary —
ten numbers per image, the columns of this table:

| column | meaning |
|--------|---------|
| `fLength` | major axis of the ellipse (mm, in the camera plane) |
| `fWidth` | minor axis (mm) |
| `fSize` | $\log_{10}$ of the total photo-electron count (brightness) |
| `fConc` | fraction of light in the two brightest pixels |
| `fConc1` | fraction of light in the single brightest pixel |
| `fAsym` | distance from the brightest pixel to the ellipse center, projected on the major axis |
| `fM3Long` | cube root of the third moment along the major axis |
| `fM3Trans` | cube root of the third moment along the minor axis |
| `fAlpha` | angle between the major axis and the line from the camera center to the ellipse center (degrees) |
| `fDist` | distance from the camera center to the ellipse center |
| `class` | `g` = gamma, `h` = hadron |

Gamma images tend to be slimmer (`fWidth`/`fLength` small), more
concentrated (`fConc` large), and — when the telescope points at a source —
better aligned (small `fAlpha`: the ellipse "points at" the source).
Hadrons are the opposite on average, with enough overlap that a single
width cut fails. Ten heterogeneous, correlated-by-construction features,
$n = 19{,}020$: Week 11's **tabular** / **ntuple** regime. BDT home turf.
Logistic regression is the linear baseline: if it already separates, the
story is "the ellipse is linearly informative", not "XGBoost is magic."

**Units and labels.** Camera millimetres and degrees stay as they are —
trees ignore monotone rescaling; logistic and (Week 16) an MLP do not, so
a `StandardScaler` lives *inside* those pipelines. Encode `g` as 1
(signal) and `h` as 0 (background) in one place, and never flip it.

## Data

**File:** MAGIC Gamma Telescope, UCI ML Repository dataset 159
(Bock et al.; simulated with CORSIKA, processed to Hillas parameters).
19,020 rows, 10 real-valued features, binary label. Class counts: 12,332
gamma (`g`) and 6,688 hadron (`h`) — *more signal than background*, the
opposite of a real observing night; say so in the writeup. Read the UCI
feature descriptions before ingest.

- UCI page: `https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope`
- Data file (no header): `magic04.data` — comma-separated, 11 columns in
  the order of the table above.
- Reuse Week 11's cached copy if it is already on disk (same path the
  Week 11 notebook used). Otherwise download once, verify a SHA-256 you
  record, and skip the download on rerun — Week 04's `fetch.py` pattern.

Write `data/PROVENANCE.md`: source URL, date fetched, checksum, row count,
and the sentence "one row = one camera image = one shower; a row split is
an event split." Do not commit the CSV if it is large; do commit the
provenance file and the code that fetches.

**Splits (frozen before any model sees a number).** Seeded
`train_test_split(..., test_size=0.20, stratify=y, random_state=0)` into
**trainval** (80%) and **test** (20%). Test is opened once, at the end, for
the ROC, the working-point metrics, and a held-out calibration check.
Nested CV and threshold choice happen on trainval only. Write the split
indices (or a `split_seed` plus the code) so Week 16 can reuse this exact
partition.

## Build steps

Do them in order; each is one of this week's exercises (E4–E7 in
`exercises.md`). E1–E3 are notebook-only unsupervised drills and are not
part of this repo.

**Stage 0 — the repo (30 min).** Week 04's packaging, spent at once:

```
cd ~/course
uv init --package magic_pid
cd magic_pid
git init
uv add numpy pandas matplotlib scikit-learn xgboost
uv add --dev pytest
```

`umap-learn` is an E3 notebook dependency, not a capstone dependency —
do not add it here unless an ingest test needs it (it does not).

Target layout:

```
magic_pid/
  pyproject.toml
  uv.lock
  .gitignore          data/*.data  data/*.csv  .venv/  __pycache__/
  run.py
  writeup.md
  data/
    PROVENANCE.md
  src/magic_pid/
    __init__.py
    ingest.py         Stage 1
    models.py         Stage 1 pipelines + grids
    nested_cv.py      Stage 2
    calibrate.py      Stage 3
    working_points.py Stage 3
    plots.py          ROC, reliability, confusion
  tests/
    test_ingest.py
    test_nested_cv.py
    test_working_points.py
  results/
    metrics.json      written by run.py; committed
  figures/
    reference/        committed ROC + reliability PDFs
```

Commit after every stage. Push to GitHub by the end of Stage 1 — the
Phase 1 gate wants the repo on GitHub, not only on disk.

**Stage 1 — pipeline (E4).** `src/magic_pid/ingest.py`: fetch-or-cache,
load with the ten Hillas names, encode `g=1`, `h=0`, freeze the split,
return arrays. `src/magic_pid/models.py`: three estimators, each a
`Pipeline` so scaling cannot leak (Week 09 E4):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])

def forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1)),
    ])

def xgb():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=400, eval_metric="auc",
            random_state=0, n_jobs=-1, tree_method="hist")),
    ])
```

Small, named grids (inner loop of Stage 2) — keep them small; this is
protocol, not a Kaggle raid:

```python
GRID_LOG = {"clf__C": [0.1, 1.0, 10.0]}
GRID_RF  = {"clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 5]}
GRID_XGB = {"clf__max_depth": [3, 4, 6],
            "clf__learning_rate": [0.03, 0.1, 0.3]}
```

`run.py` at this stage: ingest, fit each pipeline on all of trainval with
defaults (no tuning yet), print a *trainval-fit / test-once* AUC table so
you have a sanity number — then **delete that test-set use** before
Stage 2. The only test-set touch that ships is Stage 3's final evaluation.
`tests/test_ingest.py`: row count 19020, ten features, label in `{0,1}`,
split sizes 80/20, no index overlap, stratification within 1% of the
global gamma fraction.
*Accept when: fresh clone + `uv sync` + one command reproduces the result
table.* (After Stage 2 that table is the nested-CV table; at the end of
Stage 1 a default-pipeline table is enough to prove ingest + `run.py`.)

**Stage 2 — nested CV (E5).** `src/magic_pid/nested_cv.py`. Outer
`StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`, inner
`StratifiedKFold(n_splits=3, shuffle=True, random_state=0)`, scoring
`roc_auc`. For each model, `GridSearchCV(estimator, grid, cv=inner,
scoring="roc_auc")` scored by `cross_val_score(..., cv=outer)`. Same
outer splitter object (or the same `random_state` and `n_splits`) for
every model — identical-protocol. Report mean ± standard deviation of the
five outer scores. Hyperparameters chosen on an outer-fold's test data
are a gate failure; `GridSearchCV` inside `cross_val_score` makes that
structurally hard to get wrong, which is why you use it.

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
search = GridSearchCV(estimator, param_grid, cv=inner, scoring="roc_auc")
scores = cross_val_score(search, X_trainval, y_trainval, cv=outer,
                         scoring="roc_auc")
print(f"nested CV AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
```

Then refit the winning class (almost certainly XGBoost; if not, that is a
finding) on all of trainval with the inner-loop winner from a single
`GridSearchCV` on trainval — this is the model Stage 3 calibrates. Do not
open the test set to pick the class.
`tests/test_nested_cv.py`: on a tiny synthetic two-Gaussian set, nested
CV runs without error and the returned array has length 5; a second test
asserts the MAGIC table is written with three rows (logistic, RF, XGB)
and that `metrics.json` contains `mean` and `std` per model.
*Accept when: the table shows outer-fold mean ± spread and no
hyperparameter was chosen using outer-fold test data.*

**Stage 3 — calibration + working points (E6).** XGBoost ranks well and is
often miscalibrated (Week 10 §9). Split trainval once more
(`val_size=0.25` of trainval, stratified, `random_state=0`) into
**train** and **val**. Fit the Stage 2 winner on train; **Platt-scale**
on val (`CalibratedClassifierCV(..., method="sigmoid", cv="prefit")`, or
an equivalent logistic fit of `p` vs `y` on val). Reliability diagram on
val, then a held-out check on test — the test check is a *report*, not a
retune.

Two working points, **chosen on val**, **reported on test** (Week 10 E4):

- **Fixed efficiency.** Threshold where gamma TPR crosses 0.90; report
  hadron FPR (or rejection $1-\mathrm{FPR}$), plus the confusion matrix.
- **Max $S/\sqrt{S+B}$.** Sweep the threshold on val, take the maximum of
  that significance proxy (signal count and background count after the
  cut, using val's labels), apply that same threshold to test.

`src/magic_pid/working_points.py` implements both as functions of
`(y_true, p_hat) -> threshold` (for choosing) and
`(y_true, p_hat, threshold) -> metrics` (for reporting).
`tests/test_working_points.py`: lock the two *test-set* efficiencies /
significances to 1% of the values stored in `results/metrics.json` (write
the JSON from `run.py`, commit it, assert `pytest.approx(..., abs=0.01)`
on TPR at the 90% point and on the reported $S/\sqrt{S+B}$). A plot the
tests do not lock is decoration.
`plots.py`: ROC (test), reliability (val and test overlay), two confusion
matrices. Axes labeled; gamma TPR on the y-axis of the ROC, not
"accuracy."
*Accept when: the calibration plot is in the repo and both working points
are reproduced by `pytest` to 1% on the held-out set.*

**Stage 4 — writeup (E7).** `writeup.md` in the repo, ~2 pages:

- Problem (gamma vs hadron, why Cherenkov images, why Hillas).
- Data (UCI 159, $n$, class counts, the 80/20 freeze, provenance).
- Method (three pipelines, nested CV, Platt, two working points).
- Results table (nested-CV AUC mean ± spread for all three; test AUC of
  the winner, once).
- Calibration figure, embedded.
- **What failed first** — written the day it happened.
- Leakage audit (named subsection): scaler fit on all rows before the
  split? Duplicate showers on both sides? Any feature a function of the
  label? State that a row split *is* the event split. Hillas features are
  correlated by construction (Week 11 §6) — that is not leakage; a
  permutation-importance story would need the correlated-group caveat.
- Limitations (educational class balance, Hillas not pixels, no energy
  dependence, MAGIC-era features).

Every number comes from `results/metrics.json` or `run.py` stdout. If you
type `0.92` from memory, you will drift.
*Accept when: `writeup.md` exists, includes the failure section, and every
number in it is generated by the pipeline.*

**Stage 5 — one-command run.** `run.py`: ingest → nested CV table →
calibrate → working points → figures → write `results/metrics.json`.
Orchestration only. Then the real test: clone the GitHub repo into a
scratch folder, `uv sync`, `uv run pytest -q`, `uv run python run.py`.
*Accept when: that sequence is green and reprints the table.*

## Acceptance gate (from `03-Project-Roadmap.md` and `00-Syllabus.md` §5)

**End Phase 1:** Capstone-1 repo on GitHub: `pytest -q` green, nested-CV
results, calibration plot, writeup incl. what failed first. Concretely:

- Fresh clone + `uv sync` + `uv run pytest -q` green +
  `uv run python run.py` reprints the nested-CV table and writes the
  figures.
- Nested-CV table: logistic vs RF vs XGBoost, outer-fold mean ± spread,
  no hyperparameter chosen on outer-fold test data.
- ROC and reliability plots checked into `figures/` (and
  `figures/reference/`).
- Two working points, chosen on validation, reported on test, locked by
  tests to 1%.
- Leakage audit written: what you checked, what you found.
- `writeup.md` as in Stage 4. A model that loses to logistic can pass if
  the writeup diagnoses it. An unexplained win cannot.

Then the month sign-off: tag `month-03-complete`, write `retro.md` in the
Month 03 folder, open one issue. Re-derive PCA from variance maximization
cold (the month's rotating flagship) and file the page in Week 12.

## Writeup requirements

Covered in Stage 4 — `writeup.md` *is* the writeup. The repo `README.md`
is the one-screen operator's note: what this is, the three commands, a
pointer to `writeup.md`. Do not duplicate the physics essay in both.

## Stretch goals

- **Permutation importance, honestly.** On the winner, with the
  correlated-Hillas caveat in the caption (Week 11 E5).
- **Energy proxy.** `fSize` is a brightness, hence a crude energy stand-in.
  Slice the test set by `fSize` quartile and report AUC per slice — does
  the classifier quietly become an energy estimator?
- **k-means on hadrons only.** Cluster the `h` class in Hillas space
  (E1's code); is "hadron" one blob or several image-shapes? Unsupervised
  on the training hadrons only.
- **Week 11 model, identical protocol.** Load `models/magic_bdt.json` as a
  fourth row *without retuning*, scored on *this* nested-CV split. If it
  beats your retrained XGBoost, the gap is the protocol, and that is a
  finding.
