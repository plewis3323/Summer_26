# Week 01 — project

Working environment for the E1–E10 exercises in `../exercises.md`.

## Setup

```bash
cd Month1-Foundations/Week01/project
uv sync --extra dev
uv run python -m ipykernel install --user --name week01
cd notebooks
uv run jupyter lab Week01_Exercises.ipynb   # then select the "week01" kernel
```

`uv sync` installs `week01` in editable mode, so your edits under `src/` take effect
immediately — no reinstall between cells. **Start Jupyter inside `notebooks/`**: the
notebooks use paths relative to that folder (`../data`, `../results`, `../tests`).

## Layout

```
project/
├── notebooks/
│   ├── Week01_Exercises.ipynb        # E1–E10: the working notebook
│   ├── Week01_Solutions.ipynb        # worked answers (read after you're stuck)
│   └── build_exercises_notebook.py   # regenerates the .ipynb from source
├── solutions/
│   ├── TEACHER_EXPLAINER.md          # the WHY, from first principles
│   ├── build_solutions_notebook.py   # regenerates the solution .ipynb
│   ├── files/*.py                    # solutions to E2, E7, E9
│   └── apply_solutions.py            # diff / apply / restore them
├── src/week01/
│   ├── data.py      # dataset loaders (provided — download + cache)
│   ├── center.py    # E2  — you implement
│   └── fit.py       # E7  — README §8 code; you add the input guard
├── tests/
│   ├── test_data.py    # provided, passes as-is
│   ├── test_center.py  # E2 — you write these
│   └── test_fit.py     # E7 — you write these
├── run.py           # E9  — you implement
├── data/            # git-ignored, downloaded on demand
└── results/         # git-ignored, figures + run JSONs
```

## Where each exercise lives

Everything is notebook work except where `exercises.md` names a file or its acceptance
criterion is `pytest` / a CLI.

| | Exercise | Work in |
|---|---|---|
| E1 | NumPy shape calisthenics | notebook |
| E2 | Broadcasting trap | `src/week01/center.py`, `tests/test_center.py` |
| E3 | pandas: read, filter, groupby | notebook |
| E4 | matplotlib: publication figure | notebook → `results/dimuon.pdf` |
| E5 | π⁰ toy + pull distribution | notebook |
| E6 | uproot → pandas | notebook |
| E7 | pytest drill | `tests/test_fit.py` (+ a guard in `fit.py`) |
| E8 | sklearn sanity check | notebook |
| E9 | Reproducibility drill | `run.py` |
| E10 | Vectorize 10⁶ invariant masses | notebook |

## Checks

```bash
uv run pytest -q        # E2 + E7 + the provided loader tests
uv run ruff check .
```

The notebook self-grades: each exercise ends in a few `check(...)` calls that print PASS/FAIL
against the acceptance criteria on the exercise page.

## Python level

Everything here — `src/`, `tests/`, `run.py`, both notebooks and the builders — is written
with the subset of Python a few months in gets you: `import`, variables, `def` with plain
arguments, `for`, `if/else`, lists, dicts, f-strings, `with open(...)`, and library calls.
No classes, dataclasses, decorators, type hints, `lambda`, generators or comprehension
nesting. The rules are in `../../../NOTEBOOK_RULES.md`.

One consequence worth knowing up front: `fit_pi0_peak` returns a **dictionary**, so its
results are `fit["mu"]`, `fit["mu_err"]`, `fit["sigma"]` and so on. `exercises.md` writes
those as `fit.mu` and `fit.mu_err`, which would need a class to work.

## Regenerating the notebooks

Both `.ipynb` files are generated — edit the builder, not the notebook.

```bash
python3 notebooks/build_exercises_notebook.py
python3 solutions/build_solutions_notebook.py
```

House style for what goes in them: `../../../NOTEBOOK_RULES.md`.

## Where `exercises.md` and reality disagree

Each is flagged in one line in the notebook where it comes up; none is a bug in your code.

1. **The E3 data URL 404s.** The `cms-opendata-workshop/workshop2023-lesson-cmsods` link is
   dead. `week01.data` uses two live mirrors of the same 21-column file (~72 MB, 475,465 events).
2. **E5's pull tolerances are inside their own statistical error at 100 toys.** "Mean within
   0.05" has a statistical error of 0.10 at N=100, so a flawless fit can fail it. The notebook
   prints both errors next to the numbers.
3. **E10's ≥200× is not reachable here.** An honest loop-vs-vectorized timing on this machine
   is ~1.5 s vs ~0.02 s, i.e. ~70×. The ratio is a property of the CPU and of how the loop was
   written. The notebook checks ≥50× and reports the doc's 200× separately.

## Solutions

`notebooks/Week01_Solutions.ipynb` is the same notebook with every TODO filled in;
`solutions/TEACHER_EXPLAINER.md` is the reasoning behind it. The solution notebook is
read-only with respect to your work — it shows the E2/E7/E9 answers as file listings and never
writes to `src/`, `tests/` or `run.py`. See `solutions/README.md`.
