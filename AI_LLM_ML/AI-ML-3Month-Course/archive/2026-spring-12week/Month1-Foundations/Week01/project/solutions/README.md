# Week 01 — solutions

Answer key for `../notebooks/Week01_Exercises.ipynb`. Nothing here is loaded by the
exercises; your own `src/`, `tests/` and `run.py` are never touched unless you explicitly
run `apply_solutions.py --apply`.

```
solutions/
├── TEACHER_EXPLAINER.md          # WHY, from first principles -- read when stuck
├── build_solutions_notebook.py   # regenerates the solution notebook
├── apply_solutions.py            # diff / apply / restore the .py solutions
└── files/
    ├── center.py        # E2
    ├── test_center.py   # E2
    ├── fit.py           # E7 (+ the popt/pcov E9 needs)
    ├── test_fit.py      # E7
    └── run.py           # E9
```

The notebook itself is `../notebooks/Week01_Solutions.ipynb`.

## How to use this

**Get stuck first.** These exercises are the learning; this directory is what you check
yourself against afterwards, or unstick yourself with after twenty minutes of circling.

When you are stuck, in order:

1. `TEACHER_EXPLAINER.md` — find your exercise's section, read to the first idea, then
   **stop reading and go try it**.
2. `Week01_Solutions.ipynb` — the worked code.
3. `files/*.py` — the file-based exercises, commented.

There is a symptom → cause lookup table at the bottom of `TEACHER_EXPLAINER.md`.

## Running the solution notebook

```bash
cd Month1-Foundations/Week01/project/notebooks
uv run jupyter lab Week01_Solutions.ipynb    # kernel: week01
```

Every cell runs against your own tree — it imports the provided `week01.data` and
`week01.fit` only, and shows E2, E7 and E9 as file listings rather than executing them, so
nothing here can overwrite your work or report a false PASS.

Expect **59 PASS and 1 FAIL**. The failure is E10's `>=200x` criterion from `exercises.md`,
which is not reachable on this machine with an honest Python loop (~70× measured). E5's two
page criteria pass but sit inside their own statistical error at 100 toys, so they can flip.

Takes about two minutes after the first run (the 72 MB CMS CSV is cached in `data/`).

## Putting the solutions in your tree for real

```bash
python solutions/apply_solutions.py --diff      # what would change
python solutions/apply_solutions.py --apply     # overwrite, backing up to solutions/.backup/
python solutions/apply_solutions.py --restore   # undo
```

After `--apply`, `uv run pytest -q` from the project root passes 13 tests.

## Where these solutions deviate from the scaffold, and why

| file | change | why |
|---|---|---|
| `fit.py` | `ValueError` on empty input | E7 step 3 — the test demands it |
| `fit.py` | `abs(sigma)` | the model only sees `sigma**2`, so the sign is arbitrary |
| `fit.py` | `absolute_sigma=True` | Poisson errors are absolute, not relative weights |
| `fit.py` | `background_count` ÷ bin width | README §8 has counts·GeV/bin — a units bug |

Each is argued in the file's module docstring. The scaffold's returned dictionary already
carries `popt`/`pcov`: E9 cannot compute a likelihood without the fitted model.

`fit_pi0_peak` returns a dictionary rather than an object, so results read `fit["mu"]` and
`fit["mu_err"]` — `exercises.md` writes them as `fit.mu` and `fit.mu_err`, which would need a
class, and this project stays inside the beginner subset described in `../README.md`.
