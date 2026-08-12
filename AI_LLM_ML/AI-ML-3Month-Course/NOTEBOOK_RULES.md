# Notebook rules

How exercise and solution notebooks are written for this course. Applies to every week,
not just Week 01. When a notebook and this file disagree, this file wins.

## 1. Ask for the deliverables on the exercise page — nothing else

The exercise page (`exercises.md`) is the spec. The notebook restates its numbered steps and
its "Accept when" line, and asks for those. It does not invent extra sub-goals, extra
acceptance criteria, or "while you're here, also try…".

If the page is wrong or unreachable (dead URL, tolerance tighter than the statistics allow,
a speedup that the machine can't hit), say so in **one line** where it bites, keep the
page's own criterion as a check, and move on. No essays about it.

## 2. Setup is given; the answer is not

Every cell should run top to bottom the moment the TODO is filled in. So the notebook
provides:

- imports, `sys.path`, data loading, cached-download calls
- constants the page names (`mu = 0.135`, window edges, bin counts, seeds)
- array/DataFrame construction, `plt.subplots(...)` with axis labels already set
- anything that is plumbing rather than physics

and leaves as `# TODO` exactly the line(s) the exercise is about. One TODO per deliverable,
`None` placeholders so the checks below it run instead of raising `NameError`.

## 3. No advanced Python

The reader is learning the scientific stack, not the language. Use:

`import` · variables · `def` with plain positional arguments · `for` · `if/else` ·
`list.append` · f-strings · single-level list comprehensions · library calls

Do not use:

classes · `@dataclass` · decorators · type hints · `from __future__` · `global` ·
`lambda` · closures/factories · generators · `try/except` as control flow · `**kwargs` ·
walrus · nested comprehensions · `subprocess` · `importlib.reload` · `pathlib` chains

Paths are plain relative strings (`"../results/dimuon.pdf"`); the notebook is run from its
own folder. Shell-outs use the line magic `!{sys.executable} -m pytest -q ../tests/x.py`,
never the `subprocess` module.

Provided library code under `src/` may be slightly richer than this (it is read, not
written, by the student) but stays as plain as it can.

## 4. Prose is short

A markdown cell is at most ~3 lines plus, if needed, one display formula. It says what to
produce and gives the page's hint. It does not explain what the student is about to
discover — digressions on numerical analysis, benchmarking methodology, or the history of
the dataset belong in the teacher explainer, not here.

## 5. Checks are plain PASS/FAIL

One helper, defined once in setup, no module-level counters:

```python
def check(label, ok):
    if ok:
        print("PASS  " + label)
    else:
        print("FAIL  " + label)
```

2–5 checks per exercise. Each maps to the page's own acceptance criterion or to an obvious
property (shape, count, file exists). Checks report; they never raise.

## 6. Where the work lives

In the notebook, unless the exercise page names a file or its acceptance criterion is
`pytest`/a CLI. Example (from the archived 12-week course, Week 01): E2 →
`src/week01/center.py` + `tests/test_center.py`, E7 → `tests/test_fit.py`, E9 →
`run.py`; everything else was notebook cells. The same split applies to every week of
the current course.

## 7. The solution notebook mirrors the exercise notebook

Same headings, same order, same cell boundaries — every TODO filled with the simplest
correct code. Comments only where the choice is non-obvious, one line each. For file-based
exercises it shows the finished file in a fenced markdown block, not an executable cell — the
student's own `src/`, `tests/` and `run.py` stay untouched — and points at
`solutions/apply_solutions.py`. It does not stage, patch, or import from a shadow copy of the
project.

## 8. Notebooks are generated, never hand-edited

`notebooks/build_exercises_notebook.py` and `solutions/build_solutions_notebook.py` are the
source. Edit the builder, re-run it, commit both. A hand-edit to the `.ipynb` is lost on the
next build.

```bash
python3 notebooks/build_exercises_notebook.py
python3 solutions/build_solutions_notebook.py
```

## 9. Naming

| file | what |
|---|---|
| `notebooks/WeekNN_Exercises.ipynb` | the working notebook, TODOs unfilled |
| `notebooks/WeekNN_Solutions.ipynb` | the same notebook, filled in |
| `solutions/files/*.py` | finished versions of the file-based exercises |
| `solutions/TEACHER_EXPLAINER.md` | the *why*, at whatever length it needs |

Lesson notes (`*_Lesson.ipynb`) are a different format with different rules — see the
`lesson-notes` skill.
