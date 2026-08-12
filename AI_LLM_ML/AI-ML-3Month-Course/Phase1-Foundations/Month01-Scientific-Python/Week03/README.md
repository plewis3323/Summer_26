# Week 03 — Software Engineering I

Tests are to code what systematic checks are to a measurement: the difference between a
number and a result you can defend.

## Objectives

- Work in feature branches; rebase onto main; resolve a merge conflict without panic.
- Create and pin a project environment with `uv`; explain lockfile vs `pyproject.toml`.
- Lay out a real package (`src/` layout) instead of a folder of scripts.
- Write `pytest` tests including parametrized cases and a numerical-tolerance test.
- Drive `pdb` (breakpoints, stepping, inspecting frames) instead of print-debugging.

## Core material (~3 hrs)

- *Pro Git* (Chacon & Straub), Chapter 2 (Git Basics) and Chapter 3 (Branching) —
  Chapter 3 is the one that changes how you work; do the branching examples by hand.
- `uv` official docs: projects, `uv sync`, lockfiles. Short read.
- `pytest` docs: "Get started" plus the pages on assertions and `@pytest.mark.parametrize`.
- `ruff` docs: skim configuration; you will set it up once and mostly obey it.
- Python docs: the `pdb` module page — learn `b`, `n`, `s`, `c`, `p`, `up`/`down`, `q`.

## Exercises (built when the week starts)

1. **Branch/rebase kata.** Scripted sequence: branch, commit, diverge main, rebase,
   resolve the induced conflict, fast-forward merge.
   Accept when: `git log --oneline --graph` shows a linear history with all commits present.
2. **Environment from scratch.** New `uv` project for the course package; add numpy,
   pandas, matplotlib, pytest; sync.
   Accept when: fresh `uv sync` + `uv run python -c "import week03"` succeeds and `uv.lock` is committed.
3. **Package a function.** Move Week 01's invariant-mass code into
   `src/week03/kinematics.py` with a plain function interface.
   Accept when: it imports from the notebook and from pytest without `sys.path` hacks.
4. **Test suite.** Tests for the mass function: a known-value case (Z→μμ four-vectors),
   a parametrized set, and a floating-point tolerance case using `pytest.approx`.
   Accept when: `pytest -q` reports ≥5 tests, all green.
5. **Bug hunt with pdb.** A provided module computes a wrong pseudorapidity for one input
   class; find it with `pdb` only (no added prints).
   Accept when: the bug is fixed, a regression test covering it passes, and the pdb session steps are listed in ≤5 bullet points.
6. **Lint gate.** Run `ruff check` on the package, fix everything it flags, and commit a
   `ruff` config in `pyproject.toml`.
   Accept when: `ruff check .` exits clean.

## Deliverable

A repo (or subfolder) with `src/` layout, `pyproject.toml` + `uv.lock`, green
`pytest -q`, clean `ruff check`, and a branch history showing the rebase kata.

## Review

1. (Wk 01) Boolean-mask cuts: does `a[mask]` return a view or a copy? Consequence for
   memory when you chain five cuts on a 10⁷-event array?
2. (Wk 02) Sketch from memory the `groupby → agg` call that produced the per-run yield
   table.
3. (Wk 02) Why does tidy layout make the three-mass-window comparison a one-liner?
4. (Physics) Map each onto your analysis practice: unit test, regression test, lockfile.
   (Closure test, frozen cuts, …?)
