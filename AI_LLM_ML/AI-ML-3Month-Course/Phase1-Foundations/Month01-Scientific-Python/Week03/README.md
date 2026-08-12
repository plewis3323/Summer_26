# Week 03 — The scientific stack

Loops over lists got you through Week 02; science needs millions of numbers at
once. This week you learn the three packages that make Python a scientific
language — NumPy for arrays, matplotlib for figures, pandas for tables — and
end by finding a hidden peak in a synthetic dataset, the dry run for Week 04's
real one.

## Objectives

- Explain why vectorized array code beats loops (speed and clarity) and
  demonstrate it with a timing.
- Create and index arrays; predict shapes; explain views vs copies.
- Select data with boolean masks and combine conditions with `&`/`|`.
- Use broadcasting deliberately — predict output shapes before running.
- Make labeled line, scatter, and histogram plots and save them to PDF.
- Load a CSV into a DataFrame; filter rows with masks; compute per-group
  summaries with `groupby`.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text; work in a notebook alongside it.
- VanderPlas, *Python Data Science Handbook* (free online), chapter 2 (NumPy)
  — the deep pass on arrays, masks, broadcasting.
- VanderPlas, chapter 4 (matplotlib) through histograms and legends, and
  chapter 3 (pandas) through aggregation/grouping.
- NumPy user guide, "Broadcasting" page — short; read it twice.

## Exercises

See `exercises.md` (notebook generated when the week starts, per
`NOTEBOOK_RULES.md`). Seven exercises: a loop-vs-array timing, shape
prediction, mask-based cuts, a broadcast centering, the three plot types, a
first DataFrame, and a bump hunt that synthesizes the week.

## Deliverable

Completed exercise notebook (all checks PASS) plus `decay_hist.pdf` and
`bump.pdf` — every axis labeled, with units.

## Review

1. (Week 02) Rebuild the counting-dictionary pattern from memory; which
   pandas call does the same job?
2. (Week 02) A function that should return a list prints it instead — what
   does the caller get, and what is the fix?
3. (Week 01) Read this traceback bottom line and diagnose:
   `TypeError: can only concatenate str (not "float") to str` in a line
   building a report string.
4. (Week 02) What does `xs[2:5]` contain, and does modifying it change `xs`
   when `xs` is a list? (Contrast with this week's arrays.)
