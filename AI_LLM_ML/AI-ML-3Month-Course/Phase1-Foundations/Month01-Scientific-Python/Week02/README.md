# Week 02 — pandas + Visualization

An ntuple is just a DataFrame with worse ergonomics; this week you learn the DataFrame's
ergonomics and how to make figures you would actually put in a paper.

## Objectives

- Load, filter, and transform tabular event data in pandas without dropping to loops.
- Use `groupby`/aggregate and `merge` to answer per-run and cross-table questions.
- Reshape between wide and tidy ("long") layouts and say why tidy wins for analysis.
- Build a matplotlib figure from the object API (`fig, ax`) — no `plt.plot` autopilot.
- Produce one publication-quality figure: labeled axes with units, legible fonts, vector output.

## Core material (~3 hrs)

- VanderPlas, *Python Data Science Handbook*, Chapter 3 (pandas) — prioritize indexing,
  `groupby`, and `merge`; skim the rest.
- VanderPlas, Chapter 4 (matplotlib) — through subplots, ticks, and legends.
- Hadley Wickham's "Tidy Data" paper (skim §2–3): the vocabulary for why one layout is
  easier to compute on than another.
- Matplotlib docs: the "Anatomy of a figure" page. Keep it open while plotting.

## Exercises (built when the week starts)

1. **CSV → DataFrame.** Load a provided dimuon event CSV; inspect dtypes, missing values,
   memory use; fix any mis-parsed columns.
   Accept when: `df.dtypes` matches the stated schema and reported row count is exact.
2. **Cut flow table.** Reproduce Week 01's mask-based cuts in pandas and produce a
   cut-flow DataFrame (cut name, events surviving, efficiency).
   Accept when: survivor counts match the Week 01 NumPy result exactly.
3. **Per-run aggregates.** `groupby` run number: events, mean pT, mass-window yield per run.
   Accept when: the aggregate table matches a provided reference row for one spot-check run.
4. **Run-quality merge.** Merge a run-quality table and drop bad runs; count what an
   inner vs left join each would have kept.
   Accept when: post-merge event count matches the reference and the join-type difference is stated in one line.
5. **Wide → tidy.** `melt` a wide table of per-run yields in three mass windows into tidy
   form and aggregate it both ways.
   Accept when: tidy table has exactly 3× the rows and identical totals.
6. **Paper figure.** Dimuon mass spectrum: log-y histogram, labeled resonances, axis units,
   ATLAS/STAR-style clean styling, saved as PDF.
   Accept when: `results/dimuon_spectrum.pdf` exists, axes carry units, and every text element is ≥8 pt at final size.

## Deliverable

Completed exercise notebook plus `results/dimuon_spectrum.pdf` — a figure you would not
be embarrassed to show your working group.

## Review

1. (Wk 01) Predict the broadcast result shape of `(N,1) * (3,)` — and what pandas
   operation does the same alignment by *label* instead of position?
2. (Wk 01) When does slicing a DataFrame give a view vs a copy, and why is the answer
   messier than NumPy's? (The `SettingWithCopyWarning` question.)
3. (Wk 01) Write the one-line vectorized invariant-mass expression from memory.
4. (Physics) A per-run `groupby` is the software twin of what standard detector-operations
   practice?
