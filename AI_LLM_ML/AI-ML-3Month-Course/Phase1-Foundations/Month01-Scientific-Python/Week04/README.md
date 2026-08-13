# Week 04 — Reproducibility + Mini-Project (SWE I)

"Works on my machine" is the software version of an unrepeatable measurement; this week
you ship an analysis that reruns from a fresh clone with one command — and that a
machine, not just you, can check.

## Objectives

- State what full reproducibility requires: pinned deps, seeds, data provenance, one entry point.
- Control every random seed in a pipeline and demonstrate bit-identical reruns.
- Record data provenance: where the file came from, its checksum, when it was fetched.
- Open a pull request, and put `pytest` in GitHub Actions so every push is checked.
- Store a results table in sqlite (a database that is one file) so later weeks can query it.
- Use Big-O as vocabulary: say why a nested loop over 10⁵ rows is a problem and a vectorized mask is not.
- Ship the mini-project: a tested Python package rebuilding a ROOT-style dimuon analysis.

## Core material (~3 hrs)

- Sandve et al., "Ten Simple Rules for Reproducible Computational Research"
  (PLOS Computational Biology). Short; read all ten rules.
- CERN Open Data portal: the CMS dimuon spectrum educational dataset — read the record
  page and understand what the columns are before touching code.
- NumPy docs: "Random sampling" — specifically `np.random.default_rng` and why the
  legacy global seed is deprecated practice.
- Revisit *Pro Git* tagging (Chapter 2) for the month sign-off.
- GitHub Actions quickstart — enough to recognize the workflow file you will copy.
- SQLite as an application file format (official page, short) — why a `.db` file beats a pile of CSVs for results you will query later.

## Exercises (built when the week starts)

Mini-project week: the exercises are the project stages.

1. **Data fetch with provenance.** Script downloads the dimuon CSV, verifies a recorded
   SHA-256 checksum, writes `data/PROVENANCE.md` (source, date, checksum).
   Accept when: rerun skips the download (cached) and a corrupted file is rejected with a clear message.
2. **Selection module.** Port the Week 02 cut flow into `src/dimuon/select.py`, config
   values in one place, unit-tested.
   Accept when: cut-flow counts match Week 02 and tests cover each cut in isolation.
3. **Peak fit.** Fit the J/ψ region (Gaussian signal + smooth background) with
   `scipy.optimize.curve_fit`; report mass and width with uncertainties.
   Accept when: fitted J/ψ mass is within 30 MeV of the PDG value and the fit converges from the stated seed/initial values every run.
4. **Figure module.** Regenerate the full spectrum + a J/ψ zoom with fit overlay as
   paper-grade PDFs.
   Accept when: both PDFs are produced by the pipeline, not by hand, and match committed reference images by eye.
5. **One-command run.** `run.py` executes fetch → select → fit → figures end to end.
   Accept when: fresh clone + `uv sync` + `uv run python run.py` completes and prints the fitted mass.
6. **Determinism check.** Run the pipeline twice; diff all numerical outputs.
   Accept when: two runs produce identical fit parameters and identical cut-flow tables.
7. **CI + sqlite results.** Write the cut-flow and fit into a sqlite file; add a
   GitHub Actions workflow that runs `uv sync` + `pytest` on every push.
   Accept when: the workflow badge is green on `main`, and `SELECT` from the db
   returns the same cut-flow counts as the printed table.

## Deliverable

The Month 01 deliverable: `dimuon/` package with pinned env, provenance file, tests
green, CI green, a sqlite results table, and a one-command reproduction of the
spectrum and fit. Then do the month sign-off (tag `month-01-complete`, `retro.md`,
one open-question issue).

## Review

1. (Wk 01) The mass array for 10⁶ events at float64 occupies how much memory? Why might
   float32 be fine here and not in the fit covariance?
2. (Wk 02) Which join type did the run-quality merge need, and what silent error would
   the wrong one cause?
3. (Wk 03) What exactly does `uv.lock` pin that `pyproject.toml` alone does not?
4. (Wk 03) Your regression test from the pdb exercise: what bug does it pin down, from
   memory?
5. (Physics) The provenance file plays the role of what in a collaboration analysis
   (think: dataset bookkeeping / good-run lists)?
6. Why does CI catch a broken test that `git push` alone does not?
