# Week 04 — Exercises

Mini-project week: the exercises *are* the six project stages, and `project.md`
guides every one of them in detail — read it before starting E1. All six live
outside the notebook, in your `dimuon/` package repo: each names files, and each
acceptance criterion is `pytest` or a command line (per `NOTEBOOK_RULES.md` §6).
Use notebook cells only the way the lesson prescribes — prototype a cut or a fit
interactively, then promote it into `src/` — and let the notebook shell out to
`pytest -q` and `run.py` for the checks.

## E1 — Data fetch with provenance

Write `src/dimuon/fetch.py`: download the CMS dimuon CSV from CERN Open Data
into `data/`, verify the recorded SHA-256 checksum (the constant is in
`project.md` §Data), skip the download when a verified copy is already present,
and reject a corrupted file. Also write `data/PROVENANCE.md` recording source
URL, record number, date fetched, and checksum.
Hint: `urllib.request.urlretrieve` downloads in one call;
`hashlib.sha256(...).hexdigest()` fingerprints the file's bytes; `raise
ValueError(...)` on mismatch — Stage 1 of `project.md` shows every piece.
Accept when: rerun skips the download (cached) and a corrupted file is rejected
with a clear message.

## E2 — Selection module

Port the cut flow — the mask cuts of Week 02 E3 and Week 03 E3, now on real
muons — into `src/dimuon/select.py`: both muons global, opposite charge, both
`pt` above threshold, both `|eta|` in range. All cut values live in one `CUTS`
dictionary at the top; one function per cut plus `apply_cuts` returning the
selected events and the cut-flow table; unit tests on a tiny hand-built
DataFrame where you know each answer by inspection.
Hint: each cut function returns a boolean mask; `apply_cuts` chains them with
`&`, recording `int(mask.sum())` after each. Stage 2 of `project.md` lists the
five reference counts (100,000 → 38,614).
Accept when: cut-flow counts match Week 02 (the counts you get applying the
same cuts by hand in a notebook — the Stage 2 reference numbers) and tests
cover each cut in isolation.

## E3 — Peak fit

Write `src/dimuon/fit.py`: histogram the selected masses in the J/ψ window
(2.8–3.4 GeV, 60 bins) and fit Gaussian signal + linear background with
`scipy.optimize.curve_fit`, starting from the stated initial values `P0`.
Report mass and width with uncertainties (square roots of the covariance
diagonal). Test it on seeded synthetic data with a known peak before pointing
it at the real file.
Hint: fit bin *centers* vs bin *counts*; `P0 = [800.0, 3.1, 0.03, 50.0, 0.0]`
— Stage 3 of `project.md` derives the model and states the reference result.
Accept when: fitted J/ψ mass is within 30 MeV of the PDG value and the fit
converges from the stated seed/initial values every run.

## E4 — Figure module

Write `src/dimuon/plots.py`: (a) the full spectrum, 2–110 GeV on log–log axes
— four resonances visible; (b) the J/ψ zoom with the fitted curve overlaid and
the fitted mass in the title. Paper grade: every axis labeled with units. Both
saved as PDF by functions the pipeline calls; once you're happy, commit copies
under `figures/reference/`.
Hint: `np.logspace` bins for (a); for (b), evaluate the model on
`np.linspace(2.8, 3.4, 300)` and `ax.plot` it over the histogram.
Accept when: both PDFs are produced by the pipeline, not by hand, and match
committed reference images by eye.

## E5 — One-command run

Write `run.py` at the repo root: fetch → select → fit → figures, printing the
cut-flow table and the fitted mass ± uncertainty next to the PDG value.
Orchestration only — every line calls a tested function from `src/dimuon/`.
Hint: the lesson's §7 pattern; if `run.py` contains a formula or a cut, it is
in the wrong file.
Accept when: fresh clone + `uv sync` + `uv run python run.py` completes and
prints the fitted mass.

## E6 — Determinism check

Run the pipeline twice, capturing all printed output
(`uv run python run.py > run1.txt`, then again into `run2.txt`), and
`diff run1.txt run2.txt`. Explain in one sentence in your repo README *why* it
is deterministic — which of the three usual suspects (randomness, versions,
by-hand steps) is eliminated by what.
Hint: an empty `diff` is the pass; diff the printed numbers, not the PDF bytes
(PDFs embed timestamps).
Accept when: two runs produce identical fit parameters and identical cut-flow
tables.

## Review

1. (Wk 01) The mass array for 10⁶ events at float64 occupies how much memory?
   Why might float32 be fine here and not in the fit covariance?
2. (Wk 02) Which join type did the run-quality merge need, and what silent
   error would the wrong one cause?
3. (Wk 03) What exactly does `uv.lock` pin that `pyproject.toml` alone does
   not?
4. (Wk 03) Your regression test from the pdb exercise: what bug does it pin
   down, from memory?
5. (Physics) The provenance file plays the role of what in a collaboration
   analysis (think: dataset bookkeeping / good-run lists)?
