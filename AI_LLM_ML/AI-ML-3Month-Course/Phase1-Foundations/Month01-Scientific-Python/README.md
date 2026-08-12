# Month 01 — Scientific Python & Software Engineering

The arc: convert ROOT/C++ analysis instincts into the scientific-Python idiom, then wrap
that idiom in professional engineering practice. Week 01 replaces the event loop with
array expressions (NumPy). Week 02 adds structured data and publication-grade plots
(pandas + matplotlib). Week 03 adds the toolchain that makes code trustworthy — git,
`uv`, `ruff`, `pytest`, `pdb`. Week 04 spends everything at once: a reproducible,
tested rebuild of a ROOT-style dimuon analysis on CERN Open Data.

Each week's exercise notebook is built when the week starts, per `NOTEBOOK_RULES.md`.

**Month-end deliverable:** a small Python package (`src/` layout, `pyproject.toml`,
pinned deps) that downloads dimuon data, applies cuts, fits a resonance peak, and
produces the spectrum figure from one command, with `pytest -q` green.

**Sign-off:** tag `month-01-complete`, write `retro.md` (250 words) in this folder,
and open one issue for the single biggest thing you don't understand yet.
