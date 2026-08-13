# Month 01 — Programming from Zero to the Scientific Stack

The arc: go from never having written code to a working scientific-Python
pipeline you can rerun from a fresh clone. Week 01 is the terminal, Python
itself, and reading error messages calmly. Week 02 adds control flow, functions,
lists, dictionaries, and files — enough language to build a real program from
parts. Week 03 is why scientists use Python: NumPy arrays, matplotlib plots,
and pandas tables. Week 04 spends everything at once: git, tests, environments,
and a guided, tested rebuild of a CMS dimuon analysis on CERN Open Data —
including a first CI workflow and a sqlite results table.

Each week's exercise notebook is built when the week starts, per
`NOTEBOOK_RULES.md`.

**Month-end deliverable:** a small Python package (`src/` layout,
`pyproject.toml`, pinned deps, CI green) that downloads dimuon data, applies cuts, fits
the J/ψ peak, stores the cut-flow in sqlite, and produces the spectrum figure from one command, with
`pytest -q` green.

**Sign-off:** tag `month-01-complete`, write `retro.md` (250 words) in this
folder, and open one issue for the single biggest thing you don't understand
yet.
