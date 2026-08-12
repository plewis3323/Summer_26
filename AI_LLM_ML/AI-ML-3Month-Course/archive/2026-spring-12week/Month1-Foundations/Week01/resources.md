# Week 01 — Resources

Curated. Not exhaustive. Bookmark, don't binge.

## Docs (primary references you'll return to)

- **NumPy docs.** https://numpy.org/doc/stable/. The user guide is the actually useful part. The "Broadcasting" page especially.
- **pandas docs.** https://pandas.pydata.org/docs/. "Essential basic functionality" and "User Guide → Group by" are the two you'll consult most.
- **matplotlib docs.** https://matplotlib.org/stable/. The "Tutorials → Intro" and "Gallery" pages are the fastest way to find a plot type you want.
- **scipy.optimize.curve_fit.** https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html.
- **pytest docs.** https://docs.pytest.org/. "How to invoke pytest" and "How to use fixtures."
- **uv docs.** https://docs.astral.sh/uv/. "Project → Creating projects" and "Resolvers and locking."
- **ruff docs.** https://docs.astral.sh/ruff/.
- **uproot docs.** https://uproot.readthedocs.io/. For ROOT ↔ numpy.

## Tutorials and blog posts

- Jake VanderPlas — *Python Data Science Handbook* (free online). https://jakevdp.github.io/PythonDataScienceHandbook/. Already on the reading list; this entry is just the URL.
- Real Python — "Python f-Strings". https://realpython.com/python-f-strings/.
- Real Python — "Python Type Checking". https://realpython.com/python-type-checking/.
- Sebastian Raschka — "Python Machine Learning, 3rd Ed." sample notebooks on GitHub. https://github.com/rasbt/python-machine-learning-book-3rd-edition. Ch. 1–2 are Week-01-level.
- Chris Albon — numpy/pandas cheat sheets. https://chrisalbon.com/. Useful for quick syntax lookup.

## Videos

- **3Blue1Brown — Neural Networks playlist.** Already linked in `reading.md`.
- **Corey Schafer — pandas tutorials.** https://www.youtube.com/watch?v=ZyhVh-qRZPA (playlist). If you need a gentler walkthrough of pandas than VanderPlas.
- **Raymond Hettinger — "Beyond PEP 8 -- Best practices for beautiful intelligible code" (PyCon 2015)**. https://www.youtube.com/watch?v=wf-BqAjZb8M. A taste of Pythonic style.

## GitHub repos (to read, not clone-and-run)

- **`numpy/numpy`.** Reading `numpy/core/src/multiarray/arrayobject.c` when you want to know what's actually happening.
- **`pandas-dev/pandas`.** For understanding why `df.iloc` vs `df.loc` matters.
- **`matplotlib/matplotlib/examples`.** Best plot-type search engine there is.
- **`scikit-hep/uproot5`.** If you want to know how ROOT files parse in pure Python.
- **`scikit-hep/awkward`.** Jagged-array library from the HEP Python community. You'll use it in Month 2 for variable-length event structures.

## Codecademy (Pro)

- "Build a Machine Learning Model" — modules 1–2. (Already assigned.)
- "Introduction to Python 3" — only if you're rusty. Skip if you can write a decorator without looking it up.
- "Data Visualization with Python" — optional Week 01 filler if you want more matplotlib practice.

## HEP-adjacent

- **scikit-hep.** https://scikit-hep.org/. The umbrella for the Python-for-HEP ecosystem. Familiarize yourself with `uproot`, `awkward`, `hist`, `vector`.
- **`hist` package.** https://hist.readthedocs.io/. A pythonic histogramming library from the HEP Python group. Think `TH1` with a saner API.
- **`vector` package.** https://vector.readthedocs.io/. Lorentz-vector arithmetic for numpy/awkward arrays. Use this instead of rolling your own in E3.
- **CoDaS-HEP summer school materials.** https://codas-hep.org/. Excellent lecture notes on Python, numpy, and HEP workflow hygiene. Several PDFs worth stashing.

## Tooling goodies

- **`rich`** — https://rich.readthedocs.io/. Prettier Python terminal output. Useful for debugging colors.
- **`ipdb`** — pdb with better ergonomics. `uv add --dev ipdb`, then `breakpoint()` works.
- **`pre-commit`** — https://pre-commit.com/. Set this up now; thank yourself in Week 04.

## HEP data portals

- **CERN Open Data.** https://opendata.cern.ch/. Start here for real LHC data in Week 01–02.
- **CMS Open Data workshop.** https://cms-opendata-workshop.github.io/. Step-by-step recipes.
- **ATLAS Open Data.** https://opendata.atlas.cern/.
- **STAR / sPHENIX.** Internal data portals; use your collaboration credentials.

## When you get stuck

- **Stack Overflow** is still better than any LLM for syntax questions with an obvious canonical answer.
- **`#scikit-hep` on Gitter** (or their Discord now) — friendly and active community.
- **`pandas` user guide** for "how do I do X in pandas" — it's better than random blog posts of varying vintage.

---

If you finish Week 01 with time left over, read Bishop §1.5 (decision theory) and §2.1–2.2 (probability distributions). That carries into Week 02.
