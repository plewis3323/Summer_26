# Week 02 — Resources

## Docs

- **scikit-learn User Guide.** https://scikit-learn.org/stable/user_guide.html. Read "Supervised learning → Linear Models" and "Ensemble methods" sections. These are better than most textbooks on their specific topics.
- **XGBoost docs.** https://xgboost.readthedocs.io/. Key pages: "Parameters" (https://xgboost.readthedocs.io/en/stable/parameter.html), "Python API Reference", "Notes on Parameter Tuning".
- **LightGBM docs.** https://lightgbm.readthedocs.io/. "Parameters Tuning" page.
- **Optuna docs.** https://optuna.readthedocs.io/. "Key Features" tutorials; "Pruners" for mid-trial termination.
- **SHAP docs.** https://shap.readthedocs.io/. "An introduction to explainable AI with Shapley values."

## Tutorials and blog posts

- **Laurae++ XGBoost tuning guide.** https://sites.google.com/view/lauraepp/parameters. Gnarly but definitive.
- **Chris Albon — Machine Learning Flashcards.** https://machinelearningflashcards.com/. Overpriced, but the free samples are good review material.
- **Andrew Ng — "Machine Learning Yearning" (free book).** https://github.com/ajaymache/machine-learning-yearning. Short chapters on bias/variance and error analysis. Read after E4.

## Videos

- **StatQuest — Gradient Boosting Part 1–4.** https://www.youtube.com/watch?v=3CC4N4z3GJc. Good pairing with the Friedman paper.
- **StatQuest — XGBoost Part 1–4.** Clear numerical walkthrough of the XGBoost objective.
- **Scott Lundberg — "Explainable AI with SHAP" (talks).** YouTube. The creator of SHAP explaining it, with HEP-adjacent examples.

## GitHub repos

- **`dmlc/xgboost`.** Canonical. Worth browsing the Python examples in `demo/`.
- **`microsoft/LightGBM`.** Similar.
- **`optuna/optuna-examples`.** Copy-paste-able tuning scripts.
- **`scikit-hep/iminuit`.** For physicists who want `MINUIT`-style minimization with gradients from autograd. You'll use it in Month 1 extensions.
- **`scikit-hep/mplhep`.** Publication-quality HEP plot styles for matplotlib. One import and your plots look like ATLAS/CMS.

## Codecademy (Pro) courses already in Reading List §E

- "Build a Machine Learning Model" — modules 3–6.
- "Feature Engineering" — all modules.

## HEP-specific

- **CERN Open Data Portal — Higgs boson challenge data.** https://opendata.cern.ch/record/328. Use this if you prefer real HEP to HEPMASS.
- **TMVA Users Guide.** https://root.cern/tmva/. The BDT reference your collaboration almost certainly uses under the hood. Worth skimming §8 on BDTs to see the vocabulary match.
- **CMS DeepCSV / DeepJet papers.** https://arxiv.org/abs/1712.07158 and follow-ups. Real-world BDT-to-DL transition in a flagship LHC tagger. Will reappear in Week 05.

## Datasets referenced in exercises

- **HEPMASS.** https://archive.ics.uci.edu/dataset/347/hepmass.
- **Higgs Boson ML Challenge (Kaggle).** https://www.kaggle.com/c/higgs-boson.
- **CovType.** Available in `sklearn.datasets.fetch_covtype()`.
- **Wine.** `sklearn.datasets.load_wine()`.
- **make_classification synthetic.** `sklearn.datasets.make_classification(...)`.

## Tooling

- **`mplhep`** — `uv add mplhep`. Then:
  ```python
  import mplhep
  mplhep.style.use("ATLAS")  # or "CMS"
  ```
  Use this starting Week 02 for any figure you'd show your advisor.
- **`jupytext`** — https://jupytext.readthedocs.io/. Pair notebooks with plain `.py` files for git-friendliness.
- **`wandb`** — https://wandb.ai/. Free for academic. Starting Week 04 you'll log every experiment here.

## When you get stuck

- **Cross-validated** (Stats StackExchange) — better than SO for statistical issues.
- **XGBoost GitHub issues** — searchable. Most gotchas are already reported.
- **HEP Software Foundation forums.** https://www.hepsoftwarefoundation.org/. More DIY-friendly than collaboration Mattermosts for pure-ML questions.

## Further reading if you finish early

- Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*, Ch. 9 (GAMs) and Ch. 10 (boosting). Free at https://hastie.su.domains/ElemStatLearn/. Denser than Bishop; different emphasis.
- Bühlmann & van de Geer — *Statistics for High-Dimensional Data*. For L1/L2 theory if you care about the probabilistic guarantees.

---

When Week 02 is on GitHub and boxes are ticked, move to Week 03. In Week 03 you will implement autograd from scratch, and you will be glad you already know what "gradient" means in three different framings.
