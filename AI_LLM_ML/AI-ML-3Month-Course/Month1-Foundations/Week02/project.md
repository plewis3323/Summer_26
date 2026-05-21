# Week 02 — Mini-Project: BDT vs Logistic on HEP Tabular Data

## Problem

Train a signal/background classifier on a HEP tabular dataset. Benchmark a tuned XGBoost model against a properly-regularized logistic-regression baseline. Produce the standard evaluation suite (ROC, PR, calibration, feature importance, confusion matrix). Write the honest post-mortem.

## Data

Primary: **HEPMASS** 1000-particle subset. https://archive.ics.uci.edu/dataset/347/hepmass. ~10.5 M rows, 28 features, binary label. Download `1000_train.csv.gz` and `1000_test.csv.gz`; ~1 GB compressed.

Alternative: **Higgs Boson Challenge**. https://www.kaggle.com/c/higgs-boson/data. Richer feature set, 250k events, weight column, closer to a real LHC analysis.

If disk is tight, subsample HEPMASS train to 1 M rows. Report results at the subsampled size.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/hep_bdt/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py            # baseline and xgb
│   ├── eval.py              # metrics + plots
│   ├── tune.py              # optuna study
│   └── cli.py
├── configs/
│   └── default.yaml
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── notebooks/
│   └── 01_exploration.ipynb
└── results/
    ├── metrics.json
    ├── roc.png
    ├── pr.png
    ├── calibration.png
    └── feat_importance.png
```

## Acceptance criteria

1. `uv run pytest -q` passes.
2. `uv run python -m hep_bdt.cli train --config configs/default.yaml` trains both models and writes all artifacts to `results/`.
3. Tuned XGBoost test AUC > logistic baseline test AUC by at least 2 percentage points. (HEPMASS 1000 typically lets XGB reach ~0.95 vs logistic ~0.87 — plenty of margin.)
4. `metrics.json` includes AUC, PR-AUC, Brier, log-loss for both models, best hyperparameters, train/test sizes, feature list, and `git_sha`.
5. ROC, PR, and calibration plots overlay the two models with labeled legends.
6. Feature importance plot uses **permutation importance** on the test set, not gain-based.
7. Project `README.md` has a "Results" table, the figures, and a **"What went wrong"** section with at least two honest items (class imbalance, leakage check, feature scale issues, etc.).
8. `ruff check` and `ruff format --check` clean.

## Recommended sequence

1. **Day 1.** Scaffold the project; write `data.py` with streaming CSV loader (use `pandas.read_csv(chunksize=...)` if memory is tight, or convert once to Parquet). Write EDA notebook: feature distributions per class, label balance, feature correlations. Commit notebook with outputs cleared.

2. **Day 2.** Write baseline logistic with `sklearn.pipeline.Pipeline([StandardScaler(), LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000)])`. Full train + eval. Commit.

3. **Day 3.** XGBoost with default hyperparameters and early stopping. Compare to baseline. Commit.

4. **Day 4.** Optuna tuning. 30–50 trials, 5-fold CV, AUC objective. Persist study via SQLite so you can resume. Write `tune.py` as a CLI.

5. **Day 5.** Plots, calibration, feature importance. Writeup.

## Implementation notes

### `data.py`
```python
from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_hepmass(path: Path, n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=n) if n else pd.read_csv(path)
    # Convention: first column is label, rest are features
    df = df.rename(columns={df.columns[0]: "y"})
    return df
```

### `models.py` for XGBoost with early stopping
```python
from xgboost import XGBClassifier

def train_xgb(X_tr, y_tr, X_val, y_val, **hp) -> XGBClassifier:
    model = XGBClassifier(
        tree_method="hist",
        device="cuda" if _cuda_available() else "cpu",
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=50,
        **hp,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model
```

Pitfall: older XGBoost versions wanted `tree_method="gpu_hist"`; newer ones use `device="cuda"`. Check `xgboost.__version__` and the docs.

### `eval.py` for proper ROC plotting
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, preds: dict[str, np.ndarray], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for name, s in preds.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate (background mis-ID)")
    ax.set_ylabel("True positive rate (signal efficiency)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
```

HEP-flavored axis labels since those are the words your advisor will use.

### Optuna tuning sketch
```python
import optuna

def objective(trial: optuna.Trial) -> float:
    hp = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }
    cv_aucs = []
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train):
        m = train_xgb(X_train[tr], y_train[tr], X_train[va], y_train[va], **hp)
        cv_aucs.append(roc_auc_score(y_train[va], m.predict_proba(X_train[va])[:, 1]))
    return float(np.mean(cv_aucs))

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///results/optuna.db",
                            study_name="hepmass_xgb",
                            load_if_exists=True)
study.optimize(objective, n_trials=50, show_progress_bar=True)
```

### Why permutation importance
Gain-based importance is biased toward high-cardinality features and sensitive to correlated features being split between trees. Permutation importance (shuffle the test column; measure AUC drop) is model-agnostic and more interpretable to physics audiences.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
order = r.importances_mean.argsort()[::-1]
```

## Write-up template (`project/README.md`)

```
# HEPMASS BDT vs Logistic

## Problem
...

## Data
- Source, version, rows, features, class balance.

## Approach
- Baseline: logistic regression with StandardScaler, L2.
- Model: XGBoost, 50-trial optuna tuning, 5-fold CV.

## Results
| Model | Test AUC | Test PR-AUC | Brier | Log-loss |
| --- | --- | --- | --- | --- |
| Logistic | 0.872 | ... | ... | ... |
| XGBoost | 0.945 | ... | ... | ... |

(figures inlined)

## Caveats
1. Optuna budget was 50 trials; a 500-trial sweep would likely gain 0.5–1.0 points.
2. No adversarial de-biasing; features may encode run-period artifacts.
3. HEPMASS is a simplified benchmark; real LHC analyses have higher imbalance and weight columns.

## Reproducibility
- Commit: <sha>
- uv.lock committed.
- `make reproduce` runs end-to-end from raw CSV to final artifacts in ~30 min on a T4 or ~10 min on an A100.
```

## Common failure modes

- **AUC looks suspiciously high (>0.98).** Check for leakage. One-feature AUC scan.
- **Optuna converges to trivial solutions.** Your CV objective function isn't what you think. Print it.
- **XGBoost OOMs.** Reduce `max_bin`, switch to `tree_method="hist"` on CPU, or drop to 1 M rows.
- **Scaler fit on train + test.** `sklearn.pipeline.Pipeline` prevents this. Use it.
- **Random seed missed.** Every split, every model init, every optuna sampler — all seeded.

## Extensions (optional)

- Add **SHAP values** for XGBoost. `shap.TreeExplainer(model).shap_values(X_test[:5000])` → summary plot. Useful for thesis presentations.
- Replace XGBoost with LightGBM at identical hyperparameter budget and compare wall-clock.
- Repeat the whole study with `GroupKFold` on a synthesized group column. Show the AUC gap.

## Sign-off

When done, tick the Week 02 boxes in `04-Progress-Tracker.md` and push a clean repo. Next week we build PyTorch and micrograd — the gradient you see everywhere will suddenly be something you can compute by hand.
