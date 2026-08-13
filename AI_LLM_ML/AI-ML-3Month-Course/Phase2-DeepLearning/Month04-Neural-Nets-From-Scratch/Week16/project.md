# Week 16 Project — MLP vs BDT on MAGIC

## Objective

Train a PyTorch MLP on the Capstone 1 MAGIC gamma/hadron table and compare
it to the Capstone 1 BDT under an **identical CV protocol**: same splits,
same outer folds, same seeds, comparable tuning budget, same metric (AUC),
test set touched once. The roadmap criterion (`03-Project-Roadmap.md`):
**match or beat the BDT AUC, or explain why tabular ≠ deep-learning home
turf.** A BDT win with a writeup that points at sample size, lack of
spatial structure, and the Week 11 argument is a *result*. An MLP "win"
from a leaked split or a BDT that was not given the same budget is not.
This closes Month 04.

## Background — the comparison, from scratch

**What you are classifying.** Same physics as Capstone 1 (`Week12/project.md`):
MAGIC images atmospheric Cherenkov flashes from gamma-ray and hadronic
showers, summarized as ten **Hillas** ellipse parameters. Ten heterogeneous,
correlated-by-construction numbers, $n\sim 2\times 10^4$, no pixel grid.
That is the regime Week 11 spent a section on: **tabular / ntuple**, BDT
home turf. Trees ignore feature scale and monotone reparameterization; an
MLP badly does not. If you skip the scaler, E1's histograms will scream
(`fSize` next to `fAlpha`).

**What "identical protocol" means.** Week 12 / Week 24's rule has not
changed because the model is now a net.

- Same frozen trainval/test split as Capstone 1 (load the indices, or
  recreate with the same `random_state=0`, `test_size=0.20`, `stratify=y`).
- Same outer `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)`
  for the number you quote.
- Comparable tuning budget: the BDT is *not* retuned against the MLP.
  Retrain it inside this repo with Capstone 1's winning hyperparameters
  (or load the frozen Capstone 1 pipeline as a callable that does not get
  a second tuning pass). The MLP gets a val-based LR / width / early-
  stopping search of similar wall-clock, not an unbounded raid.
- Same metric: ROC AUC. Test set opened once, at the end, for both
  models together.
- Logistic baseline first. If logistic already matches the BDT, the
  story is linear Hillas structure and the MLP-vs-BDT plot is a footnote.

**What you tune on the MLP side.** He init (ReLU), LR (read the
update-to-weight ratios from E1–E6; do not grid-search blind), depth/width
(start shallow — this is not ImageNet; `10 → 32 → 32 → 1` is a reasonable
first net), batchnorm or not (E6: BN widens workable LRs; on tabular you
choose $B$, so BN is legal), early stopping on validation loss. The
diagnostic plots — activation histograms at step 0 and after training,
gradient norms, update-to-weight ratios — belong in the writeup. They are
how you know the net *trained*, as opposed to how you know it *scored*.

**What you do not do.** Do not add features the BDT was denied. Do not
touch the test set to decide depth. Do not quote Capstone 1's nested-CV
AUC as the BDT number here without rerunning it on *this* code path.

## Data

Capstone 1's MAGIC table, nothing else.

- **Source:** UCI MAGIC Gamma Telescope (dataset 159), the same CSV
  Capstone 1 ingested. Prefer loading from the Capstone 1 repo's fetch
  (same checksum, same `PROVENANCE.md` rules) over a second download.
- **Columns:** the ten Hillas features and the `g`/`h` label, encoded
  `g=1`, `h=0` exactly as Capstone 1.
- **Split:** the Capstone 1 freeze — 80/20 trainval/test, stratified,
  `random_state=0`. If you cannot import the indices, recreate them with
  the same call and assert the first-row ids match a stored hash.
- **Outer folds:** `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=0)` on trainval, for both models.

No new physics data. No pixel images. Compute: laptop CPU is enough; a
small GPU is luxury. Budget one evening for MLP tuning, not a weekend.

## Build steps

Do them in order. E1–E6 of `exercises.md` are the diagnostic drills and
are not part of this repo; E7 *is* this project.

**Stage 0 — the repo (20 min).**

```
cd ~/course
uv init --package mlp_vs_bdt
cd mlp_vs_bdt
git init
uv add numpy pandas matplotlib scikit-learn xgboost torch
uv add --dev pytest
```

Target layout:

```
mlp_vs_bdt/
  pyproject.toml
  uv.lock
  run.py
  writeup.md
  src/mlp_vs_bdt/
    __init__.py
    data.py          load MAGIC + Capstone 1 split
    bdt.py           frozen-hyperparameter XGBoost pipeline
    mlp.py           nn.Module + train/eval loop
    protocol.py      outer-fold AUC for any scorer
    plots.py         ROC overlay, diagnostics
  tests/
    test_protocol.py
    test_mlp_trains.py
  results/
    metrics.json
  figures/
    reference/
```

**Stage 1 — data + protocol harness.** `data.py` loads X, y and the
frozen split. `protocol.py` takes a callable `fit_predict_proba(X_tr,
y_tr, X_te) -> p_te` and returns the five outer-fold AUCs. Hand-check it
on a dummy scorer (`y_te.mean()` as a constant score) where you know the
AUC is 0.5. `tests/test_protocol.py`: dummy scorer gives AUC
`pytest.approx(0.5, abs=0.02)` on MAGIC's outer folds; split sizes match
Capstone 1.
*Accept when: the harness runs on a dummy model and the split assertion
passes.*

**Stage 2 — BDT under this harness.** `bdt.py`: the Capstone 1 winner's
hyperparameters inside a `Pipeline(StandardScaler, XGBClassifier(...))`,
**no new grid**. Run it through `protocol.py`. Write the five AUCs and
the mean ± spread to `results/metrics.json` under `"bdt"`. This number is
the baseline the MLP has to face — produced here, not pasted from Week 12.
*Accept when: BDT outer-fold mean AUC is in `metrics.json` and is within
0.01 of your Capstone 1 nested-CV XGBoost mean (same protocol; if it
isn't, you loaded the wrong split or the wrong hyperparameters — fix that
before touching the MLP).*

**Stage 3 — MLP.** `mlp.py`: a plain `nn.Module`, ReLU, He init, optional
`BatchNorm1d`, binary logit + `BCEWithLogitsLoss` (or two-class softmax +
cross-entropy; pick one and stay). `StandardScaler` fit on the *training
fold only*, applied to val/test. Adam (Week 08 / `torch.optim.Adam`),
early stopping on val loss, seed everything (`torch.manual_seed(0)`,
generator on the DataLoader). Tune LR / width / BN on a *single* inner
val split of trainval, then freeze those choices and run the frozen net
through the *same* outer `protocol.py`. Do not retune per outer fold
unless you also retune the BDT that way — you will not.
Log, every few steps: per-layer activation histograms, gradient norms,
update-to-weight ratios. Save a diagnostics figure from the final trainval
refit.
`tests/test_mlp_trains.py`: on a 200-row toy slice, one epoch runs, loss
is finite, `predict_proba` returns shape `(n,)` in $(0,1)$.
*Accept when: the MLP produces five outer-fold AUCs by the same harness,
and the diagnostics figure exists.*

**Stage 4 — head-to-head + test once.** One table: logistic (optional but
recommended), BDT, MLP — outer-fold mean ± spread. Then *one* test-set
evaluation of the two (three) models refit on all of trainval. ROC overlay
on test. Write `results/metrics.json` completely. If the MLP loses, the
writeup's job is the why-not, not a second architecture hunt.
*Accept when: the table exists with a win or a defensible why-not, and
the test set was touched once.*

**Stage 5 — writeup.** `writeup.md`, one page:

- The identical-protocol claim (splits, folds, seeds, budget).
- Headline table.
- Diagnostics: did the net train? (histograms, update ratios $\sim 10^{-3}$,
  no dead-ReLU epidemic at the chosen LR.)
- Why-not (if you lost): sample size, no spatial layout, correlated
  Hillas, trees' scale invariance — point at Week 11, not at "needs more
  epochs" unless the diagnostics show it actually underfit.
- **What failed first.**
- What you would do next with one more week (and why you did not do it
  to chase the BDT).

Every number from `metrics.json`.
*Accept when: the writeup exists, the failure section is there, and a
loss is explained rather than hidden.*

**Stage 6 — one command.** `run.py`: data → BDT protocol → MLP protocol →
test-once → figures → `metrics.json`. Fresh clone + `uv sync` +
`uv run pytest -q` + `uv run python run.py`.

## Acceptance gate (from `03-Project-Roadmap.md`)

**Match or beat the BDT AUC, or explain why tabular ≠ deep-learning home
turf.** Concretely:

- Fresh clone + `uv sync` + `uv run pytest -q` green +
  `uv run python run.py` reprints the table.
- Identical protocol: same split, same outer folds, BDT not retuned
  against the MLP, test set once.
- Headline number: outer-fold mean AUC, MLP vs BDT, with fold spread.
- If MLP ≥ BDT: the win survives the test-set row (no outer-fold-only
  miracle). If MLP < BDT: the writeup's why-not is specific (diagnostics
  + Week 11 argument), not "deep learning needs more data" as a slogan.
- Diagnostic plots in `figures/`. `writeup.md` includes what failed first.

Then the month sign-off: tag `month-04-complete`, write `retro.md` in the
Month 04 folder, open one issue. Re-derive Week 13's $\delta^{(l)}$
recursion cold (the month's flagship) and file it.

## Writeup requirements

Covered in Stage 5. Repo `README.md` is the operator's note: three
commands, pointer to `writeup.md`, the headline sentence (won / lost, by
how much).

## Stretch goals (only after the gate)

- **BN vs no-BN on MAGIC.** E6's LR-range plot, now on Hillas; does BN
  still widen the workable LRs when $B$ is 256 and $d=10$?
- **Depth sweep.** 1 / 2 / 4 hidden layers at fixed width; outer-fold AUC
  vs depth. The bet: more depth does not help.
- **Logistic on the same harness.** If it lands within 0.01 of the BDT,
  the MLP-vs-BDT plot is a footnote and you should say so in the first
  paragraph.
- **Permutation importance on the MLP** (input-shuffle, not weight
  magnitudes) vs the BDT, with the correlated-Hillas caveat.
