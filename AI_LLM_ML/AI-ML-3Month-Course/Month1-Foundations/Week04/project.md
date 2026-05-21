# Week 04 — Month 1 Capstone: Single-Photon vs Merged-π⁰ EMCal Shower Classifier

## Problem

Train a CNN to distinguish single-photon EMCal clusters from merged-π⁰ clusters. Two nearby photons from a π⁰ → γγ decay can merge into a single reconstructed cluster at high pT, where the opening angle drops below EMCal tower granularity. Distinguishing them from a true single photon is a real analysis need.

This is the Month-1 capstone. The goal is not a paper-worthy number; it is a clean end-to-end pipeline, an ROC curve, calibration, physics-aware failure analysis, and a reproducible repo.

## Data

Three valid paths, in decreasing order of your effort:

### Path A (recommended): simulate with the E10 generator from exercises.md
2D η×φ images of size 9×9 towers. Two classes:
- `photon`: single 2D Gaussian, σ = 0.020–0.025 rad, E ∈ (2, 30) GeV.
- `pi0_merged`: two Gaussians, opening angle ∈ (0.005, 0.05) rad, asymmetry ∈ [0, 0.8].

Add: position jitter within central 3×3, additive Poisson noise, possibly a rotated 5% tower-gain miscalibration map. Generate 100 k train / 20 k val / 20 k test.

Balance by kinematics: stratify train/val/test so each split spans the energy and opening-angle distributions evenly.

### Path B: sPHENIX / STAR MC (if you have access)
Use your collaboration's official single-photon and π⁰ MC samples. Project into 9×9 η×φ images at EMCal granularity. More realistic, more work.

### Path C: CERN HGCAL open data
https://opendata.cern.ch/. Small subset. Requires domain adaptation. Do this only if you want extra challenge.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── configs/
│   └── default.yaml
├── src/emcal_cnn/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py      # E10 logic, promoted
│   │   └── dataset.py        # torch Dataset
│   ├── models/
│   │   ├── tiny.py           # small custom CNN
│   │   └── resnet.py         # ResNet-18 adapted
│   ├── train.py              # training loop
│   ├── eval.py               # metrics, plots, failure analysis
│   └── cli.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_train.py
├── scripts/
│   ├── generate_data.py
│   └── train.sh
└── results/
    ├── loss.png
    ├── roc.png
    ├── calibration.png
    ├── gradcam_examples.png
    ├── failures_by_angle.png
    └── metrics.json
```

## Acceptance criteria

Hard requirements:

1. `uv run pytest -q` passes.
2. `uv run python scripts/generate_data.py --n 100000 --seed 42 --output data/train.pt` produces a deterministic dataset.
3. `uv run python -m emcal_cnn.cli train --config configs/default.yaml` trains the model, logs to wandb (or a local JSONL fallback), saves checkpoints, and produces all figures.
4. Test ROC AUC ≥ 0.85.
5. `metrics.json` includes AUC, PR-AUC, ECE (before and after temperature scaling), final train/val/test loss, architecture, hyperparameters, `git_sha`.
6. `results/failures_by_angle.png` stratifies errors by π⁰ opening angle. The plot must show that errors concentrate below some angle (you'll see where).
7. Grad-CAM figure for at least 4 test examples (2 photons + 2 π⁰s), correctly showing attention on the shower(s).
8. Project `README.md` has:
   - Problem and physics motivation (3–4 sentences).
   - Data description and generation recipe (or path to it).
   - Model architecture (one diagram or code excerpt).
   - Results table.
   - Figures inline.
   - **"Two physical failure modes" section** — required.
   - Reproducibility statement.

## Suggested timeline (5 days after normal reading)

- **Day 1**: scaffold repo, port the data generator, write dataset/tests, smoke-train a tiny CNN on 1000 events for 2 epochs (overfit it; confirm loss goes to zero).
- **Day 2**: full dataset generation, train the tiny CNN on it, log to wandb. Target a baseline AUC.
- **Day 3**: train ResNet-18-adapted (or a small custom CNN with BN, residual connections). Tune LR via OneCycle; 20–40 epochs.
- **Day 4**: calibration, Grad-CAM, failure stratification.
- **Day 5**: writeup and clean-up.

## Implementation notes

### `emcal_cnn/data/generator.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GenCfg:
    n: int = 100_000
    n_towers: int = 9
    eta_width: float = 0.024
    energy_range: tuple[float, float] = (2.0, 30.0)
    min_opening: float = 0.005
    max_opening: float = 0.050
    asymmetry_max: float = 0.8
    noise_per_gev: float = 0.3
    seed: int = 42

def _gaussian2d(x, y, cx, cy, sigma, amp):
    return amp * np.exp(-0.5 * (((x - cx)**2 + (y - cy)**2) / sigma**2))

def _make_single_photon(E, cx, cy, sigma, grid):
    # ...
    pass

def generate(cfg: GenCfg):
    # returns (images[N, 1, H, W] float32,
    #         labels[N] int64 (0=photon, 1=pi0),
    #         meta[N] structured)
    ...
```

Keep the generator fast. Vectorize with numpy; avoid per-event Python loops. You'll regenerate data several times.

### `emcal_cnn/models/tiny.py`

A 3-block CNN sized for 9×9 input. Not much room for pooling.

```python
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, hidden=32, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.BatchNorm2d(hidden * 2), nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 3, padding=1), nn.BatchNorm2d(hidden * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden * 4, n_classes)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))
```

### Failure stratification plot

```python
def plot_error_vs_angle(meta_test, y_true, y_pred, out):
    df = pd.DataFrame({"opening_angle": meta_test["opening_angle"],
                       "asym": meta_test["asym"],
                       "correct": y_pred == y_true,
                       "class": y_true})
    pi0 = df[df["class"] == 1]
    bins = np.linspace(pi0.opening_angle.min(), pi0.opening_angle.max(), 11)
    grouped = pi0.groupby(pd.cut(pi0.opening_angle, bins))["correct"].mean()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    grouped.plot.bar(ax=ax)
    ax.set_xlabel("π⁰ opening angle [rad]")
    ax.set_ylabel("Classification accuracy")
    fig.savefig(out, dpi=160)
```

You will see accuracy drop sharply as opening angle approaches or falls below tower pitch. That's the physical failure mode that belongs in your writeup.

### Temperature scaling
```python
def fit_temperature(logits_val, y_val) -> float:
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_val / T, y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.item())
```

## Writeup template for `project/README.md`

```
# EMCal Single-Photon vs Merged-π⁰ Classifier

## Physics motivation
At high pT, π⁰ → γγ decay photons can merge into a single EMCal cluster.
Distinguishing merged π⁰ from true prompt photons is important for ...

## Data
Synthetic, 100 k train / 20 k val / 20 k test. Generator specified in `src/emcal_cnn/data/generator.py` with seed 42.

## Model
TinyCNN (32 → 64 → 128 → GAP → Linear), BatchNorm + ReLU throughout.
Trained with Adam(lr=1e-3), OneCycleLR, 40 epochs, batch 128, label smoothing 0.05.

## Results
| Model | AUC | PR-AUC | ECE (raw) | ECE (T-scaled) | T |
| --- | --- | --- | --- | --- | --- |
| Logistic on shower-shape BDT features (baseline) | 0.803 | ... | ... | ... | — |
| TinyCNN | 0.927 | ... | ... | ... | 1.18 |
| ResNet-18 (adapted) | 0.943 | ... | ... | ... | 1.31 |

(ROC, calibration, failures-by-angle, Grad-CAM figures inline.)

## Physical failure modes
1. Below opening angle ≈ tower pitch (0.0244 rad for 9×9 EMCal), merged π⁰ images become visually indistinguishable from single photons. Accuracy drops from ~95% at Δθ = 0.04 rad to ~60% at 0.01 rad.
2. At high asymmetry (|E1 − E2|/(E1+E2) > 0.6), the sub-leading shower's contribution is swamped by noise; classifier regresses to "single photon." Could be mitigated by including a timing or layer-depth channel in future.

## Reproducibility
- Commit: <sha>
- uv.lock committed.
- ~25 min on a RTX 3060; ~8 min on an A100; ~90 min on CPU.
```

## Common failure modes during this project

- **AUC stuck at 0.5.** Label swap bug or data-label shape mismatch. Print label balance; check one training image and label by hand.
- **Val AUC >> test AUC.** Split leakage. Ensure splits happen at event-generation time, not after shuffling.
- **Training is very slow.** Your data loading is the bottleneck. Profile with `torch.profiler` or simply time a few batches; if >50% of wall-clock is in `__getitem__`, move the preprocessing to generation time or add workers.
- **Loss NaN after a handful of steps.** Check input range (if towers span 0–1000 GeV, you must normalize). `(x - x.mean()) / x.std()` per-tower or per-dataset.
- **Temperature-scaled model is worse.** Your calibration set is too small, or the model is underconfident (rare). Fit T on ≥5000 points.

## Extensions (optional; don't let them derail the capstone)

- Compare to a BDT trained on hand-engineered shower-shape features. This is the "CNN earns its place" experiment.
- Add a third class: electron. Physically justified (bremsstrahlung changes shower morphology).
- Try a Vision Transformer (tiny, 2 blocks). It'll probably lose to the CNN at this input size; that's informative.

## Sign-off

When done:
1. Tag commit `month-1-complete`.
2. Write a 250-word retrospective in `Month1-Foundations/retro.md`:
   - What was harder than you expected?
   - What was easier?
   - One specific thing you'd like to understand better before Month 2.
3. Tick all Week 04 and Month 1 boxes in `04-Progress-Tracker.md`.

When you're satisfied, report back. That's the gate before Month 2 begins.
