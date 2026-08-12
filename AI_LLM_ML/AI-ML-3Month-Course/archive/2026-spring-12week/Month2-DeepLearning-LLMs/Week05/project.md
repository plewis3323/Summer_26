# Week 05 — Mini-Project: GNN on Jet Constituents vs MLP Baseline

## Problem

Train a GNN (EdgeConv-based, ParticleNet-lite) on a jet-tagging task and prove it beats a dense MLP baseline that sees the same features. The point is not to match Particle Transformer's SOTA — it's to build a runnable GNN pipeline and understand when and why the graph prior binds.

## Data

Pick one:

### Option A: Top Tagging Reference Dataset (Butter et al. 2019)
https://zenodo.org/records/2603256. ~2M jets, binary (top vs QCD). Small enough to iterate quickly.

### Option B: JetClass (subset)
https://zenodo.org/records/6619768. 10 classes, ~100M jets. Use the "test" split subsampled to ~200k for tractability.

Either works. Option A is cleaner pedagogically (binary, standard benchmark). Option B is more research-flavored.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/jet_gnn/
│   ├── __init__.py
│   ├── data.py           # loaders for Zenodo datasets
│   ├── features.py       # log ΔR, log k_T, log z, etc.
│   ├── models/
│   │   ├── mlp.py        # baseline (pad + flatten + MLP)
│   │   ├── edgeconv.py   # ParticleNet-lite
│   │   └── attn.py       # optional — Particle Transformer-lite
│   ├── train.py
│   ├── eval.py
│   └── cli.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
└── results/
    ├── loss.png, roc.png
    ├── params_table.md
    └── metrics.json
```

## Acceptance criteria

1. `uv run pytest -q` passes.
2. MLP baseline trained to AUC ≥ 0.92 on Top Tagging (or comparable on JetClass).
3. EdgeConv GNN trained with same compute budget, beats MLP by ≥ 3 AUC points.
4. Parameter counts listed for all models; GNN should be smaller or comparable.
5. ROC curves overlaid, test AUC reported for all models.
6. Project README names **one place the GNN prior clearly helps** (sparse events, wide constituent-count variation, low-constituent edge cases) and **one place where the advantage is marginal**.
7. Wall-clock comparison reported.

## Suggested plan (4–5 days)

- **Day 1.** Data download, loader, feature engineering (log ΔR and friends). Smoke-train a dumb MLP to check data plumbing.
- **Day 2.** Full MLP baseline training run with OneCycle + Adam. Log to wandb.
- **Day 3.** EdgeConv model. Get it to train. First baseline.
- **Day 4.** Tune GNN (k, widths, depth). Try 2 configs, keep best.
- **Day 5.** Plots + writeup.

## Implementation notes

### Features
Canonical for jets:
- Constituent-level: log pT, log E, Δη = η − η_jet, Δφ = φ − φ_jet, ΔR = √(Δη² + Δφ²), log k_T = log(min(pT_i, pT_j) · ΔR_ij), log z = log(min(pT_i, pT_j)/(pT_i + pT_j)).
- Jet-level: pT_jet, η_jet, mass, multiplicity.

Include them as constituent features and optionally pair-wise features (for Particle Transformer-lite).

### Pytorch Geometric data loading
```python
from torch_geometric.data import Data, Batch

def to_pyg(jet_arrays):
    # jet_arrays: list of dicts with keys "feats" (N, F) and "label" (scalar)
    return [Data(x=torch.tensor(j["feats"], dtype=torch.float32),
                 y=torch.tensor(j["label"])) for j in jet_arrays]
```

Use PyG's `DataLoader` which builds a `Batch` (block-diagonal graph) per mini-batch.

### EdgeConv model
```python
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, global_max_pool

class ParticleNetLite(nn.Module):
    def __init__(self, in_dim=7, n_classes=2, k=16):
        super().__init__()
        def mlp(d_in, d_out):
            return nn.Sequential(nn.Linear(2 * d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(),
                                 nn.Linear(d_out, d_out), nn.BatchNorm1d(d_out), nn.ReLU())
        self.ec1 = DynamicEdgeConv(mlp(in_dim, 64), k=k)
        self.ec2 = DynamicEdgeConv(mlp(64, 128), k=k)
        self.ec3 = DynamicEdgeConv(mlp(128, 256), k=k)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes))

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.ec1(x, batch)
        x = self.ec2(x, batch)
        x = self.ec3(x, batch)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(x)
```

### MLP baseline
Pad constituents to max N (e.g. 50); flatten; feed into a 3-layer MLP. Use a mask to zero out padding before pooling if you use masked mean instead of flatten.

### Training
Adam, lr=1e-3, OneCycle schedule, ~30 epochs, label smoothing 0.05. Don't over-optimize — the point is a faithful comparison on equal compute.

## Write-up template (`project/README.md`)

```
# GNN vs MLP on Top Tagging

## Problem
Distinguish top-quark jets from QCD jets given constituent-level features.

## Data
Top Tagging Reference Dataset, Butter et al. 2019. 2M jets, 1.2M train / 400k val / 400k test. Features: (pT, η, φ, E) per constituent, up to 200 per jet.

## Models
| Model | Params | Test AUC | Test accuracy |
| --- | --- | --- | --- |
| MLP (pad 40, 3-layer, 256 hidden) | 1.2M | 0.932 | 0.886 |
| ParticleNet-lite (k=16, 3 EdgeConv) | 0.9M | 0.971 | 0.924 |
| Particle Transformer-lite (optional) | 1.5M | 0.978 | 0.931 |

## Where the graph prior helps
- Jets with few constituents (<10): MLP has to handle padding, GNN doesn't.
- High-mass top jets with wide constituent spread: GNN's local neighborhoods adapt via k-NN.

## Where it's marginal
- Low-multiplicity QCD jets where the task reduces to hardness + mass: MLP catches most of it.

## Reproducibility
commit, uv.lock, ~40 min on an RTX 3060 or ~15 min on A100.
```

## Common failure modes

- **GNN slower than MLP and no better.** You're not reusing the k-NN graph or your k is too small. Also check you don't accidentally apply Dropout inside EdgeConv blocks during eval.
- **MLP baseline too strong.** Your data features are too fully engineered. Drop the pair-wise features from the MLP — let the GNN compute them.
- **Training loss NaN.** ΔR or k_T in log space can be ≤ 0 if you don't floor. Clip with `torch.clamp(..., min=1e-8)`.
- **Over-smoothing past 3 EdgeConv layers.** Classic. Use residual connections inside EdgeConv or stay shallow.

## Extensions (optional)

- Add a Particle Transformer-lite baseline (single-head self-attention with pair-wise bias). Compare.
- Try `GATv2Conv` from PyG. Verify it beats vanilla GCN but loses to EdgeConv on jets (it usually does).
- Try mini-batch size 1024 on GPU with `torch.compile`; compare wall-clock to un-compiled.

## Sign-off

Commit, tick Week 05 boxes. Week 06 is the transformer from scratch — the single most important architectural implementation of the course.
