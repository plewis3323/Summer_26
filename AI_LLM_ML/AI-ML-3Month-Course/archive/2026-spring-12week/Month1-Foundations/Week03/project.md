# Week 03 — Mini-Project: micrograd + MLP on Synthetic Shower Response

## Problem

Implement a from-scratch scalar autograd engine (`micrograd`) with a thorough test suite. Use it to train a small MLP that learns a synthetic EMCal shower-energy response function. Then re-implement the same model in PyTorch and compare. Deliver a clean Python package with tests.

The point of this project is not to beat any particular baseline. It's to *own* autograd and training loops at the level you can diagnose what a PyTorch line is doing by recognizing its micrograd cousin.

## Data

Generate synthetically. Set:
```
E_true ~ 0.5 + Exponential(1/10) GeV       # energies 0.5 to ~50 GeV
response(E) = 0.95 + 0.02 * log(E)          # mild nonlinearity
sigma(E) = 0.05 * sqrt(E)                   # stochastic resolution
E_meas = E_true * response(E_true) + sigma(E_true) * N(0, 1)
```

Training set: 50 000 events. Validation: 10 000. Test: 10 000.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/micrograd_shower/
│   ├── __init__.py
│   ├── engine.py         # Value class
│   ├── nn.py             # Module, Neuron, Layer, MLP
│   ├── data.py           # synthetic generator
│   ├── torch_model.py    # PyTorch MLP mirror
│   ├── train.py          # loops for both
│   └── cli.py
├── tests/
│   ├── test_engine.py    # exhaustive gradient agreement
│   ├── test_nn.py
│   ├── test_data.py
│   └── test_torch.py
└── results/
    ├── loss_curve.png
    ├── bias_resolution.png
    └── metrics.json
```

## Acceptance criteria

1. `uv run pytest -q` passes. `test_engine.py` alone should have ≥ 20 gradient tests against `torch.autograd`.
2. `uv run python -m micrograd_shower.cli --engine micrograd --events 10000` trains the micrograd MLP end-to-end and produces a loss curve.
3. `uv run python -m micrograd_shower.cli --engine torch --events 50000` trains the PyTorch MLP.
4. Both engines' learning curves overlap within stochastic noise on a matched-seed run of the same size.
5. Final test-set (resolution of `E_pred − E_true / E_true`) of the PyTorch MLP is ≤ 1.3× the irreducible noise floor (which you compute from the generative model).
6. `metrics.json` includes: architecture, optimizer, lr, epochs, train/val/test loss curves, final MSE, final resolution, final bias, `git_sha`, `wall_clock_s` per engine.
7. `bias_resolution.png` shows bias and resolution vs `E_true` for the model and for a scipy polynomial-fit baseline. Honest discussion of tails.

## Suggested sequence

### Day 1 — engine.py

Build `Value` with the full op set. Write `test_engine.py`:

```python
import math, random, pytest, torch
from micrograd_shower.engine import Value

def _check(f, *xs):
    vals = [Value(x) for x in xs]
    y = f(*vals)
    y.backward()
    mg_grads = [v.grad for v in vals]

    txs = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    ty = f(*txs)
    ty.backward()
    torch_grads = [float(t.grad) for t in txs]
    for g, t in zip(mg_grads, torch_grads):
        assert abs(g - t) < 1e-6, f"grad mismatch {g} vs {t}"

def test_add_mul():
    _check(lambda a, b: a * b + a, 2.0, 3.0)

def test_tanh():
    _check(lambda a: (a*a + 1).tanh(), 1.5)

def test_complex():
    _check(lambda a, b, c: ((a * b).exp() + c.log()).tanh(), 1.0, 2.0, 0.5)
# ... 20 more
```

### Day 2 — nn.py

`Module`, `Neuron`, `Layer`, `MLP`. Keep this close to Karpathy's reference; it's already small.

```python
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i < len(nouts)-1)
                       for i in range(len(nouts))]
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
```

### Day 3 — data.py

Deterministic with explicit seed. Return numpy arrays. Write `test_data.py`: shape, range, response curvature sanity.

### Day 4 — train.py

Two train loops, one per engine. Log loss per step for micrograd (it'll be slow; use a small subset for this engine: 2000 events), per batch for PyTorch.

PyTorch loop sketch:
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

class TorchMLP(nn.Module):
    def __init__(self, nin=1, hidden=(32, 32), nout=1):
        super().__init__()
        sizes = [nin] + list(hidden) + [nout]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_torch(X_tr, y_tr, X_val, y_val, epochs=20, lr=1e-3, device="cpu"):
    model = TorchMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),
                       torch.as_tensor(y_tr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    hist = {"train": [], "val": []}
    for e in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb.to(device))
            loss = loss_fn(pred.squeeze(-1), yb.to(device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            val_pred = model(torch.as_tensor(X_val, dtype=torch.float32).to(device)).squeeze(-1)
            val_loss = loss_fn(val_pred, torch.as_tensor(y_val, dtype=torch.float32).to(device))
            hist["val"].append(float(val_loss))
    return model, hist
```

### Day 5 — evaluation and writeup

- Bias = mean((E_pred − E_true) / E_true) in each E_true bin.
- Resolution = std of the same.
- Plot bias and resolution side by side with the polynomial-fit baseline.
- Write the README with: approach, figures, honest limitations ("MLP underperforms polynomial in the high-E tail because training density drops exponentially"), next steps.

## Common failure modes

- **micrograd is agonizingly slow.** Yes. Don't use it on 50 000 points; use 1000–2000. That's enough to show the mechanism.
- **Grad mismatch in tests.** Usually `__rmul__` or `__pow__`. Add prints comparing intermediate forward values.
- **PyTorch loss is NaN.** Bad scaling. Standardize `E_meas` before feeding the network. Un-scale when computing bias/resolution.
- **Poly baseline beats MLP.** Possible. That's an honest finding and belongs in the writeup.
- **`loss.backward()` called twice.** `retain_graph=True` is a crutch; usually you want to detach something.

## Extensions (optional)

- Swap loss to a Gaussian negative-log-likelihood with learned σ — i.e. predict both mean and variance. Compare to fixed-σ MSE.
- Try Adam vs SGD+momentum and document training-curve differences.
- Add input/feature engineering: pass `[E_meas, log(E_meas), sqrt(E_meas)]` and see if the MLP learns faster.

## Sign-off

Tick the Week 03 boxes; commit. You now have the intuition to believe what's printed on `loss.backward()`. Week 04 introduces the first real architecture — CNNs — and the Month-1 capstone.
