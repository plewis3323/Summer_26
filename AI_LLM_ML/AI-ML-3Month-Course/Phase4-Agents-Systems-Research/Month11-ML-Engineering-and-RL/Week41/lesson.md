# Week 41 — Experiment Infrastructure

~4 hrs reading + setup. Before starting you should be able to: train a PyTorch
model with a training loop you wrote yourself (Week 15), diagnose a training run
from its loss curves (Week 16), use git commits and `pytest` (Week 04), and
fine-tune a model with LoRA (Week 30).

## 1. Why "I'll remember" fails

Here is a story that has happened to every ML practitioner and most physicists.

You train a model on Tuesday. Validation accuracy: 0.91. On Thursday you try a
bigger learning rate and a different weight decay — 0.89. Friday you tweak the
architecture — 0.93! Great. The following week your advisor asks: "0.93 with
which learning rate? On which version of the dataset? Can you reproduce it?"

You check your terminal history. It's gone. You check the script — you edited it
in place, so it now contains Friday's settings, except you also changed the
data-loading code on Thursday, so even re-running it doesn't give 0.93 anymore.
The best model you ever trained is a memory.

Particle physics solved this problem decades ago. Every detector run gets a run
number. Every run's configuration — magnet currents, trigger settings,
high-voltage values — is written to a database automatically. Nobody "remembers"
run conditions; the logbook does. An untracked run didn't happen.

ML training runs need exactly the same discipline, for the same reason: you will
do hundreds of them, most differ in one or two settings, and the interesting
result is always the one from two weeks ago. This week you build that discipline
as infrastructure, so it costs you nothing per run.

The four things worth recording, every run, automatically:

1. **Configuration** — every knob: learning rate, batch size, architecture,
   data version, seed.
2. **Metrics over time** — train/val loss, accuracy, learning rate, gradient
   norm, per epoch or per step.
3. **Artifacts** — the files a run produces: checkpoints, plots, the exact
   data file hash.
4. **Code identity** — which git commit produced this run.

A tool that records these is called an **experiment tracker**.

## 2. Runs, parameters, metrics, artifacts — the vocabulary

Trackers share a common vocabulary:

- A **run** is one execution of your training script. It gets a unique ID and a
  start/end time, like a detector run number.
- A **parameter** (or "param") is a config value that is fixed for the whole
  run: `lr=3e-4`, `batch_size=64`, `hidden=128`. One value per run.
- A **metric** is a number that changes during the run and is logged repeatedly
  with a step number: `val_loss` at epoch 1, 2, 3, … The tracker stores the
  whole curve.
- An **artifact** is a file attached to the run: `model.pt`, `roc.png`, a copy
  of the config file itself.
- An **experiment** is a named group of runs that belong to the same question
  ("emcal-cnn-v2"), so the hundred runs of one project don't drown the hundred
  runs of another.

That's the entire conceptual content. Everything else is tooling.

## 3. MLflow: local tracking in ten lines

We use **MLflow** in this lesson because it runs entirely on your machine — no
account, no cloud, `uv add mlflow` and you're tracking. The main hosted
alternative is **Weights & Biases (W&B)**: same concepts, a slicker web UI,
free for personal use, but it wants an account and sends your logs to their
servers. Everything below maps one-to-one onto W&B (`wandb.init`,
`wandb.log`, `wandb.Artifact`); pick whichever you prefer for your own work.
The concepts are the deliverable, not the brand.

Install and instrument:

```python
import mlflow

mlflow.set_experiment("week41-demo")

with mlflow.start_run():
    mlflow.log_param("lr", 3e-4)
    mlflow.log_param("batch_size", 64)
    for epoch in range(5):
        train_loss = 1.0 / (epoch + 1)          # stand-in for real training
        mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_artifact("config.yaml")           # any file you want attached
```

One construct to re-note: `with mlflow.start_run():` uses the `with` statement
you first met opening files in Week 02. It guarantees the run is marked
"finished" even if your code crashes inside the block — the tracking equivalent
of a file being closed properly.

MLflow writes everything into a local `mlruns/` directory next to your script.
To browse it, run in the terminal:

```
mlflow ui
```

and open `http://127.0.0.1:5000` in your browser. You get a table of runs,
sortable by any metric or param, with clickable loss curves. This table is the
logbook. When your script runs inside a git repository, MLflow also records the
current commit hash on each run automatically — that is your code identity for
free (commit before you run, or the hash points at code you didn't actually
run).

Two habits that make tracking trustworthy:

- **Log the config first, before training starts.** A crashed run with its
  config logged is evidence; a crashed run without it is noise.
- **Log the data identity.** File paths lie — `train.csv` today may not be
  `train.csv` next month. Log a hash of the file's bytes:

```python
import hashlib

def file_hash(path):
    f = open(path, "rb")
    h = hashlib.md5(f.read()).hexdigest()
    f.close()
    return h

mlflow.log_param("data_hash", file_hash("data/train.csv"))
```

`hashlib.md5` produces a short hexadecimal fingerprint of the bytes; if one
byte of the file changes, the fingerprint changes. Two runs with the same
`data_hash` trained on byte-identical data — no argument possible.

## 4. Getting the config out of the code

Tracking answers "what did run 37 use?". The companion question is "how do I
*launch* a run with different settings without editing source?" The answer:
configuration lives in a file, not in the code.

A **config file** holds every tunable choice in one place. We use YAML, a
plain-text format for nested key–value data that reads like an indented list:

```yaml
# config.yaml
lr: 0.0003
batch_size: 64
hidden: 128
epochs: 20
seed: 0
data_path: data/train.csv
```

Reading it in Python gives you an ordinary dictionary (Week 02):

```python
import yaml

f = open("config.yaml")
cfg = yaml.safe_load(f)      # -> {"lr": 0.0003, "batch_size": 64, ...}
f.close()
print(cfg["lr"])
```

(`uv add pyyaml` provides the `yaml` module. `safe_load` parses the text into
Python dicts/lists/numbers; "safe" means it refuses to execute anything.)

To choose the config file from the command line, use `argparse`, the standard
library's argument parser. It turns `python train.py --config sweep3.yaml`
into a Python value:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()
print(args.config)           # "sweep3.yaml" if passed, else "config.yaml"
```

Now the whole pattern:

```python
args = parser.parse_args()
f = open(args.config)
cfg = yaml.safe_load(f)
f.close()

with mlflow.start_run():
    mlflow.log_params(cfg)               # logs every key at once
    mlflow.log_artifact(args.config)     # and attaches the file itself
    train(cfg)                           # code reads ONLY from cfg
```

The test of a clean separation: **two different configs must produce two
different runs with zero source edits.** If you ever catch yourself editing a
number inside `train.py`, that number belongs in the config.

One paragraph on **hydra**, because you'll see it in the wild: hydra is a
config framework that composes YAML files (a base config plus per-model,
per-dataset fragments) and lets you override any key from the command line
(`python train.py lr=1e-3 model.hidden=256`). It shines when configs get deep
and numerous. For one project, YAML + argparse is fine and has no magic; choose
deliberately, not by fashion.

## 5. Sweeps: searching hyperparameters on purpose

A **hyperparameter sweep** is a batch of runs that differ only in config,
launched systematically to answer "which settings are best, and which don't
matter?" Three standard strategies:

- **Grid search**: try every combination of a few listed values. Honest but
  explodes combinatorially — 4 values × 4 values × 4 values = 64 runs.
- **Random search**: sample each hyperparameter independently at random from a
  range. Usually beats grid for the same budget, because real objectives depend
  strongly on one or two hyperparameters and weakly on the rest — random search
  tries many distinct values of the important one, while grid wastes runs
  repeating it.
- **Bayesian search**: fit a cheap model of "config → score" on the runs so far
  and pick the next config where improvement looks likely. Worth it for
  expensive runs; overkill for cheap ones.

You do not need a framework to sweep. A loop is a sweep:

```python
import random

for trial in range(12):
    cfg = dict(base_cfg)                          # copy the base config
    cfg["lr"] = 10 ** random.uniform(-4.5, -2.5)  # log-uniform: 3e-5..3e-3
    cfg["batch_size"] = random.choice([32, 64, 128])
    cfg["hidden"] = random.choice([64, 128, 256])
    with mlflow.start_run():
        mlflow.log_params(cfg)
        val = train(cfg)
        mlflow.log_metric("final_val_loss", val)
```

Note the learning rate is sampled **log-uniformly** (uniform in the exponent):
the difference between 1e-4 and 2e-4 matters about as much as between 1e-3 and
2e-3, so the search should treat multiplicative steps equally — you learned why
when you studied gradient descent step sizes in Week 05. W&B has a built-in
sweep agent and Optuna is a dedicated search library; both automate exactly
this loop, plus Bayesian strategies and early stopping of hopeless trials.

Reading a sweep is a skill of its own. In the MLflow UI, select all sweep runs
and compare. Two views to learn:

- **Parallel coordinates**: one vertical axis per hyperparameter plus one for
  the objective; each run is a line threading through its values. Good runs
  clustering through the same region of one axis = that hyperparameter matters,
  and you can read off its good range.
- **Importance / sensitivity**: which hyperparameter, when varied, moves the
  objective most. Just as valuable is the *insensitive* one — "batch size
  didn't matter from 32 to 128" is a real finding that simplifies every future
  run.

And the eternal rule from Week 09: the sweep selects on **validation** data.
The number you report at the end comes from a **test set** the sweep never saw
— otherwise the sweep has quietly overfit your model choice to the validation
set, the model-selection version of leakage.

## 6. The model registry: versions with lineage

After the sweep you have a best checkpoint. Six weeks from now you'll have
three "best" checkpoints and a deployment (Week 44) that needs to know which
one is blessed. A **model registry** is a named, versioned shelf for models:
you register a run's model under a name, it becomes `v1`; register a better one
later, `v2`. Each version keeps its **lineage** — a pointer back to the run
that made it, and through the run to the config, metrics, data hash, and git
commit. "Which model is in production and how was it trained?" becomes a
lookup, not an investigation.

In MLflow, registering happens when you log the model itself (not just the
weights file) from inside the run:

```python
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_params(cfg)
    train(cfg)
    mlflow.pytorch.log_model(model, "model",
                             registered_model_name="pid-mlp")
```

Each call under the same `registered_model_name` creates the next version.
Loading it back anywhere:

```python
model = mlflow.pytorch.load_model("models:/pid-mlp/1")   # name/version
```

(If the exact signatures have drifted by the time you run this, the MLflow
"model registry" docs page has the current form — the concept is stable, the
API less so.)

The registry earns its keep with a **regression guard**: a pytest test that
loads the registered model, evaluates it on a fixed validation batch, and
asserts the metric matches the value logged at registration time (within a
tolerance). If someone corrupts the checkpoint, retrains sloppily, or the
data pipeline drifts, `pytest -q` goes red before anything ships. You'll write
this test in the exercises, and Week 44 will lean on it.

## 7. Retrofitting an existing project

Greenfield tracking is easy; the useful skill is adding it to a project that
already exists without breaking the science. The procedure, which you'll apply
to your Capstone-2 or Capstone-3 repo:

1. **Inventory the magic numbers.** Read the training script and list every
   literal that shapes the result: rates, sizes, paths, seeds, thresholds,
   architecture choices. Grepping for `=` and digits is genuinely effective.
2. **Move them to `config.yaml`**, one commit, no behavior change. Run the
   script; confirm identical results (same seed → same numbers).
3. **Add tracking** — `start_run`, `log_params`, per-epoch `log_metric`,
   final artifacts. One commit. Re-run; confirm identical results again.
4. **Sweep** the two or three hyperparameters you never had time to tune.
5. **Register** the winner; write the regression test.

Steps 2 and 3 are separate commits on purpose: if anything changes numerically
you can bisect which edit did it. The science code should not know the tracker
exists beyond a handful of `log_*` calls; if you find yourself restructuring
the model to satisfy the tracker, stop — you're holding it wrong.

## 8. Worked example: a tracked, swept, registered classifier

Everything above, end to end, small enough to run on a laptop in a minute.
The physics stand-in: two Gaussian blobs playing "signal" and "background" —
a toy version of your Capstone-1 particle-ID problem, where signal events
(the particles you want) and background events (everything that mimics them)
overlap in feature space and a classifier must draw the boundary.

`config.yaml`:

```yaml
lr: 0.003
hidden: 32
epochs: 30
batch_size: 128
seed: 0
n_train: 4000
n_val: 1000
```

`train.py`:

```python
import argparse
import random
import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    sig = torch.randn(n // 2, 2, generator=g) * 1.0 + torch.tensor([1.5, 0.0])
    bkg = torch.randn(n // 2, 2, generator=g) * 1.5 + torch.tensor([-1.0, 0.0])
    x = torch.cat([sig, bkg])
    y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    return x, y

class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_training(cfg):
    torch.manual_seed(cfg["seed"])
    xtr, ytr = make_data(cfg["n_train"], seed=1)
    xva, yva = make_data(cfg["n_val"], seed=2)
    model = MLP(cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    n = cfg["n_train"]
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_loss = loss_fn(model(xva), yva).item()
            val_acc = ((model(xva) > 0) == (yva > 0.5)).float().mean().item()
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
    return model, val_acc

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--sweep", type=int, default=0)
args = parser.parse_args()
f = open(args.config)
base = yaml.safe_load(f)
f.close()

mlflow.set_experiment("week41-worked-example")

if args.sweep == 0:
    with mlflow.start_run():
        mlflow.log_params(base)
        mlflow.log_artifact(args.config)
        model, acc = run_training(base)
        print("val_acc", acc)
else:
    best_acc = 0.0
    for trial in range(args.sweep):
        cfg = dict(base)
        cfg["lr"] = 10 ** random.uniform(-3.5, -1.5)
        cfg["hidden"] = random.choice([8, 32, 128])
        with mlflow.start_run():
            mlflow.log_params(cfg)
            model, acc = run_training(cfg)
            mlflow.log_metric("final_val_acc", acc)
            if acc > best_acc:
                best_acc = acc
                mlflow.pytorch.log_model(model, "model",
                                         registered_model_name="toy-pid")
    print("best val_acc", best_acc)
```

Run it:

```
python train.py                  # one tracked run from config.yaml
python train.py --sweep 12       # 12-trial random sweep + registration
mlflow ui                        # browse; open http://127.0.0.1:5000
```

In the UI, sort the sweep runs by `final_val_acc`. You should see: `lr` matters
(too high diverges, too low undertrains — Week 05's picture), `hidden` barely
matters past 32 on a problem this easy, and the registered `toy-pid` model has
versions whose lineage points back at exactly the runs that produced them.
That's the whole apparatus, on a problem small enough to see through.

## Check yourself

1. Name the four things a tracker records per run, and give the detector-run
   analogy for each.
2. What's the difference between a parameter and a metric, in tracker
   vocabulary?
3. Why hash the data file instead of logging its path?
4. Your teammate's script has `lr = 3e-4` hardcoded and they say "it's fine,
   it's in git history." Give two concrete failures git history alone doesn't
   prevent.
5. Why does random search usually beat grid search on a fixed budget?
6. The sweep found its best config using validation accuracy. Why can't you
   quote that same number as your final result?
7. What question does a model registry answer that the runs table alone
   answers only painfully?
8. In the retrofit procedure, why are "move numbers to config" and "add
   tracking" separate commits?

## Answers

1. Configuration (magnet currents / trigger settings), metrics over time
   (detector rates and monitoring plots during the run), artifacts (the
   recorded data files), code identity (the DAQ software version). Every
   detector run logs all four automatically; training runs should too.
2. A parameter is one value fixed for the whole run (logged once); a metric is
   a time series logged repeatedly with a step number.
3. The path's contents can change silently; the hash fingerprints the actual
   bytes. Same hash → byte-identical data, no trust required.
4. (a) Nothing links a *result* to a *commit* — you know every value `lr` ever
   had, but not which one produced Thursday's 0.93 unless they committed
   before every run and recorded which commit each run used. (b) Comparing 30
   runs means archaeology through diffs instead of sorting a table; in
   practice nobody does it, so the comparison never happens.
5. Real objectives usually depend strongly on one or two hyperparameters.
   Random search gives you N distinct values of each hyperparameter in N runs;
   a grid gives you only a few distinct values of each, repeated over
   combinations of the others.
6. The sweep selected the config *because* it maximized validation accuracy,
   so that number is biased upward — model selection has partially fit the
   validation set. Quote a held-out test set the sweep never touched
   (Week 09).
7. "Which exact model is the current best / deployed one, and what config,
   data, and commit produced it?" — a named version with lineage, versus
   scrolling a table of hundreds of runs and hoping you tagged the right one.
8. So any numerical change can be bisected to one edit. If results shift after
   both changes land together, you can't tell whether the config extraction
   changed behavior or the tracking calls did.

## New terms

- **experiment tracker** — tool that records config, metrics, artifacts, and
  code identity per training run.
- **run** — one execution of a training script, with a unique ID.
- **parameter (param)** — a per-run config value, logged once.
- **metric** — a value logged repeatedly during a run with a step number.
- **artifact** — a file attached to a run (checkpoint, plot, config copy).
- **experiment** — a named group of related runs.
- **MLflow / Weights & Biases (W&B)** — local-first and hosted experiment
  trackers, respectively.
- **config file** — file (here YAML) holding every tunable choice, read at
  startup; code contains no magic numbers.
- **YAML** — indentation-based text format for nested key–value data.
- **argparse** — standard-library module that parses command-line arguments.
- **hash / fingerprint** — short hex string computed from a file's bytes;
  changes if any byte changes.
- **hyperparameter sweep** — a systematic batch of runs differing only in
  config (grid, random, or Bayesian strategy).
- **parallel-coordinates plot** — sweep view with one axis per hyperparameter;
  each run is a line.
- **model registry** — named, versioned store of models with lineage back to
  their runs.
- **lineage** — the chain model version → run → config + data hash + commit.
- **regression guard** — a test that reloads a registered model and asserts
  its metric still matches the logged value.
- **hydra** — config-composition framework; an alternative to plain
  YAML + argparse.

## Going deeper

- Chip Huyen, *Designing Machine Learning Systems* — the experiment tracking
  and versioning material inside the model-development chapter; the practices
  here, argued from industry failure modes.
- MLflow docs: tracking quickstart + model registry pages — the current API
  for everything this lesson sketched.
- W&B docs: Experiments quickstart, sweeps guide, model registry — the hosted
  equivalent; the sweeps guide's parallel-coordinates examples are the best
  around.
- Hydra docs: "Your first Hydra app" — read enough to decide YAML + argparse
  vs hydra deliberately.
