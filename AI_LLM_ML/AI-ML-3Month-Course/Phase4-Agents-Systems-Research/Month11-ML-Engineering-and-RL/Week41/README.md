# Week 41 — Experiment Infrastructure

An untracked training run is an unlogged detector run: whatever it measured, it didn't happen — this week every experiment gets a run number, a config, and a logbook entry.

## Objectives

- Instrument a training script with an experiment tracker (MLflow in the lesson; Weights & Biases is the hosted equivalent): metrics, configs, artifacts, and code identity per run.
- Separate configuration from code (YAML + argparse, or hydra) so any run is reproducible from its config file alone.
- Launch and interpret a hyperparameter sweep; read parallel-coordinates/importance views instead of eyeballing loss curves.
- Register model versions with lineage (which config, which data, which commit) in a model registry.
- Retrofit all of the above onto an existing Phase-2/3 project without rewriting its science.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: tracking vocabulary, MLflow end to end, config extraction, sweeps, the registry, and the retrofit procedure, with a runnable worked example.
- MLflow documentation: tracking quickstart and model registry pages — or the W&B equivalents (Experiments quickstart, sweeps guide, model registry) if you prefer hosted.
- Chip Huyen, *Designing Machine Learning Systems*, the chapter on experiment tracking and versioning within model development.
- hydra documentation basics (config composition, overrides) — read enough to decide YAML+argparse vs. hydra deliberately; either is acceptable.
- Your own Capstone-2 or Capstone-3 training script — reread it as the retrofit target and list its hardcoded choices.

## Exercises

See `exercises.md` (notebook generated from it when the week starts, per `NOTEBOOK_RULES.md`). The six exercises retrofit your chosen Phase-2/3 repo in place: config extraction → tracking → artifacts + data hash → a 12-trial sweep with report → registry v1 → a pytest regression guard.

## Deliverable

The Phase-2/3 project repo, upgraded in place: configs directory, tracked runs, one sweep report (exported or linked in `week41/SWEEP.md`), a registered model, and the regression test.

## Review

1. Week 9: why must the sweep's model selection use validation data and the final quoted metric use a held-out test set?
2. Week 16: you diagnosed broken training runs by eye. Name three logged quantities from this week that would have caught those failures automatically.
3. Week 8: for the LR you swept, what does too-high vs. too-low look like in the loss curve, and why (from GD convergence intuition)?
4. Week 4: your Month-1 project pinned seeds and deps. What new sources of irreproducibility does GPU training add that seeds don't fix?
5. Week 30: which hyperparameters did LoRA introduce, and which would you sweep first on a fixed budget?
