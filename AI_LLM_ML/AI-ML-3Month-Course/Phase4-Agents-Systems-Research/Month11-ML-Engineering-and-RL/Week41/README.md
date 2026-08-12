# Week 41 — Experiment Infrastructure

An untracked training run is an unlogged detector run: whatever it measured, it didn't happen — this week every experiment gets a run number, a config, and a logbook entry.

## Objectives

- Instrument a training script with an experiment tracker (Weights & Biases or MLflow): metrics, configs, artifacts, and system stats per run.
- Separate configuration from code (YAML + argparse, or hydra) so any run is reproducible from its config file alone.
- Launch and interpret a hyperparameter sweep; read parallel-coordinates/importance views instead of eyeballing loss curves.
- Register model versions with lineage (which config, which data, which commit) in a model registry.
- Retrofit all of the above onto an existing Phase-2/3 project without rewriting its science.

## Core material (~3 hrs)

- W&B documentation: Experiments quickstart, sweeps guide, and artifacts/model-registry pages — or the MLflow equivalents (tracking, MLflow Projects, model registry) if you prefer self-hosted.
- Chip Huyen, *Designing Machine Learning Systems*, the chapter on experiment tracking and versioning within model development.
- hydra documentation basics (config composition, overrides) — read enough to decide YAML+argparse vs. hydra deliberately; either is acceptable.
- Your own Capstone-2 or Capstone-3 training script — reread it as the retrofit target and list its hardcoded choices.

## Exercises (built when the week starts)

1. Config extraction: move every magic number in the chosen Phase-2/3 training script into one YAML config; run from config only. Accept when: two different configs produce two different runs with zero source edits.
2. Tracking: log train/val loss, LR, gradient norm, and the config to the tracker each run. Accept when: the tracker UI shows ≥2 complete runs with metrics, config, and git commit attached.
3. Artifacts: log the best checkpoint and the exact input-data hash as artifacts. Accept when: a fresh script can pull the artifact by name and reproduce the logged val metric within noise.
4. Sweep: define a search over LR, batch size, and one architecture knob (≥12 trials, random or Bayesian). Accept when: sweep report names the best config and one hyperparameter the objective is insensitive to.
5. Registry: register the sweep's best model as `v1` with its lineage; promote it over the pre-week baseline. Accept when: registry entry records metric, config reference, and commit hash for `v1`.
6. Regression guard: a `pytest` test loads the registered model and asserts val metric within tolerance of the logged value. Accept when: `pytest -q` green, and deliberately corrupting the checkpoint makes it fail.

## Deliverable

The Phase-2/3 project repo, upgraded in place: configs directory, tracked runs, one sweep report (exported or linked in `week41/SWEEP.md`), a registered model, and the regression test.

## Review

1. Week 9: why must the sweep's model selection use validation data and the final quoted metric use a held-out test set?
2. Week 16: you diagnosed broken training runs by eye. Name three logged quantities from this week that would have caught those failures automatically.
3. Week 8: for the LR you swept, what does too-high vs. too-low look like in the loss curve, and why (from GD convergence intuition)?
4. Week 4: your Month-1 project pinned seeds and deps. What new sources of irreproducibility does GPU training add that seeds don't fix?
5. Week 30: which hyperparameters did LoRA introduce, and which would you sweep first on a fixed budget?
