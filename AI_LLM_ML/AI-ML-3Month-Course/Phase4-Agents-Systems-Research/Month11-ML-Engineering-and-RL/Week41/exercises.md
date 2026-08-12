# Week 41 — Exercises

Work top to bottom. Setup (imports, data loading, constants) is given by the
notebook; you write only the lines each exercise asks for. E1–E5 are edits to
your chosen Phase-2/3 project repo (the notebook drives and checks them); E6
lives in that repo's `tests/` per NOTEBOOK_RULES §6.

Pick the retrofit target first: your Capstone-2 (Week 24) or Capstone-3
(Week 36) training script. Everything below happens to that repo, in place, on
a branch. `uv add mlflow pyyaml` in that repo before starting.

## E1 — Config extraction

Inventory every magic number in the training script (rates, sizes, paths,
seeds, architecture choices) and move them into one `config.yaml`. The script
takes `--config` via argparse and reads choices only from the loaded dict.
Hint: do it in two commits — extract with identical behavior first, verify same
results at the same seed, then never hardcode again.
Accept when: two different config files produce two different runs with zero
source edits, and the pre-extraction seed-0 result is reproduced exactly.

## E2 — Tracking

Instrument the script with MLflow: log the full config (`log_params`),
per-epoch train/val loss, learning rate, and gradient norm (`log_metric` with
`step=`), and the config file as an artifact. Commit before running so the
recorded commit hash is honest.
Hint: total gradient norm = `torch.cat([p.grad.flatten() for p in
model.parameters() if p.grad is not None]).norm()` right after `backward()`.
Accept when: `mlflow ui` shows at least 2 complete runs, each with metrics
curves, the full config, and a git commit attached.

## E3 — Artifacts and data identity

Log the best checkpoint and the training-data hash: save the best-val model to
a file, `log_artifact` it, and `log_param` an md5 hash of the training data
file's bytes.
Hint: "best" means tracked-during-training — keep `best_val` and overwrite the
saved file when it improves.
Accept when: a fresh script (given only the run ID) downloads the artifact,
reloads the model, and reproduces the logged val metric within run-to-run
noise.

## E4 — Sweep

Random-search LR (log-uniform), batch size, and one architecture knob, ≥12
trials, each its own tracked run with a `final_val_metric`. Write
`week41/SWEEP.md`: the best config, and one hyperparameter the objective is
insensitive to, with the plot or table that shows it.
Hint: sample LR as `10 ** random.uniform(lo, hi)`; use the MLflow UI's compare
view across selected runs.
Accept when: the sweep report names the best config and one insensitive
hyperparameter, both supported by the runs table.

## E5 — Registry

Register the sweep's best model as version 1 of a named registry model. Its
lineage (run → config, data hash, commit) must be reachable from the registry
entry. Reload it via `models:/<name>/1` and confirm predictions match the
logged run.
Hint: `mlflow.pytorch.log_model(model, "model", registered_model_name=...)`
inside the winning run — re-run the best config if the sweep didn't register.
Accept when: the registry entry records metric, config reference, and commit
hash for v1, and a reload reproduces the metric.

## E6 — Regression guard

In the project repo, write `tests/test_registered_model.py`: load the
registered model, evaluate it on a fixed validation batch (fixed seed), and
assert the metric is within a stated tolerance of the value logged at
registration.
Hint: pick the tolerance from observed seed-to-seed noise, not hope — run the
eval twice and look.
Accept when: `pytest -q` is green, and deliberately corrupting the checkpoint
file (flip some bytes, keep a backup) makes exactly this test fail.

## Review

1. Week 09: why must the sweep's model selection use validation data while the
   final quoted metric comes from a held-out test set?
2. Week 16: you diagnosed broken training runs by eye. Name three quantities
   logged this week that would have caught those failures automatically.
3. Week 08: for the LR you swept, what do too-high and too-low look like in
   the loss curve, and why, from gradient-descent convergence intuition?
4. Week 04: your Month-1 project pinned seeds and dependencies. What new
   sources of irreproducibility does GPU training add that seeds don't fix?
5. Week 30: which hyperparameters did LoRA introduce, and which would you
   sweep first on a fixed budget?
