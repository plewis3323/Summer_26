# Week 24 — Capstone 2

One scheduled week, two weeks' worth of building (syllabus §6's slack exists for this):
ship a tested repo that beats your CNN or generates samples a physicist (or a
careful ML reviewer) would sign off on. Science data is the default; a public
graph or image set with the same protocol is allowed.

## Objectives

- Scope, build, and ship a deep-learning project end-to-end: data, model, training,
  evaluation, tests, writeup — reproducible from a fresh clone.
- Benchmark honestly against a baseline you built yourself (Week-20 CNN, or Geant-style
  toy truth distributions).
- Validate with physics observables, not just ML metrics.
- Pass the Phase 2 gate: backprop and ELBO derivations re-done cold, on paper, no notes.

## Core material (~0 hrs new)

No new reading. Your Week-17/18 notes (track a: CNN recipe), Week-21 notes (track a:
GNN), Week-22 notes (track b: VAE), Week-23 notes (the data API — not flows), and the
Week-20 repo are the references. Flows: `optional-flows.md` if you want them.

## Choose ONE track

**(a) GNN cluster/jet classifier.** Point-cloud GNN (PyG, Week-21 pipeline) on the
Week-20 photon/π⁰ dataset — or a public jet-tagging dataset if you want scale —
benchmarked against the Week-20 CNN under an identical protocol.

**(b) VAE fast-sim of EMCal showers.** Generative model of single-photon showers,
conditioned on cluster energy, validated on physics distributions. (A flow variant or
flow comparison is a legitimate stretch goal, not a requirement.)

## Exercises (built when the week starts — the project plan)

1. Proposal (half page, committed first): question, dataset, baseline, success metric,
   risks. Accept when: it names the single number that decides success.
2. Data pipeline with `pytest -q` green: generation/loading, splits, and at least three
   physics sanity tests (e.g. energy conservation in the generator, pT–opening-angle
   kinematics). Accept when: fresh clone + `uv sync` + one command rebuilds the dataset.
3. Baseline reproduced under this repo's protocol (track a: Week-20 CNN retrained here;
   track b: truth-level toy distributions frozen as reference histograms). Accept when:
   baseline numbers are committed before the new model trains.
4. The model. Track a: GNN with one ablation (kNN k, depth, or aggregation). Track b:
   energy-conditioned VAE with Week-22's KL-warm-up recipe. Accept when: training curves
   are logged and the config that produced the headline number is in git.
5. Evaluation. Track a: AUC + rejection-at-90%-efficiency vs energy, GNN and CNN
   overlaid. Track b: energy response (mean and width of E_gen/E_true vs E) plus ≥ 2
   shower-shape distributions, sample vs truth, with a χ² or KS distance per histogram.
   Accept when: every plot has both models/curves and the writeup quotes the numbers.
6. Writeup (1–2 pages): result, comparison to baseline, what failed first (syllabus §8
   honesty policy), limitations, and what you'd do with a month. Accept when: it states
   plainly whether the success metric from Exercise 1 was met.
7. Gate check: re-derive 2-layer backprop and the ELBO cold; scan and commit. Accept
   when: both scans are in the repo and any step you had to look up is flagged for the
   month's open-question issue.

## Deliverable

Capstone-2 repo on GitHub: `pytest -q` green, one-command reproduction, writeup, and the
two cold derivations. Phase 2 gate: model beats the stated baseline or the writeup
explains why not.

Then close the month and the phase: tag `month-06-complete`, write `retro.md`, open the
open-question issue.

## Review

- (Week 13/22) The gate derivations are this week's review — backprop and ELBO, cold.
- (Week 09) What is the test set's one job this week, and how many times may you
  touch it?
- (Week 16) List the three training diagnostics you will check first when the capstone
  model's first run flatlines.
- (Week 04) Name two Week-04 reproducibility practices this repo must satisfy before
  "done" (syllabus §5's definition).
