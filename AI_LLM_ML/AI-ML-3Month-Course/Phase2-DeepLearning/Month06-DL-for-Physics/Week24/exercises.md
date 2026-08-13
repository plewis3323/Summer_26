# Week 24 — Exercises

This week is Capstone 2; the exercises are the project plan, in build order —
each one is a step of `project.md`, and finishing E1–E7 *is* finishing the
capstone. Read `project.md` before writing the proposal. Choose **one** track
on Monday and stay on it: (a) GNN cluster/jet classifier, or (b) VAE fast-sim
of EMCal showers. Most work lives in the capstone repo, not the notebook, per
`NOTEBOOK_RULES.md` §6; the notebook hosts acceptance checks (file-exists,
baseline-committed-before-model, plot-has-both-curves) and displays E5's
figures. One scheduled week, two weeks' worth of building — syllabus §6 slack
exists for this; take the second calendar week if you need it and log it.

The Phase 2 gate derivations (E7) are scheduled for a *fresh* morning, not the
last hour of the last day.

## E1 — Proposal

Half a page, committed first, before any training: question, dataset,
baseline, success metric, risks. Name the **single number** that decides
success.
Hint: track (a) that number is usually test AUC, or rejection at 90%
efficiency in the merging bin — pick one and do not change it after E4's
numbers exist. Track (b) that number is a χ² or KS on the hardest shower-shape
plus a linearity tolerance on energy response. Lesson §2's tie-breakers, in
order: which result your thesis would cite; which failure you would learn
more from; which dataset you trust. Your Week-21 E5 AUC and Week-23 E6 table
are evidence, not decoration.
Accept when: it names the single number that decides success.

## E2 — Data pipeline

Generation/loading, splits, and at least three physics sanity tests (e.g.
energy conservation in the generator, pT–opening-angle kinematics), with
`pytest -q` green. Fresh clone + `uv sync` + one command rebuilds the dataset.
Hint: track (a) imports the Week-20 generator (or a public jet dataset —
HLS4ML LHC jet tagging or the Top Quark Tagging Reference Dataset, both on
Zenodo) and converts clusters to point clouds as in Week 21 E4. Track (b)
generates *single-photon* showers only, labeled by incident energy — the
condition, not a class. The three tests are the data-integrity check; write
them as `pytest` so CI re-verifies them forever.
Accept when: fresh clone + `uv sync` + one command rebuilds the dataset.

## E3 — Baseline first

Reproduce the baseline under *this* repo's protocol, and commit the numbers
before the new model trains. Track (a): retrain the Week-20 CNN here — same
splits, same seeds, same augmentation, same early-stopping rule. Track (b):
freeze the generator's truth-level distributions as reference histograms
(energy response, resolution vs E, ≥ 2 shower shapes) before any VAE sample
is drawn.
Hint: quoting last month's AUC is not a comparison (lesson §3). Once the new
model's numbers exist, every protocol decision is contaminated. Freeze the
target first.
Accept when: baseline numbers are committed before the new model trains.

## E4 — The model

Track (a): point-cloud GNN (PyG, Week-21 pipeline) with **one** ablation (kNN
k, depth, or aggregation). Track (b): energy-conditioned VAE with Week-22's
KL-warm-up recipe.
Hint: the ablation is the Week-21 question made mechanical — does the graph
structure actually help, or is it extra parameters? Change one thing, report
both. Track (b): condition by concatenating (or FiLM-ing) incident energy
into encoder and decoder; start β at 0 and warm it up, or you will rediscover
posterior collapse on day 2 (Week 22 E5). Log every run (even if Week 41's
tracker is not in place yet: config + curves + seed in `results/`).
Accept when: training curves are logged and the config that produced the
headline number is in git.

## E5 — Evaluation

Track (a): AUC + rejection-at-90%-efficiency vs energy, GNN and CNN overlaid.
Track (b): energy response (mean and width of E_gen / E_true vs E) plus ≥ 2
shower-shape distributions, sample vs truth, with a χ² or KS distance per
histogram.
Hint: evaluate *conditionally* (vs energy), not just marginally — averages
hide the failures that matter (lesson §4). Track (b): sample grids prove
nothing; the suite has no eyeball axis. Plot resolution vs 1/√E so agreement
or failure is legible.
Accept when: every plot has both models/curves and the writeup quotes the
numbers.

## E6 — Writeup

1–2 pages: result, comparison to baseline, what failed first (syllabus §8
honesty policy), limitations, and what you'd do with a month.
Hint: the first sentence after the table is whether E1's success metric was
met. An understood negative result passes the gate; an unexplained win does
not. Write "what failed first" the day it happened.
Accept when: it states plainly whether the success metric from Exercise 1 was
met.

## E7 — Gate check

Re-derive 2-layer backprop (including softmax + cross-entropy) and the ELBO
(both routes) cold, on paper, no notes. Scan and commit both.
Hint: schedule a fresh morning. Anything you had to look up gets flagged for
the month's open-question issue (syllabus §9) — that is the point of the
cold redo, not a shame flag. Week 13 and Week 22 scans are not a substitute;
the gate is that you can still do them.
Accept when: both scans are in the repo and any step you had to look up is
flagged for the month's open-question issue.

## Review

1. (Week 13/22) The gate derivations are this week's review — backprop and
   ELBO, cold.
2. (Week 09) What is the test set's one job this week, and how many times may
   you touch it?
3. (Week 16) List the three training diagnostics you will check first when
   the capstone model's first run flatlines.
4. (Week 04) Name two Week-04 reproducibility practices this repo must
   satisfy before "done" (syllabus §5's definition).
