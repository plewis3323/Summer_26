# Week 20 — Mini-Project: Image Classifier vs Tabular Baseline

Default physics problem (fully explained in `project.md`): at high pT the two
photons from a π⁰ merge into one EMCal cluster, faking a direct photon — this
week you build the CNN that separates them, and check whether it actually beats
shower-shape variables. Allowed substitute: a public image task with a
hand-crafted-features baseline and the same eval protocol.

## Objectives

- Generate a labeled toy dataset of calorimeter shower images (single-photon vs
  merged-π⁰) with controllable energy, opening angle, and noise.
- Build the tabular baseline: shower-shape features (E-ratios, cluster moments, tower
  multiplicity) into a Phase-1-style BDT/logistic classifier.
- Train a CNN on the raw tower images using the Week-18 recipe; tune it honestly against
  a fixed validation split.
- Evaluate like a physicist: ROC and background rejection at fixed signal efficiency, as
  a function of cluster energy.
- Write up the comparison, including where the CNN wins, where it doesn't, and what it
  is looking at.

## Core material (~1 hr — this is a building week)

- Re-skim your Week-17/18 notes; they are the spec for the model and recipe.
- Skim one HEP reference for context: the CaloChallenge dataset description, or any
  paper on π⁰/γ discrimination with calorimeter shower shapes — read for the feature
  definitions, not the results. (sPHENIX EMCal: SciGlass-descended blocks, ~Δη×Δφ =
  0.024×0.024 towers; image patches of ~8×8 towers around the cluster seed are a
  reasonable choice.)

## Exercises (built when the week starts)

1. Toy shower generator: 2-photon kinematics from π⁰ decay + parametrized transverse
   shower spread deposited on a tower grid, plus single-photon events; noise and energy
   smearing. Accept when: mean opening-distance between photons decreases with pT as
   kinematics predicts (plot), and event images eyeball correctly.
2. Dataset assembly: N ≥ 20k events across a pT range where merging turns on;
   train/val/test split stratified in pT. Accept when: class balance and pT spectra per
   split are plotted and match.
3. Tabular baseline: ≥ 5 shower-shape features + BDT (Week-11 tooling). Accept when:
   ROC AUC on the test set is reported with the Phase-1 CV discipline.
4. CNN on tower images (Week-17 architecture arithmetic shown for your design). Accept
   when: training converges (curves shown) and test AUC ≥ tabular baseline, or the gap
   is reported.
5. Physics evaluation: background rejection at 90% signal efficiency vs cluster energy,
   both models on one plot. Accept when: the plot exists and the high-energy behavior
   (merging → harder) is visible and discussed in ≤ 3 lines.
6. What does it see: saliency map or occlusion scan on 4 correctly- and 4 wrongly-
   classified clusters. Accept when: the figure exists and one line says whether the CNN
   attends to the two-shower substructure.
7. Writeup (~1 page in the repo): setup, results table, honest limitations of the toy
   simulation. Accept when: a stranger could reproduce the headline number from the repo.

## Deliverable

A small tested repo (generator + training + eval, `pytest -q` green on the generator's
kinematics checks) with the CNN-vs-tabular comparison and writeup. This closes Month 05:
tag `month-05-complete`, write `retro.md`, open the open-question issue.

*(Hardware note: this is a flagged GPU week in the syllabus — Colab/Kaggle if the local
GPU is too slow.)*

## Review

- (Week 17) Compute your network's receptive field at the last conv layer. Does it
  cover the typical photon-pair separation at the merging threshold?
- (Week 12) Your Capstone-1 pipeline had nested CV. Why is a single fixed split
  defensible here, and what claim does it not license?
- (Week 10) Why report rejection at fixed efficiency rather than accuracy, for a
  trigger-like physics use case?
- (Week 08) The toy generator is a likelihood you wrote yourself. What does that mean
  for how seriously to take the CNN's advantage on real data?
