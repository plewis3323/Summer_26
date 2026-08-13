# Week 24 Project — Capstone 2

## Objective

Ship a tested deep-learning repo that either (a) classifies calorimeter
clusters (or jets) with a GNN, benchmarked against your Week-20 CNN under an
identical protocol, or (b) generates EMCal showers with an energy-conditioned
VAE, validated on physics observables against the simulator. Choose **one**
track. The gate (from `03-Project-Roadmap.md` and syllabus §5): **the model
beats the stated baseline or the writeup explains why not**, and **backprop
and the ELBO are re-derived cold** and filed. This closes Phase 2.

## Background — the physics, and the two tracks

Both tracks sit on the same detector object; the Background is written so the
spec is self-contained. Fuller versions live in Week 20 (clusters, π⁰ merging,
shower shapes) and Week 22–23 (VAEs, fast-sim, CaloChallenge).

**The calorimeter, restated.** A collider detector stops particles in a dense
**calorimeter** and measures the dumped energy. An **EMCal** is the
electromagnetic layer, built for photons and electrons. It is a grid of
**towers**; each tower reports one number. A high-energy photon does not
light a single tower: it starts an **electromagnetic shower** (pair production
γ → e⁺e⁻ alternating with bremsstrahlung), which spreads transversely over
about a **Molière radius** (a couple of cm, comparable to one tower) and is
absorbed longitudinally over ~20 **radiation lengths**. The result is a
compact blob on the grid — an image, or a small cloud of hit towers.

**Track (a) — why classify, and why a graph.** The physics problem is
**direct-photon identification**. Heavy-ion collisions (the course's resident
context is sPHENIX at RHIC) produce a **quark–gluon plasma**; photons born
inside it escape unmodified and are prized probes. The dominant fake is the
**π⁰ → γγ** decay: at high pT the two photons' opening angle shrinks as
θ ≈ 0.27 rad / E[GeV], their two blobs **merge** into one cluster, and a
naive shower-shape cut starts to fail. Week 20 asked whether a CNN on the
8×8 tower image beats those shapes. This track asks the next question: treat
the cluster as a **point cloud** — nodes = towers above threshold, features
`(log E, η, φ)`, edges from k-nearest neighbors in (η, φ) — and run a
**graph neural network** (Week 21: shared per-edge message, permutation-
invariant aggregation, global pool). The load-bearing scientific question is
not "is a GNN trendy" but **does the graph structure actually help, or is it
the extra parameters?** That is the required ablation (k, depth, or
aggregation). A public **jet** dataset (a jet is a collimated spray of
hadrons from a quark or gluon; tagging which is a standard GNN benchmark) is
an allowed scale-up if you want more particles per event than an 8×8 cluster.

**Track (b) — why generate, and why physics validation.** Full detector
simulation (Geant4) tracks every particle through a detailed geometry and
dominates collider computing budgets. **Fast simulation** learns the map from
incident particle to deposited-energy image and samples it cheaply. A **VAE**
(Week 22) does this by encoding a shower into a latent Gaussian, decoding it
back, and training a bound on the likelihood (the **ELBO**). Conditioning on
incident energy is what makes it a simulator rather than a shower collage:
the physics needs "a shower *given* 12 GeV," not "some shower."

Eyeballing sample grids proves nothing (Week 23 §9, restated in lesson §4).
A physicist signs off on a generator that matches **distributions an analysis
would cut on**, conditionally on energy:

- **Energy response / linearity:** mean of E_gen / E_true vs true energy —
  calibrated means flat at 1, at every energy, not just on average.
- **Energy resolution:** width of that ratio vs energy. Real calorimeters
  improve fractionally as ~1/√E (shower fluctuations average out); your
  generator must reproduce the *toy generator's own* width-vs-energy trend.
- **Shower shapes:** ≥ 2 spatial observables (lateral width / second moments,
  leading-tower energy fractions) — generated vs truth, each with a χ² or KS
  distance, not adjacent histograms and a shrug.

**Identical-protocol rule (both tracks).** The baseline is retrained (or
frozen, for truth histograms) *inside this repo*, same splits, same seeds,
same stopping rule, comparable budget, and its numbers are committed *before*
the new model's first real run. Quoting Week 20's AUC, or sampling the VAE
and then deciding which energies to plot, compares effort, not methods.

## Data

All public and free; continuity with Week 20 is the default.

- **Track (a), default:** the Week-20 photon/π⁰ dataset — same generator,
  same pT range (4–24 GeV), converted to point clouds (nodes = towers above
  threshold). Reuse the Week-20 split if you import it; otherwise rebuild
  under a hashed config and never mix. **Scale-up alternative:** HLS4ML LHC
  jet-tagging dataset or the Top Quark Tagging Reference Dataset (both
  Zenodo; search those names). If you switch, the CNN baseline must be
  retrained on *that* data too — you cannot compare a jet GNN to a photon
  CNN.
- **Track (b):** single-photon showers from the Week-20 generator (or a
  larger draw with the same parameters), labeled by incident energy. Do not
  mix in π⁰s — those are a different conditional distribution. Optional
  stretch: a CaloChallenge dataset (Zenodo, 1–10s of GB by tier) if you want
  Geant4 truth instead of the toy; then the validation suite runs against
  that truth, and sampling cost still belongs in the writeup.

**Compute:** the Week-20 CNN and a small GNN/VAE on 8×8 (or ~20-hit) showers
are laptop-scale. A CaloChallenge-scale VAE is a GPU week; Colab/Kaggle if
needed. Budget the second calendar week (syllabus §6) for training + the
cold derivations, not for a third architecture.

## Build steps

Do them in order; each is one of this week's exercises (E1–E7 in
`exercises.md`). The ordering *is* the identical-protocol rule made
mechanical.

1. **Proposal** (E1). Half a page, committed Monday: track, question,
   dataset, baseline, the single success number, risks (track (a): unfair
   protocol; track (b): posterior collapse / conditioning bugs). Accept when:
   it names the single number that decides success.
2. **Data pipeline** (E2). Loading/generation, splits, ≥ 3 physics sanity
   tests under `pytest`. Accept when: fresh clone + `uv sync` + one command
   rebuilds the dataset.
3. **Baseline frozen** (E3). Track (a): Week-20 CNN retrained here. Track
   (b): truth histograms frozen. Accept when: baseline numbers are committed
   before the new model trains.
4. **The model** (E4). Track (a): GNN + one ablation (k / depth /
   aggregation). Track (b): energy-conditioned VAE, KL warm-up. Accept when:
   training curves are logged and the headline config is in git.
5. **Evaluation** (E5). Track (a): AUC + rejection-at-90%-efficiency vs
   energy, GNN and CNN overlaid. Track (b): energy response (mean and width
   vs E) plus ≥ 2 shower shapes with χ² or KS, sample vs truth. Accept when:
   every plot has both models/curves and the writeup quotes the numbers.
6. **Writeup** (E6). 1–2 pages. Accept when: it states plainly whether E1's
   metric was met.
7. **Gate check** (E7). 2-layer backprop and the ELBO, cold, scanned in.
   Accept when: both scans are in the repo and looked-up steps are flagged.

## Acceptance gate (end of Week 24 — from `03-Project-Roadmap.md` and syllabus §5)

- **Model beats the stated baseline, or the writeup explains why not.** A
  clean, understood negative result passes; an unexamined win does not
  (syllabus §8). Track (a): GNN vs the retrained CNN, on the number named in
  E1, plus the ablation answering "graph vs parameters." Track (b): generated
  distributions vs frozen truth, on the number named in E1 — matching means
  while missing widths is a miss, not a win.
- **Backprop + ELBO re-derived cold** and filed in the repo (E7).
- Tested and reproducible: `pytest -q` green; fresh clone + `uv sync` + one
  command reruns the evaluation (training may load the committed checkpoint).
- The repo ships: proposal, data pipeline, baseline numbers committed first,
  model config + curves, evaluation figures, writeup, two derivation scans.

Then close the month and the phase: tag `month-06-complete`, write `retro.md`,
open the open-question issue.

## Writeup requirements (1–2 pages)

- Track chosen and why (one sentence of evidence you already owned — Week-21
  AUC or Week-23 table — not a paper you have not read).
- The headline number vs E1's success metric: met, or not, in the first
  paragraph after the table.
- Comparison to baseline under the identical protocol; track (a) includes the
  ablation row; track (b) includes per-observable χ²/KS.
- **What failed first** — written the day it happened, not reconstructed.
- Limitations: toy generator (track a/b), oversmoothing / k choice (a),
  posterior collapse and what KL warm-up did (b).
- What you would do with a month — and whether that month is Capstone 3 (b)
  (diffusion rematch) or a jet-scale GNN.

## Stretch goals (only after the gate)

- **Track (a):** a second ablation, or the public jet dataset as a
  transfer-of-method (not a replacement for the Week-20 comparison unless you
  committed that in E1).
- **Track (b):** a flow variant, or a flow comparison, using the Week-23
  coupling flow on the same showers — the README's named stretch, not a
  requirement. Report exact NLL (flow) vs ELBO (VAE) with the Week-23 caveat.
- **Both:** register the winning checkpoint in whatever logbook you are
  keeping; Week 41 will ask you to retrofit a tracker onto this repo.
