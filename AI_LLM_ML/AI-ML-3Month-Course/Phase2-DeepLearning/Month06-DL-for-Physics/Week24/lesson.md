# Week 24 — Capstone 2: Shipping a Deep-Learning Result

~1 hr reading — this week's time goes into the build (`project.md`). Before
starting you should have: the Week-20 repo with its toy shower generator and
trained CNN, the Week-21 PyG pipeline, the Week-22 VAE with the collapse study,
and the Week-23 flow-vs-VAE table. This short lesson is about *how to run the
week*: choosing a track honestly, benchmarking without fooling yourself, what
"physics validation" means, and getting through the Phase 2 gate.

## 1. What this capstone proves

Capstone 1 proved you could run a classical-ML pipeline honestly. Capstone 2
proves the Phase 2 claim: you can take a deep-learning idea from architecture
choice through training, debugging, and evaluation to a *defensible* result — one
where the comparison is fair, the validation is physical, and a stranger can
reproduce the number from a fresh clone. That is the artifact; the gate (§5 of
the syllabus) also demands the *understanding*: backprop and the ELBO re-derived
cold, because those two derivations are the load-bearing math of everything you
built this phase.

One scheduled week, two weeks' worth of building — the week README says so up
front. The syllabus §6 slack exists precisely for capstones; take the second
calendar week if you need it and log it in the tracker. What you may not do is
shrink the rigor to fit the week: cut scope (fewer ablations, smaller sweep),
never cut tests, baselines, or the writeup's honesty.

## 2. Choosing your track

Both tracks are legitimate and both are staffed by things you already built.
The choice is the first committed line of the project (project.md, step 1), so
make it deliberately, Monday, using evidence you already own:

**(a) GNN cluster/jet classifier** — the Week-21 pipeline, grown up: a
point-cloud GNN on the Week-20 photon/π⁰ data (or a public jet dataset for
scale), benchmarked against your Week-20 CNN under an identical protocol, with
an ablation that answers the question the Week-21 lesson planted: *does the
graph structure actually help, or is it just extra parameters?* Choose (a) if
the discriminative thread is your thesis-relevant one, if your Week-21 E5 GNN
came tantalizingly close to (or beat) the CNN, or if you want the
classifier-craft signal for applied-ML roles. Riskiest step: the comparison,
not the model — an unfair protocol (different splits, different tuning budgets)
invalidates the week silently.

**(b) VAE fast-sim of EMCal showers** — the Week-22 VAE, grown up: generate
single-photon showers *conditioned on cluster energy*, and validate the way
Week 23 §9 said the field validates — on physics observables, against the
generator's truth. Choose (b) if generative-models-for-simulation is your
thread (it chains directly into Capstone 3b and final-capstone track b, the
CaloChallenge line), or if the Week-23 head-to-head left you with opinions.
Riskiest step: posterior collapse and conditioning bugs eating the first days —
which is why the Week-22 KL-warm-up recipe is named in the build steps.

Tie-breakers, in order: which result your thesis or target job would actually
cite; which failure you would learn more from; which dataset you trust. Your
Week-23 E6 table is evidence for (b)'s feasibility; your Week-21 E5 AUC is
evidence for (a)'s. Do not choose by which model is trendier — the writeup, not
the architecture, is what a reader keeps. (A flow variant inside track (b) is a
stretch goal, not the spec — the README is explicit.)

## 3. Baselines first, and the identical-protocol rule

The single most common way student projects (and published papers — Week 45
will show you) go wrong: the new model gets tuning attention, real splits, and
a dozen runs; the baseline gets defaults and one run. The comparison is then
"effort vs no effort", not "GNN vs CNN".

The discipline this week, non-negotiable and encoded in the build steps:

- **The baseline is retrained inside this repo, under this protocol**, before
  the new model trains — same splits, same seeds, same augmentation, same
  early-stopping rule, comparable tuning budget. For track (a) that means the
  Week-20 CNN rebuilt here; for track (b) the "baseline" is the truth itself —
  the generator's distributions frozen as reference histograms before any
  sampling happens.
- **Baseline numbers are committed before the new model's first real run.**
  Once your model's numbers exist, every protocol decision you make is
  contaminated by knowing them. Freeze the target first.
- **The success metric is named in advance** (project step 1: the single number
  that decides success). Deciding after the results are in what "success" means
  is the same sin with a delay.

## 4. Physics validation, not eyeballs

ML metrics answer "is the model good at the loss?" Physics validation answers
"would an analysis built on this be wrong?" Different question; this week you
answer both. All the physics here is Week-20/23 material, restated in
`project.md`'s Background so the spec is self-contained.

For track (a): AUC summarizes the whole ROC curve, but a real trigger or
analysis runs at a working point. So the required plot is **background
rejection at 90% signal efficiency, as a function of cluster energy** — GNN and
CNN overlaid. Per-energy matters because the two classes converge at high
energy (the two photons from a π⁰ decay merge), and a model that wins on
average but loses exactly where the physics is hard is a worse tool.

For track (b): sample grids prove nothing (Week 23 §9's judging criteria have
no eyeball axis). The suite:

- **Energy response**: mean of $E_{\text{gen}}/E_{\text{true}}$ vs true energy
  — is the generator *calibrated*, at every energy, not just on average
  (linearity)?
- **Energy resolution**: width of the same ratio vs energy. Real calorimeters
  resolve energy better, fractionally, as energy grows (fluctuations average
  out — the stochastic $\sim 1/\sqrt{E}$ scaling); your generator must
  reproduce the *generator's own* width-vs-energy trend, whatever it is, and
  the plot vs $1/\sqrt{E}$ makes agreement or failure legible.
- **Shower shapes**: ≥ 2 distributions a photon-ID analysis would cut on
  (e.g. lateral width / second moments, energy fractions in leading towers) —
  generated vs truth, each with a quantified distance (χ² or KS), not adjacent
  histograms and a shrug.

The common thread, both tracks: evaluate *conditionally* (vs energy), not just
marginally. Averages hide exactly the failures that matter.

## 5. The gate, and how the week runs

The Phase 2 gate has two halves. The repo half: model beats the stated baseline
or the writeup explains why not — a clean, understood negative result passes;
an unexamined win does not. The derivation half: backprop (2-layer, including
softmax/cross-entropy) and the ELBO (both routes), cold, on paper, scanned into
the repo. Schedule the derivations for a *fresh* morning, not the last hour of
the last day — they are the review block and the gate simultaneously, and
rushing them defeats both. Anything you had to look up gets flagged into the
month's open-question issue (syllabus §9).

The build itself is specified in `project.md`: proposal → tested data pipeline
→ frozen baseline → model → evaluation → writeup → gate check, each with its
accept criterion. Work it in order; the ordering is the §3 discipline made
mechanical. Then close the phase: tag `month-06-complete`, write `retro.md`,
open the issue.

## Check yourself

1. The gate accepts a model that *loses* to its baseline. Under what condition,
   and why is that the right policy?
2. Why must the Week-20 CNN be retrained inside the capstone repo rather than
   its Week-20 AUC quoted?
3. Why are baseline numbers committed before the new model's first real run?
4. Track (b): why is a grid of convincing generated showers insufficient, and
   what three families of plots replace it?
5. Why does track (a) require rejection-vs-energy rather than a single AUC?
6. Which two derivations close the Phase 2 gate, and what happens to a step you
   had to look up?

## Answers

1. When the writeup explains the failure — what was tried, what the evidence
   says about *why* (representation? capacity? data size? protocol?), and what
   would change the outcome. The gate tests whether you can produce defensible
   knowledge, and an understood negative result is that; an unexplained win is
   not (syllabus §8, honesty).
2. Because the quoted number was produced under a different protocol — possibly
   different splits, seeds, preprocessing, stopping rule. Only an
   identical-protocol rerun isolates the variable you claim to test (the
   architecture).
3. Once the new model's numbers are known, protocol choices (splits, stopping,
   metric details) can no longer be made impartially — every tweak is
   implicitly tuned against the comparison. Freezing the baseline first keeps
   the target fixed.
4. Eyeballs miss tails, correlations, and conditional miscalibration — exactly
   where analyses cut. Replace with: energy response (mean ratio vs E),
   resolution (width vs E, read against the $1/\sqrt{E}$ trend), and
   shower-shape distributions with per-histogram χ²/KS distances.
5. The classes are hardest to separate exactly where the two π⁰ photons merge
   (high energy); a single AUC can hide a model that fails there. Analyses run
   at working points, binned in energy — so the evaluation must too.
6. Two-layer backprop and the ELBO, re-derived cold and scanned into the repo.
   A looked-up step gets flagged in the month's open-question issue and
   re-derived when the issue closes.

## New terms

- **identical-protocol rule** — baseline and candidate share splits, seeds,
  budget, and stopping rule; anything else compares effort, not models.
- **working point** — the specific efficiency/rejection operating threshold an
  analysis actually runs at, as opposed to curve-level summaries like AUC.
- **energy response / linearity** — mean of generated-over-true energy vs true
  energy; a calibrated generator sits flat at 1.
- **energy resolution** — the width of that ratio vs energy; fractional width
  shrinking like $1/\sqrt{E}$ is the calorimeter's stochastic scaling.
- **shower shapes** — the spatial-distribution observables (widths, moments,
  energy fractions) photon-ID cuts on; the fast-sim must get their
  *distributions* right.
- **understood negative result** — a baseline loss plus a writeup that
  diagnoses it; passes the gate where an unexplained win would not.

## Going deeper

No new reading — this week runs on your own artifacts. Re-read before starting:
your Week-20 project writeup (the baseline you must beat or explain), Week-21
§9 and your E5/E6 results (track a), Week-22 §8 and your E5 collapse study
(track b), Week-23 §8–9 and your E6 table (the track choice evidence), and the
Capstone 2 section of `03-Project-Roadmap.md`. If you must look at one external
thing, make it the CaloChallenge evaluation observables (track b) or the
ParticleNet evaluation section (track a) — as models of what "evaluated
properly" looks like in the field.
