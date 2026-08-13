# Week 20 — Exercises

This week is the mini-project; the exercises are its stages, in build order —
each one is a step of `project.md`, and finishing E1–E7 *is* finishing the
project. Read `project.md` before starting E1. Accordingly, most work lives in
the repo, not the notebook: the generator, the dataset, the BDT and CNN
training scripts, and the eval plots, per `NOTEBOOK_RULES.md` §6. The notebook
provides setup (imports, seeds, plot scaffolds), drives the scripts, and holds
the analysis cells (physics plots, saliency, writeup checks). This is a flagged
GPU week (syllabus §8), but 8×8 images train on a laptop CPU in minutes; the
flag is for the stretch goals.

## E1 — Toy shower generator

Write the parametrized generator in the repo (`src/` — see `project.md` for the
parameter table): 2-photon kinematics from π⁰ decay, each photon deposited as a
2D Gaussian on the tower grid, plus single-photon events; energy smearing,
per-tower noise, and zero-suppression. Store the 8×8 seed-centered patch, the
label, and the truth (pT, opening distance, photon energies).
Hint: lesson §4–5 and `project.md` Background — boost the rest-frame
back-to-back photons into the lab, then verify the pair's invariant mass is
0.135 GeV and that θ_min ≈ 0.27/E[GeV]. Integrate the Gaussian over tower
edges with CDFs, do not sample a point per tower. The kinematics checks belong
in `pytest`, not in a notebook cell.
Accept when: mean opening-distance between photons decreases with pT as
kinematics predicts (plot), and event images eyeball correctly.

## E2 — Dataset assembly

Generate N ≥ 20k events across the pT range where merging turns on (4–24 GeV),
balanced photon vs π⁰, and split train/val/test stratified in pT. Save images,
labels, truth, and the split indices.
Hint: stratify on pT *bins*, not on the raw float — otherwise the high-pT tail
can vanish from val/test. Plot class balance and pT spectra *per split* on one
figure; they should overlay. Freeze the split before E3; both models train on
it.
Accept when: class balance and pT spectra per split are plotted and match.

## E3 — Tabular baseline

Compute ≥ 5 shower-shape features on every event (lesson §8: seed fraction,
core fraction, second-moment widths / eccentricity, tower multiplicity,
seed–subleading separation) and train a BDT with the Week-11 tooling. Tune on
the training set with CV; touch the test set once.
Hint: the second-moment eigenvalues are PCA on 64 pixels (Week 07) — build the
energy-weighted 2×2 covariance of tower positions and `np.linalg.eigh` it.
Reuse `XGBClassifier` + `StratifiedKFold` from Week 11 E4; this is Phase-1
muscle memory, not a new model class.
Accept when: ROC AUC on the test set is reported with the Phase-1 CV
discipline.

## E4 — CNN on tower images

Design a small CNN (Week-17 architecture arithmetic shown for your design —
layer table of shapes and parameter counts in the repo), train it on the raw
8×8 patches with the Week-18 recipe: log(1+E/E0) scaling, label-preserving
flips/rotations only, Adam, early stopping on validation AUC. Same split as E3.
Hint: three 3×3 convs, stride 1, give a receptive field of 7 towers — lesson
§9; pool at most once. Tens of thousands of parameters, not millions. If test
AUC does not beat E3, that is a finding — report the gap; do not retune against
the test set.
Accept when: training converges (curves shown) and test AUC ≥ tabular
baseline, or the gap is reported.

## E5 — Physics evaluation

On the same test events, plot background rejection at 90% signal efficiency vs
cluster energy, both models overlaid. Signal = single photon, background =
merged π⁰.
Hint: bin in pT (or cluster energy), and *inside each bin* find the score
threshold that keeps 90% of photons, then compute 1 / (π⁰ efficiency at that
threshold). Expect huge rejection at 4–8 GeV, decaying toward ~1 as merging
completes — that high-energy behavior is the point of the plot, not a bug.
Accept when: the plot exists and the high-energy behavior (merging → harder)
is visible and discussed in ≤ 3 lines.

## E6 — What does it see

Saliency map or occlusion scan on 4 correctly- and 4 wrongly-classified
clusters. Occlusion: zero out one tower (or a 2×2 block) at a time and map the
score drop.
Hint: if the CNN is winning, it should key on the subleading bump — two-shower
substructure — not on a noise tower in the corner. A network staring at
zero-suppressed edges is a more important finding than a 0.01 AUC.
Accept when: the figure exists and one line says whether the CNN attends to
the two-shower substructure.

## E7 — Writeup

~1 page in the repo (`writeup.md`): setup, results table, honest limitations
of the toy simulation. A stranger should be able to reproduce the headline
number from the repo.
Hint: the limitations paragraph is lesson §7's "what the toy keeps / what it
fakes," in your words. Answer the review question about "a likelihood you
wrote yourself" in that paragraph. Budget a real hour; this is a deliverable,
not a caption. Then: `pytest -q` green on the generator's kinematics, one
command reprints the comparison table.
Accept when: a stranger could reproduce the headline number from the repo.

## Review

1. (Week 17) Compute your network's receptive field at the last conv layer.
   Does it cover the typical photon-pair separation at the merging threshold?
2. (Week 12) Your Capstone-1 pipeline had nested CV. Why is a single fixed
   split defensible here, and what claim does it not license?
3. (Week 10) Why report rejection at fixed efficiency rather than accuracy,
   for a trigger-like physics use case?
4. (Week 08) The toy generator is a likelihood you wrote yourself. What does
   that mean for how seriously to take the CNN's advantage on real data?
