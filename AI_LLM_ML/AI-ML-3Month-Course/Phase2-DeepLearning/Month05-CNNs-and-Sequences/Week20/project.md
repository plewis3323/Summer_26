# Week 20 Project — EMCal Cluster CNN vs Tabular

## Objective

Build a classifier that tells a single photon from a merged π⁰ on simulated
electromagnetic-calorimeter images, two ways, on the *same events*: (1) hand-
designed shower-shape features into a Phase-1 BDT, and (2) a CNN on the raw
tower grid. The gate (from `03-Project-Roadmap.md`): **beat the tabular-
features baseline on the same events** — or understand precisely why not, in
writing. This is Month 05's deliverable and the dataset Capstone 2 option (a)
reuses as a point cloud.

## Background — the physics, from scratch

**A calorimeter is a camera for energy.** A collider detector is nested
cylinders around the beam pipe. Inner layers (trackers) record the curved
paths of charged particles. Further out sits the **calorimeter**, whose job is
the opposite: *stop* the particle and measure the energy it dumped while
stopping. An **electromagnetic calorimeter (EMCal)** is the variant built for
photons and electrons. It is segmented into **towers** — small cells tiling
the cylinder, each reporting one number, the energy it absorbed. A tower grid
with an energy per tower is a grayscale image: towers are pixels, energy is
intensity. That is why a CNN applies at all.

Positions on the cylinder use two coordinates. **φ (azimuth)** is the angle
around the beam, 0 to 2π. **η (pseudorapidity)** is a remapping of the polar
angle from the beam, η = −ln tan(θ/2); η = 0 is perpendicular to the beam
(**midrapidity**). The course's resident detector is **sPHENIX**, at RHIC (the
Relativistic Heavy Ion Collider at Brookhaven). Its EMCal towers span
Δη × Δφ ≈ 0.024 × 0.024. At radius R ≈ 1 m that is ≈ 2.4 × 2.4 cm cells. Your
images are 8×8-tower patches centered on the hottest tower (the **seed**) — a
~19 cm postage stamp, 64 pixels.

**Transverse momentum pT** is the momentum component perpendicular to the
beam. At midrapidity a photon's energy and its pT are numerically about equal;
this project generates at η = 0 and uses E and pT interchangeably.

**Why a photon makes a blob, not a single tower.** A high-energy photon in
dense matter **pair-produces** (γ → e⁺e⁻ near a nucleus); each electron then
**bremsstrahlung**-radiates a photon; those photons pair-produce again. The
particle count roughly doubles each step while the energy per particle halves
— an **electromagnetic shower**. Two material scales set its size: the
**radiation length X₀** (longitudinal; a 10 GeV shower is absorbed in ~20 cm,
all summed into the tower's one number) and the **Molière radius R_M**
(transverse; ~90% of the energy in a cylinder a couple of cm across —
comparable to one tower). So a single photon lights a compact, roughly
circular blob: most energy in the seed, the rest in the neighboring ring. The
toy parametrizes that blob as a 2D Gaussian of width σ ≈ 1.5 cm.

**The π⁰ and why two photons become one cluster.** Smash nuclei together and
most of the spray is pions. The **neutral pion (π⁰)** has mass
m c² = 135 MeV = 0.135 GeV, lives ~10⁻¹⁷ s, and decays 98.8% of the time to
two photons: π⁰ → γγ. The calorimeter never sees the π⁰ — only the pair.

Special relativity gives the opening angle. For two photons,
(m c²)² = 2 E₁ E₂ (1 − cos θ). At fixed total energy E = E₁ + E₂ the product
E₁ E₂ is largest when the energies are equal, which *minimizes* θ:

sin(θ_min / 2) = m c² / E    ⇒    θ_min ≈ 0.27 rad / E[GeV]   (E ≫ m c²).

The angle shrinks as 1/E. At R = 1 m the minimum photon separation is 6.8 cm
(~2.8 towers) at 4 GeV and 1.4 cm (~0.56 towers) at 20 GeV. Two Gaussian
blobs of width σ ≈ 1.5 cm stop looking like two blobs below ~2σ ≈ 3 cm —
around E ≈ 9 GeV, and steadily worse above. That is **merging**: one cluster,
photon-like, that a naive eye cannot tell from a single photon. Below the
turn-on the two photons resolve (easy); above it they are the impostor.

**Why anyone cares.** Heavy-ion collisions at RHIC create a droplet of
**quark–gluon plasma**. **Direct photons** — produced in the collision
itself, not from a decay — exit the droplet unmodified and are a core sPHENIX
probe. For every direct photon there are vastly more π⁰s. Resolved pairs can
be subtracted; a high-pT **merged π⁰ cluster is the fake**. The classical
defense is **shower-shape** cuts: a merged cluster is slightly wider and
slightly elongated. The modern question, asked this week on a toy you fully
control: given the raw tower image, does a CNN extract more separation than
those hand-built shapes?

**Shower shapes (the tabular baseline).** Compress each 8×8 image into ≥ 5
features a real analysis would cut on, for example: seed fraction
E_seed / E_tot; core fraction E_{3×3} / E_tot; energy-weighted second moments
(the 2×2 covariance of tower positions — its eigenvalues are PCA on 64
pixels, Week 07 — a single photon is round, a merged pair is elongated);
eccentricity (major − minor)/(major + minor); tower multiplicity above
threshold; seed–subleading-bump separation. Feed them to a Week-11 BDT with
Phase-1 CV discipline. This baseline is not a strawman.

**The CNN, sized by the physics.** Input 1×8×8. Three 3×3 convs (stride 1)
give a receptive field of 7 towers ≈ 17 cm, covering the ~1–3 tower
separations in the merging table; pool at most once. Tens of thousands of
parameters — 20k events of 64 pixels, Week 18's variance warnings in full.
Scale inputs as log(1 + E/E₀). Valid augmentations: φ-flips, η-flips, 90°
rotations (the toy is symmetric). Invalid: any zoom or stretch (width *is*
the label). Evaluate like a physicist: not accuracy, but **background
rejection at 90% signal efficiency, vs cluster energy** (Week 10). Signal =
single photon; background = merged π⁰.

## Data

You generate it. There is no download. The generator is the dataset, and the
kinematics tests *are* the data-integrity tests.

**Parameter table** (freeze these in config; do not retune them against the
test AUC):

| parameter | value | why |
|---|---|---|
| π⁰ mass | 0.135 GeV | PDG; the invariant-mass check |
| pT range | 4–24 GeV, uniform | walks merging from resolved to hopeless (lesson §5) |
| η | 0 | so E ≈ pT |
| calorimeter radius R | 1.0 m | converts angle → cm |
| tower pitch | 2.4 cm (Δη × Δφ = 0.024) | sPHENIX-like |
| patch | 8×8 towers around the seed | week README |
| shower σ | 1.5 cm | Molière-radius caricature |
| N events | ≥ 20 000, 50/50 photon / π⁰ | enough for a small CNN; class-balanced by construction |
| split | 60/20/20 train/val/test, stratified on pT bins | both models, same events |
| energy resolution | σ_E/E = 0.08/√(E/GeV) ⊕ 0.02 | typical EMCal stochastic ⊕ constant |
| electronics noise | Gaussian per tower, σ ~ few MeV | then zero-suppress below a small threshold |
| impact jitter | uniform within the seed tower | so the net cannot key on grid alignment |
| seed | documented in config | generator, split, and both trainings |

**What one event stores:** the 8×8 image, the class label, and the truth the
classifier never sees (pT, opening distance, the two photon energies) — you
need truth for the kinematics plot and the rejection-vs-energy bins.

**What the toy keeps / fakes** (write this in the limitations paragraph):
keeps the 1/E opening-angle kinematics (exact), the compact fixed-width blob,
sparsity, noise, and resolution. Fakes shower-shape fluctuations, tails,
depth, conversions, overlapping particles, and position-dependent response.
The CNN's advantage is an advantage *under this likelihood*. That is the
Week-08 review question, and it is why a win here does not license a claim
about real sPHENIX data.

**Stretch data, not the path:** the public CaloChallenge datasets
(Zenodo-hosted Geant4 showers) are the community benchmark, but they are not
a π⁰-vs-γ problem out of the box.

**Compute** (syllabus §8): flagged GPU week. The 8×8 CNN trains in minutes on
a laptop CPU; Colab/Kaggle if you take the pretrained-backbone stretch.

## Build steps

Do them in order; each is one of this week's exercises (E1–E7 in
`exercises.md`). The order is the method: generator and tests before the
dataset, tabular baseline before the CNN, both frozen before the test-set
physics plot.

Suggested layout — build it as the stages fill it in:

```
emcal-pid/
  pyproject.toml
  uv.lock
  run.py                 one command → comparison table + figures
  writeup.md
  src/emcal_pid/
    generate.py          E1
    dataset.py           E2
    features.py          E3
    train_bdt.py         E3
    train_cnn.py         E4
    eval.py              E5
    saliency.py          E6
  tests/
    test_kinematics.py   invariant mass, θ_min vs E, energy sum
  data/                  generated; not committed (provenance note is)
  models/
  figures/
```

1. **Generator** (E1). Implement the event loop in `project.md` Background /
   lesson §7: draw class and pT → decay (π⁰) or skip (photon) → project onto
   the grid → deposit Gaussians → smear, noise, zero-suppress → cut the 8×8
   seed patch. `tests/test_kinematics.py` asserts: reconstructed two-photon
   invariant mass equals 0.135 GeV to numerical tolerance; mean opening
   distance falls with pT; deposited energy (pre-noise) matches the photon
   energies. Accept when: the opening-distance-vs-pT plot exists and event
   images eyeball correctly (one compact blob vs two, then merged).
2. **Dataset** (E2). N ≥ 20k, 4–24 GeV, 50/50, 60/20/20 stratified on pT
   bins. Accept when: class balance and pT spectra per split are plotted and
   match.
3. **Tabular baseline** (E3). ≥ 5 shower shapes + Week-11 BDT, CV on train,
   test once. Accept when: test ROC AUC is reported with that discipline.
4. **CNN** (E4). Week-17 arithmetic in a layer table in the repo; Week-18
   recipe; same split. Accept when: training curves shown and test AUC ≥
   tabular, or the gap is reported.
5. **Physics evaluation** (E5). Rejection at 90% photon efficiency vs energy,
   both models, same test events. Accept when: the plot exists and the
   high-energy (merging → harder) behavior is visible and discussed in ≤ 3
   lines.
6. **Saliency** (E6). 4 correct + 4 wrong; occlusion or gradient. Accept when:
   the figure exists and one line says whether the CNN attends to two-shower
   substructure.
7. **Writeup + one command** (E7). `writeup.md` plus `run.py` that rebuilds
   (or reloads cached) models and reprints the table. Accept when: a stranger
   can reproduce the headline number from the repo.

## Acceptance gate (from `03-Project-Roadmap.md` and the week README)

- **Beat the tabular-features baseline on the same events.** Headline number:
  test AUC, CNN vs BDT, identical split. The money plot — rejection at 90%
  efficiency vs energy — is what a physicist actually consumes; a CNN that
  wins AUC but loses at high pT has not beaten the baseline where it matters.
  If the CNN does *not* beat the BDT, the gate is not met on the number — but
  the course's honesty policy still applies: write up the negative result and
  its diagnosis (capacity? the toy's Gaussian blobs are already summarized by
  the moments? augmentation bug?), fix the most likely cause, and rerun once.
  A CNN losing to a BDT on 64-pixel Gaussian blobs is a finding, not a
  humiliation (Week 16 taught you that tabular is BDT home turf).
- Tested and reproducible: `pytest -q` green on the generator's kinematics
  checks; fresh clone + `uv sync` + one command reprints the comparison table
  and writes the figures (training may load saved checkpoints; generation may
  be cached behind a hash of the config).
- The repo ships: generator, frozen split, both models, eval plots, saliency
  figure, writeup.

Then close the month: tag `month-05-complete`, write `retro.md`, open the
open-question issue. Keep the generator's API clean — Capstone 2 (a) will
import these events as a point cloud.

## Writeup requirements (one page)

- The task, in three sentences, and the headline table (AUC and rejection at
  90% efficiency, both models, same test events).
- Data: N, pT range, split, and the parameter table (or a pointer to config).
- The money plot, with the ≤ 3-line high-energy discussion from E5.
- Saliency: one line on whether it attends to two-shower substructure, with
  the figure.
- **Limitations of the toy** — what it keeps, what it fakes, and what that
  means for taking the CNN's advantage onto real data (the Week-08 review
  question, answered).
- **What failed first** — written the day it happened, not reconstructed.
- What you would do next with one more week.

## Stretch goals (only after the gate)

- **Pretrained backbone:** upsample the 8×8 patches and fine-tune a small
  ImageNet CNN (Week 18 E4 recipe). Does transfer help on blobs this unlike
  photographs, or is from-scratch the right tool? (Week 18 E5 is your
  planning input.)
- **Asymmetric-decay slice:** report rejection vs energy *and* vs energy
  sharing |E₁ − E₂| / (E₁ + E₂). The resolvable high-pT pairs are the
  asymmetric ones — does the CNN's edge live there?
- **Feature ablation:** drop the second-moment / eccentricity features from
  the BDT. How much of the tabular baseline *was* the elongation the CNN is
  supposed to be learning from pixels?
- **Point-cloud export:** write the above-threshold towers as `(log E, η, φ)`
  lists now, so Week 21 / Capstone 2 (a) does not have to reverse-engineer
  the generator.
