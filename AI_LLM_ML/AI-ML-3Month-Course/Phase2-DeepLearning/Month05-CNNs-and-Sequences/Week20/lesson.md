# Week 20 — Mini-Project: EMCal Cluster Classifier

~1–2 hrs of reading, then build — this is a project week, and `project.md` is the
spec. Before starting you should be able to: design a small CNN and do its shape and
receptive-field arithmetic (Week 17); apply the modern training recipe —
normalization, augmentation, transfer decisions, fixed splits (Week 18); train and
validate a gradient-boosted tree honestly (Week 11); read a ROC curve and choose an
operating point for a stated goal (Week 10).

## 1. This week is a build

For four weeks you have been assembling a toolkit: convolution and its arithmetic,
skip connections, the training recipe, honest evaluation. This week you point all of
it at one real physics problem and find out whether the CNN earns its keep.

The problem: a particle detector's electromagnetic calorimeter records energy on a
grid of cells — an image. A single high-energy photon makes one compact blob. A
particle called the π⁰ ("pi-zero") decays into *two* photons which, at high energy,
land so close together that their two blobs overlap into one — a cluster that fakes a
single photon. Your job: build a classifier that tells them apart, two ways —

1. **The Phase-1 way:** compute hand-designed summary features of each cluster
   ("shower shapes") and feed them to a BDT, exactly as in Weeks 10–12.
2. **The Phase-2 way:** feed the raw pixel grid to a CNN, trained with the Week-18
   recipe.

— and then compare them *on the same events*, like a physicist: not just AUC, but
background rejection at fixed signal efficiency, as a function of energy, plus a look
at what the CNN actually attends to. The roadmap criterion (`03-Project-Roadmap.md`):
beat the tabular baseline, or understand precisely why not.

Everything runs on simulated data you generate yourself, from a toy simulator you
will write — which means you also get the experience, standard in ML-for-science, of
building the dataset before you can build the model. The rest of this lesson teaches
the physics from zero: what a calorimeter is (§2–3), what a π⁰ is (§4), why the two
classes look alike (§5–6), how the toy simulation works (§7), and how each of the two
classifiers should be built and judged (§8–9). If a term isn't defined here, that's a
course bug — file it.

## 2. A camera for energy: calorimeters

A collider experiment smashes particles together and reconstructs what flew out. The
detector is built as nested cylindrical layers around the beam pipe, each layer
answering one question. Inner layers (*trackers*) record the curved paths of charged
particles without disturbing them much. Further out sits the layer this project is
about: the **calorimeter**, whose job is the opposite — *stop* the particle entirely
and measure how much energy it deposited while stopping. An **electromagnetic
calorimeter (EMCal)** is the variant specialized for photons and electrons.

A calorimeter is made of dense material that forces the incoming particle to dump its
energy (how, in §3), instrumented so the deposited energy is converted to light or
charge and read out. Crucially, it is *segmented*: the detecting volume is divided
into **towers**, small independent cells tiling the cylinder, each reporting one
number — the energy it absorbed in this event. A tower grid with an energy per tower
is, for our purposes, exactly a grayscale image: towers are pixels, energy is
intensity. That is why Week 17's machinery applies at all.

Where are the towers? Describing positions on a cylinder takes two coordinates:

- **φ (azimuth):** the angle around the beam axis, 0 to 2π — "which way around the
  barrel."
- **η (pseudorapidity):** a remapping of the polar angle θ (the angle from the beam
  axis), defined as $\eta = -\ln \tan(\theta/2)$. It is just a coordinate along the
  barrel's length: η = 0 points perpendicular to the beam (called **midrapidity**),
  large |η| points toward the beam directions. Physicists use η instead of θ because
  particle production is roughly uniform per unit η, so equal-η cells see comparable
  activity.

Towers are laid out on a uniform η–φ grid. The course's resident detector is
**sPHENIX**, an experiment at RHIC (the Relativistic Heavy Ion Collider at Brookhaven
National Lab, which collides heavy nuclei — more on why in §6). Its EMCal towers span
Δη × Δφ ≈ 0.024 × 0.024. At the calorimeter's radius, roughly R ≈ 1 m from the beam
line, an angle of 0.024 radians subtends about 2.4 cm — so picture towers as
≈ 2.4 × 2.4 cm cells, and a cluster of interest as living inside a small patch of
them. Following the week README, your images will be 8×8-tower patches centered on
the most energetic tower (the **seed**) — a 19-cm-wide postage stamp, and, at 64
pixels, the smallest images you will ever aim a CNN at.

One more standard word: **transverse momentum, pT** — the component of a particle's
momentum perpendicular to the beam. It is the standard measure of how violent the
process that made the particle was (momentum *along* the beam is mostly just the
incoming nuclei passing through). This project works at midrapidity (η ≈ 0), where a
particle's motion is essentially all transverse, so its energy and its pT are
numerically about equal — we will use E and pT interchangeably and generate the toy
data at η = 0 to keep it that way.

## 3. What a photon does inside: electromagnetic showers

Why does a photon make a *blob*, rather than lighting up a single tower or sailing
through? Two processes, each simple on its own, take turns:

- **Pair production:** a high-energy photon traveling through matter can convert into
  an electron and its antiparticle, a positron (allowed near a nucleus, which absorbs
  some recoil). One photon becomes two charged particles, each carrying roughly half
  the energy.
- **Bremsstrahlung** ("braking radiation"): a high-energy electron or positron
  deflected by a nucleus's electric field radiates a photon, handing it a substantial
  fraction of its energy.

Chain them: photon → e⁺e⁻ pair → each radiates photons → those photons pair-produce →
... The particle count roughly doubles at each step while the energy per particle
halves — an avalanche called an **electromagnetic shower**. The doubling stops when
the individual energies drop to the few-MeV scale, where the particles lose the rest
of their energy by ionizing atoms; in an instrumented calorimeter, a fixed fraction
of all that deposited energy becomes measurable light. Total light ∝ incoming energy —
that proportionality is the whole measurement principle.

Two material constants set the shower's size, and both matter for this project:

- **Radiation length X₀** — the longitudinal scale: the distance over which an
  electron radiates away all but 1/e of its energy (and ≈ 7/9 of a photon's mean
  distance before pair-converting). Dense calorimeters have X₀ of a couple of cm; a
  10 GeV shower is fully absorbed in ~20 cm of depth. Longitudinal development is
  invisible to us — a tower reports one depth-summed number.
- **Molière radius R_M** — the transverse scale: the radius of the cylinder
  containing about 90% of the shower's energy. For compact EMCals this is on the
  order of a couple of centimeters — *comparable to one tower*.

That last comparison is the design point of the whole detector, and of this project.
Because R_M ≈ tower size, a single photon deposits most of its energy in one tower,
spills the rest into the ring of neighbors, and produces a compact, roughly circular
blob spanning a few towers. For the toy simulator we will parametrize the transverse
energy profile as a 2D Gaussian bump with width σ of order 1.5 cm — a caricature
(real showers have a narrow core plus wider tails, and fluctuate event by event), but
one that preserves the feature the classifier lives on: *one photon → one compact
blob of known width*.

## 4. The π⁰, and the two lines of relativity we need

Smash two nuclei together and most of what sprays out is **pions** — the lightest
particles made of quarks (a quark and an antiquark bound together). They come in
charged varieties (π⁺, π⁻), which the tracker sees, and a neutral one, the **π⁰**,
which it does not. Particle masses are quoted in energy units via $E = mc^2$: an
electron's mass is 0.511 MeV, a proton's is 938 MeV, and the π⁰'s is

$$m_{\pi^0} c^2 = 135.0\ \text{MeV} = 0.135\ \text{GeV}$$

(1 GeV = 1000 MeV; a "10 GeV photon" carries 10 GeV of energy).

The π⁰ is spectacularly unstable: it lives ~10⁻¹⁷ seconds and decays, 98.8% of the
time, into **two photons**:

$$\pi^0 \to \gamma\gamma .$$

It decays essentially at the collision point, so the calorimeter never sees the π⁰
itself — only its two photons. Every π⁰ in your detector is a pair of photons flying
outward from the origin with some small angle between them.

How small an angle? Two facts from special relativity — the only two we need, both
usable with plain algebra:

1. For any particle, energy, momentum and mass are related by
   $E^2 = (pc)^2 + (mc^2)^2$. For a photon, $m = 0$, so $E = pc$.
2. For a *system* of particles, the combination
   $E_{\text{tot}}^2 - |\vec p_{\text{tot}}\,c|^2$ equals $(Mc^2)^2$, where M is the
   mass of whatever the system came from. This M is the **invariant mass** — the same
   quantity you histogrammed in Week 04's dimuon project to find the J/ψ peak.

Apply fact 2 to the two photons (energies E₁, E₂, opening angle θ between their
directions). Using $E_i = |\vec p_i| c$ and
$\vec p_1 \cdot \vec p_2 = |\vec p_1||\vec p_2| \cos\theta$:

$$(m_{\pi^0} c^2)^2 = (E_1 + E_2)^2 - |\vec p_1 c + \vec p_2 c|^2
= 2 E_1 E_2\, (1 - \cos\theta).$$

This one line is the toy simulator's beating heart, and the source of everything that
follows.

## 5. The opening angle shrinks with energy

Fix the π⁰'s total energy $E = E_1 + E_2$ and stare at
$1 - \cos\theta = (m c^2)^2 / (2 E_1 E_2)$. The right side is smallest when the
product $E_1 E_2$ is largest, and for a fixed sum a product is maximized when the
factors are equal (expand $(E_1 - E_2)^2 \ge 0$ if you want the one-line proof). So
the *minimum* opening angle occurs for the symmetric decay $E_1 = E_2 = E/2$:

$$1 - \cos\theta_{\min} = \frac{2 (mc^2)^2}{E^2}
\;\;\Longrightarrow\;\;
\sin\frac{\theta_{\min}}{2} = \frac{mc^2}{E},$$

using the half-angle identity $1 - \cos\theta = 2\sin^2(\theta/2)$. For
$E \gg mc^2$ the small-angle approximation gives the number to carry in your head:

$$\theta_{\min} \approx \frac{2\, m c^2}{E} = \frac{0.27\ \text{rad}}{E\ [\text{GeV}]}.$$

The angle shrinks like 1/E. Asymmetric decays open the angle up (smaller product
$E_1E_2$), at the price of one photon being soft — so the *typical* π⁰ at high energy
is a close pair, occasionally a wider pair with one faint partner.

Convert angles to distance on the calorimeter face at radius R = 1 m
(separation d ≈ R·θ, in 2.4-cm towers):

| π⁰ energy | θ_min (mrad) | separation d | in towers |
|---|---|---|---|
| 4 GeV | 68 | 6.8 cm | 2.8 |
| 8 GeV | 34 | 3.4 cm | 1.4 |
| 12 GeV | 23 | 2.3 cm | 0.94 |
| 16 GeV | 17 | 1.7 cm | 0.70 |
| 20 GeV | 14 | 1.4 cm | 0.56 |
| 24 GeV | 11 | 1.1 cm | 0.47 |

Now recall §3: each photon's blob has width σ ≈ 1.5 cm. Two Gaussian bumps stop being
visibly two bumps when their separation falls below roughly 2σ ≈ 3 cm — which the
table says happens around E ≈ 9 GeV and gets steadily worse above. Below that, the
π⁰'s photons resolve into two clusters (easy to reject — you can even compute the
pair's invariant mass and see the 135 MeV peak); above it they **merge** into one
cluster that a naive eye cannot tell from a single photon. This is why the project
generates events across a pT range of about 4–24 GeV: it walks the classifier through
the turn-on of the merging problem, from trivial to near-impossible.

## 6. Why anyone cares: direct photons and their impostor

The physics motivation, in three steps, each explained:

Colliding heavy nuclei at RHIC creates, for ~10⁻²³ seconds, a droplet of
**quark–gluon plasma (QGP)** — matter so hot that protons and neutrons melt into
their constituent quarks and gluons; the universe was in this state microseconds
after the Big Bang. You cannot see the droplet directly; you infer its properties
from what escapes.

**Direct photons** — photons produced *in the collision itself* (in the initial hard
scatter, or radiated thermally by the plasma) rather than from a decaying particle —
are prized messengers: photons don't feel the strong nuclear force, so they exit the
droplet unmodified, carrying pristine information from inside. Measuring them is a
core sPHENIX goal.

The catch: for every direct photon, collisions produce vastly more π⁰s, each spraying
photons of its own. Decay pairs that resolve can be identified and subtracted. The
dangerous background is exactly the §5 regime: **a high-pT merged π⁰ cluster is the
impostor** — one cluster, photon-like energy, wrong physics. The classical defense is
a set of hand-crafted **shower-shape** cuts (a merged cluster is slightly wider and
slightly elongated than a single photon's — §8 turns this into features). Your
question this week is the modern one: given the raw tower image, does a CNN extract
more separation than the hand-built shapes do? On LHC and RHIC experiments this exact
question has been asked in earnest; this week you ask it end-to-end on a toy you
fully control.

## 7. The toy simulation

Real experiments answer such questions with Geant4 — a full simulation that tracks
every particle through a detailed detector model, at ~seconds per event. You need
tens of thousands of labeled events on a laptop this week, so you will write a
parametrized toy instead. This is a respectable scientific move (fast simulation is
an entire research field — Week 23 and the CaloChallenge take it further); the
discipline is knowing what your toy keeps and what it fakes.

The generator, one event at a time (full parameter table in `project.md`):

1. **Draw the truth.** Choose the class label. Draw a cluster pT uniformly in the
   range where merging turns on (≈ 4–24 GeV), at η = 0 so E ≈ pT (§2).
2. **Decay it (π⁰ events).** In the π⁰'s rest frame the two photons go back-to-back
   with $E_\gamma = mc^2/2$ each, in a random direction (uniform on the sphere —
   the π⁰ has no memory of any axis). Transform to the lab: the standard boost
   formulas give the two lab-frame photon energies and directions; verify against §4's
   invariant-mass relation and §5's minimum angle — your `pytest` checks (see
   `project.md`) assert both. Single-photon events skip this step: one photon carries
   the full energy.
3. **Project onto the tower grid.** Each photon's line lands somewhere on the
   calorimeter face at R = 1 m; convert to (η, φ) tower coordinates. Jitter the
   impact point within the central tower so the network can't key on grid alignment.
4. **Deposit energy.** Lay each photon's energy down as a 2D Gaussian of width σ on
   the face, and integrate it over each 2.4-cm tower (the difference of Gaussian CDFs
   over the tower edges — `scipy.stats.norm.cdf` — not a sample per tower). Sum the
   two photons' deposits where they overlap; energy adds.
5. **Dirty it up.** Smear each photon's total energy by the calorimeter resolution
   (a √E-scaled Gaussian — real calorimeters measure a 10 GeV photon to a few
   percent); add independent Gaussian electronics noise to every tower; then
   **zero-suppress**: set towers below a small threshold to zero, as real readout
   does. This is why calorimeter images are mostly zeros with a few bright pixels.
6. **Cut the patch.** Find the seed (hottest tower), keep the 8×8 patch around it.
   Store the image, the label, and the truth (pT, opening distance, photon energies) —
   truth you will want for the physics plots even though the classifier never sees it.

What the toy keeps: the 1/E opening-angle kinematics (exact), the compact
fixed-width blob, realistic sparsity, noise, and resolution — the ingredients the
classification problem is actually made of. What it fakes: shower-shape
fluctuations (real showers vary event to event; Gaussians don't), tails, depth
information, conversions, overlapping unrelated particles, and position-dependent
detector response. Your writeup's limitations section (§10) should say exactly this —
and the Week-20 review question about "a likelihood you wrote yourself" is worth
answering in it.

If you want real-ish showers instead: the public **CaloChallenge** datasets
(Zenodo-hosted Geant4 showers; see `project.md`) are the community benchmark — but
their geometry is not a π⁰-vs-γ problem out of the box, so the course treats them as
a stretch goal, not the path.

## 8. The tabular baseline: shower shapes

The Phase-1 arm of the comparison compresses each 8×8 image into hand-designed
features — the same features experiments actually cut on. Compute at least five, for
example (E_tot = patch energy sum; positions in tower units):

- **Seed fraction** $E_{\text{seed}} / E_{\text{tot}}$ — one photon concentrates
  energy; a merged pair dilutes the leading tower's share.
- **Core fraction** $E_{3\times3} / E_{\text{tot}}$ — energy in the 3×3 around the
  seed; same idea, next scale out.
- **Second moments.** Treat the image as a probability distribution (weights
  $w_i = E_i / E_{\text{tot}}$): compute the energy-weighted mean position, then the
  2×2 covariance matrix of tower positions. Its eigenvalues (Week 07 — this is PCA on
  64 pixels) are the squared widths along the cluster's major and minor axes. A
  single photon is round (eigenvalues equal, both small); a merged pair is elongated
  along the photon-pair axis (large major, small minor).
- **Eccentricity** (major − minor)/(major + minor), the elongation in one number.
- **Tower multiplicity** above a small threshold — two blobs light more towers.
- **Seed–subleading separation**: distance from the seed to the highest tower that is
  a local maximum outside the seed's immediate neighbors — a direct two-bump probe,
  and essentially what real π⁰-splitting algorithms do.

Feed these to the Week-11 tooling (gradient-boosted trees), with the Phase-1
discipline: hyperparameters chosen by cross-validation on the training set, the test
set touched once, at the end. This baseline is not a strawman. Shower shapes encode
decades of physics insight, and on small tabular problems BDTs are brutal opponents —
Week 16's project taught you that. If your CNN beats it, that means something; that
is the point of building the baseline first.

## 9. The CNN, sized by the physics, judged like a physicist

**Architecture.** The input is 1×8×8 — do the Week-17 arithmetic *before* coding.
Three 3×3 convs (stride 1) give a receptive field of 7 towers ≈ 17 cm, comfortably
covering the ~1–3 tower photon separations in §5's table; one 2×2 pool after the
first conv or two is plenty (pool twice and an 8×8 input is down to 2×2 — there is
almost nothing left to convolve). Keep it small: tens of thousands of parameters, not
millions — your dataset is 20k events of 64 pixels, and Week 18's variance warnings
apply in full. Write the layer table (shapes + parameter counts) into the repo; the
week README asks for it explicitly.

**Input scaling.** Tower energies span three orders of magnitude and the image is
mostly zeros — feed it raw and the seed tower dominates every gradient. Use
$\log(1 + E/E_0)$ with $E_0$ around the noise scale, then standardize (Week 18 §6
flagged exactly this for calorimeter data).

**Augmentation.** Ask the Week-18 question — is the label invariant? Flips in φ and
(at midrapidity) in η: yes, the toy detector is symmetric under both, so flips and
90° rotations are free data. Any rescaling or stretching: **no** — width *is* the
label-carrying feature. Small translations: your seed-centering already fixes the
frame; don't fight it.

**Training.** The Week-18 recipe verbatim: fixed stratified train/val/test split,
Adam, early stopping on validation AUC, training curves saved. A single fixed split
(rather than Capstone-1's nested CV) is defensible because 20k events make the
val/test estimates tight and you are comparing two models on the *same* split — but
notice what it does not license: fine-grained claims about ±0.002 AUC differences.
(The week README's review asks you to spell this out.)

**Evaluation — the physicist's axes.** Define signal = single photon, background =
merged π⁰. Accuracy is the wrong summary (Week 10): the physics use case is "keep
90% of real photons, reject as many π⁰ fakes as possible." So report:

- ROC and AUC on the test set, both models, same events.
- **Background rejection at 90% signal efficiency** — rejection = 1/(background
  efficiency at that threshold) — *as a function of cluster energy*. This is the
  money plot: expect huge rejection at 4–8 GeV (resolved pairs), decaying toward ~1
  as merging completes, with the CNN–BDT gap (if any) opening in the middle.
- **What does it see:** occlusion scans (zero out one tower or 2×2 block at a time,
  map the score drop) or a gradient saliency map, on a few correctly and wrongly
  classified clusters. If the CNN is winning, this should show it keying on the
  subleading bump — if it shows the network staring at noise towers, you have learned
  something more important than an AUC.

## 10. How the week runs

Go to `project.md` — it is the full spec: parameter table, repo layout, build steps
with acceptance criteria, the roadmap gate, and the writeup requirements. The
exercise page (`exercises.md`) walks the same build as seven exercises; unusually for
this course, most of the work lives in the repo rather than the notebook, because the
deliverable is a small tested package (the generator's kinematics run under
`pytest`). Suggested rhythm for ~10 hrs: generator + tests and dataset in the first
third, baseline quickly after (it is Phase-1 muscle memory), CNN + evaluation in the
middle, saliency + writeup at the end. Budget the writeup a real hour; it is a
deliverable, not a caption.

Hardware: this is a flagged GPU week in the syllabus, but the images are 8×8 — a
laptop CPU can train this CNN in minutes. The flag exists for the stretch goals
(fine-tuning a pretrained backbone on upsampled patches); Colab/Kaggle free tier
covers those.

This week closes Month 05: tag `month-05-complete`, write the ~250-word `retro.md`,
and open the issue for the biggest thing you don't yet understand. And keep the
generator's API clean — Capstone 2, option (a), reuses these same events as a point
cloud for a GNN, and will ask whether graph structure beats your CNN.

## Check yourself

1. Why does a single photon light up a compact *group* of towers — not exactly one
   tower, and not the whole calorimeter? Name the two material scales involved.
2. Derive $\sin(\theta_{\min}/2) = mc^2/E$ from the invariant-mass relation. Which
   energy split between the photons achieves it, and why is it the minimum?
3. A 18 GeV π⁰ at R = 1 m: minimum photon separation in cm and in 2.4-cm towers?
   Resolved or merged, given σ ≈ 1.5 cm?
4. Why are merged π⁰s specifically a *high*-pT problem, and what happens to the decay
   photons' typical energy sharing as the pair gets wider?
5. Which two shower-shape features most directly encode "two blobs, not one," and
   what Week-07 machinery turns the image into the elongation measurement?
6. Your CNN has three stacked 3×3 convs, stride 1, no pooling before them. Receptive
   field in towers — and does it cover the photon separation at the ~9 GeV merging
   threshold?
7. For the calorimeter patches: give one valid augmentation and one invalid one, with
   the reason for each.
8. Why does this project report background rejection at 90% signal efficiency instead
   of accuracy?

## Answers

1. The shower cascade (pair production + bremsstrahlung) spreads energy transversely
   over about a Molière radius R_M — a couple of cm, comparable to one tower — so the
   core tower takes most of the energy and neighbors take the tails. Longitudinally
   the shower is contained within ~20 radiation lengths X₀, all summed into the
   tower's single readout.
2. $(mc^2)^2 = 2E_1E_2(1-\cos\theta)$ with $E_1 = E_2 = E/2$ gives
   $1 - \cos\theta = 2(mc^2)^2/E^2 = 2\sin^2(\theta/2)$, hence
   $\sin(\theta/2) = mc^2/E$. Equal sharing maximizes $E_1E_2$ at fixed sum (since
   $(E_1-E_2)^2 \ge 0$), which minimizes $1-\cos\theta$.
3. $\theta \approx 0.27/18 = 15$ mrad → d ≈ 1.5 cm ≈ 0.63 towers. Separation ≈ σ,
   well under 2σ: thoroughly merged.
4. The opening angle shrinks as 1/E, so only at high pT does the pair separation fall
   below the shower width. Wider-than-minimum pairs come from asymmetric decays —
   so the resolvable high-pT pairs tend to have one soft photon, which noise and
   zero-suppression can erase, converting them into single-blob impostors too.
5. Eccentricity (or the major-axis width) and the seed–subleading-bump separation;
   the elongation comes from eigendecomposing the energy-weighted 2×2 covariance of
   tower positions — PCA, Week 07.
6. Each 3×3 (stride 1) adds 2: 3 → 5 → 7 towers ≈ 17 cm. At the merging threshold the
   separation is ~2σ ≈ 3 cm ≈ 1.3 towers — covered many times over; receptive field
   is not this problem's bottleneck.
7. Valid: φ-flip (or η-flip, or 90° rotation) — the toy detector is symmetric under
   them, so the label is invariant. Invalid: any zoom/stretch — it changes the
   apparent shower width and pair separation, which is exactly the feature carrying
   the label; you would be manufacturing mislabeled data.
8. The physics use is asymmetric and threshold-based: the analysis needs a stated
   fraction of true photons kept (efficiency) while suppressing an overwhelming π⁰
   background; accuracy at 50/50 class balance reflects neither the real prevalence
   nor the chosen operating point. Rejection at fixed efficiency is the number the
   downstream measurement actually consumes (Week 10's trigger story).

## New terms

- **calorimeter / EMCal** — detector layer that stops particles and measures their
  deposited energy; the electromagnetic variant targets photons and electrons.
- **tower** — one readout cell of the calorimeter grid; one pixel of the energy image.
- **seed** — the highest-energy tower of a cluster; the patch is centered on it.
- **pseudorapidity η** — beam-axis coordinate $-\ln\tan(\theta/2)$; η = 0
  (midrapidity) is perpendicular to the beam.
- **azimuth φ** — angle around the beam axis.
- **transverse momentum pT** — momentum component perpendicular to the beam; ≈ E at
  midrapidity for light particles.
- **electromagnetic shower** — the pair-production/bremsstrahlung avalanche by which
  a photon or electron dumps its energy in dense material.
- **pair production / bremsstrahlung** — photon → e⁺e⁻ near a nucleus / deflected
  electron radiating a photon; the shower's two alternating steps.
- **radiation length X₀** — longitudinal shower scale (energy 1/e distance).
- **Molière radius R_M** — transverse shower scale (~90% containment radius).
- **π⁰ (neutral pion)** — lightest neutral quark–antiquark particle,
  $mc^2 = 135$ MeV; decays to two photons almost instantly.
- **invariant mass** — $\sqrt{E_{\text{tot}}^2 - |\vec p_{\text{tot}}c|^2}/c^2$ of a
  particle system; equals the parent's mass (Week 04's dimuon peak, §4's π⁰ relation).
- **opening angle** — angle between the two decay photons; minimum
  $2\arcsin(mc^2/E)$, shrinking as 1/E.
- **merged cluster** — two overlapping decay-photon showers reconstructed as one;
  the single-photon impostor.
- **direct photon** — photon produced in the collision itself, not from a decay; the
  QGP probe the merged π⁰ background contaminates.
- **quark–gluon plasma (QGP)** — deconfined hot quark matter created in heavy-ion
  collisions; sPHENIX's subject.
- **shower-shape variables** — hand-designed cluster features (fractions, moments,
  multiplicity) used for photon identification; the tabular baseline.
- **zero suppression** — dropping tower readings below threshold; the source of the
  images' sparsity.
- **Geant4** — the full detector-simulation toolkit the toy generator stands in for.
- **signal efficiency / background rejection** — fraction of signal kept at a
  threshold / one over the fraction of background kept; the physics operating point.

## Going deeper

- Particle Data Group review, "Passage of Particles Through Matter"
  (pdg.lbl.gov → Reviews) — the real physics behind §3: showers, X₀, R_M, resolution;
  read the electromagnetic-cascade section and skim the rest.
- The CaloChallenge 2022 community paper (search "Fast Calorimeter Simulation
  Challenge 2022", arXiv) — what full-fidelity shower datasets and their learned
  simulators look like; §7's toy sits at the bottom of this ladder, Week 23 climbs it.
- Any experiment's photon-identification or π⁰-discrimination note (search "photon
  identification shower shape variables" for ATLAS/CMS/PHENIX examples) — read for
  the feature definitions and the efficiency/rejection reporting style, not the
  results; compare their variables with §8's list.
- Karpathy, "A Recipe for Training Neural Networks" (blog) — the build-week
  checklist: start with the data, get a dumb baseline first, overfit one batch, then
  regularize; this week is a full rep of exactly that discipline.
