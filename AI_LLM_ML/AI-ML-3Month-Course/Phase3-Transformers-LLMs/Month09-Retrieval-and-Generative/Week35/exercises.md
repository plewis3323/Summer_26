# Week 35 — Exercises

Work top to bottom, derivations first: the forward-process closed form, the
reverse posterior, the chain ELBO reduced to L_simple, and the CFG formula go
on paper (README "Derivations") and are scanned into this folder before you
start E2. This is a flagship derivation week — file the scans for the
cold-redo rotation. Setup (imports, 2-D toy data, the noise schedule
β_t, plot scaffolds, seeds) is given by the notebook; you write only the
lines each exercise asks for. E1–E4 live in the notebook. E5 is GPU work
(syllabus §8 — Colab/Kaggle if the local GPU is slow) and may save a
checkpoint under `results/`. E6 is a markdown table from your own three
implementations.

## E1 — Forward-process check

Noise 2-D data (two-moons or a Gaussian mixture) two ways: step-by-step
(iterate x_t = √(1−β_t) x_{t−1} + √β_t ε_t) and via the closed form
x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε. Compare means and covariances at selected t.
Hint: lesson §3. ᾱ_t = ∏_{s=1}^t (1−β_s). Use the same x_0 batch for both
routes; the step-by-step path needs a fresh ε_t per step, the closed form
needs one ε — they are equal in distribution, not pathwise, so compare
Monte Carlo moments, not individual points. T = 1000, Ho's linear schedule
10^{-4} → 0.02, is the default.
Accept when: means/covariances agree within Monte Carlo error at
t ∈ {10, 100, 500}.

## E2 — Train a small MLP DDPM on 2-D data

Implement Algorithm 1 (train) and Algorithm 2 (sample) on the 2-D data with a
small MLP ε_θ(x_t, t). Sample a point cloud and histogram it against truth.
Hint: lesson §7 is a complete ~50-line template — write the training loop
and the sampling loop yourself from the boxed L_simple and Algorithm 2, do
not paste. Concatenate t/T as a feature; sinusoidal embeddings are optional
on 2-D. The single-Gaussian baseline is N(μ̂, Σ̂) fitted to the data — its
2-D histogram χ² against truth should lose to a model that recovered both
modes.
Accept when: sampled point cloud visually recovers both modes and a 2-D
histogram χ² against truth beats a single-Gaussian baseline.

## E3 — Sanity: ‖ε − ε_θ‖² vs t

Using the trained E2 model, plot the mean noise-prediction loss as a function
of timestep t.
Hint: freeze the net, draw (x_0, ε, t) as in training, bin the per-sample
loss by t. Two sentences: which timesteps are hardest, and does that match
the L_simple reweighting argument (lesson §6 — dropping the ELBO weight
up-weights high-t, where global structure is decided)? If your curve is
flat, the net is underfit or t is not being used.
Accept when: curve saved with two sentences on which timesteps are hardest.

## E4 — Conditional DDPM + classifier-free guidance

Train a conditional DDPM with class labels (a labeled 2-D mixture, or
MNIST-class images if you want pixels) and sample with CFG,
ε̃ = (1+w) ε_θ(x, t, c) − w ε_θ(x, t, ∅), sweeping w ∈ {0, 1, 3, 5}.
Hint: lesson §9 — drop the condition 10–20% of the time in training so one
net learns both scores. At w = 0 you recover the plain conditional model;
increasing w trades diversity for condition adherence. Show a grid, one row
per w. For physics later: high w that nails the mean and kills the width is
a failed generator — notice it here on a toy where you can see the collapse.
Accept when: sweeping w ∈ {0, 1, 3, 5} visibly trades diversity for
condition adherence, shown in a grid.

## E5 — Calo warm-up

Train a conditional DDPM on simplified shower arrays (Capstone 2 data or a
public CaloChallenge-style set), conditioned on incident energy. This is the
on-ramp to Capstone 3 option (b).
Hint: 8×8 (or flattened) showers from the Week-20/24 generator are enough;
a U-Net is optional, an MLP or small conv net with energy concatenated (or
embedded) is the warm-up. Log(1+E) the towers as in Week 20. Generate at
several condition energies and plot mean generated energy vs condition
against the diagonal. GPU: a few thousand steps on a T4-class GPU, not a
weekend. Save the checkpoint — Week 36 track (b) starts from it.
Accept when: generated mean energy response vs condition is monotonic and
within 10% of truth across the tested range.

## E6 — Comparison table

Diffusion vs flows (Week 23) vs VAE (Week 22) — likelihood access, sampling
cost, sample quality, conditioning ease. Fill from your own three
implementations; one cited claim max.
Hint: lesson §10's table is the skeleton; your numbers go in the sampling-
cost row (wall-clock for 1k or 10k samples, same hardware, eval mode). Do
not quote a paper's FID for "quality" — use your own eyeball-plus-χ² on the
2-D toy, plus the calo energy-response note from E5 vs the Week-22/23 shower
runs. The caveat on comparing NLL (flow) to a bound (VAE, DDPM) is the same
one as Week 23 E6.
Accept when: table filled from your own three implementations, one cited
claim max.

## Review

1. Week 22: write the VAE ELBO from memory. Point to the exact line where
   the diffusion bound is that ELBO with a fixed, non-learned encoder.
2. Week 23: flows require invertible maps with tractable Jacobians. Which
   constraint does diffusion drop, and what does it pay for that freedom at
   sampling time?
3. Week 08: composing Gaussians was the key trick today. Derive
   Var(aX + bY) for independent X, Y and connect to the ᾱ_t closed form.
4. Week 08: SGD with noise resembles Langevin dynamics. In one paragraph:
   where does noise appear in each, and what plays the role of temperature?
