# Week 23 optional — Normalizing Flows exercises

Scheduled Week 23 exercises are in `exercises.md` (SQL / FastAPI / Docker).
These are the flow exercises, for generative-sim tracks only.

# Week 23 — Exercises

Work top to bottom. Setup (imports, data loading, constants, plot scaffolds, the
seeded Breit–Wigner sampler for E1, and the Week-12-style GMM baseline fit for E3)
is given by the notebook; you write only the lines each exercise asks for.
Everything lives in the notebook this week; the paper derivations (change of
variables, coupling Jacobian, mask argument — per `lesson.md` §2–6) are scanned
into this folder before you start E2.

Data: E1 and E3–E4 use small datasets built in the notebook (Breit–Wigner samples,
two-moons). E5–E6 use the Week-20 toy generator's single-photon showers (the
notebook imports your Week-20 `src/`) and reload your trained Week-22 VAE for the
comparison. Calendar note: if this week is the one the syllabus §6 cut line hits,
E1–E3 plus the derivations are the minimum viable week.

## E1 — 1D warm-up: flow a Gaussian into a Breit–Wigner

Fit a 1D flow — a chain of 4 parametrized monotone maps — to samples from a
Breit–Wigner (M = 3.1, Γ = 0.3; lesson §4), training by exact NLL. Overlay the
learned density $p_Z(f(x))\,|f'(x)|$ on the analytic curve, and overlay a
histogram of flow samples on the training samples.
Hint: use maps of the form $a x + b + c\,\tanh(d x + e)$ with $a > |cd|$ enforced
(monotone, hence invertible); get $f'(x)$ from autograd on a batch of x values,
and sample by inverting each map with a few bisection steps (scaffold given).
Accept when: the KS-style comparison passes at the stated tolerance — the notebook
computes the KS statistic between 10k flow samples and the analytic CDF, and it is
below 0.02.

## E2 — Coupling layer forward/inverse in PyTorch

Implement the affine coupling layer (lesson §6 and §10: masked split, tanh-clamped
scale, forward returning `z, logdet`, and analytic `inverse`) and verify it both
ways on random inputs.
Hint: the log-det is `s.sum(dim=1)` — if you find yourself calling `torch.det`,
re-read lesson §6.
Accept when: `inverse(forward(x))` round-trips to 1e-5 and the log-det matches
`torch.autograd.functional.jacobian` on small inputs (the notebook compares your
`logdet` against `log|det|` of the autograd Jacobian on a few 4-dim examples).

## E3 — RealNVP on two-moons

Stack ≥ 6 coupling layers with alternating masks into a flow; train on two-moons
by exact NLL (lesson §10 is the template — write the training loop and the
layer-stacking yourself). Plot: samples over the data, and the held-out
average log-likelihood next to the notebook's Week-12 GMM baseline fit on the
same split.
Hint: accumulate the per-layer log-dets in the same loop that applies the layers;
report log-likelihood per point in nats so the GMM comparison is apples-to-apples.
Accept when: samples overlay the data convincingly and held-out log-likelihood
beats a Week-12 GMM baseline.

## E4 — Mask ablation: what a single mask cannot do

Retrain the E3 flow with all layers using the *same* mask (no alternation).
Plot samples from this flow next to E3's.
Hint: lesson §6's blind-spot argument tells you exactly which dimension to look at
in the sample plot.
Accept when: the untouched dimensions are visible in the sample plot and explained
in one line (which marginal is still standard-normal, and why no training step
could ever change it).

## E5 — Flow fast-sim of single-photon showers

Train the E3 flow architecture on Week-20 single-photon showers — flattened tower
energies, or per-cluster features (total E, width, position moments) if the
64-dim version trains badly. Generate 1k showers; show a sample grid next to real
showers and overlay per-tower (or per-feature) marginal histograms, generated vs
real.
Hint: log-scale the tower energies before the flow and note the preprocessing is
itself a change of variables (lesson §7); dequantize any exactly-zero towers with
small uniform noise.
Accept when: samples pass an eyeball test and per-tower marginals roughly match —
full physics validation is Week 24's job.

## E6 — Head-to-head: this week's flow vs last week's VAE

On the same held-out shower set, produce the comparison table: wall-clock time to
generate 10k samples (flow inverse pass vs VAE decoder pass), and exact NLL (flow)
vs ELBO (VAE), both in nats per event.
Hint: time only the generation passes, models in `eval` mode, `torch.no_grad()`;
the VAE column reuses your Week-22 loss code on held-out data.
Accept when: the table exists with a one-line caveat on comparing NLL to a bound
(lesson §8 says exactly what the caveat is). Keep the table — it is an input to
your Week-24 track choice.

## Review

1. (Week 06) State the two determinant facts this week's log-likelihood formula
   leans on: the determinant of a product, and the determinant of a triangular
   matrix.
2. (Week 08) Why does maximum likelihood work with the *log* of the likelihood,
   and why does that choice turn this week's per-layer volume factors into a sum?
3. (Week 13) Backprop multiplies local Jacobians along the computational graph.
   What does this week's flow do with the determinants of that same product?
4. (Week 22) One line: what problem does the reparameterization trick solve, and
   why does a flow trained by exact NLL never meet that problem?
5. (Week 12) How does a fitted GMM assign a log-likelihood to a held-out point —
   and what made it a fair baseline for E3?
