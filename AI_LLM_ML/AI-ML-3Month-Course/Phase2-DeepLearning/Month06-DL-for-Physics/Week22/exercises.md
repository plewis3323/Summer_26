# Week 22 — Exercises

Work top to bottom. Setup (imports, the Week-20 shower images loaded and split,
constants, plot scaffolds) is given by the notebook; you write only the lines each
exercise asks for. Everything lives in the notebook this week; the paper ELBO
derivations (both routes, per `lesson.md` §4–5) are scanned into this folder before
you start E3.

## E1 — Plain autoencoder and where it breaks

Train an autoencoder (encoder → 2-d latent → decoder, MSE loss) on the Week-20
shower images. Scatter the test-set latents colored by class, then by cluster
energy. Decode 3 latent points sampled far from the occupied region and show them
next to real showers.
Hint: the model is the lesson's VAE class minus `to_logvar` and the sampling line.
Accept when: both scatter plots exist and one printed line states what the
off-manifold decodes look like and why (lesson §2.1).

## E2 — Closed-form Gaussian KL vs Monte Carlo

Implement the closed-form KL of lesson §6.1, then estimate the same KL by Monte
Carlo — average `log q(z) - log p(z)` over 100k samples $z \sim q$ — for
(μ, σ) = (0, 1), (2, 1), (0, 0.1).
Hint: build log-densities from the Gaussian formula; do not import a KL function.
Accept when: closed form and MC agree within 3 MC standard errors for all three
settings, and (0, 1) gives exactly 0 in closed form.

## E3 — The VAE

Extend E1 to a VAE: encoder outputs `mu` and `logvar`, reparameterized sampling,
loss = reconstruction + KL, both terms logged separately every epoch. Train on the
shower images, then decode 16 prior samples on a grid.
Hint: the loss is two lines once E2's formula is in hand; keep β = 1.
Accept when: total loss decreases, the separate rec/KL curves are plotted, and the
prior-sample grid shows shower-like blobs (not the E1 garbage).

## E4 — Break the gradient on purpose

Copy the E3 model but replace reparameterized sampling with sampling that detaches:
draw z with `torch.normal(mu.detach(), sigma.detach())`. Train briefly and compare
the gradient norm reaching the encoder's first layer against E3's.
Hint: `model.enc[0].weight.grad.norm()` after `backward()`.
Accept when: the detached version's encoder gradient norm is exactly 0 while E3's
is not, and one printed line names the derivation step this violates (lesson §7).

## E5 — Posterior collapse on demand

Sweep β over {0.5, 1, 4, 16, 64}, training the E3 model to convergence at each.
Plot final reconstruction error and final KL vs β on shared axes, and show one
prior-sample grid from the highest-β run.
Accept when: the plot shows KL falling toward 0 as β grows, the collapsed regime is
marked, and the high-β samples visibly lose diversity.

## E6 — Anomaly detection vs the supervised CNN

Train the E3 VAE on single-photon showers only. Score all test events (photon and
merged-π⁰) by reconstruction error and compute the ROC AUC for photon-vs-π⁰. Print
it next to your Week-20 CNN's supervised AUC.
Hint: the score is per-event summed squared error; labels only enter at evaluation.
Accept when: the anomaly AUC is > 0.5 and the printout ends with one line on the
size of the gap to the CNN and when the unsupervised approach is still the right
tool (lesson §9).

## Review

1. (Week 08) Derive in three lines why maximizing a Gaussian likelihood with fixed
   σ is minimizing squared error. (You used it again this week.)
2. (Week 12) In EM for a GMM, what does the E-step compute, and what plays that
   role in a VAE?
3. (Week 07) PCA is the optimal linear autoencoder under squared error. What
   objective did Week 07 maximize to derive it, and what was the solution?
4. (Week 16) Your training loss goes to NaN in epoch 1. List the first three
   things you check.
5. (Week 21) Why must a jet classifier's pooling operation be symmetric?
