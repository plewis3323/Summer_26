# Week 23 — Normalizing Flows + Fast Simulation (optional)

This lesson is **optional**. The scheduled Week 23 is now software engineering
for data and services (`lesson.md` in this folder). Do this file if your
Capstone 2 or 3 is a generative fast-sim track, or in slack time. Week 35
still compares flows vs VAEs vs diffusion at survey depth.

# Week 23 — Normalizing Flows + Fast Simulation

~3 hrs reading, plus the paper-and-pencil pass. Before starting you should be able
to: state what a determinant measures — the factor by which a linear map scales
volumes — and compute the determinant of a triangular matrix (Week 06); use the
chain rule for compositions, including the matrix form from the backprop derivation
(Weeks 05, 13); state maximum likelihood and explain why we work with log-likelihoods
(Week 08); and derive the ELBO both ways and say exactly why the VAE needed it
(Week 22).

## 1. The itch the VAE left

Week 22 hit a wall and went around it. The latent-variable model's likelihood

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

was intractable, so we never computed it — we maximized a lower bound instead, and
route 2 told us exactly what we gave up: the ELBO sits below $\log p_\theta(x)$ by
$\mathrm{KL}(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x))$, a gap we can neither compute
nor drive provably to zero.

A **normalizing flow** takes the wall down instead. The idea: make the map between
latent space and data space a single **invertible** function. If every $x$
corresponds to exactly one $z = f(x)$ — no integral over "which latents could have
produced this $x$", because precisely one did — then there is nothing to
marginalize, nothing to approximate, and no encoder to train separately. The price
is a real one: the network must be invertible by construction, the latent space must
have the *same* dimension as the data (no bottleneck), and we must be able to track
how the map warps probability. That warping bookkeeping is the change-of-variables
formula, and it is this week's derivation.

Two payoffs make the price worth discussing:

- **Exact likelihood.** A flow evaluates $\log p_\theta(x)$ exactly — trainable by
  plain maximum likelihood (Week 08), comparable across models in actual nats, no
  bound, no gap.
- **Sampling is one pass.** Draw $z$ from a simple base distribution, run the
  inverse map, get a sample. That combination — exact density *and* fast sampling —
  is why flows became a leading tool for learned simulation, which §9 turns to.

The name, read backwards: a flow "normalizes" data — running $f$ forward turns the
complicated data distribution into a standard normal, one invertible layer at a
time.

## 2. Change of variables in one dimension

Start where every probability-density manipulation starts: conservation of
probability mass.

Setup: a random variable $Z$ with a known density $p_Z$ (say a standard normal),
and an invertible, differentiable map so that $X = g(Z)$ — equivalently
$Z = f(X)$ with $f = g^{-1}$. Question: what is the density $p_X$ of $X$?

The wrong guess is $p_X(x) = p_Z(f(x))$ — just relabel the axis. Here is why it
fails. A density is not a probability; only its integral over an interval is.
Probability mass is what must be conserved: the mass that $Z$ puts on a tiny
interval must equal the mass $X$ puts on that interval's image, because the map
carries outcomes one-to-one:

$$P\big(x \le X \le x + dx\big) = P\big(z \le Z \le z + dz\big),$$

where $[z, z+dz]$ is the image of $[x, x+dx]$ under $f$. Write each side as
density × width (the definition of a density on a shrinking interval):

$$p_X(x)\,|dx| = p_Z(z)\,|dz|.$$

The absolute values are there because a decreasing $f$ flips the interval; width is
positive either way. Divide:

$$p_X(x) = p_Z\big(f(x)\big)\,\left|\frac{dz}{dx}\right| = p_Z\big(f(x)\big)\,\big|f'(x)\big|.$$

That is the **change-of-variables formula** in 1D. The derivative is not decoration
— it is the local stretch factor: where the map stretches an interval, the same
mass is spread over more width and the density must drop by exactly that factor.
Run it with numbers once and it sticks: take $Z \sim \mathcal{N}(0,1)$ and
$X = 2Z$, so $f(x) = x/2$, $|f'| = 1/2$. Then
$p_X(x) = p_Z(x/2)\cdot\tfrac12$: the distribution of $X$ is twice as wide, so its
density is half as tall — mass conserved. The formula does exactly what a histogram
would do if you relabeled its axis and re-normalized the bin widths.

Sanity checks worth doing on paper now (they are the E1 warm-up's foundation):

- $f(x) = (x - \mu)/\sigma$ turns $\mathcal{N}(\mu, \sigma^2)$ into
  $\mathcal{N}(0,1)$ — and the $1/\sigma$ prefactor of the Gaussian density is
  precisely the $|f'|$ factor. You have been using the change of variables since
  Week 08 without the name.
- Invertibility is load-bearing: if $f$ folds two $x$-regions onto one $z$-region
  (like $x^2$ on all of $\mathbb{R}$), masses add and the one-to-one bookkeeping
  breaks. Monotone functions are exactly the invertible ones in 1D — which is why
  E1 builds its flow from monotone maps.

## 3. Many dimensions: enter the Jacobian determinant

In $D$ dimensions the map $f: \mathbb{R}^D \to \mathbb{R}^D$ has a **Jacobian**:
the $D \times D$ matrix of all partial derivatives,
$J_f(x)_{ij} = \partial f_i / \partial x_j$ — the same object that showed up in the
matrix form of backprop (Week 13: each layer's local linearization). Near a point
$x$, $f$ acts like the linear map $J_f(x)$: it takes a tiny cube of volume $dV$ to
a tiny parallelepiped of volume $|\det J_f(x)|\,dV$. That is Week 06's geometric
meaning of the determinant, doing real work: **the determinant is the local volume
scale factor**, the $D$-dimensional generalization of $|f'|$. (Week 07's SVD view
says the same thing by decomposition: rotate, stretch each axis by a singular
value, rotate — and $|\det| = \prod_k \sigma_k$, the product of the stretches.)

Conservation of mass, verbatim from §2 but with volumes instead of widths:

$$p_X(x)\,dV_x = p_Z(z)\,dV_z, \qquad dV_z = |\det J_f(x)|\, dV_x$$

$$\boxed{\; p_X(x) = p_Z\big(f(x)\big)\,\big|\det J_f(x)\big| \;}$$

Take the log — always, per Week 08 (products of many small numbers underflow; sums
of logs do not):

$$\log p_X(x) = \log p_Z\big(f(x)\big) + \log\big|\det J_f(x)\big|.$$

### 3.1 Composition: deep flows

One invertible map is not expressive. Compose $K$ of them,
$f = f_K \circ \cdots \circ f_1$ (each an invertible "layer"), and let
$h_0 = x,\; h_k = f_k(h_{k-1}),\; h_K = z$. Two Week-06/13 facts finish the job:

- Chain rule, matrix form: $J_f = J_{f_K} \cdots J_{f_1}$ (evaluated at the
  intermediate points $h_{k-1}$) — the same product of local Jacobians that
  backprop multiplies.
- Determinant of a product = product of determinants (Week 06).

Therefore:

$$\log p_X(x) = \log p_Z(z) + \sum_{k=1}^{K} \log\big|\det J_{f_k}(h_{k-1})\big|.$$

That is the entire training objective of a normalizing flow. Choose the **base
distribution** $p_Z = \mathcal{N}(0, I)$, so
$\log p_Z(z) = -\tfrac12\|z\|^2 - \tfrac{D}{2}\log 2\pi$, and maximize
$\sum_i \log p_X(x_i)$ over the training set — exact maximum likelihood, the thing
Week 22 could not have.

One naming convention to fix, because the literature uses both directions: we call
$f: x \to z$ the **normalizing direction** (data to Gaussian; used to compute the
likelihood at training time) and $g = f^{-1}: z \to x$ the **generative direction**
(sample $z$, produce data). Papers that define the model as $x = g(z)$ write the
same formula with $\log|\det J_g|$ *subtracted*; it is the identity
$\det J_{f^{-1}} = 1/\det J_f$ wearing a different sign, nothing more.

## 4. You have done this before: Jacobians in physics

The formula is not new machinery — it is the bookkeeping from every coordinate
change you have ever integrated through. Switching a 3D integral to spherical
coordinates inserts $r^2 \sin\theta\, dr\, d\theta\, d\phi$: that factor *is*
$|\det J|$ for the map from spherical to Cartesian coordinates, accounting for how
much volume a coordinate cell $(dr, d\theta, d\phi)$ actually occupies. In particle
physics the same move runs through every **phase-space integral** — an integral
over the allowed final-state momenta of a reaction — where changing variables (say,
to invariant masses and angles) always drags a Jacobian along. A flow is that
Jacobian discipline made learnable: instead of a coordinate change you chose for
convenience, a neural network learns the coordinate change that makes your data
look Gaussian.

One physics distribution to meet now, because E1 fits a flow to it. An unstable
particle does not have a single sharp mass: quantum mechanics spreads the measured
masses of its decays around the nominal value $M$ with a width $\Gamma$ set by its
lifetime. The line shape is the **Breit–Wigner** (a Lorentzian):

$$p(m) \;\propto\; \frac{1}{(m - M)^2 + \Gamma^2/4}.$$

The J/ψ peak you fit back in Week 04's dimuon project was one of these (smeared by
detector resolution). It is a good flow target because its tails are far heavier
than a Gaussian's — the learned map must stretch hard out there, and you can watch
it do so.

## 5. The design problem

Everything now reduces to choosing the layers $f_k$. Each must satisfy three
demands simultaneously:

1. **Expressive** — stacked, they must warp a Gaussian into showers, jets, moons.
2. **Invertible** — with an inverse cheap enough to actually run (sampling runs
   it), not just to exist in principle.
3. **Tractable Jacobian determinant** — $\log|\det J|$ appears in the loss of
   every training step. A general $D \times D$ determinant costs $O(D^3)$, and
   autograd through it is worse; at shower dimensionalities that is dead on
   arrival.

An ordinary MLP layer fails 2 (ReLU throws information away; even without it,
inverting a general learned map means solving equations numerically). A general
invertible map fails 3. The escape route comes from Week 06: **the determinant of a
triangular matrix is the product of its diagonal entries** — $O(D)$, no elimination
required. So: engineer layers whose Jacobian is triangular *by construction*, and
the log-det becomes a sum of logs of diagonal entries. The best-known such layer is
the coupling layer.

## 6. Affine coupling layers (RealNVP)

The construction (Dinh, Sohl-Dickstein & Bengio's Real NVP — "real-valued
non-volume-preserving") is one clean trick: **split the dimensions in two; let one
half pass through untouched; transform the other half elementwise, with parameters
computed from the untouched half.**

Split $x \in \mathbb{R}^D$ into $x_A = x_{1:d}$ and $x_B = x_{d+1:D}$ (any fixed
partition — a **mask** — works). The layer computes

$$y_A = x_A, \qquad y_B = x_B \odot \exp\big(s(x_A)\big) + t(x_A),$$

where $s$ and $t$ ("scale" and "translate") are **arbitrary neural networks**
mapping $\mathbb{R}^d \to \mathbb{R}^{D-d}$, and $\odot$ is the elementwise
product. Check the three demands:

**Invertible — analytically.** Given $y$, you know $x_A = y_A$; so you can
recompute $s(x_A)$ and $t(x_A)$ — the *same* values the forward pass used — and
solve elementwise:

$$x_A = y_A, \qquad x_B = \big(y_B - t(y_A)\big) \odot \exp\big(-s(y_A)\big).$$

Note what was *not* required: inverting $s$ or $t$. They can be any networks,
arbitrarily deep and non-invertible — they only ever run forward, in both
directions of the flow. The $\exp$ is doing quiet work too: $e^{s} > 0$ always, so
the elementwise scale can never be zero and invertibility can never silently break.

**Triangular Jacobian.** Order the variables $(x_A, x_B)$ and differentiate:

$$J = \begin{pmatrix} \dfrac{\partial y_A}{\partial x_A} & \dfrac{\partial y_A}{\partial x_B} \\[2mm] \dfrac{\partial y_B}{\partial x_A} & \dfrac{\partial y_B}{\partial x_B} \end{pmatrix} = \begin{pmatrix} I & 0 \\ \dfrac{\partial y_B}{\partial x_A} & \mathrm{diag}\big(\exp(s(x_A))\big) \end{pmatrix}$$

Top-left: identity ($y_A = x_A$). Top-right: zero — $y_A$ does not look at $x_B$,
and this single zero block is what makes everything work. Bottom-left: some dense,
complicated block full of network derivatives — *and we never compute it*, because
the determinant of a block-triangular matrix ignores it (Week 06: expand along the
top rows). Bottom-right: diagonal, because the transform of $x_B$ is elementwise.
So

$$\det J = \det(I)\cdot\det\big(\mathrm{diag}(e^{s})\big) = \prod_j e^{s_j(x_A)}, \qquad \log\big|\det J\big| = \sum_{j} s_j(x_A).$$

The log-determinant is *the sum of the scale-network's outputs*. No determinant is
ever computed; the loss term is a `sum()`. This is the whole reason the
architecture exists.

**Expressive — only when stacked.** A single coupling layer has an obvious blind
spot: $y_A = x_A$ *identically*, so the marginal distribution of the pass-through
half is untouched — provably, not approximately. No amount of training moves it,
because no parameter touches it. The fix is **alternating masks**: the next layer
swaps the roles ($B$ passes through, $A$ is transformed conditioned on $B$), so
after two layers every dimension has been transformed at least once, conditioned
on the other half. The composition's Jacobian is a product of lower- and
upper-triangular-style factors and is no longer triangular — but we never need the
composed determinant as a matrix computation, because §3.1 already turned it into
a sum of per-layer log-dets, each still a plain sum. Six-to-ten coupling layers
with alternating (or permuted) masks is a standard recipe; RealNVP's image version
alternates checkerboard and channel masks, and later flows (Glow) learn the
permutations. E4 makes you train the no-alternation version and watch the frozen
half stay frozen.

### 6.1 One training pathology to expect

The loss contains $e^{s}$ and $\sum s$. If the scale network's raw output drifts
large — and unconstrained network outputs do (Week 16, activation statistics) —
$e^s$ overflows, the inverse pass produces garbage, and the loss NaNs; if it
drifts very negative, the layer collapses volume and the log-det term tanks.
Standard hygiene is to bound the scale: $s = c \cdot \tanh(\tilde{s})$ with the
raw network output $\tilde{s}$ and a small constant like $c = 2$, capping the
per-layer per-dimension stretch at $e^{\pm 2}$. This is the "tanh clamp" the week
README's review question points at — Week 16's lesson (keep activations in a sane
range or watch the run die) applied to a new architecture. Watch the $s$
statistics in training exactly the way you watched activation histograms.

## 7. Training and sampling, end to end

Assemble the pieces. With layers $f_1 \dots f_K$ (couplings with alternating
masks), base $\mathcal{N}(0, I)$, and a batch $\{x_i\}$, training minimizes the
exact negative log-likelihood:

$$\mathcal{L} = -\frac{1}{N}\sum_i \left[ -\tfrac12\|f(x_i)\|^2 - \tfrac{D}{2}\log 2\pi + \sum_{k=1}^{K} \text{sum-of-}s\text{-outputs}_k \right]$$

Read it as two competing forces, the flow's whole story in one line: **pull every
data point to the origin in $z$-space** (the $\|z\|^2$ term wants the data mapped
into the Gaussian's bulk) **without cheating by crushing volume** (the log-det
term charges for compression — mapping everything to the origin would need
$s \to -\infty$, which the log-det penalizes linearly). A model that ignores the
second term is the degenerate "everything is likely because I squeezed space"
solution; the Jacobian term is what makes likelihood mean something.

Sampling is the inverse pass: $z \sim \mathcal{N}(0, I)$, then apply the layer
inverses in reverse order, $x = f_1^{-1}(\cdots f_K^{-1}(z))$. For coupling
layers the inverse costs the same as the forward — one $s,t$ network evaluation
per layer — so generation is a single fast pass. (Not every flow family shares
this: autoregressive flows like MAF are fast to train and $D$-times slower to
sample, or vice versa for IAF. Coupling layers are the both-directions-cheap
compromise, which is exactly why fast simulation likes them.)

Two practical notes, one line each. Images and calorimeter energies are often
preprocessed with a log or logit transform — that transform is itself a change of
variables, and its Jacobian must be added to the reported likelihood or the
numbers are meaningless (a classic paper bug). And discrete-valued data
(integer counts, 8-bit pixels) needs **dequantization** — adding small uniform
noise — because a continuous density on discrete points can spike to infinite
likelihood; flows are continuous-density machines.

## 8. Flows vs VAEs: the honest scorecard

You now own both generative families this month teaches. The comparison that
matters for Week 24's choice (and E6 measures it):

| | VAE (Week 22) | Flow (this week) |
|---|---|---|
| Likelihood | ELBO — a lower bound, gap unknown | exact $\log p(x)$ |
| Training signal | reconstruction + KL, needs reparameterization | plain maximum likelihood |
| Latent dimension | free — a true bottleneck ($d \ll D$) | forced equal to $D$ |
| Sampling | one decoder pass | one inverse pass (coupling flows) |
| Architecture freedom | any encoder/decoder | invertible layers only |
| Failure you met | posterior collapse | frozen dimensions, scale blow-ups |

Read the table with physics eyes. The VAE's bottleneck is a *feature* when the
data truly lives near a low-dimensional manifold (a shower is largely described
by energy, position, width — Week 22's E1 showed the latent organizing itself
that way); the flow spends parameters maintaining a bijection through all $D$
dimensions, including the noise ones. The flow's exact likelihood is a *feature*
when you need calibrated densities — anomaly scores that mean something,
importance weights, direct model comparison in nats. Neither dominates. And a
warning for E6's table: an ELBO and an exact NLL are not the same quantity — the
VAE's true likelihood is *better* than its ELBO by an unknown gap — so "flow NLL
beats VAE ELBO" is evidence, not proof, that the flow is the better density.
Say so in the one-line caveat the exercise demands.

(Where is diffusion? Week 35. It will look like a flow with infinitely many tiny
noisy layers and no invertibility constraint — the third corner of the
generative-model triangle, with its own price: slow sampling.)

## 9. Why colliders need learned simulators

Now the application that motivates the whole month's generative thread.

**What simulation is for.** Collider physics runs on comparison: you cannot claim
an excess, measure an efficiency, or train a photon-vs-π⁰ classifier without
knowing what standard physics *should* look like in your detector. That knowledge
comes from simulation: generate collisions from known physics, then track every
resulting particle through a virtual copy of the detector. The workhorse for the
detector half is **Geant4** — a simulation toolkit that follows each particle
step by step through the detector material, rolling dice for every physical
process (ionization, bremsstrahlung, pair production, nuclear interactions) along
the way. It is the trusted ground truth, and it is slow.

**Why calorimeters are the bottleneck.** A **calorimeter** is the detector layer
that measures energy by absorption (Week 20 taught the EMCal version): an
incoming high-energy particle interacts, produces secondaries, which produce more
secondaries — a **shower** that multiplies one particle into thousands. Geant4
must follow every one of them. A single high-energy shower can take seconds to
minutes of CPU; calorimeter showers dominate the full-simulation budget of the
LHC experiments, and simulation in total consumes of order half their computing.
The collaborations need billions of simulated events to keep systematic
uncertainties below statistical ones, and the planned high-luminosity LHC
upgrade multiplies the data — the projected simulation demand outruns any
plausible computing budget. Classical fast simulation (parametrized shower
shapes, lookup libraries of frozen showers) buys speed by giving up accuracy in
the tails and correlations — often exactly where the physics is.

**The learned-simulator bet.** Train a generative model on Geant4 showers,
conditioned on the incoming particle's energy (and angle, and impact position);
then sample it at generation time instead of re-running the microphysics.
Thousandfold speedups are routine; the entire question is whether the samples
are *right*. A flow is a natural candidate: exact likelihood for principled
training, one-pass sampling for speed. **CaloFlow** (Krause & Shih) was the
proof of concept — a flow generating calorimeter showers that, for the first
time, a classifier struggled to tell from Geant4's.

**The CaloChallenge.** The "Fast Calorimeter Simulation Challenge 2022" turned
this into a community benchmark: public Geant4 shower datasets of increasing
dimensionality (from a few hundred voxels — 3D pixels of deposited energy — to
tens of thousands), on which anyone can train a generator and everyone is judged
the same way. Internalize the judging criteria; they are Week 24 track (b)'s
validation suite in embryo:

1. **Physics observables, as histograms.** Not "do samples look right" but: total
   and per-layer energy distributions, shower widths and depth profiles, sparsity
   — generated vs Geant4, with quantified distances.
2. **The classifier two-sample test.** Train a classifier (you have built many)
   to distinguish generated showers from Geant4 showers. If the generator is
   perfect the task is impossible and test AUC $\approx$ 0.5; every bit above
   0.5 measures a detectable flaw. This is the field's sharpest single metric —
   an adversary you did not train against.
3. **Cost.** Sampling time and memory versus Geant4 — the entire point of the
   exercise; a slow fast-sim is nothing.

Notice what the criteria are not: image-quality metrics, eyeball grids, loss
values. Physics validation means the *distributions a physicist would cut on*
match. Carry that standard into Week 24 whichever track you choose.

## 10. Worked example: a coupling flow in ~60 lines

Complete and runnable as shown; `uv add torch scikit-learn`. Two-moons is the
E3 dataset — 2D so you can *see* the learned warp; swap in shower features for
E5.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

torch.manual_seed(0)
D = 2

xy, _ = make_moons(n_samples=4000, noise=0.06)
data = torch.tensor(xy, dtype=torch.float32)
data = (data - data.mean(0)) / data.std(0)


class Coupling(nn.Module):
    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)          # 1 = pass through
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x):                           # x -> z, plus log|det J|
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=1)
        s = 2.0 * torch.tanh(s) * (1 - self.mask)   # tanh clamp (lesson 6.1)
        t = t * (1 - self.mask)
        z = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return z, s.sum(dim=1)                      # log-det = sum of s

    def inverse(self, z):
        zm = z * self.mask
        s, t = self.net(zm).chunk(2, dim=1)
        s = 2.0 * torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return zm + (1 - self.mask) * ((z - t) * torch.exp(-s))


masks = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])] * 3
layers = nn.ModuleList([Coupling(D, 64, m) for m in masks])   # 6 layers
opt = torch.optim.Adam(layers.parameters(), lr=1e-3)

for step in range(3000):
    idx = torch.randint(0, len(data), (256,))
    z, logdet = data[idx], torch.zeros(256)
    for layer in layers:                            # normalizing direction
        z, ld = layer(z)
        logdet = logdet + ld
    log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * D * torch.log(torch.tensor(2 * torch.pi))
    loss = -(log_pz + logdet).mean()                # exact NLL, in nats
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 999:
        print(f"step {step + 1}  NLL {loss.item():.3f}")

with torch.no_grad():                               # generative direction
    x = torch.randn(2000, D)
    for layer in reversed(layers):
        x = layer.inverse(x)
print("sample mean:", x.mean(0), " sample std:", x.std(0))
```

Map every line to the math: the loop over layers accumulates §3.1's sum of
log-dets; `s.sum(dim=1)` is §6's triangular-determinant result; the clamp is
§6.1; `log_pz` is the standard-normal log-density; the loss is the exact NLL —
compare against Week 22, where the analogous line was a bound. Sampling reverses
the layer list and calls `inverse`, §7. Round-tripping `inverse(forward(x))` is
E2's first check.

## Check yourself

1. Derive the 1D change-of-variables formula from conservation of probability
   mass. Why does the formula fail if the map is not invertible?
2. What geometric quantity does $|\det J_f(x)|$ measure, and which two Week-06
   facts turn a $K$-layer flow's log-likelihood into a sum?
3. Why can a flow train by exact maximum likelihood when the VAE could not?
   Point at the structural difference, not the formulas.
4. In a coupling layer, which block of the Jacobian is never computed, and why
   is that legal?
5. Why does the coupling inverse never require inverting the networks $s$ and
   $t$?
6. Prove in two lines that a single coupling layer cannot change the marginal
   distribution of its pass-through half, and name the fix.
7. Your flow's loss goes NaN in epoch 1. Name the likeliest culprit term and
   the standard mitigation.
8. Name the three axes on which CaloChallenge submissions are judged, and say
   why "the samples look right" appears on none of them.

## Answers

1. Mass on a tiny interval is conserved under a one-to-one map:
   $p_X(x)|dx| = p_Z(f(x))|dz|$, so $p_X(x) = p_Z(f(x))\,|f'(x)|$. Without
   invertibility, several $x$-regions send mass to the same $z$-region, the
   one-to-one accounting breaks, and contributions would have to be summed over
   branches.
2. The factor by which $f$ scales an infinitesimal volume at $x$. Determinant of
   a product = product of determinants (with the chain rule giving the Jacobian
   product), and $\log$ of a product = sum of logs.
3. The flow's data-to-latent map is a bijection: each $x$ has exactly one $z$,
   so $p(x)$ needs no integral over latents and no approximate posterior. The
   VAE's decoder is many-to-one with a lower-dimensional $z$, so evaluating
   $p(x)$ means marginalizing — the intractable integral the ELBO worked
   around.
4. The bottom-left block $\partial y_B / \partial x_A$ — dense and full of
   network derivatives. A block-triangular determinant is the product of the
   diagonal blocks' determinants, so the off-diagonal block never enters.
5. Because the inverse can recompute $s(y_A)$ and $t(y_A)$ exactly as the
   forward pass did — $y_A = x_A$ is available unchanged — and then solves the
   elementwise affine map explicitly. $s$ and $t$ only ever run forward.
6. The layer computes $y_A = x_A$ with no parameters involved, so the map on
   the $A$-coordinates is the identity and their distribution is untouched;
   training cannot change what no parameter influences. Fix: alternate (or
   permute) the mask across layers so every dimension is transformed.
7. The $e^{s}$ scale: unbounded scale-network outputs overflow the exponential
   (or explode the inverse). Mitigation: bound $s$ with a tanh clamp,
   $s = c\,\tanh(\tilde s)$, and watch the $s$ statistics like Week-16
   activation histograms.
8. Physics-observable histograms with quantified distances; a classifier
   two-sample test against Geant4 (AUC → 0.5 is success); sampling cost.
   Eyeballing is absent because the failure modes that matter — tails,
   correlations, conditional response — are invisible in a look-at-it grid and
   are exactly where analyses cut.

## New terms

- **normalizing flow** — an invertible network trained by exact maximum
  likelihood via the change of variables; "normalizes" data to a base Gaussian.
- **change-of-variables formula** — $p_X(x) = p_Z(f(x))\,|\det J_f(x)|$;
  densities transform with a volume factor, conserving probability mass.
- **Jacobian / log-det Jacobian** — the matrix of first partials; the log of its
  determinant's magnitude, the local log-volume change summed over layers.
- **base distribution** — the simple density (standard normal) the flow maps
  data to and samples from.
- **normalizing vs generative direction** — data→latent (likelihood) vs
  latent→data (sampling); inverses of each other.
- **coupling layer (RealNVP)** — pass half the dimensions through, transform
  the other half elementwise with scale/translate networks fed by the first
  half; block-triangular Jacobian.
- **mask / alternating masks** — the fixed split into pass-through and
  transformed dimensions; swapping it layer to layer so all dimensions move.
- **tanh clamp** — bounding the scale output $s = c\,\tanh(\tilde s)$ to keep
  $e^s$ sane.
- **dequantization** — adding uniform noise to discrete data so a continuous
  density model applies.
- **Breit–Wigner** — the Lorentzian mass line shape of an unstable particle;
  heavy-tailed, hence a good flow test target.
- **Geant4 / full simulation** — step-by-step microphysics simulation of
  particles through a detector; the accuracy standard and the CPU cost problem.
- **fast simulation / learned simulator** — replacing full simulation with a
  cheap surrogate; here, a trained generative model conditioned on particle
  energy.
- **voxel** — a 3D pixel; here, one cell of deposited calorimeter energy.
- **CaloChallenge** — the community benchmark for learned calorimeter
  simulation; judged on observable histograms, a classifier two-sample test,
  and cost.
- **classifier two-sample test** — train a classifier to separate generated
  from real; AUC above 0.5 quantifies detectable generator flaws.

## Going deeper

- Lilian Weng, "Flow-based Deep Generative Models" (blog) — the spine reading;
  covers this lesson's math plus the autoregressive-flow family (MAF/IAF) that
  §7 only waved at.
- Prince, *Understanding Deep Learning*, Ch. 16 (free PDF) — the same
  construction in a second notation, plus residual and continuous flows.
- Dinh, Sohl-Dickstein & Bengio, "Density estimation using Real NVP"
  (arXiv:1605.08803) — the coupling-layer source; read §3 closely, skim the
  multiscale image machinery.
- Krause & Shih, "CaloFlow" (arXiv:2106.05285) — flows meeting calorimeters;
  read for the framing and the evaluation, skim the architecture details.
- The Fast Calorimeter Simulation Challenge 2022 description (search
  "CaloChallenge") — the datasets and evaluation protocol §9 summarized; the
  results-comparison paper is worth a skim to see the whole zoo ranked.
- Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and
  Inference" (arXiv:1912.02762) — the field's survey, for when you want the
  full map; reference, not required.
