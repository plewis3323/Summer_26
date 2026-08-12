# Week 22 — Autoencoders & VAEs

~3–4 hrs reading, plus the paper-and-pencil pass. Before starting you should be able
to: state Bayes' theorem and the definitions of likelihood, prior, and posterior
(Week 08); write the definition of KL divergence and its two key properties
(Week 08); derive least squares from a Gaussian likelihood (Week 08); and train an
MLP or CNN in PyTorch with a custom loss (Weeks 15–17).

This week contains the course's second flagship derivation (after backprop): the
**ELBO**, derived twice, every line justified. It is part of the Phase 2 gate — you
will re-derive it cold in Week 24 — so do the paper pass properly now.

## 1. From compressing data to generating it

Everything so far in Phase 2 has been **discriminative**: given an input $x$,
predict a label $y$. A **generative model** answers a different question: what is
the distribution $p(x)$ of the data itself — and can we draw new samples from it?

Why would a physicist want that?

- **Fast simulation** (Week 23's topic in full): if a model has learned the
  distribution of calorimeter showers, sampling it can replace expensive
  simulation.
- **Anomaly detection**: a model that knows what ordinary events look like can
  flag events it finds improbable — a way to search for new physics without
  specifying what you are looking for.

We start with a humbler goal, compression, because the architecture it produces is
half of a VAE.

## 2. Autoencoders

An **autoencoder** is two networks in series. The **encoder** $e$ maps an input
$x \in \mathbb{R}^D$ (say a $D = 64$-pixel shower image) down to a low-dimensional
code $z \in \mathbb{R}^d$ with $d \ll D$ (say $d = 2$). The **decoder** $d$ maps
$z$ back up to a reconstruction $\hat{x} = d(e(x))$. Train both to minimize
**reconstruction error**, usually mean squared error:

$$\mathcal{L}(x) = \|x - d(e(x))\|^2.$$

The narrow middle — the **bottleneck** — is the point. To reconstruct a 64-pixel
shower through 2 numbers, the network must discover a 2-number summary that
captures most of what varies between showers: something like total energy and
shower width, learned rather than hand-defined. The code $z$ is called the
**latent vector**, living in **latent space** ("latent" = hidden: not observed,
inferred).

You have seen the linear version of this: PCA (Week 07) is exactly the optimal
*linear* autoencoder under squared error — encoder $V_d^\top x$, decoder $V_d z$.
An autoencoder with nonlinear networks can curve its projection to follow the data,
which is what buys reconstruction accuracy at the same $d$.

### 2.1 Why an autoencoder is not a generative model

Tempting shortcut: train an autoencoder, then feed *random* $z$ values to the
decoder and call the outputs new showers. It fails, and the failure is instructive.

The encoder places training data *somewhere* in latent space, but nothing controls
where. The occupied region can be a thin curved filament with holes; its location
and scale differ per training run. A random $z$ almost surely lands off that
filament, and the decoder — which was only ever trained on the filament — outputs
garbage there. There is no distribution over $z$ that we know how to sample from
such that decoded samples follow the data distribution.

Diagnosis: we trained a compressor and asked it to be a distribution. To generate,
the model must be *built* around a distribution from day one. That means writing
down $p(x)$ with a latent variable in it — and immediately hitting the wall that
motivates everything else this week.

## 3. Latent-variable models and the intractable integral

Here is the generative story we want, told forward:

1. Draw a latent vector from a fixed, simple **prior**: $z \sim p(z)$. We choose
   the standard normal, $p(z) = \mathcal{N}(z;\, 0, I)$ — a $d$-dimensional
   Gaussian with mean zero and identity covariance. ("Simple" is the point: we can
   sample it trivially, forever.)
2. Decode it through a network into a distribution over data:
   $x \sim p_\theta(x \mid z)$. Here $\theta$ (theta) is the decoder's weights.
   For images a common choice is a Gaussian centered on the decoder output:
   $p_\theta(x \mid z) = \mathcal{N}(x;\, d_\theta(z), \sigma^2 I)$.

The model's total probability of a data point marginalizes over the latent:

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz.$$

Training should be maximum likelihood (Week 08): choose $\theta$ to maximize
$\sum_i \log p_\theta(x_i)$ over the training set. And here is the wall: **that
integral is intractable.** It runs over all of $\mathbb{R}^d$, the integrand
involves a neural network, and no closed form exists. Numerically estimating it by
sampling $z \sim p(z)$ is hopeless in practice: for any particular $x$, almost all
prior samples decode to something far from $x$, so $p_\theta(x \mid z) \approx 0$
for nearly every draw and the estimate is all noise.

The same wall blocks the **posterior** $p_\theta(z \mid x)$ — "which latents
plausibly produced this $x$?" — since by Bayes' theorem it needs the same $p_\theta(x)$
in its denominator.

The variational move: if you cannot compute the quantity, compute a *bound* on it,
and make the bound as tight as you can. Physics does this constantly — a variational
wavefunction ansatz bounds a ground-state energy you cannot solve for exactly. Here
the ansatz is a distribution: introduce $q_\phi(z \mid x)$, a tractable family
(diagonal Gaussians whose mean and variance are computed from $x$ by an encoder
network with weights $\phi$ (phi)), intended to approximate the true posterior.
$q_\phi$ is called the **variational** or **approximate posterior**. Now derive the
bound.

## 4. The ELBO, route 1: Jensen's inequality

First the tool. A function $f$ is **concave** if every chord lies below the curve —
$\log$ is the canonical example ($\log''(u) = -1/u^2 < 0$). **Jensen's
inequality** says: for concave $f$ and any random variable $Y$,

$$f(\mathbb{E}[Y]) \;\ge\; \mathbb{E}[f(Y)].$$

Intuition with two points: $\log\left(\tfrac{a+b}{2}\right) \ge \tfrac{\log a + \log b}{2}$
— the log of an average beats the average of logs, because the chord from
$(a, \log a)$ to $(b, \log b)$ sags below the curve. Jensen extends this from
two-point averages to any expectation. Equality holds only when $Y$ is constant
(the "average" is over a single value).

Now the derivation. Fix one data point $x$ and start from the thing we cannot
compute:

$$\log p_\theta(x) = \log \int p_\theta(x, z)\, dz$$

*(Line 1: definition — the joint $p_\theta(x,z) = p_\theta(x \mid z)p(z)$
marginalized over $z$.)*

$$= \log \int q_\phi(z \mid x)\, \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\, dz$$

*(Line 2: multiply and divide by $q_\phi(z \mid x)$ — legal wherever
$q_\phi > 0$, and we choose Gaussian $q$, which is positive everywhere. This is
the one creative step: we are inserting our trial distribution into an expression
that didn't ask for it, so that the integral becomes an average under $q$.)*

$$= \log\, \mathbb{E}_{z \sim q_\phi(z \mid x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]$$

*(Line 3: rewrite the integral as an expectation — $\int q(z) g(z)\,dz = \mathbb{E}_{q}[g(z)]$
by definition of expectation.)*

$$\ge\; \mathbb{E}_{z \sim q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \;\equiv\; \mathcal{L}(\theta, \phi; x)$$

*(Line 4: Jensen, with $f = \log$ and $Y = p_\theta(x,z)/q_\phi(z \mid x)$. The
inequality direction comes from concavity.)*

That final quantity $\mathcal{L}$ is the **evidence lower bound** — **ELBO** —
so named because $p_\theta(x)$ is called the *evidence* (Week 08: the denominator
in Bayes' theorem) and $\mathcal{L}$ sits below its log. It is an expectation
under $q_\phi$ of quantities we can evaluate point by point: the joint is
decoder-times-prior, and $q_\phi$ is our own Gaussian. Tractable — estimate the
expectation by sampling $z$ from $q_\phi$.

Route 1 is fast, but it leaves a question hanging: *how loose is the bound?*
Jensen says "$\ge$" and nothing more. Route 2 answers it exactly.

## 5. The ELBO, route 2: the KL decomposition

Recall from Week 08: the KL divergence between distributions $q$ and $p$ is

$$\mathrm{KL}(q \,\|\, p) = \mathbb{E}_{z \sim q}\!\left[\log \frac{q(z)}{p(z)}\right],$$

it satisfies $\mathrm{KL} \ge 0$, with equality iff $q = p$, and it is asymmetric.

Start again from $\log p_\theta(x)$, and this time keep everything exact:

$$\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z \mid x)}\big[\log p_\theta(x)\big]$$

*(Line 1: $\log p_\theta(x)$ does not depend on $z$, so averaging it over any
distribution over $z$ changes nothing. We choose to average over $q_\phi$ — same
move as Line 2 before: bring the trial distribution into the game.)*

$$= \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x, z)}{p_\theta(z \mid x)}\right]$$

*(Line 2: the product rule of probability says $p_\theta(x,z) = p_\theta(x)\,
p_\theta(z \mid x)$; solve it for the evidence,
$p_\theta(x) = p_\theta(x,z)/p_\theta(z \mid x)$, and substitute inside the log.
The true — intractable — posterior has appeared; that is fine, because it will end
up inside a term we never have to compute.)*

$$= \mathbb{E}_{q_\phi}\!\left[\log \left(\frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)}\right)\right]$$

*(Line 3: multiply and divide by $q_\phi(z \mid x)$ inside the log — inserting the
trial distribution again, this time to split the ratio into a piece we can compute
and a piece we can interpret.)*

$$= \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]}_{\mathcal{L}(\theta, \phi;\, x)} \;+\; \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)}\right]}_{\mathrm{KL}\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)}$$

*(Line 4: log of a product = sum of logs; expectation is linear. The first term is
the ELBO from route 1; the second is, by definition, a KL divergence.)*

So, exactly, with no inequality anywhere:

$$\boxed{\;\log p_\theta(x) \;=\; \mathcal{L}(\theta, \phi;\, x) \;+\; \mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\big)\;}$$

Read off three facts:

1. **The bound, again.** Since $\mathrm{KL} \ge 0$, we get
   $\log p_\theta(x) \ge \mathcal{L}$ — route 1's conclusion, recovered without
   Jensen.
2. **The gap has a name.** The bound's slack is precisely how far the trial
   posterior $q_\phi(z \mid x)$ is from the true posterior $p_\theta(z \mid x)$.
   The bound is tight iff $q_\phi$ nails the posterior exactly.
3. **Maximizing the ELBO over $\phi$ does inference.** For fixed $\theta$, the
   left side is constant in $\phi$; so pushing $\mathcal{L}$ up must push the KL
   gap down by the same amount. Training the encoder *is* approximate posterior
   inference — the same role the E-step plays in EM for GMMs (Week 12), except
   done once by a shared network for all data points. That sharing is called
   **amortized inference**: one network, weights $\phi$, produces the approximate
   posterior for any $x$ in a single forward pass.

On paper, do both routes yourself now, then check: route 1's Jensen slack and
route 2's KL gap must be the same quantity. (They are — Check-yourself Q3.)

## 6. Unpacking the ELBO into a loss you can type

Split the joint inside the ELBO: $p_\theta(x, z) = p_\theta(x \mid z)\, p(z)$, so

$$\mathcal{L} = \mathbb{E}_{q_\phi}\big[\log p_\theta(x \mid z)\big] + \mathbb{E}_{q_\phi}\!\left[\log \frac{p(z)}{q_\phi(z \mid x)}\right] = \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{regularizer}}$$

The two terms are the whole story of a VAE:

- **Reconstruction**: encode $x$, sample a plausible $z$, and the decoder should
  put high probability on the original $x$.
- **KL to the prior**: the encoder's output distributions must stay close to
  $\mathcal{N}(0, I)$. This is what the plain autoencoder lacked — it forces the
  code cloud to fill a known region with a known shape, so that at generation time,
  prior samples land where the decoder has been trained.

### 6.1 The KL term in closed form

Encoder outputs a diagonal Gaussian: $q_\phi(z \mid x) = \mathcal{N}(z;\, \mu, \mathrm{diag}(\sigma^2))$,
where $\mu, \sigma \in \mathbb{R}^d$ are network outputs. Against the prior
$\mathcal{N}(0, I)$, the KL has a closed form. Derive the 1D case; diagonal
dimensions just add (a product of independent Gaussians turns the log-ratio into a
sum).

With $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$:

$$\log \frac{q(z)}{p(z)} = \log \frac{\frac{1}{\sqrt{2\pi}\,\sigma} e^{-(z-\mu)^2/2\sigma^2}}{\frac{1}{\sqrt{2\pi}} e^{-z^2/2}} = -\log \sigma - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}.$$

Take $\mathbb{E}_{z \sim q}$ term by term, using two Gaussian facts:
$\mathbb{E}_q[(z-\mu)^2] = \sigma^2$ (definition of variance) and
$\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$ (variance + mean squared):

$$\mathrm{KL} = -\log \sigma - \frac{\sigma^2}{2\sigma^2} + \frac{\mu^2 + \sigma^2}{2} = \tfrac{1}{2}\left(\mu^2 + \sigma^2 - 1 - \log \sigma^2\right).$$

Summing over the $d$ dimensions:

$$\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,\mathcal{N}(0,I)\big) = \tfrac{1}{2}\sum_{k=1}^{d}\left(\mu_k^2 + \sigma_k^2 - 1 - \log \sigma_k^2\right).$$

Sanity checks: it is 0 exactly at $\mu = 0, \sigma = 1$ (match the prior), grows as
$\mu$ drifts, and grows both when $\sigma \to 0$ (overconfident) and
$\sigma \to \infty$ (too broad). This formula goes straight into the loss function.
Exercise E2 checks it against a Monte-Carlo estimate.

### 6.2 The reconstruction term is (β-weighted) MSE

Choose a Gaussian decoder with fixed variance:
$p_\theta(x \mid z) = \mathcal{N}(x;\, \hat{x}(z), \sigma_x^2 I)$ where
$\hat{x}(z)$ is the decoder network's output. Then

$$\log p_\theta(x \mid z) = -\frac{\|x - \hat{x}(z)\|^2}{2\sigma_x^2} - \frac{D}{2}\log(2\pi\sigma_x^2),$$

the Week-08 Gaussian-MLE argument verbatim: maximizing a fixed-variance Gaussian
log-likelihood is minimizing squared error. The constant term has no $\theta$ or
$\phi$ in it, so training ignores it. Notice $\sigma_x^2$ sets the *relative
weight* between reconstruction and the KL term: maximizing the ELBO is minimizing

$$\|x - \hat{x}\|^2 \;+\; 2\sigma_x^2 \cdot \mathrm{KL}.$$

People usually write this weight as $\beta$ and call the model a **β-VAE**: β > 1
squeezes codes toward the prior (more regular latent space, blurrier
reconstructions); β < 1 favors reconstruction. β is not a hack bolted onto the
math — it *is* the assumed decoder variance, surfaced as a knob.

## 7. The reparameterization trick

One step of the recipe is still broken. The reconstruction term is an expectation
over $z \sim q_\phi(z \mid x)$, estimated by sampling. Gradients w.r.t. $\theta$
(decoder) flow fine — $\theta$ sits inside the expectation. But we also need
$\nabla_\phi$, and $\phi$ controls *the distribution being sampled from*. What is
the derivative of "a random draw" with respect to the parameters of the
distribution it was drawn from? As stated — undefined. A sample is a number, not a
differentiable function of $\mu$ and $\sigma$; autograd (Week 14) sees the sampling
node as a constant and reports zero gradient to the encoder. If you implement a VAE
by calling a sampler and detaching, the decoder trains and the encoder never learns
anything (Exercise E4 makes you build this broken version on purpose).

The fix is a change of viewpoint. A draw from $\mathcal{N}(\mu, \sigma^2)$ can be
manufactured from a draw from $\mathcal{N}(0, 1)$:

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I),$$

with $\odot$ meaning elementwise product. Same distribution — a Gaussian shifted by
$\mu$ and scaled by $\sigma$ — but now *the randomness is an input, not the
operation*. $z$ is an ordinary differentiable function of $\mu$, $\sigma$, and a
parameter-free noise variable. Formally, for any well-behaved $f$:

$$\nabla_\phi\, \mathbb{E}_{z \sim q_\phi}\big[f(z)\big] = \nabla_\phi\, \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}\big[f(\mu_\phi + \sigma_\phi \odot \varepsilon)\big] = \mathbb{E}_{\varepsilon}\big[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \varepsilon)\big].$$

Step one rewrites the expectation over a distribution that *depends on* $\phi$ as an
expectation over one that does not (that is what the substitution buys). Step two
swaps gradient and expectation — legal now precisely because the distribution being
averaged over no longer involves $\phi$. Autograd then differentiates
$f(\mu + \sigma \odot \varepsilon)$ by the plain chain rule (Week 13): gradients
flow through $z$ into $\mu$ and $\sigma$ and back into the encoder.

In code it is one line: `z = mu + sigma * torch.randn_like(sigma)`.

## 8. Posterior collapse

The KL term is minimized by ignoring $x$: if the encoder outputs $\mu = 0,
\sigma = 1$ for every input, the KL is exactly zero. If, early in training, the
decoder finds a way to produce passable average-looking reconstructions *without
using $z$*, the optimizer happily takes the free KL win, and you reach a stable
failure: KL ≈ 0, encoder uninformative, decoder deaf to its input. This is
**posterior collapse**. Symptoms on your logs (log both terms separately — always):
the KL term slides to ~0 while reconstruction plateaus; prior samples all look like
the same average blob; latent traversals change nothing.

Mitigations, in the order to try them:

- **KL warm-up (annealing)**: multiply the KL term by a weight ramped 0 → 1 over
  the first epochs, so the model learns to use the latent before being charged for
  it.
- **Lower β** if you raised it.
- **Weaker decoder**: a decoder with too much capacity is what makes "ignore $z$"
  viable in the first place.

Exercise E5 has you *induce* collapse deliberately with a β sweep, so you recognize
the failure before Capstone track (b), where it is the classic first-run disease.

## 9. VAEs in physics: anomaly detection

The fast-sim application waits for Weeks 23–24. The other workhorse use needs one
section. Train a (V)AE on ordinary events only — say, single-photon showers. The
model gets good at reconstructing what it has seen. Feed it something from a
different distribution — a merged-π⁰ shower, or in the LHC-search setting, a decay
of some particle nobody predicted — and reconstruction is poor. So use
**reconstruction error as an anomaly score**, and cut on it.

Why this matters: a supervised classifier (Week 20) needs labeled examples of the
signal, i.e. you must know what you are looking for. An anomaly detector only needs
to know the background. That is a real search strategy (the "LHC Olympics"
community challenge is built on it).

Its failure modes, so you use it honestly: anomalies that are *easier* to
reconstruct than the training data (e.g. lower-multiplicity, smoother) score as
normal; the score conflates "unusual physics" with "unusual noise level"; and it
will never match a supervised classifier where labels exist (E6 quantifies that gap
against your Week-20 CNN).

## 10. Worked example: a VAE in ~60 lines

Complete and runnable as shown. To stay self-contained it builds a toy dataset of
8×8 "shower" images (a 2D Gaussian blob with random center, width, and energy) —
swap in your Week-20 generator's images for the exercises. Install with
`uv add torch`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# --- toy data: 5000 blob images, 8x8, flattened to 64 ---
n, side = 5000, 8
yy, xx = torch.meshgrid(torch.arange(side * 1.0), torch.arange(side * 1.0),
                        indexing="ij")
cx = 2.5 + 3.0 * torch.rand(n, 1, 1)
cy = 2.5 + 3.0 * torch.rand(n, 1, 1)
w = 0.8 + 0.8 * torch.rand(n, 1, 1)
amp = 0.5 + torch.rand(n, 1, 1)
imgs = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * w ** 2))
data = imgs.reshape(n, side * side)

# --- model ---
class VAE(nn.Module):
    def __init__(self, d_in, d_hid, d_lat):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU())
        self.to_mu = nn.Linear(d_hid, d_lat)
        self.to_logvar = nn.Linear(d_hid, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)          # predict log sigma^2: keeps sigma > 0
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * torch.randn_like(sigma)   # reparameterization
        return self.dec(z), mu, logvar

model = VAE(64, 128, 2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
beta = 1.0

for epoch in range(30):
    perm = torch.randperm(n)
    for i in range(0, n, 256):
        x = data[perm[i:i + 256]]
        xhat, mu, logvar = model(x)
        rec = ((x - xhat) ** 2).sum(dim=1).mean()
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(dim=1).mean()
        loss = rec + beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 10 == 9:
        print(f"epoch {epoch + 1}  rec {rec.item():.3f}  kl {kl.item():.3f}")

# --- generate: sample the PRIOR, decode ---
with torch.no_grad():
    z = torch.randn(4, 2)
    samples = model.dec(z).reshape(4, side, side)
print("sample energy sums:", samples.sum(dim=(1, 2)))
```

Map every line back to the math: `rec` is $-\log p_\theta(x \mid z)$ up to the
$2\sigma_x^2$ scale (§6.2), `kl` is the §6.1 closed form written in
`logvar` $= \log\sigma^2$, the sampling line is §7, and generation draws from the
*prior* — the thing §2.1 said the plain autoencoder could not offer. The two loss
terms are logged separately because §8 said collapse hides inside the total.

## Check yourself

1. Why can't $\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z)\, dz$ be
   maximized directly, and why does naive Monte-Carlo over the prior fail?
2. In route 1, which single line introduces the approximation, and what theorem
   justifies the inequality's direction?
3. Route 2 shows the bound's gap is $\mathrm{KL}(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x))$.
   Route 1's Jensen slack must therefore equal it. When is the slack zero, in both
   languages?
4. The ELBO's KL term compares $q_\phi(z \mid x)$ to the *prior*. The gap in route
   2 compares it to the *posterior*. Why does the trainable loss contain the first
   and not the second?
5. Why is the gradient of a raw sample w.r.t. the encoder parameters meaningless,
   and what exactly does $z = \mu + \sigma \odot \varepsilon$ change?
6. Derive in two lines: why does a Gaussian decoder with fixed variance make the
   reconstruction term an MSE, and where does β live in that argument?
7. Your VAE's KL term reads 0.001 after epoch 2 and the samples all look
   identical. Name the failure and two mitigations.
8. Give one reason reconstruction-error anomaly detection can *miss* genuinely
   anomalous events.

## Answers

1. The integral over $z$ has no closed form with a network inside, and prior
   samples almost never explain any particular $x$ — $p_\theta(x \mid z) \approx 0$
   for nearly all draws — so the MC estimate has enormous variance.
2. Line 4 (Jensen's inequality): $\log$ of an expectation ≥ expectation of $\log$
   because log is concave.
3. Zero iff $q_\phi(z \mid x) = p_\theta(z \mid x)$. In Jensen language: equality
   iff the ratio $p_\theta(x,z)/q_\phi(z \mid x)$ is constant in $z$ — which,
   normalizing, is the same statement.
4. Splitting the joint turns the ELBO into reconstruction minus
   $\mathrm{KL}(q\,\|\,\text{prior})$ — both computable. The posterior-KL is the
   *gap*, never evaluated; we shrink it indirectly because ELBO + gap is fixed at
   $\log p_\theta(x)$ for fixed $\theta$.
5. A sample is a constant with respect to autograd — there is no differentiable
   path from $\phi$ to the drawn number. Reparameterization makes $z$ a
   deterministic function of $(\mu_\phi, \sigma_\phi)$ with the randomness moved
   into a parameter-free input $\varepsilon$, so the chain rule applies.
6. $\log \mathcal{N}(x; \hat{x}, \sigma_x^2 I) = -\|x - \hat{x}\|^2 / 2\sigma_x^2 + \text{const}$;
   maximizing it is minimizing MSE, and the $\sigma_x^2$ prefactor reweights
   reconstruction against KL — that ratio is β.
7. Posterior collapse. KL warm-up (anneal the KL weight from 0), reduce β, weaken
   the decoder.
8. If the anomaly is easier to reconstruct than training data (smoother, simpler),
   its error is low and it scores as normal; the score also can't separate "new
   physics" from mundane distribution shift like a noise change.

## New terms

- **generative model / discriminative model** — models $p(x)$ (can sample) vs
  $p(y \mid x)$ (predicts labels).
- **autoencoder / encoder / decoder / bottleneck / reconstruction error** — the
  compression architecture and its loss.
- **latent variable / latent space** — the unobserved code $z$ and the space it
  lives in.
- **prior $p(z)$** — the fixed simple distribution latents are drawn from.
- **posterior $p(z \mid x)$ / approximate (variational) posterior $q_\phi(z \mid x)$**
  — which latents explain $x$; the tractable stand-in the encoder computes.
- **evidence** — $p_\theta(x)$, the model's marginal likelihood of the data.
- **ELBO** — evidence lower bound; the tractable objective
  $\mathbb{E}_q[\log p(x,z)/q(z \mid x)]$.
- **Jensen's inequality / concave** — $f(\mathbb{E}[Y]) \ge \mathbb{E}[f(Y)]$ for
  concave $f$; chords below the curve.
- **amortized inference** — one encoder network producing approximate posteriors
  for every $x$, instead of a per-data-point optimization.
- **reparameterization trick** — $z = \mu + \sigma \odot \varepsilon$; moves
  randomness to an input so gradients pass through sampling.
- **β-VAE** — ELBO with a weighted KL term; β = the implicit decoder variance.
- **KL warm-up (annealing)** — ramping the KL weight from 0 early in training.
- **posterior collapse** — KL → 0, encoder ignored; the VAE degenerates to an
  unconditional decoder.
- **anomaly detection (reconstruction-based)** — scoring events by how badly a
  model trained on background reconstructs them.

## Going deeper

- Lilian Weng, "From Autoencoder to Beta-VAE" (blog) — the whole landscape in one
  post; read after your own derivation to place the variants.
- Prince, *Understanding Deep Learning*, Ch. 17 (free PDF) — the same derivation
  with different notation; a good check that you can translate.
- Kingma & Welling, "Auto-Encoding Variational Bayes" (arXiv:1312.6114), §1–3 —
  the original paper; after this lesson it should read as familiar.
- One LHC Olympics / HEP autoencoder anomaly-detection paper of your choice — read
  for how §9's idea is used and evaluated on real physics.
