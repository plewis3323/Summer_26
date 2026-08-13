# Week 35 — Diffusion Models

~3–4 hrs reading, plus the paper-and-pencil pass. Before starting you should be
able to: manipulate Gaussian densities and compute $\mathrm{Var}(aX + bY)$ for
independent $X, Y$ (Weeks 07–08); state the definition and key properties of KL
divergence and derive a Gaussian–Gaussian KL (Weeks 08, 22); derive the VAE ELBO
by both routes (Week 22); know that a normalizing flow is an invertible map with
exact likelihood (`optional-flows.md` if you did it; otherwise the survey in §10
is enough); and train a conditional model in PyTorch (Weeks 15–16, 22).

This week contains the course's fifth flagship derivation (after backprop, the
ELBO, attention, and LoRA's low-rank argument): the **DDPM training objective**,
built line by line from the forward noising process down to the simplified loss
you actually type. It enters the monthly cold-redo rotation (syllabus §7), and it
is the mathematical core of Capstone 3 track (b) — do the paper pass properly.

This is a GPU week (syllabus §8): the toy exercises run on CPU, but the
calorimeter warm-up (E5) wants a cloud GPU — Colab or Kaggle per the setup guide.

## 1. Destroying data is easy; learn to run it backwards

Week 22 ended with a diagnosis: the VAE got tractability
from a *bound* — you never optimize the true likelihood, and sample quality shows
it (blurry showers, posterior collapse). A flow (optional Week 23 reading) got *exact* likelihoods, but
paid with a straitjacket: every layer invertible, every Jacobian determinant
tractable, which rules out most architectures you would like to use.

Diffusion models take a third route, and its starting move looks like a joke:
instead of learning to generate data, learn to *destroy* it — because for the
right kind of destruction, the reverse is learnable.

The **forward (noising) process** takes a data point $x_0$ — a calorimeter shower
image, say — and adds a small amount of Gaussian noise, over and over, $T$ times
(typically $T = 1000$). By the end, $x_T$ is indistinguishable from pure noise:
every trace of the shower is gone, and we can *prove* it. This direction needs no
learning at all; it is a fixed recipe, chosen by us.

The generative model is the reverse: start from pure noise $x_T \sim
\mathcal{N}(0, I)$ — which we can sample forever, like the VAE's prior — and
learn to undo the noising one small step at a time, $x_T \to x_{T-1} \to \cdots
\to x_0$. Each *individual* step only has to remove a sliver of noise, and a
sliver-removal is a far easier function to learn than "map $z$ to a shower" in
one jump. That is the whole bet: trade one hard denoising problem for a thousand
easy ones, and pay for it at sampling time — a thousand network passes per
sample where the VAE and flow need one. Hold that cost in mind; it returns in
§10 as the central tension for calorimeter simulation.

The rest of the lesson makes each phrase precise: "the right kind of
destruction" (§3), "we can prove it" (§3), "learn to undo" (§§4–6), and "what it
costs" (§§7, 10).

## 2. The Gaussian toolkit

The entire derivation runs on three facts about Gaussians, all within reach of
Weeks 07–08. Verify each on paper before continuing — every later step cites one
of them.

**Fact 1 — linear transforms.** If $\varepsilon \sim \mathcal{N}(0, I)$ in
$\mathbb{R}^D$, then $\mu + \sigma\varepsilon \sim \mathcal{N}(\mu, \sigma^2 I)$
for any vector $\mu$ and scalar $\sigma > 0$. This is Week 22's
reparameterization statement read in the other direction: a Gaussian is standard
noise, shifted and scaled.

**Fact 2 — sums of independent Gaussians.** If $X \sim \mathcal{N}(\mu_1,
\sigma_1^2 I)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2 I)$ are independent,
then $aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2,\; (a^2\sigma_1^2 +
b^2\sigma_2^2) I)$. Means add linearly; for independent variables *variances*
add (with the coefficients squared, since $\mathrm{Var}(aX) = a^2
\mathrm{Var}(X)$) — the same fact that gave you error propagation in the lab and
$\mathrm{Var}(aX + bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y)$ in Week 08.
That the sum is again *Gaussian* is special to Gaussians, and it is the property
the whole forward process is built on.

**Fact 3 — products of Gaussian densities.** As *functions of* $x$, the product
of two Gaussian densities in $x$ is proportional to another Gaussian density in
$x$: the exponents add, and a sum of two quadratics in $x$ is a quadratic in
$x$ — complete the square (high-school algebra) and read off the new mean and
variance. You used exactly this move in Week 08 to combine a Gaussian likelihood
with a Gaussian prior into a Gaussian posterior. §4 is one long application of
Fact 3.

One notation warning before the symbols start: diffusion's noise schedule is
called $\beta_t$ by universal convention. It has nothing to do with the β of
β-VAE from Week 22 — an unfortunate collision, not a connection.

## 3. The forward process, and its closed form

Fix a **noise schedule**: a sequence of small numbers $0 < \beta_1 < \beta_2 <
\cdots < \beta_T < 1$ (Ho et al.'s default: linear from $10^{-4}$ to $0.02$ over
$T = 1000$ steps). One forward step is defined as

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(x_t;\; \sqrt{1 - \beta_t}\,x_{t-1},\;
\beta_t I\big),$$

equivalently, by Fact 1,

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0, I) \text{ independent per step.}$$

Each step shrinks the signal a little and adds a little fresh noise. Why the
peculiar $\sqrt{1 - \beta_t}$ shrink factor, rather than just adding noise?
Compute the variance after one step, assuming $x_{t-1}$ has zero mean and
identity covariance, using Fact 2:

$$\mathrm{Var}(x_t) = (1 - \beta_t)\cdot I + \beta_t \cdot I = I.$$

The shrink is tuned so that variance is *preserved*: signal variance is traded
for noise variance one-for-one, and the process converges to
$\mathcal{N}(0, I)$ — the fixed, known distribution we will sample from at
generation time — instead of blowing up. (A process that only added noise would
have variance growing without bound; "start from noise" would then mean "start
from a distribution that depends on $T$ and the data scale", and the clean
endpoint would be lost.)

**The closed form.** Simulating $t$ steps to get one training example would be
wasteful. Define

$$\alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s,$$

($\bar\alpha_t$ falls from near 1 toward 0 as $t$ grows — it is "how much of
$x_0$ survives"). Claim:

$$q(x_t \mid x_0) = \mathcal{N}\!\big(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\;
(1 - \bar\alpha_t) I\big),
\qquad\text{i.e.}\qquad
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\;\varepsilon,\;\;
\varepsilon \sim \mathcal{N}(0, I).$$

*Derivation, by composing two steps and then induction.* Substitute the step for
$x_{t-1}$ into the step for $x_t$:

$$x_t = \sqrt{\alpha_t}\Big(\sqrt{\alpha_{t-1}}\,x_{t-2} +
\sqrt{1 - \alpha_{t-1}}\,\varepsilon_{t-1}\Big) + \sqrt{1-\alpha_t}\,\varepsilon_t
= \sqrt{\alpha_t \alpha_{t-1}}\; x_{t-2}
+ \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\varepsilon_{t-1}
+ \sqrt{1-\alpha_t}\,\varepsilon_t}_{\text{two independent Gaussians}}.$$

*(Step 1: plain substitution and distributing $\sqrt{\alpha_t}$.)*

The braced sum is a sum of independent zero-mean Gaussians, so by Fact 2 it is
Gaussian with variance

$$\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t)
= \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t
= 1 - \alpha_t\alpha_{t-1}.$$

*(Step 2: Fact 2 — coefficients squared, variances added — then cancel
$\alpha_t$. Note the pattern: two steps compose into one step of exactly the
same form, with $\alpha_t\alpha_{t-1}$ playing the role of a single survival
factor.)*

So $x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} +
\sqrt{1-\alpha_t\alpha_{t-1}}\,\varepsilon$ with fresh
$\varepsilon \sim \mathcal{N}(0,I)$. Repeating the same move down to $x_0$
multiplies the survival factors into $\bar\alpha_t$ — that is the induction, and
the claim follows. *(Step 3: each composition preserves the form; after $t$
compositions the coefficient on $x_0$ is $\sqrt{\prod_s \alpha_s}$.)*

Two consequences to hold on to:

1. **One-shot noising.** A training pair at any $t$ costs one Gaussian draw:
   pick $x_0$ from data, pick $t$, draw $\varepsilon$, form
   $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$. No
   simulation of the chain. Exercise E1 verifies this against the step-by-step
   walk numerically.
2. **The proof that destruction succeeds.** As $t \to T$,
   $\bar\alpha_t \to 0$ (a product of many numbers below 1), so
   $q(x_T \mid x_0) \to \mathcal{N}(0, I)$ *for every* $x_0$: the endpoint
   distribution carries no information about the input. That is "$x_T$ is pure
   noise", proved.

## 4. The reverse process, and the posterior you can actually compute

Generation must run the chain backwards: given a noisy $x_t$, produce a slightly
less noisy $x_{t-1}$. The object we would love to know is the **true reversal**
$q(x_{t-1} \mid x_t)$. By Bayes' theorem it is

$$q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1})}{q(x_t)},$$

and there is the wall, in a familiar place: $q(x_{t-1})$ is the noised *data
distribution* — the very thing we do not have a formula for. (Same wall as Week
22 §3: the model of interest is tractable in one direction only.) So, as in Week
22, we will introduce a learned stand-in — a network — for the intractable
direction:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\big(x_{t-1};\;
\mu_\theta(x_t, t),\; \sigma_t^2 I\big),$$

a Gaussian whose mean is computed by a network $\mu_\theta$ (θ now names the
denoiser's weights) and whose variance $\sigma_t^2$ we fix by hand (Ho et al.
tried learning it and found fixing it to $\beta_t$, or to the $\tilde\beta_t$
derived below, works as well — take the simplification). A Gaussian form is not
an arbitrary convenience: for small $\beta_t$, the true reversal of a Gaussian
step is close to Gaussian — one more reason the schedule takes *many small*
steps rather than a few large ones.

Before deriving what $\mu_\theta$ should be trained toward, here is the
tractable object that makes everything work. The unconditioned reversal is
blocked, but *conditioned on the clean endpoint* $x_0$, the reversal is pure
Gaussian algebra:

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)\;
q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
\;\propto\; q(x_t \mid x_{t-1})\; q(x_{t-1} \mid x_0).$$

*(Line 1: Bayes' theorem, conditioning everything on $x_0$. Then two
simplifications: $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$ because the
forward chain is Markov — given $x_{t-1}$, the past adds nothing — and the
denominator $q(x_t \mid x_0)$ does not involve $x_{t-1}$, so as a function of
$x_{t-1}$ it is a constant we can drop and restore by normalization.)*

Both remaining factors are known Gaussians (§3). Work in one dimension — every
covariance in sight is a multiple of $I$, so dimensions decouple and the vector
result is the scalar result applied componentwise. Write out the exponents:

$$\log q(x_{t-1} \mid x_t, x_0) = -\frac{1}{2}\left[
\frac{(x_t - \sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t}
+ \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\,x_0)^2}{1 - \bar\alpha_{t-1}}
\right] + \text{const}.$$

*(Line 2: log of a product of Gaussian densities = sum of exponents; "const"
absorbs everything with no $x_{t-1}$ in it.)*

This is a quadratic in $x_{t-1}$ — Fact 3 says the answer is a Gaussian, and
completing the square finds its parameters. Collect the $x_{t-1}^2$ and
$x_{t-1}$ terms:

$$= -\frac{1}{2}\left[
\underbrace{\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\right)}_{A}
x_{t-1}^2
\;-\; 2\underbrace{\left(\frac{\sqrt{\alpha_t}}{\beta_t}\,x_t
+ \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\,x_0\right)}_{B}
x_{t-1}\right] + \text{const}.$$

*(Line 3: expand both squares, keep only $x_{t-1}$-dependent terms.)*

A Gaussian $\mathcal{N}(\mu, \sigma^2)$ has exponent
$-\frac{1}{2\sigma^2}(x^2 - 2\mu x) + \text{const}$, so matching:
$\sigma^2 = 1/A$ and $\mu = B/A$. Simplify $A$ first:

$$A = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})}
= \frac{\alpha_t - \bar\alpha_t + 1 - \alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
= \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}.$$

*(Line 4: $\alpha_t\bar\alpha_{t-1} = \bar\alpha_t$ by definition, and
$\beta_t = 1 - \alpha_t$; the $\alpha_t$'s cancel.)*

So the posterior is $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1};\,
\tilde\mu_t(x_t, x_0),\, \tilde\beta_t)$ with

$$\boxed{\;\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,
\qquad
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\,x_t
+ \sqrt{\bar\alpha_{t-1}}\,\beta_t\, x_0}{1-\bar\alpha_t}.\;}$$

*(Line 5: $\tilde\mu_t = B/A = B\tilde\beta_t$ — multiply through and simplify.
Do this multiplication yourself on paper; it is two lines.)*

Sanity-read the mean: it is a weighted average of "where the noisy point is"
($x_t$, scaled back up) and "where the clean point is" ($x_0$, scaled down to
step $t-1$'s signal level). Early in the chain ($\bar\alpha_{t-1}$ near 1,
noise small) the $x_t$ term dominates — barely move; late in the chain the
$x_0$ term matters more — the clean image pulls harder. And $\tilde\beta_t <
\beta_t$ always: knowing the endpoint *reduces* the step's uncertainty.

Of course, at generation time we do not know $x_0$ — that is what we are trying
to produce. The resolution is the derivation's punchline (§6): the network's
real job will be to *estimate the information content of $x_0$* from $x_t$, and
the ELBO will tell us that matching $\tilde\mu_t$ is exactly the right training
target.

## 5. The variational bound: Week 22's ELBO, stretched over the chain

The model is a latent-variable model: latents $x_1, \dots, x_T$, observed data
$x_0$, generative direction $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T}
p_\theta(x_{t-1} \mid x_t)$ with $p(x_T) = \mathcal{N}(0, I)$. We want maximum
likelihood on $p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}$, and that
integral is as hopeless as Week 22's — so run the same play. Take the Week 22
ELBO (route 1) with the latent $z$ replaced by the whole noise trajectory
$x_{1:T}$, and — here is the twist that makes diffusion special — for the
variational posterior $q_\phi(z \mid x)$, use the **forward process**
$q(x_{1:T} \mid x_0)$:

$$\log p_\theta(x_0) \;\ge\;
\mathbb{E}_{q(x_{1:T} \mid x_0)}\!\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right].$$

*(This is verbatim Week 22 §4 line 4 with $z = x_{1:T}$. Check the requirements:
$q$ must be a distribution over the latents we can sample from and evaluate —
the forward process is both, thanks to §3.)*

Look at what happened to the VAE's moving parts. The encoder $q$ has **no
learnable parameters** — it is the fixed noising recipe. So there is no $\phi$
to train, no reparameterization trick needed to get encoder gradients, and no
posterior collapse — the failure mode of Week 22 §8 is structurally impossible
because there is no encoder to give up. The price: the bound's gap,
$\mathrm{KL}(q \,\|\, p_\theta\text{-posterior})$ (Week 22 route 2), can only be
shrunk by moving $p_\theta$ toward $q$'s reversal, never by improving $q$. The
entire training signal lands on the denoiser.

Now reduce the bound to computable pieces. Write the loss as the negative bound
(minimize $L$ = maximize the ELBO), and expand both products inside the log:

$$L = \mathbb{E}_q\!\left[-\log p(x_T)
- \sum_{t=1}^{T} \log p_\theta(x_{t-1} \mid x_t)
+ \sum_{t=1}^{T} \log q(x_t \mid x_{t-1})\right].$$

*(Line 1: log of a product = sum of logs; sign flipped.)*

The $q$ terms face "forward" and the $p_\theta$ terms face "backward" — they
cannot be compared term by term yet. Flip the $q$ terms with Bayes. For each
$t \ge 2$ (the Markov property lets us condition on $x_0$ for free:
$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$):

$$q(x_t \mid x_{t-1}, x_0)
= \frac{q(x_{t-1} \mid x_t, x_0)\; q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}.$$

*(Line 2: Bayes' theorem within the conditional distribution given $x_0$ — the
same identity as §4 line 1, now used in the other direction. Every factor on the
right is a §3 or §4 Gaussian: this is why §4 was derived first.)*

Substitute into the sum over $t \ge 2$ and split the logs:

$$\sum_{t=2}^{T} \log q(x_t \mid x_{t-1})
= \sum_{t=2}^{T} \log q(x_{t-1} \mid x_t, x_0)
+ \sum_{t=2}^{T} \big[\log q(x_t \mid x_0) - \log q(x_{t-1} \mid x_0)\big].$$

The second sum **telescopes** — each term's negative half cancels the next
term's positive half, exactly like $(a_2 - a_1) + (a_3 - a_2) + \cdots =
a_T - a_1$ — leaving $\log q(x_T \mid x_0) - \log q(x_1 \mid x_0)$. And the
$-\log q(x_1 \mid x_0)$ cancels the $t = 1$ term of the original sum. Collect
everything:

$$L = \mathbb{E}_q\!\left[
\log \frac{q(x_T \mid x_0)}{p(x_T)}
+ \sum_{t=2}^{T} \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)}
- \log p_\theta(x_0 \mid x_1)\right].$$

*(Line 3: regroup the surviving terms against their $p$ partners.)*

Each log-ratio, averaged under $q$, is by definition a KL divergence (Week 08 —
the outer expectation includes each term's conditioning variables, so each
becomes an expected KL between two distributions over one $x_{t-1}$):

$$\boxed{\;L = \underbrace{\mathrm{KL}\big(q(x_T \mid x_0)\,\|\,p(x_T)\big)}_{L_T}
\;+\; \sum_{t=2}^{T}
\underbrace{\mathbb{E}_q\,\mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0)\,\|\,
p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}}
\;-\; \underbrace{\mathbb{E}_q \log p_\theta(x_0 \mid x_1)}_{-L_0}\;}$$

Read the three pieces:

- **$L_T$** compares the fully-noised data to the sampling prior
  $\mathcal{N}(0,I)$. No $\theta$ anywhere — the schedule fixed it, and §3
  showed it is nearly zero. Ignore it during training.
- **$L_{t-1}$**, the heart: at every step, the learned reverse
  $p_\theta(x_{t-1} \mid x_t)$ is pushed toward the *true conditioned reversal*
  $q(x_{t-1} \mid x_t, x_0)$ — precisely the Gaussian derived in §4. The bound
  itself has told us what the network's target is.
- **$L_0$**, the final decoding step, a plain reconstruction likelihood. (Ho et
  al. give it a careful discrete-pixel treatment; for continuous physics data a
  Gaussian likelihood folds it into the same form as the $L_{t-1}$ terms — one
  line in the exercises, not a separate mechanism.)

Every KL in the sum is between two *Gaussians* — this is the payoff for all the
Gaussian bookkeeping — and Gaussian KLs are closed-form. No sampling inside the
KL, no high-variance estimates: pick a random $t$, compute one closed-form term,
descend. That is the next section.

## 6. From KLs to the loss you type: ε-prediction and $L_{\text{simple}}$

**The Gaussian KL we need.** Both distributions in $L_{t-1}$ have (by our choice
of $\sigma_t^2$) the same, fixed variance; only the means differ. Derive the KL
for that case in 1-D, Week 22 §6.1 style. With $q = \mathcal{N}(\mu_1,
\sigma^2)$, $p = \mathcal{N}(\mu_2, \sigma^2)$:

$$\log\frac{q(z)}{p(z)} = \frac{(z-\mu_2)^2 - (z-\mu_1)^2}{2\sigma^2},$$

*(the $1/\sqrt{2\pi}\sigma$ prefactors are equal and cancel)*. Take
$\mathbb{E}_{z \sim q}$ using $\mathbb{E}_q[(z-\mu_1)^2] = \sigma^2$ and
$\mathbb{E}_q[(z-\mu_2)^2] = \sigma^2 + (\mu_1 - \mu_2)^2$ (variance plus
squared bias — Week 08):

$$\mathrm{KL} = \frac{(\mu_1-\mu_2)^2}{2\sigma^2},
\qquad\text{and over $D$ independent dimensions:}\qquad
\mathrm{KL} = \frac{\|\mu_1-\mu_2\|^2}{2\sigma^2}.$$

Matching Gaussians with equal variances is just matching means, in squared
error. So

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}
\big\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\|^2\right].$$

The network could regress $\tilde\mu_t$ directly. Ho et al.'s reparameterization
of this target is what made DDPMs work well, and it comes from combining two
things you have already derived. The closed form (§3) links $x_0$, $x_t$, and
the noise that was drawn:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon
\qquad\Longrightarrow\qquad
x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}.$$

Substitute this $x_0$ into §4's $\tilde\mu_t$ and simplify. The algebra, every
step:

$$\tilde\mu_t = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\,x_t
+ \sqrt{\bar\alpha_{t-1}}\,\beta_t\,
\frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}}{1-\bar\alpha_t}$$

*(Step 1: substitution. Now use
$\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$ — the
definition of $\bar\alpha$ again.)*

$$= \frac{1}{\sqrt{\alpha_t}}\cdot
\frac{\alpha_t(1-\bar\alpha_{t-1})\,x_t + \beta_t\,x_t
- \beta_t\sqrt{1-\bar\alpha_t}\,\varepsilon}{1-\bar\alpha_t}$$

*(Step 2: pull $1/\sqrt{\alpha_t}$ out of both terms — the first term absorbs it
as $\sqrt{\alpha_t} = \alpha_t/\sqrt{\alpha_t}$ — and collect over the common
denominator.)*

$$= \frac{1}{\sqrt{\alpha_t}}\cdot
\frac{(1-\bar\alpha_t)\,x_t - \beta_t\sqrt{1-\bar\alpha_t}\,\varepsilon}{1-\bar\alpha_t}
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\right).$$

*(Step 3: $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$ — the same
cancellation as §4 line 4 — then divide through.)*

Stare at the boxed content of that parenthesis: **the ideal reverse-step mean is
"take $x_t$, subtract the right multiple of the noise that was added, rescale"**.
Everything in it is known at generation time *except* $\varepsilon$ — the one
draw that noised this sample. So give the network exactly that job: predict the
noise. Define the network as $\varepsilon_\theta(x_t, t)$ and parameterize the
learned mean *in the same functional form*:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,
\varepsilon_\theta(x_t, t)\right).$$

Then the mean-matching loss collapses — the $x_t$'s cancel and the prefactors
pull out:

$$L_{t-1} = \mathbb{E}_{x_0, \varepsilon}\!\left[
\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}
\big\|\varepsilon - \varepsilon_\theta\big(
\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\; t\big)\big\|^2
\right].$$

*(Both means share the form $\frac{1}{\sqrt{\alpha_t}}(x_t - c_t \cdot)$; their
difference is $\frac{c_t}{\sqrt{\alpha_t}}(\varepsilon_\theta - \varepsilon)$
with $c_t = \beta_t/\sqrt{1-\bar\alpha_t}$; square it and divide by
$2\sigma_t^2$.)*

**The last move: drop the weight.** Ho et al. discard the $t$-dependent
prefactor and train on

$$\boxed{\;L_{\text{simple}} =
\mathbb{E}_{t \sim \mathrm{U}\{1..T\},\, x_0,\, \varepsilon}
\Big[\big\|\varepsilon - \varepsilon_\theta\big(
\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\; t\big)
\big\|^2\Big].\;}$$

Know exactly what was dropped and which way it cuts. The true bound's weight
$\beta_t^2 / (2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ is *large at small $t$*
(with $\sigma_t^2 = \beta_t$ it approaches $\tfrac{1}{2}$ as $t \to 1$) and
*small at large $t$* (order $\beta_t$, i.e. ~0.01, when $\bar\alpha_t \approx
0$). Setting every weight to 1 therefore **up-weights the high-noise steps** —
the hard denoising problems where global structure is decided — relative to the
almost-clean final touch-ups. The result is no longer exactly the ELBO; it is a
reweighted bound that trades a little likelihood for noticeably better samples,
and Ho et al. adopted it on those grounds. Honest bookkeeping: if you quote
likelihoods, say you trained on the reweighted objective.

Look at what survives of five sections of derivation: *draw a clean sample, a
timestep, and a noise vector; noise the sample in one line; ask the network to
guess the noise; mean squared error*. Every piece of the machinery — posterior,
telescope, KLs — exists to justify why that childishly simple recipe is (a
reweighting of) a principled variational bound.

## 7. Algorithms, and a DDPM in ~50 lines

The two algorithms of the DDPM paper are now one-liners to state.

**Algorithm 1 — training.** Repeat: sample $x_0$ from data,
$t \sim \mathrm{U}\{1..T\}$, $\varepsilon \sim \mathcal{N}(0,I)$; take a
gradient step on $\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 +
\sqrt{1-\bar\alpha_t}\varepsilon, t)\|^2$.

**Algorithm 2 — sampling.** $x_T \sim \mathcal{N}(0, I)$; for $t = T \dots 1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t -
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right)
+ \sigma_t z, \qquad z \sim \mathcal{N}(0, I) \text{ (except } z = 0
\text{ at } t = 1).$$

The first term is $\mu_\theta$ — §6's "subtract the predicted noise, rescale" —
and the added $\sigma_t z$ is the reverse step's own variance: the reversal is a
*distribution*, not a point, and sampling it keeps diversity. ($T$ network
passes per sample: the cost promised in §1, now visible in the loop.)

Complete and runnable as shown (CPU, ~a minute). Toy data: a two-component 2-D
Gaussian mixture, so mode coverage is checkable by eye. Install with
`uv add torch`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# --- toy data: 2-D mixture, two modes at (+-2, 0) ---
n = 4096
comp = (torch.rand(n, 1) < 0.5).float()
data = comp * torch.tensor([[2.0, 0.0]]) \
     + (1 - comp) * torch.tensor([[-2.0, 0.0]]) + 0.3 * torch.randn(n, 2)

# --- schedule ---
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
abar = torch.cumprod(alpha, dim=0)

# --- denoiser: eps_theta(x_t, t); t enters as the feature t/T ---
net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(),
                    nn.Linear(128, 128), nn.ReLU(),
                    nn.Linear(128, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for step in range(4000):                      # Algorithm 1
    x0 = data[torch.randint(0, n, (256,))]
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    a = abar[t].unsqueeze(1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps          # closed form, Sec. 3
    pred = net(torch.cat([xt, t.float().unsqueeze(1) / T], dim=1))
    loss = ((eps - pred) ** 2).mean()                  # L_simple
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 999:
        print(f"step {step + 1}  loss {loss.item():.3f}")

with torch.no_grad():                         # Algorithm 2
    x = torch.randn(2000, 2)
    for t in range(T - 1, -1, -1):
        tcol = torch.full((2000, 1), t / T)
        eps_hat = net(torch.cat([x, tcol], dim=1))
        x = (x - beta[t] / (1 - abar[t]).sqrt() * eps_hat) / alpha[t].sqrt()
        if t > 0:
            x = x + beta[t].sqrt() * torch.randn_like(x)

print("sample mean per mode sign:",
      x[x[:, 0] > 0].mean(dim=0), x[x[:, 0] < 0].mean(dim=0))
print("fraction in right mode:", (x[:, 0] > 0).float().mean())
```

Map the lines to the math: the `xt` line is §3's closed form; the `loss` line is
$L_{\text{simple}}$ with the weight already dropped; the sampling update is
Algorithm 2 with $\sigma_t^2 = \beta_t$; the mode-fraction print is the
mode-coverage check (should be near 0.5 — compare the VAE's tendency to blur
between modes). The timestep enters as a plain scaled feature — enough for a
toy; image-scale models use sinusoidal embeddings (Week 26's position-embedding
idea, recycled for time) inside a U-Net, but nothing in the *objective* changes.

## 8. What the network actually learned: scores and Langevin dynamics

One paragraph of the README asked for this connection explicitly; it deserves a
section, because it explains *why* noise-prediction is such a natural target.
The **score** of a distribution is the gradient of its log-density with respect
to the *data*, $\nabla_x \log p(x)$ — the vector field that points "uphill
toward more probable data". Compute it for the one distribution we have in
closed form, $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I)$:

$$\nabla_{x_t} \log q(x_t \mid x_0)
= -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}
= -\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}},$$

using $x_t - \sqrt{\bar\alpha_t}x_0 = \sqrt{1-\bar\alpha_t}\,\varepsilon$ from
the closed form. So the noise the network predicts *is* (a scaled negative of)
the score: $\varepsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\;
\nabla_{x_t}\log q(x_t)$. Training to predict noise is **score matching** in
disguise — the network learns, at every noise level, which direction leads back
toward the data manifold.

That reframes sampling. **Langevin dynamics** — a tool physics had first — is
the stochastic process $x \leftarrow x + \tfrac{\eta}{2}\nabla_x \log p(x) +
\sqrt{\eta}\, z$, $z \sim \mathcal{N}(0,I)$: gradient ascent on log-probability
plus injected noise, which (for small step $\eta$, long times) samples $p$
rather than merely maximizing it. You have met its structure twice: SGD's
update is a noisy gradient step (Week 08's optimizer project — noise from
minibatching rather than injection), and molecular-dynamics thermostats add
noise to sample a thermal ensemble instead of finding the energy minimum.
Algorithm 2's update is exactly a Langevin-like step: drift along the learned
score (the $\varepsilon_\theta$ term, §6's "subtract the noise") plus fresh
Gaussian noise ($\sigma_t z$), with the noise level annealed down the schedule
so the walker is guided from the pure-noise prior onto the data manifold.
Diffusion models are annealed Langevin samplers whose score was learned by
denoising — the README's phrase "Langevin dynamics run in reverse", now with
every term defined.

## 9. Conditional generation and classifier-free guidance

A calorimeter simulator that generates "some shower" is useless; the physics
needs "a shower *given* an incident photon of energy $E$ at angle $\eta$"
(Capstone 3b, and the CaloChallenge setting). So the model must learn
$p(x \mid c)$ for a condition $c$.

**The plumbing is the easy half.** Give the network the condition as an extra
input, $\varepsilon_\theta(x_t, t, c)$ — concatenated like $t$, or via an
embedding — and train $L_{\text{simple}}$ on (sample, condition) pairs. This
works, but in practice conditional models trained this way often *underuse* the
condition: $c$ explains little of the noise-prediction error at high $t$, so
the gradient pressure to respect it is weak, and generated samples obey the
condition only loosely.

**Guidance sharpens conditioning at sampling time.** The idea (classifier
guidance, the precursor): sample not from $p(x \mid c)$ but from a sharpened
distribution that overweights how *recognizable* the condition is,

$$p_w(x \mid c) \;\propto\; p(x \mid c)\, p(c \mid x)^w, \qquad w \ge 0,$$

— boost the probability of samples from which a classifier could recover $c$.
Take the score (log, then $\nabla_x$; the normalizer has no $x$ and drops):

$$\nabla_x \log p_w(x \mid c) = \nabla_x \log p(x \mid c)
+ w\, \nabla_x \log p(c \mid x).$$

The original method trained an explicit classifier $p(c \mid x_t)$ on noisy
data to supply the second term. **Classifier-free guidance** (Ho & Salimans)
eliminates it with Bayes' theorem. From
$p(c \mid x) = p(x \mid c)\, p(c) / p(x)$:

$$\nabla_x \log p(c \mid x) = \nabla_x \log p(x \mid c) - \nabla_x \log p(x)$$

*(take logs, differentiate; $\nabla_x \log p(c) = 0$ since $p(c)$ has no $x$).*
Substitute:

$$\nabla_x \log p_w(x \mid c)
= (1 + w)\, \nabla_x \log p(x \mid c) \;-\; w\, \nabla_x \log p(x).$$

The classifier is gone: the sharpened score is a linear combination of the
*conditional* and *unconditional* scores. And §8 says scores are just scaled
$\varepsilon$'s, so at each sampling step use

$$\boxed{\;\tilde\varepsilon = (1 + w)\,\varepsilon_\theta(x_t, t, c)
\;-\; w\,\varepsilon_\theta(x_t, t, \varnothing)\;}$$

in place of $\varepsilon_\theta$ in Algorithm 2. Here $\varnothing$ is a
learned **null condition** — and the one training-time change that makes this
possible is *condition dropout*: during training, replace $c$ by $\varnothing$
a small fraction of the time (10–20%), so a single network learns both the
conditional and the unconditional score. Two forward passes per sampling step
(one with $c$, one with $\varnothing$), a weighted extrapolation, done.

Read the formula as an extrapolation *away from* the unconditional model:
$w = 0$ is the plain conditional model; increasing $w$ pushes samples toward
"unambiguously consistent with $c$" and away from "generic". The price is
diversity — high $w$ collapses toward the most condition-typical samples and
distorts the distribution's tails. For images that trade is often worth it; for
*physics* it is dangerous: a calo generator is supposed to reproduce the full
shower *distribution* at each energy, fluctuations included, not the single
most-typical shower. Sweep $w$ and validate on distribution-level observables
(E4–E5) — never assume the image-generation folklore ("crank $w$ up")
transfers to simulation.

## 10. Diffusion vs flows vs VAEs, for calorimeters

You have now built the two deep generative families this course requires (VAE
and DDPM). A **normalizing flow** — an invertible map with exact likelihood —
is the third family; the full derivation is optional (`optional-flows.md` in
the Week 23 folder). (A **calorimeter**, for
any reader who skipped Phase 2: a detector layer that measures particle
energies by absorbing them — the incoming particle cascades into a **shower**
of secondaries across a grid of cells, and simulating those showers with
Geant4, the standard particle-transport simulator, dominates collider computing
budgets. Hence "fast simulation": learn the map from incident particle to
deposited-energy image, and sample it cheaply.) The honest comparison, on the
axes that matter for that job:

| | VAE (W22) | Flow (`optional-flows.md`) | DDPM (this week) |
|---|---|---|---|
| trains on | ELBO (bound) | exact NLL | reweighted bound ($L_{\text{simple}}$) |
| likelihood access | bound only | exact, cheap | bound only (exact needs ODE tricks) |
| sampling cost | 1 pass | 1 pass | $T$ passes (10²–10³) |
| sample quality | lowest (blur, averaging) | good | best at scale |
| architecture freedom | full | invertible + tractable det only | full |
| conditioning | easy (concat/embed) | easy | easy + guidance knob $w$ |
| classic failure | posterior collapse | capacity limits from invertibility | sampling too slow; over-guidance |

The pattern: diffusion buys back the flow's architecture freedom *and* beats
the VAE's sample quality by paying at sampling time — it moved the cost from
training-objective compromises to the sampling loop. For calorimeter
simulation, that is exactly the painful axis: the entire point of fast-sim is
to be *fast*, and a thousand network passes per shower can lose to Geant4's
actual cost if you are careless. This is why the CaloChallenge generation of
diffusion entries (CaloScore, CaloDiffusion — skim one this week) lean on
reduced-step samplers and distillation, and why Capstone 3b **requires**
reporting sampling cost per shower next to the physics validation. A
comparison table without the cost row is marketing.

For the physics validation itself, nothing changes from Capstone 2: energy
response linearity, resolution vs $\sqrt{E}$, shower-shape distributions
against the full simulator, with a quantitative distance per observable. The
generative family is new; the standard of evidence is not.

## Check yourself

1. Derive the two-step composition: starting from $x_t = \sqrt{\alpha_t}
   x_{t-1} + \sqrt{1-\alpha_t}\varepsilon_t$, show
   $q(x_t \mid x_{t-2}) = \mathcal{N}(\sqrt{\alpha_t\alpha_{t-1}}x_{t-2},
   (1-\alpha_t\alpha_{t-1})I)$, naming the Gaussian fact used.
2. Why $\sqrt{1-\beta_t}$ and not, say, coefficient 1? Show what property holds
   with it and fails without it, and why that matters at $t = T$.
3. In the chain ELBO, what plays the role of Week 22's $q_\phi(z \mid x)$? What
   two VAE complications disappear because of that choice, and what does
   training lose the ability to do?
4. Derive $\mathrm{KL}(\mathcal{N}(\mu_1, \sigma^2 I) \,\|\,
   \mathcal{N}(\mu_2, \sigma^2 I)) = \|\mu_1 - \mu_2\|^2 / 2\sigma^2$.
5. State the substitution that turns $\tilde\mu_t(x_t, x_0)$ into
   $\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
   \varepsilon)$, and say in one sentence why "predict $\varepsilon$" then
   becomes the natural network target.
6. $L_{\text{simple}}$ drops a $t$-dependent weight from the bound. Which
   timesteps does the true weight emphasize, which does dropping it emphasize,
   and what is the argument for the swap?
7. Derive the classifier-free guidance formula $\tilde\varepsilon = (1+w)
   \varepsilon_\theta(x_t, t, c) - w\varepsilon_\theta(x_t, t, \varnothing)$
   from $p_w(x \mid c) \propto p(x \mid c)p(c \mid x)^w$. What single
   training-time change makes both terms available from one network?
8. Your conditional calo DDPM at $w = 5$ nails the mean energy response but the
   shower-shape *widths* come out too narrow. Explain the mechanism and the fix.

## Answers

1. Substitute the $x_{t-1}$ step into the $x_t$ step:
   $x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} +
   \sqrt{\alpha_t(1-\alpha_{t-1})}\varepsilon_{t-1} +
   \sqrt{1-\alpha_t}\varepsilon_t$. The two noise terms are independent
   zero-mean Gaussians; by Fact 2 (variances add, coefficients squared) their
   sum is Gaussian with variance $\alpha_t(1-\alpha_{t-1}) + 1 - \alpha_t =
   1 - \alpha_t\alpha_{t-1}$.
2. With $\sqrt{1-\beta_t}$, unit variance is preserved:
   $(1-\beta_t) + \beta_t = 1$, so the chain converges to
   $\mathcal{N}(0, I)$ — a fixed, data-independent, samplable endpoint. With
   coefficient 1 the variance grows by $\beta_t$ every step, so $x_T$'s
   distribution depends on $T$ and the data scale and the generative process
   has no clean starting prior.
3. The fixed forward process $q(x_{1:T} \mid x_0)$. Gone: encoder gradients
   (hence no reparameterization trick) and posterior collapse (no encoder to
   collapse). Lost: the bound's gap can no longer be tightened by improving
   $q$ — only the reverse model moves.
4. The equal prefactors cancel in the log-ratio, leaving
   $[(z-\mu_2)^2 - (z-\mu_1)^2]/2\sigma^2$; under $q$,
   $\mathbb{E}[(z-\mu_1)^2] = \sigma^2$ and $\mathbb{E}[(z-\mu_2)^2] =
   \sigma^2 + (\mu_1-\mu_2)^2$, so the KL is $(\mu_1-\mu_2)^2/2\sigma^2$;
   independent dimensions add into the squared norm.
5. Solve the closed form for the clean point:
   $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\varepsilon)/\sqrt{\bar\alpha_t}$, and
   substitute into $\tilde\mu_t$. Every quantity in the resulting mean is
   available at sampling time except $\varepsilon$ — so a network that predicts
   $\varepsilon$ supplies exactly the missing ingredient of the ideal step.
6. The true weight $\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ is largest
   at small $t$ (nearly-clean steps) and tiny at large $t$. Setting it to 1
   up-weights the high-noise steps where global structure is formed. Ho et al.
   keep it because it empirically improves sample quality — at the cost that
   the objective is no longer exactly the likelihood bound.
7. Score of $p_w$: $\nabla\log p(x \mid c) + w\nabla\log p(c \mid x)$; Bayes
   gives $\nabla\log p(c \mid x) = \nabla\log p(x \mid c) - \nabla\log p(x)$;
   combining yields $(1+w)\nabla\log p(x \mid c) - w\nabla\log p(x)$, and the
   $\varepsilon \leftrightarrow$ score correspondence (§8) converts scores to
   $\varepsilon$'s. The enabling change: condition dropout — training with $c$
   replaced by a null token $\varnothing$ 10–20% of the time.
8. High guidance extrapolates toward the most condition-typical samples,
   suppressing the distribution's natural fluctuations — means survive, tails
   and widths shrink. Fix: lower $w$ (or 0) and validate on distribution-level
   observables; for simulation, guidance strength is a physics-validation
   parameter, not a quality knob.

## New terms

- **diffusion model / DDPM** — generative model that learns to reverse a fixed
  Gaussian noising process, step by step.
- **forward (noising) process** — the fixed chain
  $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$.
- **noise schedule ($\beta_t$)** — the per-step noise variances; sets how fast
  signal is destroyed (unrelated to β-VAE).
- **$\alpha_t$, $\bar\alpha_t$** — $1-\beta_t$ and its running product; the
  surviving signal fraction at step $t$.
- **variance-preserving** — the $\sqrt{1-\beta_t}$ scaling that keeps total
  variance fixed so the chain converges to $\mathcal{N}(0, I)$.
- **reverse process ($p_\theta$)** — the learned Gaussian chain
  $p_\theta(x_{t-1} \mid x_t)$ that generates by denoising.
- **forward-process posterior** — $q(x_{t-1} \mid x_t, x_0)$; Gaussian with
  mean $\tilde\mu_t$ and variance $\tilde\beta_t$; the per-step training
  target.
- **ε-prediction** — parameterizing the reverse mean via a network that
  predicts the added noise.
- **$L_{\text{simple}}$** — the unweighted noise-prediction MSE; a reweighted
  variational bound.
- **score** — $\nabla_x \log p(x)$; the direction of increasing data
  probability. Predicting $\varepsilon$ estimates it.
- **Langevin dynamics** — noisy gradient ascent on log-probability that samples
  a distribution; diffusion sampling is its annealed, learned-score version.
- **classifier-free guidance (CFG)** — sampling with
  $(1+w)\varepsilon_\theta(x,c) - w\varepsilon_\theta(x,\varnothing)$ to
  sharpen conditioning without a classifier.
- **guidance weight ($w$) / null condition ($\varnothing$) / condition
  dropout** — the CFG knob, the learned "no condition" token, and the training
  trick that teaches one network both scores.

## Going deeper

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*
  (arXiv 2006.11239), §§1–3 + Algorithms 1–2 — the derivation source; after
  this lesson its Eq. (5)–(12) should read as your own §§4–6.
- Lilian Weng, *What are Diffusion Models?* (lilianweng.github.io) — the best
  companion to the algebra, and the bridge to score-based/SDE formulations.
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (arXiv 2207.12598) —
  short; read fully and match its equations to §9.
- Prince, *Understanding Deep Learning*, Ch. 18 — the same derivation in
  different notation; translating between notations is the real check.
- One CaloChallenge diffusion entry — CaloScore or CaloDiffusion — read for how
  the field conditions on energy and which shower observables the validation
  uses; this is the standard Capstone 3b will be held to.
