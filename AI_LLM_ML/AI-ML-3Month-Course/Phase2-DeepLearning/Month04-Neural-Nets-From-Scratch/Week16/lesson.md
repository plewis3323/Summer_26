# Week 16 — Training Dynamics + Mini-Project

~5 hrs on the derivations and diagnostics, then the mini-project (`project.md`).
Before starting you should be able to: write the backprop recursion for
$\delta^{(l)}$ and the softmax/cross-entropy gradient $p - y$ (Week 13); read a
computational graph and check a backward pass against finite differences
(Week 14); write a PyTorch `nn.Module`, a five-line training loop, and a
character MLP (Week 15); state why we split data and what CV estimates
(Week 09); recall Adam and why SGD on mini-batches is noisy on purpose
(Week 08).

You can now derive a gradient, implement autograd, and call `loss.backward()`.
That is not the same as a network that *trains*. Whether a deep net trains at
all is a question about activation and gradient statistics through layers —
keep the distribution stable stage to stage, or lose it. This week derives the
two standard initializations that try to do that (Xavier, then He), teaches
you to read the plots that say which pathology you have, and adds the two
normalization layers (batchnorm, layernorm) that make the problem less
fragile. The mini-project then pits a carefully trained MLP against your
Capstone 1 BDT on MAGIC. Tabular data is the trees' home turf; a loss is a
finding.

## 1. Why training fails: statistics through layers

A 2-layer net on two-moons (Week 15) trains from almost any reasonable start.
A 5-layer net on characters, or on Hillas features, often does not: loss stuck,
loss NaN, accuracy at chance. The failure is usually not "the architecture is
wrong". It is that the *numbers flowing through the architecture* have gone
off the rails.

Forward, each layer is

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = g(z^{(l)}),$$

with $a^{(0)} = x$. If the entries of $z^{(l)}$ grow with depth, **tanh** and
**sigmoid** flatten: $g'(z) \to 0$, and Week 13's recursion

$$\delta^{(l)} = \big(W^{(l+1)\top} \delta^{(l+1)}\big) \odot g'(z^{(l)})$$

multiplies by a number near zero at every layer — **vanishing gradients**, the
early layers stop learning. If the entries of $z^{(l)}$ shrink with depth,
later layers see near-zero inputs, $W^{(l)} a^{(l-1)}$ is tiny, and the same
thing happens for a different reason. If they grow *without* a saturating $g$
(ReLU does not saturate on the positive side), $z$ and $\delta$ can explode
until the loss is NaN — **exploding gradients**.

The design goal is boring on purpose: the typical size of activations, and of
gradients, should be roughly the same at every layer. That is a *transport*
problem.

**Beam-transport analogy, from scratch.** A particle accelerator is a pipe
with a sequence of magnets. A bunch of protons (or electrons) travels down
the pipe. Each particle has a position relative to the centerline and a small
angle relative to the forward direction; the bunch as a whole has a **beam
size** — the RMS spread of those positions. Each quadrupole magnet focuses in
one plane and defocuses in the other; the art of **beam transport** is
choosing the magnet strengths so that the beam size handed to the next stage
is about the same as the size this stage received. If one stage magnifies,
the next stage sees a fatter beam and, with the same magnet, magnifies more —
the envelope grows exponentially and the beam scrapes the pipe. If one stage
demagnifies too hard, the bunch shrinks into a pencil and then a later
defocusing stage blows it up, or the signal disappears into the noise of the
aperture. Stable transport is not "focus as hard as possible". It is *match
the envelope stage to stage*.

Activations through layers are the same job. The "beam size" is
$\mathrm{Var}(a^{(l)})$, or more loosely the RMS of the pre-activations
$z^{(l)}$. Each weight matrix is a magnet. Initialization (this week's §2–3)
picks the magnet strengths *before* training so the envelope is stable at
step 0. Normalization layers (§5–6) *re-match* the envelope at every step.
The diagnostic plots (§4) are beam-position monitors: histograms and norms
per layer, read the way an operator reads an orbit plot.

The backward pass is transport in the other direction: $\delta^{(l)}$ is an
"error beam" sent back through the same magnets. Xavier's fan-average
compromise exists because you have to match *both* directions with one $W$.

## 2. Xavier initialization

**Xavier** (Glorot) initialization chooses $\mathrm{Var}(W)$ so that a layer
neither inflates nor deflates the activation variance. Derive it, then the
compromise.

**Setup.** One layer, drop the bias for a moment (or take it zero). A single
pre-activation is

$$z_j = \sum_{i=1}^{n_{\mathrm{in}}} W_{ji}\, x_i,$$

where $n_{\mathrm{in}}$ is the **fan-in** — how many inputs feed this unit.
Assume: the $x_i$ are independent, zero-mean, and share a variance
$\mathrm{Var}(x)$; the $W_{ji}$ are independent of the $x$'s and of each
other, zero-mean, and share a variance $\mathrm{Var}(W)$. Then (Week 08:
variance of a sum of independent zero-mean terms is the sum of variances)

$$\mathrm{Var}(z_j)
 = \sum_{i=1}^{n_{\mathrm{in}}} \mathrm{Var}(W_{ji} x_i)
 = n_{\mathrm{in}}\,\mathrm{Var}(W)\,\mathrm{Var}(x).$$

**Forward condition.** We want $\mathrm{Var}(z) = \mathrm{Var}(x)$, so the
next layer sees inputs of the same typical size. That forces

$$\boxed{\;\mathrm{Var}(W) = \frac{1}{n_{\mathrm{in}}}\;}$$

For **tanh**, $g'(0) = 1$ and tanh is roughly linear near zero, so
$\mathrm{Var}(a) \approx \mathrm{Var}(z)$ if we have succeeded in keeping $z$
small. That is why this variance is the tanh (and sigmoid, up to a gain
factor) result: the activation does not, by itself, rescale the variance,
*provided you are in the linear regime* — which is exactly what the condition
is trying to maintain. If you initialize too large, you leave that regime,
tanh saturates, and the derivation's assumption is already false; the
histogram in §4 will show it.

**Backward condition.** The error at the inputs of this layer (Week 13) is
$\delta^{\mathrm{in}}_i = \sum_{j=1}^{n_{\mathrm{out}}} W_{ji}\,
\delta^{\mathrm{out}}_j$ times an activation derivative we are again treating
as $\approx 1$. The same variance calculation with the **fan-out**
$n_{\mathrm{out}}$ (how many units this one feeds) gives
$\mathrm{Var}(W) = 1/n_{\mathrm{out}}$ if we want
$\mathrm{Var}(\delta^{\mathrm{in}}) = \mathrm{Var}(\delta^{\mathrm{out}})$.

**Fan-average compromise.** One matrix cannot satisfy both $1/n_{\mathrm{in}}$
and $1/n_{\mathrm{out}}$ unless the layer is square. Glorot & Bengio take the
harmonic-looking average of the two conditions:

$$\mathrm{Var}(W) = \frac{2}{n_{\mathrm{in}} + n_{\mathrm{out}}}.$$

That is **Xavier / Glorot** initialization. Sampling
$W_{ji} \sim \mathcal{N}(0, 2/(n_{\mathrm{in}}+n_{\mathrm{out}}))$ is the
normal version; the original paper used a uniform distribution with the same
variance,
$W_{ji} \sim U\big[-\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})},
+\sqrt{6/(n_{\mathrm{in}}+n_{\mathrm{out}})}\big]$. Either is "Xavier". Use
it with tanh (or sigmoid, with the paper's gain). Do not expect it to be
right for ReLU — §3 is that correction.

```python
import torch
import torch.nn as nn

def xavier_normal_(w):
    n_in, n_out = w.shape[1], w.shape[0]          # Linear: (n_out, n_in)
    std = (2.0 / (n_in + n_out)) ** 0.5
    with torch.no_grad():
        w.normal_(0.0, std)

layer = nn.Linear(128, 64)
xavier_normal_(layer.weight)
# equivalent: nn.init.xavier_normal_(layer.weight)
```

Exercise E1: a 5-layer tanh MLP, three inits at step 0 — $\mathcal{N}(0,1)$,
Xavier, He — and per-layer activation histograms. $\mathcal{N}(0,1)$ saturates
(mass at $\pm 1$); Xavier is stable; He is the ReLU answer used on tanh, so
slightly too wide. One-line reading per plot is the accept.

## 3. He initialization

**ReLU**, $g(z) = \max(0, z)$, throws away the whole negative half-line. That
changes the variance calculation, and the change is a factor of 2.

Redo §2's forward pass with $a = \mathrm{ReLU}(z)$. Keep the same assumptions
on $W$ and $x$, so $z$ is a sum of many independent terms and is approximately
symmetric about zero (central limit, Week 08). Then half of the $z$'s are
negative and map to $0$. The second moment of the output is

$$\mathbb{E}[a^2]
 = \mathbb{E}[z^2 \cdot \mathbf{1}_{z>0}]
 = \tfrac12\,\mathbb{E}[z^2],$$

because $z^2$ is even and the two halves of a symmetric distribution
contribute equally. For a zero-mean $a$ this is $\mathrm{Var}(a)$; ReLU's
mean is not quite zero (it is positive), but the second-moment statement is
the one that matters and is exact under symmetry.

From §2, $\mathbb{E}[z^2] = n_{\mathrm{in}}\,\mathrm{Var}(W)\,\mathrm{Var}(x)$.
We want $\mathrm{Var}(a) \approx \mathrm{Var}(x)$, so

$$\tfrac12\, n_{\mathrm{in}}\,\mathrm{Var}(W)\,\mathrm{Var}(x)
 \approx \mathrm{Var}(x)
 \quad\Rightarrow\quad
\boxed{\;\mathrm{Var}(W) = \frac{2}{n_{\mathrm{in}}}\;}$$

That factor of 2 is "ReLU kills half the variance, so double the weights to
put it back." This is **He** (Kaiming) initialization. The backward-pass
version uses fan-out instead of fan-in; PyTorch's
`nn.init.kaiming_normal_(w, nonlinearity="relu")` defaults to fan-in, which
is the forward condition above. `nn.Linear` itself is initialized with a
Kaiming *uniform* variant — Week 15's "already sensibly initialized" is this
family, which is why a 2-layer ReLU net often trains without you thinking
about it, and why a tanh net using those same defaults is slightly
mis-matched.

```python
def he_normal_(w):
    n_in = w.shape[1]
    std = (2.0 / n_in) ** 0.5
    with torch.no_grad():
        w.normal_(0.0, std)

# equivalent: nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
```

Pairing, which you should memorize: **tanh / sigmoid → Xavier; ReLU (and
leaky ReLU) → He.** Mixing them is the E1 "expected" wrong histogram.

## 4. Diagnosing a training run

Initialization sets step 0. Training can still wreck the statistics by step
100. You find out by instrumenting, not by staring at a scalar loss.

### 4.1 Activation histograms

After a forward pass, collect $a^{(l)}$ (or $z^{(l)}$) per layer and
histogram. At step 0 this is E1. During training it is a health check:

- **tanh / sigmoid, mass piled at $\pm 1$ (or $0$ and $1$).** Saturated.
  $g' \approx 0$, gradients to earlier layers die. Cause: weights too large
  (bad init, or LR so high that one step jumped into the flat), or inputs not
  scaled (Hillas `fSize` next to `fAlpha` — the large-range feature dominates
  $z$).
- **ReLU, mass as a spike at 0 plus a tail.** Normal. A spike that is *almost
  all* of the mass, on a unit that stays there for a whole epoch, is a dead
  unit (§4.3).
- **Any activation, variance collapsing toward 0 with depth.** The envelope
  shrank; later layers are idle.
- **Any activation, variance growing with depth.** The envelope blew up;
  saturating $g$ will clip it, ReLU will explode.

```python
@torch.no_grad()
def activation_hists(model, xb):
    x = xb
    stats = []
    for name, mod in model.named_children():
        if not isinstance(mod, nn.Linear):
            continue
        z = mod(x)
        a = torch.tanh(z)                         # or relu
        stats.append((name, a.mean().item(), a.std().item(),
                      a.detach().cpu().flatten().numpy()))
        x = a
    return stats
```

Plot one histogram per layer, same x-range, at step 0 and after a few hundred
steps. The reading is comparative: did depth *change* the shape?

### 4.2 Gradient norms and update-to-weight ratios

Two scalar summaries per parameter tensor, every few steps:

- **Gradient norm** $\|\nabla_W L\|$. If it grows with depth on the way
  *backward* (layer 1's grad much larger than layer $L$'s), you are exploding
  going back; if it shrinks to $10^{-8}$, you are vanishing. Log-scale bar
  plots vs layer index make this obvious.
- **Update-to-weight ratio**
  $\mathrm{std}(\eta\, \nabla_W L) / \mathrm{std}(W)$ — the typical size of
  this step relative to the typical size of the parameter. A well-tuned LR
  lands this around $10^{-3}$ (Karpathy's empirical rule; order-of-magnitude,
  not a law). Much smaller: the LR is a no-op, loss crawls. Much larger: one
  step rewrites the layer, often into saturation or NaN.

```python
@torch.no_grad()
def update_ratios(model, lr):
    ratios = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        w_std = p.data.std().item()
        u_std = (lr * p.grad).std().item()
        ratios[name] = u_std / (w_std + 1e-12)
    return ratios
```

These are cheap. Log them. A training run without them is a beamline without
monitors.

### 4.3 The classic pathologies

Four signatures you will diagnose in E2–E5, plus the LR cases.

**Dead ReLUs.** A ReLU unit with $z < 0$ on every example in an epoch has
$g' = 0$ on every example, so its incoming weights get zero gradient and
never recover. A large LR (or a large init) can push the pre-activation
permanently negative: one bad step, then silence. The diagnostic is the
*fraction of units that were exactly 0 for a whole epoch*, vs LR. E2: that
fraction rises with LR, and the mechanism is this paragraph in one line.

**Saturated tanh (or sigmoid).** Histogram mass at the rails; gradient norms
shrinking toward the input; update ratios $\ll 10^{-3}$ in early layers even
when the output layer still moves. Fix: Xavier, scale the inputs, lower the
LR, or switch to ReLU. E4's broken run is unscaled inputs plus bad init.

**Exploding gradients.** Loss goes NaN, or gradient norms grow by $10\times$
per layer. Cause: LR too high (E3's silent `lr=10`), or depth $\times$
too-large init without a saturating activation to clip. Fixes, in order:
lower the LR, He/Xavier as appropriate, **gradient clipping**
(`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — scale
the whole grad vector down if its norm exceeds `max_norm`). Clipping is a
seatbelt, not a substitute for matching the envelope.

**Missing `zero_grad()`.** Week 15's trap, E5's broken run: `.grad`
accumulates, so each step uses a running sum of all previous gradients.
Signature: loss decreases oddly, gradient norms grow roughly linearly with
step even at a modest LR, nothing errors. Mechanism: your Week 14 `+=`.

**LR too high vs too low**, without the drama above. Too high: loss diverges
or oscillates, update ratios $\gg 10^{-3}$. Too low: loss is almost flat,
update ratios $\ll 10^{-3}$, histograms look like step 0 forever. A **learning
rate schedule** (start higher, decay when validation plateaus; or cosine
anneal) is how you get both: big steps while the envelope is being found,
small steps while it is being refined. Goodfellow Ch. 8 is the reference;
this week you need the diagnosis more than a particular schedule.

## 5. Batch normalization

**Batch normalization (batchnorm, BN)** re-matches the envelope *inside the
network*, every forward pass, by standardizing each feature to zero mean and
unit variance over the **mini-batch**.

For a 2D batch $X \in \mathbb{R}^{B \times D}$ ($B$ examples, $D$ features —
an MLP hidden layer), BN computes, for each feature $j$,

$$\mu_j = \frac{1}{B}\sum_{b=1}^{B} z_{bj}, \qquad
\sigma_j^2 = \frac{1}{B}\sum_{b=1}^{B} (z_{bj} - \mu_j)^2,$$

then

$$\hat{z}_{bj} = \frac{z_{bj} - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}}, \qquad
y_{bj} = \gamma_j \hat{z}_{bj} + \beta_j.$$

$\varepsilon$ is a tiny stabilizer (Week 08's "don't divide by zero").
$\gamma_j, \beta_j$ are **learned** scale and shift — the layer can undo the
standardization if the loss wants a different mean or variance; what it
*cannot* undo is the coupling that forced all batch elements to share one
$(\mu_j, \sigma_j)$. What BN normalizes **over**: the batch axis (and, in a
CNN, the spatial axes too). What it does **not** normalize over: the feature
axis. Feature $j$ is standardized using only feature $j$'s values in this
batch.

**Train vs eval.** During training, $\mu$ and $\sigma$ come from the current
mini-batch — so the network's output for example $b$ *depends on which other
examples happened to share the batch*. That is noise, and it is regularizing,
and it is why BN is ill-defined for $B = 1$. The module also maintains
**running statistics**: an exponential moving average of those per-batch
$\mu, \sigma^2$. At **eval** time (`model.eval()`), BN freezes and uses the
running stats, so the network is once again a deterministic function of a
single example. Forgetting `model.eval()` at test time is a silent accuracy
drop: you are still normalizing with tiny, jittery batch stats. Forgetting
`model.train()` when you resume training is the opposite bug.

Week 07's moments: $\mu$ is a mean, $\sigma^2$ a variance, estimated
on-the-fly from $B$ samples. Small $B$ makes them noisy (the sample variance
of a variance goes as $1/B$); that is why BN likes reasonably large batches
and why the running average exists.

**Scale invariance of the loss, before BN.** Let $z = Wx$ and replace $W$ by
$cW$ with $c > 0$. Then $z \to cz$, the batch mean $\mu \to c\mu$, the batch
std $\sigma \to c\sigma$, and $\hat{z}$ is unchanged. The BN output
$y = \gamma\hat{z} + \beta$ is therefore invariant to rescaling $W$, and so
is every loss computed from $y$. (The learned $\gamma, \beta$ absorb any
scale the loss actually wants.) This is the derivation the README asked for,
and it is why BN makes training less sensitive to the overall scale of $W$ —
Xavier/He still help at step 0, but a factor-of-two mistake in $\mathrm{Var}(W)$
is no longer fatal. It is also why you can often crank the LR up once BN is
in (E6: BN widens the range of workable LRs on the Week 15 char MLP).

```python
class MLPBN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)       # D = n_hidden features
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))     # BN on pre-activations is
        return self.fc2(h)                        # the original recipe
```

Place BN *before* the activation (Ioffe's original) or after; both are used.
Be consistent, and do not BN the output logits — those need their overall
scale to mean something to the loss.

## 6. Layer normalization

**Layer normalization (layernorm, LN)** uses the same standardize-then-affine
formula, but the moments are taken **over features, for each example
separately**. For one row $z_b \in \mathbb{R}^{D}$,

$$\mu_b = \frac{1}{D}\sum_{j=1}^{D} z_{bj}, \qquad
\sigma_b^2 = \frac{1}{D}\sum_{j=1}^{D} (z_{bj} - \mu_b)^2,$$

then $\hat{z}_{bj} = (z_{bj} - \mu_b) / \sqrt{\sigma_b^2 + \varepsilon}$ and
$y_{bj} = \gamma_j \hat{z}_{bj} + \beta_j$. What LN normalizes **over**: the
feature axis of *this* example. What it does **not** use: any other example
in the batch. There is no train/eval split of statistics — the moments are
always computed from the current vector — so `model.eval()` does not change
LN's arithmetic.

```python
self.ln = nn.LayerNorm(n_hidden)              # normalized shape = feature dim
h = torch.relu(self.ln(self.fc1(x)))
```

**Why transformers prefer it (preview).** A transformer (Phase 3) processes a
*sequence* of tokens, each a $D$-vector. Batches mix sequences of different
lengths; the "batch" axis is not a set of i.i.d. copies of one feature, and
a token's representation should not depend on which other sequences happened
to share the GPU that step. LN normalizes *within* a token (or within a
layer's feature vector), so it is well-defined at batch size 1, independent
of padding, and stable when $B$ is small — the regime transformers actually
run in. BN's batch coupling, which is a feature for large-batch vision, is a
bug there. You do not need the transformer equations this week; you need the
axis: BN = over batch, per feature; LN = over features, per example.

## 7. Mini-project: MLP vs the Capstone-1 BDT

The rest of the week is the build in `project.md`. Framing, so you do not
spend the hours on the wrong question.

**The comparison.** A PyTorch MLP on the Capstone 1 MAGIC table, against the
Capstone 1 BDT, under an **identical CV protocol**. Same splits, same outer
folds, same seeds, comparable tuning budget, same metric (AUC), test set
touched once. The Week 12 / Week 24 rule has not changed because the model is
now a net. Retrain the BDT inside *this* repo (or load the frozen Capstone 1
pipeline as a callable that does not get a second tuning pass). Do not quote
a number from a different protocol.

**What "win" means.** Match or beat the BDT's outer-fold AUC, *or* lose and
explain why. The roadmap's criterion is explicit: tabular $\neq$ deep-learning
home turf. Ten heterogeneous Hillas features, correlated by construction,
$n \sim 10^4$, no spatial layout for a net to exploit — that is the regime
Week 11 spent a section on. Trees ignore scale and monotone reparameterization;
your MLP badly does not (scale the inputs; the diagnostics will scream if you
forget). A BDT win with a writeup that points at sample size, lack of
structure, and the Week 11 argument is a *result*. An MLP "win" from a leaked
split or a BDT that was not given the same budget is not.

**What you tune on the MLP side.** Init (He, if ReLU), LR (read the
update-to-weight ratios; do not grid-search blind), depth/width (start
shallow — this is not ImageNet), batchnorm or not (E6's lesson: BN widens
workable LRs; on tabular $B$ is whatever you choose, so BN is legal), early
stopping on validation loss. The diagnostic plots from §4 belong in the
writeup: they are how you know the net *trained*, as opposed to how you know
it *scored*.

**What you do not do.** Do not hunt for a net that beats the BDT by adding
features the BDT was denied. Do not touch the test set to decide depth. Do
not skip the logistic baseline — if logistic already matches the BDT, the
story is linear Hillas structure, and the MLP vs BDT plot is a footnote.

Month 04 closes here: mini-project notebook plus short writeup (MLP vs BDT,
diagnostics plots), three diagnosed-and-fixed broken runs (E3–E5), tag
`month-04-complete`, `retro.md`, one open-question issue. The month's
flagship re-derivation is Week 13's $\delta^{(l)}$ recursion, cold.

Work `project.md` in order. The ordering is the protocol made mechanical.

## Check yourself

1. In the beam-transport picture, what is the analog of $\mathrm{Var}(a^{(l)})$,
   and what goes wrong if one layer systematically magnifies it?
2. Derive $\mathrm{Var}(W) = 1/n_{\mathrm{in}}$ from the forward variance
   condition. Which assumption about tanh did you use to pass from $z$ to $a$?
3. Why is the fan-average $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$ a
   *compromise*? What two conditions can one $W$ not satisfy at once unless
   the layer is square?
4. Where does He's factor of 2 come from? Pair each init with its activation.
5. A tanh net's layer-1 activations are a spike at $-1$ and a spike at $+1$.
   Name the pathology, the effect on $\delta^{(1)}$, and two distinct causes.
6. Update-to-weight ratios are $\sim 10^{-6}$ in every layer; loss is almost
   flat. What is wrong, and what would $\sim 10^{-1}$ plus a NaN suggest
   instead?
7. Batchnorm: what axis is normalized, what is different at train vs eval,
   and why is $L$ invariant to $W \mapsto cW$ ($c>0$) just before a BN layer?
8. Layernorm: what axis is normalized instead? Give one reason transformers
   prefer it (preview is enough). Why might an MLP lose to your Capstone 1
   BDT on MAGIC, and why is that an acceptable mini-project outcome?

## Answers

1. Beam size (RMS envelope) at that stage. Systematic magnification compounds
   with depth: saturating activations clip and kill $g'$, or ReLU activations
   and gradients explode. Either way later (or earlier, backward) stages stop
   learning.
2. $z_j = \sum_{i=1}^{n_{\mathrm{in}}} W_{ji} x_i$; independent zero-mean
   $W, x$ $\Rightarrow$ $\mathrm{Var}(z) = n_{\mathrm{in}}\mathrm{Var}(W)
   \mathrm{Var}(x)$. Set $\mathrm{Var}(z)=\mathrm{Var}(x)$ $\Rightarrow$
   $\mathrm{Var}(W)=1/n_{\mathrm{in}}$. Tanh is approximately linear near 0
   ($g'(0)=1$), so $\mathrm{Var}(a)\approx\mathrm{Var}(z)$ in the regime the
   condition maintains.
3. Forward wants $1/n_{\mathrm{in}}$; backward wants $1/n_{\mathrm{out}}$. One
   matrix, two fans. Fan-avg splits the difference. Equality only when
   $n_{\mathrm{in}}=n_{\mathrm{out}}$.
4. Symmetric $z$ $\Rightarrow$ ReLU zeros half the mass $\Rightarrow$
   $\mathbb{E}[a^2]=\frac12\mathbb{E}[z^2]$. Restoring $\mathrm{Var}(a)\approx
   \mathrm{Var}(x)$ doubles $\mathrm{Var}(W)$ to $2/n_{\mathrm{in}}$. Xavier
   with tanh/sigmoid; He with ReLU.
5. Saturated tanh. $g'\approx 0$ $\Rightarrow$ $\delta^{(1)}$ (and earlier)
   vanish. Causes: too-large init ($\mathcal{N}(0,1)$ on a wide layer),
   unscaled inputs, or an LR that jumped $z$ onto the rails.
6. LR too low (or gradients already vanished): steps do not move $W$.
   $\sim 10^{-1}$ plus NaN: LR too high / exploding grads — one step rewrites
   the layer.
7. Over the batch, per feature. Train: batch $\mu,\sigma$ plus an EMA into
   running stats; eval: running stats, deterministic per example. $W\mapsto
   cW$ scales $z$, $\mu$, and $\sigma$ by $c$, so $\hat{z}$ (and the loss
   from it) is unchanged.
8. Over features, per example — no batch axis, no running stats. Transformers:
   token-wise stats, valid at $B=1$, independent of which sequences share a
   batch. MAGIC is 10 correlated tabular features, BDT home turf (Week 11);
   an understood loss under identical protocol is a finding, not a gate fail.

## New terms

- **activation / gradient statistics** — the mean, variance, and histogram
  shape of $a^{(l)}$ and $\delta^{(l)}$ (or $\nabla W^{(l)}$) at each layer;
  the quantities that say whether transport is stable.
- **vanishing / exploding gradients** — $\|\delta^{(l)}\|$ shrinking or
  growing exponentially with depth; saturating $g'$ or unmatched $\mathrm{Var}(W)$.
- **beam transport / envelope** — keeping a particle beam's RMS size matched
  stage to stage; the analog of keeping $\mathrm{Var}(a^{(l)})$ stable.
- **fan-in / fan-out** — number of inputs to a unit / number of units it
  feeds; $n_{\mathrm{in}}$ and $n_{\mathrm{out}}$ in the variance conditions.
- **Xavier (Glorot) init** — $\mathrm{Var}(W)=2/(n_{\mathrm{in}}+n_{\mathrm{out}})$
  (fan-avg), for tanh/sigmoid; forward-only form $1/n_{\mathrm{in}}$.
- **He (Kaiming) init** — $\mathrm{Var}(W)=2/n_{\mathrm{in}}$ for ReLU; the
  2 restores the half-variance ReLU drops.
- **update-to-weight ratio** — $\mathrm{std}(\text{update})/\mathrm{std}(W)$;
  typical healthy values $\sim 10^{-3}$.
- **dead ReLU** — a unit with $z<0$ on every example, hence zero gradient,
  hence stuck; more common at high LR.
- **saturated tanh** — activations piled at $\pm 1$, $g'\approx 0$.
- **gradient clipping** — rescaling $\nabla$ when $\|\nabla\|$ exceeds a cap;
  a seatbelt against explosions.
- **batchnorm** — standardize each feature over the mini-batch; running
  stats at eval; loss invariant to rescaling $W$ before BN.
- **running statistics** — EMA of per-batch mean/variance, used at eval so
  BN is a deterministic map.
- **layernorm** — standardize over features, per example; no batch coupling,
  no train/eval split of moments.
- **scale invariance (before BN)** — $W\mapsto cW$ ($c>0$) leaves BN's
  $\hat{z}$ unchanged, hence leaves the loss unchanged.

## Going deeper

- Karpathy, *Zero to Hero* — "Building makemore Part 3: Activations &
  Gradients, BatchNorm". The spine; do the histograms and the BN ablation
  along with him. E1–E6 are that video, turned into accepts.
- Glorot & Bengio, "Understanding the difficulty of training deep
  feedforward neural networks" (AISTATS 2010) — read for the variance
  argument (§2 of this lesson); skim the experiments.
- He et al., "Delving Deep into Rectifiers" (ICCV 2015) — the ReLU factor-of-2
  derivation, from the authors.
- Prince, *Understanding Deep Learning* (free PDF), Chapter 7 (Gradients and
  initialization); skim Chapter 9 (Regularization) for the BN/dropout
  neighborhood.
- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 8 (Optimization
  for training deep models) — LR schedules, clipping, and the optimizer zoo
  you already derived in Week 08, now in the deep-net setting.
