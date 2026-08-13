# Week 28 — Scaling Laws & Interpretability

~4 hrs reading, plus the paper-and-pencil pass. Before starting you should be able
to: count the parameters of your Week 26/27 GPT by hand (Week 26); train it and
read a val-loss curve (Week 27); write a least-squares / power-law fit as a
regression and say what you minimize (Weeks 07–09); state KL and MLE (Week 08);
derive PCA two ways (Week 07); name posterior collapse in one sentence (Week 22);
write scaled dot-product attention and defend the $\sqrt{d_k}$ (Week 25).

Two questions this week, both forced on you by the model you just trained. First:
if you had $10\times$ the compute, should you make the network bigger or feed it
more tokens? That is a constrained-optimization problem with a measured loss
surface, and the answer has a closed form. Second: what is the fitted object
actually *computing* — a pile of correlations, or identifiable algorithms? The
honest present-day answer sits between two metaphors you will leave able to
explain: spectroscopy and phrenology.

This is also the first rehearsal of the Phase 3 gate derivation. Before the
weekend, re-derive scaled dot-product attention and the $\sqrt{d_k}$ argument
cold, on paper, no notes (Week 25). Anything you have to look up goes in the
month's open-question issue (syllabus §9).

## 1. Kaplan's scaling form, and why a physicist should be suspicious

Train language models of many sizes on many amounts of data, plot the
**cross-entropy loss** $L$ — the average $-\log p(\text{next token})$ you have
been minimizing since Week 15 — against model size, dataset size, or training
compute, and the curves are power laws over many orders of magnitude (Kaplan
et al. 2020). A **power law** is a relation $y \propto x^{-\alpha}$: a straight
line on a log–log plot. Physicists are trained to sit up when they see one.
Kepler's $T^2 \propto a^3$, the Stefan–Boltzmann $j \propto T^4$, a calorimeter's
stochastic resolution $\sigma_E/E \propto 1/\sqrt{E}$ (Week 24) — clean exponents
usually mean the system has no preferred scale in the window you are looking at.
They also mean you can *extrapolate*, which is the whole industrial interest:
predict the loss of a model you cannot yet afford.

The working formula, written in the three-term shape Hoffmann et al. later fitted
and that Kaplan's results made the field's default, is

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.$$

Name the pieces:

- $N$ — number of **non-embedding parameters** (the weights that do the
  computing; Kaplan found embeddings follow a noisier law, so they are left out
  of $N$).
- $D$ — number of **training tokens** (the integer ids your Week 27 tokenizer
  emits; one "example" is one token, not one abstract).
- $E$ — an **irreducible loss**, the entropy floor of the true next-token
  distribution. Even an infinite model on infinite data cannot beat the
  randomness that is actually in language. Analog: the irreducible noise term in
  Week 09's bias–variance decomposition.
- $A N^{-\alpha}$ — excess loss from a finite model. Smaller $N$, worse
  approximation to the true function.
- $B D^{-\beta}$ — excess loss from finite data. Smaller $D$, more overfitting /
  less coverage of the distribution.

$A, B, \alpha, \beta, E$ are fitted, not derived. Kaplan's original paper fitted
*separate* power laws $L(N)$, $L(D)$, $L(C)$ and read the allocation off those
fits; Hoffmann wrote the joint form above so the $N$ vs $D$ tradeoff is a single
calculus problem. We work with the joint form. The functional shape is the
claim; the exponents are measurements, and they moved when the measurement
protocol did (§3).

Two immediate consequences, both already in your toolkit:

- Fitting $L$ vs $N$ at fixed $D$ is ordinary regression (Week 09) on log–log
  axes. You minimize a squared error in $\log L$ (or a weighted least squares in
  $L$) because the interesting range spans decades; an unweighted fit in $L$ is
  dominated by the largest models. Connect to Week 07/08: least squares is MLE
  under a Gaussian noise model — so you should say what you assumed about the
  residuals of the scaling fit, not just quote $\alpha$.
- The three terms compete. At tiny $N$ the model term dominates; at tiny $D$ the
  data term does; at huge $N$ and $D$ you sit on $E$. A run that looks like "loss
  vs steps" on one model is a *slice* through this surface, not the surface.

## 2. Counting compute: $C \approx 6ND$

The scarce resource is not $N$ or $D$ separately. It is **training compute**
$C$, measured in **FLOPs** (floating-point operations: one multiply or one add
counts as one FLOP). GPUs are rented by the FLOP-second; a scaling law that
ignores $C$ is a scaling law you cannot use to spend money.

Count, for one token, through a network of $N$ parameters.

A linear layer $y = Wx$ with $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$
has $n_{\text{out}} n_{\text{in}}$ parameters. One forward pass does one
multiply-add per parameter: **2 FLOPs per parameter**. Summing over layers, the
forward pass of one token is $\approx 2N$ FLOPs. (Biases and layer-norms are
lower order; we are counting the matmuls.)

Backprop (Week 13) has to form both the gradient w.r.t. the *activations* (to
keep going backward) and the gradient w.r.t. the *weights* (to take the Adam
step, Week 08). Each is about one forward's worth of matmuls, so the backward
pass is $\approx 4N$ FLOPs. Total per token:

$$2N + 4N = 6N.$$

Times $D$ tokens:

$$C \approx 6ND.$$

This is Kaplan's accounting, and it is the constraint we will differentiate
under. Three caveats, so you use it honestly:

- **Attention's $T^2$ term is omitted.** Scaled dot-product attention (Week 25)
  costs $O(T^2 d_k)$ per layer for sequence length $T$, which is *not* linear in
  $N$. For the GPT-style widths and context lengths Kaplan and Hoffmann trained,
  it is a small fraction of the MLP matmuls; at long context it is not. Your
  Week 27 model is in the safe regime.
- **Optimizer state is not in $C$.** Adam stores two running moments per
  parameter (Week 08). That is memory, not FLOPs of the 6ND count.
- **$D$ is tokens processed, not unique tokens.** If you epoch twice, you have
  doubled $D$ in this formula. Whether that *helps* as much as new tokens is a
  different question — and one of the places Kaplan's fit went wrong (§3).

Worked numbers, so the formula is a tool:

- Your Week 27 model, $\sim 12$ M parameters, $\sim 2.5\times 10^5$ tokens:
  $C \approx 6 \times 1.2\times 10^7 \times 2.5\times 10^5 = 1.8\times 10^{13}$
  FLOPs (18 TFLOP). A laptop GPU does this without ceremony.
- GPT-3: $N = 1.75\times 10^{11}$, $D = 3\times 10^{11}$,
  $C \approx 6 \times 1.75\times 10^{11} \times 3\times 10^{11} = 3.15\times 10^{23}$
  FLOPs.
- Gopher 280 B on 300 B tokens: $C \approx 5\times 10^{23}$. Chinchilla used
  essentially that same budget on a 70 B model and 1.4 T tokens — four times
  smaller, five times more data. Same $C$, much lower loss. That is the plot
  this derivation exists to explain.

## 3. Chinchilla: the compute-optimal split

Fix a budget $C$ and choose $N$ and $D$ to minimize $L(N,D)$ subject to
$6ND = C$. This is the **compute-optimal** allocation: the $(N, D)$ pair that
gets you the lowest loss for the FLOPs you will actually spend. Kaplan's
separate-fit answer was "spend most new compute on $N$" ($N \propto C^{0.73}$,
$D \propto C^{0.27}$). That is why GPT-3 and Gopher are huge models trained on
too few tokens. Hoffmann et al. (Chinchilla, 2022) refit under a cleaner
protocol and got a different split. We now derive the split from the loss form,
two ways, and then plug in their exponents.

### 3.1 Substitution

Solve the constraint for $D = C/(6N)$ and substitute:

$$L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6N}\right)^{-\beta}
       = E + A N^{-\alpha} + B\, 6^{\beta}\, C^{-\beta}\, N^{\beta}.$$

Differentiate and set to zero. The $E$ vanishes.

$$\frac{dL}{dN} = -\alpha A N^{-\alpha-1} + \beta B\, 6^{\beta}\, C^{-\beta}\, N^{\beta-1} = 0.$$

Move terms:

$$\alpha A N^{-\alpha-1} = \beta B\, 6^{\beta}\, C^{-\beta}\, N^{\beta-1},$$

multiply through by $N^{\alpha+1}$:

$$\alpha A = \beta B\, 6^{\beta}\, C^{-\beta}\, N^{\alpha+\beta},$$

and solve for $N$:

$$N^{\ast} = \left(\frac{\alpha A}{\beta B\, 6^{\beta}}\right)^{1/(\alpha+\beta)}
             C^{\beta/(\alpha+\beta)}.$$

So

$$N^{\ast} \propto C^{\beta/(\alpha+\beta)}, \qquad
  D^{\ast} = \frac{C}{6N^{\ast}} \propto C^{\alpha/(\alpha+\beta)}.$$

The exponents add to 1, as they must: $N^{\ast} D^{\ast} \propto C$. Which way
the split tilts is a fight between $\alpha$ and $\beta$. If $\alpha > \beta$
(parameters help more, per doubling, than data), $D^{\ast}$ grows faster than
$N^{\ast}$ — you should buy tokens. If $\beta > \alpha$, you should buy
parameters.

### 3.2 Lagrange multiplier

Same stationarity, different language — the one you used to derive PCA
(Week 07: maximize $w^\top S w$ subject to $\|w\|=1$). Minimize $L(N,D)$
subject to $g(N,D) = ND - C/6 = 0$. The Lagrangian is
$\mathcal{L} = L + \lambda g$. Then $\nabla L = -\lambda \nabla g$:

$$\frac{\partial L}{\partial N} = -\alpha A N^{-\alpha-1} = -\lambda D, \qquad
  \frac{\partial L}{\partial D} = -\beta B D^{-\beta-1} = -\lambda N.$$

Divide the two equations (the $\lambda$ cancels):

$$\frac{\alpha A N^{-\alpha-1}}{D} = \frac{\beta B D^{-\beta-1}}{N}
  \quad\Rightarrow\quad
  \alpha A\, N^{-\alpha} = \beta B\, D^{-\beta}.$$

Read this: **at the optimum, the two excess-loss terms, each multiplied by its
exponent, are equal.** You have balanced the marginal return of a FLOP spent on
$N$ against a FLOP spent on $D$. Combine with $ND = C/6$ and you recover the
same $N^{\ast} \propto C^{\beta/(\alpha+\beta)}$. Do the Lagrange pass on paper
even if substitution felt easier; the equal-marginal-return form is what you
want in your head.

### 3.3 Plug in the fitted exponents

Hoffmann's Approach 3 (the parametric fit of exactly our $L(N,D)$) reports

$$\alpha \approx 0.34,\qquad \beta \approx 0.28$$

(with $E \approx 1.69$, $A \approx 406.4$, $B \approx 410.7$, $N$ and $D$ in raw
counts). Then

$$\frac{\beta}{\alpha+\beta} \approx \frac{0.28}{0.62} \approx 0.45, \qquad
  \frac{\alpha}{\alpha+\beta} \approx 0.55.$$

So $N^{\ast} \propto C^{0.45}$ and $D^{\ast} \propto C^{0.55}$: **almost equal
scaling**. Double the compute, roughly double the parameters *and* double the
tokens. Kaplan's $N \propto C^{0.73}$ was a substantial misallocation — too much
$N$, not enough $D$.

The ratio $D^{\ast}/N^{\ast}$ (tokens per parameter) is *not* fixed by the
exponents alone; the prefactors $A, B$ set its overall scale, and because
$\alpha \neq \beta$ the ratio still drifts slowly with $C$. Hoffmann's other two
approaches (IsoFLOP slices; a fit of the compute-optimal frontier directly)
found $a \approx b \approx 0.50$ and, at the compute they actually ran, a ratio
of about **20 tokens per parameter**. That is the number to remember, and it is
how Chinchilla was trained: 70 B parameters, 1.4 T tokens, $1.4\times 10^{12} /
7\times 10^{10} = 20$. Your Week 27 run (~12 M parameters, ~250 k tokens) is
$\sim 0.02$ tokens per parameter — wildly data-starved by this rule, which is
why it overfits and why the val-loss minimum, not the last checkpoint, is "the
model."

Why Kaplan's exponents differed, in one paragraph, because "the literature
disagreed" is not an explanation. Kaplan counted repeated epochs as new $D$
(the formula $C=6ND$ does, but the *loss benefit* of a second epoch is not the
same as new tokens), and their learning-rate schedule was a cosine that hit
near-zero at a predetermined $D$, so longer training was not a fair comparison
at fixed $N$. Hoffmann held the protocol fixed, varied $N$ and $D$ on purpose,
and the allocation moved. Same lesson as Week 24's identical-protocol rule:
if the baseline and the candidate were not trained under the same stopping and
counting conventions, you have compared recipes, not laws.

A physicist-shaped warning: these are *empirical* power laws on a particular
architecture family (dense transformers, English-ish web text, AdamW, a
particular LR schedule). They are not the Stefan–Boltzmann law. Changing the
data distribution, the tokenizer, or the architecture (Mixture-of-Experts,
state-space models) moves $A, B, \alpha, \beta$. Use them to plan a run of the
same kind; do not treat $N^{\ast} \propto C^{0.45}$ as a constant of nature.

## 4. Emergence: a cliff, or a thermometer that only reads integers?

Wei et al. (2022) claimed that some abilities are **emergent**: absent on small
models, present on large ones, and appearing as a sharp jump on a plot of
accuracy vs log-compute (or log-$N$), unpredictable by extrapolating the small
models. Three-digit arithmetic, some BIG-bench tasks, chain-of-thought that
suddenly starts helping — the figures look like phase transitions.

Schaeffer et al. (2023) argued that many of those cliffs are a **mirage**: the
*model's* competence improves smoothly, and a nonlinear metric turns the smooth
curve into a jump. The one-line argument is a derivation, and it is this week's
second paper pass.

Let $p$ be the probability that a single generated token is correct. Suppose
$p$ rises smoothly with scale — say $p(C)$ is a saturating function of
$\log C$, exactly the kind of curve Kaplan's $L$ already predicts, since
per-token accuracy is a smooth function of cross-entropy for a well-calibrated
model. Now score the model on **exact match of $k$ tokens**: the whole answer
is right, or it is not. If token errors are independent at zeroth order,

$$P(\text{exact}) = p^{k}.$$

That is the same arithmetic as a $k$-fold coincidence trigger: $k$ independent
detectors each firing with efficiency $p$, and you require all $k$. For $k=1$
you recover $p$. For $k=10$,

$$0.5^{10} \approx 0.001, \qquad 0.9^{10} \approx 0.35, \qquad 0.99^{10} \approx 0.90.$$

A smooth walk of $p$ from $0.5$ to $0.9$ — perfectly unsurprising on a
per-token plot — becomes a walk of exact-match from $0.1\%$ to $35\%$, which on
a linear $y$-axis against $\log C$ is a cliff. Increase $k$ (longer answers,
stricter exact match) and the cliff gets later and sharper. Change the metric
to something continuous — log-likelihood of the correct answer, token-level
edit distance, a partial-credit score — and the same models, same checkpoints,
show a smooth curve. That is the mirage: **the discontinuity was in the
$y$-axis, not in the network.**

Steelmanning Wei, because the paper is not a mistake. First, exact match is
often the *task*. A fitter that gets nine of ten digits of the $J/\psi$ mass
right has not found the $J/\psi$. If users need the whole answer, $p^k$ is the
relevant operating characteristic, and "the model suddenly became able to do
this" is a true statement about the product even if $p$ was smooth. Second,
Wei's operational definition was "unpredictable from smaller models." If you
only plotted exact-match vs log-$N$ for the models you could then afford, you
really could not have predicted GPT-4's BIG-bench jumps by linear
extrapolation. That is a fact about the published figures. Third, some
qualitative changes are *not* metric artifacts: induction heads turning on
(§5), grokking, the sudden usefulness of chain-of-thought. Those need a
mechanistic story, which an accuracy plot cannot provide — but they are not
nothing.

Steelmanning Schaeffer: a claim of a new kind of scaling — a phase transition
in competence — is a claim about the *system*, not about your thermometer. If
switching from exact-match to log-likelihood removes the jump, you have learned
that exact-match has a threshold, which we already knew from $p^k$. Publishing
the thresholded plot as evidence of emergent *abilities* is the same error as
declaring a phase transition because your ADC saturates. The scientific
standard is: show the continuous metric, show the thresholded one, and do not
call the difference between them a discovery about intelligence.

**Position, not a fence.** Schaeffer wins the scientific claim. The Wei figures
do not establish discontinuous competence; they establish that nonlinear
metrics have thresholds, and $P(\text{exact})=p^k$ is the whole mechanism for a
large fraction of the famous plots. Wei wins an engineering claim that is
real and narrower: *usefulness* can appear suddenly because the user's metric
is thresholded, and you should expect that. Confusing those two is how the
debate got loud. For this course the rule is the physicist's one. A phase
transition is a claim about the system (an order parameter, a circuit that
appears, a loss-surface qualitative change). A cliff on a nonlinear metric is
a claim about the metric. Plot both. Call only the first one emergence. The
exercises make you produce the $p^k$ overlay on your own model's per-token
probabilities, so this is not a spectator opinion.

## 5. Induction heads: a circuit you can hunt

If emergence-as-plot is mostly a mirage, what would count as a real qualitative
change inside the network? A **circuit**: a small set of heads and MLPs, with a
stated algorithm, that you can *break* (ablate) and watch the behavior die.
The cleanest example in transformers is the **induction head**.

In-context learning, in the stripped-down form that makes the circuit visible,
is pattern completion on a sequence the model has never trained on:

$$[\ldots]\; A\; B\; [\ldots]\; A \;\rightarrow\; B.$$

$A$ and $B$ are tokens. The model sees $A$ then $B$ once, later sees $A$ again,
and predicts $B$. That is a copy, gated on a prefix match. Olsson et al.
(transformer-circuits.pub, *In-context Learning and Induction Heads*) showed
that a two-head algorithm implements it:

1. **Prefix matching.** A previous-token head (often in an earlier layer)
   writes into the residual stream, at token $B$, information about "the token
   that just preceded me was $A$." Then, when the *second* $A$ arrives, an
   induction head attends back to that $B$ — because $B$'s key says "I was
   preceded by $A$" and the query of the second $A$ says "I am $A$, looking for
   whoever last followed me."
2. **Copying.** The same head's **output-value (OV) circuit** — the composition
  of $W_V$ and $W_O$ for that head (Week 25's value projection, then the
  output mix) — copies the attended token's identity into the residual, so the
  unembedding predicts $B$.

Two heads, two jobs, composed across layers via the residual stream (Week 26).
A **residual stream** is the running sum of all block outputs that every layer
reads from and writes to; it is the transformer's blackboard. The induction
head does not store the pair $(A,B)$ in weights. It *looks up* the previous
occurrence in the current context. That is why it works on random tokens the
model has never seen: the algorithm is content-independent.

### Hunting them in your Week 27 model

You do not need a 70 B model. The diagnostic is a synthetic sequence of
*random* tokens with a planted repeat, fed to a model you already have.

1. Draw a random token string $x_1,\ldots,x_T$ from the vocabulary, copy a
   prefix, and concatenate: $[x_1,\ldots,x_n,\; x_1,\ldots,x_n]$. The second
   half is a pure induction task.
2. For each layer $\ell$ and head $h$, at each position $t$ in the second
   half, read the attention weight onto the position that *should* be the
   induction source (the token after the previous occurrence of $x_t$). Average.
   That is the **prefix-matching score**.
3. For the same head, estimate whether the OV circuit copies token identity:
   a standard proxy is the degree to which the head's output, at the matched
   position, raises the logit of the copied token. That is the **copying
   score**.
4. Plot a heatmap over $(\ell, h)$. A candidate induction head is high on
   *both* scores.

Your Week 27 model is small (~12 M, few layers, trained on a tiny physics
corpus). Induction heads in the Olsson sense often appear around a few to a
few tens of millions of parameters on natural-language data, after enough
tokens — which you do not have. **Absence is an acceptable result** if the
heatmap is real and the argument is "this $N$ and $D$ are below the scale
where the circuit is known to form," not "I didn't look carefully." A head
that prefix-matches but does not copy (or the reverse) is also a finding:
half the algorithm, not the circuit. Record the scores, not a vibes-based
"this head looks like it's attending to earlier tokens."

## 6. Superposition: $n$ features in $m < n$ dimensions

A **feature**, in this literature, is a direction in activation space that
corresponds to a human-nameable (or at least independently varying) property:
"this token is a verb," "the shower is wide," "the previous token was $A$."
If the network has $m$ hidden dimensions and $n > m$ features worth
representing, it cannot give each feature an orthogonal dedicated axis. The
**superposition hypothesis** is that it represents them anyway, as
*nearly* orthogonal directions, and relies on **sparsity** — most features
being off on any given input — so that interference is rare.

This is not a metaphor. Anthropic's *Toy Models of Superposition* is a model
you can train in a page.

Setup. There are $n$ features, with values $x \in \mathbb{R}^n_{\ge 0}$. Each
coordinate is usually zero and occasionally on (a Bernoulli-gated magnitude, or
a sparse ReLU of a Gaussian — the exact sparse law is not the point). An
encoder packs them into $m < n$ dimensions,

$$h = Wx, \qquad W \in \mathbb{R}^{m \times n},$$

and a **ReLU readout** tries to recover them:

$$\hat{x} = \operatorname{ReLU}(W^\top h + b) = \operatorname{ReLU}(W^\top W x + b).$$

Train $W$ and $b$ to minimize reconstruction error $\mathbb{E}\|x - \hat{x}\|^2$,
optionally with a per-feature importance weight. The matrix $W^\top W$ is the
object to plot: diagonal entries are "how much of its own dimension this
feature got"; off-diagonals are interference between features.

Two phases, swept by sparsity:

- **Dense features** (often on, they collide). The model cannot afford
  interference, so it dedicates dimensions: $m$ features get almost-orthogonal
  columns of $W$, the rest are ignored. $W^\top W$ looks like a partial identity.
- **Sparse features** (rarely on, rarely co-occur). Interference is rare, so
  the model packs $n > m$ columns into $\mathbb{R}^m$ at mutual angles. ReLU
  plus a negative bias can hide small interference: a feature that is off
  reconstructs to a slightly negative pre-activation, and ReLU zeros it.
  $W^\top W$ shows many medium-sized diagonals and a spray of off-diagonals.

That is superposition: **compression of sparse features**, not compression of
variance.

### Unlike PCA (Week 07), unlike posterior collapse (Week 22)

**PCA** (Week 07) finds the $m$ directions of highest *variance* and throws
the rest away. It is optimal $L^2$ compression for dense, Gaussian-ish data;
you derived it two ways (max $w^\top S w$, min reconstruction error) and they
agreed. A rare-but-decisive feature — a one-in-a-thousand detector-fault flag,
an induction-trigger token — has tiny variance and PCA will discard it.
Superposition does the opposite bet: keep the rare features, pack them into
leftover angles, and pay for it when two packed features fire at once
(interference, not a missing component). Same $m < n$ arithmetic, different
loss, different survivors. PCA compresses what is *common*; superposition
compresses what is *sparse*.

**Posterior collapse** (Week 22) is the other failure of a latent code. The
VAE encoder is supposed to pack information about $x$ into $z$; if the decoder
learns to ignore $z$, the KL term goes to zero and *nothing* is packed in.
Superposition is too much packed in: more features than dimensions, recovered
only because they do not co-occur. Collapse is an empty suitcase. Superposition
is a suitcase with too many clothes, folded so they almost do not wrinkle,
until you pack two that overlap. When you diagnose a representation, say which
way it failed.

## 7. What interpretability can and cannot establish

Two metaphors, both earned.

**Spectroscopy.** Heat a gas, disperse the light, and you see emission lines at
definite wavelengths. Hydrogen has a line at $656\,\text{nm}$ (H-$\alpha$)
because an electron dropping from $n=3$ to $n=2$ in a Coulomb potential
releases that photon. Matching an observed line to a known transition is a
*causal* identification: you have a mechanism (energy levels) and a
prediction (a wavelength), and an intervention (change the element, the line
moves). Finding an induction head is supposed to be this. You state an
algorithm (prefix-match + copy), you predict a pattern of attention weights
and OV behavior, you ablate the head and the copy dies, you restore it and the
copy returns. That is a spectral line with a level diagram.

**Phrenology.** Nineteenth-century claim: the shape of the skull maps onto
mental faculties — a bump here means "destructiveness," a hollow there means
"the absence of wit." It was correlation without a mechanism, did not
replicate, and was used to launder prejudice as anatomy. The ML analog is
**probe-and-declare**: train a linear classifier on a layer's activations to
predict "this sentence is about calorimeters," get a high AUC, and announce
that the layer *represents* calorimeters. A probe can succeed because the
information is there *and* because it is linearly mixed with a dozen other
things the probe is accidentally using. Without an intervention (zero the
direction, does the behavior die? patch it in, does the behavior appear?),
you have a bump on a skull.

What current interpretability **can** establish, when it is done as
spectroscopy:

- *Existence* of a circuit that implements a stated algorithm on a stated
  distribution (induction on random tokens; a syntax head on English).
- A *causal role* for that circuit, via ablation, activation patching, or
  path patching — the intervention is the experiment.
- A *lower bound* on what the model computes: it at least does this.

What it **cannot** establish, even when the circuit is real:

- **Completeness.** Finding H-$\alpha$ tells you hydrogen is in the star. It
  does not give you the star's full composition, and a missing line does not
  prove the element is absent (wrong temperature, wrong ionization, line
  buried in continuum). A found induction head does not mean in-context
  learning is "solved" or that no other copy circuit exists.
- **Identity with a human concept.** A direction that correlates with "is a
  verb" is not *the* verb feature; it is a direction that was useful for
  whatever the loss asked. Superposition makes this worse: the same neuron
  participates in many features.
- **Safety or absence.** Not finding a bad circuit is not evidence the
  behavior is absent. Phrenology-by-omission is still phrenology.
- **Distribution-free claims.** Circuits are measured on a prompt
  distribution. Week 09's generalization gap applies inside the model: the
  induction head you found on random tokens may not be the mechanism on
  physics abstracts.

The field is young, the toy models are real, and the gap between "we found a
copy head in a 2-layer attention-only transformer" and "we understand GPT-4"
is the gap between identifying one spectral line and claiming you have a
theory of stellar structure. Work on the line-identification side. Do not
write the other claim.

## Check yourself

1. In $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$, what does $E$ represent, and which
   Week 09 term is it analogous to?
2. Derive $C \approx 6ND$ in three lines (forward FLOPs per token, backward,
   times $D$). Name one effect this count ignores.
3. From $L(N,D)$ and $C=6ND$, derive $N^{\ast} \propto C^{\beta/(\alpha+\beta)}$
   by substitution *or* Lagrange (one is enough here; you will do both on
   paper). Then plug in $\alpha=0.34$, $\beta=0.28$ and state the scaling of
   $N^{\ast}$ and $D^{\ast}$.
4. Why is "20 tokens per parameter" not a pure consequence of those two
   exponents, and why is it still the right number to remember?
5. A model's per-token accuracy goes smoothly from $0.6$ to $0.95$. What
   happens to exact-match on $k=8$ tokens, and which paper does this support?
   State this week's position in one sentence.
6. An induction head does two jobs. Name them, and say why a hunt on *random*
   repeated tokens is the right diagnostic (as opposed to real abstracts).
7. Superposition vs PCA vs posterior collapse: one sentence each on what is
   being compressed or failed, and in which direction.
8. You train a linear probe on layer 4 that predicts "abstract is about
   elliptic flow" at AUC 0.92. Why is that not spectroscopy? What additional
   result would make it closer?

## Answers

1. $E$ is the entropy floor of the true next-token distribution — loss you
   cannot beat with any finite $N, D$. Analogous to Week 09's irreducible
   noise in the bias–variance decomposition.
2. Forward: $\sim 2N$ FLOPs/token (multiply-add per parameter). Backward:
   $\sim 2\times$ that, for activation gradients and weight gradients, so
   $4N$. Total $6N$ per token, times $D$ tokens. Ignores attention's $T^2$
   cost (and treats unique tokens and epochs as the same $D$).
3. Substitute $D=C/(6N)$ into $L$, set $dL/dN=0$, collect powers of $N$:
   $N^{\alpha+\beta} \propto C^{\beta}$, hence
   $N^{\ast} \propto C^{\beta/(\alpha+\beta)}$ and
   $D^{\ast} \propto C^{\alpha/(\alpha+\beta)}$. With the Chinchilla exponents,
   $N^{\ast} \propto C^{0.45}$, $D^{\ast} \propto C^{0.55}$ — nearly equal
   scaling.
4. The exponents fix how the *split* changes with $C$; the prefactors $A,B$
   (and the IsoFLOP measurements) fix the ratio $D/N$ at a given scale.
   Because $\alpha \approx \beta$ the ratio is only weakly $C$-dependent, and
   Hoffmann's frontier sits at $\sim 20$ tokens/parameter in the compute
   window they ran. Remember 20; do not pretend you derived the 20 from
   $0.34$ and $0.28$ alone.
5. $P(\text{exact})=p^{k}$ goes from $0.6^{8} \approx 0.017$ to
   $0.95^{8} \approx 0.66$ — a dramatic jump on a linear axis. Supports
   Schaeffer: the cliff is the metric. Position: do not call a $p^{k}$
   threshold "emergent competence"; reserve emergence for system-level
   qualitative changes (circuits, loss-surface phases) and always plot the
   continuous metric too.
6. Prefix matching (attend to the token that followed the previous
   occurrence of the current token) and copying (OV circuit writes that
   token's identity). Random tokens have no corpus statistics to cheat with,
   so success cannot be "the model memorized that 'beam' follows 'energy'";
   it has to be the algorithm.
7. PCA: compress dense variance into $m$ principal axes, discard low-variance
   directions. Superposition: pack $n>m$ *sparse* features into $m$ dims at
   angles, accept interference on collisions. Posterior collapse: pack
   *too little* — the latent is unused, KL $\to 0$.
8. A probe is a correlation (phrenology): the label is linearly readable,
   which does not say the model *uses* that direction for that concept.
   Closer to spectroscopy: ablate or patch the direction and show the
   elliptic-flow behavior dies or appears, with a stated mechanism.

## New terms

- **scaling law** — empirical power-law fit of loss (or another metric) to
  $N$, $D$, or $C$; used to extrapolate, not derived from first principles.
- **irreducible loss $E$** — entropy floor of the true data distribution.
- **FLOP / compute $C$** — count of arithmetic operations; $C \approx 6ND$
  for transformer training under Kaplan's accounting.
- **compute-optimal $(N^{\ast}, D^{\ast})$** — the split of a fixed $C$ that
  minimizes $L(N,D)$; Chinchilla's answer is $N$ and $D$ scaled together,
  $\sim 20$ tokens per parameter.
- **emergence (Wei)** — ability absent at small scale, present at large,
  appearing as a sharp jump, claimed unpredictable from smaller models.
- **mirage (Schaeffer)** — a cliff produced by a nonlinear metric (e.g.
  exact match) on top of a smooth per-token competence curve.
- **$P(\text{exact})=p^{k}$** — independent-token exact-match; the one-line
  mirage mechanism.
- **circuit** — a small, named set of heads/MLPs implementing a stated
  algorithm, tested by intervention.
- **induction head** — attention head that prefix-matches a previous
  occurrence and copies the token that followed it.
- **prefix-matching / copying scores** — per-head diagnostics for the two
  halves of induction, measured on repeated random tokens.
- **residual stream** — the transformer's running residual sum; the blackboard
  heads read from and write to.
- **OV circuit** — the $W_V$–$W_O$ map of one head; what it *writes* once it
  has attended.
- **feature (interp)** — a direction in activation space corresponding to an
  independently varying property.
- **superposition** — representing $n>m$ (usually sparse) features in $m$
  dimensions as nearly-orthogonal directions, with ReLU hiding interference.
- **spectroscopy vs phrenology** — causal, mechanism-plus-intervention
  identification vs correlational bump-reading; the standard for what interp
  has actually shown.

## Going deeper

- Kaplan et al., *Scaling Laws for Neural Language Models* (arXiv 2001.08361),
  §§1–3 and the $L(N)$, $L(D)$, $L(C)$ figures — the empirical discovery; note
  how they count $D$ and what they conclude about the $N$ vs $D$ split, so you
  can see exactly what Chinchilla revised.
- Hoffmann et al., *Training Compute-Optimal Large Language Models*
  (Chinchilla, arXiv 2203.15556) — Approaches 1–3 and the table that compares
  them; recover $N^{\ast} \propto C^{\beta/(\alpha+\beta)}$ from their
  Appendix before looking. The 70 B / 1.4 T paragraph is the number you
  memorize.
- Wei et al., *Emergent Abilities of Large Language Models* (arXiv 2206.07682)
  — read the definition and a handful of the cliff figures *first*, and write
  one sentence on what would falsify the claim, before opening Schaeffer.
- Schaeffer et al., *Are Emergent Abilities of Large Language Models a
  Mirage?* (arXiv 2304.15004) — the metric-choice argument; match it to your
  $p^{k}$ derivation. Then write the half-page position in `notes.md` the
  README asks for.
- Elhage / Olsson et al., *A Mathematical Framework for Transformer Circuits*
  (transformer-circuits.pub) — skim for residual stream, QK/OV decomposition;
  then *In-context Learning and Induction Heads* (main read) and *Toy Models
  of Superposition* (intro + first experiments). Hunt and toy-model before
  you read their plots, so you have a prediction to be wrong about.
