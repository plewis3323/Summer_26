# Bishop 1.1–1.3 — Q&A Log

*Running log of questions asked during the reading session and their answers.*

---

## Q1. What is a training set and a target vector? (§1.1)

**Training set** — the collection of example data the model learns from. In the curve-fitting
example it is $N$ observations of the input $x$:
$$\mathbf{x} = (x_1, \ldots, x_N)^\mathsf{T}.$$
Called "training" because the parameters $\mathbf{w}$ are *fit* on it. Each $x_n$ may itself be a
feature vector; in this example it's a single scalar.

**Target vector** — the stacked column of the desired outputs, one per training input:
$$\mathbf{t} = (t_1, \ldots, t_N)^\mathsf{T}.$$
$t_n$ is the answer paired with input $x_n$. In Bishop's example $t_n = \sin(2\pi x_n) + \epsilon_n$,
but the learner sees only the noisy $t_n$, not the generating sine.

**Relationship:** training adjusts $\mathbf{w}$ so predictions $y(x_n,\mathbf{w})$ match the targets
$t_n$; the goal is then to predict $\hat t$ for an unseen $\hat x$ (**generalization**).

*Terminology:* "target **vector**" = the $N$ scalar targets gathered into one $N$-dim column — distinct
from the (separate) idea of each input being multi-dimensional.

---

## Q2. What is generalization? (§1.1)

**Generalization** — a trained model's ability to predict accurately on **new, unseen inputs** not in
the training set. It is the real goal of ML; fitting the training data is only the means.

**Why central:** with $N=10$ noisy points we want $y(\hat x,\mathbf{w})$ correct for *any*
$\hat x\in[0,1]$, including the infinitely many never observed. Memorizing the training targets is
useless without interpolation to new $\hat x$.

**Why hard:** training data is finite and noisy. Enough flexibility (e.g. $M=9$ through all 10 points)
drives training error to zero but fits the **noise** → wild oscillation → poor predictions. That is
**over-fitting**: low training error, poor generalization.

**How measured:** not by training error (monotonically improves with complexity) but on held-out data —
a **test set** or **cross-validation** (§1.3). The train-vs-test error gap *is* the generalization gap.

**What helps:** more data, limiting complexity ($M$), or **regularization** (penalizing large
$\mathbf{w}$). Every tool in §1.1 targets generalization, not training fit.

---

## Q3. What is feature extraction? (chapter intro / §1.1)

**Feature extraction** — preprocessing that transforms raw inputs into a new representation (a set of
**features**) that makes learning easier. Part of the **preprocessing** stage.

**Why:**
- **Reduce variability / dimensionality** — raw inputs (e.g. all image pixels) are high-dimensional
  with much irrelevant variation; informative features expose patterns and speed computation.
- **Speed** — fewer features → faster training *and* prediction (matters for real-time use).
- **Inject prior knowledge** — design features to surface what matters and suppress what doesn't.

**Example (digit recognition):** instead of raw pixels, extract features like stroke count, aspect
ratio, or ink-density regions — quantities that differ between digit classes but stay roughly constant
across instances of the same digit.

**Invariance:** prefer features invariant to transformations that shouldn't change the answer
(translation, scaling) — a digit's identity is unchanged by shifting or resizing it.

**Caveat — information loss:** preprocessing can *discard* information; if a useful distinction is
thrown away, no later algorithm recovers it. Feature choice can cap achievable accuracy — a design
trade-off, not a free win.

*(added context: in §1.1 the "features" are the powers $1,x,x^2,\ldots,x^M$ — a fixed transform of
scalar $x$; generalized to basis functions $\phi_j(x)$ in Ch. 3.)*

---

## Q4. Clustering, density estimation, visualization? (chapter intro)

All three are **unsupervised learning** tasks — training data has inputs $\mathbf{x}$ but **no target
labels** $t$ (contrast supervised = classification/regression).

**Clustering** — discover groups of similar examples; partition inputs so within-cluster points are
more alike than across-cluster. Found purely from structure in $\mathbf{x}$. *E.g.* grouping customers
by behavior. (Bishop's later example: $K$-means.)

**Density estimation** — model the distribution $p(\mathbf{x})$ over input space: where data is dense
vs. sparse. *Use:* anomaly detection (low-density = outlier), generative modeling. Ties to §1.2.

**Visualization** — project high-dimensional data to 2-D/3-D for human inspection, preserving
meaningful (near/far) relationships. *E.g.* a 100-feature set → 2-D scatter. (PCA, Ch. 12 — added context.)

**Unifying idea:** extract structure from **unlabeled** data — *where are the groups* (clustering),
*where is the data* (density estimation), *what does it look like* (visualization).

**Full taxonomy:**

| Paradigm | Has targets? | Tasks |
|---|---|---|
| Supervised | yes | classification (discrete $t$), regression (continuous $t$) |
| Unsupervised | no | clustering, density estimation, visualization |
| Reinforcement | reward signal | learn actions to maximize reward |

---

## Q5. What is reinforcement learning? (chapter intro)

**Reinforcement learning (RL)** — finding suitable **actions** in a given situation to **maximize a
reward**. The learner is never told correct outputs; it must **discover** them by trial and error
through interaction.

**vs. supervised:** supervised gives the right answer per input; RL gives only a **reward** signal
(evaluative feedback), not the correct action (instructive feedback). vs. unsupervised: RL *has*
feedback (reward); unsupervised has none.

**Key features:**
- **Learning by interaction / trial and error** — try actions, observe reward, adjust; states and
  actions unfold over time.
- **Credit assignment** — rewards are often **delayed** and apply to a whole *sequence* of actions;
  must infer which earlier actions earned the outcome. (*Bishop's example:* a neural net learning
  backgammon by self-play — reward only at game end, yet every move must be evaluated.)
- **Exploration vs. exploitation** — *exploitation* uses known-good actions; *exploration* tries new
  ones that might be better. Too much of either is suboptimal; balancing them is fundamental.

Bishop notes RL is a large field the book does not cover in depth.

---

## Q6. What is regularization? (§1.1)

**Regularization** — controlling over-fitting by **adding a penalty to the error function** that
discourages large coefficients $\mathbf{w}$. Keep the flexible model, but penalize extreme solutions
(instead of shrinking the order $M$).

**Mechanism:** add a quadratic penalty to sum-of-squares error,
$$\widetilde{E}(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^{N}\big\{y(x_n,\mathbf{w}) - t_n\big\}^2
  + \frac{\lambda}{2}\lVert \mathbf{w}\rVert^2,\qquad
  \lVert\mathbf{w}\rVert^2 = \mathbf{w}^\mathsf{T}\mathbf{w} = \sum_{j} w_j^2.$$

**Why it works:** over-fitting shows up as **huge alternating-sign coefficients** that thread through
noisy points. The penalty makes large $\mathbf{w}$ expensive, trading a little training fit for smaller,
smoother coefficients → smoother curve → better generalization.

**The knob $\lambda$ (effective complexity):**
- $\lambda\to0$: penalty vanishes → over-fits.
- $\lambda$ moderate: smooth, good fit.
- $\lambda$ too large: coefficients crushed → under-fits.

It replaces the discrete knob $M$ with a continuous one.

**Names:** $L_2$ penalty = **ridge regression** (statistics) = **weight decay** (neural nets).

**Connection (§1.2):** minimizing $\widetilde{E}$ = **MAP** estimation with a Gaussian prior
$p(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I})$, where $\lambda=\alpha/\beta$.
The penalty *is* a prior that weights should be small.

---

## Q7. The ABCs of what the Bayesian interpretation means (§1.2.3)

**Two interpretations of probability:**
- **Frequentist:** probability = **long-run frequency** of a repeatable event ($p=0.5$ heads means
  half of infinitely many flips). Only meaningful for repeatable things.
- **Bayesian:** probability = **degree of belief / quantified uncertainty** — valid even for one-off,
  non-repeatable events ("70% chance of rain tomorrow"). This is the core idea.

**The ABCs of the procedure:**
- **A — Assign a prior** $p(\mathbf{w})$: belief about parameters *before* data.
- **B — Bring in data via the likelihood** $p(\mathcal{D}\mid\mathbf{w})$: how probable the observed
  data is for each $\mathbf{w}$ (a function of $\mathbf{w}$, not a distribution over it).
- **C — Combine with Bayes' theorem** to get the **posterior**:
$$p(\mathbf{w}\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\mathbf{w})\,p(\mathbf{w})}{p(\mathcal{D})},
  \qquad \text{posterior} \propto \text{likelihood}\times\text{prior}.$$

**The deep shift:**

| | Frequentist | Bayesian |
|---|---|---|
| What is $\mathbf{w}$? | fixed unknown constant | random variable with a distribution |
| Data $\mathcal{D}$ | one of many possible datasets | the single observed (fixed) dataset |
| Uncertainty in… | the estimator (error bars over repeats) | the parameters (the posterior) |
| Output | point estimate ($\mathbf{w}_\text{ML}$) | full distribution $p(\mathbf{w}\mid\mathcal{D})$ |

**Payoff:** a distribution over $\mathbf{w}$ expresses *how confident* you are → error bars, predictive
distributions, built-in over-fitting defense (prior = regularization, cf. Q6).

**Honest catch:** results depend on the **choice of prior** (criticized as subjective). Bayesians reply
that all assumptions should be explicit; as data grows the likelihood dominates and the prior's
influence fades.

---

## Q8. Why do we care about maximizing the log-likelihood? (§1.2)

**Why maximize the likelihood:** $p(\mathcal{D}\mid\mathbf{w})$ = how probable the *observed* data is
under parameters $\mathbf{w}$. **Maximum likelihood** picks the $\mathbf{w}$ making the observed data
**most probable** — the model that best explains what actually happened.

**Why the *log* (same optimum, since $\ln$ is monotonic):**
$$\arg\max_{\mathbf{w}} p(\mathcal{D}\mid\mathbf{w}) = \arg\max_{\mathbf{w}} \ln p(\mathcal{D}\mid\mathbf{w}).$$

1. **Products → sums (main reason).** i.i.d. data ⇒ likelihood is a product
   $p(\mathcal{D}\mid\mathbf{w})=\prod_{n} p(t_n\mid\mathbf{w})$; the log turns it into a sum
   $\sum_n \ln p(t_n\mid\mathbf{w})$, which is far easier to **differentiate** (and we differentiate to
   find the max).
2. **Numerical stability.** Each factor $\in(0,1)$; multiplying many **underflows** to 0 in
   floating-point. Summing logs stays in safe range — real code works in log-space.
3. **Connects to error functions.** Maximizing $\ln p$ = minimizing the **negative log-likelihood**,
   which *is* the error function. Under **Gaussian noise**, $-\ln p$ reduces to exactly the
   **sum-of-squares error** — so **least squares = ML under Gaussian noise** (the $\exp\{-\tfrac12(\cdot)^2\}$
   log-drops to the squared term).

**Takeaway:** same optimum as the likelihood, but converts an underflow-prone product into a stable,
differentiable sum that equals the familiar error function.

---

## Q9. What is over-fitting? (§1.1)

**Over-fitting** — a model fits the **training data too well, including its noise**, and so
**generalizes poorly** to new data. Low training error, high test error. The central failure mode of §1.1.

**Picture (fitting $\sin(2\pi x)$ from 10 noisy points):**
- **$M=9$**: 10 coefficients pass through all 10 points exactly ($E=0$) but **oscillate wildly** between
  them → useless predictions. *That's over-fitting.* (See `img_polyfit_overfitting.png`.)
- **$M=3$**: misses individual points but tracks the true curve → generalizes well.

**Why:** model too flexible relative to data → memorizes the *specific noise* instead of the underlying
pattern; noise won't repeat, so the fit doesn't transfer.

**Detect:**
- **Train/test error gap** — training error keeps falling as $M$ grows; test $E_\text{RMS}$ dips then
  shoots up. The gap *is* over-fitting (invisible from training error alone).
- **Coefficient blow-up** — fingerprint: huge alternating-sign coefficients ($\sim\pm10^5$ at $M=9$).

**Combat (all §1.1):** more data · reduce complexity (smaller $M$) · **regularization** (penalize large
$\mathbf{w}$, Q6).

**Deeper tension:** a higher-order polynomial contains all lower ones, so it's nominally more capable yet
generalizes worse — the fit-vs-generalization trade-off (bias–variance, Ch. 3 — added context) recurs
all book. Opposite failure = **under-fitting** ($M=0,1$): too rigid to capture the pattern.

---

## Q10. What is cross-validation, with an example? (§1.3)

**Cross-validation** — estimate generalization by rotating which portion of data is held out, so
**every point is used for both training and validation** (never simultaneously). Fixes the waste and
noisiness of a single fixed validation set.

**$S$-fold recipe:** (1) split data into $S$ equal folds; (2) for each fold $s$, train on the other
$S-1$ folds and score on fold $s$; (3) average the $S$ scores. Repeat per candidate model; pick the
best average. **LOO** = extreme $S=N$ (one point per fold). See `img_cross_validation.png`.

**Worked example — 20 points, choose $M$ via 5-fold (4 pts/fold):**

| Run | Train (16 pts) | Validate (4 pts) | Error |
|---|---|---|---|
| 1 | folds 2–5 | fold 1 | $e_1$ |
| 2 | folds 1,3,4,5 | fold 2 | $e_2$ |
| 3 | folds 1,2,4,5 | fold 3 | $e_3$ |
| 4 | folds 1,2,3,5 | fold 4 | $e_4$ |
| 5 | folds 1–4 | fold 5 | $e_5$ |

$\text{CV}(M)=\tfrac15\sum e_i$. Testing $M\in\{1,3,9\}$ might give CV errors $0.42$ (under-fit) /
$\mathbf{0.11}$ / $0.95$ (over-fit) → pick **$M=3$**, using all 20 points to decide.

**Why:** data-efficient (nothing permanently set aside — matters when data is scarce) and more reliable
(averages over $S$ splits, less luck-of-the-draw).

**Cost:** $S\times$ the training (LOO worst); with several hyperparameters the grid of runs grows
**exponentially** — motivates AIC/BIC and the Bayesian approach.
