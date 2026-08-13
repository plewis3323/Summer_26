# Week 08 — Probability, Information, and the Optimizer Zoo

~3 hrs reading, computing along as you go — denser than Weeks 05–07; take the slack
if the project spills a day. Before starting you should be able to: write Python
functions and loops (Week 02); build NumPy arrays and use `@` (Week 03); take a
gradient and run GD, including on $f(x,y)=x^2+10y^2$ (Week 05); multiply matrices,
project, and solve least squares (Week 06); form $A^\top A$, read a condition
number, and differentiate $\|Ax-b\|^2$ (Week 07).

Two threads, then they fuse. **Part A** is probability: least squares from a
Gaussian likelihood, ridge from a Gaussian prior, and the cross-entropy / KL /
likelihood chain that Weeks 10, 13, 22, and 31 quote. **Part B** is the optimizer
zoo: the step-size bound on a quadratic, the name of Week 05's zig-zag villain,
and SGD, momentum, RMSProp, and Adam. The *race* itself — `src/optim.py` on
surfaces designed to hurt — is the project (`project.md`). This lesson teaches
the ideas.

**Part A — Probability**

## 1. Random variables

A **random variable** is a number whose value is not fixed in advance — it is the
outcome of a repeatable chance process. Roll a die: the face $X$ is a random
variable. Measure a pulse height: the recorded $Y$ is a random variable (the true
energy is fixed; the electronics and the physics of energy loss jitter). Write
$X$ with a capital letter for the random thing, $x$ for a particular value it took.

### 1.1 Discrete vs continuous; pmf and pdf

**Discrete.** $X$ takes values in a list (faces 1–6; a count of hits; a class
label $0$ or $1$). Its **probability mass function (pmf)** is $p(x) = P(X = x)$,
a non-negative number per value, summing to 1: $\sum_x p(x) = 1$.

**Continuous.** $X$ takes values on the real line (an energy, a weight, a
reconstructed mass). Then $P(X = x) = 0$ for any exact $x$ — the interesting
question is $P(a \le X \le b)$. That probability is the area under a curve $p(x)$
called the **probability density function (pdf)**:

$$P(a \le X \le b) = \int_a^b p(x)\,dx, \qquad \int_{-\infty}^{\infty} p(x)\,dx = 1.$$

A pdf can exceed 1 (a spike of width $0.1$ and height $10$ still has area 1); it is
not a probability. The probability of a tiny window $dx$ is $p(x)\,dx$. Sampling
in NumPy is from named families (`rng.normal`, `rng.poisson`); a histogram of
draws *is* the pdf, up to bin width. Snippets below assume

```python
import numpy as np
rng = np.random.default_rng(0)
```

already ran.

### 1.2 Expectation, variance, covariance

The **expectation** (mean) of a discrete $X$ is the probability-weighted average
of its values:

$$\mathbb{E}[X] = \sum_x x\, p(x).$$

For a continuous $X$, replace the sum by an integral: $\mathbb{E}[X] = \int x\, p(x)\,dx$.
For any function $g$, $\mathbb{E}[g(X)] = \sum_x g(x)\,p(x)$ (or $\int$). Expectation
is linear always, independence not required:

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y].$$

The **variance** measures spread: $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
= \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. It scales as $\mathrm{Var}(aX) = a^2\mathrm{Var}(X)$.
The **standard deviation** is $\sigma = \sqrt{\mathrm{Var}(X)}$, in the same units as $X$.

The **covariance** is $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$;
the **correlation** $\rho=\mathrm{Cov}(X,Y)/(\sigma_X\sigma_Y)$ rescales it to
$[-1,1]$. (Equal variances: $\rho=\mathrm{Cov}/\sigma^2$ — Week 11's formula.)
Expanding a sum:

$$\mathrm{Var}\!\left(\sum_i a_i X_i\right)
 = \sum_i a_i^2\,\mathrm{Var}(X_i) + \sum_{i \ne j} a_i a_j\,\mathrm{Cov}(X_i, X_j).$$

If the $X_i$ are **uncorrelated** (covariances zero) the cross terms drop, and
$\mathrm{Var}(aX+bY)=a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y)$ — the lab's error
propagation, now a theorem. Independence implies uncorrelated, not conversely.

The **$q$-th quantile** of $X$ is the value below which a fraction $q$ of the mass
lies: $P(X\le x_q)=q$. The $0.5$-quantile is the **median**.

### 1.3 Joint, marginal, conditional; the product rule

Two random variables have a **joint** distribution $p(x,y) = P(X=x, Y=y)$ (discrete;
continuous: a joint density, area now a volume). The **marginal** of $X$ throws $Y$
away by summing (or integrating) it out:

$$p(x) = \sum_y p(x,y).$$

The **conditional** $p(x \mid y)$ ("$x$ given $y$") is the joint, renormalized on
the slice of fixed $y$:

$$p(x \mid y) = \frac{p(x,y)}{p(y)} \qquad\text{whenever } p(y) > 0.$$

Rearrange: the **product rule**

$$p(x,y) = p(x \mid y)\, p(y) = p(y \mid x)\, p(x).$$

Any joint of many variables factorizes the same way, in any order — the **chain
rule of probability**:

$$p(x_1, x_2, \dots, x_n) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2)\cdots p(x_n \mid x_1,\dots,x_{n-1}).$$

Week 43 writes a trajectory probability this way. **Independence** is
$p(x,y)=p(x)p(y)$ (equivalently $p(x\mid y)=p(x)$): knowing $Y$ tells you nothing
about $X$. Then $\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$ and the covariance
is zero.

The **tower property** (law of total expectation):
$\mathbb{E}[X]=\mathbb{E}\big[\mathbb{E}[X\mid Y]\big]$ — the overall mean is the
mean of the slice-means. Conditioning, then averaging the condition back out,
recovers $\mathbb{E}[X]$. Week 43 uses this to drop past rewards from a gradient.

### 1.4 Bayes' theorem, from the product rule

The two writings of the product rule are $p(y \mid x)\,p(x) = p(x \mid y)\,p(y)$.
Divide by $p(x)$:

$$\boxed{\,p(y \mid x) = \frac{p(x \mid y)\, p(y)}{p(x)}\,}$$

— **Bayes' theorem**. Four names, used all year: the **prior** $p(y)$ (belief
before seeing $x$); the **likelihood** $p(x\mid y)$ (how probable the observation
is if $y$ were true — a function of $y$, for fixed data); the **posterior**
$p(y\mid x)$ (belief after seeing $x$); the **evidence**
$p(x)=\sum_y p(x\mid y)p(y)$ (the denominator that makes the posterior sum to 1;
Week 22 names it in the ELBO). Bayes is the product rule, solved for the other
conditional.

### 1.5 Worked example: a particle-ID tagger

**Particle identification (PID)** decides which *species* of particle left a
detector signature — electron, muon, pion, proton. A **tagger** flags "this looks
like an electron." A **track** is a reconstructed path of a charged particle
through the detector. Two conditionals characterize a tagger:

- **Efficiency** $\varepsilon=P(\text{tagged}\mid\text{true electron})$: of the
  real electrons, what fraction the tagger catches.
- **Fake rate** $f=P(\text{tagged}\mid\text{not an electron})$: of the *other*
  particles, what fraction it wrongly calls electron.

What you want after it fires is $P(\text{true electron}\mid\text{tagged})$ — the
**purity**. That is a posterior. Bayes needs the **prior abundance**: how common
true electrons are *before* looking at the tagger.

At a hadron collider, **pions** (the lightest **hadrons** — particles made of
quarks) are produced copiously; **electrons** (fundamental charged leptons) are
rare. Suppose, in a sample of tracks, $1\%$ are electrons and $99\%$ are pions,
and your tagger has $\varepsilon = 0.90$, $f = 0.01$:

```python
prior_e = 0.01
prior_pi = 0.99
eff = 0.90          # P(tagged | electron)
fake = 0.01         # P(tagged | pion)

p_tagged = eff * prior_e + fake * prior_pi          # evidence
p_true_given_tagged = (eff * prior_e) / p_tagged
print(p_tagged, p_true_given_tagged)                # 0.0189  0.476...
```

A $90\%$-efficient, $1\%$-fake tagger yields a tagged set that is only $\approx 48\%$
electrons. Count it: $1000$ tracks hold $10$ electrons ($\approx 9$ tagged) and
$990$ pions ($\approx 10$ fake-tags) — purity is $9/(9+10)$. The prior ate the
tagger. The same tagger on a $50\%$ electron sample gives purity $\approx 0.99$.
The tagger did not improve; the prior did. **Always write the prior down.**

## 2. Common distributions

A **distribution family** is a pdf/pmf with a few adjustable numbers, the
**parameters**. Fitting a family to data is most of statistics; the rest of Part A
is how.

### 2.1 Bernoulli and Beta

A **Bernoulli** random variable is a single coin flip: $X \in \{0,1\}$ with
$P(X=1) = p$ and $P(X=0) = 1-p$. Compactly, $P(X=x) = p^x (1-p)^{1-x}$. This is
the model for a binary label (Week 10: signal vs background) and for a single
yes/no tag. $n$ independent Bernoullis with the same $p$ make a **Binomial** count
of successes.

The **Beta** family is a pdf on $p \in (0,1)$ itself, with two positive parameters
$\alpha, \beta$:

$$p(p; \alpha, \beta) \propto p^{\alpha-1}(1-p)^{\beta-1}.$$

It is the natural **prior** on a Bernoulli's unknown $p$. After $s$ successes and
$f$ failures the posterior is Beta$(\alpha+s,\,\beta+f)$ — add the counts to the
prior's pseudo-counts. (A family that updates this cleanly is **conjugate** to the
likelihood. Gaussian-with-Gaussian in §5 is the same pattern.) The MLE of $p$ is
the success fraction; a flat Beta$(1,1)$ prior pulls the posterior mean
$(1+s)/(2+n)$ slightly toward $1/2$.

### 2.2 The 1D Gaussian

The **Gaussian** (normal) density with mean $\mu$ and variance $\sigma^2$ is

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),$$

written $X \sim \mathcal{N}(\mu, \sigma^2)$. Bell-shaped, symmetric about $\mu$,
width set by $\sigma$. The **central limit theorem** says a sum of many small
independent jitters is approximately Gaussian, whatever the jitters' own shapes —
hence measurement noise, sample means, thermal fluctuations.

Two lemmas. (1) $aX+b\sim\mathcal{N}(a\mu+b,\,a^2\sigma^2)$. (2) The sum of
independent Gaussians is Gaussian, means adding and variances adding as in §1.2.
Week 35 reuses both.

### 2.3 The multivariate Gaussian, and Week 07's ellipses

In $k$ dimensions the Gaussian is specified by a mean vector $\mu \in \mathbb{R}^k$
and a $k \times k$ **covariance matrix** $\Sigma$, symmetric and positive definite:

$$p(x) = (2\pi)^{-k/2}\,(\det\Sigma)^{-1/2}
\exp\!\left(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right).$$

The quadratic form $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ is squared distance from $\mu$
after **whitening** (stretch each principal axis to unit variance). Level sets are
ellipsoids — exactly Week 07's PCA picture of $\Sigma$: eigenvectors are the
principal axes, eigenvalues the variances along them. A diagonal $\Sigma$ with
unequal entries is an axis-aligned ellipse; off-diagonal entries *are* the
covariances and rotate it. PCA diagonalizes them. Independent coordinates:
$\Sigma$ diagonal, joint density a product of 1D Gaussians (Week 22's encoder
output).

```python
Sigma = np.array([[2.0, 1.2], [1.2, 1.0]])
evals, evecs = np.linalg.eigh(Sigma)
print(evals)    # variances along principal axes
print(evecs)    # those axes — Week 07's PCA of this cloud
```

### 2.4 Poisson (counting)

A **Poisson** random variable counts events in a fixed window when events occur
independently at a constant average rate: $P(K=k)=e^{-\lambda}\lambda^k/k!$ for
$k=0,1,2,\dots$. Mean and variance are both $\lambda$. Physics lives here:
radioactive decays in a time interval, photoelectrons in a PMT (a light sensor
that turns photons into an electrical pulse), entries in a histogram bin. The
standard deviation is $\sqrt{\lambda}$, which is why a signal $S$ on background
$B$ has uncertainty $\sqrt{S+B}$ (Week 10). Large $\lambda$: approximately
Gaussian with that mean and variance.

## 3. Maximum likelihood

### 3.1 Likelihood and log-likelihood

You have data $x_1,\dots,x_n$, assumed **iid** (independent and identically
distributed) from a family $p(x \mid \theta)$ with unknown parameter $\theta$.
The **likelihood** is the joint probability of the data, viewed as a function of
$\theta$:

$$L(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta).$$

The **maximum-likelihood estimator (MLE)** is $\hat\theta = \arg\max_\theta L(\theta)$
— the parameter that makes the data as probable as possible under the family.

Products of many numbers in $(0,1)$ underflow to zero in floating point. Logs turn
products into sums and do not change the location of the maximum ($\log$ is
strictly increasing):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta).$$

Work with the **log-likelihood** $\ell$ always. Maximizing $\ell$ is minimizing
$-\ell$, the **negative log-likelihood (NLL)** — the training loss of most of
this course.

Differentiating $\ell$ under the sum (gradient and finite sum swap — the same
swap Week 43 uses) gives the **score**
$s(\theta)=\nabla_\theta\ell(\theta)=\sum_i\nabla_\theta\log p(x_i\mid\theta)$.

**Lemma (the MLE score has zero mean).** If the data truly come from
$p(\,\cdot\mid\theta)$, then $\mathbb{E}[\nabla_\theta\log p(X\mid\theta)]=0$.
Proof: $\int p\,\nabla_\theta\log p\,dx=\int\nabla_\theta p\,dx=\nabla_\theta\int p=\nabla_\theta 1=0$.
Week 43 quotes this lemma by name.

### 3.2 Gaussian mean and variance, derived

Take $x_1,\dots,x_n \stackrel{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$, both
parameters unknown. The log-likelihood is

$$\ell(\mu,\sigma^2)
 = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2.$$

$\partial\ell/\partial\mu = \frac1{\sigma^2}\sum_i(x_i-\mu)$. Set to zero:
$\hat\mu=\bar x=\frac1n\sum_i x_i$. Then $\partial\ell/\partial(\sigma^2)=0$
(treat $\sigma^2$ as the variable) gives

$$\hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat\mu)^2.$$

Paper derivation #1: write both derivatives out fully, without looking.

### 3.3 The variance MLE is biased

An estimator $\hat\theta$ is **unbiased** if $\mathbb{E}[\hat\theta] = \theta$ for
every true $\theta$. The sample mean is unbiased: $\mathbb{E}[\hat\mu] = \mu$.
The variance MLE is not. Because $\hat\mu$ was estimated from the same $n$ points,
the residuals $(x_i-\hat\mu)$ are slightly too small on average (they were
minimized by construction — that is what $\partial\ell/\partial\mu = 0$ did), and

$$\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = \frac{n-1}{n}\,\sigma^2.$$

The usual fix is to divide by $n-1$: $s^2=\frac1{n-1}\sum_i(x_i-\bar x)^2$ is
unbiased. `np.var(x)` divides by $n$; `np.var(x, ddof=1)` divides by $n-1$.

```python
def mle_var(x):
    return np.mean((x - x.mean())**2)      # divide by n

n, true_var = 10, 4.0
ests = [mle_var(rng.normal(0, 2.0, size=n)) for _ in range(20000)]
print(np.mean(ests), ((n-1)/n)*true_var)   # ~3.6  3.6  — biased low
```

## 4. Least squares from a Gaussian likelihood

Week 05 minimized squared error because it was convenient. Week 06 showed it is a
projection. Here is the deep reason — paper derivation #2.

**Model.** Observations $y_i$ equal a prediction $f(x_i; \theta)$ plus noise:

$$y_i = f(x_i; \theta) + \varepsilon_i, \qquad
\varepsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2).$$

That is: the noise is Gaussian, independent across $i$, and every point has the
*same* $\sigma$ (**homoscedastic** — "same scatter"). Then
$y_i \mid x_i, \theta \sim \mathcal{N}(f(x_i;\theta),\,\sigma^2)$, so the
log-likelihood is

$$\ell(\theta)
 = \text{const}(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}\big(y_i - f(x_i;\theta)\big)^2.$$

Maximizing $\ell$ in $\theta$ is exactly minimizing $\sum_i (y_i - f_i)^2$. For a
model linear in $\theta$, $f = X\theta$, that is ordinary least squares — Week 06's
normal equations, now with a probability name. **If measurement noise is Gaussian,
iid, and constant-$\sigma$, minimizing squared loss *is* maximum likelihood.**
Week 09 cites this sentence.

The assumption that breaks it is **heteroscedastic** noise: each point has its own
$\sigma_i$ (a calorimeter channel with more energy has larger stochastic
fluctuations; a counting bin with more entries has $\sigma_i = \sqrt{n_i}$). Then

$$\ell(\theta)
 = \text{const} - \sum_{i=1}^{n}\left[\log\sigma_i
 + \frac{(y_i - f_i)^2}{2\sigma_i^2}\right].$$

If the $\sigma_i$ are known, maximizing $\ell$ is **weighted least squares**:
minimize $\sum_i w_i(y_i-f_i)^2$ with $w_i=1/\sigma_i^2$. Noisy points
down-weight themselves.

```python
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([1.1, 1.9, 3.2, 3.8])
sigma = np.array([0.1, 0.1, 0.5, 0.1])     # point 3 is noisy
w = 1.0 / sigma**2
m_hat = np.sum(w * x * y) / np.sum(w * x * x)
print(m_hat)    # pulled toward the precise points, away from y=3.2
```

## 5. MAP: a Gaussian prior becomes ridge

MLE treats $\theta$ as an unknown constant. **Maximum a posteriori (MAP)** treats
$\theta$ as a random variable with a **prior** $p(\theta)$, and maximizes the
posterior $p(\theta \mid \text{data})$. Bayes, dropping the $\theta$-independent
evidence:

$$p(\theta \mid \text{data}) \propto L(\theta)\, p(\theta)
\qquad\Longrightarrow\qquad
\hat\theta_{\text{MAP}} = \arg\max_\theta\big[\ell(\theta) + \log p(\theta)\big].$$

The prior is an extra term in the objective. That is all regularization is.

Now the calculation Week 09 quotes. Linear model, Gaussian likelihood as in §4,
and a **Gaussian prior** on the weights: $w \sim \mathcal{N}(0, \tau^2 I)$
(independent coordinates, mean zero, common variance $\tau^2$). Then
$\log p(w) = \text{const} - \frac{1}{2\tau^2}\|w\|^2$, and

$$-\log p(w \mid y)
 = \frac{1}{2\sigma^2}\|Xw - y\|^2 + \frac{1}{2\tau^2}\|w\|^2 + \text{const}.$$

Minimizing this is least squares plus an $L_2$ penalty $\lambda\|w\|^2$ with
$\lambda=\sigma^2/\tau^2$. That *is* **ridge regression**. A tight prior (small
$\tau$, large $\lambda$) pulls $w$ toward zero; a vague prior recovers MLE. The
Gaussian likelihood and Gaussian prior are conjugate: the posterior is Gaussian
too (complete the square in $w$; Week 35 reuses "product of two Gaussians in $x$
is Gaussian in $x$"). Setting the gradient to zero recovers

$$(X^\top X + \lambda I)\, w = X^\top y.$$

Bonus: $\lambda I$ adds $\lambda$ to every eigenvalue of $X^\top X$, so the system
is always invertible for $\lambda>0$ — Week 07's condition-number story, now with
a prior attached.

## 6. Information: entropy, cross-entropy, KL

Three numbers, one identity, and a chain that is the rest of the course's losses.

### 6.1 Entropy

The **entropy** of a discrete distribution $p$ is

$$H(p) = -\sum_k p_k \log p_k$$

(with the convention $0\log 0 = 0$). It measures uncertainty: $H=0$ when $p$ is a
spike on one outcome (you already know the answer); $H$ is largest when $p$ is
uniform (every outcome equally unsurprising). The unit depends on the log:
$\log_2$ gives **bits**, $\ln$ gives **nats**. ML defaults to nats (natural log);
Week 11's decision trees often use bits. Convert by $H_{\text{bits}} = H_{\text{nats}}/\ln 2$.

A fair coin: $H=\log 2$ ($1$ bit, or $\ln 2$ nats). A coin that always lands
heads: $H=0$. Continuous version $H(p)=-\int p\log p\,dx$ is **differential
entropy** (can be negative; differences still mean something — KL is one).

```python
def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))          # nats

print(entropy([0.5, 0.5]), np.log(2))      # ln 2
print(entropy([1.0, 0.0]))                 # 0.0
```

A code is most efficient when its symbols are equiprobable (entropy maximized) —
why Week 30's NF4 bins sit on equal-mass quantiles.

### 6.2 Cross-entropy

The **cross-entropy** of $q$ relative to $p$ is

$$H(p, q) = -\sum_k p_k \log q_k.$$

You evaluate $-\log q$ under the *true* $p$. If $q=p$, CE equals entropy. If $q$
puts near-zero mass where $p$ puts mass, $-\log q_k$ blows up: confident wrong
predictions are expensive. That is Week 10's log-loss and Week 13's softmax
cross-entropy.

### 6.3 KL divergence, and $KL \ge 0$ by Jensen

The **Kullback–Leibler (KL) divergence** from $q$ to $p$ is

$$\mathrm{KL}(p \,\|\, q) = \sum_k p_k \log\frac{p_k}{q_k} = \mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right].$$

It is *not* a distance: $\mathrm{KL}(p\|q)\ne\mathrm{KL}(q\|p)$ (asymmetric), and
it fails the triangle inequality. It *is* the extra surprise of using $q$ when
the truth is $p$.

**Jensen's inequality.** A function $f$ is **concave** if every chord lies below
the graph; $\log$ is concave ($\log''(u) = -1/u^2 < 0$). Jensen says: for concave
$f$ and any random variable $Y$,

$$f(\mathbb{E}[Y]) \ge \mathbb{E}[f(Y)].$$

(Equality iff $Y$ is constant almost surely.) Apply it to $Y = q(X)/p(X)$ with
$X \sim p$, and $f = \log$:

$$\mathrm{KL}(p\|q)
 = -\mathbb{E}_p\!\left[\log\frac{q}{p}\right]
 \ge -\log\mathbb{E}_p\!\left[\frac{q}{p}\right]
 = -\log\sum_k p_k\cdot\frac{q_k}{p_k}
 = -\log\sum_k q_k
 = -\log 1 = 0.$$

So $\mathrm{KL}(p\|q)\ge 0$, with equality iff $p=q$. Paper derivation #3. (The
continuous case is the same integral.) Week 22 re-uses Jensen on $\log$ to bound
a marginal likelihood.

### 6.4 The chain: $\min$ CE $\Leftrightarrow$ $\min$ KL $\Leftrightarrow$ $\max$ likelihood

Expand the KL:

$$\mathrm{KL}(p\|q) = \sum_k p_k\log p_k - \sum_k p_k\log q_k = -H(p) + H(p,q).$$

Rearrange:

$$\boxed{\,H(p,q) = H(p) + \mathrm{KL}(p\|q)\,}$$

Cross-entropy equals entropy of the truth plus the KL from truth to model. $H(p)$
does not depend on $q$. Therefore **minimizing cross-entropy over $q$ is exactly
minimizing $\mathrm{KL}(p\|q)$**, which (by $KL\ge 0$) drives $q$ toward $p$, which
for empirical $p$ is **maximum likelihood**. That chain is why neural nets train
on cross-entropy, why Week 10's Bernoulli NLL is binary cross-entropy, and why
Week 13's categorical NLL is softmax cross-entropy: they are MLE in
information-theory clothes.

### 6.5 KL of two Gaussians, closed form

For $p = \mathcal{N}(\mu_0,\sigma_0^2)$ and $q = \mathcal{N}(\mu_1,\sigma_1^2)$,
expand $\mathbb{E}_p[\log p - \log q]$ using $\mathbb{E}_p[(x-\mu_0)^2]=\sigma_0^2$
and $\mathbb{E}_p[(x-\mu_1)^2] = \sigma_0^2 + (\mu_0-\mu_1)^2$:

$$\mathrm{KL}(p\|q)
 = \log\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2} - \tfrac12.$$

In $k$ dimensions, $p = \mathcal{N}(\mu_0,\Sigma_0)$, $q = \mathcal{N}(\mu_1,\Sigma_1)$:

$$\mathrm{KL}(p\|q)
 = \tfrac12\Big[\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)
 + (\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0)
 - k + \log\frac{\det\Sigma_1}{\det\Sigma_0}\Big].$$

Paper derivation #4: obtain the 1D formula from the definition, then check it is
zero when $\mu_0=\mu_1$ and $\sigma_0=\sigma_1$. Week 22's encoder–prior KL is
this formula with $q = \mathcal{N}(0,I)$.

```python
def kl_gauss_1d(mu0, s0, mu1, s1):
    return (np.log(s1/s0)
            + (s0**2 + (mu0-mu1)**2) / (2*s1**2)
            - 0.5)

print(kl_gauss_1d(0, 1, 0, 1))     # 0.0
print(kl_gauss_1d(0, 1, 1, 1))     # 0.5  — two unit Gaussians, means 1 apart
```

**Part B — Optimizers.** Week 05 gave you $x\leftarrow x-\eta\nabla f$ and showed
that $\eta$ too large diverges and that an elongated valley zig-zags. This part
derives those facts, then the four algorithms that fix them. Implement them in
`src/optim.py`; race them in the project. What follows is the why.

## 7. Gradient descent, from Taylor; the condition-number villain

### 7.1 First-order Taylor, and the update

Near $x$, $f(x+\Delta)\approx f(x)+\nabla f(x)\cdot\Delta$. To decrease $f$, you
want $\nabla f\cdot\Delta<0$. The most negative direction at fixed step length is
$\Delta\propto-\nabla f$ (Week 06 §2.5). Scale by a **learning rate** $\eta>0$:

$$x_{t+1} = x_t - \eta\,\nabla f(x_t).$$

### 7.2 The quadratic $f(x) = \tfrac12 x^\top A x$, and $\eta < 2/\lambda_{\max}$

Take $A$ symmetric positive definite (the Hessian of a convex quadratic; Week 07's
$\nabla_x(x^\top A x) = 2Ax$, so $\nabla f = Ax$). The update is linear:

$$x_{t+1} = x_t - \eta A x_t = (I - \eta A)\, x_t.$$

$A$ has an orthonormal eigenbasis (Week 07) with eigenvalues
$0 < \lambda_{\min} = \lambda_1 \le \cdots \le \lambda_n = \lambda_{\max}$. Write
$x$ in that basis: coordinate $i$ (the **mode** along eigenvector $i$) evolves
independently as

$$x^{(i)}_{t+1} = (1 - \eta\lambda_i)\, x^{(i)}_t = (1 - \eta\lambda_i)^t\, x^{(i)}_0.$$

Each mode *converges* iff $|1 - \eta\lambda_i| < 1$, i.e. $0 < \eta < 2/\lambda_i$.
The tightest such constraint is the largest eigenvalue:

$$\boxed{\,\eta < \frac{2}{\lambda_{\max}}\,}$$

— the stability bound. Week 05's parabola $f(x)=(x-3)^2$ has $f''=2$, so
$\eta<1$, the boundary you measured. Week 09 quotes this for linear regression,
where the Hessian is $2X^\top X$.

### 7.3 Per-mode rates, and $\kappa$ as the zig-zag

Even inside the stable range, modes decay at different speeds. The fastest mode
($\lambda_{\max}$) wants a *small* $\eta$ so as not to overshoot; the slowest mode
($\lambda_{\min}$) wants a *large* $\eta$ so as to actually move. One global $\eta$
cannot satisfy both. The mismatch is the **condition number**

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \ge 1.$$

With $\eta=1/\lambda_{\max}$, the slow mode shrinks by $1-1/\kappa$ per step:
$O(\kappa\log(1/\varepsilon))$ steps to shrink it by $\varepsilon$. Large $\kappa$
is slow, and the *path* is Week 05 §7.2's zig-zag — too bold on the steep axis
(overshoot, sign flip), too timid on the shallow axis (crawl). **The condition
number is the villain of gradient descent** — Week 09's name for it.

Week 05's valley $f=x^2+10y^2$ has Hessian $\mathrm{diag}(2,20)$, so $\kappa=10$.
Paper derivation #5: from $x_{t+1}=(I-\eta A)x_t$, pass to the eigenbasis, produce
both the $\eta$ bound and the $O(\kappa)$ rate.

### 7.4 Forming $A^\top A$ squares $\kappa$

Week 07: the singular values $\sigma_i$ of a matrix $A$ are the square roots of
the eigenvalues of $A^\top A$, and $\kappa(A) = \sigma_{\max}/\sigma_{\min}$.
Therefore

$$\kappa(A^\top A) = \frac{\sigma_{\max}^2}{\sigma_{\min}^2} = \kappa(A)^2.$$

The normal equations form $A^\top A$ and invert it — they square the villain.
That is why `np.linalg.lstsq` (SVD / QR, Week 07) beats `solve(A.T @ A, A.T @ y)`,
and why ridge's $+\lambda I$ (lifting $\lambda_{\min}$) is a numerical kindness as
well as a prior. GD on $\|Ax-b\|^2$ has Hessian $2A^\top A$ and inherits the
*squared* $\kappa$.

```python
A = np.array([[1.0, 0.0], [0.0, 0.1]])
print(np.linalg.cond(A), np.linalg.cond(A.T @ A))   # 10  100
```

## 8. SGD: unbiased minibatches, variance vs batch size

Full-batch GD computes $\nabla L = \frac{1}{n}\sum_{i=1}^{n}\nabla\ell_i$ exactly
each step — expensive when $n$ is huge, and *deterministic*: the same path every
run. **Stochastic gradient descent (SGD)** replaces the sum by a random
**minibatch** $B$ of size $m \ll n$:

$$g_B = \frac{1}{m}\sum_{i\in B}\nabla\ell_i(x).$$

If $B$ is drawn uniformly, $\mathbb{E}[g_B]=\nabla L$: the minibatch gradient is
**unbiased**. You still walk downhill *on average*. Cost is $O(m)$ per step, and
you get **noise**: $\mathrm{Var}(g_B)$ scales as $1/m$ times the per-example
gradient variance (§1.2). Small $m$ — noisy, cheap. $m=n$ is GD, variance zero.

That noise is not only a tax. On a non-convex loss (§12) it kicks the iterate out
of sharp poor valleys; Week 15 wants the noise to *vary*, which is why a
`DataLoader` reshuffles every epoch. The project races batch sizes; carry:
**unbiased, variance $\propto 1/m$.**

## 9. Momentum: an EMA of gradients

GD's zig-zag is velocity that keeps reversing along the steep axis. **Momentum**
keeps a running average of gradients and steps along *that*, so opposing steep-axis
kicks cancel and the shallow-axis signal accumulates.

The average is an **exponential moving average (EMA)**. For a sequence $g_t$ and
decay $\beta \in [0,1)$,

$$v_t = \beta v_{t-1} + (1-\beta)\, g_t$$

is a weighted average of past $g$'s with weights decaying as $\beta^{t-i}$. Then
$x_{t+1}=x_t-\eta\,v_t$. (Classical form $v_t=\mu v_{t-1}+g_t$ is equivalent up to
rescaling $\eta$; the EMA form matches Adam.)

Physics picture, one sentence: a particle in the loss landscape with friction —
momentum carries it along ravines, while friction (the $1-\beta$ mix) damps the
oscillation across the ravine that plain GD performs. $\beta\approx 0.9$ remembers
roughly $1/(1-\beta)\approx 10$ steps.

## 10. RMSProp: per-coordinate step normalization

Momentum fixes *direction oscillation*. It does not fix *scale mismatch*: a
coordinate whose gradients are chronically huge still takes huge steps. **RMSProp**
keeps a per-coordinate EMA of *squared* gradients and divides by its square root:

$$\begin{aligned}
s_t &= \rho\, s_{t-1} + (1-\rho)\, g_t^2
\qquad\text{(elementwise)},\\
x_{t+1} &= x_t - \eta\,\frac{g_t}{\sqrt{s_t}+\varepsilon}.
\end{aligned}$$

$\varepsilon$ (typically $10^{-8}$) is a floor against division by zero.
A coordinate with historically large $|g_j|$ gets a large $s_j$ and a *smaller*
effective step; a quiet coordinate gets a larger one. Per-coordinate learning-rate
adaptation — the zig-zag's steep axis tames itself. $\rho\approx 0.99$ is typical.
(AdaGrad summed squares without decay, so the denominator grew without bound and
steps died. RMSProp's EMA forgets; that is the fix.)

## 11. Adam: two moments, and the $1/(1-\beta^t)$ correction

**Adam** (adaptive moment estimation) is momentum and RMSProp at once: an EMA of
the gradient (first moment $m$) and an EMA of the squared gradient (second moment
$v$), then a step along $m$ scaled by $1/\sqrt{v}$.

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2.
\end{aligned}$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$. Both EMAs start at $0$. That
initialization **biases** them toward zero for the first many steps — especially
$v$, whose $\beta_2$ is so close to 1 that the memory is long. Derive the bias
and you get Adam's famous correction.

Unroll the first-moment EMA from $m_0 = 0$:

$$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i.$$

Take $\mathbb{E}[\,\cdot\,]$ and suppose $\mathbb{E}[g_i] = g$ (constant, for the
diagnosis). The sum is a geometric series:

$$\mathbb{E}[m_t]
 = (1-\beta_1)\, g \sum_{k=0}^{t-1} \beta_1^k
 = (1-\beta_1)\, g \cdot \frac{1-\beta_1^t}{1-\beta_1}
 = (1-\beta_1^t)\, g.$$

So $m_t$ estimates $g$ times $(1-\beta_1^t)$, not $g$. Divide:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}$$

is unbiased. The same argument on $v$ (with $\beta_2$, and $g^2$) gives
$\hat v_t = v_t / (1-\beta_2^t)$. Paper derivation #6: produce
$1/(1-\beta^t)$ from the geometric sum, starting at zero. Then the step is

$$x_{t+1} = x_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.$$

At $t=1$, $1-\beta_1$ exactly undoes the first EMA mix; as $t\to\infty$,
$\beta^t\to 0$ and the correction turns off. Implement this, with the correction,
in `src/optim.py` and race it in the project. Week 10 trains logistic regression
with your Adam; Week 15 wraps the same update around `model.parameters()`.

```python
# one Adam step from m=v=0, to see the correction fire
beta1, beta2, eta, eps = 0.9, 0.999, 0.1, 1e-8
g = 2.0
m, v = (1-beta1)*g, (1-beta2)*g**2
m_hat, v_hat = m/(1-beta1**1), v/(1-beta2**1)
step = eta * m_hat / (np.sqrt(v_hat) + eps)
print(m, m_hat)    # 0.2  2.0 — correction restored the gradient
print(step)        # ~0.1 — first step is about eta * sign(g)
```

## 12. Convexity, in one page

A function $f$ is **convex** if every chord lies *above* the graph:

$$f\big(t x + (1-t)y\big) \le t\, f(x) + (1-t)\, f(y)
\qquad\text{for all } x,y \text{ and all } t\in[0,1].$$

Equivalently (twice differentiable): the Hessian is positive semidefinite
everywhere — every curvature is a valley, never a saddle or a hill.

What convexity buys: **every local minimum is global.** If $\nabla f(x^*)=0$, you
are done. GD with a sane $\eta$ finds it from any start.

Least squares is convex. **Logistic regression** (Week 10) is convex — GD / Adam
find the global NLL minimum. **Deep networks are not.** Their losses have saddles
and inequivalent valleys; initialization, SGD noise, and the optimizer all matter.
Phase 2 lives there. Convexity is a property of the objective as a function of
the *parameters*: Week 10's sigmoid saturates, but log-loss composed with it is
still convex in $w$.

## Check yourself

1. A tagger has efficiency $0.8$ and fake rate $0.05$. In a sample that is $2\%$
   signal, what is $P(\text{signal} \mid \text{tagged})$? Name the three Bayes
   ingredients.
2. Write the product rule two ways and obtain Bayes' theorem in one line. What is
   the evidence?
3. For iid $\mathcal{N}(\mu,\sigma^2)$ data, state the MLE of $\mu$ and of
   $\sigma^2$. Why is the variance MLE biased, and what is $\mathbb{E}[\hat\sigma^2_{\text{MLE}}]$?
4. Under what noise assumption is least squares maximum likelihood? What single
   change of that assumption makes *weighted* least squares the MLE instead?
5. Show in two sentences that a Gaussian prior $w\sim\mathcal{N}(0,\tau^2 I)$ plus
   a Gaussian likelihood turns least squares into ridge. What is $\lambda$ in
   terms of $\sigma$ and $\tau$?
6. State the identity relating cross-entropy, entropy, and KL. Why does minimizing
   CE over the model also maximize likelihood?
7. Why is $\mathrm{KL}(p\|q)\ge 0$? (Name the inequality and the function it is
   applied to.) Give the 1D two-Gaussian KL and check it vanishes when $p=q$.
8. On $f(x)=\tfrac12 x^\top A x$ with $A$ SPD, why must $\eta < 2/\lambda_{\max}$?
   What is $\kappa$, where did you already watch it, and what does forming
   $A^\top A$ do to it?
9. Why is a minibatch gradient unbiased? How does its variance scale with batch
   size $m$? Derive Adam's factor $1/(1-\beta^t)$ from an EMA that starts at $0$.
10. What does convexity buy you about local minima? Which of logistic regression
    (Week 10) and a deep net is convex?

## Answers

1. Evidence $P(\text{tagged}) = 0.8\cdot 0.02 + 0.05\cdot 0.98 = 0.065$; posterior
   $0.016/0.065 \approx 0.246$. Prior $P(\text{signal})=0.02$, likelihoods
   $0.8$ and $0.05$, posterior $P(\text{signal}\mid\text{tagged})$.
2. $p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x)$; divide the second equality by $p(x)$ to
   get $p(y\mid x)=p(x\mid y)p(y)/p(x)$. The evidence is the denominator $p(x)$,
   equal to $\sum_y p(x\mid y)p(y)$.
3. $\hat\mu=\bar x$; $\hat\sigma^2_{\text{MLE}}=\frac1n\sum_i(x_i-\hat\mu)^2$.
   Using $\hat\mu$ (fitted on the same points) shrinks the residuals, so the
   estimator is low: $\mathbb{E}[\hat\sigma^2_{\text{MLE}}]=\frac{n-1}{n}\sigma^2$.
4. iid additive Gaussian noise with *constant* $\sigma$ (homoscedastic). If each
   point has its own known $\sigma_i$ (heteroscedastic), MLE is weighted LS with
   $w_i=1/\sigma_i^2$.
5. $-\log\text{posterior} = \frac1{2\sigma^2}\|Xw-y\|^2 + \frac1{2\tau^2}\|w\|^2
   +\text{const}$; the second term is $\lambda\|w\|^2$ (up to a constant factor
   that does not change the minimizer) with $\lambda=\sigma^2/\tau^2$.
6. $H(p,q)=H(p)+\mathrm{KL}(p\|q)$. $H(p)$ does not depend on the model $q$, so
   $\min_q H(p,q)$ is $\min_q\mathrm{KL}(p\|q)$, which (empirical $p$) is MLE.
7. Jensen on $\log$ (concave): $\mathrm{KL}(p\|q)=-\mathbb{E}_p[\log(q/p)]
   \ge -\log\mathbb{E}_p[q/p]=-\log 1=0$, equality iff $p=q$.
   $\mathrm{KL}=\log(\sigma_1/\sigma_0)+(\sigma_0^2+(\mu_0-\mu_1)^2)/(2\sigma_1^2)-1/2$;
   plug $\mu_0=\mu_1$, $\sigma_0=\sigma_1$ and every term cancels to $0$.
8. In the eigenbasis, mode $i$ multiplies by $1-\eta\lambda_i$ each step; all modes
   shrink iff $\eta<2/\lambda_{\max}$. $\kappa=\lambda_{\max}/\lambda_{\min}$ is
   the curvature mismatch — Week 05's $x^2+10y^2$ zig-zag ($\kappa=10$). Forming
   $A^\top A$ squares $\kappa$, which is why the normal equations are worse
   conditioned than $A$ and why they are GD's villain twice over.
9. $\mathbb{E}[g_B]=\nabla L$ because a uniformly drawn subset (or with-replacement
   sample) has each $\nabla\ell_i$ appearing with equal probability. Variance
   scales as $1/m$. EMA $m_t=(1-\beta)\sum_{i=1}^t \beta^{t-i}g_i$ with $m_0=0$
   has $\mathbb{E}[m_t]=(1-\beta^t)g$ if $\mathbb{E}[g_i]=g$, so dividing by
   $1-\beta^t$ unbiases it.
10. Every local min is global (and GD finds it). Logistic regression is convex;
    deep nets are not.

## New terms

- **random variable** — a number whose value comes from a chance process.
- **pmf / pdf** — probability of each discrete value; density whose areas are probabilities.
- **expectation $\mathbb{E}$** — probability-weighted average; linear always.
- **variance / standard deviation / covariance / correlation $\rho$** — spread; its square root; co-movement; scaled covariance.
- **quantile** — value below which a fraction $q$ of the mass lies.
- **joint / marginal / conditional** — both variables; one after summing the other out; one given the other.
- **product rule / chain rule of probability** — $p(x,y)=p(x\mid y)p(y)$; a joint as a product of conditionals.
- **independence / uncorrelated** — joint factors; covariance zero (weaker).
- **tower property** — $\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X\mid Y]]$.
- **Bayes' theorem / prior / likelihood / posterior / evidence** — $p(y\mid x)=p(x\mid y)p(y)/p(x)$ and the four named pieces.
- **efficiency / fake rate / purity** — $P(\text{tag}\mid\text{true})$; $P(\text{tag}\mid\text{fake})$; $P(\text{true}\mid\text{tag})$.
- **Bernoulli / Beta / conjugate prior** — coin flip; density on $p\in(0,1)$; prior that stays in-family after updating.
- **Gaussian (1D / multivariate) / covariance matrix $\Sigma$** — bell density; its $k$-D form, with ellipsoidal level sets = PCA of $\Sigma$.
- **Poisson** — counts of independent events; mean = variance $= \lambda$.
- **iid** — independent and identically distributed.
- **likelihood / log-likelihood / NLL / MLE** — $p(\text{data}\mid\theta)$; its log; minus that; the $\theta$ that maximizes it.
- **score** — $\nabla_\theta\log p$; mean zero under the true model.
- **unbiased estimator** — $\mathbb{E}[\hat\theta]=\theta$; the Gaussian variance MLE is not (divides by $n$).
- **homoscedastic / heteroscedastic** — constant $\sigma$; per-point $\sigma_i$.
- **weighted least squares** — MLE under known heteroscedastic Gaussian noise, weights $1/\sigma_i^2$.
- **MAP** — maximize $p(\theta\mid\text{data})\propto L(\theta)p(\theta)$.
- **ridge / $L_2$ regularization** — LS plus $\lambda\|w\|^2$; MAP with a Gaussian prior on $w$.
- **entropy $H(p)$ / bits / nats** — $-\sum p\log p$; $\log_2$ units; $\ln$ units.
- **cross-entropy $H(p,q)$** — $-\sum p\log q$.
- **KL divergence** — $\mathbb{E}_p[\log(p/q)]$; $\ge 0$, $=0$ iff $p=q$, asymmetric.
- **Jensen's inequality** — $f(\mathbb{E}Y)\ge\mathbb{E}f(Y)$ for concave $f$ (e.g. $\log$).
- **learning rate $\eta$ / mode** — GD step size; an eigendirection of the Hessian.
- **condition number $\kappa=\lambda_{\max}/\lambda_{\min}$** — GD's zig-zag villain; squared by $A^\top A$.
- **SGD / minibatch / unbiased gradient** — random subset of the sum; $\mathbb{E}[g_B]=\nabla L$.
- **EMA / momentum** — exponential moving average; EMA of gradients as the step direction.
- **RMSProp** — per-coordinate step $=$ gradient $/$ RMS of recent gradients.
- **Adam / bias correction $1/(1-\beta^t)$** — first+second moment EMAs; undoes the zero initialization.
- **convex function** — chords above the graph; Hessian PSD; every local min is global.

## Going deeper

- Bishop, *PRML*, §1.2 (probability, Bayes, MLE vs MAP) and §2.1–2.3
  (Bernoulli/Beta, the Gaussian). In `references/`. Part A's spine; re-derive
  every displayed equation.
- 3Blue1Brown, the Bayes' theorem video (watch before §1.5) and the
  neural-network gradient-descent video (a picture of Part B).
- StatQuest, "Maximum Likelihood" and "Entropy (for data science)" — short if
  §3 or §6 felt abstract; they do not replace Bishop.
- Ruder, "An overview of gradient descent optimization algorithms" — momentum
  through Adam. Read after you have implemented `src/optim.py`.
- Goh, "Why Momentum Really Works" (Distill) — the damped-oscillator picture of
  §9, with interactive $\beta$ and $\eta$.
- Optional: Boyd & Vandenberghe, *Convex Optimization*, opening pictures of
  convex sets and functions. Pictures only.
