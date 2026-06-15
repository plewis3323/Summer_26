# Bishop 1.2 — Probability Theory

*Grad-student summary + worked exercises. Source: Bishop, PRML §1.2.*

Probability is the framework for quantifying uncertainty — from noise in measurements and finite data. The whole chapter rests on **two rules**.

## The two rules of probability

Let $X$ take values $\{x_i\}$ and $Y$ take values $\{y_j\}$.

- **Sum rule** (marginalization):
$$p(X) = \sum_{Y} p(X, Y).$$
- **Product rule:**
$$p(X, Y) = p(Y \mid X)\, p(X).$$

From these, **Bayes' theorem**:
$$\boxed{\,p(Y \mid X) = \frac{p(X \mid Y)\, p(Y)}{p(X)}\,}, \qquad p(X) = \sum_Y p(X\mid Y)\,p(Y).$$
The denominator is the **normalization** that makes the posterior over $Y$ sum to 1.

**Vocabulary that recurs all book:**
- $p(Y)$ — **prior** (before seeing $X$).
- $p(Y\mid X)$ — **posterior** (after observing $X$).
- $p(X\mid Y)$ — **likelihood** (as a function of $Y$).
- posterior $\propto$ likelihood $\times$ prior.

**Independence:** $X,Y$ independent $\iff p(X,Y)=p(X)p(Y) \iff p(Y\mid X)=p(Y)$.

## Probability densities (continuous)

For continuous $x$, $p(x)$ is a **density**: $p(x\in(a,b)) = \int_a^b p(x)\,dx$, with $p(x)\ge0$ and $\int_{-\infty}^{\infty} p(x)\,dx = 1$. Cumulative distribution $P(z)=\int_{-\infty}^{z}p(x)\,dx$, $P'(x)=p(x)$.

⚠️ **Densities transform with a Jacobian.** Under $x=g(y)$,
$$p_y(y) = p_x(x)\left|\frac{dx}{dy}\right|.$$
Consequence: the location of the **maximum of a density is not invariant** under nonlinear reparameterization (unlike a true probability). Worth remembering when people "find the mode."

## Expectations and covariances

- **Expectation:** $\mathbb{E}[f] = \sum_x p(x) f(x)$ (discrete) or $\int p(x) f(x)\,dx$ (continuous).
- **Monte-Carlo estimate** from samples: $\mathbb{E}[f] \approx \frac{1}{N}\sum_{n=1}^N f(x_n)$.
- **Variance:** $\operatorname{var}[f] = \mathbb{E}\big[(f - \mathbb{E}[f])^2\big] = \mathbb{E}[f^2] - \mathbb{E}[f]^2.$
- **Covariance:** $\operatorname{cov}[x,y] = \mathbb{E}_{x,y}\big[(x-\mathbb{E}[x])(y-\mathbb{E}[y])\big] = \mathbb{E}[xy]-\mathbb{E}[x]\mathbb{E}[y].$ For vectors, $\operatorname{cov}[\mathbf{x}] = \mathbb{E}[(\mathbf{x}-\bar{\mathbf{x}})(\mathbf{x}-\bar{\mathbf{x}})^\mathsf{T}]$, a matrix.

## Bayesian vs. frequentist

- **Frequentist:** probability = long-run frequency; $\mathbf{w}$ is a fixed unknown; estimate it (e.g. **maximum likelihood**), quantify uncertainty via error bars / the **bootstrap** over imagined repeated datasets.
- **Bayesian:** probability = degree of belief; treat $\mathbf{w}$ as a random variable with a prior $p(\mathbf{w})$; condition on the *single* observed dataset $\mathcal{D}$:
$$p(\mathbf{w}\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\mathbf{w})\,p(\mathbf{w})}{p(\mathcal{D})}.$$
Here $p(\mathcal{D}\mid\mathbf{w})$ is the **likelihood** — *not* a probability over $\mathbf{w}$, but the data probability as a function of $\mathbf{w}$.

**Maximum likelihood (ML):** choose $\mathbf{w}$ maximizing $p(\mathcal{D}\mid\mathbf{w})$, equivalently minimizing $-\ln p(\mathcal{D}\mid\mathbf{w})$ (the "error function"). Bishop's caution: ML can **over-fit**; the Bayesian prior acts like a built-in regularizer. Critique of Bayes: results depend on the prior choice.

## The Gaussian

Univariate:
$$\mathcal{N}(x\mid\mu,\sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}}\exp\!\Big\{-\frac{1}{2\sigma^2}(x-\mu)^2\Big\},$$
mean $\mathbb{E}[x]=\mu$, variance $\operatorname{var}[x]=\sigma^2$, precision $\beta=1/\sigma^2$. Multivariate over $D$ dims:
$$\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\boldsymbol\Sigma)=\frac{1}{(2\pi)^{D/2}|\boldsymbol\Sigma|^{1/2}}\exp\!\Big\{-\tfrac12(\mathbf{x}-\boldsymbol\mu)^\mathsf{T}\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big\}.$$

**ML for a Gaussian** from i.i.d. data $\{x_n\}$: maximize $\ln p(\mathbf{x}\mid\mu,\sigma^2)$ →
$$\mu_\text{ML} = \frac1N\sum_n x_n \ \text{(sample mean)},\qquad \sigma^2_\text{ML} = \frac1N\sum_n (x_n-\mu_\text{ML})^2.$$
**Bias:** $\mathbb{E}[\sigma^2_\text{ML}] = \frac{N-1}{N}\sigma^2$ — ML *under-estimates* the variance. The unbiased fix divides by $N-1$. This bias is a small concrete instance of over-fitting and vanishes as $N\to\infty$.

## Curve fitting re-visited (probabilistic view)

Model the target as Gaussian around the polynomial:
$$p(t\mid x,\mathbf{w},\beta) = \mathcal{N}\big(t \mid y(x,\mathbf{w}),\ \beta^{-1}\big).$$
Maximizing the log-likelihood over $\mathbf{w}$ is **equivalent to minimizing sum-of-squares error** — least squares falls out of a Gaussian noise assumption. Maximizing over $\beta$ gives the noise precision.

Add a prior $p(\mathbf{w}\mid\alpha)=\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I})$ and maximize the posterior (**MAP**): this is **equivalent to regularized least squares** with $\lambda=\alpha/\beta$. So regularization = a Gaussian prior on the weights. (Connects straight back to §1.1.)

**Fully Bayesian** curve fitting integrates over $\mathbf{w}$ rather than picking one value:
$$p(t\mid x,\mathbf{x},\mathbf{t}) = \int p(t\mid x,\mathbf{w})\,p(\mathbf{w}\mid\mathbf{x},\mathbf{t})\,d\mathbf{w},$$
yielding a **predictive distribution** — a mean prediction *with* input-dependent error bars, not a point estimate.

---

## Worked exercise (self-set, Bayes' theorem)

**Q (Bishop's fruit-boxes example):** Red box chosen with $p(B{=}r)=0.4$, blue with $p(B{=}b)=0.6$. Red box: 2 apples, 6 oranges. Blue box: 3 apples, 1 orange. We draw a fruit and it's an **orange**. What's $p(B{=}r\mid \text{orange})$?

**A:** Likelihoods: $p(o\mid r)=6/8=0.75$, $p(o\mid b)=1/4=0.25$.
Evidence (sum rule):
$$p(o)=p(o\mid r)p(r)+p(o\mid b)p(b)=0.75(0.4)+0.25(0.6)=0.30+0.15=0.45.$$
Bayes:
$$p(r\mid o)=\frac{p(o\mid r)p(r)}{p(o)}=\frac{0.30}{0.45}=\frac{2}{3}\approx0.667.$$
The prior on red was $0.4$; observing an orange **raises** it to $0.667$ because oranges are more common in the red box. That update *is* learning. $\blacksquare$

See `img_bayes_fruit.png` for the prior→posterior shift.
