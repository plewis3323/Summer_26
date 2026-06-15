# Bishop 1.1 — Example: Polynomial Curve Fitting

*Grad-student summary + worked exercises. Source: Bishop, PRML §1.1.*

## The setup

We are given a training set of $N$ observations of an input variable $x$ and a target $t$:
$$\mathbf{x} = (x_1, \ldots, x_N)^\mathsf{T}, \qquad \mathbf{t} = (t_1, \ldots, t_N)^\mathsf{T}.$$

In Bishop's running example the data are generated from
$$t_n = \sin(2\pi x_n) + \epsilon_n,$$
where $\epsilon_n$ is small Gaussian noise. The **goal** is to predict the target $\hat{t}$ for a new input $\hat{x}$, *without* knowing the underlying $\sin(2\pi x)$ — we only see the noisy points. This exposes the core tension of the whole book: we must **generalize** from finite, noisy data.

## The model: polynomials linear in their parameters

We fit a polynomial of order $M$:
$$y(x, \mathbf{w}) = w_0 + w_1 x + w_2 x^2 + \cdots + w_M x^M = \sum_{j=0}^{M} w_j x^j.$$

Key idea (recurs constantly later): although $y$ is **nonlinear in $x$**, it is **linear in the coefficients $\mathbf{w}$**. Models linear in their parameters are called *linear models* — this is why the chapter generalizes so cleanly.

## Fitting by least squares

Define the **sum-of-squares error**:
$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \big\{ y(x_n, \mathbf{w}) - t_n \big\}^2.$$
The $\tfrac{1}{2}$ is for derivative convenience. $E$ is the squared vertical distance between predictions and targets; $E=0$ iff the curve passes through every point.

Because $E$ is **quadratic in $\mathbf{w}$**, its gradient is **linear in $\mathbf{w}$**, so the minimizer $\mathbf{w}^\star$ is unique and solvable in closed form (a linear system). *(added context: this is the normal-equations / least-squares solution, derived fully in §3.1.)*

## Model selection: choosing $M$ (over-fitting)

- $M=0,1$: too rigid — **under-fits**, can't capture $\sin(2\pi x)$.
- $M=3$: good fit — captures the shape.
- $M=9$: passes through **all 10 points** ($E=0$) but oscillates wildly → **over-fitting**. Excellent on training data, terrible generalization.

The paradox: a higher-order polynomial *contains* all lower orders as special cases, so it should be at least as powerful — yet it generalizes worse. The issue is that the coefficients $\mathbf{w}$ are tuned to fit the **noise**.

### Diagnostic: the coefficient magnitudes blow up

| | $M=0$ | $M=1$ | $M=3$ | $M=9$ |
|---|---|---|---|---|
| $w^\star$ magnitudes | small | small | moderate | **huge** ($\pm 10^5$) |

As $M$ grows the fitted coefficients become enormous with alternating signs — a fingerprint of over-fitting.

### Quantifying error: RMS

Compare train vs. test using the **root-mean-square error**, normalized so different $N$ are comparable:
$$E_\text{RMS} = \sqrt{\tfrac{2 E(\mathbf{w}^\star)}{N}}.$$
Test $E_\text{RMS}$ is large at small $M$ (under-fit), dips around $M=3$, then explodes at $M=9$ (over-fit), while train $E_\text{RMS}$ keeps falling toward 0. The gap between the two curves *is* over-fitting.

## More data fixes over-fitting

For fixed $M=9$, increasing $N$ (e.g. 15 → 100 points) tames the oscillations: the same flexible model behaves well with more data. Rough heuristic Bishop mentions: $N$ should be some multiple (e.g. 5–10×) of the number of parameters. He flags this as unsatisfying — model complexity shouldn't have to scale with data size — motivating the **Bayesian** treatment later.

## Regularization

Instead of throwing out flexibility, **penalize large coefficients**. Add a quadratic penalty:
$$\widetilde{E}(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^{N}\big\{y(x_n,\mathbf{w}) - t_n\big\}^2 + \frac{\lambda}{2}\lVert \mathbf{w} \rVert^2,$$
where $\lVert \mathbf{w}\rVert^2 = \mathbf{w}^\mathsf{T}\mathbf{w} = \sum_j w_j^2$. In statistics this is **ridge regression**; in NN literature, **weight decay**.

- $\ln\lambda$ very negative ($\lambda\to0$): no penalty → over-fits (back to wild $M=9$).
- $\ln\lambda$ moderate (e.g. $\approx -18$): smooth, good fit.
- $\ln\lambda$ too large: over-penalized → under-fits (curve flattens).

So $\lambda$ controls **effective complexity**. We've traded the discrete knob $M$ for a continuous knob $\lambda$.

## The honest catch: don't tune on the test set

If we pick $M$ or $\lambda$ by minimizing test error, we've used the test set *as training data* for hyperparameters. Proper practice: split into **train / validation (hold-out)**, tune on validation. Wasteful of data → motivates **cross-validation** (§1.3).

---

## Worked exercise (self-set, in the spirit of §1.1)

**Q:** Show that minimizing $E(\mathbf{w})$ leads to a *linear* system in $\mathbf{w}$.

**A:** With $y(x_n,\mathbf{w})=\sum_j w_j x_n^j$,
$$\frac{\partial E}{\partial w_i} = \sum_{n=1}^N \Big\{\sum_j w_j x_n^j - t_n\Big\} x_n^i = 0.$$
Rearranging,
$$\sum_j \underbrace{\Big(\sum_n x_n^{\,i+j}\Big)}_{A_{ij}} w_j = \underbrace{\sum_n x_n^{\,i} t_n}_{T_i}, \qquad\Rightarrow\qquad A\mathbf{w} = \mathbf{T}.$$
$A$ depends only on the inputs, $\mathbf{T}$ on inputs and targets. A single linear solve gives $\mathbf{w}^\star$ — no iterative optimization needed. This is exactly Bishop's Exercise 1.1. $\blacksquare$

See `img_polyfit_overfitting.png` for the $M\in\{0,1,3,9\}$ panel reproduced from the section.
