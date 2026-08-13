# Week 08 — Exercises

Probability block (E1–E4) lives in the notebook — setup gives the imports, a
seeded generator `rng = np.random.default_rng(8)`, the coin/tagger numbers, the
regression arrays, and labeled axes; you write only the lines asked for.
Optimizer block (E5–E8) *is* the mini-project: those four live in files, and
`project.md` guides every one of them in detail — read it before starting E5.
Each of E5–E8 names files, and each acceptance criterion is `pytest` or a
figure (per `NOTEBOOK_RULES.md` §6). Use notebook cells to prototype an update
or a contour, then promote into `src/optim.py`. Derivations asked for "on paper"
go in your derivations folder.

## E1 — Bayesian coin

Setup gives a Beta$(\alpha,\beta)$ prior on a coin's $p$ (not flat) and a
sequence of flips. Compute the MLE $\hat p = s/n$, the posterior
Beta$(\alpha+s,\,\beta+f)$, the posterior mean $(\alpha+s)/(\alpha+\beta+n)$,
and the MAP $(\alpha+s-1)/(\alpha+\beta+n-2)$ (for $\alpha,\beta>1$). Overlay
prior and posterior densities on $[0,1]$ and mark the three point estimates.
Hint: lesson §2.1; `from scipy.stats import beta as beta_dist` — the pdf is
`beta_dist.pdf(grid, a, b)`. Conjugacy means you never need to write a
normalize-the-likelihood loop.
Accept when: posterior parameters equal $\alpha+s$ and $\beta+f$ exactly, and
the plot shows the posterior peaked between the prior mean and the MLE.

## E2 — Detector Bayes

Setup gives a tagger's efficiency $\varepsilon = P(\text{tagged}\mid\text{signal})$,
fake rate $f = P(\text{tagged}\mid\text{background})$, and prior abundance
$\pi = P(\text{signal})$. Compute the evidence $P(\text{tagged})$ and the
purity $P(\text{signal}\mid\text{tagged})$ from Bayes' theorem. Repeat at a
second prior (the same tagger, a richer sample) and print both purities.
Hint: lesson §1.5; evidence is $\varepsilon\pi + f(1-\pi)$; posterior is
$\varepsilon\pi$ over that. Count it with a 1000-track table if the formula
feels slippery.
Accept when: both purities match the closed-form values to 1e-12, and a
one-line comment names which of (efficiency, fake rate, prior) moved the
purity between the two samples.

## E3 — Least squares from a Gaussian likelihood

Setup gives $x$ and noisy $y = a_{\text{true}} x + b_{\text{true}} +
\varepsilon$ with iid $\mathcal{N}(0,\sigma^2)$ noise. Write the Gaussian
log-likelihood $\ell(a,b)$ (up to a $\theta$-independent constant), maximize
it by minimizing $\sum_i (y_i - a x_i - b)^2$, and show the minimizer equals
`np.linalg.lstsq` on the design matrix with columns $x$ and ones.
Hint: lesson §4; the $\sigma$-dependent pieces do not move the $\arg\max$ in
$(a,b)$. `np.stack([x, np.ones(x.size)], axis=1)` is the design.
Accept when: MLE $(a,b)$ matches `lstsq` to 1e-10.

## E4 — Heteroscedastic twist, then KL

Setup gives the same $x$ with *known* per-point $\sigma_i$ (one point much
noisier). Fit ordinary least squares and weighted least squares
($w_i = 1/\sigma_i^2$); report both slopes. Then, on the provided discrete
$p$ and $q$ and the two 1-D Gaussians, compute entropy $H(p)$, cross-entropy
$H(p,q)$, and $\mathrm{KL}(p\|q)$, and check $H(p,q) = H(p) + \mathrm{KL}(p\|q)$
and the closed-form two-Gaussian KL (lesson §6.5) against a Monte Carlo
estimate `np.mean(log p(x) - log q(x))` on draws from $p$.
Hint: WLS slope for a through-origin line is
$\sum w_i x_i y_i / \sum w_i x_i^2$; for intercept too, solve the $2\times 2$
normal equations with weights, or `np.linalg.lstsq` on
`np.sqrt(w)[:, None] * A`. Entropy: skip $p_k=0$ (`0\log 0 = 0`).
Accept when: WLS sits closer to the true slope than OLS does; CE = H + KL to
1e-12; and the Monte Carlo KL matches the closed form within the notebook's
stated Monte Carlo band.

## E5 — Optimizer module

This one goes in files. Write `src/optim.py` with the four update rules of
`project.md` §Interface — `gd_update`, `momentum_update`, `rmsprop_update`,
`adam_update` (bias correction included) — as plain functions over NumPy
arrays, no classes. Tests in `tests/test_optim.py` lock the 1-D quadratic
stability bound, the first Adam step from $m=v=0$, and a Rosenbrock
convergence check.
Hint: lesson §§7, 9–11; Stage 1 of `project.md` is the spec, including the
exact signatures and the numerical values the tests assert.
Accept when: `uv run pytest -q` is green on the Stage 1 tests.

## E6 — Step-size threshold

On the 1-D quadratic $f(x)=\tfrac12\lambda x^2$ with the $\lambda$ the setup
gives, run your `gd_update` from a fixed $x_0$ across a sweep of $\eta$ that
straddles $2/\lambda$. Plot $|x_T|$ vs $\eta$ after a fixed $T$, and mark the
derived boundary. Repeat on the 2-D bowl $f=x^2 + 10 y^2$ (Week 05's valley)
and report $\eta < 2/\lambda_{\max}$ against the $\eta$ where your trajectory
first diverges.
Hint: lesson §7.2; $\lambda_{\max}$ of $x^2+10y^2$ is 20, so $\eta<0.1$.
Measure divergence as $\|x\|$ growing with $t$, not as "looks wiggly."
Accept when: the 1-D plot flips from decay to explosion at $2/\lambda$ (within
one grid step) and the 2-D bound matches $2/\lambda_{\max}$ the same way.

## E7 — The race

Race GD, momentum, RMSProp, and Adam on the two surfaces in `project.md`
(the elongated quadratic and Rosenbrock), from the stated starts, with the
stated $\eta$. Overlay the four trajectories on a contour plot of each
surface; save the Rosenbrock figure — it is the roadmap's named artifact.
Hint: Stage 3 of `project.md`; each optimizer is a loop calling one update
from E5. Plot $(x_t, y_t)$ as a line, not a scatter of 10,000 dots.
Accept when: all four converge where `project.md` says they should, and
`figures/rosenbrock_trajectories.pdf` exists.

## E8 — Physics fit + pathology

Fit the synthetic Breit–Wigner + polynomial mass spectrum of `project.md`
with your Adam and with `scipy.optimize.curve_fit`, from the same $p_0$.
Report mass and width against truth. Then trigger the stated pathology
(GD from the same $p_0$, or Adam from the far-away mass init) and diagnose
it from the loss curve in a markdown cell of at most 3 lines.
Hint: Stage 4 of `project.md` derives the model and the two inits; bin
centers vs counts, same as Week 04.
Accept when: Adam matches `curve_fit` on mass to the Stage 4 tolerance, and
the pathology is named with a loss-curve figure.

## Review

1. (Wk 07) Condition number of $A^\top A$ vs $A$: which did the normal
   equations suffer from, and how does that connect to this week's $\kappa$
   story?
2. (Wk 07) Write $\nabla_x\|Ax-b\|^2$ from memory; which derivation this week
   reused it?
3. (Wk 06) The projection matrix $P$ and the least-squares/likelihood fit:
   what is the geometric relationship?
4. (Wk 04) Your pseudo-experiment loops need reproducible randomness — which
   NumPy API, and why not the global seed?
