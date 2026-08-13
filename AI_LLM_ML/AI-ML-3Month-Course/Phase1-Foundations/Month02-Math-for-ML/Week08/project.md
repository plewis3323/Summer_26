# Week 08 Project — The Optimizer Race

## Objective

Hand-roll the four update rules that train every later model in this course —
gradient descent, momentum, RMSProp, and Adam (with bias correction) — as a
tested NumPy module, then race them on surfaces designed to hurt and on a
synthetic particle-physics mass-spectrum fit. This is the Month 02 deliverable
(see the Month 02 README): `src/optim.py` with tests, a Rosenbrock trajectory
plot, and an Adam-vs-SciPy comparison on a Breit–Wigner + polynomial peak.
The point is not a new physics result; it is owning the algorithms Week 10
and Week 15 will wrap around `model.parameters()`.

## Background — the surfaces, from scratch

**Gradient descent, and why it zig-zags.** Week 05 stepped
$x \leftarrow x - \eta\nabla f$ down $f(x,y)=x^2+10y^2$ and watched the path
bounce. Week 07 named the two curvatures as eigenvalues of the Hessian.
This week derived the rest: on a quadratic $f(x)=\tfrac12 x^\top A x$ the
update is $x_{t+1}=(I-\eta A)x_t$, each eigenmode $i$ converges iff
$0 < \eta < 2/\lambda_i$, and the tightest such bound is
$\eta < 2/\lambda_{\max}$. Inside the stable range the slow mode still crawls
at a rate set by the **condition number**
$\kappa=\lambda_{\max}/\lambda_{\min}$. One global $\eta$ cannot be both
small enough for the steep axis and large enough for the shallow one — that
mismatch *is* the zig-zag. The four algorithms you implement are four
responses: momentum averages away the bounce; RMSProp rescales each
coordinate by the RMS of its recent gradients; Adam does both, and corrects
the zero-initialization bias of its exponential moving averages.

**The elongated bowl.** The first race surface is the quadratic

$$f(x,y) = x^2 + 100\, y^2$$

— Hessian $\mathrm{diag}(2, 200)$, so $\lambda_{\max}=200$, $\lambda_{\min}=2$,
$\kappa=100$, and GD is stable only for $\eta < 2/200 = 0.01$. Contours are
skinny ellipses aligned with the axes. This is Week 05's valley with the
elongation turned up; it is also a cartoon of $\kappa(A^\top A)$ from Week 07.
All four methods should reach the origin from $(1,1)$ with a sane $\eta$.
GD will take the long zig-zag; the adaptive methods should cut across.

**Rosenbrock.** The second surface is the **Rosenbrock function**

$$f(x,y) = (1-x)^2 + 100\,(y-x^2)^2.$$

A narrow parabolic valley follows $y=x^2$, with the unique minimum at
$(1,1)$ where $f=0$. Along the valley floor the gradient is tiny; across it
the walls are steep. That is a *curved* $\kappa$ problem: the long direction
bends, so a method that only rescales axes (RMSProp) still has to turn, and
plain GD crawls once it falls into the valley. The standard start is
$(-1.2, 1.0)$ — the point Rosenbrock used in 1960. A trajectory plot of the
four methods on this surface is the roadmap's named figure: you should see
GD drop to the valley and creep, momentum overshoot and settle, RMSProp and
Adam take a more direct path. "Converge where they should" here means: from
that start, with the $\eta$ below, all four reach $(1,1)$ to $10^{-3}$ in the
iteration budgets given; none is required to be *fast*.

**A mass spectrum, and the Breit–Wigner.** The physics fit is a miniature of
Week 04's J/ψ job, with a different peak shape — the one you use when the
width is *intrinsic*, not detector resolution.

An unstable particle does not have a single mass. It lives for a short mean
lifetime $\tau$, and the energy–time uncertainty principle spreads the mass
into a peak of width $\Gamma \sim \hbar/\tau$ ($\hbar$ is Planck's constant
over $2\pi$; in the field's natural units one just writes $\Gamma$ in GeV).
The characteristic lineshape is the **Breit–Wigner** (also called a Cauchy
or Lorentzian)

$$B(m; M, \Gamma) = \frac{1}{(m-M)^2 + (\Gamma/2)^2}.$$

$M$ is the pole mass (the center); $\Gamma$ is the full width at half
maximum. This is *not* a Gaussian: the tails fall as $1/m^2$, much slower
than $e^{-m^2}$. Week 04's J/ψ was Gaussian because CMS resolution
($\sim 30$ MeV) dwarfs the J/ψ's intrinsic width ($\sim 0.1$ MeV). Here we
pick a resonance where the opposite is true.

The **$\rho$ meson** (rho, $770$ MeV) is a bound state of a light quark and
a light antiquark. It decays, almost always, to two pions, with a width
$\Gamma \approx 150$ MeV — comparable to its mass, so the peak is broad and
visibly non-Gaussian. A histogram of two-pion invariant masses in that
region shows a BW bump on top of a smooth **combinatorial background**:
random pion pairs that did not come from a $\rho$, whose mass distribution
is slowly varying and well-modeled by a low-order polynomial. The fit model
in this project is therefore

$$f(m) = A\cdot B(m; M, \Gamma) + a + b\, m$$

— four shape parameters $(A, M, \Gamma)$ plus a linear background $(a,b)$.
(Five numbers, same count as Week 04's Gaussian + line.) Fitting means
minimizing the summed squared residuals between $f$ and the histogram bin
counts, which (Week 08 Part A) is maximum likelihood if the bin errors are
iid Gaussian — a decent approximation when counts per bin are not tiny.

**Why this loss hurts.** $M$ and $\Gamma$ are in GeV; $A$ is a count and can
be hundreds; $a$ and $b$ are background counts per GeV. The Hessian
eigenvalues of $\|f-y\|^2$ therefore span orders of magnitude — the same
$\kappa$ villain, now in five dimensions. Worse: if you initialize $M$ far
from the peak, $B(m;M,\Gamma)$ is tiny across the whole window, the
gradient w.r.t. $M$ vanishes (the tails are flat), and GD sits still. If you
initialize $\Gamma$ needle-thin, the peak is a spike the bins can miss.
Adam's per-coordinate scaling is supposed to make the well-posed start
easy; the far-away-$M$ start is the pathology you diagnose. SciPy's
`curve_fit` (Levenberg–Marquardt, which uses curvature) is the grown-up
reference your Adam is scored against.

## Data

All synthetic — no download, no checksum. Two generators, both seeded, both
in `src/surfaces.py` (or inline in the race script; one place, not both).

**Bowl and Rosenbrock.** No data. The functions and their analytic gradients:

```python
def bowl(xy):
    x, y = xy
    return x**2 + 100.0 * y**2

def bowl_grad(xy):
    x, y = xy
    return np.array([2.0 * x, 200.0 * y])

def rosenbrock(xy):
    x, y = xy
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dy = 200.0 * (y - x**2)
    return np.array([dx, dy])
```

Starts: bowl at `(1.0, 1.0)`; Rosenbrock at `(-1.2, 1.0)`. Minimum of both
is known: origin, and `(1.0, 1.0)` respectively.

**Mass spectrum.** Draw, then histogram, then forget the unbinned list —
you fit the histogram, as in Week 04.

```python
M_TRUE = 0.770      # GeV, rho mass
G_TRUE = 0.150      # GeV, rho width
A_TRUE = 400.0
A_BKG = 80.0
B_BKG = -30.0       # linear background a + b*m
FIT_LO, FIT_HI = 0.40, 1.20
N_BINS = 60
N_SIG, N_BKG = 4000, 6000
```

Signal samples: inverse-transform or rejection-sample the BW on
`[FIT_LO, FIT_HI]` (a simple loop that proposes uniform $m$ and accepts
with probability $B(m)/B_{\max}$ is fine; $B_{\max}=B(M,\Gamma)$).
Background: `m` from the linear density $\propto a + b m$ on the same
window (or, simpler and acceptable: uniform, which is $b=0$, and let the
fit recover a small $b$). Histogram with `np.histogram(..., bins=N_BINS,
range=(FIT_LO, FIT_HI))`; bin centers vs counts are the data vector.
Seed `rng = np.random.default_rng(8)`. Write `data/spectrum.npz` (centers,
counts, edges) so the fit is bit-stable across runs; commit it or
regenerate it from the seed in `run.py` — pick one, document it, do not
mix.

## Build steps

Do them in order; each is one of this week's exercises (E5–E8 in
`exercises.md`). Probability E1–E4 are notebook-only and are not part of
this repo.

**Stage 0 — the repo (20 min).** Same packaging pattern as Week 04, smaller:

```
cd ~/course
uv init --package optim_race
cd optim_race
git init
uv add numpy matplotlib scipy
uv add --dev pytest
```

Target layout:

```
optim_race/
  pyproject.toml
  uv.lock
  .gitignore          .venv/  __pycache__/  figures/*.pdf
  run.py              Stage 3+4 orchestration
  src/optim.py        Stage 1 — the four updates
  src/surfaces.py     bowl, Rosenbrock, BW model, spectrum generator
  tests/
    test_optim.py
    test_fit.py
  figures/
    reference/        committed Rosenbrock PDF (Stage 3)
  data/
    spectrum.npz      optional; regenerate from seed if absent
```

If `uv init --package` nests the code under `src/optim_race/`, either move
`optim.py` to `src/optim.py` and teach Python the path (`pyproject.toml`
`[tool.setuptools] py-modules = ["optim"]`, or a `sys.path` insert in
`run.py` / tests), or keep the package and *import* `from optim_race.optim
import ...` — but the functions themselves live in a module named `optim`.
No classes. Commit after every stage.

**Stage 1 — optimizer module (E5).** `src/optim.py`, four functions, NumPy
arrays in and out. State (`v`, `s`, `m`) is passed in and returned; the
caller owns it. Defaults match the lesson.

```python
import numpy as np

def gd_update(x, g, eta):
    return x - eta * g

def momentum_update(x, g, v, eta, beta=0.9):
    v = beta * v + (1.0 - beta) * g
    x = x - eta * v
    return x, v

def rmsprop_update(x, g, s, eta, rho=0.99, eps=1e-8):
    s = rho * s + (1.0 - rho) * g**2
    x = x - eta * g / (np.sqrt(s) + eps)
    return x, s

def adam_update(x, g, m, v, t, eta, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    x = x - eta * m_hat / (np.sqrt(v_hat) + eps)
    return x, m, v
```

`t` is 1-based (first step is `t=1`), so the bias correction fires as in
lesson §11. `tests/test_optim.py`:

- **Stability bound.** $f(x)=\tfrac12\lambda x^2$ with $\lambda=2$ (so
  $\eta<1$). From $x_0=1$, 50 steps of `gd_update` with $\eta=0.9$ give
  $|x_{50}|<10^{-6}$; with $\eta=1.1$, $|x_{50}| > |x_0|$.
- **Adam bias correction.** One step from $m=v=0$, $g=2.0$,
  $\eta=0.1$, $\beta_1=0.9$, $\beta_2=0.999$: $\hat m$ equals $2.0$ and the
  step equals $\eta$ to 1e-10 (lesson §11 snippet).
- **Rosenbrock, Adam.** From `(-1.2, 1.0)`, `eta=0.01`, 20,000 steps: the
  iterate is within $10^{-3}$ of `(1,1)` in Euclidean norm.

*Accept when: `uv run pytest -q` is green on these three.*

**Stage 2 — step-size threshold (E6).** This one may live in the race
notebook rather than the package — it is a measurement, not a library
routine. Sweep $\eta$ on the 1-D quadratic and on the bowl; mark
$2/\lambda_{\max}$. The notebook (or a `figures/step_size.pdf` the package
writes) is the artifact. Numbers: bowl $\lambda_{\max}=200$, bound
$\eta<0.01$.
*Accept when: the 1-D plot flips at $2/\lambda$ within one grid step, and
the 2-D bound matches $2/\lambda_{\max}$ the same way.*

**Stage 3 — the race (E7).** `run.py` (or `race.py`) runs all four
optimizers on both surfaces and writes figures. Suggested budgets and
step sizes — use these unless you document a change and still meet the
gate:

| surface    | start         | GD $\eta$ | momentum $\eta$ | RMSProp $\eta$ | Adam $\eta$ | steps   |
|------------|---------------|-----------|-----------------|----------------|-------------|---------|
| bowl       | `(1, 1)`      | 0.008     | 0.008           | 0.05           | 0.05        | 400     |
| Rosenbrock | `(-1.2, 1.0)` | 0.0005    | 0.0005          | 0.01           | 0.01        | 20,000  |

GD's $\eta$ on the bowl is just inside $0.01$; on Rosenbrock it is small
because a brave $\eta$ jumps the valley wall. Overlay trajectories on
contours (`np.meshgrid` + `ax.contour` + four `ax.plot`s). Label axes
$(x,y)$, legend names the method. Save
`figures/bowl_trajectories.pdf` and
`figures/rosenbrock_trajectories.pdf`. When the Rosenbrock figure looks
right, copy it to `figures/reference/` and commit — that is the roadmap
plot.
*Accept when: all four converge where they should (bowl: all four within
$10^{-4}$ of the origin; Rosenbrock: all four within $10^{-3}$ of $(1,1)$)
and the Rosenbrock PDF exists.*

**Stage 4 — physics fit + pathology (E8).** In `src/surfaces.py`, the model
and a residual helper:

```python
def bw(m, M, G):
    return 1.0 / ((m - M)**2 + (G / 2.0)**2)

def model(m, A, M, G, a, b):
    return A * bw(m, M, G) + a + b * m
```

Fit histogram counts at bin centers. Two routes, same $p_0$:

```python
P0 = np.array([300.0, 0.75, 0.12, 60.0, 0.0])   # A, M, G, a, b
```

**(a) SciPy.** `scipy.optimize.curve_fit(model, centers, counts, p0=P0)` —
the reference. **(b) Your Adam.** Loss $L(\theta)=\sum_i (f(m_i;\theta)-y_i)^2$;
gradient by finite differences through `model` (call Week 07's `grad_check`
once on a tiny $\theta$ to make sure you did not flip a sign) or by a
hand-derived $\nabla_\theta L$. Adam: `eta=0.05`, 8,000 steps, parameters
packed as a 1-D array. Print $M\pm$ (no covariance required on the Adam
side) next to `curve_fit`'s $M$ and the true $0.770$.

**Pathology**, one of these two — run both if you like, write up one:

1. **GD from the same $P0$.** Same $\eta$ you would use on a well-scaled
   quadratic. Expected: crawls or diverges on $A$ while $M$ barely moves
   ($\kappa$ of the five-parameter Hessian).
2. **Adam from a far mass.** $P0$ with $M=0.45$ (edge of the window),
   everything else unchanged. Expected: loss flat, $M$ stuck — BW tails,
   vanishing $\partial L/\partial M$.

Plot data + both fitted curves (Adam and SciPy) on one axes; plot the
pathology loss vs step on another. Save `figures/spectrum_fit.pdf` and
`figures/pathology_loss.pdf`.
`tests/test_fit.py`: regenerate (or load) the seeded spectrum, run Adam
from `P0`, assert `|M_hat - 0.770| < 0.030` (30 MeV — the peak is 150 MeV
wide; this is not a precision measurement). A second test asserts
`curve_fit` from `P0` recovers $M$ within 15 MeV of truth.
*Accept when: Adam matches `curve_fit` on mass to 20 MeV, and the
pathology is named with a loss-curve figure.*

## Interface (summary)

| function | returns | notes |
|----------|---------|--------|
| `gd_update(x, g, eta)` | `x` | |
| `momentum_update(x, g, v, eta, beta=0.9)` | `x, v` | EMA form, lesson §9 |
| `rmsprop_update(x, g, s, eta, rho=0.99, eps=1e-8)` | `x, s` | elementwise $g^2$ |
| `adam_update(x, g, m, v, t, eta, beta1=0.9, beta2=0.999, eps=1e-8)` | `x, m, v` | $t$ 1-based; bias correction on |

All of $x,g,v,s,m$ are `np.ndarray` of the same shape. No classes, no
`**kwargs`. SciPy is used only as the physics-fit reference, not as the
optimizer under test.

## Tests

`uv run pytest -q` green means:

- Stage 1's three tests (GD bound, Adam first step, Rosenbrock Adam).
- Stage 4's two tests (Adam recovers $M$ to 30 MeV; `curve_fit` to 15 MeV).

Do not test figure pixels. Do not seed from `np.random.seed`.

## Figures

| file | what |
|------|------|
| `figures/rosenbrock_trajectories.pdf` | four paths on Rosenbrock contours; **the roadmap plot** |
| `figures/bowl_trajectories.pdf` | four paths on the elongated bowl |
| `figures/step_size.pdf` | $\|x_T\|$ vs $\eta$, boundary marked (E6) |
| `figures/spectrum_fit.pdf` | histogram + Adam curve + SciPy curve |
| `figures/pathology_loss.pdf` | loss vs step for the failed run |

Paper grade: every axis labeled, with units on the spectrum plot (GeV,
counts). Commit `figures/reference/rosenbrock_trajectories.pdf`.

## Acceptance gate (from `03-Project-Roadmap.md`)

**All four converge where they should; Rosenbrock trajectory plot.**
Concretely:

- Fresh clone + `uv sync` + `uv run pytest -q` green + `uv run python run.py`
  writes the figures and prints the physics-fit table (Adam $M$ vs SciPy $M$
  vs truth).
- Bowl: all four methods within $10^{-4}$ of the origin after the Stage 3
  budget.
- Rosenbrock: all four within $10^{-3}$ of $(1,1)$; the trajectory PDF is
  in the repo (and in `figures/reference/`).
- Physics: Adam and `curve_fit` agree on $M$ to 20 MeV; pathology diagnosed
  in the writeup, not shrugged off.

Then the month sign-off (week README / syllabus §9): tag `month-02-complete`,
write the 250-word `retro.md` in the Month 02 folder, and open one issue for
the single biggest thing you don't yet understand.

## Writeup requirements

A `README.md` in the `optim_race/` repo — one screen: what this is (the
four updates, the two surfaces, the $\rho$-like peak); how to run it (the
three commands); the result (who won Rosenbrock, Adam vs SciPy on $M$ and
$\Gamma$); **what failed first** — written the day it happened (a sign
error in `rosenbrock_grad`? Adam without the $1/(1-\beta^t)$ looking like
a damped GD? the far-$M$ init sitting still?); and what you would do next.
Negative results go in, per the syllabus honesty policy.

## Stretch goals

- **SGD on the spectrum.** Minibatch the bins (or the unbinned masses) and
  overlay SGD's noisy path on the Adam path in $(M,\Gamma)$ space.
- **A third surface.** $f(x,y)=\tfrac12\kappa x^2 + \tfrac12 y^2$ with
  $\kappa$ swept; plot iterations-to-tol vs $\kappa$ for GD vs Adam.
- **Match `torch.optim.Adam`.** One step, then 50, on a random $g$ sequence;
  your `adam_update` vs PyTorch's, same $\beta$, same $\varepsilon$,
  `atol=1e-6`. (PyTorch is allowed here as a reference, not a dependency of
  `src/optim.py`.)
- **Heavy-ball vs EMA momentum.** Implement the classical
  $v_t = \mu v_{t-1} + g_t$ form and show it is the EMA form with a
  rescaled $\eta$.
