# Week 01 — Teacher explainer

**Read this when you are stuck, not before.**

`Week01_Solutions.ipynb` shows *what* the answer is. This shows *why*, from first
principles, and — more usefully — *how you could have got there yourself*. Every section
follows the same shape:

> **The question behind the question** — what the exercise is really testing
> **From first principles** — the derivation, assuming nothing
> **How to unstick yourself** — the move to make when you are staring at it
> **The mistake everyone makes**

If you are stuck right now, the fastest general advice in this whole document is the
three-line loop at the end of §0. Try that before reading the section for your exercise.

---

## §0 — How to be stuck productively

Being stuck is the job. The difference between an hour of learning and an hour of
frustration is entirely in what you do while stuck.

**The three-line loop.** When you do not know why something is wrong:

1. **Print the shape, dtype and first three values** of every array in the expression.
   Not the values you think are there — the ones that are. More than half of NumPy bugs
   are visible in a `print(x.shape, x.dtype)` and invisible in an hour of reasoning.
2. **Shrink the input** until you can compute the answer by hand. A `(3, 5)` array of
   `np.arange(15)` is a debugger. A 475 465-row DataFrame is not.
3. **Test an invariant, not the answer.** You often cannot check that `m_inv` is right.
   You can always check that $p_x^2 + p_y^2 = p_T^2$, that a centered array has zero mean,
   that a probability sums to 1. Find the thing that must be true and assert it.

**The thing that separates scientific programming from other programming** is that your
code can be wrong in ways that produce plausible numbers. A web app that is broken shows a
blank page. A physics analysis that is broken shows a beautiful plot with the peak in the
wrong place. Nothing but your own scepticism catches that — which is why every exercise
this week ends in a check against something independently known.

**On using this document.** Read the section for the exercise you are stuck on. Read it
until you have an *idea*, then stop reading and go try the idea. Reading to the end of a
section you were only 30 seconds from solving yourself is the most expensive thing you can
do with it.

---

## §1 — E1: broadcasting is one rule, applied over and over

### The question behind the question

Not "can you call `np.mean`". It is: **do you know what shape everything is, at every
step, without running it?** Every NumPy bug you will hit this month is a shape bug wearing
a costume.

### From first principles

NumPy's broadcasting rule, in full, is three lines:

1. If the two arrays have different `ndim`, **left-pad the smaller shape with 1s**.
2. Two axes are compatible if they are **equal**, or **one of them is 1**.
3. A length-1 axis is **stretched** (conceptually — no memory is copied) to match.

Everything else follows. Work an example by hand, `(1000, 4)` against `(4,)`:

```
  (1000, 4)
       (4,)   ->  left-pad  ->  (1, 4)
  ---------
  (1000, 4)                     compatible: 4==4, and 1 stretches to 1000
```

and now `(1000, 4)` against `(1000,)`:

```
  (1000, 4)
    (1000,)   ->  left-pad  ->  (1, 1000)
  ---------
      4 vs 1000                 neither equal nor 1  ->  ValueError
```

The padding is on the **left**, which means the trailing axes align. That is the whole
source of the asymmetry that makes E2 a trap: reducing `axis=0` produces a shape that
happens to align correctly, and reducing `axis=1` produces one that does not.

`keepdims=True` sidesteps the question entirely: reducing `axis=k` with `keepdims` gives a
shape identical to the input except that axis `k` is 1 — which is compatible with the
input by construction, on every axis, for every shape.

**So the rule to internalise is:** if you are about to combine a reduction with the array
it came from, use `keepdims=True`. Not because the other way is always wrong, but because
the other way is *conditionally* right, and you do not want your correctness to be
conditional on a shape you did not choose.

### Sub-parts

**E1.2, (pt, eta, phi) → (px, py).** Colliders use cylindrical coordinates around the beam
axis. $p_T$ is the momentum in the transverse plane, $\phi$ the azimuthal angle in it, so
$p_x = p_T\cos\phi$, $p_y = p_T\sin\phi$ — polar to cartesian, nothing more. $\eta$ is
longitudinal and does not appear (it enters $p_z = p_T\sinh\eta$).

Two things worth knowing beyond the formula:

- **Verify with an invariant.** $p_x^2 + p_y^2 = p_T^2$ must hold identically, for every
  event, to machine precision. That is a much stronger check than looking at three numbers.
- **`np.arctan2(y, x)`, never `np.arctan(y/x)`.** `arctan` has range $(-\pi/2, \pi/2)$ and
  cannot distinguish quadrant II from IV — $(-1,-1)$ and $(1,1)$ give the same answer.
  `arctan2` takes both arguments and returns the full $(-\pi, \pi]$.
- **Why $\eta$ and not the polar angle $\theta$?** Because differences in
  $\eta = -\ln\tan(\theta/2)$ are invariant under boosts along the beam, and you never know
  the longitudinal boost of the colliding partons. Coordinates are chosen so that the
  quantities you measure are the ones that do not depend on what you cannot measure.

**E1.3, "rank-2 matrix".** Two different words that sound the same:

- `M.ndim` — the number of **axes**. This is what `exercises.md` means.
- `np.linalg.matrix_rank(M)` — the dimension of the column space.

$M_{ij} = \sin(i)\cos(j)$ **factorises**: row $i$ is $\sin(i)$ times the same vector
$\cos(j)$. Every column is a multiple of $\sin(i)$, so the column space is
one-dimensional and the matrix rank is 1 — always, for any outer product. Two axes, rank 1.
If you want a genuine matrix rank 2 you need a sum of two independent outer products, which
is exactly what a rank-$k$ decomposition (and, later, PCA and truncated SVD) is about.

**E1.4, Frobenius vs 2-norm.** $\|A\|_F$ flattens the matrix and takes the Euclidean
length — it does not care about matrix structure at all. $\|A\|_2$ does: it is
$\max_{\|v\|=1}\|Av\|$, the most the matrix can stretch any unit vector. In singular
values,

$$\|A\|_F = \sqrt{\textstyle\sum_k \sigma_k^2}, \qquad \|A\|_2 = \max_k \sigma_k$$

so they agree exactly when there is only one nonzero singular value — a rank-1 matrix.
`np.linalg.norm(A)` defaults to Frobenius for 2D input and to the 2-norm for 1D input,
which is a genuinely confusing default. Pass `ord=` explicitly and the confusion vanishes.

### How to unstick yourself

Write the two shapes on paper, right-aligned, one under the other. The rule is then a
visual check, not a memory test.

### The mistake everyone makes

Reaching for `.reshape(1, 4)` instead of `keepdims=True`. They give the same array. But
`keepdims` says *"I reduced this axis and kept it"*, and `reshape` says *"trust me"* —
and reshape will silently do something insane if the size ever changes, while `keepdims`
cannot.

---

## §2 — E2: why the test is the exercise

### The question behind the question

Not "can you subtract a mean". It is: **can you write a test that would have caught the bug
you did not know you had?**

### From first principles

Take `x` of shape `(3, 5)`:

| you write | mean shape | broadcast | result |
|---|---|---|---|
| `x - x.mean(axis=1)` | `(3,)` | `(3,5)` vs `(1,3)` | **ValueError** |
| `x - x.mean(axis=1, keepdims=True)` | `(3,1)` | stretch axis 1 | correct |
| `x - x.mean(axis=0)` | `(5,)` | `(3,5)` vs `(1,5)` | correct **by luck** |
| `x - x.mean(axis=0, keepdims=True)` | `(1,5)` | stretch axis 0 | correct, and says so |

Now take `x` of shape `(5, 5)`. Row 1 of that table no longer raises: `(5,)` aligns
against the last axis, so NumPy subtracts the vector of *row* means from every *column*.
Right shape. No warning. Wrong answer.

**That is the entire lesson of E2, and it generalises far beyond this function:** a bug
that raises on some inputs and silently corrupts on others is worse than a bug that always
raises. Square arrays, unit batch sizes, and equal-length sequences are where broadcasting
bugs hide, because those are the shapes that make wrong code shape-legal.

### What makes the test good

1. **Parametrized over `(3,5)`, `(5,3)`, `(1,7)`.** These are chosen, not arbitrary: two
   non-square shapes in both orientations so the bug cannot hide behind symmetry, plus a
   degenerate one where an axis has length 1.
2. **Asserts a property, not a value.** "Every row mean is ~0" is true regardless of the
   random input. A test that hard-codes expected numbers tests your arithmetic and not your
   logic, and breaks every time you touch the data.
3. **`atol`, not `rtol`.** The target is 0. Relative tolerance against 0 is meaningless:
   `|a - 0| <= rtol * |0|` is `|a| <= 0`.
4. **Tests non-mutation.** `x -= x.mean(...)` modifies the caller's array in place. That
   is a bug that only shows up in the *second* function to touch the data, which is the
   worst possible place to find it.

### How to unstick yourself

Ask, out loud: *"which axis am I reducing, and what shape must the result be for the
subtraction to broadcast the way I intend?"* Two sentences. If you can answer them the code
writes itself; if you cannot, no amount of trying variants will help.

### The mistake everyone makes

Testing on `np.arange(25).reshape(5, 5)`. It passes against the buggy implementation.

---

## §3 — E3: invariant mass, and floating point that bites

### The question behind the question

Two things. First, can you translate a physics formula into vectorized pandas. Second, and
much more important: **when your answer disagrees with the reference, can you work out
whether the fault is yours?**

### From first principles: why $M^2 = E^2 - |\vec p|^2$

Special relativity gives every particle a four-momentum
$p^\mu = (E, p_x, p_y, p_z)$ (units with $c = 1$). Under a Lorentz boost the components
mix, exactly as the components of a 3-vector mix under rotation. Rotations preserve
$x^2+y^2+z^2$; boosts preserve the **Minkowski** norm, which has one sign flipped:

$$m^2 \equiv E^2 - p_x^2 - p_y^2 - p_z^2$$

Every observer, however they are moving, computes the same $m$. That is what "invariant"
means, and it is why this quantity is the backbone of collider physics: it does not depend
on the reference frame you happened to measure in.

Four-momenta **add** (energy and momentum are conserved), so a particle that decayed into
two muons had $p^\mu = p_1^\mu + p_2^\mu$, and its mass is

$$M^2 = (E_1+E_2)^2 - \|\vec p_1 + \vec p_2\|^2$$

**Sum the components first, then take the norm.** Masses do not add:
$M \neq m_1 + m_2$. This is why a histogram of $M$ over pairs of muons shows *peaks* — at
the mass of every particle heavy enough to decay to $\mu^+\mu^-$ — sitting on a smooth
background of pairs that did not come from a common parent.

### Why your `m_inv` will not match CMS's `M` — and why that is not your bug

`np.allclose(m_inv, M, atol=1e-3)` is `False`. Roughly 93% of events agree to a milli-GeV,
99.8% to 10 MeV, and a handful are off by up to 0.8 GeV. Your algebra is fine.

The CSV stores six significant figures. Take a real event from this file: two muons with
$E \approx 1600$ GeV forming a 7.8 GeV pair. Then

$$M^2 = \underbrace{(E_1+E_2)^2}_{\approx 2.6\times10^6} - \underbrace{\|\vec p_1+\vec p_2\|^2}_{\approx 2.6\times10^6} = 61$$

Six significant figures on a number of order $10^6$ is an absolute uncertainty of a few
units. A few units of absolute error in a quantity whose true value is 61 is a few percent
— and that is *before* the square root. The **relative** error exploded even though every
input was stored to the same relative precision.

That is **catastrophic cancellation**, and the general statement is worth memorising:

> Subtracting two nearly equal numbers destroys relative precision. The absolute error
> survives the subtraction unchanged; the value does not.

It is a property of the subtraction, not of your code, and it is why numerical analysts
rewrite expressions to avoid differences of large quantities (the classic being the
quadratic formula's $-b + \sqrt{b^2-4ac}$).

### Why the "obvious fix" is worse — and the actual lesson

The natural response is to rearrange so the big subtraction never happens. For muons
($m_\mu \approx 0$):

$$M^2 = 2p_{T1}p_{T2}\left[\cosh(\eta_1-\eta_2) - \cos(\phi_1-\phi_2)\right]$$

Measured on this file, this form is **worse**: median error $2\times10^{-3}$ against
$5\times10^{-5}$, worst case ~292 GeV against ~0.8 GeV. Why? $\cosh$ grows exponentially,
so at large rapidity separation it amplifies the six-significant-figure error in $\eta$ far
harder than the subtraction ever amplified the error in $E$. Sort by disagreement and the
worst offenders are exactly the large-$|\Delta\eta|$ events, which is the prediction made
concrete.

There is no algebraic rearrangement that wins, because **the information is not in the
file**. Six significant figures is the ceiling and every formula pays for it somewhere.
The real fix is upstream: read the ROOT file (E6), where the doubles were never
round-tripped through text — median disagreement there is $8\times10^{-11}$ GeV, six orders
of magnitude better, with the *same* formula.

The transferable lesson is not about muons:

> When you rearrange a formula for numerical reasons, you are moving the amplification, not
> removing it. **Measure both forms.** Anyone who tells you which is better without
> measuring is guessing.

### Two small things that matter

**`np.clip(m2, 0, None)` before `np.sqrt`.** For a nearly-massless pair, rounding can push
$M^2$ a hair below zero; `np.sqrt(-1e-13)` is `nan`, and one `nan` poisons every `.mean()`
downstream. Clipping asserts "this is rounding noise around a genuinely zero mass" — which
is true *here*. Always print how many events you clipped before believing it; if it is
thousands rather than four, the clip is hiding a real bug.

**Log-spaced bins in E3.5.** The spectrum spans 0.2–120 GeV. With linear bins every
low-mass resonance lands in one or two bins and vanishes. `np.logspace` gives constant
*fractional* width, which is also roughly how detector resolution scales — so the physics
and the display agree.

### How to unstick yourself

Your peaks are the calibration. ρ/ω at 0.78, φ at 1.02, J/ψ at 3.097, ψ(2S) at 3.686,
Υ at 9.46, Z at 91.19. If the peaks land in the wrong places your `m_inv` is wrong, and if
they land in the right places it is right. That single plot is a better test than any
amount of re-reading the algebra.

### The mistake everyone makes

Using the file's `M` column because your own disagrees. You would be "verifying" with the
number you copied.

---

## §4 — E4: the residual panel is the point

### The question behind the question

Not "can you make two panels". It is: **can you make a figure that would let a reader catch
you being wrong?**

### From first principles: why pulls, not differences

A raw residual $N - f$ is in units of events. In a bin with 10 000 events a discrepancy of
200 is nothing ($\sqrt{10^4} = 100$, so it is 2σ); in a bin with 40 events, 200 is
catastrophic. Plotting raw residuals therefore requires the reader to mentally divide by
$\sqrt{N}$ in every bin, which nobody does.

The **pull** does it for them:

$$\text{pull}_i = \frac{N_i - f_i}{\sigma_i}, \qquad \sigma_i = \sqrt{N_i}$$

It is unitless, so a horizontal line at 0 and a band at ±1 make the panel self-calibrating:
about 68% of points inside the band, essentially all within ±3, no visible structure. Any
of those failing is a message.

And the messages are specific. Scattered points, no pattern → the model is fine. A coherent
S-shape → the peak position or width is off. Points high at the peak and low on both
shoulders → your peak shape is too narrow. **Structure in the residuals is the model
telling you what it is missing.**

### What this actually caught here

Fitting the J/ψ with a single Gaussian plus a line gives $\chi^2/\mathrm{ndof} \approx 11$.
Not marginal — badly wrong — and completely invisible in the top panel, where the fit looks
like it goes through the points.

Two physical reasons, both worth knowing:

1. **The observed peak is 100% detector resolution.** The J/ψ's natural width is 92.6 keV;
   we measure ~30 MeV, three hundred times wider. So the peak *shape* is a question about
   CMS, not about charmonium. And CMS's muon resolution varies strongly with $\eta$ and
   $p_T$, so the spectrum summed over the detector is a *sum of Gaussians of different
   widths* — which is not a Gaussian. It is a narrow core with fat tails. Two Gaussians
   sharing a mean takes $\chi^2/\mathrm{ndof}$ from 11 to 2.8.
2. **What is left is asymmetric, on the low side.** A muon that radiates a photon before
   being measured loses energy, so final-state radiation drags reconstructed mass *down*
   and never up. No sum of symmetric Gaussians can absorb a one-sided tail; the standard
   tool is a **Crystal Ball** function — Gaussian core spliced to a power-law tail.

You could have read all that in a textbook. You *found* it in a residual panel, which is a
different and much more durable kind of knowing.

### The subtlety that bites everyone: integrate the model over the drawn bins

The fit used 60 linear bins of 16.7 MeV. The spectrum is drawn in 300 log-spaced bins, ~67
MeV wide near the J/ψ. To overlay the fit you must convert to counts *per drawn bin*, and
the tempting move — multiply the curve by the width ratio — assumes the model is flat
across a bin. A $\sigma = 32$ MeV Gaussian inside a 67 MeV bin is not remotely flat, so
the curve overshoots the peak by tens of percent and looks like a broken fit.

The correct operation is to treat the fit as a **density** and integrate it over each drawn
bin. For this model the integral is analytic:

$$\int_a^b Ne^{-(x-\mu)^2/2\sigma^2}dx = N\sigma\sqrt{\tfrac{\pi}{2}}
\left[\operatorname{erf}\!\left(\tfrac{b-\mu}{\sqrt2\sigma}\right)-
\operatorname{erf}\!\left(\tfrac{a-\mu}{\sqrt2\sigma}\right)\right]$$

with the line contributing $a(b-a) + \tfrac{b}{2}(b^2-a^2)$. The general principle:
**a histogram shows integrals; a model function gives densities. Converting between them is
your job, and it is only a multiplication when the bins are uniform and narrow.**

### How to unstick yourself

If the curve does not sit on the data and you cannot see why, plot the fit in *its own*
binning first. If it fits there and not in the display binning, your problem is bin
bookkeeping, not the fit.

### The mistake everyone makes

Believing the top panel. It is the panel designed to look convincing.

---

## §5 — E5: the pull distribution, and what an error bar means

### The question behind the question

**Do you know what your error bar is claiming?** Anyone can get a fit to converge. The
exercise is checking whether the number after the ± means anything.

### From first principles: what `curve_fit` does

`curve_fit` minimises

$$\chi^2(\theta) = \sum_i \left(\frac{y_i - f(x_i;\theta)}{\sigma_i}\right)^2$$

by Levenberg–Marquardt, which is a **local** method: it walks downhill from `p0` to the
nearest minimum. Not the best minimum — the nearest one. That is why `p0` matters, and why
you should build it out of the data (`counts.max()` for the amplitude, `np.median(counts)`
for the background level) rather than out of hope.

It returns `(popt, pcov)`. `pcov` is the covariance **of the fitted parameters**, obtained
from the inverse of the approximate Hessian at the minimum. So:

- `np.sqrt(np.diag(pcov))` gives 1σ parameter errors;
- the off-diagonals give correlations, which are not decoration — a strong $\mu$/background
  correlation means quoting one without the other is meaningless;
- it is a **local, Gaussian** approximation: exactly right for a linear model with Gaussian
  errors, approximately right otherwise, and badly wrong at a parameter boundary.

### The three flags that decide whether the errors are honest

**`sigma=np.sqrt(counts + 1)`.** Bin counts are Poisson: variance = mean, so the error on a
bin with $N$ entries is $\sqrt{N}$. But $\sqrt{0} = 0$ is an error of zero, i.e. **infinite
weight** — and the fit will distort itself arbitrarily to pass exactly through every empty
bin (or divide by zero outright). The `+1` floors the error at 1. It is a pragmatic
regulariser, not a theorem: the principled fix for low counts is a Poisson **likelihood**
fit rather than a $\chi^2$ fit built on a Gaussian approximation that fails precisely when
$N$ is small. That is exactly the log-likelihood E9 makes you write.

**`absolute_sigma=True`.** By default (`False`!) `curve_fit` treats your `sigma` as
*relative* weights and rescales `pcov` by the reduced $\chi^2$ — i.e. it infers the overall
error scale from how well the fit went. That is right when you know the shape of your
errors but not their scale. It is wrong for Poisson counts, whose scale is known a priori.
On a good fit ($\chi^2/\mathrm{dof}\approx1$) the two agree to a per cent; on a bad fit
they diverge, and the default silently *absorbs your model's badness into your error bars*,
making a wrong model look precise.

**`abs(sigma)`.** The Gaussian depends on $\sigma$ only through $\sigma^2$. So $-\sigma$ is
an exactly equivalent minimum and the optimiser returns whichever side it drifted into.
Take the absolute value or bound the parameter — otherwise you have written a test that
fails on a seed you never tried.

### The pull distribution: why it is the first plot a statistician asks for

$$\text{pull} = \frac{\hat\mu - \mu_{\text{true}}}{\hat\sigma_{\hat\mu}}$$

Run many toys and histogram it. If the fit is unbiased **and** its covariance is right,
this is exactly $\mathcal{N}(0,1)$ — and that is the point: it tests the estimate and its
uncertainty *simultaneously*, which no single fit can do.

| observation | diagnosis |
|---|---|
| mean ≠ 0 | biased fit: wrong model, binning artefact, or boundary effect |
| width > 1 | errors **under**-estimated — your significances are inflated |
| width < 1 | errors **over**-estimated — you are throwing away sensitivity you paid for |
| fat tails | the fit occasionally fails badly and you are not noticing |

A width of 1.4 means every uncertainty you quote is 40% too small, so your "5σ discovery"
is really 3.6σ. This is not pedantry; it is the difference between a result and a
retraction.

### Read the acceptance criterion before you chase your tail

E5 accepts at "pull mean within 0.05, width within 0.1, for 100 toys". But the sample mean
of 100 unit-Gaussian pulls has standard error $1/\sqrt{100} = 0.10$ — **twice the
tolerance**. A flawless fit fails that criterion roughly 60% of the time. The width is
tighter than it looks too: $\mathrm{SE}[\hat\sigma] \approx 1/\sqrt{2(N-1)} = 0.071$.

Measured: at $N=100$, mean $+0.049$, width $1.097$ — a coin flip against the stated bounds.
At $N=500$ the same fit gives mean $-0.021$, width $1.030$, comfortably inside. Nothing
changed but the number of toys.

This is a genuinely important habit, so it is worth saying plainly:

> **Before you debug a failing tolerance, work out the statistical error on the quantity
> being compared to it.** If the tolerance is smaller than the noise, the test is telling
> you about your sample size, not your code.

### How to unstick yourself

Plot the fit on top of the histogram *before* you look at a single fitted number. Nine
times out of ten a bad pull distribution is visible as an obviously misplaced curve on a
single toy.

### The mistake everyone makes

Chasing a pull mean of 0.049 at 100 toys. It is noise. Run 500 and it goes away.

---

## §6 — E6: file formats, and where precision goes to die

### The question behind the question

**Where does your data actually come from, and what did the trip cost?**

### From first principles

A ROOT file is a self-describing, compressed, columnar container: a directory of named
objects, where a `TTree` is a table whose columns ("branches") are stored as compressed
baskets. Columnar matters — reading one branch of a hundred does not touch the other
ninety-nine, which is why HEP files are laid out this way and why Parquet, Arrow and every
modern analytics format converged on the same idea.

`uproot` implements that format in pure Python + NumPy. No ROOT, no C++, no compiler.

Two details that confuse everyone once:

- **`events;1`** — the `;1` is a *cycle number*. ROOT keeps previous versions of an object
  when it is rewritten. `f["events"]` gives the highest cycle, which is what you want.
- **`library=`** — `"pd"` gives a DataFrame, `"np"` a dict of arrays, `"ak"` an Awkward
  array. Use `"ak"` when branches are **jagged** (a variable number of jets per event):
  that does not fit in a rectangular DataFrame at all, and forcing it will either explode
  or silently flatten your event structure away.

### The point of step 4

"Confirm one scalar calculation matches" is the actual exercise. Recompute the invariant
mass from the four-vectors and compare against the file's own `M` branch. Median
disagreement: $8\times10^{-11}$ GeV — full double precision, because the numbers were
never round-tripped through text.

Now put that next to E3's CSV: same formula, same physics, median $5\times10^{-5}$ and a
worst case of 0.8 GeV. **Six orders of magnitude, bought by not writing your data to a CSV.**

That is E3's aside paying off. The fix for catastrophic cancellation was never algebraic —
it was "do not throw away your significant figures on the way in". CSV is a fine interchange
format and a terrible storage format for measurements.

### The mistake everyone makes

Trying to satisfy "3–5 branches" and the invariant-mass check with one load. The mass
cross-check needs eight branches (two four-vectors). They are two different loads, and that
is normal: pull a small frame to look at, pull exactly what you need to compute with.

---

## §7 — E7: a test is a specification

### The question behind the question

**Can you write a test for behaviour that does not exist yet?**

Two of the three tests confirm what the code already does. The third — `test_empty_input_raises` —
fails against the code as written. That is not a broken exercise, it is the exercise. The
test *specifies*; the code then has to comply.

### From first principles: why the empty case is nasty

You might expect `fit_pi0_peak(np.array([]), edges)` to blow up immediately. It does not.
`np.histogram([])` does not raise — it returns an array of zeros, which is a perfectly
well-formed histogram of nothing. So `curve_fit` receives a well-shaped optimisation
problem containing no information, and fails three frames down inside MINPACK with a message
about function evaluations, or converges on nonsense.

The general principle:

> **Validate at the boundary, where you still know what the caller meant.** Errors get
> harder to diagnose the further they travel from their cause.

### Why `test_fit_positive_sigma` is not trivial

It looks like a tautology. It is a real test, because the model uses $\sigma^2$ only, so
$-\sigma$ is an exactly equivalent minimum and which one you get depends on the data. It can
pass on your seed and fail on a colleague's.

Which raises the thing that actually matters about tests in scientific code: **a test that
fails one run in twenty is worse than no test**, because it trains your team to ignore red.
So pin your seeds, and — since a pinned seed only tests the seeds you picked — run a loop
over ten of them when the property is supposed to hold universally.

### On tolerance-based asserts

`abs(fit["mu"] - 0.135) < 3 * fit["mu_err"]` is a statistical statement, not a vibe. If the fit is
unbiased with correct errors, it fails 0.3% of the time by construction. Pinning the seed
converts that probabilistic claim into a deterministic one. The alternatives — a wider
tolerance, or more events — trade sensitivity for stability, and it is worth knowing which
one you chose.

### The mistake everyone makes

Editing `fit_pi0_peak` first and writing the test afterwards. You lose the only chance to
see the test fail — and a test you have never seen fail is not yet known to test anything.

---

## §8 — E8: the point is the baseline, not the model

### The question behind the question

**Do you know what your metric is worth?**

### From first principles

**`stratify=y`** forces the split to preserve class proportions. On iris (three classes, 50
each) it barely matters. On a rare-signal problem, an unstratified split can hand you three
signal events in the test set, and every number you compute afterwards is noise. Build the
habit where it is free.

**Accuracy vs balanced accuracy.** Accuracy is the fraction of all events classified
correctly. Balanced accuracy is the mean of the **per-class recalls** — every class counts
equally regardless of size. On iris they are nearly identical, which is precisely why this
is where you should learn the difference: on a 99%-background HEP sample, a classifier that
answers "background" always scores 99% accuracy and 50% balanced accuracy. One of those
numbers tells you it is useless.

**The confusion matrix** is the object both numbers are summaries of. Rows are truth,
columns are prediction, so entry $(i,j)$ is "how often class $i$ was called class $j$";
accuracy is $\mathrm{trace}/\mathrm{total}$. Read the off-diagonals: *which* classes get
confused is a physics question with a physics answer (here, versicolor and virginica
overlap in petal length — they are genuinely similar flowers).

**Fit on two features to draw a 2D boundary.** A boundary from a 4-feature model projected
into 2D is a lie: the model is using information that is not on the page, so points will sit
on the "wrong" side of a line that is not actually its line. The model has to live in the
space you are drawing.

**And notice the boundaries are straight.** Logistic regression is linear in feature space
— it can only carve the plane with lines. Where classes overlap it cannot do better, ever.
That is not a failure, it is a **baseline**: the number any more complicated model has to
beat before its complexity has earned anything.

### The mistake everyone makes

Concluding something about machine learning from iris. It is 150 rows of nearly separable
measurements from 1936. It is a smoke test for your code, not evidence about a method.

---

## §9 — E9: reproducibility is a property you can lose silently

### The question behind the question

**If someone runs your code in a year, do they get your number — and can they prove it?**

### From first principles: seeding

A pseudo-random generator is a deterministic function of its state. Seed it and the whole
sequence is fixed. The guarantee "same seed → same result" therefore holds only if **every**
draw comes from that generator.

The failure mode is nastier than it sounds. A stray `np.random.uniform()` in a fit's
starting point does not make results *obviously* random — it makes them *nearly* identical,
because the fit mostly converges to the same place. So the JSONs differ in the eighth
decimal and you do not notice for a month, by which time you cannot tell which of your
plots came from which code.

Hence: `rng = np.random.default_rng(seed)`, passed explicitly down the call stack. Explicit
passing makes the dependency **auditable** — you can read a function signature and know
whether it can be random. The legacy `np.random.seed()` sets a *global*, which any library
you call can draw from and disturb, and which nothing in a signature reveals.

When a helper takes a seed rather than a generator (like `make_pi0_toy`), derive its seed
*from* your generator — `rng.integers(0, 2**32-1)` — instead of inventing a second
independent source of randomness.

### `git_sha`, and why `-dirty` matters more than the SHA

A result is a claim about the output of a specific program. The SHA is the only compact
identifier of which program that was. But if the working tree has uncommitted changes, the
SHA names a commit whose code **is not the code that ran** — so it invites a reader (or
you, in six months) to reproduce and get a different answer with no indication why. That is
strictly worse than logging nothing.

`git status --porcelain` is empty exactly when the tree is clean, so appending `-dirty`
otherwise is three lines and converts a confident lie into an honest warning.

The same logic keeps going: same code and same seed only reproduce given the same library
versions, which is why the solution also logs `numpy.__version__` and the platform. What
you are recording is not "the answer" but "everything needed to get the answer again".

### `perf_counter` vs `time`

`time.time()` is wall-clock: it can jump backwards when NTP corrects the system clock, and
it has coarse resolution. `time.perf_counter()` is monotonic and high resolution, but its
zero point is arbitrary — it is only meaningful as a difference. So: `time.time` for *when*,
`perf_counter` for *how long*.

### The design fix the exercise forces

E9 asks you to log a log-likelihood. To compute one you need the fitted model evaluated at
the bin centres — so a `fit_pi0_peak` that returned only `mu`, `sigma` and two yields would
make the task impossible. That is why the dictionary it hands back also carries `popt` and
`pcov`.

That is worth noticing as a general move: **when an exercise seems to ask for something
impossible, check whether the data structure is the thing that is wrong.** Two extra keys
in the returned dictionary make a whole class of downstream questions answerable.

(And when you write the likelihood: clip the expectation away from zero. The model is a
Gaussian plus a *line*, and a line will happily go negative where no data constrains it.
`log(negative)` is `nan`, and one `nan` makes the whole sum `nan`.)

### The mistake everyone makes

Logging the seed and calling it reproducible, without ever running it twice and diffing.
Check the property you claim, mechanically, field by field — that is what the diff cell in
the solution notebook is for.

---

## §10 — E10: vectorization is about Python, not about math

### The question behind the question

**Do you know where the time actually goes?** Not "can you make it faster" — anyone can
delete a loop. Can you say *why* it got faster, and predict when it will not?

### From first principles

Both versions do the same arithmetic on the same IEEE doubles. NumPy's addition is not a
faster addition. What differs is the **per-operation overhead**:

The Python loop, once per event ($10^6$ times):
- interpret bytecode for the loop body,
- for each `a[k][i]`, allocate a fresh `np.float64` object wrapping one number,
- do the arithmetic,
- adjust reference counts, free the temporaries.

The vectorized version, once per **array operation** (about six times total):
- one C-level loop over contiguous memory, no boxing, no refcounting, and the memory access
  is sequential so hardware prefetching works.

The speedup is therefore roughly *the number of Python-level operations you eliminated*,
which is why it is measured in hundreds rather than in percentages — and it is also why the
ceiling exists: split the same calculation into 50 array operations over tiny arrays and the
overhead comes straight back. **Vectorization is about how many Python operations you
execute, not how much arithmetic you do.**

### Why 200× is the wrong thing to promise

`exercises.md` accepts at ≥200× and hints at ~500×. Measured here, across six honest
implementations of the same formula:

| implementation | s / 10⁶ events | × vs best |
|---|---|---|
| loop, `a[k][i]` on ndarrays | 1.39 | 106× |
| loop, `zip(a, b)` over ndarrays | 0.98 | 75× |
| loop, `zip(a, b)` over **Python lists** | 0.19 | 15× |
| vectorized, `s = a + b` then column ops | 0.023 | 2× |
| vectorized, column-wise, no `(N,4)` temporary | 0.017 | 1× |
| vectorized, `np.einsum('ij,ij,j->i', s, s, G)` | 0.013 | 1× |

The numerator spans a factor of **seven**, the denominator a factor of two. "The speedup" is
a ratio between two things you chose, so quoting 60× or 106× from the same physics on the
same machine in the same minute are both true.

The most surprising row is the third: iterating a NumPy array is **~5× slower** than
iterating an equivalent list of Python floats, because every `a[k][i]` allocates a boxed
`np.float64` that is immediately discarded, while a list already holds Python objects. The
fastest naive loop is the one that never touches NumPy — which is a good thing to know the
next time you write a loop "over an array" for convenience.

So: treat 200× as "the same order of magnitude as the reference machine". A genuine failure
looks like ~1× (you did not vectorize) or ~5–10× (you left a Python loop wrapped around
array operations).

### The `einsum` version, and why it is worth learning

```python
G = np.array([1.0, -1.0, -1.0, -1.0])         # Minkowski metric diag(+,-,-,-)
m = np.sqrt(np.maximum(np.einsum("ij,ij,j->i", s, s, G), 0.0))
```

`"ij,ij,j->i"` says: over events `i` and components `j`, multiply `s[i,j] * s[i,j] * G[j]`
and sum over everything not in the output — i.e. over `j`. That is
$s^\mu g_{\mu\nu} s^\nu$ written directly. It is the fastest row in the table because it
fuses the whole contraction into one pass with no intermediate arrays, and — more valuable
— it is the *definition* rather than a transcription of it.

### Why timing individual events is meaningless

One event takes tens of nanoseconds; `perf_counter`'s resolution and call overhead are of
the same order. Timing events individually measures the clock. Time **chunks** (200 chunks
of 5 000) and histogram the implied per-event time. That distribution is real, and its width
is cache behaviour, allocator noise and the OS scheduler.

Look at the relative widths: the *vectorized* distribution is the broader one in relative
terms, because it is fast enough that scheduler noise is a large fraction of the total. Fast
code is not just faster — it is noisier, which is its own lesson about benchmarking.

### The mistake everyone makes

Timing before checking that both versions agree. A fast wrong answer is worthless, and it is
easy to write a vectorized version that is subtly different (a missing `maximum` at 0, a
transposed axis) and only notice after you have quoted the speedup in a meeting.

---

## §11 — The five things worth carrying into Week 02

1. **Know the shape of everything.** Most numerical bugs are shape bugs, and `keepdims=True`
   is how you stop depending on an alignment rule you did not choose.
2. **Verify with an invariant, not an eyeball.** $p_T^2 = p_x^2+p_y^2$; the J/ψ is at
   3.097; a centered array has zero mean. Find what must be true and assert it.
3. **Residuals and pulls are how a fit talks back to you.** A fit prints numbers whether or
   not it fits. Only the residuals say whether to believe them, and only the pull
   distribution tests the error bar as well as the estimate.
4. **Before debugging a failing tolerance, compute the noise on the quantity.** Half the
   time the tolerance is tighter than the statistics allow, and you are about to spend an
   afternoon fixing your sample size.
5. **Record what it would take to get the number again** — seed, code SHA (with `-dirty`),
   library versions. A result you cannot regenerate is an anecdote.

---

## Appendix — Fast lookup

| Symptom | Almost always |
|---|---|
| `ValueError: operands could not be broadcast` | missing `keepdims=True` |
| Right shape, wrong values, square input | missing `keepdims=True`, and it did not raise |
| `nan` appearing after a `sqrt` | negative argument from float error → `np.clip(x, 0, None)` |
| `nan` in a log-likelihood | model went negative → clip the expectation, not the data |
| Fit disagrees with a reference in the 4th digit | precision of the input file, not your algebra |
| `curve_fit`: "Optimal parameters not found" | bad `p0` — build it from the data |
| Negative fitted `sigma` | model uses `sigma**2`; take `abs()` or bound it |
| Pull width > 1 | errors under-estimated; check `absolute_sigma` |
| Pull mean off by ~0.05 at 100 toys | noise. Run 500 |
| Fitted curve overshoots the histogram it fitted | you scaled by bin width instead of integrating over bins |
| Two runs with the same seed differ slightly | something random is not fed by your seed |
| "Speedup" is 5–10× not 100× | there is still a Python loop around your array ops |
