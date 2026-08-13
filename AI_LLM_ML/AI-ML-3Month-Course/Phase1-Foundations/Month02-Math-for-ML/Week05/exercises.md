# Week 05 — Exercises

Work top to bottom in the notebook — all exercises are notebook cells this week
(per `NOTEBOOK_RULES.md` §6; nothing goes in files or the terminal). Setup gives
the imports, a seeded generator `rng = np.random.default_rng(7)`, and the
constants each exercise names; you write only the lines asked for. Derivatives
asked for "on paper" go in your derivations folder — photograph them in.

## E1 — Watch the secant become the tangent

For `f(x) = x**2` at `x0 = 3.0`, compute the difference quotient
`(f(x0 + h) - f(x0)) / h` for `h` in `[1.0, 0.1, 0.01, 0.001, 0.0001]`, printing
`h` and the slope each time. Then compute it once more at `h = 1e-15` and note in
a one-line comment what went wrong.
Hint: the lesson's §3.1 loop; the exact secant slope for this function is
$2x_0 + h$, so you can predict every row before running.
Accept when: each printed slope matches $2x_0 + h$ to 1e-9, and the `h = 1e-15`
value differs from 6 by more than 0.1 (rounding error has taken over).

## E2 — Your numerical-derivative workhorse

Write `num_deriv(f, x)` using the central difference with `h = 1e-5` (lesson
§3.4). On paper, differentiate: $x^3$, $1/x$, $\sqrt{x}$, and $3x^4 - 2x + 7$.
Check all four with `num_deriv` at the points $x = 2, 2, 4, 1$ respectively.
Hint: expected values, in order: 12, −0.25, 0.25, 10 — if paper and code
disagree, trust the code and redo the paper.
Accept when: all four `num_deriv` results match your paper formulas to 1e-6.
Keep `num_deriv` — every derivative you derive this month gets checked with it.

## E3 — Chain rule, trusted

On paper, differentiate $f(x) = (3x+1)^2$ and $g(x) = (x^2+1)^3$ by the chain
rule, naming the inner and outer functions for each. Verify both with
`num_deriv` at $x = 2$ and $x = 1$.
Hint: expected values 42 and 24; the second needs the power rule on the outside
and rides the inner derivative $2x$ along as a factor.
Accept when: both paper formulas match `num_deriv` to 1e-6 and the markdown cell
names inner and outer functions for each.

## E4 — The gradient points uphill

For `f2(x, y) = x**2 + 3*x*y` at `(x0, y0) = (1.0, 2.0)`: compute both partial
derivatives by central differences; then brute-force search 721 angles `theta`
in `[0, 2π]` for the unit step `(cos θ, sin θ)` with the largest rate of
increase `dfdx*cos(θ) + dfdy*sin(θ)`, and compare the winning angle to
`np.arctan2(dfdy, dfdx)`.
Hint: lesson §5.2 verbatim; the grid spacing is 2π/720 ≈ 0.0087 rad, so don't
expect agreement beyond that.
Accept when: the numerical partials match the paper values (8, 3) to 1e-5, and
the best angle agrees with the gradient's angle to within 0.01 rad.

## E5 — Break gradient descent on purpose

Minimize $f(x) = (x-3)^2 + 1$ from `x = 0.0` with 200 GD steps, once for each
`eta` in `[0.01, 0.1, 0.5, 0.9, 0.99, 1.01, 1.1]`. For each run record the
first-step ratio `abs(x1 - 3) / abs(x0 - 3)` and whether the final distance to
the minimum shrank or grew relative to the start.
Hint: lesson §7.1 derived the update as $x_{\text{new}} - 3 =
(1 - 2\eta)(x_{\text{old}} - 3)$, so the per-step ratio should be $|1 - 2\eta|$
exactly.
Accept when: every measured first-step ratio matches $|1 - 2\eta|$ to 1e-9, and
each run shrinks the distance exactly when $|1 - 2\eta| < 1$ (i.e. $\eta < 1$).

## E6 — The zig-zag valley

Minimize $f(x, y) = x^2 + 10y^2$ from `(8.0, 1.0)`: run 60 GD steps at
`eta = 0.09`, store the path, and overlay it on a contour plot (labeled axes).
Then rerun at `eta = 0.11` and describe in one comment line what the
$y$-coordinate does.
Hint: gradient $(2x, 20y)$; the loss is just `x**2 + 10*y**2` at the final
point. The steep direction sets the speed limit — Week 08's project derives the
exact threshold, which sits between your two `eta` values.
Accept when: at `eta = 0.09` the final loss is below 1e-6 and the contour plot
shows the zig-zag path; at `eta = 0.11` the `abs(y)` values grow step over step
while `x` still shrinks.

## E7 — Synthesis: calibrate a detector by gradient descent

Setup gives the six gamma-ray energies `E` from the lesson and pulse heights
`y` simulated with true slope 4.2, intercept 1.5, and noise 0.05 (seed 7).
Derive $\partial L/\partial m$ and $\partial L/\partial b$ for the mean squared
error on paper (lesson §8), then run GD (`eta = 0.1`, 3000 steps) from
`m = b = 0`. Record the loss every step, plot the loss curve, and compare your
$(m, b)$ with `np.polyfit(E, y, 1)`.
Hint: `np.polyfit(E, y, 1)` returns the exact best-fit `[m, b]` — Week 06 shows
where that exact answer comes from; residuals vectorized as `m*E + b - y`.
Accept when: your GD $(m, b)$ matches `np.polyfit` to 1e-6, both land within
0.1 of the true (4.2, 1.5), and the loss curve is non-increasing over all 3000
steps.

## Review

1. (Wk 03) What is the shape of `A[:, None, :] - A[None, :, :]` for `A` of
   shape `(N, 3)`, and what pairwise physics quantity might that structure
   compute?
2. (Wk 04) You refactored a function this week and silently changed its
   behavior — which kind of test catches that, and when should it have been
   written?
3. (Wk 04) What three things did the dimuon pipeline pin to make reruns
   identical?
4. (Wk 02) Write, from memory, the loop pattern that accumulates a running
   total over a list — the same skeleton your E7 gradient-descent loop used.
