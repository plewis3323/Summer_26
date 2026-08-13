# Week 05 — Calculus for ML

Machine learning is "write down how wrong you are, then walk downhill." This
week builds the mathematics of downhill from scratch — no calculus background
assumed — and ends with you implementing gradient descent.

## Objectives

- Define a function, its graph, and slope; turn a secant into a tangent via the
  difference quotient and a limit.
- Compute derivatives by the power, product, and chain rules, and check them
  with a central-difference numerical derivative.
- Form partial derivatives and the gradient; show (numerically, then
  geometrically next week) that the gradient points uphill.
- Run gradient descent, including breaking it on purpose with a too-large step.
- Fit a two-parameter detector calibration by minimizing mean squared error
  with GD, and match `np.polyfit`.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text; work through it at the
  keyboard, typing every snippet.
- 3Blue1Brown, *Essence of Calculus* (free on YouTube) — the week's spine.
  Watch chapters 1–4 alongside §§2–4 of the lesson; watch the chain-rule
  chapter twice.
- Khan Academy, Algebra II / Precalculus — only if §§1–2 of the lesson felt
  fast. A weekend here is cheaper than being lost for a month.

## Derivations (paper first)

- Difference quotient → derivative for $x^2$, $x^3$, $1/x$, and a polynomial.
- Chain rule on $(3x+1)^2$ and $(x^2+1)^3$, naming inner and outer functions.
- Both partials of $f(x,y) = x^2 + 3xy$; the gradient at a stated point.
- The GD update on $f(x) = (x-3)^2 + 1$, including the exact per-step ratio
  $|1-2\eta|$ and the stability bound $\eta < 1$.
- $\partial L/\partial m$ and $\partial L/\partial b$ for mean squared error
  on a line $y = mE + b$.

## Exercises

See `exercises.md` (notebook built from it when the week starts, per
`NOTEBOOK_RULES.md`). Seven exercises: secants becoming tangents, a reusable
`num_deriv`, chain-rule checks, the gradient-points-uphill search, breaking GD
on purpose, the zig-zag valley, and a six-point detector calibration by GD.

## Deliverable

Completed notebook plus scanned paper derivations in this folder. Keep
`num_deriv` — every derivative you derive this month gets checked with it.

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
