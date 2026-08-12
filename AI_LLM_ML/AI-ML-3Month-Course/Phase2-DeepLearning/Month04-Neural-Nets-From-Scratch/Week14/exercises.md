# Week 14 — Exercises

Work top to bottom. Setup (imports, seeds, the toy 2D dataset, timing helpers)
is given by the notebook; you write only the lines each exercise asks for.
E1–E3 build `micrograd.py` in this week's folder (the acceptance checks import
it); E4–E6 live in the notebook.

Note: this week necessarily uses Python classes — the one construct
`NOTEBOOK_RULES.md` §3 bans that an autograd engine cannot live without.
`lesson.md` §2 teaches everything needed; keep to plain classes with `__init__`
and methods, nothing fancier.

Paper first: the local derivative of each primitive op (`+`, `*`, `pow`, `tanh`,
`exp`, `relu`) written as "grad out = local × grad in"; the two-path argument for
why gradients add; and the full hand trace of `L = (a*b + c).tanh()` at
`a=2, b=-3, c=5`. File the pages in this folder — E1 checks the engine against
your numbers.

## E1 — The `Value` class

In `micrograd.py`, implement `Value` with `data`, `grad`, `__add__`, `__mul__`,
`tanh`, and `backward()` (topological sort + local backward rules).
Hint: build it in the lesson's order — forward ops first, then `_local_backward`,
then `_collect`/`backward`. Don't add more ops until this passes.
Accept when: gradients on the hand-traced expression match your paper numbers
(`dL/da = -1.2600`, `dL/db = 0.8400`, `dL/dc = 0.4200`) to 1e-4.

## E2 — Gradient accumulation bug hunt

Evaluate `b = a + a` then `b.backward()`. Predict `a.grad` on paper before
running.
Hint: how many times does `_local_backward` on the `+` node touch `a`?
Accept when: `a.grad == 2.0` and a one-line note names the exact line of your
code (the `+=`) that makes it so.

## E3 — Finite-difference harness

Extend `Value` with `__pow__`, `exp`, and `relu`, then write
`check_op(f, x, h)` that compares the engine's gradient of `f` at `x` against a
central finite difference, and run it on every op at several random points.
Hint: relative error is `|num - eng| / (|num| + |eng| + 1e-12)`; avoid checking
`relu` at points near 0, where the kink makes finite differences lie.
Accept when: all relative errors < 1e-4 with h = 1e-5.

## E4 — An MLP on top of the engine

Using the lesson's `Neuron`/`Layer`/`MLP` pattern, train a 2-16-1 tanh network
with plain SGD on the given subsample of the Week 13 toy 2D dataset (labels
recoded to ±1, squared-error loss).
Hint: the worked example's loop is the template; the dataset is bigger than XOR,
so expect a few hundred steps and try lr around 0.05.
Accept when: the loss curve, smoothed over 10 steps, decreases monotonically
over the last 50 steps, and train accuracy (sign of the output) > 90%.

## E5 — Print the graph

Write `show_graph(v)` that prints, in topological order, one line per node:
its op (or "leaf"), its `data`, and its `grad`, for the E1 expression after
`backward()`.
Hint: `_collect` already builds the ordered list — reuse it.
Accept when: the node count and ordering match a hand count of the §1 graph
(6 nodes: three leaves, `*`, `+`, `tanh`).

## E6 — Timing reality check

Time one full forward+backward of E4's network over the whole dataset, and time
the Week 13 NumPy net doing the same on the same data; report the ratio.
Hint: `time.perf_counter()` around each; run each a few times and take the best.
Accept when: the measured slowdown is reported (expect roughly 100–10000×) and
one sentence names the reason — per-scalar Python-object overhead vs whole-matrix
BLAS calls.

## Review

1. (Week 13) Re-derive `∂L/∂z = p − y` for softmax + cross-entropy — cold, no
   notes.
2. (Week 06/07) Reverse-mode autodiff computes vector–Jacobian products. For
   `f: Rⁿ → R`, what is the Jacobian's shape, and why does reverse mode need only
   one pass?
3. (Week 08) Your engine plus which Week 08 optimizer gives you Adam? List the
   per-parameter state it would need to carry.
4. (Week 09) You tuned lr in E4 by watching the training loss. What would you
   additionally need before claiming the model generalizes?
