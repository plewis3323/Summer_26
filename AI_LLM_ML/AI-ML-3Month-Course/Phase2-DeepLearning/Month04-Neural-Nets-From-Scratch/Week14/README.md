# Week 14 — micrograd: A Scalar Autograd Engine

Last week you ran the chain rule by hand; this week you make the computer do the
bookkeeping — a computational graph is just a DAG of operations with the chain rule
walked in reverse topological order.

## Objectives

- Build a scalar `Value` object that records its inputs and local derivatives, forming a
  computational graph as expressions are evaluated.
- Implement reverse-mode autodiff: topological sort + backward accumulation, including
  correct gradient accumulation when a value is used twice.
- Explain why reverse mode is the right choice for scalar losses over many parameters
  (and when forward mode would win).
- Verify every operation's gradient against central finite differences, and know why the
  check is central, not forward, differences.
- Train a small MLP built entirely from your engine.

## Core material (~3 hrs)

- Karpathy, *Neural Networks: Zero to Hero* — "The spelled-out intro to neural networks
  and backpropagation: building micrograd" (the spine; ~2.5 hrs; build alongside, don't
  just watch).
- Read the micrograd source on GitHub after your own attempt, not before — diff your
  design against his.
- CS231n notes on backpropagation (revisit the "staged computation" / local-gradient
  view — that is exactly what the engine automates).

## Derivations (paper first)

- For each primitive op (`+`, `*`, `pow`, `tanh`, `exp`, `relu`): the local derivative,
  written as "grad flowing out = local × grad flowing in".
- The multivariate chain rule for a node used by two downstream paths — show why
  gradients add, and connect it to the `+=` in your backward pass.
- Hand-trace backprop through `L = (a*b + c).tanh()` on paper; you will check the engine
  against these numbers.

## Exercises (built when the week starts)

1. `Value` class with `+`, `*`, `tanh`, and `backward()`. Accept when: gradients on the
   hand-traced expression above match your paper numbers to 1e-6.
2. Gradient accumulation bug hunt: evaluate `b = a + a; b.backward()`. Accept when:
   `a.grad == 2` and a one-line note says which line of code makes it so.
3. Finite-difference harness: check every op's gradient at random points. Accept when:
   all relative errors < 1e-4 with h = 1e-5.
4. `Neuron`/`Layer`/`MLP` on top of `Value`; train on the Week-13 toy 2D dataset with
   plain SGD. Accept when: loss curve decreases monotonically over the last 50 steps
   smoothed, and train accuracy > 90%.
5. Graph visualization: print the DAG (topological order, op names, grads) for a small
   expression. Accept when: node count and ordering match a hand count.
6. Timing reality check: compare your scalar engine vs the Week-13 vectorized NumPy net
   on the same problem. Accept when: measured slowdown is reported and one sentence names
   the reason (Python-object overhead vs BLAS).

## Deliverable

A self-contained `micrograd`-style module (~100 lines) + notebook: all finite-difference
checks green, a trained MLP, and the paper trace it was validated against.

*(Note: the class-free rule in `NOTEBOOK_RULES.md` is relaxed this week by necessity — an
autograd engine is a class. Keep it to plain classes with `__init__` and methods, nothing
fancier; flag this in the notebook's first cell.)*

## Review

- (Week 13) Re-derive `∂L/∂z = p − y` for softmax/cross-entropy cold — no notes.
- (Week 06) Reverse-mode autodiff computes vector–Jacobian products. For `f: Rⁿ → R`,
  what is the Jacobian's shape and why does reverse mode need only one pass?
- (Week 08) Your engine + which Week-08 optimizer gives you Adam? List the state each
  parameter would need to carry.
