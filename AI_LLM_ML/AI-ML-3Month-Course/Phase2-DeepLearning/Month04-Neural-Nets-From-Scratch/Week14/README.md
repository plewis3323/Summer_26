# Week 14 — micrograd: A Scalar Autograd Engine

Last week you ran the chain rule by hand; this week you make the computer do the
bookkeeping — a computational graph is just a DAG of operations with the chain rule
walked in reverse topological order. This is also the week the course first needs a
Python class; `lesson.md` teaches classes from zero before using them.

## Objectives

- Read and write a minimal Python class: `__init__`, attributes, methods, `self`, and
  operator methods (`__add__`, `__mul__`).
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

- `lesson.md` (this folder) — the primary text: computational graphs, Python classes
  from zero, the complete `Value` engine with `if/elif` backward dispatch, reverse vs
  forward mode, and a worked example training an MLP on XOR.
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

## Exercises

See `exercises.md` (notebook generated from it when the week starts). Six exercises:
the `Value` class checked against the paper trace → the gradient-accumulation bug hunt
→ a finite-difference harness for every op → an MLP trained on the Week 13 toy data →
graph printing → a timing comparison against the NumPy net.

## Deliverable

A self-contained `micrograd`-style module (~100 lines) + notebook: all finite-difference
checks green, a trained MLP, and the paper trace it was validated against.

*(Note: the class-free rule in `NOTEBOOK_RULES.md` is relaxed this week by necessity — an
autograd engine is a class. Keep it to plain classes with `__init__` and methods, nothing
fancier; flag this in the notebook's first cell.)*

## Review

- (Week 13) Re-derive `∂L/∂z = p − y` for softmax/cross-entropy cold — no notes.
- (Week 06/07) Reverse-mode autodiff computes vector–Jacobian products. For `f: Rⁿ → R`,
  what is the Jacobian's shape and why does reverse mode need only one pass?
- (Week 08) Your engine + which Week-08 optimizer gives you Adam? List the state each
  parameter would need to carry.
