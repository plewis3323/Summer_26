# Week 13 — MLPs + Backprop on Paper

Backprop is the chain rule organized as matrix products — the same bookkeeping as error
propagation through a multi-stage detector calibration, run in reverse. This is the
flagship derivation of the course; `lesson.md` walks it scalar-first, then in matrix form.

## Objectives

- Write the forward pass of an L-layer MLP as a sequence of affine maps + nonlinearities,
  with every shape annotated.
- State the universal approximation theorem precisely and explain why it guarantees
  nothing about learnability or sample efficiency.
- Derive the softmax/cross-entropy gradient and show the famous simplification to
  `p − y`.
- Derive full backprop for a 2-layer net on paper: every `∂L/∂W`, `∂L/∂b`, and the
  backward recursion for the layer errors.
- Implement that exact derivation as NumPy forward/backward passes and verify against
  finite differences.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: neuron → layer → MLP, universal
  approximation and its fine print, the full softmax/cross-entropy and 2-layer backprop
  derivations, and a runnable NumPy worked example with finite-difference checks.
- *Understanding Deep Learning* (Prince, free PDF), Ch. 3 (Shallow neural networks) and
  Ch. 4 (Deep neural networks); skim Ch. 5 (Loss functions) for cross-entropy from MLE —
  this should feel like Week 08 again.
- 3Blue1Brown, *Neural networks* series, chapters 1–4 (re-watch chapter 4 with the paper
  derivation in hand).
- Goodfellow et al., *Deep Learning*, Ch. 6 — read §6.5 on back-propagation closely;
  treat the rest as reference.
- CS231n course notes, "Backpropagation, Intuitions" module.

## Derivations (paper first)

- Softmax + cross-entropy: show `∂L/∂z = p − y` for one-hot `y`, from scratch, including
  the softmax Jacobian.
- Full backprop for a 2-layer MLP (affine → ReLU → affine → softmax → CE): the four
  gradients `∂L/∂W2, ∂L/∂b2, ∂L/∂W1, ∂L/∂b1`, with shapes checked at every line.
- The general backward recursion `δ^(l) = (W^(l+1)ᵀ δ^(l+1)) ⊙ σ'(z^(l))` for arbitrary
  depth.
- Why gradients of matrix-valued layers batch cleanly: derive the batched forms
  `∂L/∂W = δ xᵀ` summed over the batch.

## Exercises

See `exercises.md` (notebook generated from it when the week starts). Six exercises:
forward pass with shape asserts → gradient implementations checked against finite
differences → training on two moons → a universal-approximation demo on a Breit–Wigner
curve → a break-it experiment removing the nonlinearity.

## Deliverable

Scanned paper derivations in the week folder + a NumPy notebook whose backward pass
passes all finite-difference checks and trains on a toy problem.

## Review

- (Week 06) Write the SVD of a weight matrix. What does a rank-1 `W` do to every input?
- (Week 08) Derive cross-entropy from maximum likelihood for a categorical model — the
  same three lines as this week's loss.
- (Week 08) Why does SGD noise help escape saddle points where full-batch GD stalls?
- (Week 10) Logistic regression's gradient was also `(p − y)x`. Why is that not a
  coincidence?
