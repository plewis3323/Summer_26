# Week 13 — MLPs + Backprop on Paper

Backprop is the chain rule organized as matrix products — the same bookkeeping as error
propagation through a multi-stage detector calibration, run in reverse.

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

- *Understanding Deep Learning* (Prince, free PDF), Ch. 3 (Shallow neural networks) and
  Ch. 4 (Deep neural networks); skim Ch. 5 (Loss functions) for cross-entropy from MLE —
  this should feel like Week 07 again.
- 3Blue1Brown, *Neural networks* series, chapters 1–4 (you have notes on these; re-watch
  chapter 4 with the paper derivation in hand).
- Goodfellow et al., *Deep Learning*, Ch. 6 (Deep Feedforward Networks) — read §6.5 on
  back-propagation closely; treat the rest as reference.
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

## Exercises (built when the week starts)

1. Forward pass of a 2-layer MLP in NumPy with explicit shape asserts at each layer.
   Accept when: output shape is `(batch, classes)` and rows of softmax sum to 1 within 1e-6.
2. Softmax/cross-entropy gradient implemented from your derivation. Accept when: matches
   central finite differences to relative error < 1e-5 on random inputs.
3. Full backward pass for the 2-layer net. Accept when: all four parameter gradients
   match finite differences to relative error < 1e-4.
4. Train the net with your Week-08 SGD on a 2D two-moons-style toy set. Accept when:
   decision boundary plot shows > 95% train accuracy.
5. Universal approximation demo: fit a 1-hidden-layer net to a Breit–Wigner curve at
   widths 2, 8, 64. Accept when: plot shows fit quality improving with width and a
   one-line note on what width does not fix.
6. Break it: replace ReLU with identity and re-train. Accept when: check confirms the
   deep net collapses to the accuracy of plain logistic regression.

## Deliverable

Scanned paper derivations in the week folder + a NumPy notebook whose backward pass
passes all finite-difference checks and trains on a toy problem.

## Review

- (Week 06) Write the SVD of a weight matrix. What does a rank-1 `W` do to every input?
- (Week 07) Derive cross-entropy from maximum likelihood for a categorical model — the
  same three lines as this week's loss.
- (Week 08) Why does SGD noise help escape saddle points where full-batch GD stalls?
- (Week 10) Logistic regression's gradient was also `(p − y)x`. Why is that not a
  coincidence?
