# Week 17 — Convolutions

A convolutional layer is a translation-equivariant linear map — the symmetry argument
a physicist would make: if photon showers look the same anywhere on a detector face
(the lesson explains this picture), the model shouldn't relearn them per location.

## Objectives

- Implement 2D convolution from scratch in NumPy (loops) and match
  `torch.nn.functional.conv2d`.
- Do the shape arithmetic by hand: output size from
  `out = floor((in + 2p − k)/s) + 1`, receptive field, and parameter count
  `C_out (C_in k² + 1)` for any stack of conv/pool layers — both formulas derived in
  `lesson.md`.
- Explain convolution as a weight-shared, sparsely-connected special case of a fully
  connected layer, and quantify the parameter savings.
- Trace the LeNet → AlexNet → VGG → ResNet lineage and say what problem each step
  solved.
- Show, via the gradient of a residual block, why skip connections keep deep nets
  trainable.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: convolution ground-up, the output-size
  and parameter-count derivations, receptive fields, and the residual-gradient
  derivation, ending in a worked LeNet on FashionMNIST.
- CS231n course notes, "Convolutional Networks" — do every shape calculation in it by
  hand.
- *Understanding Deep Learning* (Prince), Ch. 10 (Convolutional networks) and Ch. 11
  (Residual networks).
- He et al., "Deep Residual Learning for Image Recognition" (arXiv:1512.03385) — read
  §1, §3; skim results.
- Optional: Goodfellow et al., Ch. 9 (Convolutional Networks) for the equivariance
  formalism.

## Derivations (paper first)

- Output-size formula by window-counting (re-derive lesson §3 cold).
- Convolution as matrix multiplication: write the sparse, weight-tied matrix for a
  3×3 kernel on a 5×5 input; count shared parameters vs the dense equivalent.
- Receptive field of three stacked 3×3 convs = one 7×7, with the parameter-count
  comparison (the VGG argument).
- Gradient of a residual block: for `y = x + F(x)`, show
  `∂L/∂x = ∂L/∂y (I + ∂F/∂x)` and why the identity term is a gradient highway
  through depth.

## Exercises

See `exercises.md` (notebook generated when the week starts). Six exercises: a NumPy
convolution matched against PyTorch, a shape/parameter calculator, hand-designed
kernels on FashionMNIST, a trained LeNet, a plain-vs-skip depth experiment on
CIFAR-10, and a CNN-vs-MLP parameter audit.

## Deliverable

NumPy convolution passing parity checks, paper derivations scanned into the folder,
and a trained LeNet-class model with its architecture arithmetic documented by hand.

## Review

- (Week 13) Universal approximation says an MLP could learn any image function. Why
  build in translation equivariance anyway? (Answer in terms of sample efficiency.)
- (Week 16) Which initialization does a ReLU conv layer want, and what is `n_in` for a
  3×3×64 kernel?
- (Week 06) A convolution is a linear map. What is its null space for a pure edge
  filter — which images does it send to zero?
