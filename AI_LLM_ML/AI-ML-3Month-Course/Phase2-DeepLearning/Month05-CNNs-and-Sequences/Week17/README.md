# Week 17 — Convolutions

A convolutional layer is a translation-equivariant linear map — the same symmetry
argument you'd make in physics: if photon showers look the same anywhere on the EMCal
face, the model shouldn't relearn them per tower.

## Objectives

- Implement 2D convolution from scratch in NumPy (loops, then im2col) and match
  `torch.nn.functional.conv2d`.
- Do receptive-field and shape arithmetic by hand: output size, receptive field, and
  parameter count for any stack of conv/pool layers, from the standard formula
  `out = (in + 2p − k)/s + 1`.
- Explain convolution as a weight-shared, sparsely-connected special case of a fully
  connected layer, and quantify the parameter savings.
- Trace the LeNet → AlexNet → VGG → ResNet lineage and say what problem each step solved.
- Show, via the gradient of a residual block, why skip connections keep deep nets
  trainable.

## Core material (~3 hrs)

- CS231n course notes, "Convolutional Neural Networks: Architectures, Convolution /
  Pooling Layers" (the spine — do every shape calculation in it by hand).
- *Understanding Deep Learning* (Prince), Ch. 10 (Convolutional networks) and Ch. 11
  (Residual networks).
- He et al., "Deep Residual Learning for Image Recognition" (arXiv:1512.03385) — read
  §1, §3; skim results.
- Optional: Goodfellow et al., Ch. 9 (Convolutional Networks) for the equivariance
  formalism.

## Derivations (paper first)

- Convolution as matrix multiplication: write the (doubly Toeplitz-structured) matrix
  for a 3×3 kernel on a 5×5 input; count shared parameters vs the dense equivalent.
- Receptive field of three stacked 3×3 convs = one 7×7, with the parameter-count
  comparison (the VGG argument).
- Gradient of a residual block: for `y = x + F(x)`, show `∂L/∂x = ∂L/∂y (I + ∂F/∂x)`
  and why the identity term is a gradient highway through depth.
- Backward pass of a conv layer: show that the gradient w.r.t. the input is itself a
  convolution (with the flipped kernel).

## Exercises (built when the week starts)

1. Loop-based 2D convolution (stride, padding). Accept when: matches
   `F.conv2d` to 1e-5 on random inputs for three (k, s, p) settings.
2. Shape/receptive-field calculator for a list of layer specs. Accept when: hand-checked
   against three architectures including LeNet, all values agree.
3. Edge/blur/Sobel filters applied to a toy shower image. Accept when: plots show the
   expected responses and one line links Sobel to "learned features, layer 1".
4. LeNet-style CNN on MNIST (or FashionMNIST) using the Week-15 loop. Accept when:
   > 98% test accuracy (MNIST) inside 5 epochs on CPU/modest GPU.
5. Depth experiment: plain 20-layer convnet vs same-depth ResNet on CIFAR-10 subset.
   Accept when: training curves reproduce the plain-net degradation and the skip-net fix.
6. Parameter audit: count parameters of Exercise 4's CNN vs an MLP with the same input
   and accuracy target. Accept when: printed counts match `sum(p.numel())` and the ratio
   is stated.

## Deliverable

NumPy convolution passing parity checks, paper derivations scanned into the folder, and
a trained LeNet-class model with its architecture arithmetic documented by hand.

## Review

- (Week 13) Universal approximation says an MLP could learn any image function. Why
  build in translation equivariance anyway? (Answer in terms of sample efficiency.)
- (Week 16) Which initialization does a ReLU conv layer want, and what is `n_in` for a
  3×3×64 kernel?
- (Week 05) A convolution is a linear map. What is its null space for a pure edge
  filter, physically?
