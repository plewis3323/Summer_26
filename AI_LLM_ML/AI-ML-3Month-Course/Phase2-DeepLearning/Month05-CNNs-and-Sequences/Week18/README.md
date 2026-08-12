# Week 18 — Modern Vision Training

State-of-the-art vision practice is mostly regularization and reuse: augment what data
you have, borrow features from someone else's million-GPU-hour run, and know which
knobs matter — the analogue of not re-deriving a detector's response when a measured
parametrization already exists (the lesson unpacks this).

## Objectives

- Choose augmentations by asking which invariances the label actually has (a flipped
  sneaker is still a sneaker; a vertically-flipped one may not be a valid example).
- Explain dropout as noise injection / implicit ensembling, and why it is off at eval.
- Fine-tune a pretrained ResNet: replace the head, choose what to freeze, use
  discriminative learning rates.
- Decide between linear probe, partial fine-tune, and full fine-tune based on dataset
  size and domain gap — and articulate the domain gap for detector images vs ImageNet.
- Run a data-efficiency study: accuracy vs training-set size, from-scratch vs
  transferred.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: augmentation and the label-invariance
  test, dropout with the weight-scaling argument, transfer learning and the
  probe/partial/full decision, ending in a worked probe-then-fine-tune on CIFAR-10.
- CS231n course notes, "Neural Networks Part 2" (data preprocessing, dropout,
  regularization) and the transfer-learning section.
- *Understanding Deep Learning* (Prince), Ch. 9 (Regularization) — read the dropout
  and data-augmentation sections closely.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting" — §1–2 for the idea, skim experiments.
- PyTorch transfer-learning tutorial (official) for the mechanics of freezing and
  head-swapping.

## Derivations (paper first)

*(Light derivation week.)*

- Dropout at eval: show that scaling by the keep-probability matches the expected
  activation over dropout masks for a linear layer (and why it's only exact there).

## Exercises

See `exercises.md` (notebook generated when the week starts). Six exercises: an
augmentation gallery with label-invariance verdicts, a deliberate overfit fixed by
augmentation and dropout, a linear probe, a full fine-tune with discriminative LRs, a
from-scratch-vs-transfer data-efficiency curve, and grayscale images through an RGB
backbone.

## Deliverable

A fine-tuned classifier notebook + the data-efficiency plot, plus a short note: your
recipe (freeze policy, LRs, augmentations) for Week 20's project, decided in advance.

## Review

- (Week 16) Dropout and batchnorm interact awkwardly. What does each assume about the
  statistics of its inputs at eval time?
- (Week 09) Augmentation reduces which term in the bias–variance decomposition, and
  what tacit assumption breaks if the augmentation isn't label-preserving?
- (Week 12) You met unsupervised structure in Week 12. Why might pretrained features
  transfer to detector images at all, given ImageNet contains no calorimeters?
