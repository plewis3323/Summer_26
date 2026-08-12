# Week 18 — Modern Vision Training

State-of-the-art vision practice is mostly regularization and reuse: augment what data
you have, borrow features from someone else's million-GPU-hour run, and know which knobs
matter — the ML analogue of not re-deriving detector response when a testbeam
parametrization exists.

## Objectives

- Choose augmentations by asking which invariances the label actually has (a rotated
  shower is still a shower; a flipped ADC-vs-time trace may not be valid).
- Explain dropout as noise injection / implicit ensembling, and why it is off at eval.
- Fine-tune a pretrained ResNet: replace the head, choose what to freeze, use
  discriminative learning rates.
- Decide between linear probe, partial fine-tune, and full fine-tune based on dataset
  size and domain gap — and articulate the domain gap for detector images vs ImageNet.
- Run a data-efficiency study: accuracy vs training-set size, from-scratch vs
  transferred.

## Core material (~3 hrs)

- CS231n course notes, "Neural Networks Part 2" (data preprocessing, dropout,
  regularization) and the transfer-learning section.
- *Understanding Deep Learning* (Prince), Ch. 9 (Regularization) — read the dropout and
  data-augmentation sections closely.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting" — §1–2 for the idea, skim experiments.
- PyTorch transfer-learning tutorial (official) for the mechanics of freezing and
  head-swapping.

## Derivations (paper first)

*(Light derivation week.)*

- Dropout at eval: show that weight scaling by keep-probability p matches the expected
  activation over dropout masks for a linear layer (and why it's only exact there).

## Exercises (built when the week starts)

1. Augmentation gallery on toy EMCal shower images: flips, small rotations, tower-level
   noise, energy smearing. Accept when: grid plot renders and each augmentation has a
   one-line physics justification (or rejection).
2. Overfit on purpose: small CNN, 500 training images, no regularization — then add
   augmentation, then dropout. Accept when: three train/val-gap curves are plotted and
   ranked.
3. Linear probe: frozen pretrained ResNet as a feature extractor + logistic-regression
   head (Week 10's classifier reborn) on a small image dataset. Accept when: probe beats
   a from-scratch small CNN with < 1k training images.
4. Full fine-tune of the same ResNet with a fresh head and lower backbone LR. Accept
   when: beats the linear probe, and the frozen/unfrozen parameter counts are printed.
5. Data-efficiency curve: accuracy vs N_train ∈ {100, 300, 1k, 3k} for from-scratch vs
   fine-tuned. Accept when: the crossover (or its absence) is plotted and stated in one
   line.
6. Grayscale detector-style images through an RGB-pretrained backbone: adapt the input
   layer two ways (channel replication vs conv1 re-init). Accept when: both run and the
   accuracy difference is reported.

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
