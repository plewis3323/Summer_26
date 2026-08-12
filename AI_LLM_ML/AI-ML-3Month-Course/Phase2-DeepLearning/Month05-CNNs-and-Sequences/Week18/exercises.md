# Week 18 — Exercises

Work top to bottom. Setup (imports, data loading, subset construction, constants) is
given by the notebook; you write only the lines each exercise asks for. All exercises
live in notebook cells this week. GPU recommended for E3–E5 (Colab/Kaggle free tier
is enough).

## E1 — Augmentation gallery

For one CIFAR-10 image, build a 3×4 grid: the original plus 11 augmented variants
covering horizontal flip, vertical flip, small rotation (±15°), 90° rotation, random
crop with padding, and brightness jitter (transforms given in setup; you compose and
apply them).
Hint: call the transform repeatedly — each call redraws the randomness.
Accept when: the grid renders, and each transform has a one-line verdict in a markdown
cell: label-preserving for CIFAR-10 or not, with the reason.

## E2 — Overfit on purpose, then fight back

Train the given small CNN on 500 CIFAR-10 images three ways with the identical loop
and epochs: (a) no regularization, (b) + augmentation (flip + crop), (c) + augmentation
and dropout on the dense layer. Plot train and validation loss curves for all three.
Hint: run (a) long enough to see validation loss turn upward while training loss keeps
falling — that hook is the point.
Accept when: one figure shows all three train/val gaps, and a markdown line ranks the
three by final validation accuracy.

## E3 — Linear probe

Freeze the provided pretrained ResNet-18, replace `fc` with a 10-class head, and train
only the head on 1,000 CIFAR-10 images.
Hint: lesson §5 and §7 — freeze with `requires_grad_(False)` before swapping the head;
pass only `model.fc.parameters()` to the optimizer.
Accept when: printed trainable-parameter count is 5,130, and probe test accuracy beats
E2's best from-scratch result.

## E4 — Full fine-tune with discriminative LRs

Unfreeze the E3 model and continue training with backbone LR 1e-4 and head LR 1e-3
(two optimizer parameter groups).
Hint: build the backbone parameter list by excluding the head's parameter ids, as in
lesson §5.
Accept when: test accuracy beats the E3 probe, and frozen/trainable parameter counts
before and after unfreezing are printed.

## E5 — Data-efficiency curve (synthesis)

For N_train in {100, 300, 1000, 3000}: train the small from-scratch CNN (with E2's
best regularization) and the fine-tuned ResNet, and plot test accuracy vs N_train for
both on one axis (log-x).
Hint: reuse your E2 and E4 code as functions of the subset size; keep epochs fixed
across N so the comparison is fair.
Accept when: the plot exists with both curves labeled, and one markdown line states
where (or whether) from-scratch catches up — this number is your Week 20 planning
input.

## E6 — Grayscale through an RGB backbone

Push FashionMNIST (1-channel) through the pretrained ResNet-18 two ways: (a) replicate
the channel to 3 in the transform; (b) replace `conv1` with a fresh 1-channel
`nn.Conv2d(1, 64, 7, stride=2, padding=3)`. Linear-probe both on 1,000 images.
Hint: for (a) the provided transform stack ends with a channel-replication step; for
(b) only `conv1` and `fc` should be trainable.
Accept when: both runs complete, both accuracies are printed, and a 2-line markdown
cell says which won and why that is (or isn't) surprising given lesson §6.

## Review

1. (Week 16) Dropout and batchnorm both behave differently in `train()` vs `eval()`.
   State what each does differently at eval time.
2. (Week 09) Augmentation reduces which term in the bias–variance decomposition, and
   what tacit assumption breaks if the transform isn't label-preserving?
3. (Week 10) The linear probe trains a softmax classifier by cross-entropy. Write the
   loss for one example and name each symbol.
4. (Week 12) Why might features learned on photographs transfer to detector images at
   all, given ImageNet contains no detectors?
