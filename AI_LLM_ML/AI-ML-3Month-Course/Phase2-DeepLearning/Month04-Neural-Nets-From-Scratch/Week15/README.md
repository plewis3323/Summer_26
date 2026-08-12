# Week 15 — PyTorch Fundamentals

PyTorch is your Week-14 engine with tensors instead of scalars and a decade of
engineering — this week is about mapping what you built onto what you'll use for the rest
of the course.

## Objectives

- Manipulate tensors fluently: creation, dtype/device, broadcasting (NumPy rules from
  Week 01 carry over), views vs copies, in-place ops and why autograd hates them.
- Explain what `requires_grad`, `backward()`, `.grad`, `no_grad()`, and `detach()` do in
  terms of the computational graph you built last week.
- Structure a model as an `nn.Module` and a training loop with `Dataset`/`DataLoader`,
  optimizer, and loss — the loop skeleton you will reuse all course.
- Rebuild the Week-14 MLP in PyTorch and confirm it matches your engine's gradients.
- Train a makemore-style character-level name model: bigram counts → one-hot + linear →
  MLP.

## Core material (~3 hrs)

- Karpathy, *Zero to Hero* — "The spelled-out intro to language modeling: building
  makemore" and "Building makemore Part 2: MLP" (the spine).
- Official PyTorch tutorials: "Learn the Basics" sequence (tensors, datasets, autograd,
  optimization) — skim fast; you already know the concepts, you're learning the API.
- *Understanding Deep Learning* (Prince), Ch. 6 (Fitting models) — connects the training
  loop back to Week 08's optimizers.

## Derivations (paper first)

*(Light week on paper — the derivation muscle rests before Week 16.)*

- Show that the bigram model's cross-entropy loss is minimized exactly by the empirical
  conditional frequencies (MLE for a categorical distribution — Week 07 redux).

## Exercises (built when the week starts)

1. Tensor drills: broadcasting shapes, views vs copies, one deliberate in-place-op
   autograd error. Accept when: all shape predictions written before running are correct
   and the error message is explained in one line.
2. Gradient parity: same 2-layer net, same weights, in your micrograd and in PyTorch.
   Accept when: all parameter gradients agree to relative error < 1e-5.
3. Port the Week-13 NumPy MLP to `nn.Module` + `DataLoader` + `optim.SGD` on the same toy
   data. Accept when: reaches the Week-13 accuracy in comparable epochs.
4. Bigram name model from counts; sample names from it. Accept when: model's average
   negative log-likelihood matches the count-based model's to 3 decimals.
5. One-layer neural bigram trained by gradient descent. Accept when: converges to the
   count model's loss within 0.01 nats, confirming Exercise 4's derivation numerically.
6. MLP character model (context length 3, embedding table). Accept when: validation loss
   beats the bigram model's, with train/val curves plotted from a proper split.

## Deliverable

A PyTorch training-loop template you actually understand line-by-line (it becomes the
seed of every later project) + a character model that generates plausible fake names,
with the micrograd parity check green.

## Review

- (Week 14) In your engine, where do gradients accumulate and why must PyTorch users
  call `zero_grad()`?
- (Week 07) The char model outputs a categorical distribution. Write its likelihood and
  show minimizing cross-entropy = maximizing likelihood.
- (Week 09) You split names into train/val. What specific failure would training on all
  names and reporting training loss hide?
- (Week 01) Which NumPy broadcasting rule explains why `(32, 27) + (27,)` works but
  `(32, 27) + (32,)` fails?
