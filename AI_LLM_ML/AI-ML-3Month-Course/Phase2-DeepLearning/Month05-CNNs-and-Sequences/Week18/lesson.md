# Week 18 — Modern Vision Training

~4 hrs. Before starting you should be able to: build and train a CNN and do its shape
arithmetic (Week 17); explain overfitting and the bias–variance trade-off (Week 09);
read training/validation curves and diagnose a broken run (Week 16); train a logistic
regression and interpret its outputs (Week 10).

## 1. The real bottleneck is data, not architecture

Week 17 gave you the architecture. This week is about the uncomfortable fact that
dominates vision practice: labeled images are scarce, and CNNs are hungry. A
ResNet-scale network has millions of parameters; your dataset — especially a
scientific one, where every label costs simulation time or expert eyes — may have a
few thousand images. Train a big net on a small set and you get Week 09's classic
failure: training loss near zero, validation loss climbing. The model has memorized
the training set instead of learning the pattern.

Modern vision training is a stack of countermeasures, and almost all of them are one
of two moves:

1. **Make the data bigger or the model noisier** — augmentation and dropout
   (sections 2–3).
2. **Don't start from scratch** — transfer learning: reuse features someone else
   trained on a million images (sections 4–6).

Neither changes the CNN math from Week 17. They change what the optimizer sees.

## 2. Data augmentation

**Data augmentation** means generating extra training examples by transforming the
ones you have, using transformations that do not change the label. Flip a photo of a
sneaker horizontally: still a sneaker, still labeled "sneaker", and the network now
has two examples instead of one. Common augmentations for photographs: horizontal
flips, small rotations, random crops (shift the framing), small color/brightness
jitter, and cutting out random patches.

The discipline is in one question: **is the label really invariant under this
transformation?** Get it wrong and you are training on mislabeled data you
manufactured yourself.

- Horizontal flip of a sneaker: fine. Vertical flip: a sneaker with the sky under its
  sole — the test set contains nothing like it; at best useless.
- Rotating an MNIST "6" by 180°: it becomes a "9". Label destroyed.
- Scientific data needs the same care with different answers. A calorimeter image
  (Week 20 explains these — a grid of energy deposits from a particle detector) can be
  flipped or rotated by 90° if the detector is symmetric under those operations, and
  tower-level noise can be added because real electronics are noisy. But stretching it
  would change the physical shower width — which may be exactly the feature that
  carries the label.

Why does augmentation help, in Week 09's language? It cuts variance: the model can no
longer latch onto accidents of individual training images (this sneaker's exact pixel
alignment) because each image keeps reappearing shifted and flipped. Equivalently, it
injects an invariance assumption the architecture doesn't already have — Week 17 built
in translation; flips and rotations you add through data. The cost: if the assumption
is false, you have added bias.

In code, augmentation is a transform pipeline applied on-the-fly by the `Dataset`, so
every epoch sees a different random variant. Augment *training* data only — validation
and test must stay fixed, or your metric moves for reasons that have nothing to do
with the model:

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),   # shift the framing by up to 4 px
    transforms.ToTensor(),
])
test_tf = transforms.ToTensor()
```

## 3. Dropout

**Dropout** attacks overfitting from inside the network. During training, each unit's
activation is set to zero with probability $p$ (typically 0.1–0.5), independently, at
every forward pass; surviving activations are scaled up by $1/(1-p)$ to keep the
layer's expected output the same. At evaluation, dropout does nothing — every unit is
active, no scaling.

Two ways to understand why this regularizes:

- **No co-adaptation.** A unit cannot rely on one specific partner unit being present,
  because that partner keeps vanishing. Features are forced to be individually useful,
  redundantly encoded — which memorized noise rarely is.
- **Implicit ensemble.** Each dropout mask defines a different thinned sub-network.
  Training with dropout trains this exponentially large family of sub-networks with
  shared weights; evaluating with all units active approximates averaging the family's
  predictions. Averaging many models is a classic variance-reduction move — Week 11's
  bagging argument, replayed inside one network.

The scaling rule has a small derivation worth doing once. Take a linear unit
$y = \sum_j w_j x_j$ and drop each input independently: $\tilde{x}_j = m_j x_j /(1-p)$
with $m_j \in \{0,1\}$, $P(m_j = 1) = 1-p$. Then, over the randomness of the mask,

$$\mathbb{E}[\tilde{y}] = \sum_j w_j\, \frac{\mathbb{E}[m_j]}{1-p}\, x_j
= \sum_j w_j x_j = y,$$

so evaluation with all units active matches the *expected* training-time activation.
The match is exact only for a linear layer — a nonlinearity does not commute with the
expectation ($\mathbb{E}[\text{relu}(z)] \ne \text{relu}(\mathbb{E}[z])$) — but it is
close enough in practice, and it is why eval mode simply switches dropout off.

Practicalities: in PyTorch, `nn.Dropout(p)` between layers; `model.train()` and
`model.eval()` toggle it (same switch that toggles batchnorm, Week 16 — forgetting
`model.eval()` before measuring is a classic silent bug). Dropout is most useful on
big dense layers; modern convnets use little or none inside conv stacks, where
batchnorm plus augmentation usually does the job. Dropout and batchnorm sit awkwardly
together: batchnorm's statistics are estimated under one noise level at train time and
used under another at eval; keeping dropout out of the conv/batchnorm stack sidesteps
that.

## 4. Transfer learning: don't start from scratch

Here is the most cost-effective idea in applied deep learning. Networks trained on
**ImageNet** — the standard benchmark corpus of ~1.28 million photos labeled in 1000
categories — learn general-purpose visual machinery on the way: first-layer edge and
color detectors (Week 17's Sobel filters, rediscovered), mid-layer textures and parts,
late-layer object shapes. The early and middle layers are barely specific to
ImageNet's classes at all. They are a reusable front end for *seeing*.

**Transfer learning** means taking such a pretrained network and re-purposing it for
your task. The standard surgery: chop off the final classification layer (the "head" —
it maps features to ImageNet's 1000 classes, which you don't care about), replace it
with a fresh layer sized for your classes, and then decide how much of the rest (the
"backbone") to update.

Why should features learned on photos of dogs transfer to your problem? Because the
low-level statistics of images — edges, corners, blobs, textures — are shared across
almost all image-like data. The further your data sits from photographs (the larger
the **domain gap**), the less of the late, task-specific machinery transfers; but the
early layers travel remarkably well, even to medical scans and detector images.

## 5. Three ways to reuse a backbone

Ordered by how much you let the optimizer touch:

**Linear probe.** Freeze the entire backbone (no gradient updates), use it as a fixed
feature extractor, and train only the new head. The head is one linear layer trained
by cross-entropy — literally Week 10's logistic/softmax regression, with pretrained
features instead of hand-built ones. Cheap, hard to overfit (you are fitting a few
thousand parameters), and a strong baseline: always run the probe first.

**Partial fine-tune.** Unfreeze the last block or two along with the head. The late
layers adapt to your domain; the early edge detectors stay put.

**Full fine-tune.** Unfreeze everything, but train the backbone with a much smaller
learning rate than the head (a "discriminative learning rate", e.g. backbone at 1e-4,
head at 1e-3). The pretrained weights are a good solution already; a large LR would
blast them apart before the head has settled — the pretraining destroyed by your own
optimizer. Small backbone steps let the features drift gently toward your data.

Freezing in PyTorch is a flag on each parameter; discriminative LRs are optimizer
parameter groups:

```python
for p in model.parameters():        # freeze everything
    p.requires_grad_(False)
model.fc = nn.Linear(512, 2)        # fresh head (requires_grad defaults True)

# later, for full fine-tune with discriminative LRs:
for p in model.parameters():
    p.requires_grad_(True)
head_params = list(model.fc.parameters())
backbone_params = [p for p in model.parameters() if id(p) not in [id(q) for q in head_params]]
opt = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3},
])
```

**Which mode when?** Two axes decide: how much labeled data you have, and how big the
domain gap is.

| | small dataset (≲ a few k) | large dataset (≳ tens of k) |
|---|---|---|
| **small domain gap** (photo-like) | linear probe | full fine-tune |
| **large domain gap** (detector images, spectrograms) | probe first, then partial fine-tune | full fine-tune (or from scratch, if truly enormous) |

The logic: fine-tuning many parameters on little data is Week 09's variance problem
all over again; freezing is a regularizer. A large domain gap means late features
mis-match your data, so *some* adaptation is needed — but with little data you adapt
as few layers as you can get away with. Measure, don't guess: the probe is your
baseline, and fine-tuning must beat it to earn its cost.

## 6. Detector images through a photo-trained backbone

Two mechanical mismatches appear the moment you aim a pretrained backbone at
scientific images, and both have standard fixes.

**Channels.** Pretrained backbones expect 3-channel RGB input; detector images and
FashionMNIST are 1-channel. Either replicate the gray channel three times (zero
surgery, wastes a little compute), or replace the first conv layer with a 1-channel
version (fresh random init for that layer only — it must relearn its edge detectors,
which costs a little data). Both work; the exercises have you measure the difference.

**Statistics.** Pretrained models were trained on inputs normalized with ImageNet's
per-channel mean and standard deviation; feed them data on a wildly different scale
and the pretrained batchnorm statistics are wrong. Normalize your inputs; for
strongly non-photo-like data (a calorimeter image is mostly zeros with a few bright
towers) a log transform before normalizing often helps.

The domain-gap question for Week 20's project is worth asking now: ImageNet contains
no calorimeters, but it does contain the ingredients — bright blobs on dark
backgrounds, two-lobed shapes vs one-lobed shapes. Whether that is enough for transfer
to beat a small from-scratch CNN on toy shower images is an *empirical* question, and
exactly the kind this week's data-efficiency experiment teaches you to answer.

## 7. Worked example: probe, then fine-tune, on a small CIFAR-10 slice

CIFAR-10: 60,000 32×32 color photos in 10 classes (plane, car, bird, ...), the
standard "small but real" vision benchmark; `torchvision` downloads it. We simulate
the scientific regime by training on only 2,000 images, and compare a linear probe
against a full fine-tune of a pretrained ResNet-18 (the 18-layer residual network
from Week 17's lineage, pretrained on ImageNet; ~11M parameters).

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

torch.manual_seed(0)

# ImageNet backbones expect ~224x224 normalized RGB; upscale CIFAR's 32x32.
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), norm,
])
test_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), norm])

train_full = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
test_set = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
train_set = Subset(train_full, range(2000))            # the small-data regime
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(test_set)

def train_epochs(model, opt, n_epochs):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

# --- Stage 1: linear probe ---
model = models.resnet18(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad_(False)
model.fc = nn.Linear(512, 10)                 # fresh head, trainable
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
train_epochs(model, opt, 3)
print("linear probe:", evaluate(model))

# --- Stage 2: full fine-tune, discriminative LRs ---
for p in model.parameters():
    p.requires_grad_(True)
head_ids = [id(p) for p in model.fc.parameters()]
backbone = [p for p in model.parameters() if id(p) not in head_ids]
opt = torch.optim.Adam([
    {"params": backbone, "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
train_epochs(model, opt, 3)
print("fine-tuned:", evaluate(model))
```

Typical result (GPU recommended — 224×224 through a ResNet is slow on CPU; Colab or
Kaggle's free tier is fine): the probe lands around 75–80% test accuracy from just
2,000 images — already far above a small from-scratch CNN, which struggles to clear
50–60% at this dataset size. The fine-tune adds several points more. Read the order of
operations: probe first (cheap baseline), then unfreeze with the backbone LR one order
of magnitude below the head. That ordering, plus the augmentation choices and the
freeze policy, is the "recipe" this week's deliverable asks you to write down — you
will reuse it verbatim in Week 20.

## Check yourself

1. You augment MNIST digits with 180° rotations. What specifically goes wrong, and
   with which digits?
2. Why must augmentation be applied to training data only?
3. Dropout with $p = 0.5$: what happens at eval time in PyTorch's convention, and
   what expectation does the training-time $1/(1-p)$ scaling preserve? For which kind
   of layer is the match exact?
4. In one sentence each: how do augmentation and dropout reduce variance, in Week 09's
   sense?
5. You have 800 labeled detector images. Probe, partial fine-tune, or full fine-tune —
   and why, in terms of parameters-fitted vs data?
6. Why is the backbone LR set below the head LR in a full fine-tune?
7. A linear probe is "Week 10 reborn". What exactly is the logistic regression's
   feature vector here?
8. Name the two mechanical fixes needed to push 1-channel images through an
   RGB-pretrained backbone, and one cost of each.

## Answers

1. Digits that rotate into other digits or into non-digits: 6↔9 swap labels; 2, 3, 4,
   5, 7 become shapes the test set never contains. The label is not invariant under
   the transformation.
2. Validation/test sets must estimate performance on real, fixed data; augmenting
   them changes the measurement, and random augmentation makes the metric noisy for
   reasons unrelated to the model.
3. At eval dropout is a no-op (all units active, no scaling). The training-time
   scaling makes the expected activation over masks equal the eval activation —
   exactly for linear layers, approximately once a nonlinearity follows.
4. Augmentation: the model can't fit accidents of individual images because each
   returns transformed, so fewer spurious patterns survive. Dropout: predictions
   approximate an average over many thinned sub-networks, and averaging reduces
   variance (the bagging argument).
5. Probe first: 800 images fit a head of a few thousand parameters safely, while
   fine-tuning millions of parameters on 800 images is a variance disaster. Then try
   unfreezing the last block only if the probe's features seem to mis-match (large
   domain gap) — and keep it only if it beats the probe on validation.
6. The backbone starts at a good solution; head-sized steps would destroy the
   pretrained features before the randomly-initialized head becomes useful. Small
   steps let the backbone drift, not jump.
7. The backbone's 512-dimensional output for each image (the activations after global
   average pooling) — a learned feature vector replacing hand-crafted features.
8. Replicate the gray channel to 3 (wastes compute on redundant channels) or re-init
   the first conv for 1 channel (that layer loses pretraining and must relearn from
   your data).

## New terms

- **data augmentation** — label-preserving random transforms generating extra
  training examples on the fly.
- **label invariance** — the property a transform must have to be a valid
  augmentation: the true label is unchanged.
- **dropout** — randomly zeroing activations during training (with $1/(1-p)$
  rescaling); off at eval.
- **co-adaptation** — units depending on specific other units' presence; what dropout
  breaks.
- **ImageNet** — the ~1.28M-photo, 1000-class benchmark whose pretrained models are
  the standard reusable backbones.
- **transfer learning** — reusing a network trained on one task as the starting point
  for another.
- **backbone / head** — the reused feature-extractor body / the task-specific final
  layer(s).
- **linear probe** — frozen backbone + trained linear head; the transfer baseline.
- **fine-tuning (partial/full)** — unfreezing some/all backbone weights and training
  them at a low LR.
- **discriminative learning rates** — different LRs per parameter group, small for
  the backbone, larger for the head.
- **domain gap** — how far your data's statistics sit from the pretraining data's.
- **frozen parameters** — parameters with `requires_grad` off; excluded from updates.

## Going deeper

- CS231n course notes, "Neural Networks Part 2" (preprocessing, dropout,
  regularization) and the transfer-learning notes — the practical spine; compare
  their freeze-policy advice with §5's table.
- Prince, *Understanding Deep Learning* (free PDF), Ch. 9 (Regularization) — the
  dropout and augmentation sections put this lesson's arguments in one formal frame.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting" — §1–2 for the co-adaptation and ensemble views from the source; skim
  the experiments.
- The official PyTorch transfer-learning tutorial — the same surgery as §7 on a
  different dataset; good second rep.
