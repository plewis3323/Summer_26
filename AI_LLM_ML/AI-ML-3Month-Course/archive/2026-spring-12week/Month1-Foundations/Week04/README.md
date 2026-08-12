# Week 04 — Convolutional Neural Networks and the Month-1 Capstone

## Learning objectives

By the end of this week you will:

1. Explain *why* convolution is the right prior for images (local spatial structure, translation equivariance, compositional hierarchy). Understand what CNNs trade away compared to MLPs.
2. Know the mechanics of a 2D convolution: kernel, stride, padding, dilation, groups, channels. Compute output shape and parameter count from first principles.
3. Understand the supporting cast: pooling, batch normalization, dropout, residual connections. For each, know the intended failure mode it addresses and its cost.
4. Know the standard modern CNN architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet → ConvNeXt) and what each innovation actually solved.
5. Apply data augmentation principled-ly: *physics-respecting* augmentations for shower images vs. generic photo augmentations.
6. Ship the month-1 capstone: a CNN that classifies single-photon vs merged-π⁰ EMCal clusters, with an ROC curve, calibration plot, and honest failure analysis.

## 1. Why convolution?

MLPs work on vectors. An image is a vector if you flatten it, but flattening discards the prior that *nearby pixels are more related than distant pixels*. An MLP trained on flattened images has to learn this from data; a CNN has it baked in.

Two structural priors encoded in a convolutional layer:

**Locality.** A neuron only looks at a small spatial patch of its input. This is the same assumption you make when you use a 3×3 shower-shape window in EMCal reconstruction: the physics of shower development is local.

**Translation equivariance.** The same kernel is applied everywhere. If you shift the input image by one pixel, the output feature map shifts by one pixel. The model doesn't need to learn "the π⁰ is at row 14" and "the π⁰ is at row 15" separately. Every patch is a training example for the shared kernel.

These two priors together reduce the effective parameter count by orders of magnitude and massively improve sample efficiency — which is exactly why CNNs crush MLPs on image tasks even though MLPs are universal approximators.

### Why it matters for HEP
- EMCal cluster images: ~5×5 to 7×7 in η×φ. Small, and translationally equivariant in φ up to coordinate system boundaries.
- HCAL towers, jet images, tracker hit maps — same structure.
- Calorimeter images often have meaningful angles (pT direction) that translation equivariance alone doesn't capture. This is one motivation for rotation-equivariant networks or for deeper models that learn rotational structure from data.

## 2. The 2D convolution, carefully

Input tensor shape `(C_in, H, W)`. Kernel shape `(C_out, C_in, k_h, k_w)`. Output `(C_out, H', W')`.

For a kernel K applied at position (i, j) in the output:

$$
y_{c, i, j} = \sum_{c'=0}^{C_{in}-1} \sum_{u=0}^{k_h-1} \sum_{v=0}^{k_w-1} K_{c, c', u, v} \cdot x_{c', i+u, j+v} + b_c.
$$

(Strict mathematicians will point out this is *cross-correlation*, not convolution — for convolution proper you flip the kernel. Deep-learning libraries call it convolution anyway. Live with it.)

**Output shape** with stride s and padding p:

$$
H' = \lfloor (H + 2p - k_h) / s \rfloor + 1.
$$

**Parameter count** for one conv layer: `C_out × (C_in × k_h × k_w + 1)`. The "+1" is the bias. A 3×3 layer mapping 64→64 channels has 64×(64×9+1) = 36 928 parameters — tiny compared to a dense equivalent.

### Padding
- **Valid (p=0):** output shrinks by `k-1` per layer. Fine for one or two layers; in a deep stack you run out of spatial resolution.
- **Same:** pad so `H' = H` when `s=1`. The conventional choice in modern CNNs.
- **Reflection, replication, zero:** pad values differ. Zero padding is default; reflection is better for images where edges have meaning.

### Stride and dilation
- **Stride s > 1** downsamples.
- **Dilation d** spaces kernel taps apart. A 3×3 conv with dilation 2 looks at a 5×5 receptive field but touches only 9 pixels. Useful for semantic segmentation.

### Groups
- **Grouped convolution** divides input channels into groups; each filter only sees its own group. Historical origin: AlexNet split across two GPUs. Modern use: depthwise convolutions (groups = C_in), the core of MobileNets and ConvNeXt.

### Receptive field
The receptive field of a neuron at depth L is the region of input it can "see." With all 3×3 convs at stride 1, the receptive field grows by 2 each layer (3, 5, 7, 9, …). To see a 32×32 input you need ~15 conv layers or strided downsampling. This is why real CNNs alternate 3×3 stride 1 with 2×2 max-pool or stride-2 convs.

## 3. The supporting cast

### Pooling
- **Max pool.** Take max over a k×k window. Translation-invariant (up to k pixels). The classical downsampler.
- **Average pool.** Mean. Preserves gradient through all inputs.
- **Global average pool** at the end of a network replaces the final flatten+dense: it produces one number per channel, regardless of spatial size. Crucial for variable-size inputs.

### Batch normalization
Given a mini-batch of pre-activations z ∈ ℝᴺˣ ᴰ (N batch, D features), compute per-feature mean μ_D and variance σ²_D over the batch, then:

$$
\hat z = \frac{z - \mu_D}{\sqrt{\sigma_D^2 + \epsilon}}, \quad y = \gamma \hat z + \beta.
$$

γ and β are learned per feature. At inference, replace μ/σ² with running averages accumulated during training.

**Why it helps.** Empirically: stabilizes training, enables higher learning rates, acts as regularizer. Theoretically: the story is still contested (ICS, smoothing of the loss landscape, etc.).

**Common bugs.** Forgetting `model.eval()` at inference — running stats are frozen but dropout is still on, etc. Tiny batch sizes (≤4) make batch stats noisy; use `GroupNorm` or `LayerNorm` instead.

### Dropout
At each training step, zero out a random p-fraction of activations; scale the survivors by 1/(1-p) so the expectation is preserved. At inference, identity.

- Reduces overfitting.
- Not often combined with batchnorm (they interact oddly). Modern vision architectures use very little dropout.

### Residual connections (He et al., ResNet)
Standard layer: `h = f(x)`. Residual layer: `h = x + f(x)`. If f is a 2-conv block, this one change made 50-, 100-, 1000-layer nets trainable, when before that 20 layers was the max.

**Why.** The gradient through the `+` is the identity plus ∂f/∂x. Even if f's gradient vanishes, the identity path propagates the gradient. It's the same trick as LSTM's cell state for RNNs.

## 4. A brief architectural history

- **LeNet-5 (1998).** 7 layers, 60k params, MNIST. Proof of concept.
- **AlexNet (2012).** 8 layers, 60M params, ImageNet. Two GPUs, ReLU, dropout, local response normalization. The catalyzing paper of modern DL.
- **VGG (2014).** All 3×3 convs, deep. Clean but very parameter-heavy.
- **Inception / GoogLeNet (2014).** Parallel multi-scale branches; bottleneck 1×1 convs.
- **ResNet (2015).** Residual connections, batchnorm, >100 layers trivial. Still a strong baseline.
- **DenseNet (2017).** Every layer receives all preceding features.
- **EfficientNet (2019).** Neural architecture search + compound scaling.
- **Vision Transformers (2020+).** Patch-embed an image and treat it as a sequence. Discussed Week 05–06.
- **ConvNeXt (2022).** "What if we modernize a ResNet with transformer-era design choices?" Competitive with ViT. Worth knowing.

For this week, you'll implement a small CNN from scratch and also use a ResNet-18 (either from scratch or pretrained) as a baseline. ResNet-18 is small enough to fit on any hardware and strong enough that beating it meaningfully is evidence your domain-specific approach is doing something right.

## 5. Data augmentation

Augmentation expands the effective dataset by applying transformations that preserve the label. For natural images: rotations, flips, crops, color jitter. For physics images, you must be careful:

- **Allowed always:** additive Poisson noise (if simulating detector read noise), small spatial translations that respect the sampling grid.
- **Sometimes allowed:** rotation around the beam axis if your coordinate system is rotation-invariant in the projection you're looking at.
- **Forbidden:** horizontal flip that swaps η → -η when the detector isn't symmetric; color jitter for calibrated energy deposits (changing the color *is* changing the energy).

Your capstone will use: small (±1–2 pixel) spatial jitter, additive Poisson noise scaled by deposit energy, and possibly rotation in φ only. Not flips. Not color changes.

## 6. The training loop, the right way

A template to internalize:

```python
import torch
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    ...
```

Things to notice:
- `non_blocking=True` with `pin_memory=True` in `DataLoader` for GPU transfer overlap.
- Gradient clipping guards against explosion.
- Explicit metric accumulation over batches, not per-step averaging.
- `@torch.no_grad()` on eval disables graph construction.

### Learning rate schedules
- **Cosine decay with warmup** is a safe modern default.
- **OneCycleLR (Smith 2018)** is excellent for small/medium models. `torch.optim.lr_scheduler.OneCycleLR`.
- **Fixed LR** works for small tasks. Start there; scale up when the task demands it.

### Mixed precision
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits = model(x)
    loss = loss_fn(logits, y)
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
```
1.5–3× speedup on A100s, small-to-no accuracy loss for most tasks. Use it when you train anything nontrivial.

## 7. Evaluation for a physics classifier

Beyond AUC and a confusion matrix:
- **Calibration plot.** Predicted probability vs empirical frequency. CNNs are often overconfident.
- **Reliability vs kinematic variable.** Plot accuracy vs cluster energy, vs opening angle. Look for systematic biases.
- **Saliency / CAM.** Where does the network attend? For EMCal clusters, Grad-CAM over the shower image should light up physically meaningful regions.
- **ECE (Expected Calibration Error).** Scalar summary of calibration deviation.
- **Failure stratification.** Group errors by label-side physical variables (π⁰ mass, γγ opening angle, energy asymmetry). Failures concentrate where the physics itself is ambiguous — that's information.

## 8. Reproducibility for CNN training

Harder than for classical ML because GPU ops are non-deterministic by default. You can make training fully deterministic:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

Cost: 10–30% wall-clock. For Month 1 capstone, accept the speed hit for one "canonical" run; use non-deterministic for hyperparameter sweeps.

Also seed `torch`, `numpy`, `random`, and the `DataLoader` worker seeds.

## 9. Experiment tracking — enter wandb

This week adopt `wandb` (or MLflow if you prefer self-hosted). Why now: CNN training has more hyperparameters and more ways to silently go wrong. You want an audit trail.

```python
import wandb
wandb.init(project="emcal-classifier", config=asdict(cfg))
# inside training loop:
wandb.log({"train/loss": train_loss, "val/auc": val_auc, "epoch": epoch})
```

Wandb's "runs" page gives you side-by-side comparisons for free. Invaluable.

## 10. Common pitfalls this week

- **Training loss decreases, val loss doesn't.** Overfitting. Augmentation, regularization, or more data.
- **Val loss decreases then rises.** Overfitting onset. Early stopping on val loss.
- **Loss NaN after a few steps.** LR too high. Try 10× lower.
- **Model doesn't learn at all.** Overfit one batch first. If that fails, model or loss is wrong.
- **Test accuracy >> val accuracy.** Distribution shift or data leakage. Investigate.
- **Augmentation too aggressive.** Model's effective training set becomes out-of-distribution. Validate your augmentations against eye inspection.
- **Batchnorm at tiny batch size.** Use `GroupNorm` or increase batch size via gradient accumulation.
- **Saved checkpoint can't be loaded.** `strict=False` on `load_state_dict` hides real bugs. Figure out the key mismatch.
- **Pretrained weights' normalization stats don't match your data.** If you fine-tune an ImageNet ResNet, their mean/std are different from yours. Explicit normalization transform first.

## 11. Connections to your HEP work

This week's capstone *is* a HEP task — photon vs π⁰ discrimination on EMCal clusters. Concretely:

- If your current sPHENIX workflow extracts hand-engineered shower-shape variables (χ²prob, e3×3/e5×5, eta/phi widths), build a BDT on those too. The CNN-on-images vs BDT-on-features comparison is informative.
- Make sure your image channels carry *physics*: energy per tower in layer 1, layer 2, layer 3; possibly timing. Each channel is a physics-meaningful scalar over the η-φ plane.
- Document the data simulation (Pythia version, GEANT4 config, pileup model). Without that, nobody can trust the result.

## 12. Self-check

- Derive the output shape of a 2D conv with input (1, 28, 28), kernel 3, stride 2, padding 1.
- State the parameter count of a conv layer mapping 32 channels to 64 channels with 3×3 kernel. Answer: 64 × (32 × 9 + 1) = 18 496.
- Why does a ResNet train when a plain 50-layer CNN does not?
- Name two physics-appropriate augmentations for EMCal images and two that are forbidden.
- What does `model.eval()` change?

If solid, continue to the project. This is the Month 1 capstone; budget enough time.

---

Continue to `reading.md`, `exercises.md`, `project.md`, `resources.md`.
