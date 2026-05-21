# Week 04 — Exercises

## E1 — Conv arithmetic by hand (easy, 30 min)

For each, compute output shape and parameter count without running PyTorch, then verify with `torchinfo.summary`:

1. Conv2d(1, 32, kernel_size=3, padding=1, stride=1), input (1, 28, 28).
2. Conv2d(32, 64, kernel_size=5, padding=0, stride=2), input (32, 28, 28).
3. Conv2d(64, 64, kernel_size=3, padding=1, groups=64), input (64, 14, 14). (This is depthwise.)
4. Stack of three Conv2d(3, 64, 3×3, padding=1) → MaxPool2d(2) → Conv2d(64, 128, 3×3, padding=1) → MaxPool2d(2) → GlobalAvgPool → Linear(128, 10). Total params?

## E2 — Implement a tiny CNN on Fashion-MNIST (easy, 45 min)

**Data:** `torchvision.datasets.FashionMNIST`.

Build:
```
Conv(1→32, 3, pad=1) → ReLU → MaxPool(2)
Conv(32→64, 3, pad=1) → ReLU → MaxPool(2)
Flatten → Linear(64*7*7, 128) → ReLU → Linear(128, 10)
```

Train for 5 epochs, lr=1e-3, Adam. Target ≥ 89% test accuracy.

## E3 — Add BatchNorm and Dropout (medium, 30 min)

Starting from E2, add BN after each conv and dropout(0.2) before the final Linear. Retrain and compare:
- Convergence speed (epochs to 89% vs before).
- Overfit gap (train acc − val acc).

## E4 — From-scratch ResNet-18 (medium → hard, 120 min)

Implement ResNet-18 from scratch in PyTorch. Do not use `torchvision.models`.

Reference Table 1 of He et al. 2016. Critical details:
- 7×7 conv stem, stride 2, 64 channels.
- 4 stages with [2, 2, 2, 2] BasicBlocks.
- First block of each stage except stage 1 has stride 2 and a 1×1 projection shortcut.
- Global average pool, 512 → num_classes Linear.

Train on CIFAR-10 (adjust stem for 32×32 — use 3×3 stride 1 and skip the initial maxpool). Target ≥ 92% test accuracy after 50 epochs with cosine LR.

**Accept when:** your ResNet-18 matches torchvision's ResNet-18 in parameter count (±1) and trains to similar accuracy on CIFAR-10.

## E5 — Receptive field calculation (medium, 30 min)

For your ResNet-18, compute the receptive field of a neuron in the final feature map on the input plane. Show the per-stage calculation. Compare to a VGG-11. Discuss what this means for EMCal shower images where a full π⁰ cluster is ≈ 5×5 to 7×7 towers.

## E6 — Data augmentation audit (medium, 45 min)

Take a known-label sample of 20 EMCal-like cluster images (generated or your own).

1. Apply random rotations, horizontal flips, additive Poisson noise, and color jitter.
2. Plot the before/after side by side.
3. Write a one-paragraph evaluation for each transform: physics-preserving? Label-preserving? Recommend keep/drop.

## E7 — Confusion matrix + failure stratification (medium, 45 min)

**Data:** CIFAR-10 or any multiclass task you trained in E2–E4.

1. Compute the confusion matrix on the test set.
2. For each off-diagonal cell, pick 5 random misclassified examples and look at them.
3. Identify one systematic failure mode (e.g. cats classified as dogs when they're small kittens).
4. Propose a data or augmentation fix.

## E8 — Calibration plot (medium, 30 min)

On your best E4 model:
1. Compute softmax probabilities on the test set.
2. Plot reliability diagram (10 bins).
3. Compute ECE.
4. Fit temperature scaling: find `T > 0` that minimizes NLL on a held-out set (`logits / T` then softmax). Re-plot.

## E9 — Grad-CAM (hard, 60 min)

Implement Grad-CAM for your ResNet-18. Reference: Selvaraju et al. 2017. The implementation is one hook on the last conv layer's output plus the gradient w.r.t. that tensor.

Visualize on 10 test images. For correct predictions, the heatmap should localize on object parts. Comment on one example where the localization is unexpected.

## E10 — (Capstone prep) Pythia-like shower-image generator (hard, 90 min)

Write a generator that produces 2D η×φ images of:
- Single photon: 2D Gaussian with σ_η = σ_φ ≈ 0.02 rad, random total energy 2–30 GeV, random position.
- Merged π⁰: two Gaussians with opening angle Δθ ∈ (0.005, 0.05) rad and energy asymmetry |E1 − E2|/(E1+E2) ∈ [0, 0.8].

Add Poisson shot noise per tower scaled by deposit. Save as `(N, 1, H, W)` tensors with labels in a parquet file.

**Accept when:** you can generate 20 000 images in ~30 s; class-conditional image summaries (mean image per class) look visibly different.

---

## Solution hints

- E1 — formula: H' = floor((H + 2p − k)/s) + 1. For groups=g: params = C_out × (C_in/g × k² + 1).
- E2 — Adam + lr=1e-3 is fine; no need to tune for this exercise.
- E3 — BN accelerates convergence; the usefulness of dropout is smaller on BatchNorm'd CNNs.
- E4 — off-by-one on the stride of the first block of stage 2 is the #1 bug. Print output shapes at every stage.
- E5 — receptive field of a 3×3 conv stack at stride 1 grows by 2 per layer; at stride 2 it doubles.
- E6 — horizontal flip is generally fine for natural images, wrong for handwritten digits or for η-φ calorimeter data where the coordinate system isn't mirror-symmetric.
- E7 — look at the actual pictures. That's the whole exercise.
- E8 — CNNs are almost always overconfident. T ≈ 1.2–1.5 typical.
- E9 — Grad-CAM gives a coarse heatmap. Don't over-interpret; it's a visualization, not a proof.
- E10 — important: generate labels with balanced kinematics (don't let all π⁰s be low-energy). Also dump metadata (E, η, φ, opening angle) to the parquet so you can stratify failures later.
