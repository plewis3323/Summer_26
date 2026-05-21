# Week 04 — Resources

## Docs

- **PyTorch `torch.nn.Conv2d`.** https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html. Read the "shape" section carefully.
- **`torchvision`.** https://pytorch.org/vision/stable/. Datasets, models, transforms.
- **`torch.utils.data`.** `Dataset`, `DataLoader` idioms. The "tutorial" in PyTorch docs is a must.
- **wandb docs.** https://docs.wandb.ai/. "Track experiments" → "Quickstart". You'll be using this from now on.
- **`timm` (PyTorch Image Models).** https://github.com/huggingface/pytorch-image-models. The library of modern CNN/ViT checkpoints. Worth knowing for Month 2.

## Tutorials

- **PyTorch's official "Training a Classifier" tutorial.** https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html. Classic CIFAR-10 walkthrough.
- **Sebastian Raschka — CIFAR-10 with PyTorch** notebook. https://github.com/rasbt/deeplearning-models. Many reference implementations side-by-side.
- **d2l.ai — CNNs chapter.** https://d2l.ai/chapter_convolutional-neural-networks/. LeNet through modern architectures with runnable code.
- **Fast.ai Lesson 3** (fine-tuning) and Lesson 4 (tabular + training details). https://course.fast.ai/.

## Videos

- **Stanford CS231n** — find on YouTube. Lectures 5, 7, 9, 10 cover CNNs, training, and architectures. 2017 era but evergreen.
- **Fast.ai 2022 lectures** — https://www.fast.ai/ — Jeremy Howard's practical lens.
- **Yannic Kilcher paper reviews** on YouTube for ResNet, Batch Normalization, Adam. Not primary learning but good complement.

## GitHub repos

- **`pytorch/vision`** — study `torchvision/models/resnet.py` and `torchvision/transforms/`. They are clean, readable references.
- **`rasbt/deeplearning-models`** — Sebastian Raschka's catalog. CNN zoo, good for comparing implementations.
- **`kuangliu/pytorch-cifar`** — concise implementations of ~15 CNN architectures for CIFAR-10. Great for benchmarking.
- **`pytorch/ignite`** — if you tire of writing training loops by hand (don't yet; write one more before you abstract).

## Codecademy (Pro)

- "PyTorch for Deep Learning" — CNN modules, augmentation.
- "Deep Learning with TensorFlow" (optional) — if you want to see how TF does the same things and why PyTorch feels better for research.

## Papers worth having on hand

- **He et al. — ResNet.** arXiv:1512.03385.
- **Ioffe & Szegedy — Batch Normalization.** arXiv:1502.03167.
- **Srivastava et al. — Dropout.** JMLR 2014.
- **Smith — Super-Convergence / OneCycle.** arXiv:1708.07120.
- **Selvaraju et al. — Grad-CAM.** arXiv:1610.02391.
- **Liu et al. — ConvNeXt.** arXiv:2201.03545.
- **Guo et al. — On Calibration of Modern Neural Networks.** ICML 2017. Reference for temperature scaling.

## HEP-flavored references

- **HEP-ML Living Review.** https://iml-wg.github.io/HEPML-LivingReview/. Browse "Calorimetry", "Reconstruction".
- **de Oliveira et al., LAGAN / CaloGAN.** arXiv:1701.05927, arXiv:1705.02355. Earliest "calorimeter as image" papers.
- **Baldi, Sadowski, Whiteson.** Nature Communications 2014 — where the HEP+DL story started.
- **sPHENIX simulation docs** — if you have internal access, for generating realistic photon/π⁰ samples.

## Tooling adopted this week

- **`wandb`.** Sign up (free academic tier), `wandb login`, `uv add wandb`.
- **`torchinfo`.** For parameter counts.
- **`torchmetrics`.** `uv add torchmetrics`. Clean metric classes that handle DDP if you later need it.
- **`albumentations`.** `uv add albumentations`. Mature augmentation library; more than torchvision's transforms.

## Datasets

- `torchvision.datasets.FashionMNIST` — E2.
- `torchvision.datasets.CIFAR10` — E4, E5, E7.
- Synthetic EMCal images from your generator — E10, project.

## When you get stuck

- **PyTorch Forums.** Best for model/training bugs.
- **Stack Overflow tag `pytorch`.** For one-off syntax questions.
- **Wandb community forum.** Fast response on tooling questions.
- **Your physics-department's ML Slack (if one exists).** The people who've debugged "why doesn't my CNN train on calorimeter images" have already solved half your problems.

## Further reading (Month 2 prep)

- **Dosovitskiy et al. — ViT.** arXiv:2010.11929. Vision transformers; we'll touch them in Week 05.
- **Liu et al. — ConvNeXt.** arXiv:2201.03545.
- **Karpathy, ALL Zero-to-Hero beyond lecture 1.** Makemore and GPT videos are Week 05–06 material.

---

You're done with Month 1 once the capstone is on GitHub, tagged, and a retro is written. Report back before starting Month 2.
