# Week 04 — Reading

## Primary

1. **Goodfellow, Bengio, Courville, *Deep Learning*, Ch. 9 "Convolutional Networks".**
   - §9.1 "The Convolution Operation" (pp. 326–329). Math foundation.
   - §9.2 "Motivation" (pp. 329–335). Sparse interactions, parameter sharing, equivariance.
   - §9.3 "Pooling" (pp. 335–339).
   - §9.4 "Convolution and Pooling as an Infinitely Strong Prior" (pp. 339–341). Excellent framing.
   - §9.5 "Variants of the Basic Convolution Function" (pp. 341–347). Stride, padding, dilation.
   - §9.10 "The Neuroscientific Basis for Convolutional Networks" (pp. 364–369) — optional flavor.

2. **Goodfellow, Bengio, Courville, *Deep Learning*, Ch. 7 "Regularization".**
   - §7.1 Parameter norm penalties. §7.4 Dataset augmentation. §7.5 Noise robustness. §7.8 Early stopping. §7.12 Dropout.
   - Ch. 8 §8.1–8.5 on optimization — especially §8.3 (basic algorithms) and §8.5 (adaptive learning rates).

3. **Karpathy, *A Recipe for Training Neural Networks*.** Reread if you haven't lately. https://karpathy.github.io/2019/04/25/recipe/. The capstone will test every step.

4. **He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016). arXiv:1512.03385.**
   - Read end-to-end. §3 "Deep Residual Learning" is the important part. Understand Figure 2 (the identity mapping) precisely.

5. **Ioffe & Szegedy, *Batch Normalization* (ICML 2015). arXiv:1502.03167.**
   - Read §1–3, skim §4. The training/inference asymmetry is the thing to internalize.

## Supplementary

6. **Krizhevsky, Sutskever, Hinton — *AlexNet* (NIPS 2012).** Historical. Read the abstract and §5 (architecture).

7. **Simonyan & Zisserman — *VGG* (ICLR 2015). arXiv:1409.1556.** Skim §2 (the "all 3×3 convs" thesis).

8. **Liu et al. — *ConvNeXt* (CVPR 2022). arXiv:2201.03545.** The "modernize a ResNet" paper. Good to see what 2022 considers best-practice choices: GELU, LayerNorm, depthwise convs.

9. **Smith — *Cyclical Learning Rates / OneCycle policy* (2017).** arXiv:1708.07120 and follow-ups. Short. Explains the LR schedule that works astonishingly well out of the box.

10. **Fast.ai — *Practical Deep Learning for Coders*, lectures 1 and 3.** https://course.fast.ai/. Jeremy's top-down approach complements Goodfellow's bottom-up.

## HEP-flavored

11. **de Oliveira et al., *Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks* (Comp. & Softw. for Big Sci. 2017). arXiv:1701.05927.**
    - First major paper treating calorimeter data as images. Read §2 (data representation) and §3 (architecture). Foreshadows Week 12 track (b).

12. **Komiske et al. — *Energy Flow Networks: Deep Sets for Particle Jets* (JHEP 2019). arXiv:1810.05165.**
    - Foreshadows Week 05. You don't need to do the math; just see that the "image" framing isn't the only one.

## Codecademy (Pro)

- **"PyTorch for Deep Learning" — CNN modules.** Get the idioms; don't treat as primary.

## Videos

- **Karpathy Zero-to-Hero Lecture 3 ("Building makemore Part 2: MLP")** — you'll review training dynamics and initialization, which matter more than we pretended in Week 03.
- **Stanford CS231n lecture videos** (2017 era; find on YouTube). Lectures 5 (CNN architectures) and 7 (training).
- **3Blue1Brown — "Convolutions" (not the neural-networks playlist; the standalone video).** Mathematical intuition.

## Notes to take

- A single-page cheat sheet with: output-shape formula, parameter-count formula, receptive-field math, standard block templates (3×3 conv → BN → ReLU, residual block, bottleneck block).
- Your personal "Karpathy recipe" note from Week 03, updated with CNN-specific items: BN vs GroupNorm rules, augmentation audit, calibration check.
- One paragraph: "Why, architecturally, does ResNet train 50 layers when plain convnets plateau at ~20?" Phrase it in your own words.
