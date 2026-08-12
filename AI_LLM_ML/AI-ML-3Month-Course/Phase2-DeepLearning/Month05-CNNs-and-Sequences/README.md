# Month 05 — CNNs & Sequences

The arc: from "a net that eats flat vectors" to architectures shaped by the structure of
the data. Week 17 derives convolution as parameter sharing and walks the LeNet→ResNet
lineage, with skip connections as gradient highways. Week 18 covers how vision models are
actually trained in practice — augmentation, dropout, transfer learning — applied to
detector images. Week 19 switches from spatial to temporal structure: RNNs, LSTMs,
backprop through time, why long-range credit assignment fails, and attention as the fix
(intuition only; the full derivation opens Phase 3). Week 20 is the mini-project: a CNN
classifying single-photon vs merged-π⁰ clusters on toy EMCal shower images, benchmarked
against a Phase-1-style tabular approach.

The dependency chain is deliberate: Week 17's receptive-field arithmetic sizes Week 20's
network; Week 18's training recipes are what make it converge; Week 19 explains why the
sequence road led to attention before Phase 3 takes that road.

**Month-end deliverable:** EMCal cluster-image classifier with a CNN-vs-tabular
comparison and a physics-aware error analysis.

**Sign-off:** tag `month-05-complete`, write `retro.md` (~250 words), open one issue for
the biggest thing you still don't understand.
