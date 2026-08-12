# Month 04 — Neural Networks from Scratch

The arc: earn the right to call `loss.backward()`. Week 13 derives backprop on paper for
a 2-layer net, including the softmax/cross-entropy gradient. Week 14 turns that math into
a working scalar autograd engine (micrograd-style) verified against finite differences.
Week 15 swaps your engine for PyTorch and confirms it does the same thing, then scales up
to a makemore-style character model. Week 16 confronts the part no derivation prepares
you for — training dynamics: initialization, normalization, dead units, schedules — and
closes with a mini-project pitting an MLP against your Phase-1 BDT capstone on tabular
physics data.

Each week builds directly on the last: the gradient you derive by hand in Week 13 is the
one your engine computes in Week 14, PyTorch reproduces in Week 15, and you debug at
scale in Week 16.

**Month-end deliverable:** working autograd engine + PyTorch MLP on physics tabular data
with an honest comparison against the Capstone-1 BDT (win, or explain why not).

**Sign-off:** tag `month-04-complete`, write `retro.md` (~250 words), open one issue for
the biggest thing you still don't understand.
