# Week 21 — Graph Neural Networks

An event is not an image: tracks, hits, and jet constituents are variable-length,
unordered sets with relational structure — the physics does not change when you permute
the particle list, so the architecture shouldn't either. (`lesson.md` explains jets and
tracking from scratch before any of this is used.)

## Objectives

- Write the generic message-passing update (message → aggregate → update) and show why
  symmetric aggregation (sum/mean/max) makes graph-level outputs permutation invariant.
- Explain the design space: node features, edge construction (kNN in η–φ, static vs
  dynamic graphs), aggregation choice, depth vs oversmoothing.
- Read ParticleNet and Particle Transformer as physics papers: representation choice,
  baselines, and what actually drove the gains.
- Implement a GNN in PyTorch Geometric: `MessagePassing` subclass, batched graphs.
- Say what Exa.TrkX does — tracking as edge classification on hit graphs — and why.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: permutation invariance, message
  passing derived and proved, EdgeConv, PyG, and the HEP-ML literature map.
- distill.pub, "A Gentle Introduction to Graph Neural Networks" and "Understanding
  Convolutions on Graphs".
- *Understanding Deep Learning* (Prince), Ch. 13 (Graph neural networks).
- ParticleNet: "Jet Tagging via Particle Clouds" (arXiv:1902.08570) — read fully.
- Particle Transformer: "Particle Transformer for Jet Tagging" (arXiv:2202.03772) —
  architecture and pairwise-feature sections; skim results.
- Exa.TrkX: "Performance of a geometric deep learning pipeline for HL-LHC particle
  tracking" — skim for the problem formulation.
- PyTorch Geometric "Introduction by Example" docs.

## Derivations (paper first)

- Prove: if node embeddings are updated by permutation-equivariant layers and pooled by
  a symmetric aggregation, the graph-level readout is permutation invariant.
- Derive one full EdgeConv-style update by hand for a 3-node graph (numbers in, numbers
  out) — this becomes the unit test for your implementation.
- Show a 1D convolution is message passing on a line graph with position-dependent
  messages — CNNs as a special case.

## Exercises

See `exercises.md` (notebook generated from it when the week starts, per
`NOTEBOOK_RULES.md`). E1–E3 build message passing from a permutation stress test up
through a hand-verified PyG layer; E4–E6 convert the Week-20 clusters to point clouds
and pit a GNN against your CNN; E7 is two paper notes.

## Deliverable

A working PyG pipeline on your own detector point clouds + the hand-verified message-
passing layer + two paper notes. This is the launch pad for Capstone track (a).

## Review

- (Week 17) Convolution hard-codes translation equivariance; what symmetry does
  sum-aggregation hard-code, and why does HEP data demand it?
- (Week 19) Dynamic-graph EdgeConv rebuilds kNN in feature space each layer. Relate
  this to attention's content-based lookup from your Week-19 note.
- (Week 11) Your Phase-1 BDT ate fixed-length feature vectors. Name two things the
  point-cloud representation preserves that the feature vector destroyed.
