# Week 05 — Sequences, Attention Intuition, and GNNs for HEP

## Learning objectives

By the end of this week you will:

1. Understand why sequence models — RNNs, LSTMs, GRUs — dominated 2014–2017 and why attention killed them.
2. Derive scaled dot-product attention from first principles. Understand it as a differentiable content-addressable lookup over a "memory."
3. Know what multi-head attention buys over single-head, stated in linear-algebra terms rather than vibes.
4. Understand graph neural networks as message-passing on permutation-invariant structures, and see why they're a better prior for jets, tracking, and event-level HEP objects than images are.
5. Know the main HEP-ML GNN papers at a level where you could present ParticleNet and Particle Transformer to your group meeting.
6. Train a small GNN or EdgeConv model on a jet-tagging dataset and beat a dense-MLP baseline.

This is the conceptual hinge of Month 2. After this week, transformers and LLMs are "attention applied to text."

## 1. Why sequences were hard

A sequence x = (x_1, …, x_T) doesn't fit naturally into an MLP because:
- T varies across examples.
- Order matters.
- Long-range dependencies (an early token affecting a late output) need the gradient to flow across T steps.

The classical answer was the **recurrent neural network** (RNN):

$$
h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b), \qquad y_t = W_{hy} h_t.
$$

Shared weights across time. Memory compressed into `h_t`. Train via **backpropagation through time** (BPTT): unroll the loop and treat it as a deep feed-forward net.

### The problem: gradients over time
During BPTT, the gradient at step 1 with respect to a loss at step T involves a product of T Jacobians. If the spectral radius of W_hh is < 1, gradients vanish exponentially. If > 1, they explode. Either way, practical memory capacity was ~20 steps.

### LSTMs and GRUs
Gated recurrent units (Hochreiter & Schmidhuber 1997) add a gated "cell state" with an identity-style update, analogous to ResNet's identity shortcut but over time. Enabled training on ~200-step sequences. Ruled 2014–2017 in NLP and speech.

### Why they still died
Three issues attention solved better:
1. **Sequential computation.** RNNs are fundamentally sequential; you can't parallelize the forward pass across t. Attention is parallelizable.
2. **Long-range dependencies.** RNNs' path length from token i to token j is O(|i − j|). Attention's is O(1).
3. **Hardware fit.** Modern accelerators love matrix multiplies. RNNs are a loop of small ones; attention is a big one.

LSTMs haven't disappeared — they're still competitive on small sequences with streaming constraints — but for research-grade pretrained models, attention won.

## 2. Attention, derived

You have Q queries, K keys, V values. All are matrices of rows; each row is a vector.

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V.
$$

Read it as: each query row q_i gets scores over all keys via dot product with k_j. Scale by √d_k to keep the softmax's pre-activations from being enormous when d_k is large. Softmax row-wise to make probabilities. Weighted sum of value rows.

### Intuition: differentiable memory lookup
Imagine a dictionary `{k_j: v_j}`. A classical lookup by key `q` picks the matching v. Attention is a soft version: compute similarities between q and every k, use them as weights, blend the corresponding v's.

### Self-attention
When Q, K, V all come from the same sequence X via learned linear projections:
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V,
$$
it's **self-attention**. The output is a new sequence where each token is a weighted combination of all tokens, with weights learned.

Key structural point: self-attention is **permutation-equivariant**. Shuffle the input, the output shuffles correspondingly. That's why transformers need **positional encodings** — without them, they can't tell "the cat sat on the mat" from "mat the on sat cat the."

### Why √d_k?
Without the scaling, if q and k are unit-variance random vectors of dimension d_k, their dot product has variance d_k and magnitudes ~√d_k. Softmax of such large logits becomes nearly one-hot, making the gradient vanish. Scaling keeps variance ~1 and preserves gradients. Derive this as an exercise.

### Multi-head attention
Split Q, K, V into `h` parallel "heads," each of dimension d_k/h. Apply attention in each head. Concatenate. Project.

$$
\mathrm{MHA}(X) = \mathrm{concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W_O.
$$

**Why this helps.** Different heads can learn different relationships (syntactic, semantic, positional, "attend to the previous token"). Empirically, heads specialize. It's cheap: head dimensionality divides, so total parameters stay the same.

**Caveat:** recent papers argue most heads are redundant and prune them during post-training. Don't over-fetishize multi-head.

## 3. Position encoding

Self-attention ignores order. We add positional information to the token embeddings.

**Sinusoidal (original transformer):**
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}).
$$
Closed-form, generalizes to longer sequences than seen in training.

**Learned positional embeddings:**
Just an `nn.Embedding(max_len, d_model)`. Simpler, slightly better on sequences ≤ training length, fails to extrapolate.

**Rotary Positional Embedding (RoPE).**
Encodes position by *rotating* Q and K in pairs of dimensions by an angle proportional to position. Dot products end up depending only on relative position. Used in Llama, Gemma, Mistral.

**ALiBi (Attention with Linear Biases).**
Add a position-dependent bias directly to attention logits, no learned embedding. Extrapolates to longer sequences.

For Week 06 you'll implement learned PEs (simplest). Mention RoPE in comments so future-you knows it exists.

## 4. The transformer block (a preview of Week 06)

A decoder-only transformer block:
```
x = x + MultiHeadAttn(LayerNorm(x), mask=causal)
x = x + MLP(LayerNorm(x))
```

Components:
- **Residual connections.** Same role as ResNet: let gradients flow.
- **LayerNorm.** Normalizes across the feature dimension per token (unlike BatchNorm, which normalizes across batch per feature). More appropriate for variable-length sequences. Modern practice uses **pre-norm** (`x + f(norm(x))`) rather than original post-norm (`norm(x + f(x))`); pre-norm trains more stably.
- **MLP.** Position-wise 2-layer feed-forward. Typically hidden = 4 × d_model. GELU activation since ~2019.
- **Causal mask.** In a decoder, token t cannot see tokens > t. Implemented as a mask that sets pre-softmax logits to -∞.

The full GPT-style model stacks ~12–96 such blocks, embeds tokens + position at the front, and has a linear head back to vocab size for next-token logits.

## 5. Graph neural networks

Many HEP objects have a natural graph structure that images don't:
- **Jets:** unordered set of constituent particles, each with (pt, η, φ, m, charge, …).
- **Events:** a set of tracks and calorimeter deposits.
- **Tracking hits:** a graph over detector hits where edges indicate probable track segments.

An image treats features on a grid. A graph treats features on a set of nodes with a relational structure — edges can encode "same subjet," "close in ΔR," "from the same vertex." MLPs on flattened features throw away structure. GNNs keep it.

### Message passing, formalized
At each layer:

1. **Message**: for every edge (i, j), compute m_{i→j} = φ_e(h_i, h_j, e_{ij}).
2. **Aggregate**: for every node i, aggregate incoming messages: m_i = Σ m_{j→i} (or mean, max, attention-weighted).
3. **Update**: h_i' = φ_n(h_i, m_i).

φ_e and φ_n are MLPs. Stack L layers to propagate information L hops.

### Key GNN flavors for HEP

- **GCN (Kipf & Welling 2017).** Symmetric normalized aggregation; foundational, mostly too simple for HEP in practice.
- **GraphSAGE (Hamilton et al. 2017).** Samples neighbors; good for large graphs.
- **GAT (Graph Attention Network, Veličković 2018).** Edge weights come from attention — the intermediate step between GNN and transformer.
- **EdgeConv (Wang et al. 2019, "Dynamic Graph CNN").** Used in ParticleNet. At each layer, rebuild a k-NN graph on the current features; apply a shared MLP on concatenated (h_i, h_j − h_i); max-aggregate.

### ParticleNet (Qu & Gouskos 2020)
- Nodes = jet constituents.
- k-NN graph in (η, φ) space.
- Three EdgeConv blocks with increasing k (16, 16, 16), widths (64, 128, 256).
- Global mean + max pool.
- MLP classifier.
- Beat CNN/DNN baselines on top-quark tagging and quark-gluon discrimination.

### Particle Transformer (Qu et al. 2022)
- Applies transformer (self-attention) over jet constituents.
- Crucial trick: incorporates **pair-wise physics features** (ΔR, k_T, z, invariant mass) directly into the attention bias.
- Currently SOTA on JetClass.

### Exa.TrkX (tracking)
- Nodes = detector hits.
- Edges = candidate track segments (from a pre-filter).
- GNN classifies which edges are "true" (i.e., part of a real track).
- Pipeline: embedding → edge filtering → GNN → graph segmentation.
- Scales to HL-LHC where combinatorial tracking breaks down.

## 6. Sets: Deep Sets and Energy Flow Networks

A graph with no edges is a set. Permutation-invariant functions on sets factor as:
$$
f(\{x_1, \ldots, x_n\}) = \rho\!\left( \sum_i \phi(x_i) \right),
$$
for any permutation-invariant f (Zaheer et al. 2017, *Deep Sets*).

**Energy Flow Networks** (Komiske, Metodiev, Thaler 2019) apply this to jets:
$$
z(\{p_1, \ldots, p_n\}) = F\!\left( \sum_i E_i \Phi(\hat p_i) \right).
$$
Clean, IRC-safe (infrared-and-collinear safe) by construction. A physically motivated architectural choice — HEP-ML at its best.

## 7. Libraries for GNNs

- **PyTorch Geometric (PyG).** https://pytorch-geometric.readthedocs.io/. The dominant library. Supports many GNN layers (`GCNConv`, `GATConv`, `EdgeConv`), datasets, mini-batching via graph disjoint unions.
- **DGL.** https://www.dgl.ai/. Alternative; popular in industry.
- **`jetnet`**. HEP-specific. Jet datasets, evaluation metrics (particle FGD, etc.).
- **`weaver-core`**. https://github.com/hqucms/weaver-core. The training framework for ParticleNet and Particle Transformer. Useful reference.

For Week 05 exercises and project, use PyG on CPU for the math and small models; graduate to GPU only if graphs are large.

## 8. Inductive biases, recap

Across weeks, we've met three priors for data:

| Data structure | Prior | Typical model |
|----------------|-------|---------------|
| Tabular features | Smoothness in feature space, interactions | BDTs, MLPs |
| Images | Local spatial structure, translation equivariance | CNNs |
| Sequences | Long-range dependencies, position | Transformers |
| Sets | Permutation invariance | Deep Sets, EFN |
| Graphs | Relational structure, variable size | GNNs, GATs |

Knowing which prior fits your data is half of architecture design. For heavy-ion analyses:
- **Tabular analyses (dijet x-sections, v2 measurements):** BDTs.
- **Calorimeter images:** CNNs, ViTs.
- **Jet constituents:** GNNs, Particle Transformers.
- **Event-level with many object types:** graph of heterogeneous nodes; domain-specific architectures.

## 9. Common pitfalls this week

- **Misunderstanding "attention."** Attention is not magical. It's a weighted sum. The weights are learned. That's it.
- **Forgetting the causal mask.** Cross-contamination from future tokens into past predictions turns an LM into a trivial look-ahead copy. AUC on train is perfect; loss on val is terrible.
- **LayerNorm vs BatchNorm confusion.** LN normalizes over features within a sample; BN normalizes over the batch for each feature. For sequences, LN. Wrong axis = training fails silently.
- **Positional embeddings mis-wired.** Off-by-one between embedding table and input indices.
- **GNN mini-batching.** PyG uses `Batch` which builds a block-diagonal disjoint union of subgraphs. Don't manually batch; use `DataLoader` from PyG.
- **Too many message-passing layers.** Over-smoothing: all node representations converge. 2–4 layers is usually enough.
- **k-NN graph with too-large k.** Makes every node see everything; defeats the prior. For jets, k = 8–16.

## 10. Connections to your HEP work

Concrete:
- **ParticleNet on sPHENIX jets.** If you have a jet sample with constituent-level features (pt, η, φ, charge, PID), you can train a small ParticleNet to tag gluon vs quark jets. Strong baseline for any jet-flavor analysis.
- **GNN tracking relevance.** If sPHENIX tracking has reconstruction challenges, Exa.TrkX-style GNN filters exist as a research program. Worth knowing even if you never touch the code.
- **Event-level GNN.** For heavy-ion, the event is a dense graph of particles; GNN architectures like JEDI-net have been explored. Look them up when Week 05 is done.

## 11. Self-check

- State scaled dot-product attention. Why the scale factor?
- What is a causal mask and where is it applied?
- Why is LayerNorm preferred over BatchNorm for sequences?
- State the Deep Sets theorem in one line.
- In ParticleNet, what does "Edge" in EdgeConv refer to?
- Given jets with ~30 constituents and a choice between a CNN on (η, φ) image and a GNN, which prior is more appropriate and why?

When solid, continue.

---

Next: `reading.md`, `exercises.md`, `project.md`, `resources.md`.
