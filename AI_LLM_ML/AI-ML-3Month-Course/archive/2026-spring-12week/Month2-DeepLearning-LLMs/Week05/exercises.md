# Week 05 — Exercises

## E1 — Scaled dot-product attention by hand (easy, 45 min)

1. Implement scaled dot-product attention from scratch in NumPy (no PyTorch). Given (Q, K, V) with shapes (T, d_k), return the (T, d_v) output.
2. Verify numerically against `torch.nn.functional.scaled_dot_product_attention` for random inputs.
3. Derive, on paper, why dot products of unit-variance random vectors in dimension d_k have variance d_k. Show the √d_k scaling keeps softmax logits order-unity.

## E2 — Multi-head attention (medium, 60 min)

Build `MultiHeadAttention(d_model=128, n_heads=8)` in PyTorch from primitives only (no `nn.MultiheadAttention`).

Verify parameter count: 4 × d_model² (W_Q, W_K, W_V, W_O, ignoring biases) = 65 536.

Test: for a random (B=2, T=10, d_model=128) input, output shape is (2, 10, 128). With a random causal mask, positions past t can't leak into position t: verify by perturbing a late token and checking early outputs don't change.

## E3 — Causal-mask correctness test (medium, 30 min)

Wrap your MHA in an `attn_with_mask(x, causal=True)` and write a test:

```python
def test_causal():
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(1, 8, 64, requires_grad=True)
    y = attn_with_mask(x, causal=True)
    # Change only the last token; early outputs must be unchanged
    x2 = x.clone()
    x2[0, 7] += 1.0
    y2 = attn_with_mask(x2, causal=True)
    assert torch.allclose(y[0, :7], y2[0, :7])
```

**Accept when:** the assertion passes and swapping `causal=False` breaks it.

## E4 — Sinusoidal position encoding (easy, 20 min)

1. Implement `positional_encoding(max_len, d_model)` from the Vaswani formula.
2. Plot the encoding as an image (`d_model` vs position).
3. Plot the dot product of the encoding at position 0 with encodings at positions 1..200. Observe slow decay.

## E5 — GNN sanity on Zachary's Karate Club (easy, 30 min)

**Data:** `torch_geometric.datasets.KarateClub()`.

1. Train a 2-layer `GCNConv` model to predict community labels.
2. Visualize node embeddings via t-SNE before and after training.
3. Confirm the two communities separate after training.

## E6 — EdgeConv block from scratch (medium, 60 min)

Implement EdgeConv in PyTorch:

```python
class EdgeConv(nn.Module):
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(),
            nn.Linear(d_out, d_out), nn.BatchNorm1d(d_out), nn.ReLU(),
        )

    def forward(self, x, coords):
        # x: (N_points, d_in), coords: (N_points, D) for k-NN
        # 1) build k-NN graph in coords
        # 2) for each edge (i, j), compute MLP([h_i, h_j - h_i])
        # 3) aggregate by max over j → new h_i
        ...
```

Test on random 100-point 3D point cloud. Output shape (100, d_out).

**Hint:** `torch_geometric.nn.knn_graph` or `knn`. Aggregate via `scatter_max`.

## E7 — JetClass mini: train vs MLP baseline (medium → hard, 120 min)

**Data:** small slice of JetClass (https://zenodo.org/records/6619768). Or use the Top Tagging Reference Dataset (Zenodo, Butter et al. 2019) — smaller, faster to iterate.

1. MLP baseline: flatten constituent features (pad to fixed length, e.g. 40), feed into a 3-layer MLP. Train on top tagging.
2. GNN: build a 3-layer EdgeConv model on the same data. Train identically.
3. Compare test AUC, test accuracy, parameter counts, wall-clock.

**Accept when:** GNN AUC ≥ MLP AUC + 3 percentage points.

## E8 — Single-head attention on jets (medium, 60 min)

Replace the EdgeConv blocks in E7 with self-attention over jet constituents (single-head, d_model=64). Use absolute pair-wise features (log ΔR, log k_T, log z) as an additive attention bias like Particle Transformer does.

Compare AUC against the plain EdgeConv GNN. Discuss: did the physics-bias help?

## E9 — Over-smoothing demonstration (medium, 45 min)

Train a GCN with 2, 4, 8, 16 layers on KarateClub (or Cora). Plot:
- Final accuracy vs depth.
- Cosine similarity between node embeddings at the output vs input.

The similarity should climb with depth — all nodes collapse to the same representation. Write 150 words interpreting this.

## E10 — Build a tracking-style edge classifier (hard, 90 min)

Simulate: 500 hits in 3D, with some following circular arcs (tracks) and some noise. Construct a candidate edge set via k-NN in (x, y, z) with k=8. Label edges by whether they connect two hits on the same track.

1. Train a GNN (2 message-passing layers) that classifies edges as true/false.
2. Evaluate: edge AUC, efficiency, purity.
3. Stratify errors by pT of the underlying track.

Small-scale, but it's exactly the shape of the Exa.TrkX pipeline.

---

## Solution hints

- E1 — If your softmax output rows don't sum to 1, you forgot `axis=-1`.
- E2 — parameter count derivation: W_Q, W_K, W_V are d_model×d_model; W_O is d_model×d_model. 4 × d_model².
- E3 — the test is subtle: future tokens in causal attention must not contribute; perturbing them must not change past outputs.
- E4 — the classic encoding image has alternating smooth/oscillatory rows.
- E5 — karate club is tiny; GCN converges in seconds. A good smoke test of your PyG install.
- E6 — max aggregation is key; mean often under-performs because informative outliers get averaged out.
- E7 — if your GNN is slower than your MLP, your k-NN is being rebuilt unnecessarily or you're not batching via PyG's disjoint-union scheme.
- E8 — pair-wise features are the whole point of Particle Transformer. If they don't help on your slice of JetClass, your slice is too small or your baseline is under-tuned.
- E9 — over-smoothing is easy to see; it's why GNNs don't stack like CNNs do.
- E10 — not SOTA, not even close. The point is that the structure is learnable.
