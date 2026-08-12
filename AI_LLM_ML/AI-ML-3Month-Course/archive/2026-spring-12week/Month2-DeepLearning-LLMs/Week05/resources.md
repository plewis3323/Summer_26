# Week 05 — Resources

## Docs

- **PyTorch Geometric.** https://pytorch-geometric.readthedocs.io/. Install per the docs' matrix (PyG's CUDA wheel must match torch's).
- **DGL (alternative).** https://www.dgl.ai/.
- **`torch.nn.functional.scaled_dot_product_attention`.** https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html. Fused, fast, uses FlashAttention when available.
- **HuggingFace transformers `modeling_attention`.** Skim to see how production codebases structure attention.

## Tutorials

- **PyG — "Getting Started".** https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html. Build a GCN in 30 minutes.
- **PyG Examples.** https://github.com/pyg-team/pytorch_geometric/tree/master/examples. Copy-paste-able templates.
- **The Annotated Transformer.** http://nlp.seas.harvard.edu/annotated-transformer/. Vaswani walked through line-by-line in PyTorch. Read this the day before Week 06.
- **Distill.pub — *Understanding Convolutions on Graphs*.** https://distill.pub/2021/understanding-gnns/. Conceptual; evergreen.

## Videos

- **Karpathy — "Let's build GPT: from scratch".** https://www.youtube.com/watch?v=kCc8FmEb1nY. Watch 0:00–0:20 this week.
- **Pascal Notin — GNN lectures (Oxford).** Clean pedagogy.
- **Jan LeCun lectures on self-supervision.** NYU, YouTube. Not strictly GNN but the priors-in-architecture discussion overlaps.
- **ML4Jets conference talks.** https://indico.cern.ch/ search "ML4Jets". HEP-specific GNN / transformer applications.

## GitHub repos

- **`jet-universe/particle_transformer`.** https://github.com/jet-universe/particle_transformer. Qu et al.'s official implementation.
- **`hqucms/weaver-core`.** The training framework behind ParticleNet/Particle Transformer. Learn to read this; it's the HEP-ML production reference.
- **`karpathy/nanoGPT`.** https://github.com/karpathy/nanoGPT. Peek at `model.py` this week; implement next week.
- **`pyg-team/pytorch_geometric`.** The source itself; `nn/conv/edge_conv.py` is short and clean.
- **`deepset-ai/FARM` / `langchain-ai/langchain`.** Not strictly needed yet, but peek ahead.

## Papers (full list with Zenodo/arXiv)

- Vaswani 2017 — arXiv:1706.03762
- ParticleNet — arXiv:1902.08570
- Particle Transformer — arXiv:2202.03772
- EdgeConv / DGCNN — arXiv:1801.07829
- Deep Sets — arXiv:1703.06114
- Energy Flow Networks — arXiv:1810.05165
- GAT — arXiv:1710.10903
- GCN — arXiv:1609.02907
- Exa.TrkX — arXiv:2103.06995
- Graph Nets (Battaglia et al. 2018, "Relational inductive biases") — arXiv:1806.01261. Skim; the taxonomy is useful.

## Datasets

- **Top Tagging Reference Dataset.** https://zenodo.org/records/2603256.
- **JetClass.** https://zenodo.org/records/6619768. ~200 GB full, slices available.
- **TrackML Particle Tracking Challenge.** Kaggle. Exa.TrkX-adjacent tracking data.
- **Cora, Citeseer, Pubmed.** Stock citation-network graphs; `torch_geometric.datasets.Planetoid`.

## Tooling

- **`torch.compile`.** Wrap your model for 1.5–2× speedup in recent PyTorch. Compilation overhead is a few seconds; amortized over training, it's a big win.
- **Flash Attention.** Gets used automatically via `scaled_dot_product_attention` on supported GPUs. For A100/H100, verify it's kicking in by checking GPU utilization.

## When you get stuck

- **PyG Slack.** Official; fast responses.
- **ML4Jets community.** HEP-specific; people who've fought these battles.
- **PyTorch Forums.** For general DL issues.

## Further reading

- **Battaglia et al. "Relational inductive biases, deep learning, and graph networks" (arXiv:1806.01261).** Canonical taxonomy.
- **Shlomi, Battaglia, Vlimant — "Graph Neural Networks in Particle Physics" (Machine Learning: Science and Technology 2020).** Dedicated HEP-GNN review.

---

Continue to Week 06. We build a transformer.
