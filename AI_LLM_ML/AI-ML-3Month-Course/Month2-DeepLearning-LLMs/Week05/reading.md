# Week 05 — Reading

## Primary

1. **Vaswani et al. — *Attention Is All You Need* (NeurIPS 2017). arXiv:1706.03762.**
   - Read §1–3 thoroughly. §3.2.2 on multi-head attention; §3.3 on position-wise FFN; §3.5 on positional encoding.
   - Appendices on optimizer and schedule — useful for Week 06.
   - You'll re-read this before implementing in Week 06. Annotate generously.

2. **Qu & Gouskos — *ParticleNet: Jet Tagging via Particle Clouds* (Phys. Rev. D 2020). arXiv:1902.08570.**
   - Read end-to-end. §2 (data representation), §3 (EdgeConv), §4 (ParticleNet architecture), §5 (experiments).
   - The key insight: jets have permutation-invariant structure; a k-NN graph respects it while CNNs don't.

3. **Qu, Li, Qian — *Particle Transformer for Jet Tagging* (ICML 2022). arXiv:2202.03772.**
   - Read §2 (architecture) and §3 (pair-wise features). The "interaction matrix" is the important piece — it's how physics priors enter an attention layer.

4. **Murphy, *Probabilistic Machine Learning: Advanced Topics*, Ch. 16 "Attention".**
   - Free online. Modern mathematical treatment. Good companion to Vaswani.

## Supplementary

5. **Hamilton, Ying, Leskovec — *GraphSAGE* (NeurIPS 2017). arXiv:1706.02216.**
   - Skim. You don't need this for jets; useful to see the mini-batching paradigm for large graphs.

6. **Veličković et al. — *Graph Attention Networks* (ICLR 2018). arXiv:1710.10903.**
   - Read §2–3. The bridge between GNNs and transformers: attention as edge weights.

7. **Wang et al. — *Dynamic Graph CNN for Learning on Point Clouds* (TOG 2019). arXiv:1801.07829.**
   - Original EdgeConv paper. §3.2 for the operator; §4 for point-cloud applications.

8. **Komiske, Metodiev, Thaler — *Energy Flow Networks* (JHEP 2019). arXiv:1810.05165.**
   - Read §2–3. Deep Sets applied to jets; IRC safety as an architectural constraint.

9. **Ju et al. — *Performance of a geometric deep learning pipeline for HL-LHC particle tracking* (Exa.TrkX collaboration, EPJ C 2021). arXiv:2103.06995.**
   - Skim. §2 architecture; §4 performance. The flagship HEP tracking GNN.

10. **Zaheer et al. — *Deep Sets* (NeurIPS 2017). arXiv:1703.06114.**
    - Short. Theorem 2 is the key statement.

## Videos

- **Karpathy "Let's build GPT: from scratch" (Lecture 6).** ~1h56m. Watch 0:00–0:20 this week (embeddings, causal mask, bigram baseline). The rest is Week 06.
- **Yannic Kilcher paper review of "Attention Is All You Need".** ~45 min. A bit irreverent, but he gets the shape of things right.
- **Pascal Notin lectures on GNNs (Oxford 2021, YouTube).** ~2 hours total if you watch two lectures. Clean pedagogy.
- **HSF training — ML4Jets tutorials.** https://indico.cern.ch/ search for "ML4Jets". HEP-specific GNN tutorials from domain experts.

## HEP-ML Living Review

Browse:
- "Classification / Graph neural networks"
- "Classification / Transformers"
- "Reconstruction / Tracking" — for Exa.TrkX and follow-ups.

## Codecademy (Pro)

Nothing directly on GNNs. If you want more PyTorch practice, continue "PyTorch for Deep Learning" sequence module.

## Notes to take

- One-page derivation of scaled dot-product attention, including the variance argument for √d_k scaling.
- A table: CNN / self-attention / GNN → data structure / inductive bias / complexity.
- Summary bullets for ParticleNet and Particle Transformer: input, architecture, key physics priors, SOTA comparisons. These are reference your advisor will recognize.
- One paragraph: "If I were to use a Particle Transformer on sPHENIX jets, what pair-wise features would I include?" Phrased for a group meeting.

Reading is heavier than previous weeks. Don't skip Vaswani or ParticleNet.
