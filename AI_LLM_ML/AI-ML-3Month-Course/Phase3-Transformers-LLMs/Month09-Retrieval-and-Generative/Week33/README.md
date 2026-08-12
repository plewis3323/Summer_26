# Week 33 — Embeddings & Semantic Search

An embedding model maps text into a vector space where cosine similarity approximates
semantic relatedness — a learned inner product, and the coordinate system all of RAG
is built on.

## Objectives

- Explain how contrastive training (pull positives together, push negatives apart,
  in-batch negatives) shapes an embedding space; derive the InfoNCE loss and its
  gradient's pull/push behavior.
- Choose and justify a similarity metric: cosine vs dot product vs Euclidean, and when
  normalization makes them coincide.
- Embed a physics-paper corpus with `sentence-transformers` and build a searchable
  index.
- Compare chunking strategies (fixed tokens, overlap, section-aware) with a
  retrieval probe rather than intuition.
- Probe an embedding space for structure and for failure modes (negation, numbers,
  notation).

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: cosine similarity derived from the
  dot product, the InfoNCE derivation, chunking, vector stores, and failure probes.
- Reimers & Gurevych, *Sentence-BERT* (arXiv 1908.10084) — why pooled BERT vectors
  needed a siamese objective.
- `sentence-transformers` docs: quickstart, pooling, and the losses overview
  (MultipleNegativesRanking is the lesson's InfoNCE with in-batch negatives).
- Lilian Weng's blog post on contrastive representation learning (lilianweng.github.io).
- Skim the MTEB leaderboard/paper to see how embedding models are compared and what the
  task categories are.

## Exercises

See `exercises.md` (notebook generated when the week starts). The exercises embed
1k Week 27 abstracts, verify the metric identities numerically, probe the space
(UMAP, negation/numeral/notation), run a chunking bake-off on a long physics
document you download, and persist the winning index that Week 34 builds RAG on.

## Deliverable

`Week33_Exercises.ipynb` with checks PASS; the persisted abstracts + long-document
index with its `build_index.py` — Week 34 builds RAG directly on top of it; the
chunking bake-off table.

## Review

- Week 12: you used UMAP on embeddings today. What does UMAP preserve, what does it
  distort, and why is it a visualization tool rather than a metric?
- Week 7: cosine similarity of unit vectors relates to Euclidean distance how? Derive
  ‖u − v‖² = 2(1 − cos θ) for unit vectors.
- Week 21: permutation invariance mattered for point clouds. Where does mean-pooling
  over token embeddings have the same character, and what does it lose?
- Week 32: your extractor pulled √s_NN from abstracts. Given today's numeral-probe
  result, predict a failure mode of purely dense retrieval for energy-specific queries.
