# Week 33 — Embeddings & Semantic Search

An embedding model maps text into a vector space where cosine similarity approximates
semantic relatedness — a learned inner product, and the coordinate system all of RAG
is built on.

## Objectives

- Explain how contrastive training (pull positives together, push negatives apart,
  in-batch negatives) shapes an embedding space.
- Choose and justify a similarity metric: cosine vs dot product vs Euclidean, and when
  normalization makes them coincide.
- Embed a physics-paper corpus with `sentence-transformers` and build a searchable
  index.
- Compare chunking strategies (fixed tokens, paragraphs, section-aware, overlap) with a
  retrieval probe rather than intuition.
- Probe an embedding space for structure and for failure modes (negation, numbers,
  notation).

## Core material (~3 hrs)

- Reimers & Gurevych, *Sentence-BERT* (arXiv 1908.10084) — why pooled BERT vectors
  needed a siamese objective.
- `sentence-transformers` docs: quickstart, pooling, and the losses overview
  (contrastive / MultipleNegativesRanking — note it is InfoNCE-style with in-batch
  negatives).
- Lilian Weng's blog post on contrastive representation learning (lilianweng.github.io).
- Skim the MTEB leaderboard/paper to see how embedding models are compared and what the
  task categories are.

## Exercises (built when the week starts)

1. Embed 1k heavy-ion abstracts (Week 27 corpus) with a sentence-transformers model.
   Accept when: matrix of shape (1000, d) saved, unit-norm check passes.
2. Metric comparison: for 20 hand-picked query abstracts, retrieve top-5 by cosine, dot,
   and Euclidean, pre- and post-normalization. Accept when: table shows where rankings
   agree/diverge and the normalization identity is verified numerically.
3. Structure probe: 2-D UMAP of the embeddings colored by arXiv category. Accept when:
   plot saved and clustering (or its absence) described in two sentences.
4. Failure probes: sentence pairs testing negation ("suppressed" vs "not suppressed"),
   numerals (200 vs 5020 GeV), and notation (π⁰ vs "neutral pion"). Accept when:
   similarity table with one-line verdict per probe.
5. Chunking bake-off: chunk the sPHENIX TDR three ways (fixed 256 tokens, fixed 512
   with overlap, section-aware); build an index per strategy; score recall@5 on 15
   hand-written questions with known source sections. Accept when: recall@5 per
   strategy in a table and a winner declared.
6. Persist the winning index (chroma or lancedb) with metadata (source, section, page).
   Accept when: a fresh process loads the index and answers a query without re-embedding.

## Deliverable

`Week33_Exercises.ipynb` with checks PASS; the persisted TDR + abstracts index with its
build script — Week 34 builds RAG directly on top of it; the chunking bake-off table.

## Review

- Week 12: you used UMAP on embeddings today. What does UMAP preserve, what does it
  distort, and why is it a visualization tool rather than a metric?
- Week 7: cosine similarity of unit vectors relates to Euclidean distance how? Derive
  ‖u − v‖² = 2(1 − cos θ) for unit vectors.
- Week 21: permutation invariance mattered for point clouds. Where does mean-pooling
  over token embeddings have the same character, and what does it lose?
- Week 32: your extractor pulled √s_NN from abstracts. Given today's numeral-probe
  result, predict a failure mode of purely dense retrieval for energy-specific queries.
