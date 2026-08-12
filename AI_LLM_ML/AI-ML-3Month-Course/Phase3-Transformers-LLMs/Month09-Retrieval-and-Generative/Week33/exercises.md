# Week 33 — Exercises

Work top to bottom. Setup (imports, model loading, the Week 27 abstracts corpus,
constants) is given by the notebook; you write only the lines each exercise asks
for. E6's index build script lives in a file (`build_index.py`); everything else is
notebook cells.

You need one *long* physics document (30+ pages) for E5–E6: a review article whose
PDF you download from arXiv (e.g. search arXiv for a heavy-ion or quark-gluon
plasma review), or — variant — a document you already have, such as your
experiment's technical design report or lecture notes. The notebook's setup extracts
its text to `data/longdoc.txt`; note the source in one provenance line.

## E1 — Embed the corpus

Embed 1,000 abstracts from your Week 27 corpus with `all-MiniLM-L6-v2`
(`normalize_embeddings=True`); save the matrix to `data/abstract_emb.npy`.
Hint: `model.encode` takes a list and returns an `(n, d)` array in one call.
Accept when: saved matrix has shape (1000, 384) and every row norm is within
1e-3 of 1.0.

## E2 — Metric comparison

For 20 query abstracts (given), retrieve top-5 neighbors by cosine, dot product,
and Euclidean distance — on the raw (unnormalized) embeddings and again after
normalization. Count rank agreements and verify $\|\hat{u}-\hat{v}\|^2 =
2(1-\cos\theta)$ numerically on 100 random pairs.
Hint: re-encode with `normalize_embeddings=False` for the raw case; cosine needs
norms, dot does not.
Accept when: post-normalization, all three metrics give identical top-5 sets for
all 20 queries; the identity holds to 1e-6; the pre-normalization disagreement
count is reported.

## E3 — Structure probe

Compute a 2-D UMAP (Week 12) of the 1,000 embeddings, colored by arXiv primary
category from the corpus metadata; save the figure.
Hint: `umap-learn` on unit vectors with `metric="cosine"`.
Accept when: `results/umap_categories.png` exists plus two sentences stating
whether categories cluster and one caveat about reading UMAP geometry.

## E4 — Failure probes

Score similarity for given sentence pairs testing negation ("suppressed" vs "not
suppressed"), numerals (200 vs 5020 GeV), and notation ($\pi^0$ vs "neutral pion"
vs "pi zero"); build a table with one-line verdict per probe.
Hint: each probe is one dot product of two unit vectors; "high" means near the
paraphrase scores you saw in E2, not near 1.0.
Accept when: table has all 3 probe families, each verdict says whether the model
distinguishes the pair, and the negation and numeral similarities are compared
against a true-paraphrase similarity as reference.

## E5 — Chunking bake-off

Chunk the long document three ways: fixed 256 tokens, fixed 512 with 25% overlap,
and section-aware. Build one in-memory index per strategy. Score recall@5 on the 15
provided questions with known source sections: a hit means a top-5 chunk overlaps
the gold section.
Hint: reuse the lesson's `chunk_words`; for section-aware, split on the section
headings that survived text extraction, then merge pieces under 50 words.
Accept when: a 3-row table reports recall@5 per strategy and one sentence declares
the winner and the margin.

## E6 — Persist the winning index

Write `build_index.py`: embed the winning chunking of the long document plus the
1,000 abstracts into a persistent chromadb collection with metadata (`source`,
`section`, `page` where known). Then, in a fresh process, load and answer one query
without re-embedding.
Hint: `chromadb.PersistentClient(path=...)`; the fresh-process check is
`!{sys.executable} query_index.py` (given) from the notebook.
Accept when: `python3 build_index.py` rebuilds the index from scratch; the fresh
process returns top-5 hits with metadata for a test query in under 5 seconds; this
is the index Week 34 builds on.

## Review

1. (Week 12) You used UMAP in E3. What does UMAP preserve, what does it distort,
   and why is it a visualization tool rather than a metric?
2. (Week 07) Derive $\|u - v\|^2 = 2(1 - \cos\theta)$ for unit vectors — this
   week's E2 identity, from the algebra alone.
3. (Week 21) Mean-pooling token embeddings is permutation-invariant, like a GNN
   readout. What information does the pooling itself discard, and what restores it
   upstream?
4. (Week 32) Your extractor pulled $\sqrt{s_{NN}}$ from abstracts. Given E4's
   numeral probe, predict the failure mode of purely dense retrieval for
   energy-specific queries and name the fix Week 34 introduces.
