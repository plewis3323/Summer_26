# Week 34 — RAG over the sPHENIX TDR

Retrieval-augmented generation bolts a literature search onto a language model; like
any measurement chain, the system is only as good as its worst stage — and here that is
almost always retrieval, not generation.

## Objectives

- Build an end-to-end RAG pipeline: query → retrieve → rerank → synthesize with
  citations, over the sPHENIX Technical Design Report and your Zotero library.
- Implement hybrid search (BM25 + dense) with score fusion and show when each side wins.
- Add a cross-encoder reranker and quantify what it buys.
- Evaluate retrieval (recall@k, MRR) and generation (faithfulness/groundedness)
  separately, on a query set you wrote.
- Diagnose the classic failure modes: bad chunking, lost context, unfaithful synthesis,
  and answers that ignore retrieved text.

## Core material (~3 hrs)

- BM25: any solid reference on the scoring function (e.g. the Robertson & Zaragoza
  probabilistic-relevance framework survey) — understand tf saturation and length
  normalization terms.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* —
  the original framing; skim, the architecture differs from modern pipeline RAG.
- `chroma` or `lancedb` docs (whichever you persisted in Week 33), plus the
  `rank_bm25` or equivalent library docs.
- One reranker model card (e.g. a cross-encoder from sentence-transformers) — note the
  latency/quality trade vs bi-encoders.

## Exercises (built when the week starts)

1. Eval set first: 25 questions over the TDR + Zotero corpus, each with the gold
   source passage(s) recorded. Accept when: `eval_set.jsonl` exists with question,
   gold-source, and a reference answer per row.
2. BM25 baseline over the Week 33 chunks. Accept when: recall@5 and MRR on the eval
   set reported.
3. Dense retrieval (Week 33 index) on the same eval set. Accept when: same metrics
   reported, plus a table of three queries where BM25 wins (exact tokens: energies,
   detector acronyms) and three where dense wins (paraphrase).
4. Hybrid fusion via reciprocal rank fusion. Accept when: hybrid recall@5 ≥ each
   individual method on the eval set, or the exception is explained in two lines.
5. Cross-encoder reranking of top-20 → top-5. Accept when: recall@5 delta and added
   latency per query both reported.
6. Full RAG answers with inline source citations for all 25 questions. Accept when:
   every answer cites ≥ 1 retrieved chunk and cited chunks actually contain the
   supporting text (spot-check 10 by hand).
7. Faithfulness eval: judge each answer as grounded / partially grounded / hallucinated
   against its retrieved context (LLM-judge + your spot-check, per Week 32). Accept
   when: grounded rate reported and one hallucinated answer dissected.

## Deliverable

A `rag/` package: ingest, index, retrieve (BM25/dense/hybrid), rerank, answer — plus
`eval_set.jsonl` and a metrics table from one command. This is the retrieval core of
Capstone 3 option (a).

## Review

- Week 33: why can cosine similarity miss an exact-match query like "√s_NN = 200 GeV"
  that BM25 nails? Tie to the numeral probe.
- Week 10: recall@k is recall. Write the precision/recall definitions and state what
  plays the role of "positive" in retrieval eval.
- Week 25: a cross-encoder scores (query, passage) jointly; a bi-encoder scores dot
  products of separate encodings. Which is more expressive and why — answer in terms
  of attention across the pair.
- Week 4: your index must be rebuildable. What provenance does the ingest script need
  to record for "fresh clone → same index"?
