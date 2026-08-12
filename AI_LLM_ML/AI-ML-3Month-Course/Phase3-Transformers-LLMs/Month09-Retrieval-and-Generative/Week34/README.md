# Week 34 — RAG over a Physics Document Library

Retrieval-augmented generation bolts a literature search onto a language model; like
any measurement chain, the system is only as good as its worst stage — and here that is
almost always retrieval, not generation.

## Objectives

- Build an end-to-end RAG pipeline: query → retrieve → rerank → synthesize with
  citations, over a physics document library you assemble (arXiv PDFs from your
  Week 27/32 corpus topics, plus — as a variant — any personal library you have:
  a Zotero export, a technical design report, internal notes).
- Implement hybrid search (BM25 + dense) with score fusion and show when each side wins.
- Add a cross-encoder reranker and quantify what it buys.
- Evaluate retrieval (recall@k, MRR — defined and worked in `lesson.md`) and
  generation (faithfulness/groundedness) separately, on a query set you wrote.
- Write and freeze the Capstone 3 held-out Q&A set (≥30 questions) before any tuning.
- Diagnose the classic failure modes: bad chunking, lost context, unfaithful synthesis,
  and answers that ignore retrieved text.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: BM25 term by term, reciprocal rank
  fusion, recall@k/MRR with worked examples, reranking, citation prompts,
  faithfulness.
- BM25: Robertson & Zaragoza's probabilistic-relevance framework survey — the source
  for the tf-saturation and length-normalization terms.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* —
  the original framing; skim, the architecture differs from modern pipeline RAG.
- `chromadb` (Week 33's store) and `rank_bm25` library docs.
- One reranker model card (e.g. a cross-encoder from sentence-transformers) — note the
  latency/quality trade vs bi-encoders.

## Exercises

See `exercises.md` (notebook generated when the week starts). The exercises ingest
your assembled library, write a 25-question dev set plus the frozen ≥30-question
Capstone 3 held-out set, then climb the retrieval ladder — BM25 → dense → hybrid
RRF → cross-encoder rerank — measuring recall@5/MRR at each rung, and finish with
cited answers and a faithfulness eval from one command.

## Deliverable

A `rag/` package: ingest, index, retrieve (BM25/dense/hybrid), rerank, answer — plus
`eval/dev_set.jsonl`, the frozen `eval/heldout_set.jsonl` (hash committed), and a
metrics table from one command. This is the retrieval core of Capstone 3 option (a).

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
