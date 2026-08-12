# Week 34 — Exercises

Work top to bottom. Setup (imports, model loading, the Week 33 index, constants) is
given by the notebook; you write only the lines each exercise asks for. E1's
ingestion and the E8 pipeline live in files (a `rag/` package: `ingest.py`,
`retrieve.py`, `answer.py`); the eval sets are `.jsonl` files; everything else is
notebook cells.

## E1 — Assemble and ingest the library

Download 20–50 arXiv PDFs on your Week 27/32 corpus topics (or use your personal
library — Zotero export, technical design report — as the variant), extract text
per page with `pypdf`, chunk with the Week 33 winning strategy, and add everything
to the Week 33 index with `source`/`section`/`page` metadata plus a provenance
manifest.
Hint: keep `(page, text)` pairs through chunking so page numbers survive into
metadata.
Accept when: `python3 rag/ingest.py` rebuilds the index from `data/pdfs/`;
manifest lists every document with source id, date, page count; a spot query
returns chunks with correct page metadata.

## E2 — Two eval sets, written before tuning

Write (a) a dev set: 25 questions over your library, each with gold chunk id(s)
and a one-line reference answer, as `eval/dev_set.jsonl`; (b) the Capstone 3
held-out set: ≥30 *different* questions, same format, as `eval/heldout_set.jsonl`
— committed, hashed (record `sha256sum` output in the commit message), and not
opened again until Week 36.
Hint: write questions from the documents (point at the passage first, then phrase
the question), and vary type: factual lookup, paraphrase, exact-number, synthesis.
Accept when: both files validate (question, gold ids that exist in the index,
reference answer per row); zero question overlap between sets; the held-out hash
is committed.

## E3 — BM25 baseline

Index all chunks with `rank_bm25`; report recall@5 and MRR on the dev set.
Hint: tokenize query and chunks identically (lowercase split is fine).
Accept when: both metrics reported; hand-check one query's computation (the gold
rank and its reciprocal) against the metric code.

## E4 — Dense retrieval, same eval

Score the Week 33 dense index on the same dev set; then build a 3+3 table: three
queries where BM25 wins (exact tokens: energies, detector acronyms) and three
where dense wins (paraphrase).
Hint: "wins" = better gold rank on that query.
Accept when: recall@5 and MRR reported on the same table as E3; the 3+3 table has
a one-line reason per row.

## E5 — Hybrid fusion

Implement reciprocal rank fusion (k = 60) over the BM25 and dense rankings; score
it on the dev set.
Hint: the lesson's `rrf` function is ~7 lines; ranks count from 1.
Accept when: hybrid recall@5 ≥ each individual method on the dev set, or the
exception is explained in two lines.

## E6 — Cross-encoder reranking

Rerank hybrid top-20 down to top-5 with a `sentence-transformers` cross-encoder;
report the recall@5 and MRR deltas and the added latency per query.
Hint: `CrossEncoder.predict` on a list of (query, chunk) tuples; time with
`time.perf_counter` averaged over the dev set.
Accept when: deltas and ms/query both reported; one sentence says whether the
trade is worth it for your corpus.

## E7 — Full RAG answers with citations

Generate answers for all 25 dev questions using the lesson's citation prompt over
reranked top-5, with your Week 29–32 instruct model.
Hint: number the passages in the prompt and require "[n]" citations; parse them
back out with a plain string scan.
Accept when: every answer cites ≥ 1 retrieved chunk; for 10 hand-checked answers,
the cited chunks actually contain the supporting text.

## E8 — Faithfulness eval + one command

Judge each answer grounded / partially grounded / hallucinated against its own
retrieved context (LLM-judge + hand-label 10 to validate the judge, per Week 32).
Then wire `python3 rag/run_eval.py` to produce the full metrics table (BM25,
dense, hybrid, +rerank: recall@5, MRR; generation: grounded rate) in one command.
Hint: the judge prompt shows the answer and its passages only — not the reference
answer.
Accept when: grounded rate reported with judge–human agreement on the 10; one
hallucinated answer dissected in three sentences; the one-command table
reproduces.

## Review

1. (Week 33) Why can cosine similarity miss an exact-match query like
   "√s_NN = 200 GeV" that BM25 nails? Tie to the numeral probe.
2. (Week 10) recall@k is recall. Write the precision and recall definitions and
   state what plays the role of "positive" in retrieval evaluation.
3. (Week 25) A cross-encoder scores (query, passage) jointly; a bi-encoder scores
   dot products of separate encodings. Which is more expressive and why — answer
   in terms of attention across the pair.
4. (Week 04) Your index must be rebuildable. What provenance does `ingest.py`
   record so that "fresh clone → same index" holds?
