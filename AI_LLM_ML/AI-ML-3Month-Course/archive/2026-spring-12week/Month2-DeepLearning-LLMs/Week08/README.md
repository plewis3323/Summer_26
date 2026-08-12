# Week 08 — Retrieval-Augmented Generation (RAG), Embeddings, Vector Databases

## Where this sits

Last week you fine-tuned a model to produce good summaries. But it's still a closed system — whatever it doesn't know, it makes up. Fine-tuning can't cheaply keep pace with the arXiv firehose, nor give you citations traceable to source documents. The answer is RAG: **Retrieval-Augmented Generation**. You store a corpus as embeddings, retrieve the top-k relevant chunks for a query, stuff them into the prompt, and have the LLM answer with grounding.

This is the workflow behind most production LLM applications in 2024–2026 that do anything useful with private or dynamic data. Claude Projects, ChatGPT Custom GPTs, GitHub Copilot Chat, Perplexity — all are RAG over different corpora. You'll build one over sPHENIX / STAR / RHIC papers and internal notes. By the end, you'll have something you use.

## Learning objectives

By the end of the week you can:

1. Explain the RAG pipeline end-to-end: chunking → embedding → indexing → query embedding → retrieval → reranking → prompt assembly → generation → citation rendering.
2. Pick an embedding model (sentence-transformers family, OpenAI `text-embedding-3-*`, Cohere, Voyage, Jina) with informed tradeoffs on cost, quality, and context length.
3. Stand up a vector database (Chroma, LanceDB, Qdrant, or pgvector) — know when to reach for each.
4. Chunk a PDF corpus intelligently: by semantic structure, not fixed token counts. Handle equations, figures, tables.
5. Implement hybrid search: dense (embeddings) + sparse (BM25) with Reciprocal Rank Fusion.
6. Add a cross-encoder reranker and measure its effect on answer quality.
7. Evaluate RAG systematically: recall@k, MRR, faithfulness, answer quality. Use `ragas`.
8. Mitigate the most common failure modes: stale index, context window blow-ups, citation hallucination, irrelevant retrievals.

## The pipeline, step by step

### 1. Ingest
PDFs → text. Use `pypdf2` or `pdfplumber` for simple docs; `marker-pdf` or `nougat` for papers with equations. Parse metadata: title, authors, arXiv ID, DOI, section structure.

### 2. Chunk
Break documents into retrieval units, typically 200–800 tokens each with overlap of 50–100 tokens. Options:
- **Fixed-size.** Simple. Breaks semantic boundaries.
- **Recursive character splitting.** Try to split on paragraph / sentence / word boundaries. `langchain`'s `RecursiveCharacterTextSplitter`. Decent default.
- **Semantic chunking.** Embed sentence by sentence, group by similarity. Better for mixed-content docs.
- **Structural chunking.** Split on section headings. Best when structure is present (papers, theses).

For arXiv papers, section-aware splitting wins. A chunk should be semantically coherent; an "Introduction" paragraph is.

### 3. Embed
For each chunk, compute a vector. Embedding model choices:

| Model | Dim | Context | Cost | Best for |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 512 | Free (local) | Dev, prototyping |
| `BAAI/bge-base-en-v1.5` | 768 | 512 | Free (local) | Strong general-purpose |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | Free (local) | Quality-first local |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | Free (local) | Long-context local |
| `jinaai/jina-embeddings-v3` | 1024 | 8192 | Free (local) | Multilingual |
| OpenAI `text-embedding-3-small` | 1536 | 8192 | $0.02/M tok | API convenience |
| OpenAI `text-embedding-3-large` | 3072 | 8192 | $0.13/M tok | Top quality API |
| Voyage AI `voyage-3` | 1024 | 32000 | $0.06/M tok | Long context API |

For your corpus (5–10k papers, 1–5 GB text), a local BGE-base is fine. Switch to voyage or OpenAI only if retrieval quality is the bottleneck.

### 4. Store
Vector database. Choices:
- **Chroma.** Python-native, disk-persisted, easy. Great for <1M vectors.
- **LanceDB.** Columnar, fast. Supports hybrid search out of the box.
- **Qdrant.** Self-hosted or cloud. Production-grade. Rust core.
- **Weaviate.** Full-text + vector. Heavier.
- **pgvector.** PostgreSQL extension. If you already have Postgres, use this.
- **Pinecone / Turbopuffer / Vespa** — managed services, expensive, scalable.

For your scale, LanceDB or Chroma. Use LanceDB — it does hybrid search natively.

### 5. Query
User asks a question. Embed the query with the *same embedding model* (critical). Search the index for top-k nearest chunks (usually cosine similarity). Retrieve, say, 20 candidates.

### 6. Rerank (often skipped; often critical)
A cross-encoder takes (query, chunk) pairs and scores their relevance more accurately than the bi-encoder used for embedding. Models: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`, or API services. Rerank the 20 candidates, keep top 5.

### 7. Assemble prompt
Inject retrieved chunks into a prompt template:

```
You are an expert heavy-ion physicist. Use ONLY the provided excerpts
to answer. Cite each claim inline with [1], [2], etc.

Excerpts:
[1] {chunk_1}
[2] {chunk_2}
...

Question: {question}

Answer:
```

### 8. Generate
LLM produces the answer. Use Claude Sonnet for quality, or your locally-served fine-tune for speed.

### 9. Render
Show the answer with citations linking back to the source chunks. Highlight the excerpt that supported each claim. This is the UX that makes RAG trustworthy.

## Chunking: the unglamorous crux

RAG quality lives and dies on chunking. Bad chunks = bad retrieval = bad answers. Rules of thumb:

- Chunks should be semantically coherent and self-contained. A lone sentence is often too small; a full section often too big.
- Overlap chunks by ~15% to reduce boundary-effect misses.
- For papers, use the section headings as natural boundaries. If a section is too big, subdivide by paragraphs.
- For equations: keep the LaTeX + surrounding sentence together. Don't orphan equations.
- For tables: if possible, serialize as markdown tables. Embedded tables as text often retrieve better than image-embedded tables.
- Store metadata per chunk: `(doc_id, section, page, arxiv_id, authors, year)`. You'll use these for filtering.

## Hybrid search: dense + sparse

Dense-only retrieval misses queries with rare terms (acronyms, author names, unusual technical terms). Sparse (BM25) catches those. Combine via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ_retriever  1 / (k + rank_retriever(d))
```

with k=60 canonically. Simple, no training, works well.

LanceDB does RRF natively. So does Qdrant. Elasticsearch has it. Worth enabling.

## Reranking: diminishing returns, worth it

A cross-encoder scores each (q, d) pair. More expensive than bi-encoder embedding but more accurate. Typical gains: +5 to +15 percentage points on recall@5.

Use one. Cross-encoder of choice: `BAAI/bge-reranker-base` or `mixedbread-ai/mxbai-rerank-base-v1`. Open-weights, runs on your GPU, maybe 50ms for 20 pairs.

## Evaluation

Without evaluation, RAG quality is vibes. Metrics:

### Retrieval
- **Recall@k.** For a labeled set of (query, gold_chunks), what fraction of gold chunks appear in your top-k?
- **MRR (Mean Reciprocal Rank).** How high in the ranking is the first gold chunk on average?

### Generation
- **Faithfulness.** Does the answer derive only from the retrieved context? (Can be scored by an LLM-judge.)
- **Answer relevance.** Does the answer address the question? (LLM-judge.)
- **Citation precision/recall.** For claims in the answer, is there a retrieved chunk that supports it?

### The tool
`ragas` (https://github.com/explodinggradients/ragas) automates these. It uses an LLM-judge internally, which has its own biases, but it's the best lightweight option.

Build a gold set: 30 questions with known answers (or known source docs). Compute metrics on this. Iterate.

## Common failure modes

**Retrieved chunks look great; answer hallucinates.** Your generation prompt doesn't constrain the model. Add: "If the context doesn't contain the answer, say 'I can't find that in the provided context.'"

**Retrieval misses obvious queries.** Chunks too big (query embeds a small concept, but chunk dilutes it) or too small (lose context). Try multiple chunk sizes.

**Acronyms don't retrieve.** BM25/sparse part of hybrid search fixes this. Enable it.

**Stale index.** You re-fine-tune, or add new papers, and forget to re-embed. Build an incremental pipeline: hash each chunk, skip if unchanged.

**Context window blows up.** You retrieved k=20 chunks of 800 tokens each = 16k tokens of context, plus question, plus system prompt. Most models handle it now, but tokens are expensive. Rerank aggressively, keep top 3–5.

**Wrong embedding at query time.** You used `bge-base` for indexing and `bge-large` at query. The embeddings live in different spaces — results are garbage. Always verify model name match.

**PDF parsing inserts garbage.** Column-wrapped text from double-column papers becomes unreadable. Inspect extracted text before indexing. Use `marker-pdf` for papers.

**Reranker over-prefers lexically-similar chunks.** Some cross-encoders are biased. Evaluate with and without.

## HEP angle

This becomes useful immediately:

- Queries like "What v2 values did STAR report for charm mesons at 200 GeV Au+Au?" should surface the relevant papers and excerpts with citations.
- "Show me all sPHENIX notes mentioning the INTT alignment." — works over internal PDFs.
- "Find three papers that use similar background-subtraction techniques to what I'm writing about." — works, with discipline.

This is the tool. Building it gives you a workflow you'd use daily.

## Tooling introduced

- `sentence-transformers` — embedding models.
- `lancedb` or `chromadb` — vector DB.
- `rank-bm25` — BM25 sparse search.
- `marker-pdf` — modern PDF-to-markdown parser for scientific papers.
- `ragas` — RAG evaluation.
- `reranker` packages (via HF transformers).

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | Reading: RAG paper, Matryoshka embeddings. Install stack. |
| 2 | PDF ingestion + chunking on 50 papers. |
| 3 | Embed + index. First end-to-end retrieval. |
| 4 | Hybrid search + reranker. |
| 5 | Build 30-question gold set. Run ragas. |
| 6 | Generation loop with citations. Iterate on prompt. |
| 7 | Write-up. |

## What "done" looks like

- An indexed vector DB over 100–500 arXiv heavy-ion papers.
- A CLI or tiny web app: `ask "What is the pT range of jet quenching measurements at sPHENIX?"` → returns grounded answer with 3–5 citations.
- Gold-set evaluation metrics: recall@5, MRR, faithfulness (via ragas).
- At least one ablation: dense-only vs hybrid, no-rerank vs rerank.
- A 500-word reflection on where RAG helps most and where it's brittle.

Week 09 we add tool use — your RAG system starts making calls to Python/SQL/external APIs to get structured data, not just text. That's the first real step toward agents.
