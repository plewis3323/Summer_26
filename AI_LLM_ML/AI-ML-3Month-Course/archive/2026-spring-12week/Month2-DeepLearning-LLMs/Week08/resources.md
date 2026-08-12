# Week 08 — Resources

## Docs

- **sentence-transformers** — https://sbert.net
- **LanceDB** — https://lancedb.com/docs/
- **ChromaDB** — https://docs.trychroma.com/
- **Qdrant** — https://qdrant.tech/documentation/
- **ragas** — https://docs.ragas.io/
- **LangChain RAG** — https://python.langchain.com/docs/tutorials/rag/
- **LlamaIndex** — https://docs.llamaindex.ai/
- **marker-pdf** — https://github.com/VikParuchuri/marker
- **nougat** (Meta's scientific-PDF parser) — https://github.com/facebookresearch/nougat

## Embedding models

- **sentence-transformers/all-MiniLM-L6-v2** — 384d, fast, baseline. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **BAAI/bge-base-en-v1.5** — 768d, strong default. https://huggingface.co/BAAI/bge-base-en-v1.5
- **BAAI/bge-large-en-v1.5** — 1024d, slower, higher quality.
- **BAAI/bge-m3** — multi-functional (dense + sparse + colbert). Recent, impressive.
- **nomic-ai/nomic-embed-text-v1.5** — 768d, 8k context, Matryoshka.
- **jinaai/jina-embeddings-v3** — multilingual, long context.
- **mixedbread-ai/mxbai-embed-large-v1** — 1024d, competitive with OpenAI.
- **OpenAI `text-embedding-3-small/-large`** — API, 1536/3072d.
- **Voyage AI `voyage-3`** — API, 1024d, 32k context.
- **Cohere `embed-english-v3.0`** — API, 1024d.

For your corpus, local BGE-base is enough. Move to `text-embedding-3-large` or Voyage only if quality is the pain point.

## Rerankers

- **BAAI/bge-reranker-base / -large** — open, solid.
- **mixedbread-ai/mxbai-rerank-base-v1 / -large-v1** — competitive, sometimes better on domain text.
- **Cohere Rerank API** — Cohere API; very good, costs money.
- **Jina Reranker** — open.

## Vector DBs (beyond LanceDB)

- **Chroma** — easiest for dev.
- **Qdrant** — Rust core, production-ready, open or cloud.
- **Weaviate** — full-text + vector, heavier.
- **Milvus / Zilliz** — at-scale workhorse.
- **pgvector** — Postgres extension. Use this if you already run Postgres.
- **Turbopuffer** — serverless, cheap, fast. Hosted only.
- **Pinecone** — managed, mature, expensive.
- **Vespa** — Yahoo's, big-scale, complex.

## PDF parsers

- **pypdf / pdfplumber** — simple, fast, fails on complex layouts.
- **marker-pdf** — modern, LLM-based, outputs markdown, handles equations.
- **nougat** (Meta) — research papers, trained on arXiv.
- **unstructured** — general-purpose, handles many formats.
- **Docling** (IBM) — newer entry, strong benchmarks.
- **PyMuPDF (fitz)** — robust general-purpose, no LLM.

For papers, `marker-pdf` or `nougat`. For text-heavy reports, `pypdf` is fine.

## Tutorials

- **LangChain RAG tutorial** — https://python.langchain.com/docs/tutorials/rag/
- **LlamaIndex RAG tutorial** — https://docs.llamaindex.ai/en/stable/understanding/rag/
- **Claude cookbook RAG recipe** — https://github.com/anthropics/anthropic-cookbook/tree/main/skills/retrieval_augmented_generation
- **Weaviate academy** — https://weaviate.io/developers/academy
- **Pinecone learn** — https://www.pinecone.io/learn/

## Papers

- Lewis RAG — arXiv:2005.11401
- DPR — arXiv:2004.04906
- Sentence-BERT — arXiv:1908.10084
- Matryoshka — arXiv:2205.13147
- ColBERT — arXiv:2004.12832
- Contextual Retrieval (Anthropic blog, 2024)
- RAG survey — arXiv:2312.10997
- BEIR benchmark — arXiv:2104.08663
- MS MARCO — arXiv:1611.09268
- Self-RAG — arXiv:2310.11511
- RAFT — arXiv:2403.10131 (fine-tuning for RAG)

## Videos

- **3Blue1Brown — not relevant this week.**
- **Andrej Karpathy — not this week.**
- **DeepLearning.ai short courses — "Building Applications with Vector Databases", "Advanced Retrieval for AI".** Both ~1h, free. Worth it.
- **Cohere — *The State of Retrieval*** — various 30-min talks.
- **Pinecone Community talks** — various.

## When you get stuck

- **LangChain / LlamaIndex Discords.** Both very active.
- **r/LocalLLaMA.** RAG-heavy traffic.
- **HuggingFace Forums.** For embedding model issues.
- **Anthropic Developer Discord.**

## HEP-specific

- **INSPIRE-HEP API** — https://inspirehep.net/api. Structured metadata (citations, authors, abstracts). Adjacent to RAG; can answer many queries without full-text.
- **NASA ADS** — https://ui.adsabs.harvard.edu/help/api/. Broader than INSPIRE; covers astro too.
- **arXiv OAI-PMH** — bulk metadata harvesting.
- **sPHENIX / STAR / ALICE internal note systems** — if you have access, these are higher-signal than public papers for operational questions. Worth indexing if you can.

## Tooling

- **typer** — nice Python CLI library.
- **streamlit** — trivial web UI.
- **gradio** — alternative to streamlit; better HF integration.
- **rich** — pretty console output.
- **`tiktoken`** — count tokens to check prompt size.

## Cost awareness

- Local embeddings: free. GPU time negligible for your scale.
- marker-pdf: local. 10-30s per paper on GPU.
- Claude Sonnet for generation: ~$0.003 per question answer at typical context sizes. $10 of budget = ~3000 questions.
- ragas: 4 metrics × 30 questions × ~2 LLM calls each = ~240 LLM calls. ~$0.50.

## Guardrails

- Don't index confidential / unreleased sPHENIX notes on a public cloud vector DB. LanceDB local is fine; Pinecone is not unless you have institutional permission.
- Add a "no context found" branch to your prompt so the LLM doesn't confidently fabricate when retrieval fails.
- Keep source documents versioned. If a paper is updated, you need a re-embed strategy.

---

Week 09: tool use / function calling. The LLM stops being "text in, text out" and starts calling code. That's the doorway to agents.
