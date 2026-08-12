# Week 08 — Reading

## Primary

1. **Lewis et al. — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020). arXiv:2005.11401.**
   - The original RAG paper. §1–3 lay the framing; §4 is the architecture. Skim §5 experiments.

2. **Karpukhin et al. — *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020). arXiv:2004.04906.**
   - DPR. The bi-encoder → vector-DB → retrieve pattern in one paper. §2–3 core.

3. **Reimers & Gurevych — *Sentence-BERT* (EMNLP 2019). arXiv:1908.10084.**
   - The technique behind sentence-transformers. §3 is the architecture.

4. **Kusupati et al. — *Matryoshka Representation Learning* (NeurIPS 2022). arXiv:2205.13147.**
   - Recent, influential. Modern embedding models (OpenAI's v3, Nomic, BGE-M3) use Matryoshka — you can truncate dimensions without retraining.

5. **Anthropic — *Contextual Retrieval* (blog, 2024).** https://www.anthropic.com/news/contextual-retrieval
   - A practical technique: pre-prepend a short LLM-generated summary to each chunk before embedding. Shown to materially improve retrieval.

6. **OpenAI — *Embeddings guide*.** https://platform.openai.com/docs/guides/embeddings
   - Practitioner-grade. Covers cost, dimensions, similarity metrics.

## Secondary

7. **Gao et al. — *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023). arXiv:2312.10997.**
   - Survey. §3 (taxonomy), §5 (evaluation), §6 (frontiers). Skim; the taxonomy alone is worth it.

8. **Bajaj et al. — *MS MARCO* (2016, ongoing). arXiv:1611.09268.**
   - The benchmark dataset most retrievers are trained on. Useful to know.

9. **Thakur et al. — *BEIR* (NeurIPS 2021). arXiv:2104.08663.**
   - Retrieval benchmark across 18 domains. Useful to know state of the art.

10. **Cormack, Clarke, Büttcher — *Reciprocal Rank Fusion* (SIGIR 2009).**
    - The original RRF paper. Short. Will demystify the k=60 magic number.

11. **Izacard & Grave — *Fusion-in-Decoder* (EACL 2021). arXiv:2007.01282.**
    - An alternative to prompt-stuffing. Passes retrieved docs separately to decoder. Academic more than practical.

## Blog posts

- **Yan Eugene — *Patterns for Building LLM-based Systems & Products*.** https://eugeneyan.com/writing/llm-patterns/. The "Retrieval" section is comprehensive.
- **Pinecone — *Embeddings Tutorial*.** https://www.pinecone.io/learn/ — marketing but well-written.
- **LanceDB — *Hybrid search* blog.** https://lancedb.com/blog/hybrid-search/.
- **Jerry Liu (LlamaIndex founder) on RAG tradeoffs.** Various blog posts on llamaindex.ai.

## Tutorials

- **LangChain RAG tutorial.** https://python.langchain.com/docs/tutorials/rag/. The standard walkthrough. Useful even if you don't use LangChain going forward.
- **LlamaIndex RAG tutorial.** https://docs.llamaindex.ai/en/stable/understanding/rag/. Similar.
- **Haystack RAG pipeline.** https://haystack.deepset.ai/tutorials. Another option.
- **Claude cookbook — RAG with Claude.** https://github.com/anthropics/anthropic-cookbook/tree/main/skills/retrieval_augmented_generation

## Videos

- **Cohere — *The State of Retrieval* (various talks, 2024).** Short, practitioner-focused.
- **Pinecone — *Building a RAG application.* ~40 min.** 
- **Anthropic — *Contextual Retrieval* announcement video.** ~10 min.

## Textbooks

Nothing in the canonical ML textbooks on RAG yet — too new. Manning / Raghavan / Schütze *Introduction to Information Retrieval* is the classic on IR; Chapter 6 (scoring and weighting) gives you BM25 foundations. Free online.

## Codecademy

"Build a RAG app" — if present in your Pro catalog, do it. If not, skip.

## HEP-specific references

- **INSPIRE-HEP API** — not RAG per se, but worth knowing. https://inspirehep.net/api. Structured metadata for every paper, DOI, BibTeX. Adjacent to what you'll build — metadata alone can answer many queries without hitting the full-text.
- **arXiv full-text API and bulk access** — https://info.arxiv.org/help/bulk_data_s3.html. If you want the full corpus locally.
- **Marker-pdf on a physics paper.** Try it. It handles equations better than naive pypdf.

## Notes to take

- A decision table: when to reach for dense vs sparse vs hybrid retrieval, with an example query for each.
- A one-page spec of your chunker: chunk size target, overlap, handling of equations, handling of tables, metadata fields.
- A short write-up: "Five queries about heavy-ion physics I'd expect a decent RAG system to answer and five I'd expect it to struggle with."
- Performance notes: embedding throughput (embeddings/s), retrieval latency, generation latency. End-to-end time budget for "ask → answer".

## Reading plan

| Day | What |
| --- | --- |
| 1 | Lewis RAG paper. LangChain RAG tutorial. |
| 2 | DPR paper. Sentence-BERT paper §3. |
| 3 | Contextual Retrieval blog. Matryoshka paper. |
| 4–5 | Reference as you build. Reranker docs. |
| 6 | ragas docs. Evaluation-focused reads. |
| 7 | Survey (Gao et al.) skim. |

## Order-of-operations nuance

Read the LangChain tutorial first (Day 1) to get vocabulary. Then read the papers. The papers hit differently once you've seen the pipeline.
