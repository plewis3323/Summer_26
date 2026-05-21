# Week 08 — Exercises

## E1 — Install and verify (easy, 15 min)

```bash
uv add sentence-transformers lancedb rank-bm25 ragas pypdf marker-pdf
```

Write a 10-line script that:
1. Loads `sentence-transformers/all-MiniLM-L6-v2`.
2. Embeds "hello world" and "goodbye world".
3. Computes cosine similarity.

**Accept when:** similarity is ~0.6–0.7 (same structure, different tokens).

## E2 — Embedding model comparison (easy, 30 min)

Embed the following 6 sentences with two models (`all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`):

```
S1 = "Elliptic flow measures the second Fourier coefficient of the azimuthal distribution."
S2 = "The v2 coefficient characterizes anisotropic particle emission in heavy-ion collisions."
S3 = "Pion production rates scale with the number of participants."
S4 = "Kaon yields depend on centrality."
S5 = "The sPHENIX experiment uses a hadronic calorimeter."
S6 = "Cats are warm animals that purr."
```

Compute pairwise cosine similarity matrices. Expected: S1-S2 very high; S3-S4 high; S1-S5 moderate; anything-S6 low.

Report: which model discriminates better on the physics-adjacent pairs?

## E3 — Chunk a physics paper (medium, 45 min)

Pick an arXiv heavy-ion paper PDF. Chunk it three ways:
1. Fixed 500-character chunks, 100-char overlap.
2. `RecursiveCharacterTextSplitter` at 500 tokens with 50 overlap.
3. Split on Markdown headings (`marker-pdf` produces nicely-headed markdown).

For each, print:
- Number of chunks
- Average chunk length (tokens)
- 3 random chunks with their boundaries

Write 100 words on which method preserves semantic coherence best for papers.

## E4 — Build a small index with LanceDB (medium, 60 min)

```python
import lancedb
from sentence_transformers import SentenceTransformer

db = lancedb.connect("./lancedb")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

chunks = [...]  # list of dicts with "text" and metadata

vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
for c, v in zip(chunks, vecs):
    c["vector"] = v.tolist()

tbl = db.create_table("heavy_ion_chunks", data=chunks)
```

Query: "What is elliptic flow?"
```python
q_vec = model.encode(["What is elliptic flow?"], normalize_embeddings=True)[0].tolist()
results = tbl.search(q_vec).limit(5).to_pandas()
```

**Accept when:** top-5 results include semantically relevant chunks (check qualitatively).

## E5 — Hybrid search with RRF (medium, 60 min)

Add BM25 via `rank-bm25`:

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search(query, k=10):
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    dense = tbl.search(q_vec).limit(20).to_pandas()
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(enumerate(bm25_scores), key=lambda x: -x[1])[:20]
    # RRF
    scores = {}
    for rank, row in enumerate(dense.itertuples()):
        scores[row.id] = scores.get(row.id, 0) + 1 / (60 + rank)
    for rank, (idx, _) in enumerate(bm25_top):
        cid = chunks[idx]["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return top
```

Test with 5 queries (one of which contains a rare acronym like "INTT"). Compare to dense-only results.

## E6 — Reranker (medium, 45 min)

Add a cross-encoder:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, candidates, k=5):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    return [c for _, c in ranked[:k]]
```

Pipeline: hybrid retrieve 20 → rerank → keep top 5. Compare side-by-side with no-rerank for the 5 queries in E5.

## E7 — Generation with grounded citations (medium, 60 min)

```python
from anthropic import Anthropic
client = Anthropic()

def answer(query, chunks):
    ctx = "\n\n".join(f"[{i+1}] ({c['arxiv_id']}, p{c['page']}): {c['text']}" for i, c in enumerate(chunks))
    prompt = f"""You are a heavy-ion physicist. Use ONLY the provided excerpts.
Cite each claim inline with [1], [2], etc. If the context is insufficient, say so.

Excerpts:
{ctx}

Question: {query}

Answer:"""
    r = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=600,
        messages=[{"role":"user","content":prompt}],
    )
    return r.content[0].text
```

Test with 5 questions. Verify citations appear.

## E8 — Build a 30-question gold eval set (medium, 90 min)

For your corpus of papers, write 30 questions where you know the answer or can identify which paper(s) contain it. Save as `gold.jsonl`:

```json
{"question": "What is the hadronic calorimeter energy resolution at sPHENIX?", "gold_arxiv_ids": ["2301.02961"], "expected_answer_keywords": ["HCal", "sampling", "%", "/√E"]}
```

**Accept when:** 30 entries; each with at least one `gold_arxiv_id`.

## E9 — Recall@k sweep (easy, 30 min)

For each gold question, run retrieval, check whether any of the gold arxiv_ids appears in top-k. Plot recall@1, recall@3, recall@5, recall@10 for:
- Dense-only
- Hybrid (RRF)
- Hybrid + rerank

Expectation: hybrid+rerank > hybrid > dense. Numeric gap of 10–20 percentage points is realistic.

## E10 — Faithfulness with ragas (medium, 60 min)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

data = Dataset.from_list([{
    "question": q["question"],
    "answer": your_generated_answer,
    "contexts": [c["text"] for c in retrieved_chunks],
    "ground_truth": q.get("answer", ""),
} for q in gold])

result = evaluate(data, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

Report the four scores for your system. Iterate: can you improve one without hurting the others?

## E11 — Contextual retrieval (hard, 90 min)

Implement Anthropic's contextual retrieval: before embedding each chunk, prepend a short LLM-generated summary situating the chunk within its document.

```python
def contextualize(chunk, full_doc, client):
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content":
            f"<document>{full_doc[:20000]}</document>\n\nHere is a chunk:\n\n<chunk>{chunk}</chunk>\n\nProvide a 1-2 sentence context situating this chunk."}],
    )
    return r.content[0].text.strip() + "\n\n" + chunk
```

Re-embed 100 random chunks with and without context. Rerun recall@5 on the gold set. Report delta.

Anthropic claims ~35% retrieval error reduction on their benchmarks. For your corpus you may see 10–30% — report honestly.

## E12 — Incremental indexing (medium, 45 min)

Write a pipeline that:
1. Hashes each chunk (content + metadata).
2. Stores hashes in a sidecar SQLite.
3. On re-run, only re-embeds chunks whose hash changed or is new.

Test: re-run on the same corpus — zero new embeddings. Add one paper — only new chunks embedded.

## E13 — CLI / tiny Streamlit app (easy, 45 min)

Wrap the end-to-end pipeline in a CLI:
```bash
uv run hep-rag ask "What is jet quenching at RHIC?"
```

Or a 50-line Streamlit app with a text box and a results panel. Show retrieved chunks with metadata and the generated answer with citations.

Use whichever you prefer — CLI for speed, Streamlit for show-and-tell.

---

## Solution hints

- E2 — bge-base is nearly always better than MiniLM-L6 on discrimination tasks. The cost is 2× the embedding time.
- E3 — chunk boundaries matter. A chunk that starts mid-equation is almost useless. Use marker-pdf.
- E4 — `normalize_embeddings=True` is essential for cosine similarity. Forgetting it is the #1 RAG bug.
- E5 — RRF's magic number k=60 is empirical. It rarely needs tuning.
- E6 — cross-encoders are ~100× slower than bi-encoders per pair. Rerank only top candidates, not everything.
- E7 — Claude citing well depends on the prompt. If it hallucinates citations, tighten the "use ONLY" instruction.
- E8 — building a gold set is tedious. It's the most important 90 minutes of the week.
- E9 — if dense beats hybrid, you have a bug. Dense should basically always lose to hybrid when the query has any rare tokens.
- E10 — ragas is slow (LLM calls). Run on a subset first to verify.
- E11 — contextual retrieval is powerful but doubles your indexing cost. Worth it only if retrieval is your bottleneck.
- E12 — incremental indexing is unglamorous but the first thing you'll miss when you don't have it.
- E13 — Streamlit is `pip install streamlit; streamlit run app.py`. Zero-effort UI.
