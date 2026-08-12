# Week 08 — Mini-Project: Heavy-Ion Paper RAG ("HEP-RAG")

## Problem

Build a retrieval-augmented generation system over a corpus of ~300–500 heavy-ion physics papers. Ingest PDFs, chunk them intelligently, index with hybrid dense+sparse search, rerank, generate grounded answers with inline citations. Evaluate on a 30-question gold set. Ship a CLI (and optionally a Streamlit UI).

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/hep_rag/
│   ├── __init__.py
│   ├── ingest.py              # PDF → text via marker-pdf
│   ├── chunk.py               # structural chunking + metadata
│   ├── embed.py               # sentence-transformers, batched
│   ├── index.py               # LanceDB create / update / query
│   ├── bm25.py                # sparse index
│   ├── retrieve.py            # hybrid RRF + rerank
│   ├── generate.py            # Claude call with grounded prompt
│   ├── eval.py                # ragas + recall@k harness
│   └── cli.py                 # typer CLI
├── tests/
│   ├── test_chunk.py
│   ├── test_retrieve.py
│   └── test_cli.py
├── data/
│   ├── papers/                # PDFs
│   ├── parsed/                # marker output (markdown + metadata)
│   ├── index/                 # LanceDB store
│   └── gold.jsonl             # 30 question eval set
├── configs/
│   └── hep_rag.yaml
└── results/
    ├── recall_at_k.png
    ├── ragas_scores.json
    └── ablation_table.md
```

## Acceptance criteria

1. `uv run pytest -q` passes.
2. Ingested ≥ 300 unique arXiv papers from hep-ex / nucl-ex / nucl-th.
3. LanceDB index with both dense vectors and BM25 sidecar.
4. Retrieval pipeline with three stages: (a) hybrid dense+sparse, (b) reranker, (c) reranked top-5 assembled into prompt.
5. 30-question gold set constructed. Recall@5 ≥ 70% with hybrid+rerank.
6. Generation produces answers with inline `[1]`-style citations that map to the retrieved chunks.
7. ragas evaluation: faithfulness ≥ 0.85, answer_relevancy ≥ 0.85 (LLM-judge numbers are noisy; ballpark).
8. At least 2 ablations: (a) dense-only vs hybrid, (b) no-rerank vs rerank.
9. CLI: `uv run hep-rag ask "..."` returns answer + citations; `uv run hep-rag ingest path/to/pdf/` adds new papers incrementally.
10. README with architecture diagram, sample queries, metrics, limitations.

## Suggested plan (7 days)

- **Day 1.** Corpus download. `marker-pdf` to parse 50 papers end-to-end. Inspect output quality.
- **Day 2.** Chunker. Embed. First LanceDB index.
- **Day 3.** BM25. Hybrid search. Reranker.
- **Day 4.** Generation. Grounded prompt. Citations.
- **Day 5.** 30-question gold set. Recall@k sweep.
- **Day 6.** ragas eval. Iterate.
- **Day 7.** CLI polish, ablations, write-up.

## Corpus acquisition

Option A: Use your Week 06 arXiv ID list + bulk-download PDFs via arxiv API:
```python
import arxiv
from pathlib import Path
client = arxiv.Client(page_size=100, delay_seconds=3.0)
for r in client.results(arxiv.Search(id_list=arxiv_ids)):
    r.download_pdf(dirpath="data/papers/", filename=f"{r.entry_id.split('/')[-1]}.pdf")
```

Option B: Curated list of papers you've actually read or plan to — around 50 papers. Higher quality queries, smaller corpus.

Either works. Option B is more tightly scoped; Option A is more impressive.

## PDF parsing

```python
# src/hep_rag/ingest.py
import subprocess
from pathlib import Path

def parse(pdf: Path, out_dir: Path):
    subprocess.run(["marker_single", str(pdf), str(out_dir), "--workers", "1"], check=True)
```

`marker-pdf` outputs clean markdown with preserved equations (as LaTeX) and tables. ~10 seconds per paper on GPU, ~30s on CPU. Run it in a batch over your corpus.

Validate by opening a few and confirming equations survive as `$...$` blocks.

## Chunking

```python
# src/hep_rag/chunk.py
import re, hashlib

def chunks_from_md(md: str, arxiv_id: str, max_tokens: int = 500, overlap: int = 50):
    # Split on headings first
    sections = re.split(r"\n#+\s+", md)
    out = []
    for sec in sections:
        if not sec.strip(): continue
        # Further split if too long — use paragraphs as secondary split
        tokens = sec.split()
        if len(tokens) <= max_tokens:
            out.append({"arxiv_id": arxiv_id, "text": sec,
                        "hash": hashlib.sha256(sec.encode()).hexdigest()[:16]})
        else:
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk = " ".join(tokens[i:i + max_tokens])
                out.append({"arxiv_id": arxiv_id, "text": chunk,
                            "hash": hashlib.sha256(chunk.encode()).hexdigest()[:16]})
    return out
```

Enrich with metadata during ingest:
```python
chunk["arxiv_id"] = arxiv_id
chunk["title"] = metadata["title"]
chunk["authors"] = metadata["authors"]
chunk["year"] = metadata["year"]
chunk["section"] = section_name
```

## Indexing

```python
# src/hep_rag/index.py
import lancedb, pyarrow as pa
from sentence_transformers import SentenceTransformer

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("arxiv_id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("section", pa.string()),
    pa.field("text", pa.string()),
    pa.field("hash", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 768)),
])

def build_index(chunks, db_path="data/index"):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    for c, v in zip(chunks, vecs):
        c["vector"] = v.tolist()
        c["id"] = c["hash"]
    db = lancedb.connect(db_path)
    tbl = db.create_table("chunks", data=chunks, schema=SCHEMA, mode="overwrite")
    tbl.create_fts_index("text")   # enables BM25-style search in LanceDB directly
    return tbl
```

LanceDB's FTS index + vector index + hybrid search method gives you RRF out of the box:
```python
result = tbl.search(query).query_type("hybrid").limit(20).to_pandas()
```

## Retrieval + rerank

```python
# src/hep_rag/retrieve.py
from sentence_transformers import CrossEncoder

def retrieve(tbl, model, query, k_retrieve=20, k_final=5):
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    hybrid = tbl.search((q_vec.tolist(), query)).query_type("hybrid").limit(k_retrieve).to_pandas()
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    pairs = list(zip([query]*len(hybrid), hybrid["text"].tolist()))
    scores = reranker.predict(pairs)
    hybrid["rerank_score"] = scores
    return hybrid.sort_values("rerank_score", ascending=False).head(k_final)
```

## Generation

```python
# src/hep_rag/generate.py
from anthropic import Anthropic

SYSTEM = """You are a senior physicist specializing in heavy-ion collisions.
Answer the user's question using ONLY the provided excerpts.
Cite every claim inline as [1], [2], etc.
If the excerpts do not contain the answer, say so explicitly."""

def answer(client, query, chunks):
    ctx = "\n\n".join(f"[{i+1}] ({c['arxiv_id']}, §{c.get('section','')}): {c['text']}"
                      for i, c in enumerate(chunks))
    msg = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=800, system=SYSTEM,
        messages=[{"role":"user","content":f"Excerpts:\n\n{ctx}\n\nQuestion: {query}"}],
    )
    return msg.content[0].text
```

## CLI

```python
# src/hep_rag/cli.py
import typer
app = typer.Typer()

@app.command()
def ingest(path: str):
    # parse, chunk, embed, upsert
    ...

@app.command()
def ask(query: str, k: int = 5):
    tbl = lancedb.connect("data/index").open_table("chunks")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    chunks = retrieve(tbl, model, query, k_final=k)
    client = Anthropic()
    ans = answer(client, query, chunks.to_dict("records"))
    print(ans)
    print("\nSources:")
    for i, c in enumerate(chunks.to_dict("records")):
        print(f"[{i+1}] {c['arxiv_id']} — {c.get('title','')}")

if __name__ == "__main__":
    app()
```

## Evaluation plan

Build `gold.jsonl` with 30 entries. For each, record:
- `question`
- `gold_arxiv_ids`: list of papers you expect to be retrieved
- Optional `expected_answer_keywords` to spot-check generations

Run:
```python
def recall_at_k(gold, tbl, model, k):
    hits = 0
    for q in gold:
        retrieved = retrieve(tbl, model, q["question"], k_final=k)
        retrieved_ids = set(retrieved["arxiv_id"].tolist())
        if set(q["gold_arxiv_ids"]) & retrieved_ids:
            hits += 1
    return hits / len(gold)
```

Plot for k ∈ {1, 3, 5, 10, 20} for three configurations: dense-only, hybrid, hybrid+rerank. Include in `results/recall_at_k.png`.

Then run ragas on the full gold set. Store scores in `results/ragas_scores.json`.

## Write-up (`project/README.md`)

```markdown
# HEP-RAG — Retrieval-augmented QA over heavy-ion physics papers

## Architecture
PDFs → marker-pdf (markdown) → structural chunker → bge-base-en embeddings
     → LanceDB (vector + FTS) → hybrid RRF → bge-reranker-base → Claude Sonnet

## Corpus
372 arXiv papers (hep-ex, nucl-ex, nucl-th, 2015–2025). 14,800 chunks after chunking.

## Metrics
| Config | Recall@5 | MRR | Faithfulness | Answer Rel. |
| --- | --- | --- | --- | --- |
| Dense-only | 0.58 | 0.41 | 0.81 | 0.86 |
| Hybrid (RRF) | 0.71 | 0.55 | 0.83 | 0.87 |
| Hybrid + rerank | 0.81 | 0.68 | 0.88 | 0.89 |

## Example query
> "What is the hadronic calorimeter energy resolution at sPHENIX?"
Answer: The sPHENIX HCal energy resolution is reported as σ_E/E ≈ 75%/√E ⊕ 14% [1], consistent with a sampling calorimeter of the design described in [2].

Sources:
[1] 2301.02961 — sPHENIX Technical Design Report
[2] 2206.04769 — ...

## Limitations
- Retrieval struggles with queries phrased as questions rather than statements (e.g., "why is the QGP described as a perfect fluid?"). Hybrid search picks up mostly definitional chunks.
- Some figure/table content is lost in PDF extraction.
- Citations sometimes map to chunks that are semantically adjacent but not quite the source of the claim.

## Reproducibility
Corpus via `hep-rag ingest`. uv.lock committed. LanceDB store portable.
```

## Common failure modes

- `marker-pdf` gives inconsistent output on old scans / non-native PDFs → fall back to `pypdf` for those.
- Hybrid worse than dense → FTS index has not been built or BM25 scoring is broken; check LanceDB version.
- Reranker flips good results to bad → cross-encoder trained on MS MARCO has a bias toward lexical similarity; sometimes mxbai-rerank-base-v1 works better on physics-ese.
- Claude refuses to cite → your system prompt is unclear about the `[n]` format. Give it an example.
- ragas slow → it calls an LLM per metric per example. Budget 10–15 minutes for 30 questions.
- Out-of-memory when embedding → reduce batch size.

## Extensions (optional)

- Add contextual retrieval (E11). Measure recall gain.
- Add a "follow-up" loop: if the first answer says "insufficient context", re-query with the LLM's own reformulation.
- Swap your Week 07 fine-tuned summarizer in as the generator. Compare quality with Claude-generated answers.
- Build a web app with Streamlit, show side-by-side retrieved chunks next to the answer.
- Integrate with INSPIRE-HEP: if the answer mentions a paper, link it by DOI.

## Sign-off

Commit, tick Week 08. Month 2 done. Next week: tool use — we teach the LLM to call Python, SQL, and external APIs, and we stop relying purely on text.
