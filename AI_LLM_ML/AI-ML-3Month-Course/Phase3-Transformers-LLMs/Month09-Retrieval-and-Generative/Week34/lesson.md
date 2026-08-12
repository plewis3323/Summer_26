# Week 34 — RAG over a Physics Document Library

~3 hrs. Before starting you should be able to: build and query a persisted
embedding index (Week 33), run an open-weight instruct model with a chat template
(Week 29), explain precision and recall (Week 10), and describe LLM-as-judge and
its biases (Week 32).

## 1. What RAG is, and why generation is the easy half

Ask a language model "what is the design energy resolution of our electromagnetic
calorimeter?" and it will answer — fluently, confidently, and quite possibly
wrongly. The model's knowledge is whatever was in its training data: frozen at its
cutoff date, missing your internal documents entirely, and stored lossily in the
weights. Week 32 gave this failure a name and a measurement: hallucination.

**Retrieval-augmented generation (RAG)** attacks the problem by splitting the job:

1. **Retrieve** — find the handful of document chunks most relevant to the query
   (Week 33 machinery, plus what this week adds);
2. **Generate** — hand those chunks to an instruct model with the instruction
   "answer *from these passages* and cite them".

The model stops being an oracle and becomes a reading assistant. When the pipeline
fails, it is almost always the retrieval stage: if the right passage never reaches
the prompt, no amount of generation quality can recover it — and a fluent model
will paper over the gap with invention. So this week is mostly about retrieval and
about *measuring* retrieval, separately from generation, like calibrating each
stage of a measurement chain before quoting the combined systematic.

## 2. The library: documents you assemble

RAG needs a corpus worth asking questions about. This week you assemble a **physics
document library** of your own:

- **Full-text arXiv PDFs.** Pick 20–50 papers from the topics of your Week 27
  abstracts corpus (heavy-ion measurements, detector papers, one or two long
  reviews — search arXiv and download the PDFs). Your Week 32 corpus topics work
  too.
- **The Week 33 long document** you already chunked and indexed.
- **Variant — your personal library.** If you have your own collection (a Zotero
  export, your experiment's technical design report, internal notes), use it
  instead of or alongside the arXiv set. The pipeline is identical; only ingestion
  paths change. This variant makes every downstream answer immediately checkable
  against documents you know well — worth the setup if you have them.

PDFs must become text. Install with `uv add pypdf`:

```python
from pypdf import PdfReader

reader = PdfReader("data/pdfs/some_paper.pdf")
pages = []
for i, page in enumerate(reader.pages):
    pages.append({"page": i + 1, "text": page.extract_text()})
print(pages[0]["text"][:300])
```

Look at that output before building anything on it. PDF extraction of physics
papers mangles two-column layouts, drops or scrambles equations, and interleaves
figure captions with prose. Perfect cleanup is not the assignment; *knowing your
noise floor* is. Keep per-page records — the page number goes into chunk metadata,
because "see page 7 of arXiv:XXXX" is what a citation is.

Record provenance at ingest (Week 04 discipline): for every document, its source
(arXiv id or file name), download date, and page count, written to a manifest file.
Fresh clone + one command must rebuild the same index.

## 3. Lexical retrieval: BM25

Week 33 ended on the complementarity point: dense retrieval nails paraphrase and
misses exact tokens; keyword search is the mirror image. The standard keyword
scorer — decades old, still a strong baseline — is **BM25** ("best matching 25",
the 25th scoring variant in a 1990s retrieval system; the name stuck).

BM25 scores a document $d$ against a query by summing one contribution per query
term $t$:

$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$

Name every symbol: $f(t,d)$ is the term's count in the document; $|d|$ is the
document length in words; avgdl is the average document length in the corpus;
$k_1 \approx 1.5$ and $b \approx 0.75$ are tuning constants. IDF is the **inverse
document frequency**,

$$\text{IDF}(t) = \ln\!\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right),$$

with $N$ the number of documents and $n_t$ the number containing $t$. Three ideas
live in that formula:

- **Rare terms count more.** IDF is large when $n_t \ll N$. "Quenching" appearing
  in 5 of 100 documents is informative; "the" appearing in all of them is worthless
  ($n_t \approx N$ drives IDF toward $\ln 1{+}\ldots \approx 0$... more precisely
  toward $\ln(1 + 0.5/(N+0.5)) \approx 0$).
- **Term-frequency saturation.** As $f(t,d)$ grows, the fraction rises but flattens
  toward $k_1 + 1$: mentioning "quenching" 40 times is barely better evidence than
  10 times. $k_1$ sets where the flattening happens. (Compare: raw term-count
  scoring grows without bound and lets keyword-stuffed documents win.)
- **Length normalization.** The $b$ term inflates the denominator for
  longer-than-average documents: a long document matches every query a little by
  accident, and $b$ discounts that.

**Worked number.** $N = 100$ documents, "quenching" appears in $n_t = 5$ of them.
$\text{IDF} = \ln\!\left(\frac{100 - 5 + 0.5}{5 + 0.5} + 1\right) = \ln(18.36) = 2.91$.
A document of average length ($|d| = \text{avgdl}$, so the length factor is 1) with
$f = 3$, $k_1 = 1.5$, $b = 0.75$ scores
$2.91 \times \frac{3 \times 2.5}{3 + 1.5} = 2.91 \times 1.667 = 4.85$
from this term. Push $f$ to 30 and the fraction only reaches
$\frac{30 \times 2.5}{31.5} = 2.38$ — saturation at work (the cap is $k_1+1 = 2.5$).

In code, install with `uv add rank-bm25`:

```python
from rank_bm25 import BM25Okapi

tokenized = [c.lower().split() for c in chunks]      # chunks: list of strings
bm25 = BM25Okapi(tokenized)

query = "jet quenching at sqrt(s_NN) = 200 GeV"
scores = bm25.get_scores(query.lower().split())      # (n_chunks,)
```

Note what BM25 *cannot* do: "parton energy loss" shares zero terms with "jet
quenching", so BM25 scores it zero. Exact strings, no meaning — precisely the hole
dense retrieval fills.

## 4. Hybrid search: fusing dense and lexical

Run both retrievers and merge. The merging problem: BM25 scores and cosine
similarities live on incomparable scales, so you cannot just add them.
**Reciprocal rank fusion (RRF)** sidesteps scales by using only *ranks*. Each
retriever returns a ranked list; a document's fused score is

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)},$$

with $k = 60$ by convention (it damps the difference between rank 1 and rank 2 so
one retriever's top pick cannot dominate alone) and rank counted from 1; a document
missing from a retriever's list contributes nothing from that retriever.

**Worked example.** Dense returns $[A, B, C]$, BM25 returns $[C, D, A]$, $k = 60$:

- $A$: dense rank 1, BM25 rank 3 → $\frac{1}{61} + \frac{1}{63} = 0.0326$
- $C$: dense rank 3, BM25 rank 1 → $\frac{1}{63} + \frac{1}{61} = 0.0326$
- $B$: dense rank 2 only → $\frac{1}{62} = 0.0161$
- $D$: BM25 rank 2 only → $\frac{1}{62} = 0.0161$

Fused order: $A, C$ (tied), then $B, D$ (tied). Documents that *both* retrievers
liked float to the top; each retriever's solo favorites still survive. That is the
whole trick, and it is remarkably hard to beat. In code:

```python
def rrf(rankings, k):
    scores = {}
    for ranking in rankings:                 # each: list of doc ids, best first
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

## 5. Measuring retrieval: recall@k and MRR

Before improving retrieval you must be able to measure it. Both standard metrics
assume an **eval set**: queries paired with their **gold passages** — the chunks a
correct answer must come from, recorded by a human (you) who checked.

**Recall@k.** Week 10 defined recall as the fraction of true positives found. In
retrieval, the "positives" for a query are its gold passages, and the "found" ones
are those appearing in the top $k$ results:

$$\text{recall@}k(q) = \frac{|\text{gold}(q) \cap \text{top-}k(q)|}{|\text{gold}(q)|},$$

averaged over queries. With one gold passage per query it is simply: did the right
chunk make the top $k$ — 1 or 0. Recall@k is *the* number that bounds a RAG system:
if the passage isn't retrieved, the answer cannot be grounded in it.

**MRR (mean reciprocal rank)** adds *position* sensitivity. For each query, find
the rank (from 1) of the first gold passage in the results; take its reciprocal,
$1/\text{rank}$; use 0 if no gold appears in the list; average over queries:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}.$$

Reciprocal rank falls fast: rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores
0.2. That mirrors reality — chunks early in the prompt get more of the model's
attention, and shorter prompts are cheaper — so a system that finds gold at rank 1
genuinely beats one that finds it at rank 5, even though both have the same
recall@5.

**Worked example.** Four queries; retrieval returns 5 results each; one gold
passage per query. The gold lands at:

| query | gold rank | in top 5? | reciprocal rank |
|-------|-----------|-----------|-----------------|
| Q1 | 1 | yes | 1.000 |
| Q2 | 3 | yes | 0.333 |
| Q3 | not in top 5 | no | 0.000 |
| Q4 | 2 | yes | 0.500 |

$$\text{recall@5} = \frac{3}{4} = 0.75, \qquad
\text{MRR} = \frac{1 + 1/3 + 0 + 1/2}{4} = \frac{1.833}{4} = 0.458.$$

Check the edge cases against your intuition: a perfect system has recall@5 = MRR =
1.0; a system that always buries gold at exactly rank 5 keeps recall@5 = 1.0 but
MRR = 0.2. Report both — recall@k tells you *whether* the system finds evidence,
MRR tells you *where it puts it*.

**One rule with teeth: write the eval set before tuning.** If you pick queries
after seeing what the system does well, you are measuring your own selection bias —
Week 32's contamination lesson, self-inflicted. This week you write two sets: a
~25-question *dev set* you tune against freely, and a **≥30-question held-out set**
that you freeze (commit, hash, do not look at again) for Capstone 3. Same
discipline as a blinded analysis: the answer box stays taped shut until the method
is frozen.

## 6. Reranking with a cross-encoder

Week 33's embedding model is a **bi-encoder**: it encodes query and passage
*separately*, then compares two vectors. Fast — passage vectors are precomputed —
but the query never "sees" the passage during encoding; all interaction is
squeezed through one dot product.

A **cross-encoder** concatenates query and passage into a single input and runs the
transformer over the pair, emitting one relevance score. Now attention (Week 25)
operates *across* the pair — every query token can attend to every passage token —
which is strictly more expressive than a dot product of independent summaries. The
price: no precomputation. Scoring is one forward pass per (query, passage) pair,
so scoring a whole corpus per query is out of the question.

The standard architecture uses both, each where it is affordable:

1. **Retrieve** top ~20 candidates cheaply (hybrid search over the whole corpus);
2. **Rerank** those 20 with the cross-encoder; keep the top ~5 for the prompt.

```python
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, c) for c in candidates]        # ~20 candidates from hybrid
ce_scores = ce.predict(pairs)                   # one relevance score per pair
order = ce_scores.argsort()[::-1][:5]
```

Reranking is a measurable trade, not a free lunch: exercise E5 reports both the
recall@5 change *and* the added latency per query. On a clean corpus with an easy
eval set the delta can be small; on paraphrase-heavy questions it is often the
single biggest upgrade in the pipeline.

## 7. Answer synthesis with citations

Generation, finally. Take the top-5 reranked chunks and build a prompt for the
instruct model you have used since Week 29:

```python
context = ""
for i, c in enumerate(top_chunks):
    context = context + f"[{i+1}] ({c['source']}, p.{c['page']}) {c['text']}\n\n"

prompt = (
    "Answer the question using ONLY the numbered passages below. "
    "Cite passages like [2] after each claim. "
    "If the passages do not contain the answer, say so instead of guessing.\n\n"
    + context + "Question: " + question + "\nAnswer:"
)
```

Three deliberate choices in that prompt. Numbered passages with source metadata
make citations checkable — a cited number maps back to a chunk, a page, a document;
this is why §2 insisted on metadata at ingest. The "ONLY" instruction defines the
task as reading, not recalling. And the licensed "I don't know" gives the model an
alternative to invention when retrieval failed — without it, the model *will*
answer from its weights, and you will not be able to tell.

Be clear about what prompting buys: these instructions lower the hallucination
rate; nothing makes it zero. The model can still cite [2] for a claim that [2]
does not contain. Which is why generation gets its own evaluation.

## 8. Evaluating generation: faithfulness

An answer is **grounded** (faithful) if every claim in it is supported by the
retrieved passages — not by the world, by the *passages*. Groundedness is a
property you can check mechanically per answer, unlike "correctness", which needs
world knowledge. Three-way label per answer, judged against its own retrieved
context: **grounded** / **partially grounded** (some claims supported, some not) /
**hallucinated** (key claims unsupported).

Do the judging the Week 32 way: an LLM judge scores all answers cheaply, and you
hand-label a subsample to validate the judge, remembering its biases (verbosity,
position, self-preference). Report the grounded rate next to the retrieval metrics
— *separately*. The whole point of the two-stage evaluation is diagnosis:

| symptom | retrieval metrics | groundedness | stage to fix |
|---|---|---|---|
| wrong-answer, right passage retrieved | good | poor | generation (prompt, model) |
| wrong-answer, passage never retrieved | poor | — | retrieval (chunking, hybrid, rerank) |
| right answer, wrong citation | good | poor | prompt format / citation parsing |
| "I don't know" on answerable question | good | grounded | generation too conservative |

A single end-to-end "accuracy" number cannot distinguish these rows. Two dials,
measured separately, can.

## 9. Worked example: metrics by hand on a toy pipeline

Small enough to check every number by eye. Three documents, three queries, gold
passages known.

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

chunks = [
    "Jet quenching is observed in central Au+Au collisions at 200 GeV.",   # 0
    "The EMCal energy resolution is 16%/sqrt(E) in test beam data.",       # 1
    "Elliptic flow of charged hadrons is measured in Pb+Pb at 5020 GeV.",  # 2
]
queries = ["How do fast partons lose energy in heavy-ion collisions?",
           "What is the EMCal energy resolution?",
           "What is measured at sqrt(s_NN) = 5020 GeV?"]
gold = [0, 1, 2]

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(chunks, normalize_embeddings=True)
bm25 = BM25Okapi([c.lower().split() for c in chunks])

def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

recip_ranks = []
hits_at_1 = 0
for q, g in zip(queries, gold):
    qv = model.encode([q], normalize_embeddings=True)[0]
    dense_rank = list(np.argsort(-(emb @ qv)))
    bm25_rank = list(np.argsort(-bm25.get_scores(q.lower().split())))
    fused = rrf([dense_rank, bm25_rank], 60)
    rank_of_gold = fused.index(g) + 1
    recip_ranks.append(1.0 / rank_of_gold)
    if rank_of_gold == 1:
        hits_at_1 = hits_at_1 + 1
    print(q[:40], "| dense", dense_rank, "| bm25", bm25_rank,
          "| fused", fused, "| gold rank", rank_of_gold)

print("recall@1 =", hits_at_1 / 3, " MRR =", round(sum(recip_ranks) / 3, 3))
```

Run it and read the per-query lines against the lesson: query 1 is pure paraphrase
("lose energy" / "quenching" share no terms), so expect dense to rank the gold
first while BM25 flounders; query 2 has the exact rare token "EMCal", BM25's home
turf; query 3's "5020" is the Week 33 numeral trap — BM25 matches the digits
exactly, dense may not separate 5020 from 200. Fusion should put the gold chunk
first for all three, giving recall@1 = 1.0 and MRR = 1.0 — each retriever covering
the other's blind side, which is the thesis of this whole week in nine lines of
output. (Model nondeterminism aside: if a dense ranking differs slightly for you,
the *fused* gold ranks are the numbers to check.)

The exercises scale this exact skeleton to your real library, ~25 dev questions,
five retrieval configurations, and judged generation on top.

## Check yourself

1. In the BM25 formula, what happens to a term's contribution as $f(t,d) \to
   \infty$, and which constant controls it? Why is that behavior desirable?
2. Why does BM25 score "parton energy loss" as irrelevant to "jet quenching", and
   which component of the hybrid system covers that case?
3. Five queries have gold ranks 1, 2, 4, >5 (miss), 1 in a top-5 list. Compute
   recall@5 and MRR.
4. A system change moves every gold passage from rank 4 to rank 1. What happens to
   recall@5 and to MRR? What does that tell you about reporting only recall@5?
5. Why must the Capstone 3 held-out Q&A set be written and frozen *this* week,
   before you tune anything?
6. In attention terms (Week 25), why is a cross-encoder strictly more expressive
   than a bi-encoder, and what does that expressiveness cost at query time?
7. An answer states the correct energy resolution but cites a passage that does not
   contain it. Is it grounded? Which stage needs fixing?
8. Why is groundedness judged against the retrieved passages rather than against
   the true answer?

## Answers

1. The contribution saturates at $\text{IDF}(t) \cdot (k_1 + 1)$; $k_1$ sets how
   fast. Desirable because the 30th occurrence of a term is barely more evidence of
   relevance than the 10th, and unbounded counts reward keyword stuffing.
2. Zero shared terms means every $f(t,d) = 0$, so the sum is 0 — BM25 sees strings,
   not meaning. Dense retrieval (the Week 33 embeddings) handles the paraphrase.
3. recall@5 = 4/5 = 0.8. MRR = $(1 + 1/2 + 1/4 + 0 + 1)/5 = 2.75/5 = 0.55$.
4. recall@5 is unchanged (gold was already in the top 5); MRR jumps from 0.25 to
   1.0 for those queries. recall@5 alone hides large, real improvements in where
   evidence lands in the prompt — report both.
5. Once you tune against a set, it measures your tuning, not the system (selection
   bias / contamination, Week 32). Freezing before integration guarantees the
   capstone benchmark is an out-of-sample measurement.
6. The cross-encoder runs attention across the concatenated (query, passage) pair,
   so every query token can attend to every passage token; a bi-encoder compresses
   each side to one vector first and all interaction is one dot product. Cost: no
   precomputation — one full forward pass per candidate pair per query, hence
   rerank-only-top-20.
7. Not grounded — groundedness requires the *cited* passages to support the claim;
   the model answered from its weights. Fix generation: prompt format, citation
   instructions, or a stronger instruct model.
8. Because groundedness isolates the generation stage: it asks "did the model read
   faithfully?", checkable from the context alone. Comparing to the true answer
   mixes in retrieval quality and world knowledge, un-attributing the failure.

## New terms

- **RAG (retrieval-augmented generation)** — pipeline that retrieves relevant
  chunks and instructs a model to answer from them with citations.
- **BM25** — lexical scoring function combining IDF, saturated term frequency, and
  document-length normalization.
- **IDF (inverse document frequency)** — $\ln\!\big(\frac{N-n_t+0.5}{n_t+0.5}+1\big)$;
  rare terms weigh more.
- **term-frequency saturation** — bounded growth of a term's score with its count,
  controlled by $k_1$.
- **hybrid search** — running dense and lexical retrieval together and fusing
  results.
- **reciprocal rank fusion (RRF)** — rank-based fusion:
  $\sum_r 1/(k + \text{rank}_r)$; ignores incomparable score scales.
- **eval set / gold passage** — human-written queries paired with the chunks a
  correct answer must come from.
- **recall@k** — fraction of gold passages appearing in the top $k$ results,
  averaged over queries.
- **MRR (mean reciprocal rank)** — average of $1/\text{rank of first gold}$; 0 on a
  miss.
- **dev set vs held-out set** — the set you tune on vs the frozen set you touch
  only at final evaluation.
- **bi-encoder / cross-encoder** — separate encodings compared by dot product vs
  joint encoding of the pair; fast-and-shallow vs slow-and-expressive.
- **reranking** — rescoring a small candidate set with an expensive model after
  cheap first-stage retrieval.
- **grounded / faithful** — every claim in the answer is supported by the retrieved
  passages themselves.

## Going deeper

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
  — the original RAG paper; skim to see how the modern pipeline differs from the
  jointly-trained original.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* —
  where §3's formula comes from, with the probabilistic justification.
- `rank_bm25` and `chromadb` docs — the two libraries the exercises stand on.
- A cross-encoder model card from `sentence-transformers` (e.g. the ms-marco
  series) — read the training data and intended-use notes; rerankers inherit the
  biases of their training queries.
