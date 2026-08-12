# Week 33 — Embeddings & Semantic Search

~3 hrs. Before starting you should be able to: explain what a transformer encoder
does to a sequence of tokens (Week 25–26), load and run a pretrained HuggingFace
model (Week 29), compute dot products and vector norms in NumPy (Weeks 03, 06),
and write softmax + cross-entropy from memory (Weeks 10, 13).

## 1. The problem: search that understands paraphrase

You have the 5,000 heavy-ion abstracts you scraped in Week 27. (Heavy-ion physics
collides large nuclei — gold, lead — at near light speed to create a droplet of
quark–gluon plasma, the state of matter from the early universe. Its papers are full
of jargon with many names for the same idea.) You want every abstract about *jet
quenching* — the phenomenon where a high-energy spray of particles loses energy
crossing the plasma. Some abstracts say "jet quenching". Others say "parton energy
loss", "suppression of high-$p_T$ hadrons", or "$R_{AA} < 1$". A keyword search
finds only the first group.

Keyword search matches *strings*. What you want is to match *meaning*. The tool for
that is an **embedding**: a learned function that maps a piece of text to a vector
(a point in $\mathbb{R}^d$, typically $d$ = 384–1024) such that texts with similar
meaning land near each other. Search then becomes geometry: embed the query, embed
every document, return the documents whose vectors are closest.

This week you build that machinery and — just as important — measure where it fails.
Next week (Week 34) it becomes the retrieval half of a RAG system; in Week 36 it is
the core of Capstone 3 track (a).

## 2. From the dot product to cosine similarity

"Closest" needs a definition. Start from what you know (Week 06): the dot product of
$\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ is

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i b_i,
\qquad \|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}}.$$

Here $a_i$ is the $i$-th component of $\mathbf{a}$ and $\|\mathbf{a}\|$ is its length
(Euclidean norm). The dot product hides an angle. To see it, compute the squared
distance between the two vectors two ways.

**Way 1 — algebra.** Expand using distributivity of the dot product:

$$\|\mathbf{a} - \mathbf{b}\|^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b})
= \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\,\mathbf{a} \cdot \mathbf{b}.$$

**Way 2 — geometry.** $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{a} - \mathbf{b}$ form a
triangle with the angle $\theta$ between $\mathbf{a}$ and $\mathbf{b}$ opposite the
side $\mathbf{a}-\mathbf{b}$. The law of cosines (high-school trigonometry: the
Pythagorean theorem with a correction term for non-right angles) says

$$\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2
- 2\,\|\mathbf{a}\|\,\|\mathbf{b}\| \cos\theta.$$

The two right-hand sides describe the same triangle, so equate them and cancel:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\| \cos\theta
\qquad\Longrightarrow\qquad
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}.$$

That last quantity is the **cosine similarity**: +1 for parallel vectors (same
direction, "same meaning"), 0 for orthogonal ones (unrelated), −1 for opposite. It
ignores vector *length* and keeps only *direction* — useful because embedding norms
often track incidental properties like text length rather than meaning.

**Normalization collapses the metric choices.** If you rescale each vector to unit
length, $\hat{\mathbf{u}} = \mathbf{u}/\|\mathbf{u}\|$, three things coincide:

1. cosine similarity equals the plain dot product:
   $\cos\theta = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$;
2. Euclidean distance becomes a monotone function of cosine — set
   $\|\hat{\mathbf{u}}\| = \|\hat{\mathbf{v}}\| = 1$ in Way 1:
   $$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|^2 = 2 - 2\cos\theta = 2(1 - \cos\theta);$$
3. therefore ranking neighbors by cosine, by dot product, or by (negative) Euclidean
   distance gives the *same order*.

So the practical rule: **normalize your embeddings, then use the dot product** — it
is the cheapest of the three, and the choice of metric stops mattering. On
*unnormalized* vectors the three metrics can rank differently, and you will verify
that numerically in exercise E2.

## 3. Where sentence embeddings come from

A transformer encoder (Week 25–26) turns a token sequence into one output vector
*per token*. A search index wants one vector *per text*. The standard bridge is
**pooling**: combine the token vectors into a single vector. **Mean pooling** —
averaging the token vectors — is the common choice. Note what averaging does: like
the sum in a GNN readout (Week 21), it is permutation-invariant, so pooling alone
would discard word order; the order information survives only because the
transformer's position embeddings already mixed it into each token vector.

Here is the catch, discovered by Reimers & Gurevych (the *Sentence-BERT* paper):
if you mean-pool a model that was pretrained only on masked-word prediction, the
resulting sentence vectors are poor — often worse for similarity search than
counting words. Nothing in the pretraining objective ever asked "do these two
*sentences* mean the same thing?", so the geometry of pooled vectors carries no
such promise. You must *train for* the property you want. That training is
contrastive.

## 4. Contrastive training and the InfoNCE loss

**Setup.** You have pairs of texts that should mean the same thing: (question,
answer passage), (paper title, its abstract), (sentence, its paraphrase). Call each
pair an **anchor** $a_i$ and its **positive** $p_i$. An encoder $f_\phi$ (the
transformer + pooling, with parameters $\phi$) maps text to a unit vector. Write
the similarity between anchor $i$ and positive $j$ as

$$s_{ij} = \frac{f_\phi(a_i) \cdot f_\phi(p_j)}{\tau},$$

where $\tau > 0$ is a scalar called the **temperature** — same role as sampling
temperature in Week 29: it rescales scores before a softmax. We want $s_{ii}$ large
(anchor near its own positive) and $s_{ij}$, $j \neq i$, small (anchor far from
everyone else's positives — those are the **negatives**).

**Turning "near its own positive" into a loss.** Take a batch of $N$ pairs. For
anchor $i$, treat "which of the $N$ positives belongs to me?" as an $N$-way
classification problem. You already know the loss for classification (Week 10):
softmax + cross-entropy. The probability the model assigns to the correct match is

$$P_i = \frac{\exp(s_{ii})}{\sum_{j=1}^{N} \exp(s_{ij})},$$

and the loss is the negative log of it, averaged over the batch:

$$\mathcal{L}_{\text{InfoNCE}}
= -\frac{1}{N} \sum_{i=1}^{N} \log
\frac{\exp(s_{ii})}{\sum_{j=1}^{N} \exp(s_{ij})}.$$

This is the **InfoNCE loss** (also called the multiple-negatives ranking loss in
`sentence-transformers`, and it is the same family as NT-Xent/SimCLR in vision).
"NCE" is *noise-contrastive estimation* — the correct pair is contrasted against
"noise" alternatives. Note the trick that makes it cheap: the other $N-1$ positives
*in the same batch* serve as the negatives — **in-batch negatives** — so a batch of
$N$ pairs supplies $N(N-1)$ negative comparisons for free.

**What the gradient does.** Expand one term:

$$\mathcal{L}_i = -s_{ii} + \log \sum_{j} \exp(s_{ij}).$$

Differentiate with respect to the similarities, remembering
$\partial \log\sum_j e^{s_{ij}} / \partial s_{ik} = P_{ik}$ where
$P_{ik} = \exp(s_{ik})/\sum_j \exp(s_{ij})$ is the softmax weight (this is the same
softmax-gradient computation you did for cross-entropy in Week 13):

$$\frac{\partial \mathcal{L}_i}{\partial s_{ii}} = -(1 - P_{ii}) \le 0,
\qquad
\frac{\partial \mathcal{L}_i}{\partial s_{ik}} = P_{ik} \ge 0 \quad (k \neq i).$$

Gradient descent moves *against* the gradient, so it **pulls** the positive
similarity up with strength $(1 - P_{ii})$ and **pushes** each negative down with
strength $P_{ik}$. Read the second expression closely: the push on negative $k$ is
proportional to its softmax weight, so the negatives most similar to the anchor —
the *hard* negatives, the ones currently confusing the model — get pushed hardest.
The loss automatically focuses effort where the space is wrong.

**What the temperature does.** Cosine similarities live in $[-1, 1]$; a softmax
over numbers that close together is nearly uniform and its gradients are weak.
Dividing by a small $\tau$ (typical values 0.01–0.1) stretches the score range,
sharpening the softmax. Smaller $\tau$ concentrates the push on the very hardest
negatives; too small, and one mislabeled "negative" that is actually a paraphrase
dominates the batch gradient. $\tau$ is a real hyperparameter, not decoration.

After training on hundreds of millions of pairs, the geometry finally means
something: direction ≈ meaning. That is the object you download.

## 5. Using a pretrained embedding model

You will not train an embedding model this week — you will use one trained exactly
as in §4. Install with `uv add sentence-transformers`. A good small default is
`all-MiniLM-L6-v2` (384 dimensions, fast on CPU); check the MTEB leaderboard
(a public benchmark ranking embedding models across tasks) before assuming any
model is "the best" — leaders change monthly.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Jet quenching in central Au+Au collisions at 200 GeV",
    "Suppression of high-pT hadron yields in heavy-ion collisions",
    "Measurement of the Higgs boson mass in the diphoton channel",
]
emb = model.encode(texts, normalize_embeddings=True)   # shape (3, 384)

print(emb.shape, np.linalg.norm(emb[0]))               # (3, 384) 1.0
sims = emb @ emb[0]                                    # cosine, since unit-norm
print(np.round(sims, 3))
```

The first two texts — different words, same physics — should score visibly higher
with each other than either does with the third. `normalize_embeddings=True` is the
§2 rule applied: unit norm, then dot product.

Searching is a matrix–vector product plus a sort:

```python
q = model.encode(["parton energy loss in the quark-gluon plasma"],
                 normalize_embeddings=True)[0]
scores = emb @ q                       # (n,) cosine scores
top = np.argsort(-scores)[:5]          # indices of the 5 best, best first
for i in top:
    print(round(float(scores[i]), 3), texts[i][:60])
```

For a few thousand vectors this brute-force search runs in milliseconds. Keep that
in mind before reaching for anything fancier.

## 6. Chunking: what exactly do you embed?

Abstracts are a gift: one abstract ≈ one topic ≈ one vector. Long documents — a
40-page arXiv review, a thesis chapter, a detector design report — are not. Two
problems force you to split them:

1. **Model limits.** Embedding models truncate input beyond a maximum length
   (often 256–512 tokens; recall from Week 27 that a token is roughly ¾ of a word).
   Everything past the limit is silently ignored.
2. **Dilution.** Even within the limit, one vector for ten topics is the *average*
   of ten meanings — near none of them. A query about one topic misses it.

So you split the document into **chunks** and embed each chunk. The three standard
strategies, in increasing order of effort:

- **Fixed-size:** every $k$ tokens (or words) becomes a chunk. Trivial to write;
  happily cuts sentences and equations in half.
- **Fixed-size with overlap:** consecutive chunks share, say, 25% of their text, so
  an idea straddling a boundary appears whole in at least one chunk. Costs extra
  storage and can return near-duplicate hits.
- **Structure-aware:** split on the document's own joints — sections, paragraphs —
  merging tiny pieces and splitting oversized ones. Best retrieval, most code, and
  it needs a document whose structure survived PDF extraction.

A minimal fixed-size chunker with overlap (plain Python, Week 02 material):

```python
def chunk_words(text, size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start = start + (size - overlap)
    return chunks
```

**Do not pick a strategy by intuition.** Chunking changes retrieval quality more
than swapping embedding models does, and the winner depends on the document. The
honest procedure — exercise E5 — is a *retrieval probe*: write questions whose
answers you can point to in the document, build one index per strategy, and score
each index on whether the right chunk comes back in the top 5. Numbers, not vibes.

## 7. Storing the index: vector stores

`emb @ q` plus `np.argsort` *is* a vector index. Its two weaknesses appear at
scale: brute force is $O(nd)$ per query (fine at $10^4$ vectors, painful at $10^8$
— that is what approximate-nearest-neighbor structures like HNSW solve, and all you
need to know this month is that they trade a little recall for a lot of speed), and
a bare matrix forgets where each row came from. The second problem is yours today.
A **vector store** is a small database that persists embeddings *with their texts
and metadata* so a fresh process can search without re-embedding the corpus.
Install with `uv add chromadb`:

```python
import chromadb

client = chromadb.PersistentClient(path="index")
col = client.get_or_create_collection("abstracts",
                                      metadata={"hnsw:space": "cosine"})
col.add(
    ids=["a0", "a1", "a2"],
    embeddings=emb.tolist(),
    documents=texts,
    metadatas=[{"source": "arxiv", "section": "abstract", "page": 1},
               {"source": "arxiv", "section": "abstract", "page": 1},
               {"source": "arxiv", "section": "abstract", "page": 1}],
)

hits = col.query(query_embeddings=[q.tolist()], n_results=2)
print(hits["documents"][0])
print(hits["metadatas"][0])
```

The metadata dictionary is not decoration. Week 34's RAG system must *cite its
sources* — "which document, which section, which page" — and citation is only
possible if ingestion recorded it. Decide the metadata schema now: at minimum
`source` (file or arXiv id), `section`, and `page` where known. And keep the build
script that produced the index under version control (Week 04 discipline): an index
you cannot rebuild from scratch is an index you cannot trust.

## 8. Probing the space for failure modes

Embedding models are trained mostly on general web text, and their notion of
"similar" is *topical*, not *logical*. Three failure families matter for physics
work, and you will measure all three in exercise E4:

- **Negation.** "Jet quenching is observed" and "jet quenching is *not* observed"
  share almost every content word; their embeddings are typically very close.
  Contrastive training never forced "X" and "not X" apart — both sentences are
  *about* the same topic. Statements and their negations retrieve each other.
- **Numbers.** "$\sqrt{s_{NN}} = 200$ GeV" vs "$\sqrt{s_{NN}} = 5020$ GeV" — for a
  physicist these pick out different colliders (RHIC vs the LHC); for the model
  they are two near-identical strings that each burn a few tokens. Expect the
  similarity between them to be high, and dense retrieval for an energy-specific
  query to return the wrong energy confidently. (Week 27's tokenizer autopsy told
  you digits tokenize badly; this is the retrieval-side bill for it.)
- **Notation.** "$\pi^0$" vs "neutral pion" vs "pi zero" name one particle three
  ways. Sometimes the model knows; often — especially for detector acronyms and
  collaboration jargon that never appeared in training data — it does not.

The pattern behind all three: **embeddings are excellent at paraphrase and topic,
weak at logic, arithmetic, and rare exact symbols.** Keyword search has exactly the
mirrored profile — it nails "200 GeV" and "EMCal" and misses every paraphrase. That
complementarity is not a footnote; it is the reason Week 34 builds *hybrid* search
instead of going all-in on dense vectors.

## 9. Worked example: index and probe a mini-corpus

End to end on eight abstracts-in-miniature, so every moving part fits on one page.
Runnable as shown (first run downloads the model, ~90 MB).

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Jet quenching observed in central Au+Au collisions at 200 GeV.",
    "Parton energy loss in the quark-gluon plasma suppresses high-pT hadrons.",
    "No significant suppression of high-pT hadrons is observed in p+p data.",
    "Neutral pion spectra measured with the electromagnetic calorimeter.",
    "The pi0 yield is reconstructed from photon pairs in the EMCal.",
    "Elliptic flow of charged hadrons at sqrt(s_NN) = 5020 GeV in Pb+Pb.",
    "Machine learning methods for jet classification at the LHC.",
    "Weather patterns over the Atlantic show increased storm frequency.",
]
emb = model.encode(docs, normalize_embeddings=True)     # (8, 384)

def search(query, k):
    qv = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ qv
    order = np.argsort(-scores)[:k]
    for i in order:
        print(f"  {scores[i]:.3f}  {docs[i][:64]}")

print("Q: energy loss of fast partons in hot nuclear matter")
search("energy loss of fast partons in hot nuclear matter", 3)

print("Q: neutral pion measurement")
search("neutral pion measurement", 3)

print("negation probe:", round(float(emb[1] @ emb[2]), 3))   # suppressed vs not
print("notation probe:", round(float(emb[3] @ emb[4]), 3))   # pi0 vs neutral pion
```

What you should see, and why it matches §§2–8: the first query returns docs 0–2 —
paraphrase retrieval working exactly as the contrastive geometry promises (§4–5) —
but note that doc 2, which asserts the *opposite* conclusion, scores nearly as high
as docs 0–1: the negation failure of §8, now as a number on your screen. The second
query finds both the "$\pi^0$" and "neutral pion" phrasings — notation handled here
because both phrasings are common in training text; do not extrapolate that to
"sPHENIX" or "$R_{AA}$". The weather document scores near zero everywhere: topical
separation is the one thing this machinery does essentially perfectly.

This pattern — embed, normalize, dot product, sort, *then probe where it lies to
you* — is the whole week in one loop, and the exercises scale it to 5,000 real
abstracts and a real long document.

## Check yourself

1. Starting from the law of cosines, derive
   $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$.
2. For unit vectors, show $\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|^2 = 2(1-\cos\theta)$.
   Why does this make the cosine/dot/Euclidean choice irrelevant after
   normalization?
3. Write the InfoNCE loss for a batch of $N$ pairs. Where do the negatives come
   from, and how many negative comparisons does the batch contain?
4. Show that $\partial \mathcal{L}_i / \partial s_{ik} = P_{ik}$ for $k \neq i$.
   What does this imply about which negatives the training signal focuses on?
5. Why does mean-pooling a masked-language-model's token vectors give poor sentence
   embeddings without contrastive fine-tuning?
6. Your query is "dijet asymmetry at $\sqrt{s_{NN}} = 200$ GeV" and dense retrieval
   keeps returning 5020 GeV papers. Name the failure mode and one mitigation.
7. A colleague chunks a 40-page report into 2000-token chunks "to keep context".
   Give two distinct reasons retrieval will suffer.
8. What must ingestion record, per chunk, for Week 34's system to cite sources?

## Answers

1. Algebraic expansion gives $\|\mathbf{a}-\mathbf{b}\|^2 = \|\mathbf{a}\|^2 +
   \|\mathbf{b}\|^2 - 2\,\mathbf{a}\cdot\mathbf{b}$; the law of cosines gives
   $\|\mathbf{a}-\mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 -
   2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ for the same triangle. Equate, cancel
   the norms, divide by $-2$.
2. Set $\|\hat{\mathbf{u}}\| = \|\hat{\mathbf{v}}\| = 1$ in the expansion:
   $1 + 1 - 2\cos\theta = 2(1-\cos\theta)$. Distance is a strictly decreasing
   function of cosine, and cosine equals the dot product at unit norm, so all three
   metrics produce the same ranking.
3. $\mathcal{L} = -\frac{1}{N}\sum_i \log\left[\exp(s_{ii}) / \sum_j
   \exp(s_{ij})\right]$ with $s_{ij} = f(a_i)\cdot f(p_j)/\tau$. The negatives are
   the other pairs' positives in the same batch (in-batch negatives): $N(N-1)$
   comparisons per batch.
4. $\mathcal{L}_i = -s_{ii} + \log\sum_j e^{s_{ij}}$, and the derivative of the
   log-sum-exp with respect to $s_{ik}$ is the softmax weight $P_{ik}$. Since the
   push on each negative is proportional to $P_{ik}$, the most-similar (hardest)
   negatives receive the largest updates.
5. Pretraining optimized masked-word prediction; no term in that objective compares
   two sentence vectors, so the pooled geometry was never shaped for sentence
   similarity. Contrastive fine-tuning supplies exactly that missing objective —
   the Sentence-BERT result.
6. The numeral failure mode: embeddings barely distinguish "200" from "5020".
   Mitigations: hybrid search with a keyword component (BM25, Week 34), or metadata
   filtering on an extracted energy field (your Week 32 extractor).
7. Truncation: chunks beyond the model's maximum sequence length are silently cut,
   so much of each chunk is never embedded. Dilution: many topics averaged into one
   vector put it near none of its topics, so specific queries miss.
8. At minimum an identifier for the source document (file name or arXiv id) plus
   locators — section and page — attached as metadata to every chunk at ingest.

## New terms

- **embedding** — a learned map from text to a vector such that semantic similarity
  becomes geometric closeness.
- **cosine similarity** — $\cos\theta = \mathbf{a}\cdot\mathbf{b} /
  (\|\mathbf{a}\|\|\mathbf{b}\|)$; direction agreement, ignoring length.
- **pooling / mean pooling** — collapsing per-token transformer outputs into one
  vector, usually by averaging.
- **contrastive training** — learning by pulling matched pairs together and pushing
  mismatched pairs apart.
- **anchor / positive / negative** — the reference text, its true match, and the
  non-matches used for contrast.
- **InfoNCE loss** — cross-entropy of picking the true positive from a candidate
  set under a temperature-scaled softmax over similarities.
- **in-batch negatives** — using the other pairs in a batch as each anchor's
  negatives, giving $N(N-1)$ contrasts per batch of $N$.
- **temperature ($\tau$)** — scale factor sharpening the softmax over similarities;
  controls how much training focuses on hard negatives.
- **hard negative** — a negative currently embedded close to the anchor; receives
  the largest gradient push.
- **chunking** — splitting long documents into pieces small enough to embed as
  single coherent vectors.
- **vector store** — a database persisting embeddings with their texts and
  metadata, searchable without re-embedding.
- **approximate nearest neighbor (ANN)** — index structures (e.g. HNSW) that trade
  a little recall for large speedups over brute-force search.
- **MTEB** — a public benchmark suite ranking embedding models across retrieval,
  clustering, and classification tasks.

## Going deeper

- Reimers & Gurevych, *Sentence-BERT* (arXiv 1908.10084) — the paper behind §3:
  why pooled BERT needed a siamese contrastive objective, with the numbers.
- Lilian Weng, *Contrastive Representation Learning* (lilianweng.github.io) — the
  wider family (SimCLR, MoCo, CLIP) that InfoNCE lives in.
- `sentence-transformers` documentation, the losses overview — see
  MultipleNegativesRankingLoss and recognize §4's equation in production form.
- MTEB paper/leaderboard (HuggingFace) — how embedding models are actually
  compared, and what "best model" claims are worth.
