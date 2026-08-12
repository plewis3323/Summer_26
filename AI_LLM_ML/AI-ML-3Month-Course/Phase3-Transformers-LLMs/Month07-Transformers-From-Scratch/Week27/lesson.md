# Week 27 — Tokenization + nanoGPT (GPU week)

~12 hrs (GPU week — budget the training run). Before starting you should be able to:
run your `model.py` GPT with a hand-verified parameter count (Week 26); write a
training loop with AdamW and checkpointing (Weeks 15–16); fetch and cache a remote
dataset with provenance (Week 04); state why loss ≈ ln(vocab) at init (Weeks 16, 26).

Two builds this week. First: the tokenizer — the byte-pair encoding algorithm, derived
step by step on a corpus small enough to trace by hand. Second: the training run — your
Week 26 model plus your tokenizer, trained on real physics abstracts on a free GPU.
This is the month's flagship derivation number three: BPE, worked in full.

## 1. The tokenization problem

A language model consumes integer token ids. Something must convert raw text to
integers and back. That something — the **tokenizer** — is chosen *before* training and
frozen forever after; every strength and quirk it has is baked into the model. Think of
it as the detector layer of an LLM: a lossy, quirky digitization that every downstream
inference inherits.

The two obvious designs both fail:

- **Character-level** (your Week 15 model): tiny vocabulary (~100), no unknown-text
  problem — but sequences are long. A 1000-word abstract is ~6000 characters, and
  attention cost grows as $T^2$ (that $(T, T)$ score matrix from Week 26's shape-flow
  table). Long sequences also spread meaning thin: the model spends capacity learning
  that q-u-a-r-k spells quark.
- **Word-level**: short sequences, meaningful units — but the vocabulary is unbounded.
  "Hyperon", "$\sqrt{s_{NN}}$", "eta_meson_v2", a typo, a new detector name: each is
  either in your fixed table or an unknown token `<UNK>`, and `<UNK>` destroys
  information irreversibly.

The compromise every modern LLM uses is **subword tokenization**: frequent strings
(common words) become single tokens; rare strings fall apart into smaller pieces, down
to single bytes if necessary. Nothing is ever unknown, and typical text is ~4 characters
per token. The dominant algorithm for building the subword vocabulary is **byte-pair
encoding (BPE)** — originally a 1994 compression trick, and simple enough to derive
completely in one page.

## 2. BPE derived: the algorithm on a corpus you can trace

The idea in one sentence: **start from the smallest possible units, then repeatedly
merge the most frequent adjacent pair into a new token, until the vocabulary is as big
as you want.** Frequency decides everything; the data designs its own vocabulary.

The algorithm:

1. Split the training text into base units (characters here; bytes in Section 3).
2. Count every *adjacent pair* of units in the corpus.
3. Find the most frequent pair. Tie-break rule: the pair whose first occurrence is
   earliest in the text (any consistent rule works; this is the one our reference code
   implements, so use it).
4. Merge: replace every occurrence of that pair with a single new token; record the
   merge rule.
5. Repeat from step 2 until you have done `vocab_size − base_size` merges.

Now trace it, fully, on a tiny corpus (37 characters):

```
"beam energy beam energy beam energies"
```

**Step 1 — pair counts.** Scanning the string once (spaces are units too — pairs 15,
occurrences 36 total):

| pair | count | pair | count |
|---|---|---|---|
| (b,e) | 3 | (n,e) | 3 |
| (e,a) | 3 | (e,r) | 3 |
| (a,m) | 3 | (r,g) | 3 |
| (m,␣) | 3 | (g,y) | 2 |
| (␣,e) | 3 | (y,␣) | 2 |
| (e,n) | 3 | (␣,b) | 2 |
|  |  | (g,i), (i,e), (e,s) | 1 each |

Nine pairs tie at count 3. The tie-break picks `(b,e)` — earliest first occurrence
(position 0). **Merge 1: `(b,e) → be`.** The corpus becomes (now 34 units):

```
be a m ␣ e n e r g y ␣ be a m ␣ e n e r g y ␣ be a m ␣ e n e r g i e s
```

**Merges 2–4.** Recount, merge, repeat. Each time, the winner is the pair that extends
the growing chunk (count 3, earliest occurrence):

| merge | pair | new token | corpus length after |
|---|---|---|---|
| 2 | (be, a) | `bea` | 31 |
| 3 | (bea, m) | `beam` | 28 |
| 4 | (beam, ␣) | `beam␣` | 25 |

After four merges, the corpus reads

```
beam␣ e n e r g y ␣ beam␣ e n e r g y ␣ beam␣ e n e r g i e s
```

The frequent word "beam" (with its trailing space) has *become a single token* — nobody
told the algorithm about words; frequency alone found it. This is why LLM tokenizers
are full of tokens like `␣the` and `␣energy`: the space travels with the word.

**Merges 5–6 — and a lesson.** Continue mechanically: the most frequent pairs are now
`(beam␣, e)` then `(beam␣e, n)` (count 3 each, earliest occurrence). Raw BPE happily
merges *across* the word boundary, building the useless token `beam␣en`. Frequency has
no concept of "word". Real tokenizers stop this: GPT-2 first splits text into chunks
with a regular expression (word-ish pieces, number pieces, punctuation, leading space
attached) and runs BPE *within* chunks only, so merges never cross a word boundary.
Remember both facts: plain BPE crosses spaces; production BPE adds a splitting pattern.

**Encoding and decoding.** Training produced an ordered list of merge rules. To
*encode* new text: split into base units, then apply every merge rule **in training
order** wherever it matches. Order matters — rule 3 `(bea, m)` can only fire after
rules 1–2 have built `bea`. To *decode*: every token knows the string it was built
from; concatenate. Decoding is trivial and lossless — always, by construction — which
is exercise E1's round-trip test.

Try it: encode `"beam energy"` with the six rules above. You should get
`[beam␣en, e, r, g, y]` — wait, no: encoding applies rules to `"beam energy"`, whose
final chunk is `energy` not `energies`; work it through and you get
`[beam␣en, e, r, g, y]` only if rule 6 matches — it does ($beam␣$ then $e$, $n$). Five
tokens for eleven characters. If your hand answer disagrees with your code in E1, the
bug is almost always applying merges in frequency order instead of training order.

## 3. Byte-level BPE: why bytes, and what UTF-8 is

Characters were fine for tracing, but real tokenizers start from **bytes**. Here is
why, and what that means.

Computers store text as bytes — integers 0–255 — under an encoding. **UTF-8**, the
universal standard, represents each character as 1–4 bytes: plain English letters take
1 byte; π takes 2 (`0xCF 0x80`); the superscript ⁰ takes 3; so the string `"π⁰"` is
5 bytes for 2 characters. (A π⁰, while we are here, is the neutral pion — the lightest
meson, which decays to two photons; its symbol shows up constantly in heavy-ion text
and is a perfect tokenizer stress test.)

If BPE's base units are the 256 possible bytes, then *any* text in any language — math
symbols, emoji, corrupted files — is representable with zero unknown tokens: worst
case, it falls back to raw bytes. The vocabulary is exactly

$$\text{vocab} = 256 \text{ base bytes} + \text{number of merges},$$

and new token ids count up from 256. That is GPT-2's design (§2.2 of the paper) and
what you build in E1: `train` on the *bytes* of the corpus (`text.encode("utf-8")`),
merge byte pairs exactly as in Section 2, decode by concatenating byte sequences and
calling `bytes.decode("utf-8", errors="replace")` — the `errors="replace"` guards the
one sharp edge, a token boundary landing mid-character in a multi-byte sequence.

## 4. Tokenizer pitfalls: know your detector's dead channels

Once text becomes tokens, the model never sees letters again. A whole family of LLM
failures is really tokenizer physics:

- **Counting letters.** "How many r's in strawberry?" is hard because the model
  receives `strawberry` as one or two tokens — opaque ids — not eleven letters. It must
  have *memorized* spelling facts about each token; it cannot look.
- **Whitespace sensitivity.** `"energy"` and `" energy"` are different tokens with
  different learned vectors (Section 2's traveling space). Prompts that end with a
  trailing space put the model off-distribution — a classic silent quality bug.
- **Digits.** BPE chunks numbers by frequency, not place value: `200` might be one
  token, `201` two (`20`, `1`). Arithmetic across arbitrary chunkings is hard, and
  physics text is full of numbers: `√s_NN = 200 GeV` tokenizes very differently from
  `√s_NN = 62.4 GeV`.
- **Non-ASCII science.** π⁰, √, subscripts, Greek in general: multi-byte characters a
  general-purpose tokenizer saw rarely become long byte-fallback sequences — many
  tokens for one glyph. Your domain tokenizer, trained on physics abstracts, will merge
  them tightly. E3's autopsy quantifies exactly this difference against GPT-2's
  tokenizer via `tiktoken`.

The design tension behind all of this: bigger vocabulary → shorter sequences (cheaper
attention) but a bigger embedding table and rarer, worse-trained tokens. GPT-2 chose
50257; you will choose ~4096, sized to a ~1 MB corpus.

## 5. The corpus: fetching abstracts from the arXiv API

You need training text. We use abstracts from **arXiv**, the physics preprint server —
free, public, and equipped with a machine-readable API. The categories: `nucl-ex`
(nuclear experiment — heavy-ion collisions live here), `nucl-th` (nuclear theory), and
`hep-ex` (high-energy experiment). A **heavy-ion collision**, for the record, is two
heavy nuclei (gold, lead) collided at near light speed to create quark–gluon plasma —
the deconfined state of quarks and gluons; its abstracts are dense with exactly the
symbols from Section 4.

The fetch-script pattern below is a reusable skill (Week 34's RAG corpus and Week 45's
paper pipeline reuse it): query in pages, be polite, parse, deduplicate, record
provenance. The API returns **Atom XML** — a structured text format where each paper is
an `<entry>` element; Python's built-in `xml.etree.ElementTree` walks it. Complete and
runnable (install with `uv add requests`):

```python
import json
import time
import requests
import xml.etree.ElementTree as ET

BASE = "http://export.arxiv.org/api/query"        # stable, documented endpoint
NS = {"atom": "http://www.w3.org/2005/Atom"}      # XML namespace for Atom elements
QUERY = "cat:nucl-ex OR cat:nucl-th OR cat:hep-ex"
PAGE = 200                                        # results per request
N_WANTED = 5000

seen_ids = []
abstracts = []
start = 0
while len(abstracts) < N_WANTED:
    params = {"search_query": QUERY, "start": start, "max_results": PAGE,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    r = requests.get(BASE, params=params, timeout=30)
    entries = ET.fromstring(r.text).findall("atom:entry", NS)
    if len(entries) == 0:                         # API hiccup or end of results
        time.sleep(10)
        continue
    for e in entries:
        arxiv_id = e.find("atom:id", NS).text
        summary = e.find("atom:summary", NS).text
        if arxiv_id not in seen_ids:              # deduplicate by arXiv id
            seen_ids.append(arxiv_id)
            abstracts.append(" ".join(summary.split()))   # collapse whitespace
    start = start + PAGE
    print(len(abstracts), "abstracts")
    time.sleep(3)                                 # arXiv asks for 3 s between calls

with open("corpus.txt", "w") as f:
    f.write("\n\n".join(abstracts))

provenance = {"query": QUERY, "endpoint": BASE, "count": len(abstracts),
              "date": time.strftime("%Y-%m-%d"), "page_size": PAGE}
with open("provenance.json", "w") as f:
    json.dump(provenance, f, indent=2)
```

Points of technique, each of which will bite you somewhere else if unlearned:
**paging** (`start`/`max_results` — never ask for everything at once), **politeness**
(the 3-second sleep is in arXiv's terms of use; APIs ban abusers), **retry on empty**
(the API occasionally returns an empty page; back off and repeat rather than crash),
**deduplication** (papers get cross-listed between categories, so the same id arrives
twice), and **provenance** (Week 04's rule: a data file without the query, date, and
count that produced it is not reproducible — API contents drift, and "within API
drift" is the best a re-run can promise).

Expect ~5000 abstracts ≈ 1 MB of text. That is *small* — hold that thought for
Section 7.

## 6. The training run: nanoGPT-style practice on a free GPU

**The recipe.** You have all the pieces: Week 26's `model.py`, your tokenizer, a
corpus. What Week 16's training loop needs, upgraded to language-model practice (these
are the conventions in Karpathy's `nanoGPT` `train.py`, which you should read this week
as literature — code written by someone who has made every mistake already):

- **Data pipeline**: encode the whole corpus once to a flat array of token ids; a
  batch is `batch_size` random windows of `ctx+1` consecutive ids — inputs are the
  first `ctx`, targets the same window shifted by one. Split 90/10 train/val *by
  position in the corpus*, not per-window, so val windows never overlap train text.
- **AdamW** (Week 08's Adam + decoupled weight decay, ~0.1 on the weight matrices,
  none on LayerNorm/bias parameters).
- **LR warmup + cosine decay**: ramp linearly from 0 over the first few hundred steps
  — a fresh AdamW has garbage moment estimates and pre-norm or not, big early steps
  can wreck the run — then decay smoothly to ~1/10 of peak. Peak lr ≈ 1e-3 at this
  model size.
- **Gradient clipping** at norm 1.0 (Week 19's exploding-gradient insurance).
- **Checkpointing**: save model + optimizer state every ~500 iters *to storage that
  survives the session* — free GPUs disconnect without warning.
- **Dropout** ≈ 0.2: your corpus is small (see below); regularize.

**The size that is honest for a free GPU.** Colab and Kaggle free tiers give you a
T4-class GPU (16 GB) in sessions of a few hours, with weekly quotas — the spec, not
your laptop, sets the budget. A config that trains in well under an hour there:

| n_layer | n_head | d_model | ctx | vocab | dropout | ≈ params |
|---|---|---|---|---|---|---|
| 6 | 6 | 384 | 256 | 4096 | 0.2 | ~12 M |

Hand-check that count with Week 26's formula before trusting it: $12d^2 + 13d$ per
block with $d = 384$ gives $1{,}774{,}080$; six blocks $10{,}644{,}480$; tied
embeddings $4096 \times 384 = 1{,}572{,}864$; positions $256 \times 384 = 98{,}304$;
final LN $768$. Total $\approx 12.3$ M. Batch size 64 at ctx 256 fits a T4 with room
to spare; ~5000 iterations is a reasonable first budget.

**The honesty check: know your token budget.** ~1 MB of physics text at roughly 4
bytes/token is ~250 k tokens. Week 28 will teach you the compute-optimal ratio (~20
tokens per parameter); by that yardstick a 12 M-parameter model "wants" ~240 M tokens —
you have *one thousandth* of that. Consequences to expect and report, not hide: the
model will loop over the corpus many times, train loss will keep falling while **val
loss bottoms out and rises — overfitting** (Week 09) — so track both, keep the
checkpoint at the val minimum, and treat the val curve as the result. This is also the
argument against training something bigger: more parameters would memorize the corpus
faster, not model it better. If you want a genuinely better model, the lever is *more
abstracts* (the fetch script caps at `N_WANTED`, not at what arXiv has).

**The baseline.** "My model learned something" needs a number to beat. The **bigram
model** is the classic floor: $P(\text{next token} \mid \text{current token})$
estimated by counting pairs in the training set — a giant conditional-probability
lookup table, the maximum-likelihood estimate you derived in Week 08. Its val loss
costs one array pass to compute. Uniform guessing scores $\ln(4096) \approx 8.32$ nats;
the bigram typically lands around 5–6 on text like this; your acceptance bar (E5) is
beating it by ≥ 0.5 nats. Sampling from the bigram also calibrates your eye: its output
is word-salad with plausible local pairs, so whatever your transformer produces beyond
that is what attention bought you.

**Free-GPU workflow** (Colab and Kaggle differ in buttons, not substance): enable the
GPU runtime; upload `model.py`, `bpe.py`, `corpus.txt` (or clone your repo); verify
`torch.cuda.is_available()`; attach persistent storage (Colab: mount Google Drive;
Kaggle: `/kaggle/working` persists with the saved notebook) and point checkpoints at
it; train; download the checkpoint, loss curves, and samples before the session ends.
Treat the session as pre-emptible: if you cannot resume from your latest checkpoint
after a simulated disconnect (E5 requires demonstrating this once), you do not have
checkpointing, you have wishful thinking.

**Sampling.** Generation is Week 15's loop at scale: feed a prompt, take the logits at
the last position, divide by a **temperature** (0.8 here — mild sharpening; Week 29
treats sampling properly), softmax, sample, append, repeat. Judge the samples like a
physicist reading a student's abstract (E6): the grammar will impress you; then look
for fabricated references, impossible energies, detector names in the wrong experiment.
Plausible-but-wrong is the failure mode of the entire field — meet it early, in a model
you built, where you know *exactly* what it was and wasn't trained on.

## 7. Worked example: BPE end to end in 40 lines

Complete and runnable — this is the reference implementation for E1, on the Section 2
corpus so you can check it against the hand trace:

```python
def pair_counts(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out

def train(text, n_merges):
    ids = list(text.encode("utf-8"))          # base units: bytes 0..255
    merges = []                                # ordered list of (pair, new_id)
    for step in range(n_merges):
        counts = pair_counts(ids)
        best = max(counts, key=counts.get)     # ties: first-seen pair wins
        new_id = 256 + step
        ids = merge(ids, best, new_id)
        merges.append((best, new_id))
    return merges

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges:                # training order, not frequency order
        ids = merge(ids, pair, new_id)
    return ids

def decode(ids, merges):
    parts = {i: bytes([i]) for i in range(256)}
    for pair, new_id in merges:
        parts[new_id] = parts[pair[0]] + parts[pair[1]]
    return b"".join(parts[i] for i in ids).decode("utf-8", errors="replace")

corpus = "beam energy beam energy beam energies"
merges = train(corpus, 6)
for (a, b), new_id in merges:
    print(new_id, "<-", a, "+", b)

toks = encode("beam energy", merges)
print(toks, "->", repr(decode(toks, merges)))
print("round trip ok:", decode(encode(corpus, merges), merges) == corpus)
```

Run it. The six printed merges must reproduce Section 2's table exactly (ids 256–261
building `be`, `bea`, `beam`, `beam␣`, `beam␣e`, `beam␣en` — as bytes now, but the same
pairs in the same order), `"beam energy"` must encode to 5 tokens, and the round trip
must print `True`. When it does, you have written the same algorithm that tokenizes
every prompt sent to a production LLM — smaller vocabulary, identical mechanics. E2
then checks your merge order against Karpathy's `minbpe` on a larger text, and the rest
of the week is Section 5's fetch and Section 6's run.

## Check yourself

1. Why not character-level tokens? Why not word-level? One cost each.
2. Recite the BPE training loop in four steps, including the tie-break we use.
3. In the worked corpus, why did the space end up *inside* the token `beam␣`, and what
   do production tokenizers add to stop merges crossing into the next word?
4. Why must encoding apply merges in training order?
5. Byte-level BPE has what base vocabulary size, and what happens to text the tokenizer
   never saw during training?
6. Why is "how many r's in strawberry" hard for an LLM?
7. Your 12 M-parameter model trains on ~250 k tokens. What will the train and val loss
   curves do, and what do you keep as "the model"?
8. What three things does the provenance file record, and what claim does it license?

## Answers

1. Characters: sequences ~4–6× longer, and attention cost grows as $T^2$; capacity
   wasted learning spelling. Words: unbounded vocabulary — anything unseen becomes
   `<UNK>` and its information is destroyed.
2. Count adjacent pairs; take the most frequent (tie → earliest first occurrence in the
   text); replace all its occurrences with a new token and record the rule; repeat
   until the target number of merges.
3. `(m,␣)` and then `(beam,␣)` were among the most frequent pairs because every "beam"
   is followed by a space — frequency does not know about words. Production tokenizers
   (GPT-2) pre-split text with a regex and run BPE within chunks only.
4. Later merges are built from the outputs of earlier ones — `(bea, m)` cannot match
   until `(b,e)` and `(be,a)` have fired. Applying rules out of order produces
   different (wrong) tokenizations of exactly the strings the model was trained on.
5. 256 (all byte values). Unseen text falls back to shorter and shorter merges, in the
   worst case raw bytes — many tokens, but never unknown.
6. The model receives "strawberry" as one or two opaque token ids, not letters; it
   cannot inspect spelling, only recall memorized facts about those tokens.
7. Train loss falls steadily; val loss bottoms out and rises as the model memorizes the
   small corpus (overfitting). Keep the checkpoint at the val-loss minimum and report
   the val curve.
8. Query, date, and count (plus endpoint/page size). It licenses reproducibility "up to
   API drift": a re-run with the same query on a later date should produce nearly the
   same corpus and explain any difference.

## New terms

- **tokenizer** — fixed text↔integer-ids map chosen before training; the model's
  "digitization layer".
- **subword tokenization** — frequent strings become single tokens, rare ones split
  into pieces; no unknown tokens.
- **byte-pair encoding (BPE)** — greedy vocabulary builder: repeatedly merge the most
  frequent adjacent pair into a new token.
- **merge rule** — one recorded pair→new-token replacement; applied in training order
  at encode time.
- **byte-level BPE** — BPE with the 256 byte values as base units; ids 256+ are merges.
- **UTF-8** — the standard text encoding; 1–4 bytes per character (π is 2, ⁰ is 3).
- **Atom XML** — the structured feed format the arXiv API returns; parsed with
  `ElementTree`.
- **provenance file** — recorded query/date/count for a fetched dataset (Week 04's
  reproducibility rule applied to APIs).
- **bigram model** — next-token predictor from pair counts alone; this week's baseline.
- **temperature** — divisor on logits before softmax during sampling; lower = sharper.
- **quark–gluon plasma** — deconfined quark/gluon state made in heavy-ion collisions;
  the subject of your corpus.

## Going deeper

- Karpathy, *Let's build the GPT Tokenizer* (video), with the `minbpe` repo as
  reference code — watch *after* your E1 passes; his regex-splitting and special-token
  sections extend Section 3 to production reality.
- Karpathy's `nanoGPT` repo, `train.py` — read as literature for the training-loop
  practices in Section 6 (warmup + cosine, clipping, AdamW grouping, checkpoint/resume).
- Radford et al., GPT-2 paper, §2.2 — the one-page rationale for byte-level BPE, from
  the people who shipped it at 50257 tokens.
- arXiv API documentation (search "arXiv API user manual") — the query grammar,
  paging, and rate-limit rules behind Section 5's script.
