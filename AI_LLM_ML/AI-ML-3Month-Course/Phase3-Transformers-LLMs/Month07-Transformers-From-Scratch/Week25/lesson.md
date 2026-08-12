# Week 25 — Attention Derived

~10 hrs. Before starting you should be able to: explain why an RNN's fixed-size hidden
state throttles long-range information flow (Week 19); write softmax and its Jacobian
role in the cross-entropy gradient (Weeks 10, 13); multiply matrices and reason about
their shapes (Week 06); build a small `nn.Module` in PyTorch (Week 15).

This is the third flagship derivation of the course (after backprop in Week 13 and the
ELBO in Week 22). By the end of the week you should be able to write scaled dot-product
attention from a blank page and defend every symbol in it — especially the $1/\sqrt{d_k}$.

## 1. The problem: reading the right part of the past

Week 19 ended with a diagnosis. An RNN reads a sequence one token at a time and squeezes
everything it has seen into one fixed-size hidden vector. A **token** is one unit of the
sequence — for now think "one word" or "one character"; Week 27 makes the definition
precise. By the time token 500 arrives, whatever token 3 contributed has been overwritten
hundreds of times. Gradients flowing back through those steps shrink or explode
(backprop through time, Week 19), so long-range credit assignment fails.

What we actually want is simpler to state than to build: when processing token $t$, the
model should be able to *look back at any earlier token directly* and pull in exactly
the information it needs, no matter how far back it is. Not a summary of the past — a
lookup into the past.

Concrete example. In the sentence

> "The muon left the tracker and deposited almost nothing in **it**."

to interpret "it", the model needs "the calorimeter"... except the sentence says
"tracker". (A muon is a heavy cousin of the electron that passes through most detector
material; a tracker is the detector layer that records charged-particle paths.) Whatever
"it" refers to, resolving it means *fetching* an earlier token's content based on what
the current token is asking for. That fetch operation is attention.

## 2. From dictionary lookup to soft lookup

Start with something you have used since Week 02: a Python dictionary.

```python
d = {"color": "blue", "shape": "round", "mass": "0.135"}
print(d["shape"])   # round
```

A dictionary stores (key, value) pairs. You hand it a **query** ("shape"), it finds the
key that *exactly matches* the query, and returns that key's value. Three roles:

- **query** — what I am looking for,
- **key** — what each stored item advertises itself as,
- **value** — what you actually get back if that item is chosen.

Two problems for a neural network. First, exact matching is all-or-nothing — a query
either hits a key or it doesn't, and "how similar is this query to that key" has no
meaning. Second, and fatally: the lookup is not differentiable. `d["shape"]` is a hard
branch; there is no gradient through "which key matched", so gradient descent (Week 05)
cannot learn what to look up.

The fix is the same move the course has made twice before (hard max → softmax in
Week 10, hard latent choice → distribution in Week 22): **replace the hard choice with
a weighted average**.

Let queries, keys, and values be vectors instead of strings. Measure how well query $q$
matches key $k_i$ with a dot product $q \cdot k_i$ — large when the vectors point the
same way (Week 06). Turn the list of match scores into a probability distribution with
softmax, and return the *average of all the values, weighted by match probability*:

$$\text{out} = \sum_i \alpha_i\, v_i, \qquad \alpha_i = \operatorname{softmax}_i\!\left(q \cdot k_1,\; q \cdot k_2,\; \dots\right) = \frac{e^{\,q\cdot k_i}}{\sum_j e^{\,q\cdot k_j}}.$$

Here $q$ is the query vector, $k_i$ and $v_i$ are the key and value vectors of stored
item $i$, and $\alpha_i$ is the **attention weight** — the fraction of the output that
comes from item $i$. The weights are positive and sum to 1, so the output is an
*expectation of the values under a learned distribution over items*. If one score is
much larger than the rest, softmax puts nearly all the weight there and this reduces to
dictionary lookup. If scores are similar, the output blends several values. Every step —
dot products, exponentials, division, weighted sum — is differentiable, so the whole
lookup can be trained end to end.

Physics aside: $\alpha_i = e^{s_i}/\sum_j e^{s_j}$ is a Boltzmann distribution with the
score playing the role of (negative) energy. The $1/\sqrt{d_k}$ factor we derive in
Section 4 acts exactly like a temperature. Keep this picture; it makes the derivation
obvious.

## 3. Self-attention: Q, K, V are learned projections

In **self-attention** the queries, keys, and values all come from the same sequence —
each token both asks questions of the past and answers questions from the future.

The input is a matrix $X$ of shape $(T, d_{\text{model}})$: $T$ tokens, each represented
by an embedding vector of length $d_{\text{model}}$. An **embedding** is a learned
vector standing in for a discrete token — you built these in Week 15's character model.
From $X$, three learned weight matrices produce three views of every token:

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V,$$

with $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and
$W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. Row $t$ of $Q$ is token $t$'s query;
row $i$ of $K$ and $V$ are token $i$'s key and value. $d_k$ is the query/key length and
$d_v$ the value length (often all equal, but keep them distinct in your head — they have
different jobs).

Why three *separate* matrices, instead of using $X$ directly for all three roles? Because
the three jobs are different. The query of "it" should encode "I am a pronoun looking
for a recently mentioned object". The key of "tracker" should encode "I am a mentionable
object". Its value should carry the content that is useful *once chosen* — which may
emphasize different features than the ones that made it findable. Distinct projections
let the network learn each role independently. This is the whole trainable content of
attention: the lookup machinery is fixed; only what-asks-what and what-gets-returned is
learned.

One more property worth naming: the same $W_Q, W_K, W_V$ are applied to *every* token,
just like a convolution applies the same filter at every pixel (Week 17) and a GNN
applies the same message function at every node (Week 21). Attention has no idea, so
far, *where* a token sits — only what it contains. Week 26 fixes that with position
embeddings; this week it is a feature (fewer parameters, works for any $T$).

## 4. Scaled dot-product attention, and deriving the √d

Stack all the per-token lookups into one matrix expression. $QK^\top$ has shape
$(T, T)$; entry $(t, i)$ is $q_t \cdot k_i$, token $t$'s match score against token $i$.
Apply softmax across each *row* (each query distributes its weight over all keys), then
multiply by $V$:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.$$

Every piece is Section 2 except the $1/\sqrt{d_k}$. Deriving it is this week's
centerpiece. The argument has two halves: (a) dot-product scores *grow* with $d_k$, and
(b) large scores *kill gradients* through the softmax. Together they force the scaling.

### 4a. The variance of a dot product grows like d_k

Model the situation at initialization: the components of $q$ and $k$ are independent
random variables with mean 0 and variance 1 (that is what sensible init gives you —
Week 16). The score is a sum of $d_k$ products:

$$s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i.$$

**Mean.** Because $q_i$ and $k_i$ are independent,
$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0$, so $\mathbb{E}[s] = 0$.

**Variance.** The terms $q_i k_i$ are independent of each other (different components,
all independent), and the variance of a sum of independent terms is the sum of the
variances (Week 08). For one term, using $\mathbb{E}[q_i k_i] = 0$:

$$\operatorname{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1 \cdot 1 = 1,$$

again splitting the expectation because $q_i^2$ and $k_i^2$ are independent. Summing
$d_k$ of these:

$$\operatorname{Var}(q \cdot k) = d_k.$$

So the typical score size — the standard deviation — is $\sqrt{d_k}$. A head with
$d_k = 64$ produces scores of typical size 8; with $d_k = 1024$, size 32. The bigger the
head, the wilder the raw scores, for no informational reason at all — it is pure
dimension bookkeeping, the same random-walk scaling as $\sqrt{N}$ statistical error bars
on a counting measurement. Dividing by $\sqrt{d_k}$ makes the scaled score have variance
1 *regardless of head size*:

$$\operatorname{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1.$$

### 4b. Why large scores are fatal: softmax saturation

Why care that scores are $O(\sqrt{d_k})$ instead of $O(1)$? Because softmax with large
inputs **saturates**: one entry gets probability $\approx 1$, the rest $\approx 0$.
Compute softmax of $(8, 0, -8)$: weights $(0.99966, 0.00034, 10^{-7})$. The soft lookup
has silently become a hard one — and hard choices have no gradient.

Make that precise. Write $p = \operatorname{softmax}(z)$. The Jacobian — how weight
$p_i$ responds to a nudge in score $z_j$ — is

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j),$$

where $\delta_{ij}$ is 1 if $i = j$ and 0 otherwise. (You derived this in Week 13 inside
the cross-entropy gradient; re-derive it now on paper — two cases, $i = j$ and
$i \neq j$, quotient rule on $p_i = e^{z_i}/\sum_k e^{z_k}$.) Now read the formula in
the saturated regime: if some $p_m \approx 1$ and every other $p_i \approx 0$, then
*every* entry of the Jacobian is a product of something near 0 with something bounded —
$p_i(\delta_{ij} - p_j) \approx 0$ for all $i, j$. The gradient through the softmax is
essentially zero, in every direction.

Consequence: without scaling, a large-$d_k$ attention layer initializes *already
saturated*. Gradients cannot flow back to $W_Q$ and $W_K$, so the network can never
learn to change what it attends to — the lookup is frozen at its random initialization.
With the $1/\sqrt{d_k}$, scores start at variance 1, softmax stays soft, gradients flow.
This is the same species of failure as saturated sigmoids and dead ReLUs from Week 16,
appearing one level up.

That is the full derivation: **dot products of $d_k$-dimensional random vectors have
standard deviation $\sqrt{d_k}$ (part a), softmax saturates and its Jacobian
$p_i(\delta_{ij} - p_j)$ vanishes when inputs are large (part b), so divide scores by
$\sqrt{d_k}$ to keep them $O(1)$ at any head size.** Do it on paper, both halves, and
put the scan in this folder. It is the Phase 3 gate derivation.

## 5. Causal masking: don't read the future

We will use attention for next-token prediction: given tokens $1..t$, predict token
$t{+}1$ (your Week 15 character model, upgraded). During training the whole sequence is
in the input at once — so nothing stops position $t$'s query from matching position
$t{+}5$'s key. That is cheating: the model would "predict" the future by reading it, get
excellent training loss, and be useless for generation.

The fix is a **causal mask**: before the softmax, set every score with $i > t$ (key
later than query) to $-\infty$. Since $e^{-\infty} = 0$, those positions get exactly
zero attention weight, and the remaining weights still sum to 1 over the allowed
positions. In code, $-\infty$ is a large negative constant like `-1e9`, added to the
upper triangle of the $(T, T)$ score matrix (`np.triu`, Week 06).

Masking *before* softmax matters. Zeroing weights *after* softmax would leave the row
summing to less than 1 and would still leak gradient into future keys. With the pre-softmax
mask, the output at position $t$ is mathematically independent of every token after $t$
— a property you will verify numerically in exercise E4, by perturbing a future token
and checking the change at earlier positions is exactly zero.

## 6. Multi-head attention

One attention layer computes one weighted average per token — one "look". But "it" in
Section 1 might need to simultaneously find its referent, agree with a verb, and pick up
sentence-level context. Softmax weights sum to 1: one head attending hard at one place
cannot also attend hard elsewhere.

**Multi-head attention** runs $h$ independent attention operations ("heads") in
parallel, each in a lower-dimensional space, and concatenates the results:

$$\text{head}_j = \operatorname{Attention}(X W_Q^{(j)},\; X W_K^{(j)},\; X W_V^{(j)}), \qquad
\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O.$$

Each head has its own projections with $d_k = d_v = d_{\text{model}} / h$, so the
concatenation lands back at width $d_{\text{model}}$, and one final learned matrix
$W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ mixes the heads'
findings together. Different heads are free to learn different lookup patterns — in
trained models you find heads that attend to the previous token, heads that track
matching brackets, heads that find repeated names. Reading such patterns is exercise E6
and the seed of Week 28's interpretability work.

**Parameter count** (derive it, you will need the skill in Week 26). Per head,
$W_Q^{(j)}, W_K^{(j)}, W_V^{(j)}$ are each $d_{\text{model}} \times (d_{\text{model}}/h)$.
Across all $h$ heads that is $3 \times h \times d_{\text{model}} \cdot d_{\text{model}}/h
= 3\,d_{\text{model}}^2$ — note $h$ cancels: heads slice the space, they don't add
parameters. Add $W_O$: $d_{\text{model}}^2$ more. Total

$$4\,d_{\text{model}}^2 \quad (+\,4\,d_{\text{model}} \text{ if biases are used}).$$

For GPT-2's $d_{\text{model}} = 768$: $4 \cdot 768^2 = 2{,}359{,}296$ weights per
attention layer. Memorize the "$4d^2$" shape of this answer.

In code, all heads are computed at once: project $X$ to shape $(T, 3 d_{\text{model}})$,
split into $Q, K, V$, reshape each to $(h, T, d_k)$, and let batched matrix multiply do
the rest. Same math, one kernel.

## 7. Worked example: attention by hand, then in NumPy

First by hand, small enough to trust. Two tokens, $d_k = 2$, one head, no scaling
suspense — we include it. Let

$$q_2 = (1, 1), \qquad k_1 = (1, 0),\; k_2 = (0, 1), \qquad v_1 = (2, 0),\; v_2 = (0, 4).$$

Token 2's scores against both keys: $q_2 \cdot k_1 = 1$, $q_2 \cdot k_2 = 1$. Scale by
$1/\sqrt{2}$: both $0.7071$. Softmax of two equal scores: $\alpha = (0.5, 0.5)$. Output:
$0.5 \cdot (2,0) + 0.5 \cdot (0,4) = (1, 2)$. A blend, because the query matched both
keys equally.

Now the general implementation, runnable as shown:

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # subtract row max: same output, no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(X, W_Q, W_K, W_V, causal):
    T, d_model = X.shape
    d_k = W_Q.shape[1]
    Q = X @ W_Q                              # (T, d_k)
    K = X @ W_K                              # (T, d_k)
    V = X @ W_V                              # (T, d_v)
    scores = Q @ K.T / np.sqrt(d_k)          # (T, T): scores[t, i] = q_t . k_i / sqrt(d_k)
    if causal:
        mask = np.triu(np.ones((T, T)), k=1) # 1s strictly above the diagonal (future)
        scores = scores - 1e9 * mask         # -inf in effect: zero weight after softmax
    A = softmax(scores)                      # (T, T): each row sums to 1
    return A @ V, A                          # (T, d_v), and the weights for inspection

rng = np.random.default_rng(0)
T, d_model, d_k = 5, 8, 4
X = rng.normal(size=(T, d_model))
W_Q = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
W_K = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
W_V = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)

out, A = attention(X, W_Q, W_K, W_V, causal=True)
print(out.shape)                  # (5, 4)
print(np.round(A, 3))             # lower-triangular: no weight on the future
print(A.sum(axis=1))              # each row sums to 1
```

Read the printed $A$: row $t$ is token $t$'s attention distribution. Row 0 must be
$(1, 0, 0, 0, 0)$ — the first token can only attend to itself. Every entry above the
diagonal must be 0. If you can predict those two facts before running the cell, you
understand the mask.

In the exercises you will extend exactly this code to batches and multiple heads, then
check it against PyTorch's built-in to five decimal places.

## Check yourself

1. In the dictionary analogy, which of query/key/value does each of these play: "what
   the current token is looking for", "what a stored token returns when selected",
   "what a stored token advertises"?
2. State $\operatorname{Var}(q \cdot k)$ for $q, k$ with i.i.d. zero-mean unit-variance
   components in $d_k$ dimensions, and give the two-line argument.
3. Softmax Jacobian: what is $\partial p_i / \partial z_j$, and what happens to it when
   one $p_m \to 1$?
4. Why is the causal mask applied before the softmax rather than zeroing weights after?
5. A multi-head layer has $d_{\text{model}} = 512$ and $h = 8$. What is $d_k$ per head,
   and how many weight parameters does the whole layer have (no biases)?
6. Attention applies the same $W_Q$ to every token. Name the analogous
   parameter-sharing property in CNNs (Week 17) and in GNNs (Week 21).
7. Why can attention connect token 500 to token 3 with a healthy gradient when an RNN
   cannot?
8. What would go wrong if we used one shared matrix for both keys and values?

## Answers

1. Query = what the current token is looking for; value = what a stored token returns
   when selected; key = what it advertises.
2. $\operatorname{Var}(q \cdot k) = d_k$. Each term $q_i k_i$ has mean 0 and variance
   $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$; the $d_k$ terms are independent, so
   variances add.
3. $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$. When $p_m \to 1$ and the rest
   $\to 0$, every entry is (near-0) × (bounded), so the whole Jacobian $\to 0$:
   gradients stop flowing through the softmax.
4. Pre-softmax masking with $-\infty$ gives *exactly* zero weight and keeps the
   remaining weights summing to 1; post-softmax zeroing breaks normalization and the
   masked scores would still have received gradient through the softmax denominator.
5. $d_k = 512/8 = 64$; parameters $= 4 d_{\text{model}}^2 = 4 \cdot 512^2 = 1{,}048{,}576$.
6. CNN: the same filter slides over every spatial position. GNN: the same message/update
   function is applied at every node. All three reuse one function across positions, so
   parameter count is independent of input size.
7. The attention output at token 500 is a *direct* weighted sum that includes $v_3$ —
   one softmax and one matmul between them, not 497 recurrent steps. The gradient path
   has length ~2 regardless of distance.
8. Findability and content would be forced to be the same vector. A token could not
   advertise itself one way ("I am a noun") while delivering different information once
   selected; keys optimized for matching would corrupt values optimized for content, and
   vice versa.

## New terms

- **token** — one unit of a sequence (word, character, or Week 27's subword).
- **attention** — differentiable lookup: output = values averaged under softmax-ed
  query–key match scores.
- **query / key / value** — learned projections of a token: what it seeks / what it
  advertises / what it returns.
- **attention weight** ($\alpha_i$) — softmax-ed match probability; how much of value
  $i$ enters the output.
- **self-attention** — attention where Q, K, V all come from the same sequence.
- **scaled dot-product attention** — $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$; the
  scaling keeps score variance at 1 for any $d_k$.
- **softmax saturation** — large-magnitude inputs drive softmax to a near-one-hot
  output, where its Jacobian $p_i(\delta_{ij}-p_j)$ vanishes.
- **causal mask** — pre-softmax $-\infty$ on future positions so token $t$'s output
  cannot depend on tokens $> t$.
- **multi-head attention** — $h$ parallel attentions in $d_{\text{model}}/h$-dimensional
  subspaces, concatenated and mixed by $W_O$; $4d_{\text{model}}^2$ weights total.
- **head** — one of the parallel attention operations, with its own Q/K/V projections.

## Going deeper

- Karpathy, *Let's build GPT: from scratch, in code, spelled out* — first half, through
  the self-attention block. Watch *after* your own implementation works; his "version 1
  → version 4" build-up will then feel like watching your own path.
- 3Blue1Brown, *But what is a GPT?* and *Attention in transformers, visually explained*
  — the best animations of the Q/K/V geometry; excellent for making Section 3 visual.
- Prince, *Understanding Deep Learning*, Ch. 12, self-attention sections — a second,
  more formal pass over exactly this material, free PDF.
- Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762), §3.2 only this week —
  read the primary source for the equation you just derived; note their one-sentence
  footnote version of your variance argument. The rest of the paper is Week 26.
