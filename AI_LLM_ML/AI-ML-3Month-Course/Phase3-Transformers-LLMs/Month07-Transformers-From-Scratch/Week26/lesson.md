# Week 26 — The Full Transformer

~10 hrs. Before starting you should be able to: write scaled dot-product attention with
the √d_k argument and causal masking, and have a working `MultiHeadAttention` module
(Week 25); explain LayerNorm and why init should give loss ≈ ln(vocab) (Week 16);
state the "gradient highway" argument for ResNet skip connections (Week 17); train a
character-level next-token model in PyTorch (Week 15).

This week you assemble the attention module into a complete GPT-style language model,
count its parameters by hand down to the last bias, trace a batch's shape through every
operation, and then read the two founding papers knowing which parts matter.

## 1. What we are building

A **decoder-only transformer** (the GPT architecture) is a next-token predictor: given
integer token ids $t_1, \dots, t_T$, it outputs, at every position, a probability
distribution over the vocabulary for the *next* token. Training is Week 15's setup at
scale — cross-entropy between the predicted distribution at position $i$ and the actual
token $t_{i+1}$, all positions in parallel thanks to the causal mask.

The name needs decoding itself. The original transformer (Vaswani et al., 2017) was
built for translation and had two towers: an *encoder* that reads the source sentence
with unmasked attention, and a *decoder* that generates the target sentence with causal
attention plus "cross-attention" into the encoder. GPT keeps only the decoder tower,
drops cross-attention, and trains on plain next-token prediction. That is the
architecture behind essentially every modern LLM, and the one you build here.

The structure, top to bottom:

```
token ids (B, T)
  → token embedding + position embedding        → (B, T, d)
  → Block 1: [LayerNorm → attention → add] then [LayerNorm → MLP → add]
  → Block 2: same
  → ... n_layer blocks ...
  → final LayerNorm
  → linear head to vocabulary                   → (B, T, vocab)
```

Five configuration numbers pin the whole model down: `n_layer` (number of blocks),
`d_model` (embedding width, called `n_embd` in code), `n_head`, `vocab_size`, and
`ctx` (maximum sequence length, a.k.a. context length or block size). GPT-2 small is
(12, 768, 12, 50257, 1024). By Section 5 you will turn those five numbers into an exact
parameter count.

## 2. Token and position embeddings

Attention, as derived last week, is **permutation-equivariant**: shuffle the input
tokens (and their mask) and the outputs shuffle the same way — nothing in
$\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ knows *where* a token sits, only what it
contains. Language is not permutation-invariant ("dog bites man" ≠ "man bites dog"), so
position information must be injected explicitly.

The GPT solution is blunt and effective: alongside the **token embedding** table
$W_E \in \mathbb{R}^{\text{vocab} \times d}$ (row $t$ = the learned vector for token
id $t$, exactly Week 15's `nn.Embedding`), keep a **position embedding** table
$W_P \in \mathbb{R}^{\text{ctx} \times d}$ (row $p$ = a learned vector for *being at
position $p$*), and add them:

$$x_p = W_E[t_p] + W_P[p].$$

Now two identical tokens at different positions enter the network as different vectors,
and attention can learn position-dependent behavior (e.g. "attend to the previous
position"). The cost: the model cannot run beyond `ctx` positions — there is no row
1025 in a 1024-row table.

Two alternatives you must be able to discuss:

**Sinusoidal encodings** (the original paper): no learned table; position $p$ gets a
fixed vector of sines and cosines at geometrically spaced frequencies,

$$PE(p)_{2i} = \sin(p/10000^{2i/d}), \qquad PE(p)_{2i+1} = \cos(p/10000^{2i/d}).$$

The design property — derive it on paper this week — is that a *shift* in position is a
*fixed linear map*. Fix an offset $k$ and look at one frequency's $(\sin, \cos)$ pair.
The angle-addition formulas from high-school trig give

$$\begin{pmatrix}\sin(\omega_i(p+k))\\ \cos(\omega_i(p+k))\end{pmatrix}
= \begin{pmatrix}\cos(\omega_i k) & \sin(\omega_i k)\\ -\sin(\omega_i k) & \cos(\omega_i k)\end{pmatrix}
\begin{pmatrix}\sin(\omega_i p)\\ \cos(\omega_i p)\end{pmatrix},$$

with $\omega_i = 1/10000^{2i/d}$ — a rotation matrix that depends on $k$ but *not on
$p$*. Stacking the per-frequency rotations block-diagonally: $PE(p+k) = M_k\,PE(p)$ for
every $p$. So "the token $k$ steps back" is one linear transform away, and a linear
layer can learn relative offsets without ever being told about them. You will verify
this numerically in exercise E4 by fitting $M_k$ from data and checking the error is at
machine precision.

**RoPE (rotary position embedding)**, used by most current open models: instead of
adding position vectors to the input, rotate each 2-dimensional slice of the *query and
key* vectors by an angle proportional to position — $q_p \mapsto e^{i p \theta} q_p$ in
complex notation, one $\theta$ per slice. A rotation by $p\theta$ on the query and
$p'\theta$ on the key changes their dot product by exactly the *difference*
$(p - p')\theta$ — the same phase-difference trick as an interferometer — so attention
scores depend only on relative position, which is what grammar actually needs. Concept
level is enough this week; Su et al. §3.2 has the details.

## 3. The residual stream

Here is the mental model that makes transformers legible. Write one block's computation
as

$$x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x)), \qquad
x \leftarrow x + \operatorname{MLP}(\operatorname{LN}(x)).$$

The vector $x$ that flows down the depth of the network — width $d$, one per position —
is called the **residual stream**. Notice what the equations say: no module ever
*replaces* $x$; every module reads $x$ (through a LayerNorm), computes a correction, and
*adds* it back. The stream is a shared bus that all blocks read from and write to. A
physicist's picture: the identity map is the zeroth-order solution, and each block adds
a perturbative correction on top of everything accumulated so far.

**Why this wins: gradient flow.** Recall Week 17's ResNet argument, now made exact.
For a chain $x_{l+1} = x_l + F_l(x_l)$, the chain rule gives

$$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F_l}{\partial x_l},$$

and across $L$ blocks

$$\frac{\partial L_{\text{loss}}}{\partial x_0}
= \frac{\partial L_{\text{loss}}}{\partial x_L}\prod_{l=0}^{L-1}\left(I + \frac{\partial F_l}{\partial x_l}\right)
= \frac{\partial L_{\text{loss}}}{\partial x_L}\left(I + \sum_l \frac{\partial F_l}{\partial x_l} + \dots\right).$$

Expand the product: the leading term is the identity — a gradient path from the loss to
the very first layer that passes through *no* weight matrices at all. However badly some
$\partial F_l/\partial x_l$ behaves, that clean path exists. Compare a plain stack
$x_{l+1} = F_l(x_l)$, where the gradient is a product of $L$ Jacobians and shrinks or
explodes geometrically — Week 19's vanishing-gradient story. Residual connections are
why 12, 48, or 96 blocks train at all.

**Pre-norm vs post-norm.** Where LayerNorm sits decides whether that clean path
survives. (LayerNorm, from Week 16: normalize each token's $d$-vector to mean 0 and
variance 1, then apply a learned per-channel scale and shift — $2d$ parameters.)

- *Post-norm* (the 2017 paper): $x \leftarrow \operatorname{LN}(x + F(x))$. The LN sits
  *on* the residual path — every gradient, even the identity term, must pass through
  it, and its rescaling compounds over depth. Post-norm transformers famously need
  learning-rate warmup to survive their first steps.
- *Pre-norm* (GPT-2 and nearly everything since): $x \leftarrow x + F(\operatorname{LN}(x))$.
  The LN sits on the *branch*; the identity path is untouched from the embedding to the
  final LayerNorm. Deep models train stably without ceremony.

You will make this concrete in exercise E5 by training both variants and watching the
per-layer gradient norms.

## 4. The MLP block, and the block assembled

Attention moves information *between* positions; it is a weighted sum, hence linear in
the values. Something has to do the nonlinear *processing* at each position — that is
the **MLP block** (also called the feed-forward network): two linear layers with a
nonlinearity, applied to each position independently,

$$\operatorname{MLP}(x) = W_2\, \phi(W_1 x + b_1) + b_2,$$

with $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$: expand
to $4d$, apply $\phi$, project back. The factor 4 is convention, inherited from the
original paper and rarely questioned; it puts roughly two thirds of a block's
parameters in the MLP. The nonlinearity $\phi$ is **GELU** — a smooth ReLU variant,
$\phi(u) = u\,\Phi(u)$ where $\Phi$ is the Gaussian CDF (Week 08); think "ReLU with
rounded corner", which avoids the dead-neuron problem from Week 16.

One **decoder block** is those two sub-layers wired into the residual stream, pre-norm:

```python
import torch
import torch.nn as nn
from attention import MultiHeadAttention   # your Week 25 module

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)   # causal inside
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Division of labor, worth memorizing: **attention routes, the MLP computes.** Attention
decides which positions' information to combine; the MLP transforms the combined vector
at each position, same weights everywhere.

## 5. Parameter counting: the full derivation

This is the week's first flagship derivation: given the config, count every parameter
by hand, then reconcile with the famous "124M". The skill transfers — Week 30's LoRA
sizing and Week 42's memory budgeting are this arithmetic again. Rule: a
`Linear(n_in, n_out)` has $n_{\text{in}} \cdot n_{\text{out}}$ weights $+\ n_{\text{out}}$
biases; a LayerNorm has $2d$; an embedding table has (rows × width).

**One block, symbolically** (GPT-2 uses biases everywhere, so we count them):

| piece | weights | biases |
|---|---|---|
| LN1 | $d$ (scales) | $d$ (shifts) |
| attention: fused QKV `Linear(d, 3d)` | $3d^2$ | $3d$ |
| attention: output proj `Linear(d, d)` | $d^2$ | $d$ |
| LN2 | $d$ | $d$ |
| MLP up `Linear(d, 4d)` | $4d^2$ | $4d$ |
| MLP down `Linear(4d, d)` | $4d^2$ | $d$ |

$$\text{params/block} = 12d^2 + 13d.$$

Sanity-check the structure of that answer: $4d^2$ from attention (Week 25's count),
$8d^2$ from the MLP — the MLP is twice the attention. The $13d$ of biases and LN
parameters is a 0.1%-level correction at $d = 768$, but "matches exactly" is the
acceptance criterion, so it stays.

**GPT-2 small, numerically** ($d = 768$, $L = 12$, vocab $= 50257$, ctx $= 1024$):

- Per block: $12 \cdot 768^2 + 13 \cdot 768 = 7{,}077{,}888 + 9{,}984 = 7{,}087{,}872$.
- All blocks: $12 \times 7{,}087{,}872 = 85{,}054{,}464$.
- Token embeddings: $50257 \times 768 = 38{,}597{,}376$.
- Position embeddings: $1024 \times 768 = 786{,}432$.
- Final LayerNorm: $2 \times 768 = 1{,}536$.
- Output head `Linear(768, 50257)`, no bias: $38{,}597{,}376$ — *or zero*, see below.

**Weight tying.** The output head maps a $d$-vector to vocab logits — a
$\text{vocab} \times d$ matrix, exactly the shape of the token embedding table. GPT-2
**ties** them: the same matrix $W_E$ embeds tokens on the way in and produces logits on
the way out ($\text{logits} = x W_E^\top$). Rationale: both matrices relate "token
identity" to "direction in $d$-space", and sharing saves 38.6M parameters — 31% of this
model. So:

$$\text{total (tied)} = 85{,}054{,}464 + 38{,}597{,}376 + 786{,}432 + 1{,}536
= \mathbf{124{,}439{,}808} \approx 124\text{M}.$$

Untied it would be $163{,}037{,}184$. The quoted "124M" checks out exactly — and in
exercise E2 you will demand the same exactness from your own implementation:
`sum(p.numel() for p in model.parameters())` must equal your hand count to the digit. A
mismatch of $9{,}984$ means you forgot the biases-and-LN term; a mismatch of
$38{,}597{,}376$ means you double-counted the tied head. The error *tells you* what you
forgot — that is why we count exactly.

## 6. Shape-flow: one batch, every shape

Second flagship derivation: trace a batch through the whole model and write the shape
after every operation. Do it on paper for GPT-2 small with batch $B = 8$, $T = 64$.
Every debugging session you will ever have with transformers is this table.

| step | operation | shape | why |
|---|---|---|---|
| 0 | input token ids | $(8, 64)$ | integers in $[0, 50257)$ |
| 1 | token embedding lookup | $(8, 64, 768)$ | each id → its $W_E$ row |
| 2 | + position embedding | $(8, 64, 768)$ | $W_P[0{:}64]$ is $(64, 768)$, broadcast over batch (Week 03) |
| 3 | LN1 | $(8, 64, 768)$ | normalizes the last axis, shape-preserving |
| 4 | fused QKV projection | $(8, 64, 2304)$ | Linear $768 \to 3 \cdot 768$ |
| 5 | split + reshape to heads | 3 × $(8, 12, 64, 64)$ | $2304 \to 3 \times 12 \text{ heads} \times d_k{=}64$; head axis moved before T |
| 6 | scores $QK^\top/\sqrt{64}$ | $(8, 12, 64, 64)$ | last two axes: (query pos, key pos) — this one is $T{\times}T$, not heads! |
| 7 | causal mask + softmax | $(8, 12, 64, 64)$ | rows sum to 1, upper triangle 0 |
| 8 | weights @ V | $(8, 12, 64, 64)$ | back to (…, T, $d_k$) |
| 9 | merge heads + output proj | $(8, 64, 768)$ | concat 12 × 64 → 768, Linear $768 \to 768$ |
| 10 | residual add | $(8, 64, 768)$ | same shape by construction — that is the point |
| 11 | LN2 → MLP up → GELU | $(8, 64, 3072)$ | $4d = 3072$ |
| 12 | MLP down + residual add | $(8, 64, 768)$ | stream width restored |
| 13 | × 12 blocks, final LN | $(8, 64, 768)$ | blocks are shape-preserving maps of the stream |
| 14 | head: $x W_E^\top$ | $(8, 64, 50257)$ | logits: one score per vocab entry per position |
| 15 | cross-entropy vs targets | scalar | targets = inputs shifted left by one, shape $(8, 64)$ |

Three things to internalize. (1) The residual stream's shape $(B, T, d)$ never changes
from step 2 to step 13 — every block is a shape-preserving update, which is what lets
you stack any number of them. (2) In step 6, the *coincidence* that $T = d_k = 64$ here
makes four axes the same size — deliberately chosen so you learn to track axes by
*meaning*, not by size; relabel with $T = 100$ to check yourself. (3) Steps 14–15: the
model predicts at *every* position simultaneously, each position seeing only its past
(the mask), so one forward pass yields $B \times T$ training examples.

## 7. Reading the founding papers critically

You now know enough to read the primary sources as a peer, not a supplicant. Questions
to hold while reading:

*Attention Is All You Need* (Vaswani et al., 2017). What was essential: attention
without recurrence, multi-head, residual + norm structure, the parallel-training
argument. What was incidental: the encoder–decoder split (you built neither tower's
cross-attention and lost nothing for language modeling), sinusoidal encodings (learned
and rotary both work), post-norm (replaced by pre-norm), the specific optimizer recipe.
Note how thin the justification for $1/\sqrt{d_k}$ is — one footnote stating your
Week 25 variance argument. Papers compress; derivations live in readers.

*Language Models are Unsupervised Multitask Learners* (GPT-2, Radford et al., 2019).
Read §§1–2 and the model description. The architecture is your model — the paper's
contribution is a *claim*: train a big enough decoder on enough diverse text with the
dumbest possible objective, and task ability emerges without task-specific training.
Check what evidence the paper actually offers (zero-shot benchmark tables), what it
does not (ablations, error bars), and notice the model-size ladder (124M → 1.5B) —
that ladder becomes Week 28's scaling laws.

## 8. Worked example: the full GPT, assembled and sanity-checked

Complete and runnable (with `attention.py` from Week 25 in the same folder; install
with `uv add torch`):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

# Block as defined in Section 4 (paste that class here or import your own).

class GPT(nn.Module):
    def __init__(self, n_layer, n_head, d_model, vocab_size, ctx):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight      # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                         # (B, T, vocab)

torch.manual_seed(0)
model = GPT(n_layer=4, n_head=4, d_model=128, vocab_size=1000, ctx=64)

n_params = sum(p.numel() for p in model.parameters())
print("params:", n_params)          # check against your hand count!

idx = torch.randint(0, 1000, (8, 32))
logits = model(idx)
print("logits:", logits.shape)      # (8, 32, 1000)

targets = torch.randint(0, 1000, (8, 32))
loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1))
print("loss:", loss.item(), " ln(vocab):", math.log(1000))
```

The last line is Week 16's init check doing real work: an untrained model should assign
roughly uniform probability $1/\text{vocab}$ to every token, so cross-entropy should
start near $\ln(1000) \approx 6.91$. If your printed loss is near 6.9, embeddings,
blocks, tying, and loss plumbing are all at least *plugged in* correctly. If it is 15,
something (usually an un-normalized residual stream or a bad init) is broken before you
have trained a single step. Also hand-count this toy config
($12d^2 + 13d$ per block with $d = 128$, 4 blocks, embeddings $1000 \times 128$ tied +
$64 \times 128$, final LN) and check it against the printed number — that is exercise
E2's skill in miniature.

The decisive test, though, is exercise E3: load the real GPT-2 weights from HuggingFace
into *your* classes and match its logits to 1e-4. Passing it proves your architecture
is not "GPT-like" — it *is* GPT-2. This exact model is what you train in Week 27.

## Check yourself

1. Why does a transformer need position information at all, when an RNN did not?
2. State the sinusoidal shift property in one equation and name the trig fact it
   follows from.
3. Write the pre-norm and post-norm block equations. Which leaves an identity gradient
   path from loss to embeddings, and why does that matter at depth 48?
4. From memory: parameters in one decoder block as a function of $d$ (weights and
   biases), and which sub-layer owns the biggest share.
5. Your GPT-2-small count comes out 38,597,376 high. What did you forget?
6. In the shape-flow table, which single step produces a tensor whose last two axes are
   *not* (positions, channels), and what are they instead?
7. Why is loss ≈ ln(vocab) at init the right check, and what loss would a model with
   vocab 50257 show?
8. Attention routes, the MLP computes: which sub-layer moves information between
   positions, and which is applied to every position independently?

## Answers

1. Self-attention is permutation-equivariant — reorder the tokens and outputs reorder
   identically — so word order is invisible unless injected. An RNN processes tokens
   one at a time, so order is built into its computation.
2. $PE(p{+}k) = M_k\,PE(p)$ with $M_k$ independent of $p$; it follows from the angle
   addition formulas $\sin(a{+}b) = \sin a\cos b + \cos a \sin b$ (and its cosine
   twin), which make each frequency pair transform by a fixed rotation.
3. Pre-norm: $x + F(\operatorname{LN}(x))$. Post-norm: $\operatorname{LN}(x + F(x))$.
   Pre-norm keeps the identity path clean; post-norm inserts an LN on it. At depth 48
   the product of 48 LN Jacobians on the highway distorts gradient scale enough to
   require warmup or fail outright.
4. $12d^2$ weights $+ 13d$ (biases + LN): $3d^2 + d^2$ attention, $4d^2 + 4d^2$ MLP.
   The MLP owns $8d^2/12d^2 = 2/3$ of the block.
5. You counted the output head as a separate matrix — it is tied to the token
   embedding, so those $50257 \times 768$ parameters exist once, not twice.
6. Step 6–8, the attention scores/weights: last two axes are (query position, key
   position) — a $T \times T$ object per head, not positions × channels.
7. An untrained model should be maximally ignorant: uniform $1/\text{vocab}$ predictions
   give cross-entropy $\ln(\text{vocab})$. Higher means broken plumbing or bad init
   (confidently wrong); much lower means leakage. $\ln(50257) \approx 10.82$.
8. Attention moves information between positions (the weighted sum over other tokens'
   values); the MLP computes per-position with shared weights.

## New terms

- **decoder-only transformer** — GPT architecture: causal-attention blocks only, trained
  on next-token prediction; no encoder, no cross-attention.
- **context length (ctx)** — maximum sequence length; sets the position-table size.
- **token / position embedding** — learned tables mapping token identity / position
  index to $d$-vectors, summed at the input.
- **permutation-equivariant** — reorder inputs ⇒ outputs reorder identically; why
  attention alone cannot see word order.
- **sinusoidal encoding** — fixed sin/cos position vectors with $PE(p{+}k) = M_k PE(p)$.
- **RoPE** — rotate Q and K by position-proportional angles so scores depend only on
  relative position.
- **residual stream** — the width-$d$ vector per position that every block reads from
  and additively writes to.
- **pre-norm / post-norm** — LayerNorm on the branch (identity path clean) vs on the
  residual path (needs warmup).
- **MLP block / feed-forward** — per-position `Linear(d,4d) → GELU → Linear(4d,d)`;
  two thirds of a block's parameters.
- **GELU** — smooth ReLU variant, $u\,\Phi(u)$.
- **weight tying** — one matrix serves as both token embedding and output head.
- **logits** — pre-softmax scores over the vocabulary, shape $(B, T, \text{vocab})$.

## Going deeper

- Karpathy, *Let's build GPT* — second half: blocks, LayerNorm, residuals, the full
  model. Watching him assemble what you just assembled is the fastest error-check
  available.
- Vaswani et al., *Attention Is All You Need* (arXiv 1706.03762) — full read this week,
  with Section 7's critical questions in hand.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) —
  §§1–3 and model description; the config table is your E2/E3 ground truth.
- Prince, *Understanding Deep Learning*, Ch. 12, remaining sections — position
  encodings and the full architecture, more formally.
- Su et al., *RoFormer* (arXiv 2104.09864), §3.2 only — the rotary embedding
  construction; the phase-rotation idea reads naturally after Section 2.
