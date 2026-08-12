# Week 19 — Sequences: RNNs, LSTMs, and Why Attention Won

~5 hrs, including the paper derivation. Before starting you should be able to: derive
backprop through a small network by hand, including the multivariate chain rule
(Week 13); state what a matrix's singular values say about how it stretches vectors
(Week 07); train a character-level model with an embedding table in PyTorch (Week 15);
explain why a residual block's gradient contains an identity term (Week 17).

## 1. Data that comes in order

Everything so far ate fixed-size inputs: a feature vector, a 28×28 image. But much of
the world arrives as a **sequence** — an ordered list of items whose length varies and
whose order carries the meaning. Text is a sequence of characters. An audio clip is a
sequence of samples. A detector readout is a sequence too: many particle detectors
record, for each channel, a waveform — the signal voltage sampled every few
nanoseconds — and the *shape over time* distinguishes a real pulse from noise.

Fixed-size models handle sequences badly for two reasons. An MLP needs a fixed input
length, so you must pad or truncate, and — Week 17's argument transposed — it has no
built-in notion that position 7 and position 8 are adjacent, or that "the same pattern
later" is the same pattern. Week 15's character model dodged this by using a fixed
context window (the previous 3 characters), which works until the relevant information
sits 50 characters back.

What we want is a model that reads a sequence one step at a time, of any length,
applying the *same* rule at every step — weight sharing across time, exactly as a
convolution shares weights across space — while carrying along a memory of what it has
seen.

## 2. The recurrent neural network

A **recurrent neural network (RNN)** maintains a **hidden state** $h_t$, a vector that
acts as the network's memory, and updates it at each step from the new input and the
previous state:

$$h_t = \tanh\!\bigl(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b\bigr).$$

Symbols: $x_t$ is the input at step $t$ (say, one character's embedding vector);
$h_{t-1}$ is the previous hidden state ($h_0$ is zeros); $W_{xh}$ maps input to
hidden, $W_{hh}$ maps previous hidden to hidden (the **recurrent weights**); $b$ is a
bias; $\tanh$ squashes elementwise to $(-1, 1)$. When an output is needed at step $t$
(e.g. "probabilities for the next character"), a readout layer produces
$\hat{y}_t = W_{hy} h_t + b_y$, followed by softmax.

The crucial property: the *same three weight matrices* are used at every step. To see
what that buys, **unroll** the recurrence — draw the computation for a 4-step sequence
as a chain: $h_0 \to h_1 \to h_2 \to h_3 \to h_4$, with $x_1 \ldots x_4$ feeding in
from below and every arrow labeled with the same $W_{hh}$. Unrolled, an RNN is just a
deep feedforward network — one layer per time step — with **tied weights**: depth
equals sequence length, and a 100-character sequence makes a 100-layer network.

That reframing is the whole story of this week. Everything you learned about deep
networks — Week 16's signal propagation, Week 17's gradient products — applies to
RNNs with "layer" replaced by "time step". Including the failure mode.

## 3. Backprop through time

Training data for a character-level RNN: at each step, predict the next character;
the loss is the sum (or mean) of the per-step cross-entropies,
$L = \sum_{t=1}^{T} L_t$. Computing gradients through the unrolled network is called
**backprop through time (BPTT)**. It is ordinary backprop — Week 13, nothing new is
being invented — but the tied weights make the bookkeeping worth doing slowly once.

Set $a_t = W_{xh} x_t + W_{hh} h_{t-1} + b$, so $h_t = \tanh(a_t)$. We want
$\partial L / \partial W_{hh}$. The subtlety: $W_{hh}$ is used at *every* step, so
(Week 14's accumulation rule — a value used in several places gets the *sum* of the
gradients from each use) its total gradient is a sum of contributions, one per use.

**Step 1 — how one loss term reaches one use of $W_{hh}$.** The use of $W_{hh}$ at
step $k$ affects $h_k$ directly; $h_k$ affects $L_t$ (for $t \ge k$) only through the
chain $h_k \to h_{k+1} \to \cdots \to h_t$. Chain rule:

$$\frac{\partial L_t}{\partial W_{hh}}\bigg|_{\text{use at }k}
= \frac{\partial L_t}{\partial h_t}\;
\underbrace{\frac{\partial h_t}{\partial h_{t-1}}
\frac{\partial h_{t-1}}{\partial h_{t-2}} \cdots
\frac{\partial h_{k+1}}{\partial h_{k}}}_{\text{the product of Jacobians}}\;
\frac{\partial h_k}{\partial W_{hh}}\bigg|_{\text{this use}}.$$

**Step 2 — sum over uses and loss terms.**

$$\frac{\partial L}{\partial W_{hh}}
= \sum_{t=1}^{T} \sum_{k=1}^{t}
\frac{\partial L_t}{\partial h_t}
\left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right)
\frac{\partial h_k}{\partial W_{hh}}\bigg|_{\text{use }k}.$$

**Step 3 — what each Jacobian is.** Differentiate $h_j = \tanh(a_j)$ with respect to
$h_{j-1}$. Only the $W_{hh} h_{j-1}$ term of $a_j$ depends on $h_{j-1}$, and tanh acts
elementwise, so

$$\frac{\partial h_j}{\partial h_{j-1}}
= \mathrm{diag}\!\bigl(1 - h_j^2\bigr)\; W_{hh},$$

using $\tanh'(a) = 1 - \tanh^2(a)$ (derive it once from
$\tanh = (e^a - e^{-a})/(e^a + e^{-a})$ if you never have). Here
$\mathrm{diag}(1-h_j^2)$ is the diagonal matrix of elementwise tanh slopes at step $j$.

On paper this week you unroll $T = 3$ explicitly and write out all the terms of
$\partial L_3/\partial W_{hh}$ — three paths, carrying products of zero, one, and two
Jacobians. Do it component-free (matrices) or index-by-index, but do it; the exercise
notebook then checks your formula against finite differences.

The per-step structure gives the practical algorithm: run forward saving all $h_t$,
then sweep backward accumulating $\delta_t = \partial L/\partial h_t$ — each step
receives the local loss gradient plus $\delta_{t+1}$ pushed back through one Jacobian.
Cost: $O(T)$ memory for the saved states. For very long sequences one truncates
(**truncated BPTT**): backprop only some window of steps — cheaper, but gradients
beyond the window are simply *zero*, so nothing further back can be credited at all.

## 4. Vanishing and exploding gradients through time

Look at the product in Step 2. A loss at step $t$ teaches the network about an input
at step $k$ only through

$$\prod_{j=k+1}^{t} \mathrm{diag}(1 - h_j^2)\, W_{hh},$$

a product of $t - k$ Jacobians — the same object as Week 17's plain-network gradient,
with time lag playing the role of depth, plus one aggravation: here it is the *same*
matrix $W_{hh}$ multiplied in at every step.

**The clean case first.** Drop the nonlinearity (pretend the slopes are all 1). The
product is $W_{hh}^{\,t-k}$. Week 07 tells you exactly how repeated multiplication by
one matrix behaves: let $\sigma_{\max}$ be the largest singular value of $W_{hh}$ (the
biggest stretch factor it applies to any vector). Then
$\lVert W_{hh}^{\,n} v \rVert \le \sigma_{\max}^{\,n} \lVert v \rVert$, and for a
generic vector the growth/decay is governed by $\sigma_{\max}^n$:

- $\sigma_{\max} < 1$: the gradient **vanishes** — shrinks geometrically with lag.
  At $\sigma_{\max} = 0.9$ and lag 50, the factor is $0.9^{50} \approx 0.005$.
- $\sigma_{\max} > 1$: the gradient **explodes**. At $1.1^{50} \approx 117$, a few
  more steps and the update is NaN.

**Now restore tanh.** Each factor gains $\mathrm{diag}(1 - h_j^2)$, whose entries lie
in $(0, 1]$ — and are near 0 whenever a unit saturates (Week 16's story). So the norm
bound becomes

$$\left\lVert \prod_{j=k+1}^{t} \mathrm{diag}(1-h_j^2)\, W_{hh} \right\rVert
\;\le\; \bigl(\gamma\, \sigma_{\max}\bigr)^{\,t-k}, \qquad
\gamma = \max_j \bigl\lVert \mathrm{diag}(1-h_j^2) \bigr\rVert \le 1.$$

The nonlinearity can only make shrinking worse; it never rescues a too-small
$\sigma_{\max}$. Explosion is still possible when $\sigma_{\max}$ is large enough to
overcome $\gamma$.

**What this means for learning.** The gradient *is* the credit assignment: it is how
"the loss at step 200 was high" gets traced back to "because of the input at step 3".
If the product has vanished, that credit never arrives — the model literally cannot
learn dependencies across long lags, no matter how long you train. This is the
**long-range credit assignment problem**, and it is a structural property of the
recurrence, not a tuning issue.

The two failure modes get very different remedies. Exploding gradients have a blunt,
effective fix you met in Week 16: **gradient clipping** — if the gradient norm exceeds
a threshold, rescale it down. Clipping saves the training run from the occasional
cliff. But note what clipping cannot do: it can shrink big gradients; it cannot
amplify vanished ones. Vanishing needs an architectural fix.

## 5. The LSTM: an additive path through time

Week 17 fixed vanishing gradients through *depth* by adding an identity path:
$y = x + F(x)$ puts an $I$ inside every Jacobian factor. The **LSTM** (Long
Short-Term Memory, 1997 — it predates ResNet by 18 years) is the same trick through
*time*, plus learned valves controlling the path.

The LSTM carries *two* state vectors: the hidden state $h_t$ (also the output, as
before) and a new **cell state** $c_t$ — the protected memory. Three **gates**, each a
sigmoid-activated vector in $(0,1)^d$ computed from the current input and previous
hidden state, control traffic ($\sigma$ = sigmoid, $\odot$ = elementwise product;
each gate has its own $W, U, b$):

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \qquad \text{forget gate}$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \qquad \text{input gate}$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \qquad \text{output gate}$$
$$g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g) \qquad \text{candidate memory}$$

and the two updates:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad
h_t = o_t \odot \tanh(c_t).$$

Read the cell update in words: keep a per-coordinate fraction $f_t$ of the old memory,
and add in a gated amount of new content. The forget gate decides what to erase, the
input gate what to write, the output gate what to reveal to the rest of the network.

**Why this fixes vanishing** — the derivation to do on paper. Differentiate the cell
update along the cell path, holding the gate values fixed (they also depend on
$h_{t-1}$, but the cell-to-cell path is the dominant memory channel and the one worth
isolating):

$$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t).$$

Compare the vanilla RNN's $\mathrm{diag}(1-h^2)\,W_{hh}$. No weight matrix in the
loop; no repeated multiplication by the same $W$. The gradient across a lag of $n$
steps along the cell path is $\prod \mathrm{diag}(f_j)$ — and the network *learns*
$f$. Where a memory must persist, the forget gate saturates near 1 and the gradient
passes essentially unattenuated, for hundreds of steps; where memory should be
dropped, $f \to 0$ *by choice*, not by decay. Memory is **additive** ($c_t$ is
old-plus-new, like a residual block's $x + F(x)$), not **multiplicative**
(squashed-transform-of-old, like $\tanh(W h)$). One practical echo of the analysis:
initialize the forget-gate bias $b_f$ positive (≈1–2), so gates start open and the
network begins life able to remember.

The LSTM is the "big" gated recurrent cell; the GRU is a popular lighter variant with
two gates and no separate cell state — same additive idea. In PyTorch, `nn.LSTM`
runs the whole recurrence over a batch of sequences in one call (§7).

## 6. Attention, as intuition

LSTMs stretch usable memory from tens of steps to hundreds. But look at what any
recurrent model still does: it *compresses the entire past into one fixed-size
vector*. By step 500, everything the model may ever need about steps 1–499 must
already be packed into $h_{500}$ (and $c_{500}$) — a bottleneck that no gating
cleverness removes. And information still travels step-by-step: the path from step 3
to step 500 is 497 applications of the update rule, each a chance to lose a little.

**Attention** abandons the compression. Keep *every* past step's representation
around. When producing step $t$'s output, look back over all of them and take a
weighted average, with weights the model computes based on *content* — how relevant
each past step is to what step $t$ is trying to do. Sketch, in the language Week 25
will make precise: step $t$ emits a **query** vector ("what am I looking for?"); each
past step $j$ offers a **key** vector ("what am I about?") and a **value** vector
("what I contribute if chosen"). Scores are dot products (query · key); a softmax —
the same softmax you have used since Week 10 — turns scores into weights summing to 1;
the output is the weight-averaged values.

Two consequences, worth stating now because they explain the next three months of
this course:

- **Path length 1.** Step 500 reaches step 3 directly — one weighted lookup, not 497
  hops. The gradient flows back along that same direct edge. Long-range credit
  assignment stops being a fight against a product of 497 Jacobians.
- **No bottleneck.** Nothing must be prematurely packed into one vector; the model
  retrieves what it needs when it needs it.

The price: every step attends to every other, so cost grows with the *square* of
sequence length, and the sequential nature of RNN computation is replaced by big
parallel matrix products — which is a price GPUs love to pay. The transformer
(Week 25 onward) is what you get when you keep attention and drop the recurrence
entirely. This week's deliverable includes a one-paragraph "why attention" in your own
words; Week 25 you will grade it against the real derivation.

## 7. Worked example: a character-level LSTM

The task from Week 15, upgraded: read a name one character at a time and predict the
next character; sample from the trained model to generate new names. Use the same
`names.txt` you used in Week 15 (the makemore names file — ~32k names, one per line;
if you no longer have it, search "karpathy makemore names.txt" on GitHub). Where the
Week 15 MLP saw a fixed 3-character window, the LSTM carries state over the whole
name.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}   # 0 is reserved for "." start/end
itos = {i: s for s, i in stoi.items()}
itos[0] = "."
V = len(itos)                                    # vocabulary size (27)

# Build one padded tensor of inputs and targets: "." + name predicts name + "."
MAXLEN = max(len(w) for w in words) + 1
X = torch.zeros(len(words), MAXLEN, dtype=torch.long)
Y = torch.full((len(words), MAXLEN), -1, dtype=torch.long)   # -1 = ignore (padding)
for i, w in enumerate(words):
    seq = [0] + [stoi[c] for c in w] + [0]
    for t in range(len(seq) - 1):
        X[i, t] = seq[t]
        Y[i, t] = seq[t + 1]

n_train = int(0.9 * len(words))
perm = torch.randperm(len(words))
train_idx, val_idx = perm[:n_train], perm[n_train:]

class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 24)           # Week 15's embedding table
        self.lstm = nn.LSTM(24, 128, batch_first=True)
        self.head = nn.Linear(128, V)

    def forward(self, x):
        e = self.emb(x)                          # (batch, T, 24)
        out, _ = self.lstm(e)                    # (batch, T, 128), runs the recurrence
        return self.head(out)                    # (batch, T, V) logits per step

model = CharLSTM()
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)   # padding steps contribute no loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(2001):
    ib = train_idx[torch.randint(len(train_idx), (64,))]
    logits = model(X[ib])
    loss = loss_fn(logits.reshape(-1, V), Y[ib].reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Week 16's clipping
    opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            vlogits = model(X[val_idx])
            vloss = loss_fn(vlogits.reshape(-1, V), Y[val_idx].reshape(-1))
        print(step, "train", round(loss.item(), 3), "val", round(vloss.item(), 3))

# Sample: feed one character, take the model's distribution, draw, repeat.
for _ in range(10):
    ctx = torch.zeros(1, 1, dtype=torch.long)    # start token "."
    h = None
    out = ""
    for _ in range(MAXLEN):
        e = model.emb(ctx)
        o, h = model.lstm(e, h)                  # pass state forward step by step
        probs = torch.softmax(model.head(o[0, -1]), dim=0)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == 0:
            break
        out = out + itos[nxt]
        ctx = torch.tensor([[nxt]])
    print(out)
```

A few minutes on CPU gets validation loss near 2.0 nats — comfortably below the
Week 15 fixed-window MLP on the same split — and the samples look like plausible
names. Note the three places this week's theory shows up in the code: `nn.LSTM` hides
the §5 cell equations (open its documentation next to the equations once); the
per-step logits mean the loss is the §3 sum over time; and the clipping line is §4's
insurance against the exploding half of the gradient problem.

## Check yourself

1. In what precise sense is an unrolled RNN "a 100-layer network"? What is shared
   that an ordinary 100-layer network does not share?
2. Why is $\partial L/\partial W_{hh}$ a double sum? What does each summation index
   run over?
3. Write the vanilla-RNN Jacobian $\partial h_j/\partial h_{j-1}$ and name both
   factors.
4. Gradient at lag 40 with $\sigma_{\max}(W_{hh}) = 0.95$ and slopes ≈ 1: roughly how
   big relative to lag 0? What if $\sigma_{\max} = 1.05$?
5. Gradient clipping fixes which of the two gradient pathologies, and why can't it
   fix the other?
6. Write the LSTM cell update and give $\partial c_t/\partial c_{t-1}$ (gates held
   fixed). What plays the role that the identity map plays in a ResNet block?
7. Why does initializing the forget-gate bias positive help early training?
8. Give the two structural reasons attention beats recurrence for long-range
   dependencies, and the price attention pays.

## Answers

1. Unrolling one update per time step gives a 100-step-deep computation graph. All
   100 "layers" use the same $W_{xh}, W_{hh}, b$ — tied weights — unlike a feedforward
   net's independent per-layer weights.
2. Outer sum over loss terms $L_t$ (one per step); inner sum over the uses $k \le t$
   of $W_{hh}$, because the same matrix is applied at every step and each use
   contributes gradient (Week 14's accumulation rule).
3. $\partial h_j/\partial h_{j-1} = \mathrm{diag}(1 - h_j^2)\, W_{hh}$: the diagonal
   matrix of tanh slopes at step $j$, times the recurrent weight matrix.
4. About $0.95^{40} \approx 0.13$ — an order of magnitude down, and worse with
   realistic tanh slopes. With $1.05$: $1.05^{40} \approx 7$, and growing
   geometrically — heading toward explosion.
5. Exploding only. Clipping rescales oversized gradients down; a vanished gradient is
   a *direction-and-magnitude signal that never arrived* — there is nothing left to
   rescale up.
6. $c_t = f_t \odot c_{t-1} + i_t \odot g_t$;
   $\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)$. The forget-gated additive
   cell path plays the identity map's role: with $f \approx 1$ it is the ResNet skip,
   but learnable.
7. Gates start open ($f \approx \sigma(1..2) \approx 0.73..0.88$), so gradients flow
   along the cell path from step one; with $b_f = 0$ half the memory decays per step
   ($f \approx 0.5$) before the network has learned what to keep.
8. Path length 1 between any two positions (no product of Jacobians over the lag) and
   no fixed-size bottleneck (the past is retrieved, not compressed). Price: compute
   and memory grow quadratically with sequence length.

## New terms

- **sequence** — variable-length ordered data; order carries meaning.
- **recurrent neural network (RNN)** — applies one shared update rule per step while
  carrying a hidden state.
- **hidden state** $h_t$ — the fixed-size vector serving as the network's memory.
- **recurrent weights** $W_{hh}$ — the matrix mapping previous state to current;
  reused every step.
- **unrolling** — drawing the recurrence as a feedforward graph, one layer per step.
- **tied weights** — the same parameters reused at many places in a graph (here:
  every time step).
- **backprop through time (BPTT)** — ordinary backprop on the unrolled graph, with
  gradients summed over each weight's many uses.
- **truncated BPTT** — backpropagating only a window of steps; beyond it, credit is
  exactly zero.
- **vanishing / exploding gradients** — geometric decay/growth of the
  product-of-Jacobians with time lag, governed by $\sigma_{\max}(W_{hh})$.
- **long-range credit assignment** — connecting a loss to a cause many steps earlier;
  what the vanishing product destroys.
- **LSTM** — recurrent cell with a gated, additive cell-state path that preserves
  gradients across long lags.
- **cell state** $c_t$ — the LSTM's protected additive memory.
- **gate (forget/input/output)** — sigmoid-valued vectors in $(0,1)$ multiplying a
  signal elementwise; learned valves.
- **GRU** — a lighter gated cell: two gates, no separate cell state.
- **attention** — content-based weighted lookup over all past positions; queries,
  keys, values (derived properly in Week 25).

## Going deeper

- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (blog) —
  what char-RNNs learn in the wild; read after the worked example and compare samples.
- Christopher Olah, "Understanding LSTM Networks" (colah's blog) — the canonical
  gate-by-gate pictures; the diagrams to hold in your head for §5.
- Goodfellow, Bengio & Courville, *Deep Learning* (free online), Ch. 10 — the formal
  BPTT treatment and the long-term-dependency sections; checks your §3 derivation.
- 3Blue1Brown, the attention chapter of the neural-network video series — pictures
  for §6 only; watch for intuition now, Week 25 does the math.
