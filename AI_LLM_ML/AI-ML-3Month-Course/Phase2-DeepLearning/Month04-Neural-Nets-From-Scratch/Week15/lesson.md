# Week 15 — PyTorch Fundamentals

~6 hrs, mostly at the keyboard. Before starting you should be able to: explain
what a computational graph is and what `backward()` does to it, and read a plain
Python class (Week 14); use NumPy arrays, broadcasting, and boolean masks
(Week 03); derive the softmax/cross-entropy gradient (Week 13); state why we
split data into train and validation sets (Week 09).

You have now built an autograd engine. **PyTorch** is that engine, industrialized:
the nodes of the graph are whole arrays instead of single scalars, every local
backward rule is compiled code, and the same program runs on a GPU. Nothing
conceptually new happens this week — the goal is to map each thing you built onto
the API you will use for the rest of the course, then use it for something your
scalar engine could never afford: a character-level language model.

Install with `uv add torch`. Everything this week runs fine on a CPU.

## 1. Tensors

A **tensor** is PyTorch's array: an n-dimensional grid of numbers, exactly like a
NumPy array, plus two superpowers — it can live on a GPU, and it can remember the
graph that produced it. The NumPy skills from Week 03 transfer almost verbatim:

```python
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from data
z = torch.zeros(3, 5)                         # shapes, not a list of them
r = torch.randn(3, 5)                         # standard normal entries
print(a.shape, a.dtype)                       # torch.Size([2, 2]) torch.float32

b = a + a          # elementwise, like NumPy
c = a @ a          # matrix product, like NumPy
m = a[a > 2.0]     # boolean masks work
row = a.sum(dim=1) # NumPy's axis= is called dim=
```

Three differences worth flagging on day one:

- **dtype.** PyTorch defaults to `float32` (NumPy: `float64`). Deep learning
  runs in 32-bit or less; mixing dtypes raises errors, and `x.float()`,
  `x.long()` convert. Integer class labels live in `int64` tensors (`long`).
- **Broadcasting** follows the same rules you learned in Week 03: align shapes
  from the right, dimensions of size 1 stretch. `(32, 27) + (27,)` works;
  `(32, 27) + (32,)` does not — same rule, same reason.
- **device.** `x.to("cuda")` moves a tensor to the GPU; operations require all
  operands on the same device. Not needed this week, but the reason PyTorch
  exists.

Conversion both ways is cheap: `torch.from_numpy(arr)` and `t.numpy()` — and both
share memory with the original, which brings us to the first real trap.

**Views vs copies.** Like NumPy slices, many tensor operations return a **view**
— a new tensor object looking at the *same* memory. `x.view(2, 6)`,
`x.reshape(...)` (usually), slicing, and `.T` are views; writing through one
changes the other. `x.clone()` forces a real copy. If a result surprises you,
ask: did I copy, or am I looking at the same numbers twice?

**In-place operations.** Methods with a trailing underscore modify a tensor's
memory directly: `x.add_(1.0)`, `x.relu_()`, `x += 1`. They save memory — and
they are the enemy of autograd. Your Week 14 backward rules read *stored forward
values* (`self.data` in the tanh rule); overwrite a value in place after it was
used, and the stored number backward needs is gone. PyTorch detects this and
raises "a variable needed for gradient computation has been modified by an
inplace operation". Until you have a reason otherwise: no in-place ops on
anything that requires gradients.

## 2. Autograd: your engine, renamed

Every piece of your `Value` class has a PyTorch counterpart:

| your engine (Week 14) | PyTorch |
|---|---|
| `Value(2.0)` leaf | `torch.tensor(2.0, requires_grad=True)` |
| `.data` | the tensor's values (`.data` exists but hands-off) |
| `.grad`, starts at 0, accumulates via `+=` | `.grad`, starts at `None`, accumulates |
| `_parents`, `_op` | `grad_fn` (each tensor knows its producer) |
| `L.backward()` topo walk | `loss.backward()` |
| `p.grad = 0.0` before each step | `optimizer.zero_grad()` |
| not building the graph at all | `with torch.no_grad():` |
| cutting a value out of the graph | `x.detach()` |

Watch it reproduce Week 14's hand trace:

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(5.0, requires_grad=True)
L = (a * b + c).tanh()
L.backward()
print(L.item())                      # -0.7616  (.item(): 1-element tensor -> float)
print(a.grad, b.grad, c.grad)        # -1.2600, 0.8400, 0.4200
```

`requires_grad=True` marks a leaf as "track me": every tensor computed from it
records its parents (peek at `L.grad_fn` — a `TanhBackward` node, your `_op`
string grown up). `backward()` runs the reverse topological walk and deposits
gradients on the leaves.

Two behaviors you already understand better than most PyTorch users:

- **Gradients accumulate.** Call `backward()` twice and `.grad` doubles — it is
  the `+=` from your `_local_backward`, needed because a value used by many
  paths must *sum* its contributions. The framework cannot know when you are
  done with the old gradient, so *you* zero it between steps. Forgetting is the
  classic silent bug: training still runs, loss still falls (slowly, weirdly),
  nothing errors. Week 16 hands you exactly this broken run to diagnose.
- **Not everything should be tracked.** The parameter update
  `w = w - lr * w.grad` is itself arithmetic on `w` — track it and the graph
  grows across steps, leaking memory and corrupting gradients. Wrap
  update/evaluation code in `with torch.no_grad():` (a block whose operations
  are not recorded), or use `x.detach()` to get a copy of `x` severed from the
  graph (e.g. before `.numpy()` for plotting).

## 3. `nn.Module`: models as classes

You could build networks from raw tensors (and in E5's neural bigram you will).
But juggling parameter lists by hand does not scale, so PyTorch gives a base
class, `torch.nn.Module`, that any model inherits from. **Inheritance** is one
small addition to Week 14's classes: writing `class MLP(nn.Module)` makes `MLP` a
*child* of `nn.Module` — it gets all the parent's methods for free, and
`super().__init__()` on the first line runs the parent's `__init__` so that
machinery is set up before you add your own attributes. That is all the
inheritance this course needs.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)

model = MLP(2, 16, 2)
logits = model(X)          # calls forward(X) for you
```

What the machinery buys you: `nn.Linear(n_in, n_out)` is a ready-made affine
layer holding a weight matrix and bias (already sensibly initialized — Week 16
derives how); assigning modules to `self` inside `__init__` *registers* them, so
`model.parameters()` yields every weight and bias in the whole tree — exactly the
`parameters()` lists you concatenated by hand in Week 14, automated. You write
`forward`; calling `model(X)` invokes it (plus framework hooks). Note `forward`
returns **logits** — raw scores, no softmax. Hold that thought for §5.

## 4. `Dataset` and `DataLoader`

In Week 08 you saw that SGD updates on small random batches beat full-batch
descent. PyTorch splits that job in two: a `Dataset` is anything with a length
and integer indexing that returns one example; a `DataLoader` wraps it and serves
shuffled mini-batches, reshuffling every **epoch** (one full pass through the
data). For tensors already in memory:

```python
from torch.utils.data import TensorDataset, DataLoader

ds = TensorDataset(X_train, y_train)     # pairs X_train[i], y_train[i]
loader = DataLoader(ds, batch_size=32, shuffle=True)

for xb, yb in loader:                    # one epoch
    ...                                  # xb: (32, d), yb: (32,)
```

`shuffle=True` matters: serve the data in a fixed order and the model sees the
same gradient sequence every epoch — Week 08's argument for why SGD's noise
helps needs the noise to actually vary.

## 5. The loss and the optimizer

`torch.nn.functional.cross_entropy(logits, labels)` computes softmax and
cross-entropy **fused in one call** — you pass raw logits and *integer* class
labels (not one-hot vectors; PyTorch indexes the correct logit directly). Fused
is not just convenience: you found in Week 13 that softmax overflows without the
max-subtraction trick and that the combined gradient collapses to `p − y`;
doing both inside one function is how PyTorch gets the stable and cheap version.
Passing already-softmaxed probabilities into `cross_entropy` is a classic bug —
the loss silently computes softmax *twice* and trains badly.

The optimizer packages your Week 08 update rules over `model.parameters()`:

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1)
opt.step()        # p.data -= lr * p.grad, for every parameter
opt.zero_grad()   # p.grad = 0, for every parameter
```

Swap `SGD` for `torch.optim.Adam(...)` and nothing else changes — that is the
point of the abstraction.

Assembled, here is the five-line heartbeat you will type for the rest of your
life. It is Week 14's worked-example loop, line for line:

```python
for xb, yb in loader:
    logits = model(xb)                # forward
    loss = F.cross_entropy(logits, yb)
    opt.zero_grad()                   # zero the accumulators
    loss.backward()                   # backward through the graph
    opt.step()                        # gradient descent update
```

## 6. A character-level language model

Now something new: a model of *names*. The dataset is `names.txt` from
Karpathy's makemore repository (github.com/karpathy/makemore) — about 32,000
English first names, one per line. The task: learn the distribution of names
well enough to generate new plausible ones. This tiny problem is the exact shape
of GPT — predict the next token given context — with 27 characters standing in
for 50,000 word-pieces, and it seeds three weeks of Phase 3.

**Setup.** Alphabet plus a boundary marker `.` used for both "start of name" and
"end of name", and lookup tables both ways:

```python
words = open("names.txt").read().splitlines()
chars = ["."] + [c for c in "abcdefghijklmnopqrstuvwxyz"]
stoi = {}
for i in range(len(chars)):
    stoi[chars[i]] = i          # "." -> 0, "a" -> 1, ...
```

**The bigram model.** A **bigram** is an adjacent pair of characters. The
simplest possible model: predict each character from only the one before it —
i.e., learn $P(\text{next} \mid \text{current})$, a 27 × 27 table. Counting gives
it directly:

```python
N = torch.zeros(27, 27)
for w in words:
    chs = "." + w + "."
    for i in range(len(chs) - 1):
        N[stoi[chs[i]], stoi[chs[i + 1]]] += 1

P = (N + 1) / (N + 1).sum(dim=1, keepdim=True)   # rows are P(next | current)
```

The `+ 1` (**smoothing**) puts a tiny count in every cell so no pair has
probability exactly zero — one unseen pair in evaluation would otherwise make the
loss infinite. Row-normalizing turns counts into probabilities; `keepdim=True`
keeps the sum shaped `(27, 1)` so broadcasting divides each row by its own sum.

Generate names by starting at `.` and repeatedly sampling the next character
from the current character's row until `.` comes up again
(`torch.multinomial(p, num_samples=1)` draws an index with probabilities `p`).
The output is charmingly name-ish and mostly wrong — a bigram sees one character
of context.

**How good is it, in numbers?** The same score as always: average negative
log-likelihood (NLL) of the data, $-\frac{1}{n}\sum \log P(\text{pair})$ — which
is exactly cross-entropy. On paper this week you show the counting model is not
just *a* model but the *best possible* bigram model: minimizing cross-entropy
over all conditional tables gives exactly the empirical frequencies (MLE for a
categorical distribution — Week 08's derivation reused).

**The same model, neurally.** Re-cast counting as learning: input = current
character one-hot encoded (Week 13), model = a single linear layer `27 → 27`
producing logits, loss = cross-entropy, optimizer = gradient descent. This is
softmax regression on characters. Because the counting answer is the global
optimum of this exact loss, training must converge to the counting model's NLL —
a rare pleasure: a deep-learning pipeline with a known exact answer. E5 confirms
it to two decimals; if it does not converge there, the pipeline is broken, not
the model.

**Then make it deep.** The bigram's ceiling is its one-character context. The
fix (E6, and the seed of Phase 3): take the previous *three* characters, map
each through a shared **embedding table** — a learned `(27, d)` matrix whose
row $i$ is a $d$-dimensional vector representing character $i$; one-hot times a
matrix *is* row lookup, so this is the linear layer's first step made explicit —
concatenate the three vectors, and feed a tanh hidden layer, then 27 output
logits. Trained with the identical five-line loop, validation NLL drops clearly
below the bigram's. That gap is what context is worth.

## 7. Worked example

The full template, end to end, on Week 13's two-moons data — same task your
NumPy net and scalar engine solved, now in the form you will reuse all course.
Complete and runnable:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- data: two moons (Week 13), as tensors, with a real split (Week 09) ---
n = 500
angles = rng.uniform(0.0, np.pi, size=n)
X0 = np.stack([np.cos(angles), np.sin(angles)], axis=1)
X1 = np.stack([1.0 - np.cos(angles), 0.5 - np.sin(angles)], axis=1)
X = np.concatenate([X0, X1]) + rng.normal(0.0, 0.1, size=(2 * n, 2))
t = np.concatenate([np.zeros(n), np.ones(n)])

perm = rng.permutation(2 * n)
X = torch.from_numpy(X[perm]).float()      # float32 for the model
t = torch.from_numpy(t[perm]).long()       # int64 for cross_entropy
n_train = 800
train_ds = TensorDataset(X[:n_train], t[:n_train])
X_val, t_val = X[n_train:], t[n_train:]
loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# --- model ---
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)              # logits — no softmax here

model = MLP(2, 16, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.5)

# --- train ---
for epoch in range(40):
    for xb, yb in loader:
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        with torch.no_grad():              # evaluation: no graph needed
            val_logits = model(X_val)
            val_loss = F.cross_entropy(val_logits, t_val).item()
            val_acc = (val_logits.argmax(dim=1) == t_val).float().mean().item()
        print(epoch, "val loss", round(val_loss, 4), "val acc", round(val_acc, 3))
```

Validation accuracy should clear 0.95 within the 40 epochs. Before moving on, do
the line-by-line audit the deliverable asks for: for every line, say which week
built it. Data split — Week 09. `float()`/`long()` — §1. The class — Week 14 +
§3. Logits, no softmax — Week 13 + §5. The five-line loop — Week 14's worked
example. `no_grad` at eval — §2. If any line is "because the tutorial said so",
you are not done with it yet.

## Check yourself

1. Your NumPy code used `float64` everywhere. What dtype will bite you first in
   PyTorch, and where do integer tensors show up?
2. `y = x.view(2, 6)` then `y[0, 0] = 99`. What happened to `x`, and why?
3. Why do in-place operations break autograd? Answer using your Week 14 tanh
   backward rule.
4. What exactly does `requires_grad=True` change about subsequent computations?
5. Training works but the loss decreases oddly and gradients seem too big. You
   suspect a missing `zero_grad()`. Explain the mechanism, from your own engine.
6. `F.cross_entropy` wants logits and integer labels. Name the two things it
   does internally that you implemented by hand in Week 13.
7. Why must the counting bigram model and the trained one-layer neural bigram
   reach the same loss?
8. In the char-MLP, applying the `(27, d)` embedding table to a one-hot vector
   equals what simpler operation?

## Answers

1. `float32` vs `float64` mismatches (e.g. `torch.from_numpy` on a float64 array
   fed to a float32 model — convert with `.float()`). Integer (`long`) tensors
   carry class labels for `cross_entropy` and indices for lookups.
2. `x[0, 0]` is now 99. `view` returns a view — same memory, new shape; writing
   through either name changes the one underlying tensor.
3. The tanh rule reads the stored forward output (`1 - self.data ** 2`). An
   in-place op overwrites stored values that backward still needs, so the
   computed gradient would be wrong; PyTorch errors instead of guessing.
4. The tensor becomes a tracked leaf: every operation on it records parents and
   the producing op (`grad_fn`) — the graph gets built — and `backward()` will
   deposit a gradient in its `.grad`.
5. `.grad` accumulates with `+=` (needed so multiple paths sum). Without
   zeroing, each step's gradient adds onto all previous steps', so updates use a
   runaway sum of stale gradients — training limps or explodes but nothing errors.
6. The max-subtraction stability shift inside softmax, and the fused
   softmax+cross-entropy backward that collapses to `p − y`.
7. They minimize the same cross-entropy over the same model class (one
   conditional distribution per current character), and you prove on paper that
   its unique minimizer is the empirical frequencies — the counting answer. GD
   just walks to it.
8. Selecting row $i$ of the table — one-hot times a matrix is row lookup, so an
   embedding is a linear layer whose input is always one-hot, stored efficiently.

## New terms

- **tensor** — PyTorch's n-dimensional array; NumPy semantics plus GPU support
  and graph recording.
- **`requires_grad`** — flag marking a leaf tensor for gradient tracking.
- **`grad_fn`** — a tensor's record of the operation that produced it; your
  `_op`/`_parents` industrialized.
- **view** — a tensor sharing memory with another; reshapes and slices are views.
- **in-place operation** — one that overwrites a tensor's memory (`add_`, `+=`);
  hostile to autograd.
- **`no_grad` / `detach`** — run code without building the graph / return a
  tensor severed from the graph.
- **inheritance** — `class Child(Parent)`: the child class receives the parent's
  methods; `super().__init__()` runs the parent's setup.
- **`nn.Module`** — PyTorch's model base class: registers parameters, provides
  `parameters()`, makes `model(x)` call your `forward`.
- **epoch** — one full pass through the training data.
- **`DataLoader`** — serves shuffled mini-batches from a `Dataset`.
- **logits** — raw class scores fed to a fused softmax+cross-entropy (Week 13's
  word, now load-bearing).
- **bigram** — an adjacent pair of characters (or tokens); a bigram model
  predicts each from the previous one.
- **smoothing** — adding a small count everywhere so no event has probability 0.
- **NLL (negative log-likelihood)** — average $-\log P(\text{data})$; identical
  to cross-entropy loss.
- **embedding table** — a learned matrix whose rows are vector representations
  of discrete symbols; lookup = one-hot times matrix.

## Going deeper

- Karpathy, *Zero to Hero* — "The spelled-out intro to language modeling:
  building makemore" and "Building makemore Part 2: MLP" (the spine; §6 is the
  map, these are the territory — build along).
- Official PyTorch tutorials, "Learn the Basics" sequence — skim fast; you know
  the concepts, you are collecting API detail.
- Prince, *Understanding Deep Learning* (free PDF), Ch. 6 (Fitting models) —
  connects this training loop back to Week 08's optimizers.
