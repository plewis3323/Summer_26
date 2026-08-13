# Week 15 — Exercises

Work top to bottom. Setup (imports, seeds, the two-moons data as tensors, the
cached download of `names.txt`, and the `sys.path` line that imports your Week 14
`micrograd.py`) is given by the notebook; you write only the lines each exercise
asks for. All exercises live in the notebook this week — no file-based
deliverables. Classes stay in play from Week 14 onward (`nn.Module` is one);
keep them as plain as the lesson's.

Paper first (a light week — the derivation muscle rests before Week 16): show
that minimizing the bigram model's cross-entropy over all conditional
probability tables gives exactly the empirical conditional frequencies — MLE
for a categorical distribution, Week 08's derivation run once per row. File the
page in this folder; E5 is its numerical confirmation.

## E1 — Tensor drills

For each expression in the setup's list (eight broadcasting shapes, two
view-vs-copy pairs), write your prediction — result shape, or "error" — in the
markdown cell *before* running anything. Then run the deliberate in-place
snippet the setup provides and explain its autograd error.
Hint: align shapes from the right, size-1 dimensions stretch (Week 03 rules,
lesson §1); for the view pairs ask "did I copy, or am I looking at the same
memory twice?"; for the error, your Week 14 tanh backward rule says which
stored value got destroyed.
Accept when: all shape predictions written before running are correct and the
error message is explained in one line.

## E2 — Gradient parity: micrograd vs PyTorch

Build Week 14's 2-16-1 tanh network twice — once with your `micrograd.py` MLP,
once from raw tensors with `requires_grad=True` — copy the *same* weights into
both, run one forward + squared-error backward on the same small batch, and
compare every parameter gradient.
Hint: walk both `parameters()` lists in the same fixed order when copying;
relative error is Week 14 E3's formula; keep the PyTorch side in float64
(`torch.double`) so the comparison is limited by math, not dtype.
Accept when: all parameter gradients agree to relative error < 1e-5.

## E3 — Port the Week-13 net to PyTorch

Rebuild the Week-13 NumPy MLP as `nn.Module` + `DataLoader` + `optim.SGD` on
the same two-moons data, and do the line-by-line audit from lesson §7 as you
go: for every line, name the week that built it.
Hint: the worked example *is* the template — same capacity (hidden width 16),
lr 0.5, batch size 32; `F.cross_entropy` takes logits and integer labels, so no
one-hot matrix and no softmax in `forward`.
Accept when: reaches the Week-13 accuracy (train accuracy > 95%) in comparable
epochs (the worked example's 40 are enough).

## E4 — Bigram name model from counts

Build the 27 × 27 count matrix over all bigrams in `names.txt` (with the `.`
boundary marker), smooth with + 1, row-normalize into `P`, sample 10 names from
it, and compute the dataset's average negative log-likelihood by indexing `P`.
Hint: lesson §6 has every piece; `torch.multinomial(p, num_samples=1)` draws
the next character; the NLL is the mean of `-log P[current, next]` over every
bigram in the dataset.
Accept when: the model's average negative log-likelihood matches the
count-based model's to 3 decimals (the check recomputes it with an independent
plain-Python loop over pairs — your vectorized number must agree with it).

## E5 — One-layer neural bigram

Recast E4 as learning: one-hot encode the current character, apply a single
`27 → 27` linear layer to get logits, train with `F.cross_entropy` and plain
gradient descent on all bigram pairs.
Hint: this is softmax regression, so the loss surface is convex — a large lr
(around 10, with the mean loss) and a few hundred full-batch steps converge;
smoothing shifts the count model's loss by less than the tolerance, so don't
chase it.
Accept when: converges to the count model's loss within 0.01 nats, confirming
the paper derivation (counting = the MLE optimum) numerically.

## E6 — MLP character model

Context length 3: map each of the previous three characters through a shared
`(27, 10)` embedding table, concatenate, feed a 200-unit tanh hidden layer,
then 27 output logits. Split the *names* 90/10 into train/val before building
example rows, train with minibatches, and plot train and validation loss
curves.
Hint: makemore Part 2 is the walkthrough; `emb.view(-1, 30)` flattens the three
embeddings — your first load-bearing view; split by names, not by rows, so no
name contributes context windows to both sides.
Accept when: validation loss beats the bigram model's, with train/val curves
plotted from a proper split.

## Review

1. (Week 14) In your engine, where do gradients accumulate and why must PyTorch
   users call `zero_grad()`?
2. (Week 08) The char model outputs a categorical distribution. Write its
   likelihood and show minimizing cross-entropy = maximizing likelihood.
3. (Week 09) You split names into train/val. What specific failure would
   training on all names and reporting training loss hide?
4. (Week 01) Which NumPy broadcasting rule explains why `(32, 27) + (27,)`
   works but `(32, 27) + (32,)` fails?
