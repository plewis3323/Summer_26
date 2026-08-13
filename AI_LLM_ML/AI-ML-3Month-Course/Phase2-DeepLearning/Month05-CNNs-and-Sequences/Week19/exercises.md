# Week 19 — Exercises

Work top to bottom. Setup (imports, the toy sequence and its weights, the copy-task
generator, the names corpus with the Week-15 split and baseline number, constants) is
given by the notebook; you write only the lines each exercise asks for. All exercises
live in notebook cells this week — no file-based deliverables. Everything trains in
minutes on CPU.

## E1 — Vanilla RNN forward + BPTT (in NumPy)

Write `rnn_forward(xs, h0, Wxh, Whh, b, Why, by)` returning the per-step hidden
states, logits, and total cross-entropy loss on the provided 5-step toy sequence,
then `rnn_backward(...)` returning `dWhh`, `dWxh`, `db` via the backward sweep.
Hint: lesson §3 — sweep t from T down to 1 accumulating
`delta_t = (local loss gradient at t) + delta_{t+1}` pushed through one Jacobian
`diag(1 - h_{t+1}**2) @ Whh`; `Whh` is used at every step, so its total gradient is
the *sum* of the per-use contributions (Week 14's accumulation rule).
Accept when: gradients match finite differences to relative error < 1e-4.

## E2 — Vanishing and exploding, measured

Using E1's backward pass on the provided 50-step sequence, rescale `Whh` to spectral
radius 0.9, 1.0, and 1.1 (rescaling helper given in setup) and plot the gradient norm
$\lVert \partial L_T / \partial h_k \rVert$ against time lag $T - k$ for all three on
one log-y axis.
Hint: use only the *final* step's loss, so each curve is a single product of
Jacobians; the norm of `delta` as the backward sweep passes step k is the curve.
Log-y turns geometric decay/growth into straight lines — compare their slopes to your
paper bound $(\gamma\,\sigma_{\max})^{t-k}$.
Accept when: plot shows vanish/stable/explode as derived on paper.

## E3 — The copy task: RNN vs LSTM

Train the given small `nn.RNN` and `nn.LSTM` classifiers on the copy task — recall
the first token after k blank steps — for each k in the provided list, and plot final
test accuracy vs k for both models on one axis.
Hint: keep hidden size and training budget identical for the two models — the
architecture is the only variable. Call a model "failed" at the first k where test
accuracy drops below 90% (chance level is given in setup) and print that k for both.
Accept when: plot shows the RNN failing at much smaller k than the LSTM, with the
failing k reported.

## E4 — A character-level LSTM

Fill in the `CharLSTM` forward pass, the training step, and the sampling loop
(the lesson §7 skeleton, loaders, and split are given) and train on the Week-15 names
corpus.
Hint: this is lesson §7 — write the forward from the §5 equations and the shape
comments, not from memory. The Week-15 MLP char model's validation loss on this exact
split is provided as `MLP_BASELINE`; don't forget the clipping line.
Accept when: validation loss beats the Week-15 MLP char model.

## E5 — Gradient clipping ablation

Rerun E4's training twice at the deliberately high learning rate given in setup,
identical seeds: once with `clip_grad_norm_`, once without, logging loss every step.
Plot both loss curves on one axis.
Hint: also log the *pre-clip* gradient norm — the spike that kills the unclipped run
shows up there first (lesson §4: clipping shrinks explosions; it cannot resurrect a
vanished signal). Cap the y-axis, or the dead run's curve will flatten the live one.
Accept when: the run with clipping survives a loss spike the unclipped run dies on
(both curves shown).

## E6 — Attention teaser (no training)

From the provided query vector and the 20 past-position key vectors, compute the
dot-product scores, softmax them, and bar-plot the resulting weights over past
positions; add one markdown line below the figure.
Hint: this is Week 10's softmax applied to similarity scores — three lines, no
learning. The provided keys make two past positions relevant; the plot should show
the weights finding them regardless of how far back they sit.
Accept when: the plot is produced and one line states what BPTT limitation this
bypasses.

## Review

1. (Week 17) Draw the parallel explicitly: LSTM cell path ↔ ResNet skip connection.
   What plays the role of the identity map in each?
2. (Week 06) Why does the spectral radius (top singular value) of $W_{hh}$ govern the
   T-step gradient? One line, citing your Week-06 SVD notes.
3. (Week 14) Unrolling an RNN for T steps in your micrograd engine: where does the
   repeated-use gradient accumulation rule bite?
4. (Week 15) The Week-15 char model saw a fixed 3-character window through an
   embedding table. What information can it never use, no matter how long it trains —
   and which E3 curve is that same fact, measured?
