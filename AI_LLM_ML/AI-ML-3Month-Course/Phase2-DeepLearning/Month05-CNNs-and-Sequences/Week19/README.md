# Week 19 — Sequences: RNNs, LSTMs, and Why Attention Won

Unrolled through time, an RNN is a very deep net with tied weights, so the T-step
gradient is a product of T Jacobians — the same instability as repeated transfer matrices.

## Objectives

- Write the RNN recurrence and unroll it; implement backprop through time (BPTT) by
  hand for a small vanilla RNN.
- Derive the vanishing/exploding gradient problem from the product of Jacobians, and
  connect it to the recurrent weight matrix's largest singular value.
- Explain the LSTM cell gate-by-gate, and why the additive cell-state path is the same
  trick as a ResNet skip connection.
- State precisely why long-range credit assignment fails in recurrent models, and give
  the intuition (no derivation yet) for attention as content-based lookup that makes the
  path length between any two positions O(1).
- Train a character-level RNN/LSTM and inspect what it learned.

## Core material (~3 hrs)

- CS231n lecture notes/slides on recurrent networks, plus Karpathy's blog post "The
  Unreasonable Effectiveness of Recurrent Neural Networks".
- Christopher Olah, "Understanding LSTM Networks" (colah's blog) — the canonical
  gate-by-gate walkthrough.
- Goodfellow et al., Ch. 10 (Sequence Modeling) — §on BPTT and on long-term dependency
  challenges.
- Attention intuition only: 3Blue1Brown's attention video, watched for pictures not
  equations — Week 25 derives it properly.

## Derivations (paper first)

- BPTT for a vanilla RNN: unroll 3 steps on paper, derive ∂L/∂W_hh as a sum over paths,
  each carrying a product of Jacobians.
- Show that with a linear recurrence, the T-step gradient scales as the T-th power of
  the largest singular value of W_hh (Week 06's SVD doing real work); state the bound
  when tanh is included.
- LSTM cell-state gradient: show the derivative along the cell path is controlled by the
  forget gate — additive memory, not multiplicative.

## Exercises (built when the week starts)

1. Vanilla RNN forward + BPTT in NumPy on a 5-step toy sequence. Accept when: gradients
   match finite differences to relative error < 1e-4.
2. Gradient-norm vs time-lag plot for the RNN, with W_hh scaled to spectral radius 0.9 /
   1.0 / 1.1. Accept when: plot shows vanish/stable/explode as derived on paper.
3. Copy task (recall a token after k blanks): RNN vs LSTM as k grows. Accept when: plot
   shows the RNN failing at much smaller k than the LSTM, with the failing k reported.
4. Character-level LSTM in PyTorch on the Week-15 names corpus (or physics-abstract
   text). Accept when: validation loss beats the Week-15 MLP char model.
5. Gradient clipping ablation on Exercise 4 with a deliberately high LR. Accept when:
   the run with clipping survives a loss spike the unclipped run dies on (both curves
   shown).
6. Attention teaser (no training): from provided query/key vectors, plot the softmax
   similarity weights over past positions. Accept when: the plot is produced and one
   line states what BPTT limitation this bypasses.

## Deliverable

Paper BPTT derivation + notebook with the vanishing-gradient measurement matching it,
a working character LSTM, and a one-paragraph "why attention" note in your own words —
to be graded against yourself in Week 25.

## Review

- (Week 17) Draw the parallel explicitly: LSTM cell path ↔ ResNet skip connection.
  What plays the role of the identity map in each?
- (Week 06) Why does the spectral radius (top singular value) of W_hh govern the
  T-step gradient? One line, citing your Week-06 SVD notes.
- (Week 14) Unrolling an RNN for T steps in your micrograd engine: where does the
  repeated-use gradient accumulation rule bite?
