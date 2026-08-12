# Week 13 — Exercises

Work top to bottom. Setup (imports, toy-data generation, plotting axes, seeds) is
given by the notebook; you write only the lines each exercise asks for. All
exercises live in the notebook this week — no file-based deliverables.

Before the notebook: do the paper pass. Derive, by hand, (a) the softmax Jacobian
and $\partial L/\partial z = p - y$, (b) all four parameter gradients of the
2-layer net, (c) the general recursion $\delta^{(l)} = (W^{(l+1)\top}\delta^{(l+1)})
\odot g'(z^{(l)})$ — shapes annotated on every line. Scan or photograph the pages
into this folder. E2 and E3 are your derivation typed in; if the checks fail, fix
the paper first.

## E1 — Forward pass with shape asserts

Implement the 2-layer forward pass (affine → ReLU → affine → softmax) for a batch,
with an `assert a.shape == (...)` line after every intermediate. Include the
row-max stability shift in softmax.
Hint: `lesson.md` §8 has the batched formulas; `keepdims=True` keeps broadcasting
honest.
Accept when: output shape is `(batch, classes)` and every softmax row sums to 1
within 1e-6.

## E2 — Softmax/cross-entropy gradient

Implement `grad_logits(P, Y)` returning $\partial L/\partial Z^{(2)}$ for the
mean batch loss, straight from your paper result.
Hint: it is one line; do not forget the $1/n$ from the mean.
Accept when: matches central finite differences on random logits to relative
error < 1e-5.

## E3 — Full backward pass

Implement `backward(X, Y, Z1, H, P, W2)` returning all four gradients
`dW1, db1, dW2, db2`.
Hint: two matrix products, one broadcasted mask, two column sums — if you are
writing loops, revisit §8.
Accept when: all four gradients match central finite differences (h = 1e-5) to
relative error < 1e-4 on random small nets.

## E4 — Train on two moons

Train the net with plain gradient descent (your Week 08 update rule) on the
two-moons toy set from the lesson; plot the decision boundary over the data.
Hint: the worked example's hyperparameters (m = 16, lr = 0.5, 2000 steps) work;
for the boundary, predict on a grid via `np.meshgrid` and use `plt.contourf`.
Accept when: train accuracy > 95% and the plotted boundary visibly curves between
the moons.

## E5 — Universal approximation on a Breit–Wigner

A Breit–Wigner is the characteristic bump shape a short-lived particle leaves in a
mass spectrum: when experimenters plot how often collisions produce a given
invariant mass, an unstable particle appears as a peak centered on its mass whose
width reflects its lifetime, $f(E) \propto 1/\left[(E - M)^2 + \Gamma^2/4\right]$.
Fit a 1-hidden-layer regression net (same code, squared-error loss — the setup
gives you the loss and its gradient $\partial L/\partial \hat{y} = \hat{y} - y$)
to a Breit–Wigner curve at hidden widths 2, 8, and 64; plot all three fits over
the target.
Hint: only the output layer changes: one linear output unit, no softmax; the
backward recursion is untouched.
Accept when: the plot shows fit quality improving with width, plus a one-line note
naming something width does not fix (e.g. extrapolation outside the training
interval, or needing more steps to converge).

## E6 — Break it: remove the nonlinearity

Replace ReLU with the identity function and retrain on two moons. Also train
plain logistic regression (Week 10) on the same data for reference.
Hint: change one line in `forward` and the mask line in `backward`.
Accept when: the check confirms the "deep" net's accuracy is within 2 points of
logistic regression's — the collapse from `lesson.md` §3, live.

## Review

1. (Week 06) Write the SVD of a weight matrix $W = U\Sigma V^\top$. What does a
   rank-1 $W$ do to every input vector?
2. (Week 08) Derive cross-entropy from maximum likelihood for a categorical
   model — the same three lines as this week's loss.
3. (Week 08) Why does SGD's noise help escape saddle points where full-batch GD
   stalls?
4. (Week 10) Logistic regression's gradient was $(\hat{p} - y)x$. Why is that not
   a coincidence, given this week's $p - y$?
