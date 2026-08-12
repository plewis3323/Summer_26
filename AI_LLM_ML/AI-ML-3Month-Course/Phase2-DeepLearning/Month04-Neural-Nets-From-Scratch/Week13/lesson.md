# Week 13 — MLPs + Backprop on Paper

~5 hrs reading + 2 hrs paper derivation. Before starting you should be able to:
apply the chain rule to nested functions and compute partial derivatives (Week 05);
multiply matrices and track shapes through a product (Week 06); use the
matrix-calculus conventions for gradients of scalars with respect to vectors and
matrices (Week 07); derive the logistic-regression gradient and explain softmax and
cross-entropy (Week 10).

This is the flagship derivation of the course. Everything in Phase 2, and most of
Phase 3, is this week's math wearing different clothes. Take it slowly, do every
step on paper, and do not move on until the finite-difference check in the worked
example passes.

## 1. From logistic regression to a neuron

In Week 10 you built logistic regression: take an input vector $x$, compute a
weighted sum plus a bias, and squash the result through a sigmoid to get a
probability:

$$\hat{p} = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

That whole object — weights, bias, squashing function — is a **neuron**. There is
nothing more to the biological metaphor than that. A neuron computes one number
from many numbers: a linear score, then a nonlinear function of the score. The
nonlinear function is called the **activation function**, and the linear score $z$
is called the **pre-activation**.

Logistic regression is a single neuron. Its limitation, which you saw in Week 10,
is that a single linear score can only carve the input space with one straight
boundary (a line in 2D, a plane in 3D, a hyperplane in general). Real data —
signal vs background in a detector, one particle species vs another — usually
needs curved boundaries. The fix is not a cleverer single neuron. It is many
neurons, arranged in layers.

## 2. Layers and the multilayer perceptron

Take $m$ neurons, each with its own weight vector and bias, and feed them all the
same input $x \in \mathbb{R}^d$. Stack the $m$ weight vectors as rows of a matrix
$W \in \mathbb{R}^{m \times d}$ and the biases into $b \in \mathbb{R}^m$. Then all
$m$ pre-activations at once are one matrix-vector product:

$$z = W x + b \in \mathbb{R}^m,$$

and the $m$ outputs are $a = g(z)$, with the activation $g$ applied to each entry
separately (elementwise). This is a **layer**: an affine map (matrix times vector
plus a shift — Week 06) followed by an elementwise nonlinearity.

A **multilayer perceptron (MLP)** chains layers: the outputs of one layer become
the inputs of the next. The layers between input and output are **hidden layers**,
because their values are not part of the data — the network invents them. For a
2-layer MLP classifying into $K$ classes:

$$z^{(1)} = W^{(1)} x + b^{(1)}, \quad h = g(z^{(1)}), \quad
z^{(2)} = W^{(2)} h + b^{(2)}, \quad p = \mathrm{softmax}(z^{(2)}).$$

Shapes, which you should write next to every line you ever derive:

| object | shape | meaning |
|---|---|---|
| $x$ | $d$ | input features |
| $W^{(1)}, b^{(1)}$ | $m \times d$, $m$ | first (hidden) layer |
| $z^{(1)}, h$ | $m$ | hidden pre-activations, activations |
| $W^{(2)}, b^{(2)}$ | $K \times m$, $K$ | second (output) layer |
| $z^{(2)}, p$ | $K$ | class scores ("logits"), class probabilities |

The vector $z^{(2)}$ of raw class scores is often called the **logits** — the
numbers that softmax turns into probabilities, generalizing the log-odds from
Week 10.

Common activation functions:

- **ReLU** (rectified linear unit): $\mathrm{ReLU}(z) = \max(0, z)$. Cheap,
  and its derivative is just 0 or 1. The modern default.
- **tanh**: $\tanh(z)$, squashes to $(-1, 1)$. Smooth, zero-centered, but flat
  ("saturated") for large $|z|$ — Week 16 shows why that hurts.
- **sigmoid**: $\sigma(z)$, squashes to $(0, 1)$. Mostly retired as a hidden
  activation for the same saturation reason; still the right output for binary
  probabilities.

## 3. Why the nonlinearity is not optional

Suppose you drop $g$ and stack two affine layers:

$$z^{(2)} = W^{(2)} (W^{(1)} x + b^{(1)}) + b^{(2)}
        = (W^{(2)} W^{(1)}) x + (W^{(2)} b^{(1)} + b^{(2)}).$$

That is a single affine map with matrix $W^{(2)} W^{(1)}$. Any number of stacked
affine layers collapses to one — the "deep" network is exactly logistic (or
softmax) regression with extra arithmetic. The elementwise nonlinearity is the
only thing standing between an MLP and a linear model. Exercise E6 makes you
watch this collapse happen in training.

## 4. Universal approximation, and why it doesn't save you

The **universal approximation theorem** says: a network with one hidden layer and
a non-polynomial activation can approximate any continuous function on a bounded
region to any desired accuracy, if you allow the hidden layer to be wide enough.

Precisely: for any continuous $f$ on a closed bounded set, and any tolerance
$\varepsilon > 0$, there exists a width $m$ and weights such that the 1-hidden-layer
network is within $\varepsilon$ of $f$ everywhere on that set.

Read the fine print. The theorem guarantees such weights *exist*. It does not say:

1. **how wide** — $m$ may need to grow exponentially with the input dimension;
2. **that gradient descent finds those weights** — existence is not learnability;
3. **anything about generalization** — approximating the training points well says
   nothing about new points (Week 09's whole subject).

So "neural networks can represent anything" is true and almost useless. What makes
them work in practice is that gradient descent on the right architectures finds
good-enough weights from data — and the rest of this month is about making that
actually happen.

## 5. The loss: softmax + cross-entropy

For $K$-class classification, softmax (Week 10) converts logits to probabilities:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}},$$

where from here on $z$ means $z^{(2)}$, the logits. Each $p_k > 0$ and
$\sum_k p_k = 1$.

The label is **one-hot encoded**: $y$ is a vector of length $K$ that is 1 at the
true class $t$ and 0 elsewhere. The **cross-entropy loss** for one example is

$$L = -\sum_{k=1}^{K} y_k \log p_k = -\log p_t,$$

the negative log-probability the network assigned to the correct class. You
derived in Week 08 that minimizing cross-entropy is exactly maximizing the
likelihood of a categorical model — this is not an arbitrary choice of loss.

## 6. The softmax/cross-entropy gradient (scalar form)

Here is the first half of the flagship derivation. Goal: $\partial L / \partial z_j$
for every logit $z_j$. Chain rule through the probabilities:

$$\frac{\partial L}{\partial z_j}
 = \sum_{k=1}^{K} \frac{\partial L}{\partial p_k}\,\frac{\partial p_k}{\partial z_j}.$$

**Step 1 — loss w.r.t. probabilities.** From $L = -\sum_k y_k \log p_k$:

$$\frac{\partial L}{\partial p_k} = -\frac{y_k}{p_k}.$$

**Step 2 — the softmax Jacobian.** We need $\partial p_k / \partial z_j$. Write
$S = \sum_i e^{z_i}$, so $p_k = e^{z_k}/S$. Two cases, by the quotient rule
(Week 05).

Case $k = j$ (a logit's own probability):

$$\frac{\partial p_k}{\partial z_k}
 = \frac{e^{z_k} S - e^{z_k} e^{z_k}}{S^2}
 = p_k - p_k^2 = p_k (1 - p_k).$$

Case $k \neq j$ (raising $z_j$ steals probability from class $k$ through the
denominator only):

$$\frac{\partial p_k}{\partial z_j}
 = \frac{0 \cdot S - e^{z_k} e^{z_j}}{S^2}
 = -p_k\, p_j.$$

Both cases in one line, using the Kronecker delta $\delta_{kj}$ (1 if $k = j$,
else 0):

$$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j).$$

**Step 3 — combine.** Substitute both pieces into the chain-rule sum:

$$\frac{\partial L}{\partial z_j}
 = \sum_k \left(-\frac{y_k}{p_k}\right) p_k (\delta_{kj} - p_j)
 = -\sum_k y_k (\delta_{kj} - p_j)
 = -y_j + p_j \sum_k y_k.$$

The $p_k$'s cancelled — that is the magic step, and it only happens because the
loss is the *log* of a softmax. Since $y$ is one-hot, $\sum_k y_k = 1$, so:

$$\boxed{\ \frac{\partial L}{\partial z_j} = p_j - y_j\ }
\qquad \text{or in vector form} \qquad
\frac{\partial L}{\partial z} = p - y.$$

The gradient at the output is simply *prediction minus truth*. Compare Week 10:
logistic regression's gradient was $(\hat{p} - y)x$ — the same $p - y$ with the
input attached. Not a coincidence: sigmoid + binary cross-entropy is the $K = 2$
special case of softmax + cross-entropy. Whenever you pair an exponential-family
output with its matching log-loss, the gradient at the scores collapses to
prediction minus truth.

## 7. Backprop through the 2-layer net (scalar form)

Now the second half: gradients for all four parameter blocks
$W^{(2)}, b^{(2)}, W^{(1)}, b^{(1)}$. We work backwards from the loss — hence
**backpropagation**. Define the shorthand

$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = p - y \in \mathbb{R}^K,$$

the **error** at layer 2 — how much the loss changes per unit change of each
output pre-activation. We just derived it. Everything else is chain rule.

**Gradient of $W^{(2)}$.** Entry $W^{(2)}_{kj}$ touches the loss only through
$z^{(2)}_k = \sum_j W^{(2)}_{kj} h_j + b^{(2)}_k$. So

$$\frac{\partial L}{\partial W^{(2)}_{kj}}
 = \frac{\partial L}{\partial z^{(2)}_k}\,
   \frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}
 = \delta^{(2)}_k\, h_j.$$

Every (row $k$, column $j$) entry is a product of the $k$-th error and the $j$-th
hidden activation. That is precisely the **outer product** (Week 06):

$$\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} h^\top
\qquad (K \times m \ \checkmark).$$

Sanity-check the shape: gradient of a scalar w.r.t. a $K \times m$ matrix must be
$K \times m$; a $(K \times 1)(1 \times m)$ outer product is. Always run this check.

**Gradient of $b^{(2)}$.** $z^{(2)}_k$ depends on $b^{(2)}_k$ with derivative 1:

$$\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} \qquad (K \ \checkmark).$$

**Gradient flowing into $h$.** Each hidden unit $h_j$ feeds *all* $K$ outputs, so
its derivative is a sum over the paths (multivariate chain rule, Week 05):

$$\frac{\partial L}{\partial h_j}
 = \sum_{k=1}^{K} \frac{\partial L}{\partial z^{(2)}_k}
   \frac{\partial z^{(2)}_k}{\partial h_j}
 = \sum_k \delta^{(2)}_k W^{(2)}_{kj}
 = \left(W^{(2)\top} \delta^{(2)}\right)_j.$$

The transpose is not decoration: forward, $W^{(2)}$ maps hidden $\to$ output;
backward, $W^{(2)\top}$ maps output errors $\to$ hidden errors. Same wires, walked
in reverse.

**Through the ReLU.** $h_j = \mathrm{ReLU}(z^{(1)}_j)$, which has derivative 1
where $z^{(1)}_j > 0$ and 0 where $z^{(1)}_j < 0$ (at exactly 0 we just pick 0 —
it almost never matters and code has to pick something). Define layer 1's error:

$$\delta^{(1)}_j = \frac{\partial L}{\partial z^{(1)}_j}
 = \left(W^{(2)\top} \delta^{(2)}\right)_j \cdot
   \mathbf{1}\!\left[z^{(1)}_j > 0\right],$$

where $\mathbf{1}[\cdot]$ is 1 when the condition holds, else 0. In vector form,
with $\odot$ meaning elementwise product:

$$\delta^{(1)} = \left(W^{(2)\top} \delta^{(2)}\right) \odot g'(z^{(1)})
\qquad (m \ \checkmark).$$

**Gradients of $W^{(1)}, b^{(1)}$.** Identical structure to layer 2, with $x$
playing the role of $h$:

$$\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^\top \quad (m \times d\ \checkmark),
\qquad
\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)} \quad (m\ \checkmark).$$

That is the whole derivation. Four gradients, one pattern:

> **error at a layer $\times$ input to that layer $=$ weight gradient;**
> **transpose-times-error, gated by the activation derivative $=$ error one layer down.**

## 8. The general recursion and the batched (matrix) form

For an $L$-layer network the same steps give a recursion you can run to any depth:

$$\delta^{(L)} = p - y, \qquad
\delta^{(l)} = \left(W^{(l+1)\top} \delta^{(l+1)}\right) \odot g'(z^{(l)}),
\qquad
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} a^{(l-1)\top},$$

with $a^{(l-1)}$ the activations entering layer $l$ (and $a^{(0)} = x$). One
forward pass storing every $z^{(l)}$ and $a^{(l)}$, one backward pass — the total
cost is about twice a forward pass, no matter how many parameters. That
efficiency, not the math (which is 300-year-old chain rule), is why deep learning
is computationally possible.

**Batching.** In code we process $n$ examples at once, stored as *rows* of a
matrix $X \in \mathbb{R}^{n \times d}$ (the pandas/NumPy convention from Week 03).
With row vectors, every formula transposes. Forward:

$$Z^{(1)} = X W^{(1)\top} + b^{(1)}, \qquad H = g(Z^{(1)}), \qquad
Z^{(2)} = H W^{(2)\top} + b^{(2)},$$

where the bias adds to every row by broadcasting (Week 03). The batch loss is the
*mean* of per-example losses. Each example contributes its own outer product
$\delta_i a_i^\top$, and the sum over the batch of outer products is itself one
matrix product — the same "sum of outer products" identity you met in Week 06:

$$\frac{\partial L}{\partial W^{(2)}} = \frac{1}{n}\,\Delta^{(2)\top} H,
\qquad
\frac{\partial L}{\partial b^{(2)}} = \frac{1}{n}\sum_{i=1}^n \Delta^{(2)}_{i,:},$$

with $\Delta^{(2)} = P - Y \in \mathbb{R}^{n \times K}$ holding one row of errors
per example, and likewise

$$\Delta^{(1)} = \left(\Delta^{(2)} W^{(2)}\right) \odot g'(Z^{(1)}), \qquad
\frac{\partial L}{\partial W^{(1)}} = \frac{1}{n}\,\Delta^{(1)\top} X.$$

Nothing new happened — check any single entry and you recover the scalar formulas
of §7 averaged over examples. This is why backprop "batches cleanly": the chain
rule per example never couples different examples, so the batch versions are just
the per-example versions stacked and averaged.

## 9. Worked example

A complete, runnable NumPy implementation of everything above, checked against
finite differences. This is also the reference for the exercises.

```python
import numpy as np

rng = np.random.default_rng(0)

# --- toy data: two interleaved half-circles ("two moons"), 2 classes ---
n = 200
angles = rng.uniform(0.0, np.pi, size=n)
X0 = np.stack([np.cos(angles), np.sin(angles)], axis=1)
X1 = np.stack([1.0 - np.cos(angles), 0.5 - np.sin(angles)], axis=1)
X = np.concatenate([X0, X1]) + rng.normal(0.0, 0.1, size=(2 * n, 2))
t = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
Y = np.eye(2)[t]                      # one-hot labels, shape (2n, 2)

d, m, K = 2, 16, 2                    # input dim, hidden width, classes
W1 = rng.normal(0.0, 0.5, size=(m, d))
b1 = np.zeros(m)
W2 = rng.normal(0.0, 0.5, size=(K, m))
b2 = np.zeros(K)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1                # (n, m)
    H = np.maximum(0.0, Z1)           # ReLU, (n, m)
    Z2 = H @ W2.T + b2                # (n, K)
    Z2s = Z2 - Z2.max(axis=1, keepdims=True)   # stability shift
    expZ = np.exp(Z2s)
    P = expZ / expZ.sum(axis=1, keepdims=True) # softmax rows, (n, K)
    return Z1, H, P

def loss(P, Y):
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))

def backward(X, Y, Z1, H, P, W2):
    n = X.shape[0]
    D2 = (P - Y) / n                  # delta at layer 2, (n, K)
    dW2 = D2.T @ H                    # (K, m)
    db2 = D2.sum(axis=0)              # (K,)
    D1 = (D2 @ W2) * (Z1 > 0)         # delta at layer 1, (n, m)
    dW1 = D1.T @ X                    # (m, d)
    db1 = D1.sum(axis=0)              # (m,)
    return dW1, db1, dW2, db2

# --- finite-difference check on a few entries of W1 ---
Z1, H, P = forward(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(X, Y, Z1, H, P, W2)
h = 1e-5
for idx in [(0, 0), (3, 1), (10, 0)]:
    W1p = W1.copy(); W1p[idx] += h
    W1m = W1.copy(); W1m[idx] -= h
    _, _, Pp = forward(X, W1p, b1, W2, b2)
    _, _, Pm = forward(X, W1m, b1, W2, b2)
    num = (loss(Pp, Y) - loss(Pm, Y)) / (2 * h)
    rel = abs(num - dW1[idx]) / (abs(num) + abs(dW1[idx]) + 1e-12)
    print(idx, "analytic", dW1[idx], "numeric", num, "rel err", rel)

# --- train with plain gradient descent (Week 05/08) ---
lr = 0.5
for step in range(2000):
    Z1, H, P = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, Y, Z1, H, P, W2)
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    if step % 400 == 0:
        acc = np.mean(P.argmax(axis=1) == t)
        print(step, "loss", round(loss(P, Y), 4), "acc", round(acc, 3))
```

Run it. The three relative errors should print around $10^{-8}$ or smaller — if
one is $10^{-2}$, your backward pass is wrong *in that path*, which is exactly why
finite differences are the unit test of this whole month. Training should pass 95%
accuracy well before 2000 steps: a curved boundary, from nothing but the chain
rule and gradient descent.

Note the two implementation habits worth copying forever: subtracting the row max
before `exp` (softmax overflows for logits around 700 otherwise — the shift
changes nothing mathematically because it cancels in the ratio), and the
`+ 1e-12` inside the log (so a confidently-wrong $p = 0$ prints `inf` loss instead
of crashing).

## Check yourself

1. A 3-layer MLP uses the identity function as its activation. What model is it
   equivalent to, and why?
2. State one thing the universal approximation theorem guarantees and two things
   it does not.
3. Derive $\partial p_k / \partial z_j$ for softmax, both cases, without looking.
4. Why does $\partial L / \partial z = p - y$ come out so simple — which two
   choices had to be paired for the cancellation to happen?
5. In $\delta^{(1)} = (W^{(2)\top} \delta^{(2)}) \odot g'(z^{(1)})$, explain in one
   sentence each what the transpose does and what the elementwise product does.
6. $W^{(1)}$ is $m \times d$. What shape is $\partial L / \partial W^{(1)}$, and
   what fact makes that answer instant?
7. Your finite-difference check uses $(f(\theta + h) - f(\theta - h)) / 2h$. Why is
   this *central* form preferred over $(f(\theta + h) - f(\theta)) / h$?
8. Backprop for one batch costs roughly how many forward passes' worth of compute,
   independent of parameter count?

## Answers

1. Softmax (multinomial logistic) regression. Stacked affine maps compose into a
   single affine map (§3), so without a nonlinearity depth adds nothing.
2. Guarantees: weights *exist* for a 1-hidden-layer net to approximate any
   continuous function on a bounded set to any accuracy. Does not guarantee: the
   required width is practical (it can grow exponentially), that gradient descent
   finds those weights, or that the fit generalizes beyond the training points.
3. $\partial p_k / \partial z_k = p_k(1 - p_k)$;
   $\partial p_k / \partial z_j = -p_k p_j$ for $k \ne j$; together
   $p_k(\delta_{kj} - p_j)$. Rerun §6 Step 2 if either case felt shaky.
4. Softmax paired with *cross-entropy* (its matching log-likelihood loss). The
   $1/p_k$ from the log exactly cancels the $p_k$ factored out of the softmax
   Jacobian; with any other loss the cancellation dies.
5. The transpose routes each output error backward along the same weights that
   carried the activation forward (output errors $\to$ hidden errors). The
   elementwise product gates each hidden error by how responsive its activation
   was — a ReLU that was off ($z \le 0$) passes zero gradient.
6. $m \times d$ — the gradient of a scalar with respect to any array always has
   the shape of that array, because it holds one partial derivative per entry.
7. Taylor-expanding, the forward difference has error $O(h)$ while the central
   difference's even-order terms cancel, leaving $O(h^2)$ — far more accurate for
   the same $h$, which matters when you demand relative errors near $10^{-6}$.
8. About two-to-three: one forward pass plus a backward pass of comparable cost
   (each backward step reuses the stored forward quantities).

## New terms

- **neuron** — one linear score plus one nonlinear activation; logistic regression
  is a single neuron.
- **activation function** — the elementwise nonlinearity (ReLU, tanh, sigmoid).
- **pre-activation** — the linear score $z$ before the activation is applied.
- **layer** — an affine map plus an elementwise activation applied to a whole
  vector of neurons at once.
- **hidden layer** — a layer whose values are neither input nor output; the
  network's internal representation.
- **multilayer perceptron (MLP)** — layers chained one after another.
- **logits** — the raw class-score vector that softmax converts to probabilities.
- **one-hot encoding** — a label written as a vector with a single 1 at the true
  class.
- **universal approximation theorem** — one wide-enough hidden layer can
  approximate any continuous function on a bounded set; an existence statement only.
- **backpropagation** — the chain rule organized layer-by-layer from the loss
  backward, reusing stored forward values.
- **error ($\delta^{(l)}$)** — $\partial L / \partial z^{(l)}$, the gradient of the
  loss at a layer's pre-activations.
- **Kronecker delta** — $\delta_{kj}$, equal to 1 when $k = j$ and 0 otherwise.
- **outer product** — column vector times row vector, giving a matrix; the shape
  of every weight gradient.

## Going deeper

- Prince, *Understanding Deep Learning* (free PDF), Ch. 3–4 — shallow and deep
  networks with excellent figures; Ch. 5 re-derives cross-entropy from maximum
  likelihood, which should feel like Week 08 again.
- 3Blue1Brown, *Neural networks* series, chapters 1–4 — the animations put
  pictures to $\delta$; re-watch chapter 4 with your paper derivation beside you.
- Goodfellow, Bengio & Courville, *Deep Learning* (free online), §6.5 — the same
  derivation in fully general computational-graph notation; a preview of Week 14.
- CS231n course notes, "Backpropagation, Intuitions" — the staged-computation,
  local-gradient view that becomes your autograd engine next week.
