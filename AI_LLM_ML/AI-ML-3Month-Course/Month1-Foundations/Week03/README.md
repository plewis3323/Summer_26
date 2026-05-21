# Week 03 — PyTorch, MLPs, and Backprop from Scratch (Karpathy micrograd)

## Learning objectives

By the end of this week you will:

1. Build a scalar automatic-differentiation engine from scratch — one `Value` class that supports `+`, `*`, `tanh`, `relu`, `exp`, `log`, and a `backward()` that does reverse-mode AD on the resulting graph.
2. Re-derive, on paper, the gradients for MSE, binary cross-entropy, and softmax+cross-entropy. Know exactly why the `(p̂ - y)` form is the same for logistic and softmax layers.
3. Translate your scalar engine into PyTorch tensors and understand the parallels: `Tensor.requires_grad`, `loss.backward()`, `grad_fn`, and `torch.no_grad()`.
4. Train a small MLP on a real (if toy-sized) dataset, diagnose training pathologies (vanishing/exploding gradients, dead ReLUs, loss plateaus), and know which lever to pull.
5. Have an honest mental model of the universal approximation theorem and what it does and does not tell you.

If you skip the from-scratch autograd, you will forever think of PyTorch's `.backward()` as a black box. One afternoon investing in this pays off every week for the rest of the course.

## 1. Reverse-mode automatic differentiation — the whole idea

Forget neural networks for a moment. You have a scalar function f: ℝⁿ → ℝ built from elementary ops (+, *, sin, exp, …). You want ∇f efficiently.

Two numerical methods that don't work well:
- **Finite differences.** O(n) function evaluations per gradient component. Numerical noise. Stop using this after undergrad.
- **Symbolic differentiation.** Produces enormous expressions by the product rule. Expression swell.

**Automatic differentiation** sits between. You decompose f into a DAG of elementary ops, compute the forward pass storing intermediate values, then apply the chain rule **backward** through the DAG, reusing intermediate values.

For any elementary op, you need two things:
1. **Forward**: compute the output from inputs.
2. **Backward** (the local Jacobian-vector product): given the gradient of the *loss* with respect to the op's output, compute the gradient with respect to its inputs.

That's it. The rest is bookkeeping — stitching together the DAG and walking it in topological order backward.

### Why "reverse"?
If f: ℝⁿ → ℝ (many inputs, one output — the usual case for loss functions), reverse mode computes the full gradient in **one** backward pass, cost ≈ 2× forward. Forward mode would take n passes. For neural networks with millions of parameters, that's why we always use reverse.

### The chain rule, stated for computation graphs

Given a graph where node `v` has parents `p_1, ..., p_k` and you've computed:

$$
v = f(p_1, \ldots, p_k),
$$

the **local** gradients are ∂v/∂p_i. During backward, if you already know ∂L/∂v (the gradient of the final loss with respect to v), you compute:

$$
\frac{\partial L}{\partial p_i} \mathrel{+}= \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial p_i}.
$$

The `+=` is because a node can be used in multiple places; each use contributes. Missing the `+=` is the #1 bug in from-scratch AD.

## 2. Build `micrograd` (this is what you'll do in the exercises)

Karpathy's `micrograd` is ~100 lines of Python. Here's the structural outline:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._children = _children       # tuple of parent Values
        self._op = _op
        self._backward = lambda: None    # closure that updates parents' grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v in visited: return
            visited.add(v)
            for c in v._children:
                build(c)
            topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

That's the whole thing (modulo a few more ops and `__radd__` / `__rmul__` fluff for arithmetic with plain numbers). Read it three times. The only cleverness is the topological sort and the `+=` on gradients.

## 3. Derivations you must do on paper this week

### MSE
$$
\ell(\hat y, y) = \tfrac{1}{2}(\hat y - y)^2, \qquad \frac{\partial \ell}{\partial \hat y} = \hat y - y.
$$
The ½ is a convention to make the gradient clean. When you skip the ½, the gradient just has a factor of 2.

### Binary cross-entropy with sigmoid
Logits z → p̂ = σ(z) = 1/(1+e⁻ᶻ). Loss:
$$
\ell = -y \log \hat p - (1-y)\log(1 - \hat p).
$$
Derivative with respect to z (not p̂!):
$$
\frac{\partial \ell}{\partial z} = \hat p - y.
$$
Derive this yourself with the quotient rule on σ. The fact that `∂ℓ/∂z = residual` is the reason binary classification is numerically well-behaved.

### Softmax with cross-entropy
Logits z ∈ ℝᴷ. Softmax:
$$
\hat p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}.
$$
Cross-entropy with one-hot y:
$$
\ell = -\sum_k y_k \log \hat p_k.
$$

Gradient with respect to logits:
$$
\frac{\partial \ell}{\partial z_k} = \hat p_k - y_k.
$$

Same residual form. This is *not* a coincidence. The softmax is the exponential-family canonical link for the multinomial, and cross-entropy is its negative log-likelihood. Derive it cleanly once. It's ~half a page.

### ReLU
$$
\mathrm{ReLU}(z) = \max(z, 0), \qquad \frac{d}{dz}\mathrm{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \\ \text{undefined} & z = 0. \end{cases}
$$
In practice we just pick 0 at z = 0. Subgradient. Works fine.

### Linear layer
y = Wx + b where W ∈ ℝᵐˣⁿ, x ∈ ℝⁿ, y ∈ ℝᵐ. Given ∂L/∂y, the backward pass is:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \, x^T, \qquad
\frac{\partial L}{\partial x} = W^T \frac{\partial L}{\partial y}, \qquad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}.
$$

Memorize the pattern: "gradient through a linear op swaps data and parameter roles." This is the shape you'll see everywhere.

## 4. From scalars to tensors — enter PyTorch

Your micrograd engine operates one scalar at a time. That's pedagogically clean but computationally insane. Real ML moves tensors through ops in parallel.

PyTorch's `torch.Tensor` is a GPU-capable ndarray with tracking for AD. The API you use 95% of the time:

```python
import torch

x = torch.randn(100, 3, requires_grad=False)
y = torch.randn(100, 1)

W = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

pred = x @ W + b
loss = ((pred - y) ** 2).mean()

loss.backward()

print(W.grad.shape, b.grad.shape)  # matches W and b shapes
```

Key mental anchors:
- **`requires_grad=True`** marks a tensor as a leaf in the computation graph.
- Every op records itself; intermediate tensors have a `.grad_fn`.
- **`.backward()`** on a scalar triggers the reverse pass. Accumulates into `.grad` of leaf tensors. You must `.zero_()` between steps.
- **`torch.no_grad()`** or **`.detach()`** for eval/inference to skip graph construction.
- Shape broadcasting rules are the same as numpy's.

### The training loop, skeletonized
```python
opt = torch.optim.SGD([W, b], lr=0.01)
for step in range(n_steps):
    opt.zero_grad()
    pred = x @ W + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    opt.step()
```

Notice: you don't write the gradient update. `opt.step()` applies `param -= lr * param.grad` (for SGD). For Adam etc., it does more, but the interface is identical.

### `nn.Module` — organizing parameters
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

Everything under `self.fcN` (which are `nn.Parameter`s) automatically tracks grads and shows up in `model.parameters()`.

### Losses
```python
loss_fn = nn.CrossEntropyLoss()  # expects raw logits, NOT softmaxed probabilities
```

Common bug: you softmax inside your model, then pass to `CrossEntropyLoss`, and nothing trains. PyTorch's `CrossEntropyLoss` fuses log-softmax and NLL for numerical stability. Pass **logits**. If you need probabilities for eval, call `torch.softmax` outside the loss path.

## 5. MLPs — the humblest deep model

An MLP with L hidden layers:
$$
h_1 = \phi(W_1 x + b_1), \quad h_2 = \phi(W_2 h_1 + b_2), \quad \ldots, \quad \hat y = W_{L+1} h_L + b_{L+1},
$$
where φ is an activation (ReLU, tanh, GELU, …). Everything else in deep learning is either (a) structured layers (CNN, attention) or (b) tricks to train MLPs of this form stably.

### Universal approximation theorem
An MLP with one hidden layer and enough units can approximate any continuous function on a compact set arbitrarily well. What it does *not* say:
- How many units (the number can be exponential in input dimension).
- How to find the weights (non-convex optimization).
- That the approximation generalizes (it's a *function class* statement; nothing about data).

People quote this theorem to justify using neural nets. The real justification is empirical: stochastic gradient descent on a reasonable architecture finds good solutions in practice, for reasons we only partially understand.

### Activation functions
- **Sigmoid and tanh.** Bounded, smooth. Vanishing gradient when saturated. Good for output layers in probability/regression with bounded targets; bad as hidden activations in deep nets.
- **ReLU.** `max(z, 0)`. Solved the vanishing-gradient crisis. Default for hidden layers. Can "die": if a ReLU neuron's input stays < 0, gradient is 0 forever and it never recovers.
- **Leaky ReLU, ELU, GELU, SiLU.** Variants. GELU is the default in modern transformers.

### Initialization
Why it matters: if weights are too large, activations explode and gradients blow up; too small, activations collapse to zero and gradients vanish. Carefully initialized networks start with activation variances roughly equal across layers.

- **Xavier/Glorot** (for tanh/sigmoid): `W ~ N(0, 2/(fan_in + fan_out))`.
- **Kaiming/He** (for ReLU): `W ~ N(0, 2/fan_in)`.
- PyTorch's `nn.Linear` uses Kaiming-uniform by default. Usually fine.

### Optimizers (just enough for this week)
- **SGD.** `w ← w - η ∇L`. Simple, strong. Usually paired with momentum: `v ← β v + ∇L; w ← w - η v`.
- **Adam.** Per-parameter adaptive learning rates via running estimates of the gradient mean and variance. Default for deep nets — not because it's always best, but because it's usually fine out of the box.

Don't memorize Adam's update equations yet; we'll derive them in Week 04. Just know: if your vanilla SGD won't learn, try Adam; if Adam produces worse test accuracy than SGD, try SGD+momentum.

## 6. Training pathologies and the diagnostic toolkit

When training doesn't work, there is always a reason. The common ones:

### Loss is NaN after a few steps
- Learning rate too large.
- Exploding gradients; check `grad.norm()` per layer.
- Division by zero, log of zero, softmax with huge logits. Use numerically stable primitives (`log_softmax`, `logsumexp`).

### Loss doesn't decrease
- LR too small.
- Wrong loss: e.g. you forgot the `.mean()` and you're summing a batch.
- You forgot `opt.zero_grad()` and your gradients accumulate from step to step.
- Your model's output layer is wrong for the task (sigmoid before CrossEntropyLoss, etc.).
- Weights not updating because they aren't registered as `nn.Parameter`.

### Loss decreases on train, explodes on val
- Overfitting.
- Batchnorm / dropout in the wrong mode (forgot `model.eval()` at inference).

### Activations saturate, dead ReLUs
- Diagnose by histogram of activations per layer. ReLU outputs stuck at 0 mean that neuron is dead.
- Fix: reduce LR, use Leaky ReLU, better initialization, add batchnorm.

### Gradient vanishes with depth
- Classic symptom of pre-ResNet architectures. Gradient norm halves with every layer going backward.
- Fix: residual connections (Week 04), proper initialization, batchnorm, shorter networks.

### Karpathy's recipe (canonical)
Karpathy's blog "A Recipe for Training Neural Networks" is your diagnostic doctrine. Key moves:
1. **Become one with the data.** Plot it. Stare at it. Look at the labels. Check for corrupted rows.
2. **Set up the end-to-end training/eval skeleton, get dumb baselines.** Fixed model, fixed hyperparameters, see a number move.
3. **Overfit to a single batch first.** If you can't overfit 4 examples, there's a bug. No amount of data will save you.
4. **Add regularization.** Once overfitting works, throttle it back with dropout/weight decay/augmentation.
5. **Tune.** Learning-rate schedule, optimizer, architecture width/depth.

## 7. Hardware and wall-clock intuition

A simple MLP (2 hidden layers, 256 units, 10-class output, 60k MNIST-sized train set) trains in:
- ~10 seconds on a modern laptop CPU.
- ~2 seconds on a consumer GPU.
- The GPU is overkill. Only when models and data grow (Week 04's CNNs, Week 06's transformers) does GPU become essential.

Don't fetch a GPU for Week 03. Do your PyTorch work on CPU. It will force you to write small and fast.

## 8. Common pitfalls this week

- **In-place ops on leaf tensors.** `x += 1` where `x.requires_grad = True` without a `.detach()` will raise at backward time. Use `x.add_(1)` only on non-leaf tensors.
- **Forgetting `opt.zero_grad()`.** Your gradients accumulate across steps and your loss explodes mysteriously. Every training loop has `zero_grad` at the top.
- **Weights not moving.** Model has no parameters in the optimizer. `opt = torch.optim.SGD(model.parameters(), ...)` — if `model.parameters()` is empty, double-check every layer is an `nn.Module` attribute (not a list or dict, which don't auto-register).
- **`DataLoader` shuffling gotcha.** `shuffle=True` reshuffles every epoch. If you want deterministic iteration, seed inside the sampler.
- **Training-eval mode.** `model.train()` vs `model.eval()` changes batchnorm and dropout behavior. Switching is mandatory; the bug hides for epochs.
- **`torch.Tensor` vs `torch.tensor`.** The former is a constructor that doesn't copy; the latter always copies. Use `torch.tensor` when converting from numpy.

## 9. Connections to your HEP work

- Fit a tiny MLP to your EMCal shower-energy response function (the thing that maps cluster energy to true photon energy). Compare to the polynomial fit you probably have.
- Try a tiny MLP on the HEPMASS features from Week 02. It will be competitive with logistic regression on 28 hand-engineered features, probably worse than XGBoost. This teaches you a core lesson: deep learning does not magically outperform tabular methods on tabular data. It usually *loses*.
- Mentally stage for Week 04: the shower-classifier capstone is the natural next step — but for images, where CNNs beat MLPs cleanly.

## 10. Self-check

- Derive `∂ℓ/∂z` for binary cross-entropy with sigmoid in three lines.
- Explain why the backward pass through a linear layer transposes W.
- State the universal approximation theorem without the word "any" doing too much work.
- Describe three training pathologies and their diagnostic.
- In PyTorch, what happens if you call `loss.backward()` twice without `opt.zero_grad()`?

If any of these is hazy, revisit — Week 04's CNN work builds directly on this mental model.

---

Continue to `reading.md`, `exercises.md`, `project.md`, `resources.md`.
