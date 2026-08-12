# Week 03 — Exercises

Most of these are on paper or in a single notebook. Hand-derivations first; then implement.

## E1 — Hand-derive three gradients (easy, 60 min, paper)

Derive on paper, with every step shown:

1. ∂/∂z of `-log σ(z)` where σ is the sigmoid.
2. ∂/∂zₖ of softmax+cross-entropy with one-hot label y.
3. ∂/∂W of `L(Wx + b)` for a vector L and matrix W.

Check each with a numerical finite-difference test in a notebook.

## E2 — Complete `micrograd` (medium, 120 min)

**Data:** none.

Starting from the `Value` class outlined in `README.md §2`, extend it to support:

1. `__neg__`, `__sub__`, `__truediv__`, `__pow__` (for integer and float exponents).
2. `exp`, `log`.
3. `relu`.
4. `__radd__`, `__rmul__`, so you can do `2 + Value(3)`.

For each op, add a test that:
- verifies the forward value.
- verifies the backward via finite differences (ε = 1e-5) against the analytic `grad`.

Also implement the topological-sort `backward()` if you haven't.

**Accept when:** all ~20 tests pass; micrograd can compute gradients for expressions like `((2*a + b.exp()).tanh()).log()`.

## E3 — Gradient-agreement test against PyTorch (medium, 30 min)

Generate 50 random symbolic expressions using your operators and 3–5 input variables. For each, compute:
- The gradient using your micrograd.
- The gradient using `torch.autograd.grad` on the identical expression.

Assert max-abs difference < 1e-6.

**Hint:** use `torch.tensor([...], dtype=torch.float64, requires_grad=True)`. With float32, disagreement at 1e-4 is normal.

## E4 — Build an MLP in micrograd (medium, 90 min)

Build `Module`, `Linear`, `MLP` classes on top of `Value`. Each `Linear(n_in, n_out)` has `n_in * n_out + n_out` `Value` weights and biases. `parameters()` returns them all.

Then:
1. Generate a 2D half-moons dataset (`sklearn.datasets.make_moons(n=200, noise=0.2)`).
2. Train a 2-hidden-layer MLP (16, 16) with tanh activations and MSE loss in pure micrograd.
3. Plot decision boundary.

**Accept when:** training loss goes down monotonically and decision boundary is non-linear.

## E5 — Translate to PyTorch (medium, 60 min)

Re-implement E4 using PyTorch: `nn.Module`, `nn.Linear`, `nn.Tanh`, `torch.optim.SGD`, `nn.MSELoss`. Same architecture, same data, same seed.

1. Verify the learning curve matches micrograd's within noise.
2. Measure wall-clock of both. PyTorch on CPU should be 10–100× faster because of vectorized ops (micrograd does one scalar at a time).

## E6 — MNIST MLP (medium → hard, 90 min)

**Data:** `torchvision.datasets.MNIST`.

1. Build an MLP: 784 → 256 → 128 → 10.
2. Train with SGD+momentum, lr=0.01, momentum=0.9, 10 epochs, batch size 64.
3. Evaluate accuracy on the test set. Target ≥ 97%.
4. Plot training and test loss vs epoch.

**Hint:** use `CrossEntropyLoss` on raw logits. Don't softmax inside the model.

## E7 — Overfit one batch (medium, 30 min)

This is Karpathy's recipe step 3 in exercise form.

1. Take your MLP from E6 but feed it the *same* batch of 16 images every step.
2. Train for 2000 steps.
3. Loss should go below 1e-5 on that batch. If it doesn't, your model or training loop is broken.

Artificially break it: remove the `opt.zero_grad()`; observe the loss explode. Restore. Observe the fix.

## E8 — Diagnose a dying ReLU (hard, 45 min)

1. Build an MLP with ReLU activations and very large initial weights (scale `nn.init.normal_(W, std=3)`).
2. Train on MNIST for 1 epoch.
3. Plot a histogram of each layer's activations after 1 epoch.
4. Observe dead ReLUs (histograms spike at 0).
5. Fix with Kaiming initialization.
6. Re-plot. Spike should vanish.

## E9 — Implement Adam by hand (hard, 60 min)

In PyTorch, write an `Adam` optimizer class from scratch that matches `torch.optim.Adam` exactly. Implement bias correction.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat m_t = m_t / (1-\beta_1^t), \quad \hat v_t = v_t / (1-\beta_2^t)
$$
$$
\theta_t = \theta_{t-1} - \eta \, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
$$

Unit test: on a quadratic with known minimum, your Adam and `torch.optim.Adam` should produce identical trajectories given matched seeds.

## E10 — Regression on synthetic shower-energy response (hard, 60 min)

**Data:** simulate. Let true photon energy `E_true ~ Exp(1/10) + 0.5 GeV`. Measured cluster energy `E_meas = E_true * resp(E_true) + sigma(E_true) * randn()`, where `resp(E) = 0.95 + 0.02 * log(E)` and `sigma(E) = 0.05 * sqrt(E)`.

Task:
1. Generate 50 000 training events.
2. Train a small MLP to predict `E_true` given `E_meas`.
3. Compare the bias and resolution vs energy to:
   - A polynomial fit (scipy.optimize).
   - A simple analytical inverse of the response function.
4. Plot (bias, resolution) vs E_true for all three methods.

**Accept when:** MLP matches or beats the polynomial fit in the bulk and shows honest failure in the tails (<0.5 GeV and >50 GeV).

---

## Solution hints

- E1 — if you get the sign wrong on softmax, you built `y - p̂`, which minimizes the wrong direction. Double-check.
- E2 — when you implement `__pow__`, be careful: `x**n` for integer n has the gradient `n * x**(n-1)`, but for general float, use `exp(n * log(x))`.
- E3 — finite-differences on float32 loses precision fast. Use float64 tensors in the ground-truth PyTorch side.
- E4 — if training is slow, shrink the dataset to 64 points; micrograd is scalar-by-scalar.
- E6 — 97% is easy with a proper MLP on MNIST. If you're stuck at 90%, check your label shape (should be class indices, not one-hot).
- E7 — overfitting one batch catches 80% of bugs. Always do this first.
- E8 — dead ReLUs look like histogram spikes at exactly 0 covering >50% of the units.
- E9 — bias correction matters for the first ~10 steps. Without it, Adam underestimates the adaptive learning rate.
- E10 — MLP will excel in the bulk and underperform analytical methods in the tails. That's the generalization story in miniature: deep models are great interpolators, poor extrapolators.

---

Commit solutions. Move to the project.
