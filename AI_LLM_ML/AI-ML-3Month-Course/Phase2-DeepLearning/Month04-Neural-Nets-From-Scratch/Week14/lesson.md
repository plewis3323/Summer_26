# Week 14 — micrograd: A Scalar Autograd Engine

~5 hrs building + 1 hr paper. Before starting you should be able to: derive
backprop for a 2-layer net and state the backward recursion (Week 13); apply the
multivariate chain rule — a variable feeding several paths contributes a sum of
terms (Week 05); write and call your own functions, use lists and loops fluently
(Week 02); check a derivative with central finite differences (Week 13).

Last week you ran the chain rule by hand and hard-coded the result for one fixed
architecture. Change the network — add a layer, swap the loss — and you re-derive
and re-code the backward pass. This week you make the computer do the bookkeeping
once and for all: an **autograd engine** (automatic gradient engine) that watches
any expression being computed and then produces exact derivatives of its output
with respect to every input. PyTorch's autograd, which you adopt next week, is
this week's ~100 lines with tensors instead of scalars and a decade of
engineering. Build it yourself and PyTorch will never be a black box.

## 1. The idea: record the computation, then walk it backwards

Take a small expression and compute it the way a computer actually does — one
operation at a time, each producing an intermediate value:

$$L = \tanh(a \cdot b + c)
\qquad\text{becomes}\qquad
u = a \cdot b, \quad v = u + c, \quad L = \tanh(v).$$

Draw each value as a node and each operation as arrows from its inputs to its
result and you get a **computational graph**:

```
a ──┐
    (*)── u ──┐
b ──┘         (+)── v ──(tanh)── L
c ────────────┘
```

The graph is a **DAG** — a directed acyclic graph: arrows have a direction
(inputs to outputs) and you can never follow arrows in a circle, because a value
is computed from things that already exist.

Two passes over this graph do everything:

- **Forward pass:** compute each node's value from its inputs, left to right.
  You have been doing this since Week 01; it is just evaluating the expression.
- **Backward pass:** compute $\partial L / \partial(\text{node})$ for every node,
  right to left, starting from $\partial L / \partial L = 1$.

The backward pass works because each operation only needs its **local
derivative** — the derivative of its own output with respect to its own inputs,
ignoring the rest of the graph. The chain rule then says:

> gradient flowing *into* an input $=$ local derivative $\times$ gradient flowing
> *out of* the result.

For our expression, walking right to left with $\partial L/\partial L = 1$:

$$\frac{\partial L}{\partial v} = 1 - \tanh^2(v), \qquad
\frac{\partial L}{\partial u} = \frac{\partial L}{\partial v} \cdot 1, \qquad
\frac{\partial L}{\partial c} = \frac{\partial L}{\partial v} \cdot 1,$$
$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial u} \cdot b, \qquad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial u} \cdot a.$$

Every rule is a one-liner. Week 13's $\delta^{(l)}$ recursion was exactly this,
with whole layers as the nodes. The engine's job is to store the graph while the
forward pass runs, then apply these one-liners in the right order — no matter
what expression you wrote.

Do the paper trace now with $a = 2$, $b = -3$, $c = 5$: forward gives $u = -6$,
$v = -1$, $L = \tanh(-1) \approx -0.7616$; since $1 - \tanh^2(-1) \approx 0.4200$,
backward gives $\partial L/\partial c \approx 0.4200$,
$\partial L/\partial a \approx -1.2600$, $\partial L/\partial b \approx 0.8400$.
Keep these numbers; the engine must reproduce them.

## 2. Python classes, from zero

To record a graph, each value must carry luggage: its number, its gradient, which
values produced it, and which operation did. A plain `float` can't carry luggage.
Python's tool for bundling data with the functions that work on that data is the
**class** — and this is the first time this course needs one, so here it is from
nothing.

A class is a blueprint. From the blueprint you stamp out **objects** (also called
**instances**), each holding its own data. The data slots are **attributes**; the
functions bundled inside are **methods**.

```python
class Particle:
    def __init__(self, name, mass):
        self.name = name
        self.mass = mass

    def energy(self, momentum):
        return (momentum ** 2 + self.mass ** 2) ** 0.5

mu = Particle("muon", 0.1057)
print(mu.name)          # muon
print(mu.energy(2.0))   # 2.0027923...
```

Reading this line by line:

- `class Particle:` starts the blueprint. Everything indented under it belongs
  to the class.
- `__init__` is a special method Python calls automatically when you create an
  object: `Particle("muon", 0.1057)` runs `__init__` with those arguments. Its
  job is to fill in the attributes. The double underscores mark it as a method
  Python itself calls; you never call `__init__` directly.
- `self` is the object being worked on. Every method's first parameter is
  `self`, and Python fills it in for you: `mu.energy(2.0)` means
  "run `energy` with `self = mu` and `momentum = 2.0`".
- `self.mass = mass` creates an attribute. Afterwards `mu.mass` reads it from
  the object, anywhere, in any method.

Each object is independent: `pi = Particle("pion", 0.1396)` has its own `name`
and `mass`; changing `pi.mass` never touches `mu.mass`.

One more piece and we have everything the engine needs. Python translates
operators into specially named methods: `x + y` actually runs `x.__add__(y)`,
and `x * y` runs `x.__mul__(y)`. Define those methods on your class and your
objects work with ordinary `+` and `*` signs. This is called **operator
overloading**, and it is why the engine below feels like normal arithmetic.

That is the whole classes lesson: `class`, `__init__`, `self`, attributes,
methods, operator methods. Nothing fancier appears this week or, per the course
rules, anywhere in Phase 2.

## 3. The `Value` class: forward pass

Each `Value` wraps one number and remembers how it was made:

```python
class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data          # the number itself
        self.grad = 0.0           # dL/d(this value), filled by backward()
        self._parents = parents   # the Values this one was computed from
        self._op = op             # which operation made it ("+", "*", ...)

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def tanh(self):
        import math
        t = math.tanh(self.data)
        return Value(t, (self,), "tanh")
```

Notes on the three unfamiliar bits: `parents=()` gives a default — leaf values
you create by hand have no parents; `(self, other)` is a tuple, a list you can't
modify, holding the inputs; `isinstance(other, Value)` asks "is this already a
`Value`?" so that `v + 3.0` works by wrapping the plain number first. The leading
underscore in `_parents` is just a naming convention meaning "internal — users of
the class shouldn't poke this".

Now watch the graph build itself:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(5.0)
L = (a * b + c).tanh()
print(L.data)        # -0.76159...
print(L._op)         # tanh
print(L._parents[0]._op)   # +
```

Writing the expression *is* the forward pass, and every intermediate `Value`
records its parents. The graph from §1 now exists in memory.

## 4. The backward pass: local rules plus one subtlety

Each operation needs its local rule, "grad flowing out $=$ local derivative
$\times$ grad flowing in," written as a method that pushes gradient from a node
to its parents:

```python
    def _local_backward(self):
        if self._op == "+":
            self._parents[0].grad += 1.0 * self.grad
            self._parents[1].grad += 1.0 * self.grad
        elif self._op == "*":
            self._parents[0].grad += self._parents[1].data * self.grad
            self._parents[1].grad += self._parents[0].data * self.grad
        elif self._op == "tanh":
            self._parents[0].grad += (1.0 - self.data ** 2) * self.grad
```

(This method goes inside the `Value` class, indented like the others. The tanh
rule uses $\tanh'(x) = 1 - \tanh^2(x)$, and since `self.data` already *is*
$\tanh(x)$, the local derivative is `1 - self.data ** 2`.)

**Why `+=` and not `=`.** Suppose a value is used twice: `y = x * x + x`. Then
$x$ reaches $L$ through two paths, and the multivariate chain rule (Week 05) says
its derivative is the *sum* of the contributions from every path. Each path calls
`_local_backward` on a different node, and each call adds its share into
`x.grad`. Write `=` instead and the second path silently erases the first — the
single most classic autograd bug, and exercise E2 makes you catch it red-handed.
The same accumulation is why every training loop must reset `grad` to zero before
each backward pass, and why PyTorch has `zero_grad()` — remember this next week.

## 5. Ordering the backward pass: topological sort

The local rules only work if, by the time a node pushes gradient to its parents,
its own `grad` is complete — every path from $L$ down to it has already
delivered its share. So nodes must run in an order where each node comes after
everything that depends on it. The standard tool is a **topological order** of
the DAG: a listing in which every node appears after all of its parents (leaves
first, output last) — run the backward pass over that list *reversed* and each
node fires only after all its dependents have. The clean way to build one is
with a function that calls itself:

```python
    def _collect(self, order, visited):
        if self not in visited:
            visited.append(self)
            for p in self._parents:
                p._collect(order, visited)
            order.append(self)

    def backward(self):
        order = []
        visited = []
        self._collect(order, visited)
        self.grad = 1.0
        for node in reversed(order):
            node._local_backward()
```

A function calling itself is **recursion**: `_collect` on a node first collects
all of that node's parents (each of which collects *its* parents, and so on down
to the leaves, which have no parents and stop), and only then appends the node
itself. So `order` lists leaves first and the output last; `visited` guarantees
a node used twice is only collected once. Walking `order` in *reverse* starts at
the output and reaches every node only after all its dependents have run —
exactly what the local rules require. `self.grad = 1.0` seeds the recursion:
$\partial L / \partial L = 1$.

## 6. Why backwards? Reverse mode vs forward mode

You could also push derivatives *forward*: start at one input, set its derivative
to 1, and carry $\partial(\text{node})/\partial(\text{that input})$ through the
graph alongside the values. That is **forward-mode** autodiff, and it is perfectly
correct — but one pass gives the derivative with respect to *one input*. A
network with a million parameters would need a million forward passes.

**Reverse mode** — what we built — runs once from the single scalar output and
delivers $\partial L/\partial(\text{everything})$ in that one sweep. In Week 07's
language: for $f:\mathbb{R}^n \to \mathbb{R}$ the Jacobian is one row ($1 \times
n$), and reverse mode computes a vector–Jacobian product, capturing the whole row
in one pass. Forward mode wins in the opposite regime, few inputs and many
outputs — rare in ML, where the loss is one number and the parameters are
millions. That asymmetry is the entire reason deep learning trains in reasonable
time.

## 7. The complete engine

Here is the full class, with the extra operations a network needs: power (for
squared-error losses), negation and subtraction (built from `*` and `+`), ReLU
and exp, and the reflected methods `__radd__`/`__rmul__` so `3.0 * v` works when
the plain number is on the left (Python asks the number first, fails, then asks
`Value`'s reflected method). This is the module the exercises grow.

```python
import math

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._exponent = None      # used only by ** nodes

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __pow__(self, k):          # v ** k, k a plain number
        out = Value(self.data ** k, (self,), "pow")
        out._exponent = k
        return out

    def __neg__(self):             # -v
        return self * -1.0

    def __sub__(self, other):      # v - w
        return self + (-other)

    def __radd__(self, other):     # 3.0 + v
        return self + other

    def __rmul__(self, other):     # 3.0 * v
        return self * other

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")

    def relu(self):
        return Value(max(0.0, self.data), (self,), "relu")

    def exp(self):
        return Value(math.exp(self.data), (self,), "exp")

    def _local_backward(self):
        if self._op == "+":
            self._parents[0].grad += 1.0 * self.grad
            self._parents[1].grad += 1.0 * self.grad
        elif self._op == "*":
            self._parents[0].grad += self._parents[1].data * self.grad
            self._parents[1].grad += self._parents[0].data * self.grad
        elif self._op == "pow":
            k = self._exponent
            self._parents[0].grad += k * self._parents[0].data ** (k - 1) * self.grad
        elif self._op == "tanh":
            self._parents[0].grad += (1.0 - self.data ** 2) * self.grad
        elif self._op == "relu":
            if self._parents[0].data > 0:
                self._parents[0].grad += self.grad
        elif self._op == "exp":
            self._parents[0].grad += self.data * self.grad

    def _collect(self, order, visited):
        if self not in visited:
            visited.append(self)
            for p in self._parents:
                p._collect(order, visited)
            order.append(self)

    def backward(self):
        order = []
        visited = []
        self._collect(order, visited)
        self.grad = 1.0
        for node in reversed(order):
            node._local_backward()
```

Every local rule is the calculus you already own: $\frac{d}{dx}x^k = kx^{k-1}$,
$\frac{d}{dx}e^x = e^x$ (so the local derivative is the node's own output),
ReLU passes gradient only where its input was positive — Week 13's mask, one
scalar at a time.

Trust nothing until it survives a **central finite-difference** check
(Week 13): nudge one input by $\pm h$, difference the outputs, compare. Every
new operation you ever add to an engine gets this test before it gets used.

## 8. From engine to network

A neuron (Week 13) is weights, a bias, a dot product, an activation — all
buildable from `Value`s. Three small classes stack up exactly like §2's
blueprint, no new Python:

```python
import random

class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)

    def out(self, x):
        s = self.b
        for i in range(len(self.w)):
            s = s + self.w[i] * x[i]
        return s.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def out(self, x):
        return [n.out(x) for n in self.neurons]

    def parameters(self):
        params = []
        for n in self.neurons:
            params = params + n.parameters()
        return params

class MLP:
    def __init__(self, n_in, sizes):
        self.layers = []
        prev = n_in
        for s in sizes:
            self.layers.append(Layer(prev, s))
            prev = s

    def out(self, x):
        for layer in self.layers:
            x = layer.out(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params = params + layer.parameters()
        return params
```

One design choice to note: we use tanh outputs and labels of $\pm 1$ with a
squared-error loss, rather than Week 13's softmax + cross-entropy. Softmax
couples a whole vector of outputs through its denominator, which is clumsy one
scalar at a time; squared error keeps every gradient flowing through ops the
engine already has. The training loop and the learning are otherwise identical.

## 9. Worked example

Everything assembled: verify the paper trace from §1, then train an MLP on XOR —
the classic dataset a single neuron provably cannot separate (label is $+1$ when
the two inputs differ, $-1$ when they agree; no single straight line splits
that). Run after the class definitions above.

```python
# --- 1. the paper trace from section 1 ---
a = Value(2.0)
b = Value(-3.0)
c = Value(5.0)
L = (a * b + c).tanh()
L.backward()
print("L", round(L.data, 4))        # -0.7616
print("dL/da", round(a.grad, 4))    # -1.2600
print("dL/db", round(b.grad, 4))    #  0.8400
print("dL/dc", round(c.grad, 4))    #  0.4200

# --- 2. finite-difference check of the same numbers ---
h = 1e-5
Lp = (Value(2.0 + h) * Value(-3.0) + Value(5.0)).tanh()
Lm = (Value(2.0 - h) * Value(-3.0) + Value(5.0)).tanh()
numeric = (Lp.data - Lm.data) / (2 * h)
print("dL/da numeric", round(numeric, 4), "engine", round(a.grad, 4))

# --- 3. train a 2-4-1 MLP on XOR ---
random.seed(0)
X = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
Y = [-1.0, 1.0, 1.0, -1.0]

model = MLP(2, [4, 1])
lr = 0.2
for step in range(200):
    loss = Value(0.0)
    for i in range(len(X)):
        pred = model.out(X[i])[0]
        loss = loss + (pred - Y[i]) ** 2
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():
        p.data = p.data - lr * p.grad
    if step % 40 == 0:
        print(step, "loss", round(loss.data, 5))

for i in range(len(X)):
    print(X[i], "->", round(model.out(X[i])[0].data, 3), "target", Y[i])
```

The loss should fall below 0.05 and the four predictions should land near their
targets. Read the training loop's four moves in order — forward, zero the grads,
backward, update — because they are, line for line, next week's PyTorch loop:
`model(x)` / `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`.
You have now written the thing those calls abstract.

Two honest limitations to carry forward. First, speed: each scalar `Value` is a
full Python object, and Python-level bookkeeping per multiply is thousands of
times slower than Week 13's NumPy, which hands whole-matrix products to BLAS —
compiled, hardware-tuned linear-algebra routines. Exercise E6 measures the gap.
Second, memory: the graph stores every intermediate value until `backward()`
consumes it. Both are why real engines record the same graph over tensors (whole
arrays per node) rather than scalars — which is precisely PyTorch, next week.

## Check yourself

1. In the graph for `L = (a * b + c).tanh()`, list the nodes in one valid
   topological order.
2. What does `__init__` do, and who calls it?
3. In `mu.energy(2.0)`, what is `self`?
4. `d = v * w` — which method actually runs, and what three things does the new
   `Value` record?
5. Why must `_local_backward` use `+=`? Give the two-path expression that breaks
   under `=`.
6. Why does `backward()` walk the topological order in *reverse*?
7. A model has $10^6$ parameters and one scalar loss. How many passes does
   reverse mode need for all gradients? Forward mode?
8. Your engine and NumPy compute the same gradients. Name the main reason the
   engine is thousands of times slower.

## Answers

1. `a, b, u=a*b, c, v=u+c, L` (leaves may be in any order among themselves; each
   computed node must follow its parents).
2. It fills in a new object's attributes; Python calls it automatically when you
   write `Value(2.0)` — you never call it yourself.
3. The object `mu` — Python passes the object before the dot as the first
   parameter of the method.
4. `v.__mul__(w)`. The new `Value` records the product in `data`, the tuple
   `(v, w)` in `_parents`, and `"*"` in `_op`.
5. Gradients from multiple paths must *add* (multivariate chain rule). In
   `y = x * x + x`, `x` gets contributions from three uses; with `=` each write
   erases the previous ones and `x.grad` ends up holding only one path's share.
6. A node may only push gradient to its parents once its own `grad` is complete —
   i.e., after every node that depends on it has already run. Reverse topological
   order guarantees exactly that, starting from the output.
7. Reverse: one backward pass (plus the forward). Forward mode: $10^6$ passes,
   one per parameter — the reason ML uses reverse mode.
8. Per-operation Python-object overhead (allocating a `Value`, method dispatch,
   graph bookkeeping for every single multiply) versus NumPy dispatching whole
   matrix products to compiled BLAS routines.

## New terms

- **autograd engine** — code that records computations as they run and produces
  exact gradients of the output with respect to every input.
- **computational graph** — values as nodes, operations as directed edges from
  inputs to results.
- **DAG (directed acyclic graph)** — a graph with one-way edges and no cycles.
- **local derivative** — an operation's derivative w.r.t. its own inputs,
  independent of the surrounding graph.
- **class / object / instance** — a blueprint, and the things stamped from it.
- **attribute** — a data slot on an object (`self.data`).
- **method** — a function bundled into a class; its first parameter is the object.
- **`__init__`** — the special method Python runs when an object is created.
- **`self`** — the object a method is acting on.
- **operator overloading** — defining `__add__`, `__mul__`, … so your objects
  work with `+` and `*`.
- **topological order** — an ordering of a DAG in which every node appears after
  its parents (so the reverse walks output-to-leaves).
- **recursion** — a function calling itself on a smaller piece of the problem.
- **reverse-mode / forward-mode autodiff** — propagating derivatives from the
  output backwards (one pass for all parameters) vs from one input forwards.
- **BLAS** — compiled, hardware-tuned linear-algebra routines; what NumPy calls
  for matrix products.

## Going deeper

- Karpathy, *Neural Networks: Zero to Hero* — "The spelled-out intro to neural
  networks and backpropagation: building micrograd" (the spine, ~2.5 hrs). Build
  alongside; his `_backward` uses a Python feature (functions stored inside
  functions) where ours uses the `if/elif` dispatch — same machine, compare them.
- The micrograd source on GitHub (karpathy/micrograd) — read it *after* your own
  attempt, and diff the design decisions against yours.
- CS231n notes on backpropagation — the "staged computation" and local-gradient
  view; exactly what your engine automates.
