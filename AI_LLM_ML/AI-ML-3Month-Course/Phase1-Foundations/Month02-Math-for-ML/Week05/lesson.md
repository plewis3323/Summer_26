# Week 05 — Calculus for ML

~3 hrs reading, computing along as you go. Before starting you should be able to:
write and run a Python function with a `for` loop (Week 02); build NumPy arrays and
do arithmetic on them (Week 03); make a labeled matplotlib plot (Week 03).

Machine learning, at its core, is this: write down a number that measures how wrong
your model is, then adjust the model to make that number smaller. The mathematics of
"which way is smaller" is calculus. This week builds it from scratch — no calculus
background assumed — and ends with you implementing the adjustment algorithm, gradient
descent, that trains essentially every model in this course.

## 1. Functions and their graphs

A **function** is a rule that takes in a number and returns a number. Write it as
$f(x)$: "$f$ of $x$". The rule $f(x) = x^2$ takes 3 and returns 9, takes $-2$ and
returns 4. The input has a name, $x$; the output is whatever the rule produces.

In Python you already know this object:

```python
def f(x):
    return x**2

print(f(3))    # 9
print(f(-2))   # 4
```

The **graph** of a function is the picture you get by plotting input on the
horizontal axis and output on the vertical axis:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)   # 200 evenly spaced inputs from -3 to 3
plt.plot(x, f(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = x^2")
plt.show()
```

Functions are everywhere in physics and in ML:

- A **calibration curve**: a particle detector converts deposited energy into an
  electronic pulse. (A particle passing through matter loses energy; the detector
  turns that energy into an electrical signal you can record.) Pulse height is a
  function of energy.
- A **model**: "predicted house price as a function of floor area" is a function.
- A **loss function**: "how wrong the model is, as a function of the model's
  parameters". This one is the star of the course. Making it small is training.

## 2. Slope of a line

Start with the simplest function: a straight line, $f(x) = mx + b$. You know this
from school. $b$ is where the line crosses the vertical axis. $m$ is the **slope**:

$$m = \frac{\text{rise}}{\text{run}} = \frac{\text{change in output}}{\text{change in input}}.$$

If you move 1 unit right, the line moves $m$ units up. Slope is a *sensitivity*: it
tells you how strongly the output responds to the input. A calibration line with
slope 5 (pulse units per unit energy) means each extra unit of energy adds 5 units
of pulse height.

Two facts about lines that matter all week:

1. The slope is the *same everywhere* on the line.
2. If you know the slope and one point, you know the whole line.

## 3. Slope of a curve — the derivative

Now look at $f(x) = x^2$. What is its slope? The question seems broken: near
$x = 0$ the graph is almost flat, near $x = 3$ it climbs steeply. A curve does not
have *a* slope — it has a slope *at each point*. Our job is to compute it.

### 3.1 Secant lines: rise over run between two nearby points

Pick a point, say $x = 3$. Pick a second point a small step $h$ to the right,
at $x = 3 + h$. Draw the line through the two graph points — a **secant line** —
and compute its slope the high-school way:

$$\text{slope of secant} = \frac{f(3+h) - f(3)}{h}.$$

This fraction is called a **difference quotient**: change in output over change
in input. Compute it in Python for shrinking $h$:

```python
def f(x):
    return x**2

x0 = 3.0
for h in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    slope = (f(x0 + h) - f(x0)) / h
    print(h, slope)
```

Output:

```
1.0     7.000000000000
0.1     6.100000000000
0.01    6.009999999999
0.001   6.000999999999
0.0001  6.000100000001
```

The slopes settle down toward **6**. As the second point slides toward the first,
the secant line tilts toward the line that just grazes the curve at $x = 3$ —
the **tangent line** — and 6 is its slope.

### 3.2 The limit, and the definition

"Settles down toward 6 as $h$ shrinks" has a name: the **limit** as $h \to 0$.
We will use limits informally — a limit is the value a quantity approaches, even
if you never plug in $h = 0$ itself (you can't: $0/0$ is meaningless).

The **derivative** of $f$ at $x$ is the limiting slope:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.$$

Read $f'(x)$ as "$f$ prime of $x$": the slope of the graph of $f$ at the point $x$.
Another common notation is $\frac{df}{dx}$, which you can read as "an infinitesimally
small change in $f$ per infinitesimally small change in $x$" — the same rise-over-run
idea. The derivative is itself a function: feed it a location, get back the slope there.

### 3.3 First principles, by algebra

For $f(x) = x^2$ we can do the limit exactly, with nothing but algebra:

$$\frac{f(x+h) - f(x)}{h} = \frac{(x+h)^2 - x^2}{h}
= \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h.$$

As $h \to 0$, this approaches $2x$. So:

$$f(x) = x^2 \quad\Rightarrow\quad f'(x) = 2x.$$

Check it against the table: at $x = 3$, $f'(3) = 6$. Exactly what the numbers
crept toward. This is the whole game: the *numerical* difference quotient
approximates; the *algebraic* limit is exact.

### 3.4 A better numerical derivative

Computers can't take $h$ all the way to zero (rounding error takes over — try
$h = 10^{-15}$ in the table above and watch it fall apart). A good practical
recipe is the **central difference**, which straddles the point symmetrically:

```python
def num_deriv(f, x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)

print(num_deriv(f, 3.0))   # 5.999999999...
```

Keep `num_deriv`. All month — and again when you build backpropagation in Phase 2 —
you will use it to *check* every derivative you derive on paper. A derivative that
passes a numerical check is a derivative you can trust.

## 4. Rules: power, product, chain

Working the limit from first principles every time is slow. A handful of rules,
each derived once, covers everything we need.

### 4.1 Power rule

Run the first-principles computation on $x^3$:
$(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$, so the difference quotient is
$3x^2 + 3xh + h^2 \to 3x^2$. The pattern from $x^2 \to 2x$ and $x^3 \to 3x^2$
continues for every power:

$$\frac{d}{dx} x^n = n\,x^{n-1}.$$

It also holds for negative and fractional $n$ (so $\frac{d}{dx}\frac{1}{x} =
\frac{d}{dx}x^{-1} = -x^{-2}$, and $\frac{d}{dx}\sqrt{x} = \tfrac{1}{2}x^{-1/2}$).

Two companions, both provable straight from the definition:

- **Constant multiple:** $\frac{d}{dx}\left[c\,f(x)\right] = c\,f'(x)$.
- **Sum rule:** $\frac{d}{dx}\left[f(x) + g(x)\right] = f'(x) + g'(x)$.

Together these differentiate any polynomial. Example:
$\frac{d}{dx}(3x^4 - 2x + 7) = 12x^3 - 2$ (the derivative of a constant is 0:
a flat line has slope 0).

### 4.2 Product rule

For a product $f(x)g(x)$, the derivative is *not* $f'g'$. Derive the right answer
from the definition, using the add-and-subtract trick:

$$\frac{f(x+h)g(x+h) - f(x)g(x)}{h}
= \frac{f(x+h)g(x+h) - f(x)g(x+h) + f(x)g(x+h) - f(x)g(x)}{h}$$

$$= \frac{f(x+h) - f(x)}{h}\,g(x+h) \;+\; f(x)\,\frac{g(x+h) - g(x)}{h}.$$

As $h \to 0$, the first fraction becomes $f'(x)$, $g(x+h)$ becomes $g(x)$, and the
last fraction becomes $g'(x)$:

$$\frac{d}{dx}\left[f(x)g(x)\right] = f'(x)\,g(x) + f(x)\,g'(x).$$

Picture: a rectangle with side lengths $f$ and $g$. Grow both sides a little; the
new area comes from two thin strips — one of size (change in $f$) × $g$, one of
size $f$ × (change in $g$). The tiny corner piece vanishes in the limit.

### 4.3 Chain rule — read this section twice

The chain rule handles **composition**: one function applied to the output of
another. $f(x) = (3x + 1)^2$ is a composition: first compute the inner value
$u = 3x + 1$, then apply the outer rule $u^2$.

Here is the rule, and then why it is true:

$$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx}
\qquad\text{— rates multiply along a chain.}$$

**Why.** A derivative is a sensitivity: if $x$ changes by a small amount
$\Delta x$, then $u$ changes by approximately $\Delta u \approx \frac{du}{dx}\,\Delta x$
(that is what slope *means*: output change ≈ slope × input change). Feed that change
into the outer function: $\Delta f \approx \frac{df}{du}\,\Delta u$. Substitute one
into the other:

$$\Delta f \approx \frac{df}{du}\cdot\frac{du}{dx}\,\Delta x.$$

The sensitivity of $f$ to $x$ is the product of the sensitivities along the chain.
If $u$ moves 3 times as fast as $x$, and $f$ moves $2u$ times as fast as $u$, then
$f$ moves $2u \cdot 3$ times as fast as $x$.

**Worked:** $f(x) = (3x+1)^2$. Inner: $u = 3x + 1$, so $\frac{du}{dx} = 3$.
Outer: $f = u^2$, so $\frac{df}{du} = 2u$. Chain:

$$f'(x) = 2u \cdot 3 = 6(3x + 1).$$

Check numerically at $x = 2$: the formula says $6 \times 7 = 42$;
`num_deriv(lambda...)` — no, we don't use `lambda` in this course; define it plainly:

```python
def f(x):
    return (3*x + 1)**2

print(num_deriv(f, 2.0))   # 42.00000000...
```

**Why this rule is the spine of backprop.** A neural network is a long chain of
simple functions: the input goes through layer 1, its output through layer 2, and
so on, ending in a single loss number. To train the network you need the derivative
of the loss with respect to every parameter — and every one of those derivatives is
a product of local derivatives along the chain, by exactly this rule.
Backpropagation (Week 13) is nothing but the chain rule, organized so no product is
computed twice. If the chain rule feels solid, backprop will feel inevitable.

Chains longer than two links work the same way — multiply all the local rates:
for $f(g(h(x)))$, $\;\frac{df}{dx} = \frac{df}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}$.

## 5. Functions of several variables

A model never has one parameter. A line fit has two ($m$ and $b$); the networks
in Phase 3 have billions. So we need derivatives of functions of several inputs.

$f(x, y) = x^2 + 3xy$ takes *two* numbers and returns one. Its graph is a surface
over the $xy$-plane; the practical picture is a **contour plot** — curves of equal
height, like a topographic map:

```python
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)          # grids of all (x, y) pairs
Z = X**2 + 3*X*Y
plt.contour(X, Y, Z, levels=20)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

### 5.1 Partial derivatives

Ask the slope question one input at a time. The **partial derivative** of $f$ with
respect to $x$, written $\frac{\partial f}{\partial x}$ (curly $\partial$, read
"partial f partial x"), is: *hold every other input fixed, treat $f$ as a function
of $x$ alone, differentiate as usual.*

For $f(x, y) = x^2 + 3xy$:

- $\frac{\partial f}{\partial x} = 2x + 3y$  (treat $y$ as a constant — the way $3$
  is a constant in $3x$)
- $\frac{\partial f}{\partial y} = 3x$  (now $x$ is the constant)

Numerically, nudge one input and leave the other alone:

```python
def f2(x, y):
    return x**2 + 3*x*y

h = 1e-5
x0, y0 = 1.0, 2.0
dfdx = (f2(x0 + h, y0) - f2(x0 - h, y0)) / (2 * h)   # expect 2*1 + 3*2 = 8
dfdy = (f2(x0, y0 + h) - f2(x0, y0 - h)) / (2 * h)   # expect 3*1 = 3
print(dfdx, dfdy)
```

### 5.2 The gradient

Collect the partials into a vector (for now, "vector" just means an ordered list of
numbers — Week 06 gives it the full geometric treatment):

$$\nabla f = \left(\frac{\partial f}{\partial x},\; \frac{\partial f}{\partial y}\right).$$

The symbol $\nabla$ is read "grad" or "nabla". At the point $(1, 2)$ above,
$\nabla f = (8, 3)$.

What does the gradient *mean*? Suppose you stand at $(x, y)$ and take a small step:
$x$ changes by $\Delta x$ and $y$ by $\Delta y$, both at once. Each input contributes
its own (slope × change), and for small steps the contributions simply add — this is
the several-variable chain rule:

$$\Delta f \approx \frac{\partial f}{\partial x}\,\Delta x
             + \frac{\partial f}{\partial y}\,\Delta y.$$

(Week 06 will name this combination — multiply matching entries, add them up — the
*dot product*.)

**Claim: the gradient points in the direction of steepest ascent.** Among all
possible unit-length steps, the one that increases $f$ fastest is the step in the
direction of $\nabla f$; the step that *decreases* $f$ fastest is exactly opposite,
along $-\nabla f$. Rather than prove it (Week 06's dot-product geometry makes the
proof one line), verify it numerically. A step of length 1 in direction $\theta$ is
$(\cos\theta, \sin\theta)$ — recall from school trig that the point at angle
$\theta$ on the unit circle has those coordinates:

```python
best_angle = None
best_rate = -1e9
for theta in np.linspace(0, 2*np.pi, 721):
    rate = dfdx * np.cos(theta) + dfdy * np.sin(theta)
    if rate > best_rate:
        best_rate = rate
        best_angle = theta

grad_angle = np.arctan2(dfdy, dfdx)   # the angle the vector (dfdx, dfdy) points at
print(best_angle, grad_angle)         # the same angle
```

The best angle found by brute force is the gradient's own direction. Remember the
sentence, because the rest of the course leans on it: **the gradient points uphill;
minus the gradient points downhill fastest.**

## 6. Finding minima

Training a model means finding parameter values that make the loss as small as
possible — finding the **minimum** of a function. Calculus offers two routes.

### 6.1 Route 1: set the derivative to zero

At the bottom of a smooth valley the tangent line is flat: $f'(x) = 0$. Points where
the derivative vanishes are called **critical points**. So: differentiate, set to
zero, solve.

Example: $f(x) = x^2 - 4x + 7$. Then $f'(x) = 2x - 4 = 0$ gives $x = 2$, and
$f(2) = 3$. (Cross-check with the school formula for a parabola's vertex,
$x = -b/2a = 4/2 = 2$. Same answer — the vertex formula *is* this calculation,
done once and memorized.)

Caution: $f'(x) = 0$ also happens at maxima (top of a hill is flat too) and at
saddle-shaped points. The sign of the **second derivative** $f''(x)$ — the
derivative of the derivative, which measures how the slope itself is changing —
distinguishes them: $f'' > 0$ means the slope is increasing through zero, a valley;
$f'' < 0$ means a hill.

With several variables, a minimum needs *every* partial derivative to vanish:
$\nabla f = 0$.

### 6.2 Route 2: follow the negative gradient

Route 1 needs you to *solve* $\nabla f = 0$ by algebra. For a quadratic, fine. For
a loss function defined through a million-parameter model, there is nothing to
solve by hand. Route 2 gives up on solving and just walks downhill:

> Stand somewhere. Compute the gradient. Take a small step in the direction
> $-\nabla f$ (downhill fastest). Repeat.

This is **gradient descent** — the algorithm that trains essentially everything in
this course. As an update rule, with $\eta$ (Greek "eta") the step size, called the
**learning rate**:

$$x_{\text{new}} = x_{\text{old}} - \eta\, f'(x_{\text{old}}).$$

## 7. Gradient descent in code

### 7.1 One dimension

Minimize $f(x) = (x - 3)^2 + 1$. We know the answer (a parabola with vertex at
$x = 3$, minimum value 1), which is exactly why it makes a good first test.
$f'(x) = 2(x - 3)$ by the chain rule.

```python
def f(x):
    return (x - 3)**2 + 1

def dfdx(x):
    return 2 * (x - 3)

x = 0.0          # start anywhere
eta = 0.1        # learning rate
history = [x]
for step in range(40):
    x = x - eta * dfdx(x)
    history.append(x)

print(x)         # 2.9996... -> converging to 3
```

Plot `history` and you'll see $x$ glide toward 3, fast at first (steep slope, big
gradient, big step) and slowing as the valley flattens.

Now break it, deliberately — the learning rate is the knob that matters:

- `eta = 0.01`: still converges, just slowly. Safe but wasteful.
- `eta = 0.9`: overshoots the minimum each step, landing on the far wall, but each
  overshoot is smaller — it converges by ricochet.
- `eta = 1.1`: each overshoot is *bigger* than the last. Diverges to infinity.

For this parabola the exact boundary is $\eta = 1$: the update is
$x_{\text{new}} - 3 = (1 - 2\eta)(x_{\text{old}} - 3)$, so the distance to the
minimum shrinks only if $|1 - 2\eta| < 1$, i.e. $0 < \eta < 1$. You will rediscover
this stability boundary experimentally in the exercises, and derive its general form
in the Week 08 project.

### 7.2 Two dimensions

Minimize $f(x, y) = x^2 + 10y^2$ — a valley that is 10 times steeper in $y$ than
in $x$. Gradient: $\nabla f = (2x, 20y)$.

```python
def grad(x, y):
    return 2*x, 20*y

x, y = 8.0, 1.0
eta = 0.09
xs = [x]
ys = [y]
for step in range(60):
    gx, gy = grad(x, y)
    x = x - eta * gx
    y = y - eta * gy
    xs.append(x)
    ys.append(y)

print(x, y)   # both near 0
```

Overlay the path on the contour plot:

```python
gx_ = np.linspace(-9, 9, 100)
gy_ = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(gx_, gy_)
plt.contour(X, Y, X**2 + 10*Y**2, levels=25)
plt.plot(xs, ys, marker="o", markersize=3, color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Look at the shape of the path: it zig-zags across the steep $y$-direction while
crawling along the shallow $x$-direction. The single learning rate is too timid for
$x$ and too bold for $y$ at the same time. This mismatch — some directions steep,
some shallow — is the central *practical* problem of optimization, and fixing it is
the whole point of the Week 08 mini-project (momentum, RMSProp, Adam).

## 8. Worked example: fitting a line by gradient descent

Time to do something real: fit a model to data by gradient descent, end to end.

**The physics.** Some atomic nuclei are radioactive and emit gamma rays — flashes
of light-like radiation — at *exactly known* energies, the same everywhere in the
universe. Point such sources at a detector and you know precisely how much energy
went in; record the electronic pulse height that came out. If the detector is
linear, pulse height $y$ relates to energy $E$ as $y \approx mE + b$, and finding
$m$ and $b$ is called **calibrating** the detector.

**The data** (simulated, so we know the right answer — true $m = 5.0$, $b = 2.0$):

```python
import numpy as np

rng = np.random.default_rng(42)
E = np.array([0.51, 0.66, 0.84, 1.17, 1.27, 1.33])   # known gamma energies (MeV)
y = 5.0 * E + 2.0 + rng.normal(0, 0.1, size=E.size)  # measured pulses, with noise
```

**The loss.** Measure how wrong a candidate $(m, b)$ is by the **mean squared
error** — the average of the squared miss at each point:

$$L(m, b) = \frac{1}{N}\sum_{i=1}^{N} \big(mE_i + b - y_i\big)^2,$$

where $N$ is the number of data points and the sum runs over them. Squaring makes
every miss positive (misses can't cancel) and punishes big misses hardest. $L$ is a
function of *two variables* — the parameters $m$ and $b$. The data are constants.
Training is minimizing $L$.

**The gradient, by the chain rule.** Write $r_i = mE_i + b - y_i$ (the $i$-th
**residual**, the signed miss). Then $L = \frac{1}{N}\sum r_i^2$. Differentiate with
respect to $m$: the outer function is $r^2$ (derivative $2r$), the inner is
$r_i = mE_i + b - y_i$ (derivative with respect to $m$: $E_i$). Chain rule, then sum:

$$\frac{\partial L}{\partial m} = \frac{1}{N}\sum_i 2\,r_i\,E_i,
\qquad
\frac{\partial L}{\partial b} = \frac{1}{N}\sum_i 2\,r_i\,.$$

(For $b$, the inner derivative is 1.) That's the entire derivation — one chain rule
and one sum.

**The descent.**

```python
m, b = 0.0, 0.0                 # start ignorant
eta = 0.1
for step in range(2000):
    r = m * E + b - y           # residuals, all points at once (Week 03!)
    dm = 2 * np.mean(r * E)
    db = 2 * np.mean(r)
    m = m - eta * dm
    b = b - eta * db

print(m, b)                     # ~4.98, ~2.03  (true: 5.0, 2.0)
loss = np.mean((m * E + b - y)**2)
print(loss)
```

The fit lands within noise of the true values. Plot data plus the fitted line, and
also plot $L$ against the step number: the loss curve should fall steeply, then
flatten. That falling curve is the picture you will stare at for the rest of this
course — every neural network you ever train produces one.

Two things to notice before moving on:

1. Nothing here was specific to lines. Swap in any model with any parameters: as
   long as you can compute $\partial L/\partial(\text{each parameter})$, the same
   loop trains it. That is why this algorithm, plus the chain rule to get the
   partials, runs all of deep learning.
2. We *could* have solved this one exactly with Route 1 ($\nabla L = 0$ gives two
   linear equations — Week 06 shows how to solve those systematically, and Week 08
   derives why squared error was the right loss in the first place). Gradient
   descent matters for all the problems where Route 1 is impossible.

## Check yourself

1. In your own words: what does $f'(4) = -2$ tell you about the graph of $f$ near
   $x = 4$?
2. Use first principles (the limit of the difference quotient) to find $f'(x)$ for
   $f(x) = 5x + 1$. Does the answer surprise you?
3. Differentiate $f(x) = 2x^5 - x^2 + 6$.
4. Differentiate $f(x) = (x^2 + 1)^3$ using the chain rule, naming the inner and
   outer functions.
5. For $f(x, y) = x^2 y + y^3$: compute both partial derivatives, and the gradient
   at the point $(2, 1)$.
6. You are minimizing a function and the gradient at your current point is
   $(0.4, -1.2)$. In which direction should you step, and why that one?
7. In the 1D gradient descent of §7.1, what goes wrong with $\eta = 1.1$,
   mechanically — what does each update do to the distance from the minimum?
8. In the worked example, why does the formula for $\partial L/\partial m$ have an
   extra factor $E_i$ that $\partial L/\partial b$ lacks?

## Answers

1. Near $x = 4$ the graph slopes *downward*, falling about 2 units of output per
   unit of input; a small increase $\Delta x$ changes $f$ by about $-2\,\Delta x$.
2. $\frac{(5(x+h)+1) - (5x+1)}{h} = \frac{5h}{h} = 5$ for every $h$, so
   $f'(x) = 5$. No surprise: the function is a line, and its slope is 5 everywhere —
   the derivative recovers exactly the school notion of slope.
3. $f'(x) = 10x^4 - 2x$ (power rule term by term; the constant 6 contributes 0).
4. Inner $u = x^2 + 1$ with $du/dx = 2x$; outer $u^3$ with derivative $3u^2$.
   Chain rule: $f'(x) = 3(x^2+1)^2 \cdot 2x = 6x(x^2+1)^2$.
5. $\partial f/\partial x = 2xy$; $\partial f/\partial y = x^2 + 3y^2$. At $(2,1)$:
   $\nabla f = (4, 7)$.
6. Step along $-(0.4, -1.2) = (-0.4, 1.2)$ (suitably scaled): the negative gradient
   is the direction of steepest *descent*, and you want the function to decrease.
7. The update multiplies the distance to the minimum by $1 - 2\eta = -1.2$: sign
   flips (overshoot) and magnitude *grows* by 20% each step, so the iterates bounce
   across the minimum with ever-larger amplitude — divergence.
8. Chain rule: the inner function $r_i = mE_i + b - y_i$ has derivative $E_i$ with
   respect to $m$ but derivative 1 with respect to $b$. The inner derivative rides
   along as a factor.

## New terms

- **function** — a rule assigning one output number to each input.
- **graph** — the plot of a function's outputs against its inputs.
- **slope** — change in output per unit change in input; for a line, constant.
- **secant line** — the line through two points on a graph.
- **difference quotient** — $(f(x+h)-f(x))/h$, the secant's slope.
- **tangent line** — the line that grazes a curve at a point; its slope is the derivative.
- **limit** — the value a quantity approaches as a parameter (here $h$) shrinks to zero.
- **derivative** $f'(x)$, $\frac{df}{dx}$ — the slope of $f$ at $x$; the limit of the difference quotient.
- **central difference** — numerical derivative estimate $(f(x+h)-f(x-h))/(2h)$.
- **power rule / product rule / chain rule** — derivative rules for powers, products, and compositions.
- **composition** — applying one function to the output of another.
- **partial derivative** $\partial f/\partial x$ — derivative with respect to one input, others held fixed.
- **gradient** $\nabla f$ — the vector of all partial derivatives; points in the direction of steepest ascent.
- **contour plot** — map of a two-input function by curves of equal output.
- **critical point** — a point where the derivative (or every partial) is zero.
- **second derivative** — derivative of the derivative; its sign separates valleys from hills.
- **gradient descent** — repeated small steps along $-\nabla f$ to find a minimum.
- **learning rate** $\eta$ — the step-size factor in gradient descent.
- **loss function** — a function measuring model wrongness as a function of its parameters.
- **mean squared error** — average of squared residuals; this week's loss.
- **residual** — the signed miss, prediction minus observation, for one data point.

## Going deeper

- 3Blue1Brown, *Essence of Calculus* (all chapters, free on YouTube) — the week's
  spine. Animated geometry for everything in §§2–4; watch chapters 1–4 alongside
  §§2–4 here, and the chain-rule chapter twice.
- Khan Academy, Algebra II / Precalculus units — if §§1–2 felt fast, spend a weekend
  here first. Normal, and it costs nothing but time.
