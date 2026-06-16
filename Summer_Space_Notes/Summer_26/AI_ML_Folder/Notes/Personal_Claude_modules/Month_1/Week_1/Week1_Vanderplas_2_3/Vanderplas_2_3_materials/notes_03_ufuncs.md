---
# Notes 03 — Computation on Arrays: Universal Functions (Vanderplas Ch. 2.3)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary

CPython is slow on element-by-element loops because its dynamic typing forces type-checking and function dispatch on every iteration — the cost is the per-element overhead, not the arithmetic itself. NumPy fixes this with **vectorized operations** implemented as **universal functions (ufuncs)**: a single statement (e.g. `1.0 / values`) pushes the loop down into a compiled layer, executing the same computation hundreds of times faster.

A naive Python-loop reciprocal over 1 million values takes ~2.91 s; the vectorized `1.0 / values` takes ~4.6 ms — over 600x faster. The speedup grows with array size.

ufuncs come in two flavors: **unary** (one input, e.g. `np.negative`) and **binary** (two inputs, e.g. `np.add`). They work on scalar-and-array, array-and-array, and multi-dimensional arrays alike.

## Key ideas / idioms

- Vectorized expression replaces an explicit loop; the loop runs in compiled code.
- All standard Python operators are wrapped as ufuncs and obey normal precedence / can be chained.
- Operators ↔ ufuncs:

| Operator | ufunc | Operation |
|----------|-------|-----------|
| `+` | `np.add` | addition |
| `-` | `np.subtract` | subtraction |
| `-` (unary) | `np.negative` | negation |
| `*` | `np.multiply` | multiplication |
| `/` | `np.divide` | division |
| `//` | `np.floor_divide` | floor division |
| `**` | `np.power` | exponentiation |
| `%` | `np.mod` | modulus |

- Math families: `np.abs`/`np.absolute`; trig (`np.sin/cos/tan`, `np.arcsin/arccos/arctan`); exponents (`np.exp`, `np.exp2`, `np.power`); logs (`np.log`, `np.log2`, `np.log10`); precise small-value variants `np.expm1`, `np.log1p`.
- For complex input, `np.abs` returns the magnitude $|a+bi| = \sqrt{a^2+b^2}$.
- Advanced ufunc methods: `out=` (write in place), `.reduce()` (collapse to one value), `.accumulate()` (keep intermediates), `.outer()` (all input pairs).
- `expm1`/`log1p` give better precision near zero:
$$\texttt{expm1}(x) = e^x - 1, \qquad \texttt{log1p}(x) = \ln(1+x)$$

## Worked code examples (runnable)

```python
import numpy as np

# --- Arithmetic operators and their ufunc equivalents ---
x = np.arange(4)            # [0 1 2 3]
print(x + 5)                # [5 6 7 8]
print(x * 2)                # [0 2 4 6]
print(x ** 2)               # [0 1 4 9]
print(-(0.5 * x + 1) ** 2)  # [-1. -2.25 -4. -6.25]
print(np.add(x, 5))         # same as x + 5
print(np.multiply(x, 2))    # same as x * 2
```

```python
import numpy as np

# --- Absolute value (real and complex) ---
x = np.array([-2, -1, 0, 1, 2])
print(abs(x))          # [2 1 0 1 2]
print(np.abs(x))       # [2 1 0 1 2]  (np.absolute is the same)

z = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
print(np.abs(z))       # [5. 5. 2. 1.]  (magnitudes)
```

```python
import numpy as np

# --- Trigonometric functions ---
theta = np.linspace(0, np.pi, 3)
print(np.sin(theta))   # [0. 1. ~0]
print(np.cos(theta))   # [1. ~0 -1.]
print(np.tan(theta))   # [0. ~large ~0]

xs = [-1, 0, 1]
print(np.arcsin(xs))   # [-1.5708  0.  1.5708]
print(np.arccos(xs))   # [3.1416  1.5708  0.]
print(np.arctan(xs))   # [-0.7854  0.  0.7854]
```

```python
import numpy as np

# --- Exponents and logarithms ---
x = [1, 2, 3]
print(np.exp(x))       # [2.718  7.389  20.086]
print(np.exp2(x))      # [2. 4. 8.]
print(np.power(3, x))  # [3 9 27]

x = [1, 2, 4, 10]
print(np.log(x))       # natural log
print(np.log2(x))      # [0. 1. 2. 3.3219]
print(np.log10(x))     # [0. 0.301 0.602 1.]

# Precise versions for tiny inputs
x = [0, 0.001, 0.01, 0.1]
print(np.expm1(x))     # e^x - 1
print(np.log1p(x))     # ln(1 + x)
```

```python
import numpy as np
from scipy import special

# --- Specialized ufuncs via scipy.special ---
x = [1, 5, 10]
print(special.gamma(x))     # [1. 24. 362880.]
print(special.gammaln(x))   # log of gamma
print(special.beta(x, 2))

x = np.array([0, 0.3, 0.7, 1.0])
print(special.erf(x))       # error function
print(special.erfc(x))      # complement
print(special.erfinv(x))    # inverse (last -> inf)
```

```python
import numpy as np

# --- Advanced: specifying output with out= ---
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)               # [0. 10. 20. 30. 40.]

y = np.zeros(10)
np.power(2, x, out=y[::2])   # write into a strided view
print(y)               # [1. 0. 2. 0. 4. 0. 8. 0. 16. 0.]
```

```python
import numpy as np

# --- Advanced: reduce, accumulate, outer ---
x = np.arange(1, 6)            # [1 2 3 4 5]
print(np.add.reduce(x))        # 15
print(np.multiply.reduce(x))   # 120
print(np.add.accumulate(x))    # [1 3 6 10 15]
print(np.multiply.accumulate(x))  # [1 2 6 24 120]

print(np.multiply.outer(x, x)) # 5x5 multiplication table
```

## Why this matters / intuition

The performance bottleneck in pure-Python numerics is overhead, not math: each loop iteration pays for dynamic type-checking and function dispatch. ufuncs move the loop into compiled C, so the per-element overhead vanishes and the CPU runs tight, cache-friendly code. The practical rule from the chapter: *when you see a loop over a NumPy array, ask whether it can be rewritten as a vectorized ufuncs expression* — it is nearly always faster, increasingly so as arrays grow.

The `out=` argument matters for large arrays because it avoids allocating a temporary result array, saving memory and a copy. *(added context)* This is the same in-place-write idea behind augmented assignment like `x += 1`.

## Gotchas

- Floating-point round-off: `np.sin(np.pi)` returns ~1.22e-16, not exactly 0 — values near machine precision are effectively zero.
- Use `np.expm1`/`np.log1p` instead of `np.exp`/`np.log` for very small inputs; the plain versions lose precision near zero.
- `out=` requires the target array to already exist with the right shape/dtype; it does not allocate for you. It also works with views (e.g. `y[::2]`), writing into strided memory.
- `scipy.special` is a separate import (`from scipy import special`), not part of NumPy.
- `.reduce()` and `.accumulate()` are ufunc *methods*; for common cases prefer the dedicated `np.sum`, `np.prod`, `np.cumsum`, `np.cumprod` (covered next section).

## Suggested figure (optional)

A two-bar (log-scale) timing chart contrasting the Python-loop reciprocal (~2.91 s) against the vectorized `1.0 / values` (~4.6 ms) over 1,000,000 elements, with the ~600x gap annotated — visually driving home that vectorization wins, and by how much.

---

## 💬 Q&A (captured during session)

### Q: What are ufuncs?

**Ufuncs (universal functions)** are NumPy's vectorized, element-wise operations — functions that apply the *same* operation to every element of an array (or pair of arrays) in one go, via fast precompiled C loops instead of a Python `for` loop.

**The problem they solve.** Python loops are slow because every iteration re-does type-checking and dispatch (the dynamic-typing overhead from Notes 01). For an array, that per-element overhead dominates:

```python
import numpy as np

big = np.random.randint(1, 100, size=1_000_000)

# Slow: Python-level loop, type-checked every element
def reciprocals_loop(arr):
    out = np.empty(len(arr))
    for i in range(len(arr)):
        out[i] = 1.0 / arr[i]
    return out

# Fast: one ufunc call, C loop under the hood
fast = 1.0 / big          # this is the ufunc np.divide
```

Same result, but the ufunc version is typically tens to hundreds of times faster — the loop happens in C, once.

**Two flavors:**
- **Unary** — operate on one array: `np.abs`, `np.exp`, `np.log`, `np.sin`, `-x`
- **Binary** — operate on two: `np.add`, `np.multiply`, `np.power`, `x > y`

**Key idea: operators *are* ufuncs.** The arithmetic you already write is just sugar for ufuncs:

```python
x = np.arange(4)        # [0 1 2 3]
x + 5                   # np.add(x, 5)        -> [5 6 7 8]
x ** 2                  # np.power(x, 2)      -> [0 1 4 9]
x % 2                   # np.mod(x, 2)        -> [0 1 0 1]
```

**Handy advanced features:**

```python
x = np.arange(1, 6)               # [1 2 3 4 5]

out = np.empty(5)
np.multiply(x, 10, out=out)       # write straight into out, no temp array

np.add.reduce(x)                  # 15  (sum all -> like np.sum)
np.add.accumulate(x)              # [ 1  3  6 10 15]  (running sum)
np.multiply.outer(x, x)           # full multiplication table (5x5)
```

**Mental model:** whenever you're tempted to loop over a NumPy array, stop — there's almost always a ufunc (or combination) that does it element-wise in C. *"Vectorize, don't iterate"* is the core NumPy idiom, and ufuncs are the machinery behind it. They also obey **broadcasting** (Notes 05).

### Q: What are wrappers?

A **wrapper** is a thin piece of code that goes *around* another piece of code to provide a more convenient or different interface to it — without changing what the underlying thing actually does. It "wraps" the real work and hands off to it.

This came up above: when `x + 5` is "sugar for `np.add(x, 5)`", the `+` operator is acting as a **wrapper** around the `np.add` ufunc.

**The idea — a wrapper:**
1. Takes a call in a simple/familiar form,
2. maybe does a little setup or translation,
3. calls the real ("wrapped") function that does the heavy lifting,
4. hands back the result.

You get an easier interface; the actual computation is unchanged.

**NumPy example.** NumPy makes the `+`, `-`, `**`, `%`, `>` operators *wrap* the corresponding ufuncs so you can write natural math instead of verbose calls:

```python
import numpy as np

x = np.arange(4)

x + 5            # convenient wrapper...
np.add(x, 5)     # ...around this actual ufunc — identical result

x ** 2           # wrapper around np.power(x, 2)
-x               # wrapper around np.negative(x)
```

*(added context — general programming concept, not from the book)* A wrapper in plain Python, the pattern behind **decorators**:

```python
import time

def slow_add(a, b):
    time.sleep(0.1)
    return a + b

# A "timing wrapper" around slow_add
def timed(func):
    def wrapper(*args, **kwargs):          # same call signature
        start = time.perf_counter()
        result = func(*args, **kwargs)     # call the wrapped function
        print(f"took {time.perf_counter() - start:.3f}s")
        return result                      # hand back its result
    return wrapper

fast_to_call = timed(slow_add)
fast_to_call(2, 3)     # prints timing, still returns 5
```

Here `wrapper` adds timing *around* `slow_add` but doesn't change the addition itself.

**Why it matters.** Wrappers are everywhere in data-science code:
- **Operators wrapping ufuncs** → readable math.
- **Pandas methods wrapping NumPy** — e.g. `df.sum()` wraps `np.sum` but adds index/label handling (Notes 12).
- **Convenience constructors** — `pd.read_csv` wraps a lot of parsing machinery behind one call.

The mental model: *a wrapper changes how you call something, not what it ultimately does.*
