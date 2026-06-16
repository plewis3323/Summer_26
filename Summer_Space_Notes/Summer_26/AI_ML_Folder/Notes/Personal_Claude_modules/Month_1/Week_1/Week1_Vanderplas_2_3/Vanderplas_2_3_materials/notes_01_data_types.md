---
# Notes 01 — Understanding Data Types in Python (Vanderplas Ch. 2.1)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Effective data-heavy computing requires understanding *how* Python stores and manipulates data. Python's flexibility (dynamic typing) comes at a cost: every value is a full object with bookkeeping overhead, and standard containers (`list`) store pointers to scattered objects. NumPy fixes this by providing **fixed-type, contiguous arrays** that trade flexibility for the efficiency needed for numerical work. This section explains the trade-off and shows the many ways to build NumPy arrays.

## Key ideas / idioms
- **Dynamic typing:** a Python variable can hold any type and change type freely (`x = 4`, then `x = "four"` is fine). In a statically-typed language like C you must declare the type up front and it cannot change.
- **A Python integer is not just an integer.** It is a pointer to a C structure (`struct _longobject`) holding:
  - `ob_refcnt` — reference count (for memory management)
  - `ob_type` — the variable's type
  - `ob_size` — size of the following data members
  - `ob_digit` — the actual integer value
  This makes Python ints much larger than a raw C int (which is just bytes in memory).
- **A Python list is not just a list.** Because each element is a full Python object, a list stores a pointer to a block of pointers, each pointing to its own object. This allows **heterogeneous** lists but adds memory + indirection overhead.
- **Fixed-type arrays** store a single pointer to one **contiguous block** of data of one type. Much more efficient, but homogeneous (all same type).
- Python ships a built-in `array` module for compact same-type arrays, but **NumPy's `ndarray`** adds efficient *operations* on that data — that's why we use NumPy.
- When making an array from a list, **types are upcast** to a common type if possible (e.g., an int mixed with floats becomes all floats).
- You can force a type with the `dtype` keyword.
- NumPy arrays can be **multidimensional** (nested lists become 2D, etc.).
- Types can be specified as strings (`dtype='int16'`) or NumPy objects (`dtype=np.int16`).
- **No math required for this section**; conceptually, list vs array overhead scales as $$O(n)$$ extra pointers/objects for a list of length $n$, vs a single contiguous buffer for an array. *(added context: the big-O framing is mine, not in the text.)*

## Worked code examples (runnable)
```python
import numpy as np

# --- Dynamic typing (pure Python) ---
x = 4
x = "four"   # legal in Python; would be a type error in C

# --- Built-in array module (same-type, compact) ---
import array
L = list(range(10))
A = array.array('i', L)   # 'i' => integer type code
print(A)

# --- Creating NumPy arrays from Python lists ---
print(np.array([1, 4, 2, 5, 3]))

# Integers get upcast to float when mixed with floats:
print(np.array([3.14, 4, 2, 3]))

# Force a data type explicitly:
print(np.array([1, 2, 3, 4], dtype='float32'))

# Nested lists -> multidimensional array:
print(np.array([range(i, i + 3) for i in [2, 4, 6]]))

# --- Creating arrays from scratch ---
print(np.zeros(10, dtype=int))                 # length-10 array of zeros
print(np.ones((3, 5), dtype=float))            # 3x5 array of ones
print(np.full((3, 5), 3.14))                   # 3x5 array filled with 3.14
print(np.arange(0, 20, 2))                     # like range(): start, stop, step
print(np.linspace(0, 1, 5))                    # 5 values evenly spaced in [0, 1]
print(np.random.random((3, 3)))                # uniform random in [0, 1)
print(np.random.normal(0, 1, (3, 3)))          # normal(mean=0, std=1)
print(np.random.randint(0, 10, (3, 3)))        # random ints in [0, 10)
print(np.eye(3))                               # 3x3 identity matrix
print(np.empty(3))                             # uninitialized (whatever is in memory)

# --- Specifying NumPy data types ---
print(np.zeros(5, dtype='int16'))              # via string
print(np.zeros(5, dtype=np.int16))             # via NumPy object
```

## Why this matters / intuition
- For data science you operate on *millions* of values; the per-object overhead of Python lists makes element-wise math slow and memory-hungry.
- A contiguous, single-type buffer is what lets NumPy push work down to fast compiled (C/Fortran) loops — this is the foundation everything later in the book (vectorization, broadcasting, pandas) is built on.
- Knowing `dtype` exists matters for **memory** (e.g., `int8` vs `int64`) and for avoiding silent precision surprises.

## Gotchas
- **Upcasting is silent:** `np.array([3.14, 4, 2, 3])` quietly becomes all floats. If you expected ints, set `dtype` explicitly.
- **NumPy arrays are homogeneous.** Unlike a Python list, you can't freely mix types in one array.
- **`np.empty` is NOT zeros.** It returns whatever happens to already be in that memory — never assume initial values.
- **`np.arange` stop is exclusive**, and with float steps it can hit floating-point edge cases; prefer `np.linspace` when you need an exact number of evenly spaced points.
- A NumPy array indexed/sliced will coerce assigned values to the array's `dtype` (e.g., writing a float into an int array truncates). *(added context: implied by fixed-type storage; the explicit truncation example appears in the next sections.)*

## Suggested figure (optional)
Two side-by-side memory diagrams:
1. **Python list:** a contiguous block of pointers, each arrow leading off to a separately-allocated Python integer object (each object box showing `ob_refcnt / ob_type / ob_size / ob_digit`). Emphasizes scattered memory + double indirection.
2. **NumPy array:** a single header (dtype, shape, strides) pointing to one contiguous block of raw values packed back-to-back. Emphasizes one allocation, no per-element object overhead.
This is the classic "C int vs Python int / list vs array" layout that makes the efficiency argument visual.
