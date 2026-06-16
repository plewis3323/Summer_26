---
# Notes 09 — Structured Data: NumPy's Structured Arrays (Vanderplas Ch. 2.9)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Structured arrays let a single NumPy array hold **heterogeneous, compound records** — fields of different types (e.g., a string name, an int age, a float weight) packed together with named fields. They solve the problem of keeping related data spread across separate parallel arrays (where the relationship between entries is only implicit). For everyday work the book notes that **Pandas `DataFrame`s** are usually the better tool; structured arrays shine mainly for simple operations and for mapping onto C/Fortran binary layouts.

## Key ideas / idioms
- A structured array is defined by a compound **`dtype`** listing field names and per-field formats.
- Several equivalent ways to specify the dtype: dictionary of `names`/`formats`, list of `(name, format)` tuples, or a comma-separated string (which auto-names fields `f0, f1, ...`).
- Format codes combine a **type character** + **byte size** (and optional endianness prefix `<`/`>`):
  - `'U10'` = Unicode string up to 10 chars; `'i4'` = 32-bit int; `'f8'` = 64-bit float; `'S10'` = byte string.
- Access patterns:
  - by **field name**: `data['name']` returns that column as an array;
  - by **index**: `data[0]` returns one record (a tuple-like row);
  - **chained**: `data[-1]['name']`;
  - **boolean masking on a field**: `data[data['age'] < 30]['name']`.
- Fields can themselves be sub-arrays: e.g. a `(3, 3)` matrix per record — this maps directly onto C `struct` definitions.
- **`np.recarray`** is a view that adds **attribute-style** access (`data_rec.age`) at the cost of speed.

Format-code reference table from the section:

| Char | Meaning | Example |
|------|---------|---------|
| `'b'` | Byte | `np.dtype('b')` |
| `'i'` | Signed integer | `np.dtype('i4') == np.int32` |
| `'u'` | Unsigned integer | `np.dtype('u1') == np.uint8` |
| `'f'` | Floating point | `np.dtype('f8') == np.float64` |
| `'c'` | Complex float | `np.dtype('c16') == np.complex128` |
| `'S'`, `'a'` | String | `np.dtype('S5')` |
| `'U'` | Unicode string | `np.dtype('U') == np.str_` |
| `'V'` | Raw (void) data | `np.dtype('V') == np.void` |

*(added context)* Per-field access is cheap because each field is a strided view into the same contiguous buffer — no copy is made.

## Worked code examples (runnable)

```python
import numpy as np

# The "before" problem: parallel lists with only implicit relationships
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# Create a structured array via the dictionary spec
data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                          'formats': ('U10', 'i4', 'f8')})
print(data.dtype)
# [('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]

# Fill the named fields
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# Access by field name (returns the whole column)
print(data['name'])

# Access a single record by index, and a field of it
print(data[0])
print(data[-1]['name'])

# Boolean filtering on a field: names of everyone under 30
print(data[data['age'] < 30]['name'])
```

```python
import numpy as np

# Equivalent dtype specifications
d1 = np.dtype({'names': ('name', 'age', 'weight'),
               'formats': ((np.str_, 10), int, np.float32)})  # Python types
d2 = np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])  # list of tuples
d3 = np.dtype('S10,i4,f8')  # comma string -> auto field names f0, f1, f2
print(d1)
print(d2)
print(d3)
```

```python
import numpy as np

# Compound type with a per-record sub-array (3x3 matrix)
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])          # (0, [[0,...]])  id plus a 3x3 block
print(X['mat'][0])   # the 3x3 matrix for record 0
```

```python
import numpy as np

data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                          'formats': ('U10', 'i4', 'f8')})
data['age'] = [25, 45, 37, 19]

# Record array: attribute-style access
data_rec = data.view(np.recarray)
print(data_rec.age)   # array([25, 45, 37, 19], dtype=int32)
```

## Q&A capture — "what *is* a structured array?" (plain version)

> *(My question during reading: just tell me what it is, simply.)*

A normal NumPy array is **homogeneous** — every slot holds **one value** of **one type**
(all ints, all floats). A **structured array** lets every slot instead hold a **record**: a
bundle of several **named fields**, each with its own type. Each element becomes a mini-row of
a table.

```python
data[0]        # ('Alice', 25, 55.0)  ← one record = labeled bundle of mixed types
data['name']   # whole 'name' column as an array
```

The job it does: instead of three fragile **parallel lists** (where "Alice/25/55.0" is linked
only by all sharing index 0, and sorting one breaks the link), it **fuses them into one
container** so each record stays glued together.

```
  normal array        structured array
  ┌────┐              ┌──────────────────────────────┐
  │ 25 │ one value    │ name='Alice' age=25 wt=55.0  │ one record
  │ 45 │ per slot     │ name='Bob'   age=45 wt=85.5  │ (named fields, mixed types)
  └────┘              └──────────────────────────────┘
```

The compound **`dtype`** is what makes it work — it names each field and its type+size
(`'U10'` = 10-char string, `'i4'` = 32-bit int, `'f8'` = 64-bit float), laid out back-to-back
in memory. **One line:** a structured array is a NumPy array whose every element is a labeled,
mixed-type record — a whole table packed into one contiguous buffer. (For daily work, reach for
a Pandas `DataFrame`; structured arrays earn their keep mainly for C/Fortran binary layouts.)

## Why this matters / intuition
Structured arrays give you a **single, self-describing container** for tabular/record data while keeping NumPy's contiguous-memory efficiency. Because the layout is explicit (named fields, fixed sizes, endianness), they are ideal for **interoperating with C/Fortran binary formats and legacy libraries** — the memory image lines up with a C `struct`. They are also the conceptual ancestor of the Pandas `DataFrame`, so understanding them clarifies how Pandas stores typed columns under the hood.

## Gotchas
- **`recarray` attribute access is slow.** The book's timings: `data['age']` ≈ 241 ns; `data_rec['age']` ≈ 4.61 µs; `data_rec.age` ≈ 7.27 µs. The attribute convenience adds real overhead — prefer dict-key access in hot loops.
- The comma-string spec (`'S10,i4,f8'`) gives you **no real field names** (`f0, f1, f2`); use it only when names don't matter.
- `'S'` is a **byte string** while `'U'` is **Unicode** — mixing them up can cause encoding surprises *(added context: `'S'` fields compare against `bytes`, not `str`)*.
- For routine data wrangling, reach for **Pandas** instead; structured arrays are best for simple ops and C-layout mapping.

## Suggested figure (optional)
A side-by-side diagram: on the left, three separate boxes labeled `name`, `age`, `weight` with dashed lines hinting at the implicit row correspondence; on the right, a single contiguous memory strip divided into records, each record sub-divided into colored slots (`U10` | `i4` | `f8`) with a header naming the fields — visually showing how a structured array fuses the parallel arrays into one typed buffer.
