# Week 03 — The scientific stack

~10 hrs. Before starting you should be able to: write `for` loops and `if`
selections over lists (Week 02); define and call functions (Week 02); read a
CSV-like text file line by line (Week 02); run scripts and notebooks (Week 01).

Week 02 ended with loops over a book. Science means loops over *numbers* —
millions of them — and plain Python loops are too slow and too wordy for that.
This week you meet the three packages that make Python the language of
science: **NumPy** (fast arrays of numbers), **matplotlib** (plots), and
**pandas** (tables). Every later week of this course runs on these three.

Install them into this week's project first:

```
uv add numpy matplotlib pandas
```

Work in a notebook this week (`uv run jupyter lab`) — arrays and plots want
immediate feedback.

## 1. Why arrays

Suppose you have a million energy measurements and want them all doubled.
Week 02 style:

```python
doubled = []
for e in energies:
    doubled.append(2.0 * e)
```

This works. But Python executes it one item at a time, checking at every step
what type `e` is and what `*` means for it — bookkeeping repeated a million
times. An **array** fixes this: a block of numbers, all the same type, stored
side by side in memory, on which one operation applies to *every element at
once*:

```python
doubled = 2.0 * energies      # if energies is a NumPy array
```

The loop still happens — but inside NumPy, in compiled code, without the
per-item bookkeeping. Measure it yourself (`time.perf_counter()` reads a
high-resolution clock in seconds):

```python
import time
import numpy as np

values = list(range(1_000_000))
arr = np.arange(1_000_000)

t0 = time.perf_counter()
out1 = [2.0 * v for v in values]
t1 = time.perf_counter()
out2 = 2.0 * arr
t2 = time.perf_counter()

print(f"loop:  {t1 - t0:.4f} s")
print(f"array: {t2 - t1:.4f} s")
```

Typical result: the array version is 50–100× faster. (The underscores in
`1_000_000` are just readable digit grouping; `import ... as np` gives the
package a short nickname — `np` is the universal one.)

Speed is half the argument. The other half is *clarity*: `2.0 * arr` says
what happens to the data in one line, the way a formula does. Scientific code
written with arrays reads like the algebra it implements. This style is
called **vectorized** code, and making it your reflex is the goal of the week.

## 2. Making arrays

The array type is `np.ndarray`; you will just say "array". Ways to get one:

```python
import numpy as np

a = np.array([4.0, 7.0, 1.5])       # from a Python list
z = np.zeros(5)                     # [0. 0. 0. 0. 0.]
n = np.arange(10)                   # 0..9, like range
x = np.linspace(0.0, 1.0, 5)        # 5 evenly spaced points from 0 to 1
```

`np.arange` counts like `range`; `np.linspace(start, stop, num)` gives `num`
evenly spaced values *including* both endpoints — the standard way to make an
x-axis. Every array has two attributes you should check reflexively:

```python
>>> a.dtype
dtype('float64')
>>> a.shape
(3,)
```

The **dtype** is the one type shared by all elements (`float64` — Week 01's
floats; `int64` for integers). The **shape** is a tuple of sizes along each
**axis** (dimension). `(3,)` means one axis of length 3. A 2D array — a grid,
like a table or an image — has shape `(rows, columns)`:

```python
m = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])     # m.shape == (2, 3)
```

Arithmetic on arrays is **elementwise** — applied to each element, or to
matching pairs when both sides are arrays of the same shape:

```python
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([10.0, 20.0, 30.0])
>>> a + b
array([11., 22., 33.])
>>> a ** 2
array([1., 4., 9.])
>>> np.sqrt(a)
array([1.        , 1.41421356, 1.73205081])
```

`np.sqrt`, `np.exp`, `np.log`, `np.sin` and friends all work this way. And
whole-array questions have one-word answers: `a.sum()`, `a.mean()`, `a.min()`,
`a.max()`, `a.std()` (standard deviation — a measure of spread you will meet
properly in Week 08).

## 3. Indexing and slicing

Indexing works like lists (from 0, negatives from the end), and slices work
like list slices — same half-open convention:

```python
>>> x = np.arange(10) * 10
>>> x[0], x[-1]
(np.int64(0), np.int64(90))
>>> x[2:5]
array([20, 30, 40])
```

Two dimensions take one index per axis, in one bracket — row first, then
column — and `:` alone means "everything along this axis":

```python
>>> m[0, 2]      # row 0, column 2
3.0
>>> m[1, :]      # all of row 1
array([4., 5., 6.])
>>> m[:, 0]      # column 0 of every row
array([1., 4.])
```

One behavior that *differs* from lists: a slice of an array is a **view** —
a window onto the same memory, not a copy. Modify the slice and you modify
the original:

```python
>>> y = x[2:5]
>>> y[0] = -999
>>> x[2]
np.int64(-999)
```

Views make slicing free even on gigabyte arrays; the cost is this
occasionally surprising sharing. When you need an independent copy, say so:
`y = x[2:5].copy()`.

## 4. Boolean masks: cuts on data

Comparisons on arrays are elementwise too, and produce a boolean array called
a **mask**:

```python
>>> e = np.array([13.7, 0.4, 88.0, 2.1, 250.0])
>>> mask = e > 2.0
>>> mask
array([ True, False,  True,  True,  True])
```

Index an array with a mask and you get only the elements where the mask is
`True`:

```python
>>> e[mask]
array([ 13.7,  88. ,   2.1, 250. ])
```

That is Week 02's "cut" pattern — the loop with `if` and `append` — in one
line. Conditions combine elementwise with `&` (and), `|` (or), `~` (not);
parentheses around each comparison are required because of how Python groups
the symbols:

```python
selected = e[(e > 2.0) & (e < 100.0)]
```

Three companions to know: `mask.sum()` counts the `True`s (booleans add as
1/0 — the fastest way to ask "how many pass?"); `np.where(mask, a, b)` picks
from `a` where true and `b` where false; and masks filter *other* arrays of
the same length — `times[e > 2.0]` gives the timestamps of the events that
passed the energy cut. That last idiom is the backbone of every analysis in
this course.

## 5. Broadcasting

What happens when shapes don't match? If one operand is a single number,
NumPy **broadcasts** it — stretches it, conceptually, to every element:

```python
>>> e * 2.0            # scalar broadcast over shape (5,)
array([ 27.4,   0.8, 176. ,   4.2, 500. ])
```

The same stretching works between arrays of different shapes when their
shapes are *compatible*. The rule, checked axis by axis from the right: two
sizes are compatible if they are equal, **or one of them is 1** (the size-1
axis is stretched to match). A missing axis on the left counts as size 1.

The workhorse example — normalize each column of a table. `m` has shape
`(2, 3)`; its column means have shape `(3,)`:

```python
>>> col_means = m.mean(axis=0)       # mean down each column, shape (3,)
>>> centered = m - col_means         # (2,3) - (3,) broadcasts to (2,3)
>>> centered
array([[-1.5, -1.5, -1.5],
       [ 1.5,  1.5,  1.5]])
```

(`axis=0` means "collapse axis 0" — compute down the rows, leaving one value
per column; `axis=1` collapses the columns, one value per row.) Check the
rule: `(2, 3)` vs `(3,)` → right-align → `3` vs `3` equal; `2` vs missing
(=1) → stretch. Result `(2, 3)`.

When you get broadcasting wrong, NumPy tells you:
`operands could not be broadcast together with shapes (2,3) (2,)`. The habit
that prevents it: *predict the output shape before you run the line*. If you
can't, print `.shape` on the inputs first.

## 6. First plots with matplotlib

Numbers become understanding when you can see them. matplotlib's standard
recipe is four lines, and it is always the same four lines:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("time [s]")
ax.set_ylabel("position [m]")
```

`plt.subplots()` creates a **figure** (the whole canvas, `fig`) and an
**axes** (one coordinate system on it, `ax`); all plotting happens through
`ax`. In a notebook the figure appears under the cell. The three plot types
that cover 90% of science:

```python
x = np.linspace(0.0, 10.0, 200)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin")          # line plot: y vs x
ax.plot(x, np.cos(x), label="cos")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
```

`label=` names a curve and `ax.legend()` displays the names. Second,
the **scatter plot** — one dot per (x, y) pair, for data that isn't a curve:

```python
fig, ax = plt.subplots()
ax.scatter(heights, weights)
ax.set_xlabel("height [cm]")
ax.set_ylabel("weight [kg]")
```

Third — and the one physics uses most — the **histogram**: chop the value
range into equal **bins** and draw a bar counting how many values fall in
each. It answers "how are these values distributed?":

```python
rng = np.random.default_rng(42)
energies = rng.normal(10.0, 2.0, size=5000)   # 5000 values, bell curve

fig, ax = plt.subplots()
ax.hist(energies, bins=50)
ax.set_xlabel("energy [GeV]")
ax.set_ylabel("counts per bin")
```

Two things sneaked in. `np.random.default_rng(42)` creates a random-number
generator; `.normal(10.0, 2.0, size=5000)` draws 5000 values from a bell
curve centered at 10 with spread 2. The `42` is a **seed** — it fixes the
"random" sequence so reruns give identical data; Week 04 makes that a
principle. Real detector data will replace the generator in Week 04's project;
the histogram code stays the same.

House rule, now and forever: **every axis gets a label, with units.** An
unlabeled plot is a rumor. And to keep a figure, save it — vector PDF for
documents:

```python
fig.savefig("energies.pdf")
```

## 7. First DataFrames with pandas

NumPy arrays hold one kind of number. Real datasets are *tables* — each row
an observation, each column a differently-named, differently-typed quantity.
pandas provides the **DataFrame**: a table whose columns are NumPy arrays
with names. Given a **CSV file** (comma-separated values — a plain-text
table, one row per line, commas between fields, names in the first line):

```
run,particle,energy,quality
17,muon,13.7,good
17,pion,2.1,good
18,muon,88.0,bad
```

one call parses everything — names, types, all of it:

```python
import pandas as pd

df = pd.read_csv("events.csv")
```

(`pd` is pandas' universal nickname; `df` is the conventional name for a
DataFrame.) First-look tools, in the order you should reach for them:

```python
df.head()        # first 5 rows, as a formatted table
df.shape         # (n_rows, n_columns)
df.dtypes        # the type pandas inferred for each column
df["energy"]     # one column — indexing by NAME, not position
```

A single column is a **Series** — essentially a named NumPy array, so
everything from Sections 2–4 works on it: `df["energy"].mean()`,
`df["energy"] * 2`, and — the important one — masks:

```python
good = df[df["quality"] == "good"]
high = df[(df["quality"] == "good") & (df["energy"] > 5.0)]
```

Read `df[mask]` exactly like NumPy: keep the rows where the condition is
true. Same `&`/`|`, same required parentheses. New columns are created by
assignment, vectorized as always:

```python
df["energy_j"] = df["energy"] * 1e9 * 1.602e-19
```

The last tool of the week is the most powerful: **groupby** splits the rows
into groups by a column's value, applies a computation to each group, and
returns one row per group:

```python
>>> df.groupby("particle")["energy"].mean()
particle
muon    50.85
pion     2.10
```

Read it as three steps: group rows by `particle`; within each group take the
`energy` column; compute its mean. Counting per group is
`df.groupby("run").size()`. This split-apply-combine move replaces an entire
Week 02 dictionary-accumulation loop with one line — it *is* that loop,
vectorized. (That is the intro; `groupby` has depths we return to when
projects need them.)

## 8. Worked example — from raw numbers to a figure

End to end: simulate a small detector dataset, write it to CSV, load it with
pandas, cut, group, and plot. In a notebook, cell by cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 2000

run = rng.integers(1, 6, size=n)                  # run number 1..5
signal = rng.normal(91.0, 3.0, size=n // 4)       # a peak near 91
background = rng.exponential(40.0, size=n - n // 4)
energy = np.concatenate([signal, background])     # glue the two arrays
rng.shuffle(energy)
```

The story: a detector records `n` events across 5 runs. A quarter of the
events are **signal** — something real at 91 GeV producing a bump — and the
rest are **background**, a featureless falling distribution (that is what
`exponential` gives). Mixed together and shuffled, the peak is invisible in
the raw numbers. (This is a cartoon of Week 04's project, where the peak will
be a real particle in real data.) Package and save it:

```python
df = pd.DataFrame({"run": run, "energy": energy})
df.to_csv("toy_events.csv", index=False)
```

`pd.DataFrame({...})` builds a table from named arrays — a dictionary,
Week 02's own tool, with column names as keys. Now pretend we're the analyst
receiving the file:

```python
df = pd.read_csv("toy_events.csv")
print(df.shape)
print(df.head())
print(df.groupby("run").size())
```

Five runs, roughly 400 events each. Cut to the interesting region and look:

```python
sel = df[(df["energy"] > 60.0) & (df["energy"] < 120.0)]
print(f"{len(sel)} of {len(df)} events in window")

fig, ax = plt.subplots()
ax.hist(sel["energy"], bins=60)
ax.set_xlabel("energy [GeV]")
ax.set_ylabel("counts per bin")
ax.set_title("Toy spectrum, 60-120 GeV window")
fig.savefig("toy_spectrum.pdf")
```

A clear bump at 91 riding on a falling floor. One number to quantify it —
compare the peak window to a background-only window of equal width using
masks on the raw array:

```python
e = sel["energy"]
in_peak = ((e > 85.0) & (e < 97.0)).sum()
off_peak = ((e > 100.0) & (e < 112.0)).sum()
print(f"peak window: {in_peak}   off-peak window: {off_peak}")
```

More counts in the peak window than an equal-width window beside it: the bump
is real, not a fluctuation of the eye. Total new-style code: ~25 lines, no
`for` loop anywhere. Next week this exact pipeline — load, cut, histogram,
find the peak — runs on real CERN data, wrapped in tests and version control.

## Check yourself

1. Give the two reasons vectorized code beats a Python loop for numerical
   work.
2. `a = np.zeros((3, 4))` — what are `a.shape`, `a.dtype`, and `a[2, 0]`?
3. `x = np.arange(6); y = x[1:4]; y[0] = 99`. What is `x[1]`, and why? How
   would you prevent it?
4. Write one line keeping elements of `e` that are between 5 and 10 or
   exactly 0.
5. Shapes `(4, 3)` and `(4,)` — can they broadcast? Shapes `(4, 3)` and
   `(4, 1)`? Explain using the rule.
6. Which plot type answers "how are these 5000 measured masses distributed?"
   and what two things must the finished figure carry?
7. In pandas, what do `df["energy"]`, `df[df["energy"] > 5]`, and
   `df.groupby("run")["energy"].mean()` each return, in one phrase apiece?
8. What Week 02 pattern does `groupby` replace?

## Answers

1. Speed — the per-element loop runs in compiled code without Python's
   per-item bookkeeping (typically 50–100×); clarity — `2 * a` reads like
   the formula it implements.
2. `(3, 4)`; `float64` (`np.zeros` defaults to floats); `0.0`.
3. `x[1]` is `99` — a slice is a view sharing the original's memory, so
   writing through `y` writes `x`. Use `y = x[1:4].copy()` for independence.
4. `e[((e > 5) & (e < 10)) | (e == 0)]` — elementwise `&`/`|`, parentheses
   around each comparison.
5. `(4, 3)` vs `(4,)`: right-align → `3` vs `4` — incompatible, error.
   `(4, 3)` vs `(4, 1)`: `3` vs `1` → stretch the 1; `4` vs `4` → equal.
   Result `(4, 3)`.
6. A histogram; labeled axes with units (and a sensible bin count).
7. One column as a Series; the sub-table of rows passing the mask; a
   one-value-per-run summary (mean energy by run).
8. The dictionary accumulation loop — "if key in counts: update, else:
   create" — i.e. grouping and aggregating by hand.

## New terms

- **NumPy / matplotlib / pandas** — the scientific stack: arrays / plots / tables.
- **array (`ndarray`)** — a block of same-type numbers operated on all at once.
- **vectorized** — written as whole-array operations instead of Python loops.
- **`np.arange` / `np.linspace` / `np.zeros`** — count like `range` / evenly spaced points incl. endpoints / all zeros.
- **dtype** — the single element type of an array (`float64`, `int64`).
- **shape / axis** — tuple of sizes per dimension / one dimension; `axis=0` collapses down rows, `axis=1` across columns.
- **elementwise** — applied to each element (or each matching pair) independently.
- **view vs copy** — a slice shares the parent's memory; `.copy()` makes it independent.
- **mask** — boolean array from an elementwise comparison; `a[mask]` keeps the `True` rows; combine with `&`, `|`, `~` (parentheses required).
- **broadcasting** — the shape-stretching rule: axes compatible if equal or 1, checked from the right.
- **figure / axes** — the canvas / one coordinate system on it (`fig, ax = plt.subplots()`).
- **line plot / scatter plot / histogram / bin** — curve vs x / dot per pair / counts per interval / the interval.
- **legend / `label=`** — the box naming each plotted curve.
- **random-number generator / seed** — `np.random.default_rng(seed)`; the seed fixes the sequence for reproducibility.
- **DataFrame / Series** — pandas table / one named column of it.
- **CSV** — plain-text table, comma-separated, header line first; `pd.read_csv` parses it.
- **groupby** — split rows by a column's values, aggregate each group, one row per group.
- **signal / background** — the events you are looking for / everything else underneath them.

## Going deeper

- VanderPlas, *Python Data Science Handbook* (free online), chapter 2 (NumPy)
  — the full treatment of arrays, masks, and broadcasting; the week's spine.
- VanderPlas, chapter 4 (matplotlib) — through "Histograms" and
  "Customizing Legends"; skim the rest for now.
- VanderPlas, chapter 3 (pandas) — read up through "Aggregation and
  Grouping"; the later reshaping material waits until a project needs it.
- NumPy user guide, the "Broadcasting" page — the rule stated by its authors,
  with pictures; short, read it twice.
