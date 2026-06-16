---
# Notes 19 — Vectorized String Operations (Vanderplas Ch. 3.10)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/03.10-working-with-strings.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
Pandas adds **vectorized string operations** through the `.str` accessor on a Series (or Index). NumPy's vectorization speeds up numeric arrays element-wise but has no clean path for arrays of strings, and plain Python loops/comprehensions break on missing data. The `.str` accessor exposes Python's familiar string methods (plus regex helpers) in a vectorized, missing-value-aware form.

The motivating pain point: a list comprehension over data containing `None` raises `AttributeError` (illustrative — this block intentionally fails, so it is shown as display, not run):

```
data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
[s.capitalize() for s in data]   # AttributeError: 'NoneType' has no attribute 'capitalize'
```

Pandas Series carry a `.str` attribute that applies the operation element-wise and **skips missing values gracefully**.

## Key ideas / idioms
- `.str` mirrors nearly all Python `str` methods, just vectorized: `lower`, `upper`, `capitalize`, `len`, `startswith`, `strip`, `split`, etc.
- Return type follows the operation: strings stay strings, `len()` gives integers, `is*` tests give booleans.
- A second family wraps Python's `re` module: `match`, `contains`, `extract`, `findall`, `count`, `replace`, `split`.
- A miscellaneous family handles indexing and reshaping: `get`, `slice`, indexing via `str[...]`, `get_dummies`, plus `cat`, `repeat`, `pad`, `wrap`, `join`, `normalize`.
- Chaining works: `s.str.split().str.get(-1)` re-applies `.str` after a method that returns lists.
- Boolean results from `contains`/`match` slot directly into masking and counting (`.sum()` counts `True`).

There is no real math in this section; the only "quantity" idiom is counting boolean hits:
$$\text{count} = \sum_i \mathbb{1}[\text{pattern matches } x_i].$$

## Worked code examples (runnable)
```python
import numpy as np
import pandas as pd

# --- The .str accessor handles missing values ---
data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
names = pd.Series(data)
print(names.str.capitalize())   # None is preserved, not an error

# --- Python-like string methods ---
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
print(monte.str.lower())
print(monte.str.len())                 # integers
print(monte.str.startswith('T'))       # booleans
print(monte.str.split())               # lists of words

# --- Regex methods ---
# extract: pull the first group from each entry (first names here)
print(monte.str.extract('([A-Za-z]+)', expand=False))
# findall: e.g., names that start and end with a consonant
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'))
# contains: boolean search
print(monte.str.contains('Terry'))

# --- get / slice / indexing ---
print(monte.str[0:3])                  # first 3 chars of each
print(monte.str.split().str.get(-1))   # last name (last token)

# --- get_dummies: split coded indicators into columns ---
# (added context) inline data standing in for the recipe database,
# which the book downloads from a large JSON file.
full = pd.DataFrame({
    'name': monte,
    'info': ['B|C|D', 'B|D', 'A|C', 'B|C', 'B|C|D', 'B|D'],
})
print(full['info'].str.get_dummies('|'))

# --- Counting boolean hits (mirrors the recipe analysis) ---
desc = pd.Series(['quick breakfast bowl', 'hearty dinner', 'Breakfast tacos'])
print(desc.str.contains('[Bb]reakfast').sum())   # -> 2
```

## Why this matters / intuition
Real-world data is messy and text-heavy, and Vanderplas stresses that **cleaning/munging often makes up the majority of data-science work**. The `.str` accessor lets you express that cleaning concisely and safely: no manual loops, no crashes on `None`, and regex power on tap. The recipe-database example shows the payoff — boolean `str.contains` filters turn ~173,000 recipes into a tiny targeted set, which is the kernel of a simple ingredient-based recommender.

## Gotchas
- Methods that return lists (`split`, `findall`) require a second `.str` to keep operating element-wise (e.g., `.str.split().str.get(-1)`).
- `extract` needs a capture group `(...)`; `expand=False` returns a Series instead of a DataFrame.
- Regex methods are case-sensitive by default — the book uses `[Bb]reakfast`/`[Cc]innamon` character classes to catch capitalization; misspellings (e.g., "cinamon") are simply missed unless you account for them.
- Missing values propagate as `NaN`/`None` through `.str` operations rather than raising — convenient, but watch for them in downstream counts.
- `.str` is for object/string dtype; calling it on numeric data won't help.

## Suggested figure (optional)
A two-column "cheat-sheet" graphic: left column lists the Python-like methods grouped by return type (string-returning, integer-returning `len`, boolean `is*`/`startswith`); right column lists the regex family (`match`, `contains`, `extract`, `findall`, `count`, `replace`, `split`) with a one-word purpose beside each, plus a small footer box showing the `get_dummies('|')` split turning one coded column into several 0/1 indicator columns.
---
