# Reading-Session Prompt — Vanderplas Ch. 2–3 (Personal Claude ML/AI Class)

Filled-in for this session. Copy the block below into a new Claude Code session to start the reading.
Source book online: https://jakevdp.github.io/PythonDataScienceHandbook/

---

## ▶️ Copy-paste prompt

> This is a **reading session** for my personal Claude ML/AI class.
> **Source:** Jake VanderPlas, *Python Data Science Handbook* (https://jakevdp.github.io/PythonDataScienceHandbook/)
> **Scope:** Chapter 2 — *Introduction to NumPy*, and Chapter 3 — *Data Manipulation with Pandas*
> **Week/Module:** Week 1, Month 1
> **Working folder:** .../Month_1/Week_1/Week1_Vanderplas_2_3
>
> Run the session like this:
> 1. Create a `Vanderplas_2_3_materials/` folder in the working folder for note `.md` files and images.
> 2. As I send **questions, figures, images, and text** back and forth, capture them into
>    numbered note files (`notes_01_*.md`, `notes_02_*.md`, …). Use LaTeX (`$$ … $$`) for math,
>    keep prose readable, and describe/recreate any figures so they make sense standalone.
> 3. When I describe a figure or ask for one, **generate a clean image** (matplotlib/PIL) into the
>    materials folder and reference it from the notes.
> 4. **Augment with grad-student notes:** alongside my chunks, summarize each section the way a
>    grad student would — concise conceptual summary, key derivations/idioms worked out, *try the
>    exercises and code examples*, and short "why this matters / intuition" remarks. For this book
>    that means runnable NumPy/Pandas snippets. Keep my captured chunks and your generated notes
>    visually distinct.
> 5. Keep notes faithful to the source — don't invent content I didn't read or discuss; flag
>    anything you add beyond the source as *(added context)*.
> 6. When I say **"done"**, compile everything into a single **readable Jupyter notebook**
>    (markdown cells for notes + math, code cells that are runnable/compilable, embedded images).

---

## Conventions Claude should follow

- **Folder layout** (mirrors the Codecademy reading sessions):
  ```
  Week1_Vanderplas_2_3/
    Vanderplas_2_3_materials/
      notes_01_<slug>.md
      notes_02_<slug>.md
      img_<slug>.png
    Vanderplas_2_3_Notes.ipynb        # built at the end, on "done"
  ```
- **Note files:** one focused subtopic each, `#`-titled, math in `$$ … $$`, figures described in words
  even when an image exists (so the text stands alone).
- **Images:** generated programmatically, saved as `img_<slug>.png`, captioned in the notes.
- **Notebook on "done":** markdown cells carry the prose/math; code cells are runnable; images embedded.
  Should render cleanly in Jupyter and be compilable if executed top-to-bottom. Code cells should
  `import numpy as np` / `import pandas as pd` as needed so the notebook runs standalone.
- **Fidelity:** only record what came from the reading or our discussion. Flag anything Claude adds
  as context with *(added context)*.

---

## Scope reference (Vanderplas, Ch. 2–3)

- **Chapter 2 — Introduction to NumPy:** data types & arrays, array attributes/indexing/slicing,
  reshaping & concatenation, ufuncs, aggregations, broadcasting, boolean masks & comparison,
  fancy indexing, sorting, structured arrays.
- **Chapter 3 — Data Manipulation with Pandas:** `Series`/`DataFrame`/`Index` objects, indexing &
  selection, ufuncs & index alignment, handling missing data, hierarchical indexing (MultiIndex),
  concat/append, merge & join, aggregation & `groupby`, pivot tables, vectorized string ops,
  working with time series, `eval`/`query` for high-performance ops.

---

## Trigger words during a session
- **"done"** → build the Jupyter notebook from all note files + images.
- **"new note: …"** → start a fresh numbered note file.
- **"figure: …"** → generate an image into the materials folder and link it in the notes.
