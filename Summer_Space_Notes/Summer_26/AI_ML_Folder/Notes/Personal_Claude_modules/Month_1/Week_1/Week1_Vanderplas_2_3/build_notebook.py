#!/usr/bin/env python3
"""Compile all notes_*.md files in Vanderplas_2_3_materials/ into one Jupyter notebook.

Markdown prose (incl. $$..$$ math and ascii ``` blocks) -> markdown cells.
```python fenced blocks -> runnable code cells. Order preserved.
"""
import json
import os
import re
import glob
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS = os.path.join(HERE, "Vanderplas_2_3_materials")
OUT = os.path.join(HERE, "Vanderplas_2_3_Notes.ipynb")


def split_md(text):
    """Yield ('markdown'|'code', content) segments. Only ```python fences become code."""
    lines = text.splitlines()
    segs = []
    buf = []
    mode = "markdown"
    i = 0
    while i < len(lines):
        line = lines[i]
        if mode == "markdown" and line.strip() == "```python":
            if buf:
                segs.append(("markdown", "\n".join(buf).strip("\n")))
                buf = []
            mode = "code"
            i += 1
            continue
        if mode == "code" and line.strip() == "```":
            segs.append(("code", textwrap.dedent("\n".join(buf))))
            buf = []
            mode = "markdown"
            i += 1
            continue
        buf.append(line)
        i += 1
    if buf:
        kind = "markdown" if mode == "markdown" else "code"
        segs.append((kind, "\n".join(buf).strip("\n")))
    # drop empty markdown segments
    return [(k, c) for k, c in segs if c.strip()]


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _splitlines_keepends(src)}


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _splitlines_keepends(src)}


def _splitlines_keepends(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [s]


cells = []

# Title / front matter
cells.append(md_cell(
    "# Python Data Science Handbook — Ch. 2–3 Reading Notes\n\n"
    "**Jake VanderPlas, *Python Data Science Handbook*** — "
    "Chapter 2 (Introduction to NumPy) & Chapter 3 (Data Manipulation with Pandas)\n\n"
    "Week 1, Month 1 — Personal Claude ML/AI class. Source: "
    "https://jakevdp.github.io/PythonDataScienceHandbook/\n\n"
    "*Grad-student reading notes (Claude-generated run-through) plus captured session Q&A. "
    "Faithful to the source; anything beyond it is flagged* `(added context)`.\n\n"
    "Code cells import `numpy as np` / `pandas as pd` where needed and are meant to run "
    "top-to-bottom within each section.\n\n"
    "---\n\n"
    "## Contents\n\n"
    "**Ch. 2 — NumPy:** 01 Data types · 02 Array basics · 03 Ufuncs · 04 Aggregations · "
    "05 Broadcasting · 06 Boolean masks · 07 Fancy indexing · 08 Sorting · 09 Structured arrays\n\n"
    "**Ch. 3 — Pandas:** 10 Pandas objects · 11 Indexing & selection · 12 Operating on data · "
    "13 Missing data · 14 Hierarchical indexing · 15 Concat & append · 16 Merge & join · "
    "17 Aggregation & grouping · 18 Pivot tables · 19 String ops · 20 Time series · 21 eval/query"
))

note_files = sorted(glob.glob(os.path.join(MATERIALS, "notes_*.md")))
assert note_files, "no note files found"

for path in note_files:
    with open(path) as f:
        text = f.read()
    # strip a leading/trailing standalone '---' horizontal rule that brackets the file
    text = re.sub(r"\A\s*---\s*\n", "", text)
    for kind, content in split_md(text):
        cells.append(md_cell(content) if kind == "markdown" else code_cell(content))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {OUT}")
print(f"  source note files: {len(note_files)}")
print(f"  cells: {len(cells)} total -> {n_md} markdown, {n_code} code")
