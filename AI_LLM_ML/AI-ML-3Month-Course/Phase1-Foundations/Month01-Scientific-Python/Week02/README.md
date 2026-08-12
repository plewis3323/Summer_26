# Week 02 — Control flow, functions, and data

Week 01's programs ran every line exactly once. This week your programs
decide, repeat, and organize — and by the end you assemble the pieces into a
real tool: a word-frequency counter that chews through whole books.

## Objectives

- Branch with `if`/`elif`/`else` and combine conditions with `and`/`or`/`not`.
- Loop with `for` (over lists, `range`) and `while`; know which fits when.
- Define functions with parameters and `return`; explain local variables and `None`.
- Use lists (indexing, slicing, `append`) and dictionaries (the counting pattern).
- Read and write text files line by line, handling `"\n"` and type conversions.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text; type every example.
- Severance, *Python for Everybody*, chapters 3–9 — many more worked examples
  of exactly this week's topics; skim fast, slow down where needed.
- The official Python Tutorial, sections 4–5 — control flow and data
  structures, second pass.

## Exercises

See `exercises.md` (notebook generated when the week starts, per
`NOTEBOOK_RULES.md`). Seven exercises building from single functions and loops
through dictionary counting and file parsing, ending with a word-frequency
counter that writes a ranked report file.

## Deliverable

Completed exercise notebook (all checks PASS) plus `wordcount2.py` producing a
correct `report.txt` from a real book-length text file.

## Review

1. (Week 01) Reconstruct the `uv` commands that create a project folder and
   run a script inside it.
2. (Week 01) Why does `input` + arithmetic always need `int(...)` or
   `float(...)` in between?
3. (Week 01) Which error types did Week 01 name, and what does each mean in
   one phrase?
4. (Week 01) Write the f-string that prints a float `m` as `m = 3.10 GeV`.
