# Week 02 — Exercises

Work top to bottom. Setup (test data, sample file creation) is given by the
notebook; you write only the lines each exercise asks for. E7 lives in a
`.py` file run from the terminal (per `NOTEBOOK_RULES.md` §6); E1–E6 are
notebook cells.

## E1 — Classifier

Write a function `classify(energy)` returning `"low"` for values below 1.0,
`"medium"` for 1.0 up to 100.0, and `"high"` for 100.0 and above.
Hint: test conditions from one end; early `return`s remove the need for `elif`.
Accept when: `classify(0.5)`, `classify(1.0)`, `classify(99.9)`,
`classify(100.0)` return `"low"`, `"medium"`, `"medium"`, `"high"`.

## E2 — Loop statistics

Setup gives a list `energies` of 200 floats. Using only a `for` loop and
variables (no `sum`/`min`/`max`), compute `total`, `lowest`, and `highest`.
Hint: initialize `lowest` and `highest` to `energies[0]`, then compare inside
the loop.
Accept when: your three values equal `sum(energies)`, `min(energies)`, and
`max(energies)` exactly.

## E3 — Cuts

From the same `energies`, build `selected` containing values between 2.0 and
50.0 (exclusive) — once with a `for`/`if`/`append` loop, once as a list
comprehension.
Hint: two comparisons joined by `and`.
Accept when: both versions are equal as lists and the notebook's reference
count matches `len(selected)`.

## E4 — Half-life countdown

A sample starts at `activity = 10000.0` decays per second; each half-life
divides it by 2 (the lesson explains half-lives). Using a `while` loop, count
how many half-lives pass before activity drops below 10.0.
Hint: one counter, one halving, per pass; make sure the loop can end.
Accept when: your counter equals `10` and the final activity is
`9.765625`.

## E5 — Particle census

Setup gives a list `hits` of particle-name strings (`"muon"`, `"pion"`, ...).
Build a dictionary `counts` mapping each name to its number of appearances,
then print one `name: count` line per particle, most common first.
Hint: the lesson's six-line counting pattern; sort `(count, name)` tuples.
Accept when: `counts` totals to `len(hits)` and the printed first line is the
most frequent particle.

## E6 — Runlog reader

Setup writes `runlog.txt`, lines like `run_017 2843 good`. Read it and build:
`n_good` (count of lines ending in `good`) and `events_good` (total of the
second field over good runs only).
Hint: `.strip()` then `.split()`; the event count needs `int(...)`.
Accept when: `(n_good, events_good)` equals the notebook's stated reference
pair exactly.

## E7 — Synthesis: frequency counter with a report file

Extend the lesson's worked example into `wordcount2.py` in your `week02`
folder: count words in a text file whose path the user types, then *write*
the full ranking (all words, most frequent first, one `count word` line each)
to `report.txt`, and print only the top 5 and the distinct-word count to the
screen.
Hint: reuse `count_words` and `top_n` from the lesson; open `report.txt` with
mode `"w"` and remember `"\n"`.
Accept when: running it on a Project Gutenberg book produces a `report.txt`
whose line count equals the printed distinct-word count, with the same top-5
order as the screen.

## Review

1. (Week 01) In the terminal, you are in `~/course`: give the commands to
   create `week02/data` and end up inside `data`.
2. (Week 01) What does `float(input("x? "))` do, and why is the `float`
   necessary before arithmetic?
3. (Week 01) `"7" * 3` — what is the result and why? What one change makes it
   `21`?
4. (Week 01) Name the error each produces: `enrgy + 1` with no such variable;
   `int("12.5")`; `"3" + 4`. Which Week 01 habit diagnoses all three in
   seconds?
