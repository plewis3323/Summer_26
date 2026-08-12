# Week 01 — Exercises

Work top to bottom. Setup (imports, constants) is given by the notebook; you
write only the lines each exercise asks for. E1, E5, and E7 live outside the
notebook (the terminal and `.py` files, per `NOTEBOOK_RULES.md` §6); E2–E4 and
E6 are notebook cells.

## E1 — Terminal warm-up

In a terminal, starting from your home folder: create the folder tree
`course/week01/scratch`, move into `scratch`, prove where you are, then move
back up to `week01` — using Tab completion for every folder name you type.
Paste the command sequence into the notebook's E1 markdown cell.
Hint: `mkdir` can only make one level at a time unless you create them in order;
`cd ..` goes up.
Accept when: your pasted sequence contains only `pwd`/`ls`/`cd`/`mkdir`, and the
final `pwd` output ends in `course/week01`.

## E2 — Calculator

Compute, each in its own cell: the number of seconds in a (365-day) year; how
many 25 ns bunch crossings fit in one second (25 ns is $25 \times 10^{-9}$ s —
the time between proton-bunch collisions at the LHC, CERN's large accelerator);
and `17 // 5` plus `17 % 5` recombined to reconstruct 17.
Hint: scientific notation in Python looks like `25e-9`.
Accept when: the three results are `31536000`, `40000000.0`, and an expression
using `//` and `%` that evaluates to exactly `17`.

## E3 — Variables and update

Given `count = 0` in setup: add 3 event batches of sizes 120, 87, and 45 to
`count` using only the update pattern `count = count + ...` (one line per
batch), then compute `mean_batch` as a float.
Hint: the mean is the total divided by the number of batches.
Accept when: `count` is `252` and `mean_batch` is `84.0`.

## E4 — String surgery

Setup gives `record = "  run_042,muon,13.7  "`. Produce: `clean` (whitespace
stripped), `parts` (the three comma-separated fields as a list), and
`energy` (the third field as a float). Then build the f-string
`summary` reading exactly `run_042 saw a muon at 13.7 GeV`.
Hint: chain `.strip()` then `.split(",")`; list items are `parts[0]`,
`parts[1]`, `parts[2]` (counting starts at 0).
Accept when: `energy == 13.7` and `summary` matches the target string exactly.

## E5 — An input script

Write `half.py` in your `week01` folder: ask the user for a starting number of
radioactive nuclei (a whole number), and print how many remain after one, two,
and three half-lives. (A **half-life** is the time in which half of a
radioactive sample decays — so each step divides the count by 2.) Use `//` so
the counts stay whole numbers.
Hint: convert the `input` string with `int(...)` once, then reuse the variable.
Accept when: `uv run python half.py` with input `1000` prints `500`, `250`,
and `125` (one per line, labels up to you).

## E6 — Deliberate breakage

The notebook gives four broken cells (a misspelled name, a missing quote,
`"3" + 4`, and `int("12.5")`). For each: run it, and in the cell below write
one comment line naming the error type and one line fixing the code so it runs.
Hint: read the traceback bottom line first; it names the error for you.
Accept when: all four fixed cells run without error and each names the right
error type (`NameError`, `SyntaxError`, `TypeError`, `ValueError`).

## E7 — Synthesis: a lab-notebook greeter

Write `greeter.py`: ask for the user's name and a beam energy in GeV, then
print a three-line report — a greeting using the name, the energy converted to
joules (use `EV_TO_JOULES = 1.602e-19` and 1 GeV = `1e9` eV, formatted with
`:.3e`), and the number of characters in the name. Run it from the terminal.
Hint: this is the lesson's worked example plus E4's `len` trick; two `input`
calls, one `float` conversion.
Accept when: input `Parker` / `200` produces a line containing `3.204e-08`
and a line containing `6`, with no traceback.

## Review

Week 01 has no earlier weeks; these draw on the setup guide (`02-Setup-Guide.md`,
Day 0 and Week 01 sections).

1. On Windows, what does WSL2 give you, and why does this course use it?
2. What single rule about `sudo` keeps a beginner's terminal safe?
3. Which command creates a new uv project, and which command runs a script
   inside that project's environment?
4. Name the two keys that save the most typing and re-typing in a terminal.
