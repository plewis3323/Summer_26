# Week 03 — Exercises

Work top to bottom in the notebook — all exercises are notebook cells this
week (per `NOTEBOOK_RULES.md` §6). Setup gives the imports, a seeded
generator `rng = np.random.default_rng(42)`, and the synthetic datasets each
exercise names; you write only the lines asked for.

## E1 — Loop vs array

Setup gives `values` (a list) and `arr` (the same 1,000,000 numbers as an
array). Time computing the square root of everything both ways —
comprehension with `math.sqrt` vs `np.sqrt(arr)` — using `time.perf_counter`.
Hint: the lesson's Section 1 timing scaffold; subtract clock readings.
Accept when: both results agree (spot-check index 12345 to 1e-9) and the
array version is at least 20x faster.

## E2 — Shapes on paper first

Setup gives arrays of shapes `(6,)`, `(2, 3)`, `(3,)`, and `(2, 1)`. For the
five listed pairwise operations, *write your shape prediction as a comment
before running*, then verify with `.shape` — one of the five is an error, and
predicting that counts.
Hint: right-align the shapes; a size must be equal or 1.
Accept when: all five comments match the observed shape (or observed error).

## E3 — Cuts as masks

Setup gives `energy` (10,000 floats) and `quality` (10,000 strings,
`"good"`/`"bad"`). Compute: how many events have `energy > 20`; the mean
energy of good events; and `clean`, the energies of good events between 20
and 150.
Hint: `mask.sum()` counts; masks built from one array can index another.
Accept when: the three values match the notebook's stated reference triple
exactly.

## E4 — Center the columns

Setup gives `hits`, shape `(500, 3)` — x, y, z positions of detector hits.
Using one broadcast subtraction (no loop), produce `centered` whose column
means are zero.
Hint: `hits.mean(axis=0)` has shape `(3,)`.
Accept when: `centered.shape == (500, 3)` and every column mean of `centered`
is below 1e-12 in absolute value.

## E5 — Three plots

From setup's `decay_time` array (exponentially distributed): (a) a histogram
with 40 bins; (b) a scatter of `decay_time[:200]` vs setup's `pulse[:200]`;
(c) a line plot of `np.exp(-t / 2.2)` for `t = np.linspace(0, 10, 100)`.
Every axis labeled with units (times in microseconds); save (a) as
`decay_hist.pdf`.
Hint: one `fig, ax = plt.subplots()` per plot; labels before saving.
Accept when: the three cells produce the three plot types, no axis is
unlabeled, and `decay_hist.pdf` exists next to the notebook.

## E6 — First DataFrame

Setup wrote `muon_runs.csv` (columns `run`, `n_events`, `mean_energy`,
`quality`). Load it; report the shape; make `good`, the sub-table of rows
with `quality == "good"`; and add a column `total_energy` equal to
`n_events * mean_energy`.
Hint: `pd.read_csv`, one mask, one vectorized assignment.
Accept when: the shape matches the stated reference, `len(good)` matches, and
`total_energy` for the first row equals `n_events[0] * mean_energy[0]`.

## E7 — Synthesis: find the bump

Setup gives `events.csv` (columns `run`, `energy`; a hidden peak on a falling
background). Load it; histogram the energy in 60 bins over the stated window
with labeled axes; locate the peak by eye; then compute counts in the peak
window vs an equal-width off-peak window, and events-per-run via `groupby`.
Save the figure as `bump.pdf`.
Hint: this is the lesson's worked example with a different peak location —
reuse its window-counting masks.
Accept when: `bump.pdf` exists, the in-peak count exceeds the off-peak count
by the notebook's stated margin, and the groupby table has one row per run.

## Review

1. (Week 02) Write, from memory, the dictionary pattern that counts
   occurrences of items in a list — then name the pandas one-liner from this
   week that replaces it.
2. (Week 02) Why does a line read from a text file usually need `.strip()`
   before comparison?
3. (Week 01) `float("1e3")` — value and type? And `"1e3" + "2"`?
4. (Week 02) When must you use `while` rather than `for`, and what bug is the
   risk?
