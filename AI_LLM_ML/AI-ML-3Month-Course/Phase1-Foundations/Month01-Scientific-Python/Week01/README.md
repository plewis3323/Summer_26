# Week 01 — Python + NumPy

The event loop you'd write in ROOT/C++ becomes a single array expression here — this week
is about making that translation reflexive.

## Objectives

- Write vectorized NumPy code that replaces explicit `for`-loops over events.
- Use broadcasting deliberately (predict output shapes before running).
- Apply boolean-mask selections the way you'd apply cuts in an analysis.
- Explain views vs copies and C-contiguous memory layout, and say when each bites.
- Translate a ROOT idiom (TTree loop, TH1 fill) into its NumPy equivalent on sight.

## Core material (~3 hrs)

- VanderPlas, *Python Data Science Handbook*, Chapter 2 (NumPy) — the whole chapter;
  it is the backbone of the week.
- NumPy official docs: the "Broadcasting" section of the user guide. Read it twice;
  the rules are short and everything follows from them.
- Skim VanderPlas Chapter 1 (IPython/Jupyter) for `%timeit`, `?`, and debugging magics.
- Optional: NumPy docs "NumPy for MATLAB users" reads well as "NumPy for ROOT users"
  if you mentally substitute TTree branches for matrices.

## Exercises (built when the week starts)

1. **Event loop → array expression.** Compute dimuon invariant mass from `(E, px, py, pz)`
   arrays two ways: a Python loop and a vectorized expression.
   Accept when: results agree to 1e-9 and the vectorized version is ≥50× faster on 10⁶ events (`%timeit`).
2. **Cuts as masks.** Apply pT > 2 GeV and |η| < 1.1 selections with boolean masks; chain them.
   Accept when: surviving-event count matches a loop-based reference exactly.
3. **Broadcasting shapes.** Given arrays of shapes (N,3), (3,), (N,1), predict then verify
   the shape of each pairwise operation; center a hit collection per coordinate.
   Accept when: all shape predictions written before running are correct and the centered array has zero column means to 1e-12.
4. **Views vs copies.** Modify a slice and a fancy-indexed selection of the same array;
   explain which mutated the parent.
   Accept when: printed parent arrays match a stated prediction for both cases.
5. **Histogram without TH1.** Bin a mass array with `np.histogram`, then reproduce the
   counts with pure broadcasting/`searchsorted`.
   Accept when: both bin-count arrays are identical for 100 bins over a fixed range.
6. **Fill-pattern translation.** Rewrite a provided ROOT-style pseudocode snippet
   (nested loop over events and tracks) as ≤5 lines of NumPy.
   Accept when: output array equals the loop reference and no Python-level loop over events remains.

## Deliverable

`notebooks/Week01_Exercises.ipynb` completed (all checks PASS), plus a short
`notes.md` listing five ROOT→NumPy idiom translations in your own words.

## Review

(Week 1 draws on your physics background.)

1. Why is the invariant mass of a muon pair invariant under boosts — and which array
   operation in Exercise 1 corresponds to the frame-independent quantity?
2. In a trigger, cuts are applied in a deliberate order for rate reasons. Does the order
   of chained boolean masks change the result here? The cost?
3. Your C++ event loop touches each event once; the vectorized version makes several
   full passes over arrays. Why is it still faster in Python?
4. What does a TH1 own that a plain `np.histogram` output does not (errors, over/underflow)?
