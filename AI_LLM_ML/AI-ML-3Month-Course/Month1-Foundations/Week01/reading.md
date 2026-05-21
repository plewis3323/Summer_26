# Week 01 — Reading

## Primary

1. **Bishop, *Pattern Recognition and Machine Learning*, Chapter 1.**
   - §1.1 "Example: Polynomial Curve Fitting" (pp. 4–11). Read this carefully. It is the cleanest introduction to bias–variance, overfitting, and regularization you'll find. Connect it to over-binning and under-binning in a histogram fit.
   - §1.2 "Probability Theory" (pp. 12–24). Skim if comfortable; it's a probability refresher.
   - §1.3 "Model Selection" (pp. 32–33). Introduces train/validation/test logic.

2. **VanderPlas, *Python Data Science Handbook*, Chapters 2 and 3.** Freely available at https://jakevdp.github.io/PythonDataScienceHandbook/.
   - Ch. 2 NumPy: "Understanding Data Types", "Introduction to NumPy Arrays", "Computation on NumPy Arrays", "Computation on Arrays: Broadcasting", "Boolean Arrays and Masks", "Fancy Indexing".
   - Ch. 3 pandas: "Introducing Pandas Objects", "Data Indexing and Selection", "Operating on Data in Pandas", "Handling Missing Data", "Hierarchical Indexing", "Combining Datasets: merge and join", "Aggregation and Grouping".

3. **3Blue1Brown, *But what is a neural network?* and the next three videos.** YouTube playlist "Neural networks". Watch all four (total ~60 min). Timestamps:
   - Video 1 *What is a neural network?* (19:13): 00:00–07:00 intuition; 07:00–19:13 math.
   - Video 2 *Gradient descent, how neural networks learn* (21:01). Watch all.
   - Video 3 *What is backpropagation really doing?* (13:54). Watch all.
   - Video 4 *Backpropagation calculus* (10:17). Can defer to Week 03.

   These are intuition priming. You are not expected to implement from them — you'll do that in Week 03 from Karpathy's series.

## Supplementary

4. **Python 3 for scientists.** If you haven't written modern Python in a while:
   - Real Python — "Python's F-String for String Interpolation" (f-strings came in 3.6, use them).
   - Real Python — "Type Hints and Annotations in Python" (you will add type hints to your course code).

5. **Hunter, J. (2007). *Matplotlib: A 2D graphics environment*.** Computing in Science & Engineering. Skim — the history helps you understand why matplotlib has two APIs (state-based vs object-oriented) and which to prefer (OO).

## Codecademy (you have Pro)

6. **"Build a Machine Learning Model" — modules 1 and 2.**
   - Module 1: "Foundations of  Machine Learning" — terminology and workflow. ~1 hour. Do this first.
   - Module 2: "Supervised Learning" introduction — ~2 hours. Save the regression/classification implementations for Week 02; the framing is useful now.

## HEP-adjacent context (optional but recommended)

7. **Guest, Cranmer, Whiteson — *Deep Learning and its Application to LHC Physics* (Annu. Rev. Nucl. Part. Sci. 2018).** Sections 1 and 2 only this week. Just to calibrate what "ML in HEP" looks like at a review level. Skim. You'll re-read §3–4 in Week 02 and §5–6 in Weeks 04–05.

## What to take notes on

Write answers to these in a notebook (plain markdown is fine):

- Why does the test-set error eventually go up if you keep increasing polynomial degree in Bishop §1.1?
- Give a 3-sentence explanation of broadcasting rules you could tell a labmate.
- What is the `pandas` equivalent of a `TTree::Draw` cut?
- In matplotlib's object-oriented API, what do `fig` and `ax` represent, and why do we prefer it to `plt.plot`-style code?
- Name three things that must be seeded to make a numpy-only ML pipeline reproducible.

Keep these — they're your Week 01 lightning reference.
