# K-Means Handwriting Recognition Project — Progress Tracker

**Lesson:** Codecademy MLE Path → Unsupervised Learning → K-Means Clustering → *Handwriting Recognition project*
**Note file:** `Handwriting_Recognition_KMeans_Lesson.ipynb`
**Started:** 7/8/26

---

## Status: DONE (7/8/26)
Full project (Steps 1-18) folded in from real exercise code/instructions and verified to run top-to-bottom
clean (`nbconvert --execute`, exit 0). TL;DR written.

**Key design fix during this pass:** originally each code cell independently re-created and re-fit `model`
(mirroring the prior Iris "hard way" cumulative-cell style). For this project that was actually wrong —
`KMeans` has no `random_state`, so independent re-fits per cell scrambled which cluster index meant which
digit *between* cells, silently breaking the Step 10 centroid grid -> Step 17 decode relationship. Restructured
so `model` is fit once (Steps 7-8 combined into one cell) and reused by every later cell, matching how a real
Jupyter kernel actually shares state top-to-bottom.

**Real, ground-truth-verified result:** user drew `2, 0, 2, 9` in Codecademy's `test.html`. Step 17's
hardcoded index->digit map (built for Codecademy's own captured run) decoded this run's `new_labels`
(`[6 6 4 6]`) as `4464` — flagged as unreliable, since that map only matches *their* run. Added a new
verification cell that builds the *real* map for this run using `digits.target` majority-vote per cluster
(same idea as Iris's `pd.crosstab()` evaluation) — real decoded result: **`4, 4, 7, 4`**, i.e. **0/4 correct**,
with three different true digits collapsing into one cluster. Framed in Step 18 as a genuine illustration of
K-Means' limitations (raw pixel similarity, no digit-identity concept) and of the 1990s-Turkish-handwriting
training data not generalizing to mouse-drawn digits.

## Sublesson checklist
Mark `[x]` when the step's code/text is folded into the notebook **from the real source**.

- [x] **Step 1** — import `datasets`, `load_digits()`, print `digits`
- [x] **Step 2** — print `digits.DESCR`; Q&A: image size (8x8 px), dataset origin (UCI ML repo test set, NIST-preprocessed)
- [x] **Step 3** — print `digits.data`
- [x] **Step 4** — print `digits.target`
- [x] **Step 5** — visualize `digits.images[100]` via `plt.matshow`; confirmed label = 4
- [x] **Step 6** — import `KMeans` from `sklearn.cluster`
- [x] **Steps 7-8** — choose k=10 (Socratic Q&A captured), `model = KMeans(n_clusters=10)`, `model.fit(digits.data)` — combined into one cell, model reused everywhere after
- [x] **Steps 9-11** — visualize all 10 cluster centroids as 8x8 images (figure + subplot grid + show), using the shared fitted model
- [x] **Step 12** — optional sklearn example, noted/skipped
- [x] **Steps 13-14** — `test.html` hand-drawing → array (external/interactive, described only)
- [x] **Step 15** — real `new_samples` array (4 hand-drawn digits, user drew `2, 0, 2, 9`) folded in
- [x] **Step 16** — `model.predict(new_samples)` → real captured `new_labels = [6 6 4 6]`
- [x] **Step 17** — index→digit decoding loop (faithful to instructions' hardcoded map); real output `4464`, flagged as unreliable for this run
- [x] **Verification cell (added)** — empirical cluster→digit map via `digits.target` majority vote; real decoded result `4, 4, 7, 4` — 0/4 match vs. what user actually drew
- [x] **Step 18** — reflection written with real verified comparison and analysis of why the model likely failed
- [x] Q/A captured (k derivation + Step 10 argument breakdown)
- [x] Real Codecademy exercise code swapped in (faithful transcription)
- [x] TL;DR written

## How we work this lesson (so Claude stays consistent)
1. You paste/describe a **step's instructions** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** → I add it to the relevant section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell, cumulative script style (matches how the
   prior `Unsupervised_KMeans_Lesson.ipynb` handled the hard-way K-Means steps).
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify notebook runs clean, commit if asked.

## Open questions / to verify
- None — project complete, notebook verified end-to-end, TL;DR written.
