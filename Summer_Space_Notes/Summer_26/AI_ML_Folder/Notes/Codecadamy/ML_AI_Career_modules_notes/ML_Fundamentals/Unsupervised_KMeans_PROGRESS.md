# Unsupervised Learning & K-Means — Progress Tracker

**Lesson:** Codecademy MLE Path → Unsupervised Learning → *Intro article + K-Means Clustering*
**Note file:** `Unsupervised_KMeans_Lesson.ipynb`
**Started:** 7/2/26

---

## Status: ✅ DONE (7/7/26)
Full lesson folded in from real Codecademy source: intro article, K-Means theory, Iris dataset, hard-way
K-Means (Steps 1-4), scikit-learn `KMeans`, inference on new data, visualization, cluster evaluation
(crosstab), and choosing k via inertia + the elbow method. TL;DR written. Whole notebook verified to run
top-to-bottom clean (`nbconvert --execute`, exit 0) on final pass. Centroids/labels are random (no seed),
so printed/plotted values vary per run — that's expected and noted throughout.

**Not covered:** the lesson's final "Review" sublesson text was never pasted in, so there's no dedicated
Review section — the TL;DR serves that purpose instead.

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded into the notebook **from the real source**.

- [x] **Intro article — What is Unsupervised Learning?** (supervised vs unsupervised; 3 uses: clustering, dim-reduction, auto-labelling; module covers PCA + K-Means)
- [x] **K-Means — Intro to Clustering** (clustering = finding groups in unlabelled data; apps: recsys, search, market seg, image seg, text; Iris = 3 natural clusters)
- [x] **K-Means — the algorithm overview** (k = #clusters, Means = avg dist to centroid; iterate: place k centroids → assign → recompute → repeat to convergence; training vs inference)
- [x] **Iris dataset** (`datasets.load_iris()`; 150 samples x 4 features: sepal len/wid, petal len/wid; 3 species = target clusters) — code verified
- [x] **Iris — exploring data/target/DESCR** (real exercise: `.data`, `.target` = ground truth, `.DESCR`; Q's answered — published 1936, units cm; commented out Codecademy-only `codecademylib3_seaborn` import) — verified
- [x] **Visualize before K-Means** (real exercise: `samples[:,0]` sepal length vs `samples[:,1]` sepal width scatter; NumPy `[:,col]` = `[all_rows, column]`) — verified
- [x] **K-Means Step 1 — place k random centroids** (k=3; `np.random.uniform(min,max,k)` for x/y; `np.array(list(zip(...)))` -> (k,2); plot centroids over samples) — real checkpoint solution folded in & verified. (Removed my earlier seed to match the exact exercise; output is random per run.)
- [x] **K-Means Step 2 — assign points to nearest centroid** (Euclidean vs taxicab; `distance()` fn; `assign_to_centroid()` helper + `np.argmin`; loop fills `labels` 0/1/2) — real checkpoint solution folded in & verified. (Their code passes full 4D `samples[i]` but `distance()` only uses dims 0-1, so effectively sepal len/width.)
- [x] **K-Means — Step 3: update centroids (mean of assigned points)** (`deepcopy` old centroids first; for each of k clusters, collect assigned `sepal_length_width` points into `points`, then `centroids[i] = np.mean(points, axis=0)`) — real checkpoint solution folded in & verified.
- [x] **K-Means — convergence / iterate until stable (Step 4)** (Steps 2-3 wrapped in `while error.all() != 0:`; `error[i] = distance(centroids[i], centroids_old[i])` tracks per-centroid movement; final colored scatter plot of the 3 clusters + diamond centroid markers) — real checkpoint solution folded in & verified. Noted a subtlety: `.all()` exits the loop as soon as *any one* centroid stops moving, not once *all* have converged.
- [x] **K-Means with scikit-learn** (`sklearn.cluster.KMeans`) (`from sklearn.cluster import KMeans`; `model = KMeans(n_clusters=3)`; `model.fit(samples)` does Steps 2-3 to convergence in one call; `model.predict(samples)` = inference, using all 4 Iris features) — real checkpoint solution folded in & verified.
- [x] **New Data? — inference on unseen samples** (fitted `model.predict(new_samples)` on 3 brand-new iris measurements; `iris.target_names[label]` to convert numeric labels to species names) — real checkpoint solution folded in & verified.
- [x] **Visualize After K-Means** (sepal length vs. width scatter, colored by predicted cluster via `plt.scatter(x, y, c=labels, alpha=0.5)`) — real checkpoint solution folded in & verified.
- [x] **Evaluating the Clusters** (species names via `iris.target_names[...]`, `pd.DataFrame`, `pd.crosstab(df['labels'], df['species'])`; Q&A grounds the "how accurate?" reflection question in a real captured crosstab, ~89% accuracy, setosa perfectly separated) — real checkpoint solution folded in & verified.
- [x] **Choosing k** (elbow method / inertia) (`model.inertia_`; loop k=1..8, collect inertias, `plt.plot(num_clusters, inertias, '-o')`; elbow at k=3 for Iris, matching the 3 real species) — real checkpoint solution folded in & verified.
- [ ] **Review** (never pasted — TL;DR stands in for it)
- [x] Q/A captured
- [x] Real Codecademy exercises swapped in

## How we work this lesson (so Claude stays consistent)
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** for that sublesson → I add it to the **Q & A** section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant section.
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify notebook runs clean, commit if asked.

## Open questions / to verify
- Confirm Codecademy's exact sublesson titles & order vs. this scaffold.
- Does the lesson use a specific dataset (e.g. Iris)? Swap in the real one.
- Does it cover the **elbow method** for choosing k, or just fixed k?
