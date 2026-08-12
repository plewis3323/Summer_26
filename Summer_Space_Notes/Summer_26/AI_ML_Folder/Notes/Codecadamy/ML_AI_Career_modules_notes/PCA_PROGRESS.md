# PCA (Principal Component Analysis) — Progress Tracker

**Lesson:** Codecademy MLE Path → Unsupervised Learning → *PCA article + PCA in Python* (last lesson of the ML Fundamentals block)
**Note file:** `PCA_Lesson.ipynb`
**Started:** 7/8/26

---

## Status: ✅ DONE

Both the PCA article and the full **PCA in Python** lesson are folded in and verified (`nbconvert --execute --inplace`,
exit 0, final pass 7/9/26). Every sublesson uses real Codecademy checkpoint code and real datasets except the intro
article's `pizza.csv`/`pizza_new.csv` (synthetic stand-ins, since those files were never provided — see the open
checklist item below). TL;DR added at the end covering both halves of the lesson. One real discrepancy is flagged
in-notebook rather than papered over: the "PCA as Features" checkpoint's actual scores contradict the lesson text's
claim (16 original features scored *higher* than 4 PCA features on this dataset/split).

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded into the notebook **from the real source**.

- [x] **PCA article — Intro to PCA** (why dimensionality reduction matters, what PCA does)
- [x] **PCA article — Laying the groundwork** (variance as information; Coefficient of Variation; CV code on synthetic `pizza.csv`) — verified
- [x] **PCA article — Coding question (`pizza_new.csv` CV ranking)** (user's exercise scaffold folded in faithfully; `df` renamed to `df_new` to avoid clobbering the `df` used by later cells — flagged to user; `importance_rank = df_new.apply(cv).sort_values(ascending=False).index.tolist()`) — verified on synthetic `pizza_new.csv` stand-in
- [x] **PCA article — The Math Behind PCA** (data matrix, covariance matrix formula + `df.cov()`, eigenvalues/eigenvectors, $\det(A-\lambda I)=0$) — verified
- [x] **PCA article — Principal Components in practice** (`sns.scatterplot`, `sns.pairplot`, `sklearn.decomposition.PCA` reducing 5 features -> 2 components, plotting the reduced data) — verified
- [x] **PCA article — Fill-in-the-blank quiz** (covariance matrix -> matrix factorization -> eigenvectors/Principal Components + eigenvalues) — answered directly, resolved from article text
- [x] **PCA article — How, Where, and Why of PCA** (downstream ML input, PCA as unsupervised clustering / preprocessing, image processing use case)
- [x] **PCA in Python — Introduction to Implementing PCA** (lesson goals: NumPy PCA step-by-step, sklearn PCA, PCA features into an ML model, visualize PCA on image data; dry bean dataset intro)
- [x] **PCA in Python — Loading the dataset** (`Dry_Beans.csv` — real dataset, moved from Desktop + cleaned; `df.head()`, split into `data_matrix` (dropped `Class`) and `classes`) — verified, real Codecademy checkpoint code
- [x] **PCA in Python — Implementing PCA in NumPy I** (real Codecademy checkpoint code: `data_matrix.csv` -> `.corr()` -> correlation heatmap (`sns.diverging_palette` + `sns.heatmap`) -> `np.linalg.eig()` for 16 eigenvalues + 16 eigenvectors) — verified, real checkpoint code
- [x] **PCA in Python — Implementing PCA in NumPy II - Analysis** (real Codecademy checkpoint code, split into its two checkpoint cells; loads `eigenvalues.csv` -> `info_prop` -> scree plot; `cum_info_prop = np.cumsum(info_prop)` -> cumulative plot with `hlines`/`vlines` at the 95%/4-component crossing) — verified, real checkpoint code. Changed the scaffold's `plt.clf()` after the scree plot to `plt.show()` since these are now separate cells (flagged to user) — otherwise fine as-is.
- [x] **PCA in Python — sklearn implementation** (`sklearn.decomposition.PCA`; standardize `data_matrix` -> `data_matrix_standardized` -> `pca.fit(...).components_` for eigenvectors -> `pca.explained_variance_ratio_` for the variance ratios; real Codecademy checkpoint code (3 checkpoints), real dry-bean `data_matrix` — swapped in verbatim after user pasted the actual exercise; dropped the platform-only `import codecademylib3` line since it isn't installable locally, flagged to user) — verified, confirms first 4 components ~95% of variance matching the earlier NumPy scree-plot finding
- [x] **PCA in Python — Projecting the data onto the principal axes** (real Codecademy checkpoint code, real data: `data_matrix_standardized.csv` + `classes.csv` downloaded from Codecademy and moved from Desktop into this notebook's directory; `PCA(n_components=4)` since 4 PCs hit ~95% of variance; `pca.fit_transform()` -> `data_pcomp` with columns `PC1`-`PC4`; `sns.lmplot(x='PC1', y='PC2', hue='bean_classes', fit_reg=False)`; dropped the platform-only `import codecademylib3` line, flagged to user) — verified, dry bean classes visibly cluster using just PC1/PC2
- [x] **PCA in Python — PCA as Features / training a model on principal components** (real Codecademy checkpoint code: `data_matrix_standardized.csv` + `classes.csv`, `PCA(n_components=4)` -> `LinearSVC` vs. the same model on all 16 features, `train_test_split(..., test_size=0.33, random_state=42)`) — verified, but flagged a real discrepancy: actual scores are 4-PC = 0.847 vs. 16-feature = 0.917 (16 features wins), contradicting the lesson text's claim that 4 PCs score higher; also noted `.score()` is mean accuracy, not "average likelihood" as the lesson calls it
- [x] **PCA in Python — PCA for Images I** (real Codecademy checkpoint code: `sklearn.datasets.fetch_olivetti_faces()` — downloads/caches to `~/scikit_learn_data`, 400 images x 4096 pixels; standardize -> `side_length = sqrt(n_features)` = 64 -> plot first 15 faces reshaped to 64x64 with `plt.cm.bone`; dropped the platform-only `import codecademylib3` line) — verified, real download succeeded
- [x] **PCA in Python — PCA for Images II** (real Codecademy checkpoint code + real `faces_standardized.csv`, moved from Desktop; `PCA(n_components=40)` -> `.components_` = eigenfaces plotted 64x64 -> `pca.transform()` + `pca.inverse_transform()` to reconstruct faces from the compressed components; dropped the platform-only `import codecademylib3` line; code reflects the checkpoint's second setting (`n_components=40`, ~0.09% of original pixels) rather than the initial `n_components=400`, per the exercise instructions) — verified
- [x] **Review** — no separate review sublesson was given; the TL;DR below serves as the recap
- [x] Q/A captured (article Q&A section + sklearn-implementation Q&A both folded in)
- [ ] Real Codecademy checkpoint code / real `pizza.csv` & `pizza_new.csv` files — never provided; article section still runs on synthetic stand-ins matching the article's reported stats

## How we work this lesson (so Claude stays consistent)
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** for that sublesson → I add it to the **Q & A** section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant section.
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify notebook runs clean, commit if asked.

## Open questions / to verify
- Confirm Codecademy's exact sublesson titles & order for this PCA lesson.
- Does the lesson use a specific dataset? Swap in the real one when given.
