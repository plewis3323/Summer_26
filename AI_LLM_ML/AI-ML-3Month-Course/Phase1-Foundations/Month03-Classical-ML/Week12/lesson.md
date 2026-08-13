# Week 12 — Unsupervised Learning + Capstone 1

~3 hrs on the unsupervised methods, then the rest of the week into the capstone
(`project.md`). Before starting you should be able to: derive PCA from variance
maximization (Week 07); apply Bayes' theorem and write a Gaussian likelihood
(Week 08); split data, run k-fold CV, and name leakage (Week 09); train a
classifier, read a ROC, calibrate probabilities, and choose a working point
(Week 10); train a random forest and XGBoost and say why trees win on tabular
data (Week 11). You already have a tuned MAGIC BDT from Week 11; this week you
put it inside an honest pipeline and ship Phase 1.

Supervised learning needs a label for every example — gamma vs hadron, signal vs
background. A lot of detector data has no such oracle. **Unsupervised learning**
asks what structure is in the features themselves. First half: k-means, then
Gaussian mixtures via EM, then what you may and may not read off a PCA or
UMAP embedding. Second half: a Week-24-style "how to run the week" for
Capstone 1, pointing at `project.md` for the build steps.

## 1. Unsupervised vs supervised

Week 09's frame: examples $x$, labels $y$, a hypothesis class, a loss, a
discipline for estimating true risk. Drop the labels. What remains is a cloud
$x_1, \dots, x_n \in \mathbb{R}^d$ and the question "what structure is in it?"

Three jobs, all this week:

- **Clustering** — partition into internally similar groups (k-means, GMMs).
  No $y$, so "right" means "useful description", not "this AUC".
- **Dimensionality reduction** — a shorter vector that keeps the structure
  you care about. PCA (Week 07) keeps global variance directions; UMAP tries
  to keep local neighborhoods.
- **Density estimation** — a $p(x)$ on the cloud. A GMM is a density;
  k-means is a hard, degenerate cousin of one.

None of these replaces a classifier. MAGIC *has* labels; Capstone 1 is
supervised. Unsupervised tools are for looking: "do the classes already
separate in 2D?" is a legitimate exploratory question, and a dangerous one
if you then *tune the classifier on the picture*. Exploration on the training
fold; decisions on validation; the number from the test set. Same Week 09
rule, new plots.

## 2. k-means

You are handed $n$ points and a number of groups $K$ (chosen by you — it is a
hyperparameter, not estimated by the algorithm). **k-means** partitions the
points into $K$ groups and represents each group by a single point, its
**centroid** $\mu_k$ (the mean of the points assigned to it). The assignment of
point $i$ to cluster $k$ is a hard yes/no: write $r_{ik} = 1$ if $x_i$ belongs
to cluster $k$ and $0$ otherwise, with $\sum_k r_{ik} = 1$ (each point in
exactly one cluster).

### 2.1 Lloyd's algorithm and the WCSS objective

The quantity k-means actually minimizes is the **within-cluster sum of squares
(WCSS)**, also called inertia:

$$J(\{r_{ik}\}, \{\mu_k\})
 = \sum_{i=1}^{n}\sum_{k=1}^{K} r_{ik}\,\|x_i - \mu_k\|^2.$$

Read it: every point pays the squared Euclidean distance to *its* centroid, and
we add them up. Small $J$ means compact, well-separated-from-their-own-center
clusters. It does *not* mean the clusters are the "true" groups in nature — it
means this particular geometric description is tight.

**Lloyd's algorithm** (the thing everyone means by "k-means") alternates two
steps from a starting guess of the centroids:

1. **Assignment.** For each point, set $r_{ik} = 1$ for the centroid $\mu_k$
   closest to it, $0$ for the others. Ties: pick one, it almost never matters.
2. **Update.** For each cluster, set $\mu_k$ to the mean of the points assigned
   to it. Empty cluster: re-seed that centroid (sklearn does this; so should
   you).

Repeat until assignments stop changing. One seed, in NumPy:

```python
import numpy as np

rng = np.random.default_rng(0)
mu = X[rng.choice(len(X), size=K, replace=False)]   # K random points as centroids
for _ in range(100):
    r = ((X[:, None, :] - mu) ** 2).sum(axis=2).argmin(axis=1)   # nearest
    mu_new = np.stack([X[r == k].mean(axis=0) for k in range(K)])
    # empty cluster: mean-of-empty will NaN — re-seed that row, as sklearn does
    if np.allclose(mu_new, mu):
        break
    mu = mu_new
J = ((X - mu[r]) ** 2).sum()                        # WCSS for this seed
```

Run that from many seeds and keep the smallest $J$ (`n_init`). scikit-learn
adds better seeding (k-means++, which spreads the initial centroids) and the
same `.fit` API as Week 09:

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, random_state=0)
km.fit(X)                    # km.labels_, km.cluster_centers_, km.inertia_
```

`inertia_` is exactly $J$. Exercise E1 has you implement Lloyd on 2D blobs,
plot centroid tracks, and match sklearn to $10^{-4}$ from the best of 20 seeds.

### 2.2 Coordinate descent, monotonicity, and local minima

Why does Lloyd work, and why does it get stuck? $J$ is a function of two blocks
of variables: the assignments $\{r_{ik}\}$ and the centroids $\{\mu_k\}$.
**Coordinate descent** means: freeze one block, minimize over the other; swap;
repeat. That is Lloyd, and each half has a closed-form minimizer.

**Assignment step, $\mu$ frozen.** $J$ splits into a sum over points. For point
$i$ the inner sum is $\|x_i - \mu_k\|^2$ for whichever $k$ we pick, so the
minimizer is "assign to the nearest centroid". $J$ cannot increase.

**Update step, $r$ frozen.** $J$ splits into a sum over clusters. For cluster
$k$,

$$J_k(\mu_k) = \sum_{i:\, r_{ik}=1} \|x_i - \mu_k\|^2.$$

Differentiate (Week 05) and set to zero: $\nabla_{\mu_k} J_k =
-2\sum_{i \in C_k}(x_i - \mu_k) = 0$, so $\mu_k$ is the mean of its points.
(Week 08: this is also the MLE for the mean of a spherical Gaussian.) $J$
cannot increase.

So $J$ is **monotone non-increasing** along the path Lloyd takes, and $J \ge 0$,
so the sequence of $J$ values converges. That is the guarantee — and it is the
accept criterion in E1: plot $J$ vs iteration and it must never tick up.

What the guarantee does *not* say: that you have found the global minimum.
$J(r, \mu)$ is not jointly convex. Different starting centroids fall into
different basins. A standard failure: one centroid initialized inside a tight
blob and the other two sharing a second blob — Lloyd happily reports a split
of the second blob and a wasted singleton, with $J$ locally minimal and
globally silly. The practical fix is the `n_init` loop above: run from many
seeds, keep the smallest $J$. k-means++ (sklearn's default) spends the seed
budget more intelligently by placing new centroids far from existing ones, but
it is still a heuristic. Always report the objective, not just the picture.
$K$ is a choice, not an estimate: $J$ always drops as you add clusters, so
"minimize $J(K)$" is meaningless. On labeled MAGIC you already know there are
two physics classes; you might still cluster hadrons alone to ask whether
that class is a mixture of image-shapes.

## 3. Gaussian mixtures and EM

k-means gives hard assignments and spherical clusters of equal weight. Real
clouds overlap, stretch, and have different sizes. A **Gaussian mixture model
(GMM)** is a density that can describe that: the data are drawn from one of
$K$ Gaussians, and we do not observe which.

### 3.1 The model

A **multivariate Gaussian** in $d$ dimensions (Week 08, now with a covariance
instead of scalar $\sigma^2$) has density

$$\mathcal{N}(x \mid \mu, \Sigma)
 = (2\pi)^{-d/2}\,(\det\Sigma)^{-1/2}
   \exp\!\Big(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\Big).$$

$\mu$ is the center; $\Sigma$ is the $d \times d$ **covariance matrix**
(symmetric, positive definite) — spreads $\Sigma_{jj}=\mathrm{Var}(x_j)$ and
tilts $\Sigma_{jk}=\mathrm{Cov}(x_j,x_k)$. Level sets are ellipsoids. A
**mixture** of $K$ of them is

$$p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k),$$

with **mixing weights** $\pi_k \ge 0$, $\sum_k \pi_k = 1$. Generative story:
pick cluster $k$ with probability $\pi_k$, then draw $x$ from that Gaussian.
Overlapping blobs, not a partition.

### 3.2 Complete-data log-likelihood

If we observed the cluster identity of every point, fitting would be easy.
Introduce a latent one-hot vector $z_i \in \{0,1\}^K$ with $z_{ik} = 1$ iff
point $i$ came from cluster $k$. The **complete data** are the pairs
$(x_i, z_i)$. Their likelihood (independence across $i$) is

$$p(X, Z \mid \theta)
 = \prod_{i=1}^{n}\prod_{k=1}^{K}
   \Big[\pi_k\,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)\Big]^{z_{ik}},$$

where $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$. The **complete-data
log-likelihood** is therefore

$$\ell_c(\theta; X, Z)
 = \sum_{i=1}^{n}\sum_{k=1}^{K} z_{ik}
   \Big[\log\pi_k + \log\mathcal{N}(x_i \mid \mu_k, \Sigma_k)\Big].$$

If the $z_{ik}$ were known, this would separate: cluster $k$'s points would
give the ordinary Gaussian MLE for $(\mu_k, \Sigma_k)$ and $\pi_k$ would be
the fraction of points in $k$. They are not known. **Expectation-maximization
(EM)** is the algorithm that fills them in with their expectations under the
current parameters, then maximizes, then repeats.

### 3.3 E-step: responsibilities are Bayes

The **E-step** replaces each unknown $z_{ik}$ by its posterior probability
given $x_i$ and the current $\theta$ — the **responsibility** $\gamma_{ik}$:

$$\gamma_{ik}
 := \mathbb{E}[z_{ik} \mid x_i, \theta]
 = p(z_{ik} = 1 \mid x_i, \theta).$$

This is Week 08's Bayes theorem with a new name. Prior $p(k) = \pi_k$,
likelihood $p(x_i \mid k) = \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$, evidence
the mixture density:

$$\boxed{\;\gamma_{ik}
 = \frac{\pi_k\,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
        {\sum_{j=1}^{K}\pi_j\,\mathcal{N}(x_i \mid \mu_j, \Sigma_j)}\;}$$

$\gamma_{ik}$ is a *soft* assignment: point $i$ can be 0.7 cluster 1 and 0.3
cluster 2. The $\gamma$'s for one point sum to 1. That is the entire E-step.

### 3.4 M-step

The **M-step** maximizes the expected complete-data log-likelihood
$\mathbb{E}_{Z \mid X, \theta^{\mathrm{old}}}[\ell_c]$ — which is just
$\ell_c$ with $z_{ik}$ swapped for $\gamma_{ik}$. Define the effective count
$N_k = \sum_i \gamma_{ik}$ (soft number of points in cluster $k$). Setting
derivatives to zero (Lagrange multiplier on $\sum_k \pi_k = 1$ for the
weights; matrix calculus from Week 07 for $\Sigma_k$) gives

$$\pi_k = \frac{N_k}{n}, \qquad
\mu_k = \frac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}\, x_i, \qquad
\Sigma_k = \frac{1}{N_k}\sum_{i=1}^{n}
  \gamma_{ik}\,(x_i - \mu_k)(x_i - \mu_k)^\top.$$

These are the usual MLE formulas with each point *weighted* by how much it
belongs to the cluster. Iterate E, M, E, M. Each full EM step is guaranteed
not to decrease the *observed-data* log-likelihood
$\ell(\theta) = \sum_i \log p(x_i \mid \theta)$ — that is EM's theorem, and
E2's accept criterion (monotone $\ell$, parameters matching sklearn to
$10^{-3}$ from the same init).

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type="full",
                      n_init=5, random_state=0)
gmm.fit(X)
print(gmm.weights_, gmm.means_.shape)     # π, and (K, d)
gamma = gmm.predict_proba(X)              # responsibilities, shape (n, K)
```

`covariance_type="diag"` or `"tied"` shrinks the number of covariance
parameters when $n$ is small relative to $d$ — a full $\Sigma_k$ has
$d(d+1)/2$ entries, and ten Hillas features already make that a lot.

### 3.5 k-means as the $\varepsilon \to 0$ limit

Put the same isotropic covariance $\Sigma_k = \varepsilon I$ on every
component, and equal mixing weights. The log-density of component $k$ at $x_i$
is $-\|x_i - \mu_k\|^2 / (2\varepsilon)$ plus a $k$-independent constant. As
$\varepsilon \to 0$, the component with the *smallest* $\|x_i - \mu_k\|^2$
dominates the Bayes posterior exponentially: $\gamma_{ik} \to r_{ik}$, a hard
assignment to the nearest centroid. The M-step for $\mu_k$ becomes the
ordinary mean of the assigned points. Lloyd's algorithm is EM in this hard,
spherical, equal-weight limit. That is why k-means clusters are spherical and
why overlapping blobs want a GMM instead.

## 4. PCA in practice vs UMAP

Week 07 derived PCA two ways — maximize projected variance, or minimize
reconstruction error — and showed they agree: the top-$k$ principal axes are
the top-$k$ eigenvectors of the covariance (equivalently, the top-$k$ right
singular vectors of the centered data matrix). This week you *use* it, and you
meet a nonlinear cousin.

**What PCA preserves.** PCA is a rotation (and optional truncation). PC1 is
the longest direction in the cloud; PC2 is the longest remaining direction
orthogonal to it; and so on. In the full $d$ dimensions, distances are the
original Euclidean ones. In a truncated $k$-D projection they are those
distances *after dropping the low-variance axes* — a global statement. The
axes are linear combinations of named features, so you can ask what PC1 is
made of.

**What UMAP tries to preserve.** **UMAP** (Uniform Manifold Approximation and
Projection) is a nonlinear embedding: it builds a weighted graph of
*local neighborhoods* in the original space (each point connected to its
nearest neighbors), then finds a 2D (or 3D) layout of points whose local
neighborhood graph looks like that one. The objective cares about nearby
points staying nearby. It does not, as a goal, preserve global distances,
global angles, cluster sizes, or densities.

**What you may not read off an embedding.** The pretty picture hides this.

- **UMAP:** inter-cluster distances, blob sizes, and densities are not
  preserved — weakly connected groups get pushed to opposite corners because
  that satisfies the local constraints; a tight cluster and a diffuse one can
  occupy similar area. Seed, rotation, and flip rearrange the islands; the
  islands' *internal* structure is the signal, their placement on the page is
  not.
- **PCA:** a curved sheet ("manifold") can look overlapped in the top-2 PCs
  even when the classes separate along the curve. Linear axes miss that.
- **Both:** coloring MAGIC by gamma/hadron after embedding is fine
  exploration. Choosing a classifier cut from the picture, or plotting the
  test set to admire "separation", is leakage with extra steps.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap                         # uv add umap-learn

X_std = StandardScaler().fit_transform(X_train)     # PCA wants comparable axes
Z_pca = PCA(n_components=2, random_state=0).fit_transform(X_std)
Z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1,
                   random_state=0).fit_transform(X_std)
# plot Z_pca and Z_umap, colored by class; caption: one distortion each
```

Scale first: length in mm and an angle in degrees must not compete for
"largest variance" on units. Trees did not care (Week 11); PCA does, because
it *is* a variance method. UMAP uses nearest neighbors, so scale still
matters. E3: both embeddings, shared legend, caption stating one distortion
of each. The accept is the caption, not the prettiness.

## 5. How to run Capstone 1

The unsupervised half is the week's new math. The rest of the hours go into
the build, specified in `project.md`. This section is about *how to run that
build* without fooling yourself — the same job Week 24's lesson does for
Capstone 2.

### 5.1 What this capstone proves

Capstone 1 proves the Phase 1 claim: a real tabular physics classifier, from
a raw file through nested CV, calibration, and working points, to a
*defensible* number — tests that lock it, a writeup that includes what failed
first. The artifact is the repo. The gate also wants PCA re-derived cold (the
month's rotating flagship). Syllabus §6 slack exists for capstones; take a
second calendar week if you need it and log it. Cut a model from the
comparison if you must; never cut tests, calibration, the leakage audit, or
the failure section.

### 5.2 The physics, from scratch

The dataset is the **MAGIC Gamma Telescope** sample from the UCI ML
Repository: 19,020 showers, 10 real-valued features, a binary label `g`
(gamma) or `h` (hadron). Read the UCI feature list; here is what the
experiment *is*, from zero.

A charged particle moving through a medium faster than light travels *in that
medium* (light is slowed by the refractive index; the particle is not) emits a
shock-wave of photons — the optical analog of a sonic boom. That light is
**Cherenkov radiation**. Earth's atmosphere is the medium.

A high-energy **gamma ray** (a photon, typically GeV–TeV for this telescope)
hits a nucleus in the upper atmosphere and converts to an electron–positron
pair; those radiate more photons, which pair-produce again. The cascade is an
**electromagnetic shower**: a pancake of $e^\pm$ and photons, compact and
regular, plunging down. The charged particles in it emit Cherenkov light, a
faint nanosecond flash. A dish on the ground images that flash onto a camera
of photomultiplier pixels.

A **hadron** — a proton or nucleus from ordinary cosmic rays — also makes an
atmospheric shower, but a **hadronic shower**: it hits a nucleus and produces
pions, which produce more pions, plus a messy electromagnetic sub-cascade
from $\pi^0 \to \gamma\gamma$, plus penetrating muons. The Cherenkov image is
lumpier, wider, and less regularly elliptical.

**MAGIC** (Major Atmospheric Gamma-ray Imaging Cherenkov) is a pair of 17 m
telescopes on La Palma. The camera image is a blob. **Hillas parameters**
(Hillas, 1985) summarize it as an ellipse: `fLength`/`fWidth` (axes),
`fSize` (brightness), concentration (`fConc`, `fConc1`), asymmetry and third
moments, `fDist` (center-to-camera-center), `fAlpha` (angle between the major
axis and the line to the camera center). Gamma images tend to be slimmer,
more concentrated, and — when pointing at a source — better aligned (small
`fAlpha`). Hadrons are the opposite on average, with enough overlap that a
single width cut fails.

Ten heterogeneous, correlated-by-construction features, $n \sim 2\times 10^4$:
Week 11's **tabular** / **ntuple** regime. BDT home turf. Your Week 11 MAGIC
model is the right *model class*; this week you put it in the right
*protocol*.

### 5.3 The pipeline

Work the `project.md` steps in order. The shape, matching the roadmap:

**ingest → nested CV (logistic vs random forest vs XGBoost) → calibration →
working points.**

Ingest: seeded CSV load, documented 10 Hillas columns plus label, a
trainval/test split frozen *before* any model sees a number, and a
`Pipeline` so scaling lives inside the estimator (Week 09's leakage fix).
Logistic needs the scaler; trees do not — a scaler in a forest pipeline is
harmless and keeps one API.

Logistic is the linear baseline (Week 10): if it already separates, the story
is "the ellipse is linearly informative", not "XGBoost is magic". The forest
is the almost-tuning-free ensemble (Week 11 §2). XGBoost is the class Week 11
argued for on 10 tabular features. Retrain it *inside this repo, under this
protocol*. Quoting last week's AUC is the sin Week 24 names: different split,
budget, possibly preprocessing.

### 5.4 Nested CV

Week 09 taught k-fold CV: it estimates a *procedure*, and a score you used to
pick hyperparameters is optimistic. If the same CV loop both chooses
`max_depth` and reports "the model's AUC", you kept the setting that got
lucky on those folds. **Nested CV** spends that idea:

- **Outer loop** — scores "tune, then fit". Each outer fold is held out
  *completely* from tuning.
- **Inner loop** — on the remaining data, ordinary CV (or a grid search)
  picks hyperparameters.

Outer-fold scores estimate the *tuned procedure*. Hyperparameters chosen on
an outer-fold's test data are a gate failure. Then refit the winner (almost
certainly XGBoost; if not, that is a finding) on all of trainval, and touch
the test set once.

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
search = GridSearchCV(estimator, param_grid, cv=inner, scoring="roc_auc")
# outer scores: the tuned *procedure*, no outer-fold labels in the inner search
scores = cross_val_score(search, X_trainval, y_trainval, cv=outer,
                         scoring="roc_auc")
print(f"nested CV AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
```

Report mean ± fold spread for every model in the comparison, same outer
splitter, same scoring. That is the identical-protocol rule applied to three
classical models rather than to "new architecture vs baseline".

### 5.5 Calibration and working points

XGBoost (and forests) rank well and are often miscalibrated (Week 10 §9).
**Calibrate** on held-out data — Platt scaling is the robust default here —
and put the reliability diagram in the repo. A "90% gamma efficiency"
threshold on miscalibrated probabilities delivers some other efficiency.

Two working points, chosen on validation, reported on test (Week 10 §8):

- **Fixed efficiency.** Threshold where gamma TPR crosses 0.90; report the
  hadron rejection (or FPR) you bought, plus the confusion matrix.
- **Max $S/\sqrt{S+B}$.** Sweep the threshold, take the maximum of that
  significance proxy, same report.

`pytest` should reproduce both to 1% on the held-out set. A plot the tests
do not lock is decoration.

### 5.6 The identical-protocol rule and the test set

Same splits, seeds, outer folds, comparable budgets. Logistic numbers exist
*before* you spend the XGBoost budget — otherwise every tweak is contaminated
by knowing the BDT's score. The test set is touched **once**, at the end
(ROC, working-point metrics, held-out calibration check). Nested-CV numbers
come from trainval; thresholds are chosen on validation, not test.

**Leakage audit** (writeup section): scaler fit on all rows before the split?
Duplicate showers on both sides? Any feature a function of the label? MAGIC
is one row per image, so a row split *is* the event split — say so. Hillas
features are correlated by construction (Week 11 §6); that is not leakage,
but a permutation-importance story needs the correlated-group caveat.

## 6. The Phase 1 gate

Syllabus §5, as a checklist:

- [ ] `pytest -q` green on a fresh clone after `uv sync`.
- [ ] Nested-CV table: logistic vs RF vs XGBoost, outer-fold mean ± spread,
      no hyperparameter chosen on outer-fold test data.
- [ ] ROC and reliability (calibration) plots checked into the repo.
- [ ] Two working points, chosen on validation, reported on test, locked by
      tests to 1%.
- [ ] Leakage audit written: what you checked, what you found.
- [ ] `writeup.md` (~2 pages): problem, data, method, results table,
      calibration figure, **what failed first**, limitations. Every number
      generated by the pipeline.
- [ ] Tag `month-03-complete`, write `retro.md`, open one issue. PCA
      re-derived cold (the month's rotating flagship) and filed.

A model that loses to logistic can pass if the writeup diagnoses it. An
unexplained win cannot. That is the Capstone 2 policy, practiced on trees.
The build order is in `project.md`. Work it in order. Then close the phase.

## Check yourself

1. k-means assignment cannot increase $J$, and neither can the centroid
   update. Why, in one sentence each? Why does that *not* imply a global
   minimum?
2. You run k-means from one seed and get a pretty picture. What must you
   still look at, and what does `n_init` actually do to $J$?
3. Write the GMM responsibility $\gamma_{ik}$ from Bayes' theorem. Which
   pieces are the prior, the likelihood, and the evidence?
4. Write the M-step updates for $\pi_k$, $\mu_k$, $\Sigma_k$. What is $N_k$?
5. Explain the $\varepsilon \to 0$ limit that recovers k-means. Which GMM
   assumptions become k-means assumptions?
6. Name one thing PCA preserves that UMAP does not, and one thing you must
   *not* read off a UMAP plot of MAGIC.
7. Why is nested CV's outer loop the number you quote for "how well does
   tuned XGBoost do?", rather than the inner loop's best score?
8. Why must the Week 11 MAGIC BDT be retrained inside the capstone repo
   rather than its CV AUC quoted? On which split do you choose the 90%
   efficiency threshold, and on which do you report it?

## Answers

1. Assignment: with centroids frozen, each point independently chooses the
   nearest $\mu_k$, which is the per-point minimizer of its term in $J$.
   Update: with assignments frozen, $J_k(\mu_k)$ is minimized uniquely at the
   mean of cluster $k$. $J(r,\mu)$ is not jointly convex, so a monotone
   coordinate descent can stop at a local minimum.
2. The value of $J$ (and, ideally, $J$ vs iteration — monotone). `n_init`
   reruns Lloyd from different seeds and keeps the smallest $J$; the pretty
   picture from one seed may be a bad basin.
3. $\gamma_{ik} = \pi_k \mathcal{N}(x_i\mid\mu_k,\Sigma_k) /
   \sum_j \pi_j \mathcal{N}(x_i\mid\mu_j,\Sigma_j)$. Prior $\pi_k$,
   likelihood $\mathcal{N}(x_i\mid\mu_k,\Sigma_k)$, evidence the mixture
   density in the denominator. (Week 08 Bayes, E-step.)
4. $N_k = \sum_i \gamma_{ik}$ (soft count). $\pi_k = N_k/n$,
   $\mu_k = N_k^{-1}\sum_i \gamma_{ik} x_i$,
   $\Sigma_k = N_k^{-1}\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$.
5. Equal weights, $\Sigma_k = \varepsilon I$ for all $k$; as $\varepsilon
   \to 0$ the posterior collapses to a hard nearest-centroid assignment and
   the $\mu$ update becomes the ordinary mean. Spherical equal-weight
   clusters: k-means's geometry.
6. PCA preserves global variance directions (and, in the untruncated case,
   Euclidean geometry up to rotation). Do not read inter-cluster distances,
   blob sizes, or densities off UMAP — it preserves local neighborhoods, not
   those global quantities.
7. The inner loop's best score was used to *choose* hyperparameters, so it is
   optimistically biased (Week 09: a validation score you selected on is not
   a test score). Each outer fold is held out from that choice, so the outer
   mean estimates the tuned procedure.
8. Last week's number was a different protocol (splits, seeds, budget).
   Identical-protocol means retrain here. Choose the threshold on validation;
   report efficiency/rejection on the untouched test set (Week 10).

## New terms

- **unsupervised learning** — structure from $x$ without labels $y$.
- **centroid / WCSS (inertia)** — cluster mean; sum of squared distances to
  assigned centroids, k-means's $J$.
- **Lloyd's algorithm** — nearest-centroid assignment then mean update.
- **coordinate descent** — minimize one block of variables at a time; Lloyd
  on $(r,\mu)$ is monotone in $J$.
- **Gaussian mixture (GMM)** — $p(x)=\sum_k\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k)$.
- **mixing weight $\pi_k$** — prior probability of cluster $k$.
- **complete data / latent $z$** — observations plus unobserved cluster IDs.
- **responsibility $\gamma_{ik}$** — posterior $p(z_{ik}=1\mid x_i)$; E-step
  = Bayes (Week 08).
- **EM** — E-step then weighted MLE for $\pi,\mu,\Sigma$; never decreases
  observed-data log-likelihood.
- **UMAP** — nonlinear embedding that tries to preserve local neighborhoods,
  not global distances or densities.
- **Hillas parameters** — ellipse-summary of a Cherenkov camera image.
- **Cherenkov radiation** — light from a charged particle moving through a
  medium faster than light in that medium.
- **electromagnetic vs hadronic shower** — compact $e^\pm$/photon cascade
  from a gamma; messy pion/muon cascade from a cosmic-ray hadron.
- **nested CV** — outer loop scores the tuned procedure; inner loop chooses
  hyperparameters; outer-fold test data never enters the inner loop.
- **identical-protocol rule** — shared splits, seeds, budget, stopping;
  otherwise you compare effort, not models.

## Going deeper

- Bishop, *PRML* (in `references/`) Chapter 9 — §9.1 k-means, §9.2 the GMM
  derivation you just did. Skim §12.1 to reconnect PCA.
- UMAP documentation, "How UMAP works", and Coenen & Pearce, "Understanding
  UMAP" — what the embedding does *not* preserve. E3's caption is that list
  applied to MAGIC.
- MAGIC Gamma Telescope, UCI ML Repository — read the Hillas feature
  descriptions before ingest. (Bock et al. / CORSIKA are optional context.)
- No new capstone reading. Re-read Week 09 leakage, Week 10 §8–9, Week 11
  §5 and §7, and Capstone 1 in `03-Project-Roadmap.md` before pipeline code.
