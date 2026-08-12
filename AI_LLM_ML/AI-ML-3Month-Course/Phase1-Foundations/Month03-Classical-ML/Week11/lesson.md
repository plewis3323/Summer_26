# Week 11 — Trees & Ensembles

~10 hrs. Before starting you should be able to: frame a supervised problem with
loss/risk and validate it with CV and an untouched test set (Week 09); explain the
bias–variance decomposition (Week 09); train a classifier and compare models by AUC
(Week 10); compute entropy of a discrete distribution (Week 08); take a gradient
and run gradient descent (Weeks 05, 08).

Linear models draw one straight boundary through feature space. This week's models
draw axis-aligned boxes — and then combine hundreds of them. On **tabular data**
(rows = examples, columns = heterogeneous named features: a length in meters next
to a count next to an angle), tree ensembles have been the strongest practical
method for two decades, and they remain the model to beat before anyone reaches for
a neural network. Your Capstone 1 baseline comes from this week. The intellectual
centerpiece is the derivation of gradient boosting as gradient descent *in function
space* — the week's paper-first derivation.

## 1. Decision trees

A **decision tree** is a flowchart of yes/no questions about individual features:
"is `width` < 0.11? if yes, is `length` < 0.24? …", ending in **leaves** that give
the prediction. Internal **nodes** each test one feature against one threshold, so
every leaf corresponds to an axis-aligned rectangular region of feature space, and
the tree predicts one value per region (for classification: the majority class, or
the class fractions as probabilities).

Trees are attractive for reasons that matter on real scientific data:

- No feature scaling needed — a split at `energy < 3.2` doesn't care about units,
  and any **monotone transform** of a feature (log, square root — anything
  order-preserving) leaves the tree's behavior unchanged, since only the ordering
  of values enters.
- Mixed feature types and wildly different scales coexist peacefully.
- Small trees are readable — you can print the flowchart.

### 1.1 How a split is chosen

Growing a tree is greedy. At each node, holding the examples that reached it, try
every feature and every candidate threshold; score how much each split would
"purify" the labels; take the best; recurse on the two children. The score needs a
measure of label impurity at a node. Let $p_k$ be the fraction of the node's
examples in class $k$. Two standard impurity measures:

- **Gini impurity**: $G = 1 - \sum_k p_k^2$ — the probability that two examples
  drawn from the node (with replacement) disagree in class. Zero for a pure node;
  maximal ($1 - 1/K$) for a uniform mix.
- **Entropy** (Week 08, in bits or nats): $H = -\sum_k p_k \log p_k$ — zero for a
  pure node, maximal for a uniform mix.

A candidate split sends fractions $f_L$ and $f_R$ ($f_L + f_R = 1$) of the node's
examples left and right. Its quality is the **information gain** — impurity before
minus the weighted impurity after:

$$\Delta = I(\text{node}) - f_L\, I(\text{left}) - f_R\, I(\text{right}),$$

with $I$ either Gini or entropy (they nearly always pick the same split; Gini is
cheaper and is scikit-learn's default). "Try every threshold" is less heroic than
it sounds: for one feature, sort the node's values — only the midpoints between
consecutive distinct values can matter, so a feature with $m$ values offers at most
$m - 1$ thresholds, scannable in one sorted pass.

Worked micro-example (do the arithmetic yourself — the exercises hand you a bigger
one). Ten examples at a node, 5 signal + 5 background: Gini $= 1 - (0.5^2 + 0.5^2)
= 0.5$. A candidate split sends 4 examples left (4 signal, 0 background: Gini 0)
and 6 right (1 signal, 5 background: $1 - (1/6)^2 - (5/6)^2 = 10/36 \approx
0.278$). Gain $= 0.5 - 0.4\cdot 0 - 0.6\cdot 0.278 = 0.333$. A split producing
(3 signal, 2 background) and (2 signal, 3 background) gains only
$0.5 - 0.5\cdot 0.48 - 0.5\cdot 0.48 = 0.02$. The first split wins, matching the
eye's judgment.

For **regression trees**, replace impurity with the variance of the node's $y$
values and predict each leaf's mean — the same greedy machinery. Keep this in your
pocket: gradient boosting (§4) is built from regression trees even when the final
task is classification.

### 1.2 Why single trees overfit — and are unstable

Grow the tree until every leaf is pure and you have built a lookup table: perfect
training accuracy, terrible generalization — a maximal-variance model in Week 09's
language. The usual restraints: cap the depth (`max_depth`), require a minimum
number of examples per leaf (`min_samples_leaf`), or require a minimum gain.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)                       # same API as Week 09 — always
print(tree.score(X_val, y_val))                  # .score = accuracy for classifiers
```

The deeper disease is **instability**: because splitting is greedy, a tiny change
in the training sample can swap which split wins at the root, and the entire
subtree below it changes. Redraw the training set and the fitted function jumps
around — high variance in the precise Week 09 sense of $\mathbb{E}_D[(\hat y -
\bar h)^2]$. A high-variance, low-bias estimator is exactly the raw material that
averaging can improve. That observation is the whole of §2.

## 2. Bagging: averaging away variance

If you had $B$ *independent* training sets, you could fit $B$ trees and average
their predictions; averages of independent quantities have $1/B$ of the variance.
You have one training set. The **bootstrap** manufactures pseudo-replicas: draw $n$
examples from your $n$-example dataset *with replacement* (each replica omits ~37%
of the originals and duplicates others — $(1 - 1/n)^n \to e^{-1} \approx 0.37$).
Fit one deep (deliberately overfit, low-bias) tree per replica and average.
**Bagging** = *b*ootstrap *agg*regating.

### 2.1 What correlation does to the average — the derivation

How much does averaging actually help? Let each tree's prediction at a fixed query
point be a random variable (randomness = the bootstrap draw) with variance
$\sigma^2$, and let each *pair* of trees have correlation $\rho$ — they are not
independent, since all replicas came from the same original data. (Correlation,
from Week 08: $\rho = \mathrm{Cov}(\hat h_i, \hat h_j)/\sigma^2$, so the covariance
of any pair is $\rho\sigma^2$.) The bagged prediction is
$\bar{h} = \frac{1}{B}\sum_{i=1}^B \hat h_i$. Expand its variance using the Week 08
rule $\mathrm{Var}(\sum_i a_i) = \sum_i \mathrm{Var}(a_i) + \sum_{i \ne j}
\mathrm{Cov}(a_i, a_j)$:

$$\mathrm{Var}(\bar{h})
 = \frac{1}{B^2}\Big[\underbrace{B\,\sigma^2}_{\text{diagonal}}
 + \underbrace{B(B-1)\,\rho\sigma^2}_{\text{off-diagonal pairs}}\Big]
 = \frac{\sigma^2}{B} + \frac{B-1}{B}\,\rho\sigma^2.$$

Let $B \to \infty$:

$$\boxed{\;\mathrm{Var}(\bar h) \;\longrightarrow\; \rho\sigma^2
 \;+\; \frac{(1-\rho)\sigma^2}{B}\Big|_{\to 0}\;}$$

(the boxed form is the standard regrouping of the line above). Read it: the
$(1-\rho)\sigma^2/B$ piece dies as you add trees — averaging is free variance
reduction, and more trees never hurt (only cost compute). But the $\rho\sigma^2$
floor **does not die**. Correlated trees make correlated mistakes, and no amount of
averaging removes a shared error. Adding trees attacks $B$; to lower the floor you
must attack $\rho$.

### 2.2 Random forests: attacking ρ

A **random forest** is bagging plus one decorrelating trick: at *every node* of
every tree, only a random subset of the features (typically $\sqrt{d}$ of them for
classification) is allowed to compete for the split. Trees are thereby forced to
disagree — a dominant feature can't sit at the root of every tree — which lowers
$\rho$ at the price of slightly worse (higher-bias, higher-$\sigma^2$) individual
trees. The trade is empirically excellent, and the formula above says exactly why
it can be: the floor is $\rho\sigma^2$, so cutting $\rho$ can beat a small rise in
$\sigma^2$.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
p = forest.predict_proba(X_val)[:, 1]   # fraction of trees voting "1"
```

Forests are the great default: almost tuning-free (`n_estimators` big enough to
flatten the error curve; maybe `min_samples_leaf`), hard to badly overfit, decent
out of the box. A free bonus: each tree's ~37% left-out examples give an unbiased
validation signal without a split — the **out-of-bag (OOB) score**
(`oob_score=True`). One caveat for §Week 10 sensibilities: forest
`predict_proba` outputs are vote fractions, not calibrated probabilities —
recalibrate before treating them as beliefs.

## 3. Boosting: the other direction

Bagging builds strong (deep) learners in parallel and averages away *variance*.
**Boosting** does the mirror image: build **weak learners** — models only slightly
better than chance, typically stumpy trees of depth 1–4 — *sequentially*, each one
focused on the errors the ensemble has made so far, and *add* them up. Averaging
independent strength vs accumulating corrections: bagging attacks variance,
boosting attacks *bias*, one shallow correction at a time.

The founding algorithm (AdaBoost, 1997) reweighted misclassified examples. The
modern, general view — the one that produced XGBoost and LightGBM — is that
boosting is gradient descent in function space. That derivation is next, and it is
the week's centerpiece.

## 4. Gradient boosting as functional gradient descent

**The problem.** Find a function $F$ minimizing total loss on the training set,

$$\mathcal{L}(F) = \sum_{i=1}^{n} \ell\big(y_i,\, F(x_i)\big),$$

for any differentiable loss $\ell$ — squared error, log loss, whatever the task
needs. We will build $F$ additively:
$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu\, h_m(x)$, a sum of $M$ small trees $h_m$
scaled by a **learning rate** $\nu$.

**Step 0 — the key change of viewpoint.** The training loss $\mathcal{L}$ touches
$F$ only through the $n$ numbers $F(x_1), \dots, F(x_n)$ — the function's values
*at the training points*. Collect them into a vector
$\mathbf{F} = (F(x_1), \dots, F(x_n)) \in \mathbb{R}^n$ and the loss becomes an
ordinary function of $n$ real variables:
$\mathcal{L}(\mathbf{F}) = \sum_i \ell(y_i, \mathbf{F}_i)$. Minimizing over
"all functions" has become minimizing over a vector — Week 05/08 territory.

**Step 1 — write gradient descent in this space.** The gradient of
$\mathcal{L}$ with respect to $\mathbf{F}$ has one component per training example:

$$g_i = \frac{\partial\, \ell(y_i, F(x_i))}{\partial F(x_i)},$$

and plain gradient descent would update each predicted value by
$F(x_i) \leftarrow F(x_i) - \nu\, g_i$. Sanity-check on squared loss
$\ell = \frac{1}{2}(y_i - F(x_i))^2$: there $g_i = F(x_i) - y_i$, so
$-g_i = y_i - F(x_i)$ — **the negative gradient is the residual**, the part of $y$
still unexplained. Gradient descent literally says "nudge each prediction toward
its target." For log loss with $y_i \in \{0,1\}$ and $F$ the logit,
$-g_i = y_i - \sigma(F(x_i))$ — Week 10's residual again. The pattern is general;
$-g_i$ is called the **pseudo-residual**.

**Step 2 — the obstruction.** The update above adjusts predictions *only at the
$n$ training points*. It says nothing about $F$ anywhere else — it is a lookup
table, and lookup tables don't generalize. We need the update to be a legitimate
function of $x$, one we can evaluate on new data.

**Step 3 — project the step onto a hypothesis class.** So: take the ideal descent
direction, the vector $(-g_1, \dots, -g_n)$, and find the *function* in some
manageable class $\mathcal{H}$ (small regression trees) that best imitates it —
i.e., fit a regression tree $h_m$ to the dataset $\{(x_i,\, -g_i)\}_{i=1}^n$ by
ordinary least squares:

$$h_m = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^n \big(-g_i - h(x_i)\big)^2.$$

This is a projection in the Week 06 sense: among realizable directions, the closest
one to the true negative gradient. Because $h_m$ is a genuine function of $x$, it
carries the correction to unseen points too — the tree's split structure decides
*which regions of feature space* share each correction. That is where
generalization re-enters.

**Step 4 — step size, then repeat.** Optionally pick each leaf's output (or a
global multiplier $\gamma_m$) by a one-dimensional minimization of the true loss —
a line search; then damp with the learning rate $\nu \in (0, 1]$ (typically
0.02–0.3) and update:

$$F_m(x) = F_{m-1}(x) + \nu\, \gamma_m\, h_m(x).$$

Recompute pseudo-residuals under the new $F_m$; fit the next tree; repeat $M$
times. Initialize $F_0$ with the best constant (the mean of $y$ for squared loss;
the log-odds of the base rate for log loss).

**The algorithm, complete:**

1. $F_0 \leftarrow$ best constant prediction.
2. For $m = 1, \dots, M$:
   a. $g_i \leftarrow \partial \ell(y_i, F_{m-1}(x_i)) / \partial F_{m-1}(x_i)$
      for all $i$.
   b. Fit a small regression tree $h_m$ to targets $-g_i$.
   c. (Line search for $\gamma_m$ per leaf, if used.)
   d. $F_m \leftarrow F_{m-1} + \nu\, \gamma_m h_m$.
3. Return $F_M$.

Two knobs deserve comment. The **learning rate** $\nu$ enters exactly where the
step size entered Week 05's gradient descent — and empirically, smaller $\nu$ with
proportionally more trees $M$ generalizes better ("shrinkage"): many timid steps
follow the loss surface more faithfully than few bold ones. And unlike bagging,
**more rounds eventually overfit** — boosting keeps driving *training* loss toward
zero, memorizing noise one correction at a time. The standard guard is **early
stopping**: monitor validation loss each round and stop when it stops improving.
Boosting has no $\rho\sigma^2$-floor safety net; it needs the Week 09 discipline.

Exercise E3 has you implement this loop in ~20 lines for a 1D regression and watch
$F_m$ assemble itself round by round — do the paper derivation first, then the
code, and the code will feel inevitable.

## 5. XGBoost: second-order boosting, engineered

**XGBoost** (and its cousin **LightGBM**) is gradient boosting plus a sharper
mathematical step and a decade of engineering; it is the default serious model for
tabular data and your Capstone 1 workhorse. Install with `uv add xgboost`. The
mathematical upgrade is worth deriving — it reuses your Taylor-expansion skills
from Week 05 and previews how curvature information improves optimization.

At round $m$, instead of using only the gradient, expand each example's loss to
*second* order in the increment $h(x_i)$ around the current prediction (writing
$g_i$ for the first derivative as before and $h_i^{(2)}$ — conventionally $H_i$ —
for the second):

$$\ell\big(y_i, F_{m-1}(x_i) + h(x_i)\big) \;\approx\;
\ell\big(y_i, F_{m-1}(x_i)\big) + g_i\, h(x_i) + \tfrac{1}{2} H_i\, h(x_i)^2.$$

Add a complexity penalty on the new tree — $T$ leaves with output values
$w_1, \dots, w_T$:

$$\Omega(h) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2,$$

($\gamma$ taxes each leaf, $\lambda$ is an L2 penalty on leaf outputs — ridge's
idea (Week 09) transplanted into the tree). Since a tree assigns every example to
exactly one leaf, group the sum by leaf: with $G_j = \sum_{i \in \text{leaf } j}
g_i$ and $H_j = \sum_{i \in \text{leaf } j} H_i$, the objective (dropping
constants) is

$$\sum_{j=1}^{T}\Big[\,G_j w_j + \tfrac{1}{2}(H_j + \lambda)\, w_j^2\,\Big]
 + \gamma T$$

— a separate one-variable quadratic per leaf. Minimize each by differentiating and
setting to zero, exactly as in Week 05:

$$\boxed{\;w_j^* = -\frac{G_j}{H_j + \lambda}\;}
\qquad\text{giving objective}\qquad
-\frac{1}{2}\sum_j \frac{G_j^2}{H_j + \lambda} + \gamma T.$$

That closed form is why XGBoost is fast and principled at once: any candidate
split's value can be scored *exactly* (under the quadratic approximation) by how
much it increases $\sum_j G_j^2/(H_j + \lambda)$, minus the $\gamma$ toll for the
extra leaf — a built-in, loss-aware version of §1's information gain, with
regularization included.

Practical knobs, mapped to the theory: `max_depth` (weak-learner capacity;
3–8 typical), `learning_rate` ($\nu$), `n_estimators` ($M$; set it high and let
early stopping choose), `early_stopping_rounds` (the overfitting guard),
`subsample` / `colsample_bytree` (bagging-style randomness grafted onto boosting),
`reg_lambda` ($\lambda$). Tune depth and learning rate by CV (Week 09); everything
else starts at defaults.

```python
from xgboost import XGBClassifier

bdt = XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.05,
                    early_stopping_rounds=50, eval_metric="auc",
                    random_state=0)
bdt.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("best round:", bdt.best_iteration)
p = bdt.predict_proba(X_test)[:, 1]
```

Same fit/predict API — Week 09's one-time investment paying off across libraries.

## 6. Feature importance, and its lies

Everyone asks "which features mattered?" Tree ensembles offer answers; both common
ones can mislead, and Capstone 1's writeup should use them with stated caveats.

**Impurity importance** (`feature_importances_`): credit each feature with the
total information gain of the splits that used it, summed over the ensemble. Cheap
— it falls out of training — but twice-biased: (i) features offering many candidate
thresholds (continuous or high-cardinality ones) get more chances to look good,
even on noise; (ii) it is computed on *training* data, so it partly reports what
the model overfit to, not what generalizes.

**Permutation importance**: on *held-out* data, shuffle one feature's column
(destroying its relationship to the label while keeping its distribution), and
measure how much the model's score drops. Model-agnostic and validation-based —
strictly more honest — but with one famous failure mode both methods share:
**correlated features split the credit**. If `width` and `width_squared` are both
present, the model can lean on either; permuting one barely hurts (the other covers
for it), and impurity gain is split between them. Either way each looks
individually dispensable while the *pair* is essential. Physics features — many
derived from the same underlying measurement — are correlated by construction, so
this bites constantly. Diagnosis: check feature correlations first, permute
correlated *groups* together, or drop one of a pair and refit. Exercise E5 builds
the failure and both diagnoses.

## 7. Why particle physics loved BDTs

Some history you are inheriting, told from zero. A collider experiment's raw data
is enormous, but the working currency of an analysis is the **ntuple**: a flat
table with one row per collision event (or per reconstructed particle) and one
column per derived quantity — an energy, an angle, a track count, a shower width.
In other words: tabular data, tens of heterogeneous, physically meaningful,
mutually correlated features, often with modest usable event counts after
selection. That is precisely the regime of this week.

Around 2005–2015, "multivariate analysis" swept HEP via **TMVA** (Toolkit for
MultiVariate Analysis), a package bundled with ROOT, the field's standard analysis
framework. TMVA's flagship method was the **BDT** — boosted decision tree, HEP's
name for exactly the gradient/adaptive boosting of §4 — and for a decade "we
trained a BDT" appears in a large fraction of published analyses, including
Higgs-discovery-era results. The fit was real, not fashion: trees don't care about
units or monotone re-parameterizations (physicists' derived variables come in
arbitrary conventions), they handle correlated inputs without preprocessing, work
at ntuple-scale statistics, and boosting squeezed out accuracy that mattered when
signals were a handful of events. TMVA standardized the workflow —
train/test/evaluate with standard plots — which spread good habits, but it also hid
the machinery: a generation of analysts used the button without the derivation you
now own. What you should carry forward: on tabular physics features, a tuned BDT
remains the benchmark; when Week 16 pits a neural network against your Capstone 1
BDT, the network is not guaranteed to win — tabular data is the trees' home turf.

## 8. Worked example: one dataset, the whole zoo

Tree, forest, and BDT on the same problem, compared honestly by CV AUC — the
template Capstone 1 scales up. Runnable as shown.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Synthetic tabular stand-in: 10 informative-ish features, mild class overlap.
X, y = make_classification(n_samples=4000, n_features=10, n_informative=6,
                           n_redundant=2, flip_y=0.05, class_sep=1.0,
                           random_state=0)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # keeps class ratio per fold

models = [
    ("logistic", Pipeline([("scale", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000))])),
    ("tree d3", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ("tree deep", DecisionTreeClassifier(random_state=0)),
    ("forest", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
    ("xgboost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              random_state=0)),
]
for name, model in models:
    scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="roc_auc")
    print(f"{name:10s}  CV AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
```

Typical output pattern (your numbers will differ slightly): the *unlimited-depth*
single tree loses to the depth-3 tree — variance in action; the forest beats both
by a wide margin — §2's averaging; XGBoost edges the forest — §4's bias attack;
and logistic regression trails on this nonlinear problem but remains the honest
baseline every comparison should include. Note `StratifiedKFold`: CV folds that
preserve the class ratio (Week 10's `stratify` idea, applied per fold) — use it for
every classification CV from now on.

## Check yourself

1. Compute the Gini impurity of a node with 8 signal and 2 background examples,
   and the gain of a split producing (8 signal, 0 background) and (0, 2).
2. Why does an unlimited-depth tree have low bias and high variance? Which
   mechanism makes small data perturbations change the whole tree?
3. Derive $\mathrm{Var}(\bar h) = \rho\sigma^2 + (1-\rho)\sigma^2/B$. Which term
   does adding trees kill, and which knob of a random forest attacks the other?
4. In gradient boosting, what exactly is the "space" the gradient descent runs in,
   and why must the raw gradient step be replaced by a fitted tree?
5. Show that for squared loss the pseudo-residual is the ordinary residual, and for
   log loss it is $y - \sigma(F)$.
6. From XGBoost's per-leaf quadratic $G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2$,
   derive $w_j^* = -G_j/(H_j + \lambda)$. What role does $\lambda$ play in it?
7. More trees: harmless for bagging, dangerous for boosting. Why, in one sentence
   each — and what is the standard guard for boosting?
8. Your two most "important" features are 0.95-correlated, and permutation
   importance says each is negligible. What happened, and what would you do?

## Answers

1. $G = 1 - (0.8^2 + 0.2^2) = 0.32$. Both children are pure (Gini 0), so the gain
   is $0.32 - 0.8\cdot 0 - 0.2\cdot 0 = 0.32$.
2. Grown fully, it can carve a region per training point (memorization → near-zero
   bias, huge variance). Greedy splitting is the instability mechanism: perturb the
   data, a different split wins near the root, and everything below changes.
3. $\mathrm{Var}(\frac{1}{B}\sum \hat h_i) = \frac{1}{B^2}[B\sigma^2 +
   B(B-1)\rho\sigma^2] = \rho\sigma^2 + (1-\rho)\sigma^2/B$. Adding trees kills the
   $(1-\rho)\sigma^2/B$ term; per-node random feature subsetting
   (`max_features`) attacks $\rho$.
4. $\mathbb{R}^n$, the vector of the model's values at the $n$ training points.
   The raw step only changes predictions *at those points* — a lookup table — so
   it is projected onto the tree class by least-squares-fitting a tree to the
   pseudo-residuals, giving a function defined everywhere.
5. Squared loss $\frac{1}{2}(y - F)^2$: $\partial/\partial F = F - y$, so
   $-g = y - F$. Log loss with logit $F$: $\ell = -[y\log\sigma(F) +
   (1-y)\log(1-\sigma(F))]$, and Week 10's chain-rule cancellation gives
   $\partial\ell/\partial F = \sigma(F) - y$, so $-g = y - \sigma(F)$.
6. Differentiate: $G_j + (H_j + \lambda)w_j = 0 \Rightarrow w_j^* =
   -G_j/(H_j+\lambda)$. $\lambda$ shrinks leaf outputs toward zero (ridge on
   leaves) and keeps the step sane when a leaf's curvature $H_j$ is tiny.
7. Bagging averages same-target estimators, so extra trees only lower variance
   toward the $\rho\sigma^2$ floor; boosting keeps adding corrections that drive
   training loss down, eventually fitting noise. Guard: early stopping on a
   validation set.
8. They cover for each other — permuting one, the model leans on its twin, so
   individual importances collapse while the pair is essential. Permute the group
   together, or drop one and refit, and report the caveat.

## New terms

- **tabular data / ntuple** — rows-by-named-columns data; HEP's flat
  one-row-per-event table of derived quantities.
- **decision tree / node / leaf / stump** — the flowchart model; a test point; a
  terminal prediction region; a depth-1 tree.
- **Gini impurity / entropy impurity / information gain** — node label-mixing
  measures; impurity drop from a split.
- **monotone transform invariance** — trees see only value order, so
  order-preserving feature transforms change nothing.
- **instability** — greedy splits amplify small data changes into different trees.
- **bootstrap / bagging / out-of-bag (OOB)** — resampling with replacement;
  averaging models fit on bootstrap replicas; the ~37% left-out examples used as
  free validation.
- **random forest / `max_features`** — bagged trees + per-node random feature
  subsetting; the knob controlling that subsetting (decorrelation, attacks $\rho$).
- **weak learner / boosting** — a slightly-better-than-chance model; sequentially
  adding weak learners that fix current errors.
- **pseudo-residual** — $-\partial\ell/\partial F(x_i)$, the per-example negative
  gradient a boosting round fits.
- **functional gradient descent** — gradient descent on the vector of a model's
  training-point values, projected onto a hypothesis class each step.
- **learning rate (shrinkage) $\nu$ / line search / early stopping** — damping of
  each boosting step; per-step 1D loss minimization; halting on stalled validation
  loss.
- **XGBoost / LightGBM** — second-order, regularized gradient-boosting libraries.
- **optimal leaf weight** — $w_j^* = -G_j/(H_j + \lambda)$ from the per-leaf
  quadratic.
- **impurity vs permutation importance** — training-time gain credit vs held-out
  score drop under column shuffling; both fooled by correlated features.
- **BDT / TMVA / ROOT** — HEP's name for boosted trees; the ROOT-bundled toolkit
  that standardized them; the field's standard analysis framework.
- **StratifiedKFold** — CV splitter preserving class ratios in every fold.

## Going deeper

- StatQuest: decision trees, random forests (parts 1–2), gradient boost
  (parts 1–4), and XGBoost series — genuinely the clearest visual treatment of the
  split math and the boosting loop; watch after your paper pass, at speed.
- scikit-learn User Guide: "Decision Trees" and "Ensemble methods" — the
  bias–variance framing of bagging vs boosting; plus the permutation-importance
  page's correlated-features caveat, which §6 compressed.
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" — read §2 only: the
  regularized objective and second-order approximation you just derived, from the
  authors.
- Bishop, *PRML* (in `references/`) Chapter 14 — the committee/boosting framing,
  for a probabilistic angle on ensembles.
- TMVA Users Guide, introduction — skim to know the culture a decade of HEP
  analyses came from: what it standardized, and what it hid.
