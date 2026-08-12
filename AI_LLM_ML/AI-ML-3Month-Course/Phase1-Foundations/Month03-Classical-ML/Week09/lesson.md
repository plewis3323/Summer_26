# Week 09 — The ML Frame + Linear Regression

~10 hrs. Before starting you should be able to: write and run Python functions and
loops (Week 02); build and slice NumPy arrays and make labeled matplotlib plots
(Week 03); take a gradient of a function of several variables and run gradient
descent on it (Weeks 05, 08); multiply matrices and describe the column space of a
matrix (Week 06); derive least squares from a Gaussian likelihood (Week 08).

This week you learn what "machine learning" actually is, formally. Everything in the
rest of the course — logistic regression, boosted trees, neural networks,
transformers — sits inside the frame this week builds: a model with adjustable
parameters, a loss that scores it, and a discipline for estimating how it will do on
data it has never seen. The model this week is the simplest one there is, linear
regression, which you already met from the math side in Weeks 06–08. Now you meet it
as machine learning.

## 1. Supervised learning, stated precisely

Here is a concrete problem. A telescope records images of particle showers in the
atmosphere. Each image is summarized by 10 numbers (lengths, widths, angles of the
bright blob). For some recorded showers we know what caused them; for a new shower we
want the machine to tell us. That is **supervised learning**: learning a rule from
examples where the answer is given.

The pieces, with names you will use all year:

- An **example** (or sample, or data point) is one recorded case: one shower, one
  house sale, one collision event.
- The **features** of an example are the numbers describing it, collected into a
  vector $x \in \mathbb{R}^d$ — $d$ numbers per example. For the shower, $d = 10$.
- The **label** (or target) $y$ is the answer we want to predict. If $y$ is a real
  number (a price, an energy), the task is **regression**. If $y$ is a category
  (gamma vs hadron), the task is **classification** — that's Week 10.
- A **dataset** is $n$ example–label pairs $(x_1, y_1), \dots, (x_n, y_n)$. We stack
  the feature vectors as rows of a matrix $X$ of shape $(n, d)$, called the
  **design matrix**, and the labels into a vector $y$ of length $n$.
- A **model** (or hypothesis) is a function $h$ that maps a feature vector to a
  prediction: $\hat{y} = h(x)$. The hat on $\hat{y}$ means "predicted", as opposed
  to the true $y$.
- The **hypothesis class** $\mathcal{H}$ is the set of functions we allow ourselves
  to search over. For linear regression it is all functions of the form
  $h(x) = w^\top x + b$, one function per choice of the **weights** $w$ (a vector of
  $d$ numbers) and the **bias** $b$ (one number, the intercept — nothing to do with
  the statistical "bias" in section 5; the name collision is unfortunate and
  permanent).

**Learning** means: use the dataset to pick the member of $\mathcal{H}$ that will
predict well on *future* examples. That last clause is the entire subject. Doing
well on the examples you already have is easy — a lookup table does it perfectly.
Doing well on examples you haven't seen is called **generalization**, and every
technique in this lesson exists to measure it honestly or improve it.

## 2. Loss and risk

To pick "the best" function we need a score. A **loss function** $\ell(\hat{y}, y)$
is a number saying how bad the prediction $\hat{y}$ is when the truth is $y$ —
smaller is better, zero is perfect. For regression the default is **squared loss**:

$$\ell(\hat{y}, y) = (\hat{y} - y)^2.$$

Why squared? Week 08 gave the deep reason: if measurement noise is Gaussian,
minimizing squared loss *is* maximum likelihood. It is also differentiable
everywhere, which gradient descent needs.

Now the key distinction. Imagine (as a thought experiment) that examples are drawn
from some fixed probability distribution $P$ over pairs $(x, y)$ — nature's
distribution of showers, or house sales. The quantity we actually care about is the
**true risk** (or expected risk, or generalization error):

$$R(h) = \mathbb{E}_{(x,y)\sim P}\big[\ell(h(x), y)\big],$$

the average loss over *all possible* examples, weighted by how likely they are.
($\mathbb{E}$ is the expectation from Week 08.) We can never compute $R(h)$ — we
don't have $P$, we have $n$ samples from it. What we can compute is the
**empirical risk**, the average loss on our dataset:

$$\hat{R}(h) = \frac{1}{n}\sum_{i=1}^{n} \ell(h(x_i), y_i).$$

Training a model means minimizing $\hat{R}$ over $h \in \mathcal{H}$ — this is
called **empirical risk minimization**. The gamble is that $\hat{R}$ is close to
$R$. For a *fixed* function, it is: the law of large numbers says an average of
samples converges to the expectation. The trap is that we *chose* $h$ to make
$\hat{R}(h)$ small — we searched over many functions and kept the luckiest-looking
one — so $\hat{R}$ of the chosen model is biased low as an estimate of its $R$. The
gap $R(h) - \hat{R}(h)$ is the **generalization gap**, and when it is large we say
the model has **overfit**: it learned the noise in this particular dataset, not the
pattern. The opposite failure, a hypothesis class too rigid to capture the pattern
at all, is **underfitting**. You will produce both, on purpose, in the exercises.

## 3. Linear regression, three ways

The model: $\hat{y} = w^\top x + b$. A standard trick folds $b$ into $w$: append a
constant feature 1 to every $x$, so $x$ becomes $(x_1, \dots, x_d, 1)$ and the last
entry of $w$ plays the role of $b$. From here on assume that's done, so
$\hat{y} = w^\top x$ and, for the whole dataset at once, $\hat{y} = Xw$.

Empirical risk with squared loss (dropping the harmless $1/n$):

$$L(w) = \|Xw - y\|^2 = (Xw - y)^\top (Xw - y).$$

### 3.1 The normal equations (closed form)

Week 07's matrix calculus gives the gradient directly:
$\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$. (If that formula isn't at your
fingertips, re-derive it: expand the quadratic, differentiate term by term, or
check it numerically with finite differences as in Week 05.) At a minimum the
gradient is zero:

$$2X^\top(Xw - y) = 0 \quad\Longrightarrow\quad X^\top X\, w = X^\top y.$$

These are the **normal equations**. If $X^\top X$ is invertible (it is when $X$'s
columns are linearly independent — rank $d$, Week 06), the solution is
$w^* = (X^\top X)^{-1} X^\top y$.

The geometry, from Week 06: $Xw$ ranges over the column space of $X$ as $w$ varies.
Minimizing $\|Xw - y\|$ means finding the point in that subspace closest to $y$ —
the orthogonal projection of $y$ onto it. "Normal" means perpendicular: the residual
$y - Xw^*$ is orthogonal to every column of $X$, which is exactly what
$X^\top(y - Xw^*) = 0$ says.

One numerical warning you'll verify in the exercises: forming $X^\top X$ squares the
condition number (Week 08's villain), so in code prefer `np.linalg.lstsq(X, y)`,
which solves the same problem via a stable factorization (the SVD route from
Week 07), over explicitly inverting $X^\top X$.

### 3.2 Gradient descent

You already own this machinery. The gradient is $2X^\top(Xw - y)$; start from
$w = 0$ and iterate $w \leftarrow w - \eta \cdot 2X^\top(Xw - y)$ with step size
$\eta$. Because $L$ is a convex quadratic, GD converges to the same $w^*$ for any
$\eta$ below the stability threshold you derived in Week 08
($\eta < 2/\lambda_{\max}$, with $\lambda_{\max}$ the largest eigenvalue of
$2X^\top X$). Why bother, given the closed form? Because for the models coming after
this week there *is* no closed form — GD is the general tool, and linear regression
is where you check it against an exact answer.

### 3.3 scikit-learn

**scikit-learn** (imported as `sklearn`) is the standard Python library for
classical ML — it provides tested implementations of essentially every model in
Weeks 09–12. Install it in your project environment with `uv add scikit-learn`.

You have used libraries by calling functions (`np.mean(x)`). scikit-learn works
slightly differently: you create a **model object**, then call functions *attached
to* that object (attached functions are called **methods** — you've used them
already: `.append` on lists, `.mean()` on arrays). The object remembers what it
learned, so you don't pass the fitted parameters around by hand. Every scikit-learn
model — linear regression today, random forests in Week 11 — follows the same
three-step pattern, which is why the library is worth learning once:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(80, 1))        # 80 examples, 1 feature
y = 3.0 * X[:, 0] - 1.0 + rng.normal(0, 0.5, size=80)

model = LinearRegression()                  # 1. create the model object
model.fit(X, y)                             # 2. learn w and b from data
y_hat = model.predict(X)                    # 3. predict

print("w =", model.coef_, " b =", model.intercept_)
print("mean squared error =", np.mean((y_hat - y) ** 2))
```

Conventions to memorize: `X` is always shape `(n_examples, n_features)` — even one
feature must be a column, hence `size=(80, 1)`; `.fit(X, y)` trains in place;
`.predict(X)` returns predictions; learned parameters get a trailing underscore
(`coef_`, `intercept_`) meaning "exists only after fit". Run all three solutions —
normal equations, your GD, `LinearRegression` — on the same data in Exercise E1 and
watch them agree to numerical precision.

### 3.4 Features can be nonlinear; the model stays linear

"Linear" means linear *in $w$*, not in the raw inputs. Replace $x$ by hand-built
features and the same machinery fits curves. Polynomial features
$(1, x, x^2, \dots, x^p)$ turn linear regression into polynomial fitting, and $p$
(the **degree**) becomes a knob controlling how flexible the model is — the
**capacity** of the hypothesis class. That knob is a **hyperparameter**: a setting
you choose *before* training, as opposed to a **parameter** like $w$ that training
itself adjusts. scikit-learn builds the features for you:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)      # shape (80, 6): 1, x, x^2, ..., x^5
```

Objects like `PolynomialFeatures` that reshape data rather than predict are called
**transformers**; they use `.fit_transform` / `.transform` instead of
`.fit` / `.predict`. Note the pattern is the same: fit learns (here: nothing much),
transform applies.

## 4. Splitting data: train, validation, test

Empirical risk on the training data cannot measure generalization — the model was
chosen to make it small. The fix is blunt and non-negotiable: hold some data out.

- **Training set** — the model fits its parameters here.
- **Validation set** — you compare models and choose hyperparameters (degree,
  regularization strength) here. Because you make choices using it, its score is
  also slightly optimistic — you kept the model that got lucky on it.
- **Test set** — touched **exactly once**, at the very end, to report the final
  number. Never to make a decision. The moment a test score influences any choice,
  it has become a validation set and its number is no longer honest.

Physics runs this same discipline under the name **blind analysis**: an experiment
fixes its event-selection cuts before looking at the data region where the signal
would be, precisely so the cuts can't be (even unconsciously) tuned to manufacture
a discovery. The test set is your signal region. Freeze everything first.

```python
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
```

`random_state` seeds the shuffle (Week 04's reproducibility rule: seed everything).

### Cross-validation

With small datasets, carving out a fixed validation set wastes data and its score is
noisy — it depends on which points landed in it. **k-fold cross-validation (CV)**
reuses the data: split the training data into $k$ equal parts ("folds", commonly
$k = 5$); for each fold in turn, train on the other $k-1$ and score on the held-out
fold; average the $k$ scores. Every point gets scored exactly once by a model that
never saw it. The fold-to-fold spread also tells you how uncertain the estimate is —
always report it alongside the mean.

Be precise about what CV estimates: the average performance of *the procedure*
("fit this model class to $\sim\frac{k-1}{k}$ of the data") on unseen data — not
the performance of the one final model you then train on everything. It is the
right tool for *comparing* procedures and picking hyperparameters. The final,
report-once number still comes from the test set.

```python
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_trainval, y_trainval, cv=cv,
                         scoring="neg_mean_squared_error")
print("CV MSE:", -scores.mean(), "+/-", scores.std())
```

(scikit-learn's convention: scorers are "higher is better", so losses appear
negated — hence `neg_mean_squared_error` and the minus sign.)

### Data leakage

**Leakage** is any path by which information from outside the training fold sneaks
into training. It silently inflates validation scores, and it is the most common
way real ML projects fool their authors. The classic case: **standardizing**
features (subtracting the mean and dividing by the standard deviation — many models
want features on comparable scales) using the mean of the *whole* dataset before
splitting. The training folds then know something about the held-out points — their
contribution to that mean. The fix is to fit any preprocessing inside each training
fold only. scikit-learn's `Pipeline` chains preprocessing and model into one
estimator object so CV re-fits the whole chain per fold automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler()),
                 ("model", LinearRegression())])
scores = cross_val_score(pipe, X_trainval, y_trainval, cv=cv,
                         scoring="neg_mean_squared_error")
```

A `Pipeline` has `.fit` and `.predict` like any model — the pattern composes. Other
leakage flavors to check for in any project: duplicated rows landing on both sides
of a split; features computed *from* the label; in physics, two rows derived from
the same underlying collision event split across train and test. Exercise E4 makes
you produce leakage and measure the optimism it buys.

## 5. The bias–variance decomposition

Why do models overfit and underfit? The cleanest answer is a short derivation —
this is the week's paper-first derivation, so follow it with a pencil.

**Setup.** Fix one query point $x$. Assume nature generates labels as
$y = f(x) + \varepsilon$, where $f$ is the true (unknown) function and
$\varepsilon$ is noise with $\mathbb{E}[\varepsilon] = 0$ and
$\mathrm{Var}[\varepsilon] = \sigma^2$. Our learning procedure, given a random
training set $D$ of $n$ points, produces a model $\hat{h}_D$; its prediction at $x$
is $\hat{y} = \hat{h}_D(x)$. Note $\hat{y}$ is a *random variable*: redraw the
training set and you get a different fitted model, hence a different prediction.

We ask: on average — over both the randomness of the training set $D$ and the noise
$\varepsilon$ in the test label — how large is the squared error at $x$?

$$\mathbb{E}_{D,\varepsilon}\big[(y - \hat{y})^2\big] = \;?$$

**Two shorthands.** Let $\bar{h}(x) = \mathbb{E}_D[\hat{y}]$ — the *average*
prediction at $x$ across all possible training sets. And note $y - \hat{y} =
\underbrace{\varepsilon}_{y - f(x)} + \underbrace{(f(x) - \hat{y})}_{\text{model error}}$.

**Step 1 — split off the noise.** Square and expand:

$$\mathbb{E}\big[(y-\hat{y})^2\big]
 = \mathbb{E}[\varepsilon^2]
 + 2\,\mathbb{E}\big[\varepsilon\,(f(x)-\hat{y})\big]
 + \mathbb{E}\big[(f(x)-\hat{y})^2\big].$$

The cross term dies: the test noise $\varepsilon$ is independent of the training
set, so $\mathbb{E}[\varepsilon (f - \hat y)] = \mathbb{E}[\varepsilon]\,
\mathbb{E}[f - \hat y] = 0$ (independent factors, and $\mathbb{E}[\varepsilon]=0$;
that's the Week 08 fact that expectations of independent products factor). And
$\mathbb{E}[\varepsilon^2] = \sigma^2$. So

$$\mathbb{E}\big[(y-\hat{y})^2\big] = \sigma^2 + \mathbb{E}_D\big[(f(x)-\hat{y})^2\big].$$

**Step 2 — split the model error around the average prediction.** Insert
$\bar{h}(x)$ by adding and subtracting it:

$$f(x)-\hat{y} = \big(f(x) - \bar{h}(x)\big) + \big(\bar{h}(x) - \hat{y}\big).$$

Square and take $\mathbb{E}_D$:

$$\mathbb{E}_D\big[(f-\hat{y})^2\big]
 = \big(f - \bar{h}\big)^2
 + 2\big(f - \bar{h}\big)\,\mathbb{E}_D\big[\bar{h} - \hat{y}\big]
 + \mathbb{E}_D\big[(\bar{h} - \hat{y})^2\big].$$

The first term has no randomness left ($f$ and $\bar h$ are fixed numbers at fixed
$x$), so the expectation drops. The cross term dies again, this time by
construction: $\mathbb{E}_D[\bar{h} - \hat{y}] = \bar{h} - \mathbb{E}_D[\hat{y}] =
\bar h - \bar h = 0$ — $\bar h$ was *defined* as the mean of $\hat y$.

**Result.**

$$\boxed{\;\mathbb{E}_{D,\varepsilon}\big[(y-\hat{y})^2\big]
 = \underbrace{\big(f(x) - \bar{h}(x)\big)^2}_{\text{bias}^2}
 + \underbrace{\mathbb{E}_D\big[(\hat{y} - \bar{h}(x))^2\big]}_{\text{variance}}
 + \underbrace{\sigma^2}_{\text{irreducible noise}}\;}$$

Read each term:

- **Bias**: how far the *average* fitted model is from the truth. A degree-1
  polynomial fitting a sine curve is biased — no amount of data fixes it, because
  the hypothesis class simply cannot bend that way. High bias = underfitting.
- **Variance**: how much the fitted model jumps around when you redraw the training
  set. A degree-15 polynomial through 20 noisy points swings wildly between draws.
  High variance = overfitting.
- **Irreducible noise** $\sigma^2$: the label noise itself. No model beats it; if
  your test error approaches $\sigma^2$, stop tuning.

Be explicit about what each expectation is over — this is the part people fumble.
The bias and variance terms are averages over *training sets* $D$ at a fixed query
$x$; the $\sigma^2$ came from averaging over *test-label noise*. Total expected
test error also averages over query points $x$, which just averages this identity
pointwise.

The practical picture: as capacity (polynomial degree) increases, bias² falls and
variance rises; their sum is U-shaped, and the best generalization sits at the
bottom of the U. Exercise E2 has you *measure* all three terms by simulation —
possible only in simulation, where you know $f$ and can redraw $D$ hundreds of
times — and watch the decomposition hold to within noise.

## 6. Ridge regression: buying less variance with a little bias

If high variance is the disease, one cure is to restrain $w$. **Ridge regression**
adds a penalty on the size of the weights to the loss:

$$L_\lambda(w) = \|Xw - y\|^2 + \lambda \|w\|^2,$$

with **regularization strength** $\lambda \ge 0$ — a hyperparameter. Any added term
that discourages complexity is called **regularization**. Setting the gradient to
zero exactly as in §3.1 (the penalty contributes $2\lambda w$):

$$X^\top X w + \lambda w = X^\top y
\quad\Longrightarrow\quad
w^*_\lambda = (X^\top X + \lambda I)^{-1} X^\top y,$$

where $I$ is the identity matrix. Two limits sanity-check it: $\lambda \to 0$
recovers ordinary least squares; $\lambda \to \infty$ crushes $w \to 0$. In
between, coefficients **shrink** smoothly — you'll plot that path in Exercise E5
and pick $\lambda$ by cross-validation, which is exactly what CV is for.

You have seen this formula before wearing different clothes: in Week 08 you showed
MAP estimation with a Gaussian prior on $w$ gives L2-regularized least squares.
Ridge *is* that MAP solution; $\lambda$ encodes how strongly the prior pulls toward
zero. Bonus: $X^\top X + \lambda I$ is always invertible for $\lambda > 0$ (it adds
$\lambda$ to every eigenvalue), so ridge also rescues the ill-conditioned cases
that break the plain normal equations.

Practical rule: standardize features before ridge — the penalty treats all
coefficients alike, which is only fair if the features share a scale. Inside a
`Pipeline`, of course.

## 7. Worked example: degree selection done honestly

Everything above, end to end, on data where we know the truth. Runnable as shown.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold

rng = np.random.default_rng(42)

# Nature: f(x) = sin(2x), noise sigma = 0.3
n = 120
X = rng.uniform(-2, 2, size=(n, 1))
y = np.sin(2 * X[:, 0]) + rng.normal(0, 0.3, size=n)

# Freeze the test set FIRST. It will be touched exactly once, at the end.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Compare polynomial degrees by 5-fold CV on the train+val data only.
cv = KFold(n_splits=5, shuffle=True, random_state=0)
degrees = [1, 2, 3, 5, 7, 9, 12, 15]
cv_mse = []
for d in degrees:
    pipe = Pipeline([("poly", PolynomialFeatures(degree=d)),
                     ("scale", StandardScaler()),
                     ("model", LinearRegression())])
    scores = cross_val_score(pipe, X_trainval, y_trainval, cv=cv,
                             scoring="neg_mean_squared_error")
    cv_mse.append(-scores.mean())
    print(f"degree {d:2d}: CV MSE = {-scores.mean():.4f} +/- {scores.std():.4f}")

best_degree = degrees[int(np.argmin(cv_mse))]
print("chosen degree:", best_degree)

# Refit the chosen pipeline on ALL train+val data, then the single test evaluation.
final = Pipeline([("poly", PolynomialFeatures(degree=best_degree)),
                  ("scale", StandardScaler()),
                  ("model", LinearRegression())])
final.fit(X_trainval, y_trainval)
test_mse = np.mean((final.predict(X_test) - y_test) ** 2)
print(f"test MSE (reported once): {test_mse:.4f}   noise floor sigma^2 = {0.3**2:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(degrees, cv_mse, "o-", label="CV MSE")
ax.axhline(0.3 ** 2, linestyle="--", label="noise floor $\\sigma^2$")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("mean squared error")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
plt.show()
```

Things to notice when you run it: the CV curve is U-shaped (bias falls, then
variance takes over); the chosen degree is small (the truth is a sine — degree 5-ish
captures it on this range); the test MSE lands near the CV estimate and near the
noise floor $\sigma^2 = 0.09$. If your test score ever comes out *far* better than
CV predicted, suspect leakage, not genius.

## Check yourself

1. In one sentence each: what is empirical risk, what is true risk, and why is the
   empirical risk of a *trained* model a biased estimate of its true risk?
2. Derive the normal equations from $\nabla_w \|Xw - y\|^2 = 0$. What property of
   $X$ makes the solution unique?
3. In the bias–variance decomposition, which expectation is the variance term taken
   over — test noise, training sets, or query points?
4. A degree-1 fit to quadratic data has high ______; a degree-15 fit to 20 points
   has high ______. Which one does collecting more data help?
5. You standardize your features using the full dataset's mean, then run 5-fold CV.
   What goes wrong, and what is the one-line fix in scikit-learn?
6. What does 5-fold CV estimate, and why is it still not a substitute for the final
   test-set evaluation?
7. Write the ridge solution and give both limits, $\lambda \to 0$ and
   $\lambda \to \infty$. Which Week 08 idea is ridge the twin of?
8. In the scikit-learn API, what do `.fit`, `.predict`, and a trailing underscore
   (as in `coef_`) each signify?

## Answers

1. Empirical risk: average loss on your $n$ samples. True risk: expected loss over
   nature's distribution $P$. The trained model was *selected* to minimize the
   empirical risk, so its training score is optimistic — you kept the function that
   looked luckiest on this sample.
2. $\nabla_w = 2X^\top(Xw - y) = 0 \Rightarrow X^\top X w = X^\top y$. Unique when
   $X^\top X$ is invertible, i.e. the columns of $X$ are linearly independent
   (rank $d$).
3. Over training sets $D$, at a fixed query point $x$:
   $\mathbb{E}_D[(\hat{y} - \bar{h}(x))^2]$.
4. High bias; high variance. More data mainly cures variance — bias is a property
   of the hypothesis class and stays.
5. The held-out fold's statistics leaked into training via the shared mean, so CV
   scores are optimistic. Fix: put `StandardScaler` and the model in a `Pipeline`
   and cross-validate the pipeline, so scaling is re-fit inside each training fold.
6. The average unseen-data performance of the *fitting procedure* on datasets of
   size $\sim\frac{k-1}{k}n$. You then used those scores to make choices (degree,
   $\lambda$), so the winning score is again slightly optimistic — the untouched
   test set settles it.
7. $w^*_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$; $\lambda \to 0$ gives
   ordinary least squares, $\lambda \to \infty$ gives $w \to 0$. It is MAP
   estimation with a Gaussian prior on $w$ (Week 08).
8. `.fit(X, y)` learns parameters from data and stores them on the object;
   `.predict(X)` applies the learned model; a trailing underscore marks an
   attribute that exists only after fitting (a learned quantity).

## New terms

- **supervised learning** — learning a prediction rule from labeled examples.
- **feature / label / example** — input numbers $x$ / target $y$ / one $(x, y)$ pair.
- **regression / classification** — predicting a number / a category.
- **design matrix** — the $(n, d)$ matrix $X$ with one example per row.
- **model (hypothesis)** — a function from features to prediction.
- **hypothesis class** — the set of functions the learner searches.
- **weights / bias (intercept)** — the parameters $w$, $b$ of a linear model.
- **loss function** — per-example badness score $\ell(\hat{y}, y)$.
- **empirical risk / true risk** — average loss on the sample / expectation under $P$.
- **generalization (gap)** — performance on unseen data (and its shortfall vs training).
- **overfitting / underfitting** — fitting noise / failing to fit signal.
- **normal equations** — $X^\top X w = X^\top y$, the least-squares optimum.
- **hyperparameter** — a setting chosen before training (degree, $\lambda$, $k$).
- **capacity** — how flexible a hypothesis class is.
- **train / validation / test split** — fit / choose / report-once partitions.
- **k-fold cross-validation** — rotate a held-out fold; average the $k$ scores.
- **data leakage** — outside-the-fold information reaching training.
- **standardization** — rescaling a feature to mean 0, standard deviation 1.
- **Pipeline** — a chained preprocessing+model object that CV re-fits per fold.
- **bias / variance / irreducible noise** — the three terms of expected squared error.
- **regularization / ridge / shrinkage** — penalizing weight size;
  $\lambda\|w\|^2$; the resulting pull of coefficients toward zero.
- **estimator / transformer / method** — scikit-learn's fit-able objects; the
  data-reshaping kind; a function attached to an object.
- **blind analysis** — physics practice of freezing selections before looking at
  the signal region; the test set is the same idea.

## Going deeper

- VanderPlas, *Python Data Science Handbook* (free online), Chapter 5 — the
  hyperparameters/model-validation and linear-regression sections: a second pass on
  the scikit-learn API with more worked figures.
- Bishop, *PRML* (in `references/`) §1.1 and §3.1–3.2 — the polynomial-fitting
  example is this whole lesson in miniature; §3.2 gives the textbook bias–variance
  treatment.
- scikit-learn User Guide, "Cross-validation" — especially the pitfalls section on
  leakage; short and worth reading verbatim.
- StatQuest, "Machine Learning Fundamentals: Bias and Variance" — a 7-minute
  intuition pass if the decomposition feels abstract.
