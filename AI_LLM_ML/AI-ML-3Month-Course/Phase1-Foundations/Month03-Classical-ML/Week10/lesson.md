# Week 10 — Classification

~10 hrs. Before starting you should be able to: state the supervised-learning frame
and use scikit-learn's fit/predict pattern with a proper train/val/test split
(Week 09); write a likelihood, take its negative log, and connect cross-entropy to
KL divergence (Week 08); run gradient descent, including your own Adam (Weeks 05,
08); apply the chain rule to composed functions (Week 05).

Week 09 predicted numbers. This week predicts categories: is this shower a gamma ray
or a hadron, is this email spam, is this collision event worth keeping? The plan:
derive the standard classifier (logistic regression) from a likelihood, get its
gradient by hand — a derivation Phase 2's backprop will reuse verbatim — then spend
the second half on the part practitioners actually get wrong: turning scores into
*decisions*, measuring classifiers honestly, and surviving class imbalance.

## 1. The classification problem

Setup as in Week 09 — examples $x \in \mathbb{R}^d$, dataset of $n$ pairs — except
the label is now a category. For **binary classification** we code the two classes
as $y \in \{0, 1\}$: 1 for the class we care about finding (the **positive class**,
"signal"), 0 for the other (**negative class**, "background"). With more than two
classes ($y \in \{1, \dots, K\}$) it's **multiclass classification** — §5.

First instinct: code the classes as numbers and run linear regression. It fails
informatively. Squared loss punishes the model for predicting "2.3" on a confident
positive — being *very right* costs loss — and predictions aren't interpretable:
what would $\hat{y} = 1.4$ mean? What we actually want from a classifier is a
**probability**: $p(y = 1 \mid x)$, the model's belief that this example is
positive given its features (the conditional probability from Week 08). Probability
outputs are exactly what decision-making downstream needs — §6 onward runs on them.

## 2. From a line to a probability: the sigmoid

Keep the linear machine $z = w^\top x + b$ — it's too useful to abandon — but its
output ranges over all of $\mathbb{R}$, and probabilities live in $(0, 1)$. We need
a squashing function. The standard choice is the **sigmoid** (or logistic function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}.$$

Check its behavior: $\sigma(0) = 1/2$; as $z \to +\infty$, $\sigma \to 1$; as
$z \to -\infty$, $\sigma \to 0$; and it is smooth and monotonic. The model is then

$$p := p(y = 1 \mid x) = \sigma(w^\top x + b),$$

and this model is **logistic regression** — a classifier, despite the historical
name. The number $z$ before squashing is called the **logit** or score.

Why this particular squash? Invert it. Solving $p = \sigma(z)$ for $z$ gives

$$z = \log\frac{p}{1 - p},$$

the **log-odds**: the log of (probability for) / (probability against). So logistic
regression is precisely the statement "the log-odds is linear in the features."
Each weight $w_j$ has a clean meaning: one unit of feature $j$ adds $w_j$ to the
log-odds. Evidence adds; probabilities multiply — the sigmoid is the bridge.

One derivative, worth doing once on paper because everything below uses it. Write
$\sigma = (1 + e^{-z})^{-1}$ and apply the chain rule (Week 05):

$$\sigma'(z) = -(1 + e^{-z})^{-2}\cdot(-e^{-z})
 = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
 = \sigma(z)\,\big(1 - \sigma(z)\big),$$

using $\frac{e^{-z}}{1+e^{-z}} = \frac{(1+e^{-z}) - 1}{1+e^{-z}} = 1 - \sigma(z)$.
So $\sigma' = \sigma(1-\sigma)$: the slope is largest at $p = 1/2$ (maximally
undecided) and vanishes as the model becomes certain either way.

## 3. The loss: negative log-likelihood

What loss should we minimize? Week 08's recipe: write the likelihood of the data
under the model, take $-\log$. The label is a coin flip with example-dependent
probability — a **Bernoulli** random variable: $y = 1$ with probability $p$,
$y = 0$ with probability $1 - p$. A compact way to write both cases at once:

$$P(y \mid x; w) = p^{\,y}\,(1-p)^{1-y},$$

(check it: $y=1$ gives $p$, $y=0$ gives $1-p$). Assuming independent examples, the
likelihood of the whole dataset is the product over examples, and the negative
log-likelihood — our loss — is the sum:

$$L(w) = -\sum_{i=1}^{n}\Big[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\Big],
\qquad p_i = \sigma(w^\top x_i + b).$$

This is the **log loss**, also called **binary cross-entropy** — and that name is
literal: for each example it is the cross-entropy (Week 08) between the true label
distribution (all mass on $y_i$) and the predicted distribution $(p_i, 1-p_i)$.
Minimizing it minimizes the KL divergence from truth to model, since they differ by
an entropy term that doesn't depend on $w$. Read its shape: if $y_i = 1$ the loss
is $-\log p_i$ — gentle when $p_i$ is near 1, unbounded as $p_i \to 0$. Confident
wrong answers are punished without limit; that is what makes the outputs honest
probabilities rather than mere scores.

Unlike least squares, setting $\nabla L = 0$ has no closed-form solution — the
sigmoid's nonlinearity sees to that. But $L(w)$ is convex (bowl-shaped, no spurious
valleys — the Week 08 property), so gradient descent finds the global optimum. We
need the gradient.

## 4. The gradient, derived by hand

This is the week's flagship derivation. Do it on paper alongside the text; Phase 2
will ask you for it again from a cold start.

Work with one example (drop the subscript $i$; the full gradient is the sum over
examples). Define the chain we're differentiating through:

$$z = w^\top x + b, \qquad p = \sigma(z), \qquad
\ell = -\big[y\log p + (1-y)\log(1-p)\big].$$

We want $\partial \ell / \partial w$, via the chain rule:
$\frac{\partial \ell}{\partial w} =
 \frac{d\ell}{dp}\cdot\frac{dp}{dz}\cdot\frac{\partial z}{\partial w}$.

**Link 1: $d\ell/dp$.** Differentiate the loss with respect to $p$
($\frac{d}{dp}\log p = 1/p$ from Week 05):

$$\frac{d\ell}{dp} = -\frac{y}{p} + \frac{1-y}{1-p}
 = \frac{-y(1-p) + (1-y)p}{p(1-p)}
 = \frac{p - y}{p(1-p)}.$$

**Link 2: $dp/dz$.** Already done: $dp/dz = \sigma'(z) = p(1-p)$.

**Link 3: $\partial z/\partial w$.** $z = w^\top x + b$ is linear in $w$, so
$\partial z/\partial w = x$ (and $\partial z/\partial b = 1$).

**Multiply.** The $p(1-p)$ factors cancel — this is the punchline:

$$\frac{\partial \ell}{\partial w}
 = \frac{p - y}{p(1-p)}\cdot p(1-p)\cdot x
 = (p - y)\,x,
\qquad
\frac{\partial \ell}{\partial b} = p - y.$$

Sum over examples and stack into matrix form (rows of $X$ are the $x_i^\top$):

$$\boxed{\;\nabla_w L = X^\top\big(\sigma(Xw) - y\big)\;}$$

where $\sigma$ applies elementwise. Pause on how clean this is. It is *identical in
form* to linear regression's gradient $X^\top(Xw - y)$ — features times
**residual**, prediction minus truth — even though the loss and model are
different. The cancellation is not luck: the log-likelihood was built from the same
exponential-family structure as the sigmoid, so the loss's curvature and the
sigmoid's saturation exactly offset. One practical payoff: even when the sigmoid
saturates ($\sigma' \approx 0$, which would strangle a squared-loss gradient), the
log-loss gradient $(p - y)x$ stays healthy whenever the prediction is wrong.

Training is now Week 08 material: run GD or Adam on $\nabla_w L$. In practice add
L2 regularization $\lambda\|w\|^2$ exactly as in ridge (scikit-learn's
`LogisticRegression` regularizes by default — its knob is `C` $= 1/\lambda$, so
*larger* `C` means *less* regularization; a classic gotcha). Exercise E1 has you
train with your own Adam and match sklearn's coefficients.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 400
X = np.vstack([rng.normal(-1.0, 1.0, size=(n // 2, 2)),
               rng.normal(+1.0, 1.0, size=(n // 2, 2))])
y = np.repeat([0, 1], n // 2)

model = LogisticRegression(C=1e6)     # near-unregularized, to match a scratch fit
model.fit(X, y)
print("w =", model.coef_, " b =", model.intercept_)
print("first 3 probabilities:", model.predict_proba(X[:3])[:, 1])
print("first 3 hard labels:  ", model.predict(X[:3]))
```

New API surface: classifiers have `.predict_proba(X)`, returning one column per
class (column 1 is $p(y=1\mid x)$), alongside `.predict(X)`, which returns hard 0/1
labels by thresholding at 0.5. Keep them straight: probabilities are the model's
real output; hard labels are a decision someone made — §6.

## 5. More than two classes: softmax

For $K$ classes, give each class its own score: $z_k = w_k^\top x + b_k$, stacked
into a vector $z \in \mathbb{R}^K$. To convert $K$ scores into $K$ probabilities
that are positive and sum to 1, exponentiate and normalize — the **softmax**:

$$p_k = \mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

Sanity checks: with $K = 2$, softmax reduces to the sigmoid applied to $z_1 - z_0$
(two-class softmax is logistic regression); adding a constant to every $z_k$
changes nothing (only score *differences* matter).

The loss generalizes too. Write the label **one-hot**: $y$ is a length-$K$ vector
with 1 in the true class's slot, 0 elsewhere. Cross-entropy loss:
$\ell = -\sum_k y_k \log p_k = -\log p_{\text{true class}}$.

Now the gradient with respect to the scores $z$ — derived once, used for years.
Take $\partial \ell/\partial z_j$ and split $\ell = -\sum_k y_k z_k +
\log\sum_j e^{z_j}$ (expand $\log p_k$ and use $\sum_k y_k = 1$ to pull the
log-sum term out). Differentiating each piece:

$$\frac{\partial}{\partial z_j}\Big(-\sum_k y_k z_k\Big) = -y_j,
\qquad
\frac{\partial}{\partial z_j}\log\sum_i e^{z_i}
 = \frac{e^{z_j}}{\sum_i e^{z_i}} = p_j.$$

So

$$\boxed{\;\frac{\partial \ell}{\partial z} = p - y\;}$$

— prediction minus one-hot truth, the same residual form again. **Flag this page
in your notes.** When Phase 2 derives backpropagation for a neural network ending
in softmax + cross-entropy, this identity is the first step of the backward pass;
you will be glad it is already burned in. (To finish the parameter gradient here:
chain through $z_k = w_k^\top x + b_k$ to get
$\partial\ell/\partial w_k = (p_k - y_k)\,x$.)

## 6. From probabilities to decisions: thresholds

A probability is not yet an action. To act you pick a **decision threshold** $t$
and call the example positive when $p \ge t$. Nothing says $t$ must be 0.5 —
that default is only right when both error types cost the same, and they rarely do.

Every choice of $t$ produces a **confusion matrix** — the 2×2 count of what
happened, whose four cells you should be able to recite:

|  | predicted 1 | predicted 0 |
|---|---|---|
| **actual 1** | true positive (TP) | false negative (FN) |
| **actual 0** | false positive (FP) | true negative (TN) |

From these, the derived rates (each answers a different question):

- **Accuracy** $= \frac{TP + TN}{n}$ — fraction of all calls that were right.
- **Recall**, also **true positive rate (TPR)**, also (in physics) **efficiency**
  $= \frac{TP}{TP + FN}$ — of the real positives, how many did we catch?
- **False positive rate (FPR)** $= \frac{FP}{FP + TN}$ — of the real negatives,
  how many did we wrongly flag?
- **Precision**, also (in physics) **purity** $= \frac{TP}{TP + FP}$ — of the ones
  we flagged, how many are real?

Raising $t$ makes the classifier pickier: FP falls (precision rises), but TP falls
too (recall drops). Lowering $t$ does the reverse. This tradeoff is not a defect to
be fixed but the entire content of the decision; the metrics exist to expose it.

### Why accuracy lies under imbalance

**Class imbalance** means one class vastly outnumbers the other — the normal
situation in physics, fraud, and medicine. Two lines of algebra kill accuracy as a
metric. Let the positive class be a fraction $\pi$ of the data, and consider the
lazy classifier "always say 0." Its accuracy is $1 - \pi$. At $\pi = 0.01$
(a 99:1 problem) that is **99% accurate** while catching *zero* positives. Any
reported accuracy must beat this floor to mean anything, and near the floor the
metric can't distinguish a useful model from the lazy one. Under imbalance, report
recall, precision, and the curves below — never bare accuracy.

## 7. ROC and PR curves: all thresholds at once

Rather than defend one threshold, sweep $t$ from 1 down to 0 and trace what
happens. Two standard pictures:

The **ROC curve** ("receiver operating characteristic", a radar-era name that
stuck) plots TPR (y) against FPR (x), one point per threshold. It starts at
$(0,0)$ ($t=1$: call nothing positive), ends at $(1,1)$ ($t=0$: call everything
positive). A perfect classifier hugs the top-left corner; a coin-flip classifier
traces the diagonal $\mathrm{TPR} = \mathrm{FPR}$. The **AUC** (area under the
curve) compresses it to one number in $[0.5, 1]$, and has a beautiful
interpretation worth memorizing: *AUC is the probability that a randomly chosen
positive example receives a higher score than a randomly chosen negative one.* It
measures ranking quality, independent of any threshold and — because TPR and FPR
are each normalized within their own class — independent of class balance.

The **PR curve** plots precision (y) against recall (x). Because precision has FP
in the denominator against TP, it is acutely sensitive to how many negatives there
are. That makes PR the more honest picture under heavy imbalance: an ROC curve can
look superb at $\pi = 0.001$ while the PR curve reveals that at your operating
recall, 95% of your flagged events are junk. Rule of thumb: balanced or mildly
imbalanced → ROC; rare positives where precision is what you'll live with → PR
(and check both; they're cheap).

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

p_val = model.predict_proba(X_val)[:, 1]        # scores on validation data
fpr, tpr, thresholds = roc_curve(y_val, p_val)
prec, rec, thr_pr = precision_recall_curve(y_val, p_val)
print("AUC =", roc_auc_score(y_val, p_val))
```

In Exercise E3 you build both curves from scratch with a threshold loop and match
sklearn's — after that, `roc_curve` holds no mysteries.

## 8. The physics of thresholds: triggers

Here is where a particle physicist meets this material daily. Some background,
from zero. A collider like the LHC or RHIC smashes particles together millions to
billions of times per second; each crossing that produces recorded detector
signals is an **event**. Fully reading out and storing one event costs from
hundreds of kilobytes to megabytes — at tens of MHz of collisions, no storage
system on Earth keeps up. So every experiment runs a **trigger**: a fast, automated
yes/no decision, made in microseconds by dedicated hardware and software, about
whether each event is written to disk or discarded *forever*. The trigger is a
classifier with a threshold, run at wire speed, whose false negatives are
permanently unrecoverable.

Trigger designers speak exactly the language of §6, with their own dialect:
**efficiency** (what fraction of the interesting physics events pass the cut) is
recall; **purity** (what fraction of the kept events are interesting) is
precision; the **background rejection** (what fraction of junk is discarded) is
$1 - \mathrm{FPR}$. "Tightening the cut" is raising the threshold.

How do they pick the working point? Not by symmetric accuracy — signal events may
be one in $10^6$. Two standard recipes, both in Exercise E4:

- **Fixed efficiency.** "Keep 90% of signal; make rejection as high as possible."
  Choose the threshold on validation data where TPR crosses 0.90, then report the
  FPR you bought. This is the mode when missing signal is the cardinal sin.
- **Maximize significance.** When the goal is to *discover* a signal over a
  background, the figure of merit is roughly $S/\sqrt{S + B}$, where $S$ and $B$
  are the expected counts of signal and background events passing the cut. The
  intuition: the background that passes fluctuates statistically by about
  $\sqrt{S+B}$ (Poisson counting, Week 08), so this ratio counts "how many standard
  deviations of fluctuation" your signal excess is. Sweep the threshold, compute
  $S/\sqrt{S+B}$ at each, take the max. Note it deliberately trades some efficiency
  away when the background is nasty.

Discipline point (Week 09's rule wearing a hard hat): choose the threshold on
*validation* data, then measure efficiency and purity on data the choice never saw.
A threshold is a hyperparameter.

## 9. Calibration: do the probabilities mean anything?

A model can rank perfectly (AUC $\to$ 1) while its probabilities are nonsense.
**Calibration** asks: among all events where the model said "70%", do about 70%
actually turn out positive? If yes, the model is calibrated, and its outputs can be
plugged into expected-count and significance formulas as real probabilities. If it
says 99% and is right only 80% of the time, it is **overconfident**.

The diagnostic is the **reliability diagram**: bin validation predictions by
predicted probability (e.g. 10 bins), and for each bin plot the *observed* positive
fraction against the *mean predicted* probability. A calibrated model lies on the
diagonal; an overconfident model traces an S below-then-above it. The
**Brier score**, $\frac{1}{n}\sum_i (p_i - y_i)^2$ — mean squared error between
predicted probability and the 0/1 outcome — gives a single number that rewards both
discrimination and calibration (lower is better).

Well-regularized logistic regression is usually decently calibrated (it was trained
on a likelihood, after all). Models optimized for ranking or accuracy — boosted
trees (Week 11), SVMs, many neural nets — often are not. The repair, at the level
you need now: fit a small *second* model that maps the classifier's scores to
corrected probabilities, using held-out data. **Platt scaling** fits a logistic
function to the scores (two parameters, robust on small data); **isotonic
regression** fits a flexible monotonic step function (needs more data, fixes
weirder shapes). scikit-learn wraps both:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)  # Platt
calibrated.fit(X_trainval, y_trainval)
frac_pos, mean_pred = calibration_curve(y_val, p_val, n_bins=10)  # reliability pts
```

Calibration is a Capstone 1 gate requirement, and for good physics reason: a
threshold placed to achieve "90% efficiency" using *miscalibrated* probabilities
delivers some other efficiency entirely.

## 10. Living with imbalance

Beyond choosing better metrics, three levers when positives are rare, in the order
to try them:

1. **Move the threshold.** Often sufficient. Train as usual, then place $t$ by the
   §8 recipes instead of defaulting to 0.5.
2. **Weight the classes.** Multiply each example's loss by a class weight, making
   one positive "worth" many negatives to the optimizer:
   `LogisticRegression(class_weight="balanced")` sets weights inversely
   proportional to class frequency. Equivalent to changing the costs in the loss.
3. **Resample.** Oversample positives or undersample negatives in the *training
   folds only*. Resampling before splitting duplicates events across the split —
   textbook leakage (Week 09).

And one honest caveat: weighting and resampling deliberately distort the training
distribution, so the resulting probabilities are miscalibrated by construction —
recalibrate (§9) on unweighted validation data before trusting them.

## 11. Worked example: an imbalanced trigger, end to end

A 1:50 signal/background problem, from raw arrays to a chosen working point.
Runnable as shown.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

rng = np.random.default_rng(7)
n_bkg, n_sig = 20000, 400                      # 1:50 imbalance
X_bkg = rng.normal(0.0, 1.0, size=(n_bkg, 2))
X_sig = rng.normal(1.8, 1.0, size=(n_sig, 2))  # signal shifted in both features
X = np.vstack([X_bkg, X_sig])
y = np.concatenate([np.zeros(n_bkg), np.ones(n_sig)])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.3,
                                            random_state=0, stratify=y_tr)

model = LogisticRegression(class_weight="balanced")
model.fit(X_tr, y_tr)
p_val = model.predict_proba(X_val)[:, 1]

lazy_accuracy = 1 - y_val.mean()
print(f"lazy 'always background' accuracy: {lazy_accuracy:.4f}")
print(f"AUC on validation: {roc_auc_score(y_val, p_val):.4f}")

# Working point 1: 90% signal efficiency, chosen on validation data.
fpr, tpr, thresholds = roc_curve(y_val, p_val)
i = int(np.argmax(tpr >= 0.90))                 # first threshold reaching 90% TPR
t_eff90 = thresholds[i]
print(f"t for 90% efficiency: {t_eff90:.3f}  (buys FPR = {fpr[i]:.4f})")

# Working point 2: maximize S/sqrt(S+B) over thresholds, on validation data.
sig_like = []
for t in thresholds:
    passed = p_val >= t
    S = np.sum(passed & (y_val == 1))
    B = np.sum(passed & (y_val == 0))
    if S + B > 0:
        sig_like.append(S / np.sqrt(S + B))
    else:
        sig_like.append(0.0)
t_maxsig = thresholds[int(np.argmax(sig_like))]
print(f"t for max S/sqrt(S+B): {t_maxsig:.3f}")

# Evaluate BOTH frozen thresholds once, on the untouched test set.
p_te = model.predict_proba(X_te)[:, 1]
for name, t in [("eff90", t_eff90), ("maxsig", t_maxsig)]:
    cm = confusion_matrix(y_te, p_te >= t)
    tn, fp, fn, tp = cm.ravel()
    print(f"{name}: efficiency={tp/(tp+fn):.3f}  purity={tp/(tp+fp):.3f}  "
          f"rejection={tn/(tn+fp):.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_val, p_val):.3f}")
ax.plot([0, 1], [0, 1], "--", label="coin flip")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate (efficiency)")
ax.legend()
fig.tight_layout()
plt.show()
```

Things to notice: the lazy accuracy is ~0.98 — beating it is table stakes, not
success; `stratify=y` keeps the class ratio identical across splits (with 400
positives, an unlucky split could starve validation); the two working points land
at different thresholds because they answer different questions; and both were
*chosen* on validation, *reported* on test. That two-step is the week's habit.

## Check yourself

1. Show that solving $p = \sigma(z)$ for $z$ gives the log-odds. What does a weight
   $w_j = 0.7$ mean in those terms?
2. Derive $\sigma' = \sigma(1 - \sigma)$ from the definition.
3. Walk the three chain-rule links from the Bernoulli NLL to
   $\nabla_w L = X^\top(\sigma(Xw) - y)$. Which two factors cancel, and what is the
   practical benefit of that cancellation?
4. For softmax + cross-entropy, what is $\partial\ell/\partial z$, and why does the
   course keep flagging it?
5. A dataset is 99% negative. What accuracy does "always predict negative" get, and
   which two metrics would expose it as useless?
6. Define AUC in one sentence via the two-random-examples interpretation. Why is it
   threshold-independent?
7. Your trigger needs 90% signal efficiency. On which data split do you place the
   threshold, and on which do you report the resulting purity?
8. A model has AUC 0.98 but a badly bowed reliability curve. What is wrong, what is
   NOT wrong, and which two repairs does §9 offer?

## Answers

1. $p = \frac{1}{1+e^{-z}} \Rightarrow e^{-z} = \frac{1-p}{p} \Rightarrow
   z = \log\frac{p}{1-p}$. A unit increase in feature $j$ adds 0.7 to the log-odds,
   i.e. multiplies the odds by $e^{0.7} \approx 2$.
2. $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot
   \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$.
3. $d\ell/dp = \frac{p-y}{p(1-p)}$; $dp/dz = p(1-p)$; $\partial z/\partial w = x$;
   the $p(1-p)$ factors cancel, leaving $(p-y)x$. Benefit: no vanishing gradient
   when the sigmoid saturates on a *wrong* confident prediction — the error signal
   stays proportional to $p - y$.
4. $\partial\ell/\partial z = p - y$ (prediction minus one-hot truth). It is the
   first step of the backward pass through any softmax + cross-entropy head —
   Phase 2's backprop derivation starts exactly there.
5. 99% accuracy. Recall (0 — it catches nothing) and precision (undefined/0 — it
   flags nothing); either exposes it, as does the PR curve collapsing.
6. AUC is the probability that a random positive example scores higher than a
   random negative one — a statement about the *ranking* of scores, which never
   mentions a threshold.
7. Place it on validation data; report efficiency/purity on the untouched test set.
8. Its probabilities are miscalibrated (the stated 90% isn't an observed 90%); its
   *ranking* is fine — AUC is unaffected by any monotone rescaling of scores.
   Repairs: Platt scaling (logistic map, small data) or isotonic regression
   (monotone step function, more data), both via `CalibratedClassifierCV`.

## New terms

- **binary / multiclass classification** — predicting one of 2 / of $K$ categories.
- **positive & negative class (signal & background)** — the sought class vs the rest.
- **sigmoid / logit / log-odds** — the squash $\frac{1}{1+e^{-z}}$; the pre-squash
  score; $\log\frac{p}{1-p}$.
- **logistic regression** — linear log-odds model trained on the Bernoulli NLL.
- **Bernoulli** — a single yes/no random variable with success probability $p$.
- **log loss / binary cross-entropy** — $-[y\log p + (1-y)\log(1-p)]$, summed.
- **softmax / one-hot** — score-vector → probability-vector map; the 0/1 label
  encoding with a single 1.
- **decision threshold** — the cutoff $t$ turning $p$ into an action.
- **confusion matrix / TP / FP / FN / TN** — the 2×2 outcome counts.
- **accuracy, precision (purity), recall (efficiency, TPR), FPR, background
  rejection** — the derived rates; rejection $= 1 - \mathrm{FPR}$.
- **class imbalance / class weights / stratified split** — rare positives; per-class
  loss multipliers; splits that preserve the class ratio.
- **ROC curve / AUC / PR curve** — TPR-vs-FPR across thresholds; its area
  (rank-quality probability); precision-vs-recall across thresholds.
- **event / trigger** — one recorded collision; the real-time keep/discard
  classifier every collider experiment runs.
- **significance $S/\sqrt{S+B}$** — signal excess measured in units of the
  background's Poisson fluctuation.
- **calibration / reliability diagram / Brier score / overconfidence** — predicted
  probabilities matching observed frequencies; the binned diagnostic plot; mean
  squared probability error; predicting more certainty than reality delivers.
- **Platt scaling / isotonic regression** — post-hoc calibration by logistic map /
  by monotone step fit.

## Going deeper

- Bishop, *PRML* (in `references/`) §4.3 — probabilistic discriminative models: the
  same logistic-regression derivation with more generality (and IRLS, the
  second-order training method we skipped).
- StatQuest, "Logistic Regression" and "ROC and AUC" videos — fast intuition passes
  that pair well before/after the §4 derivation.
- scikit-learn User Guide — "Model evaluation" (ROC, precision-recall) and the
  "Probability calibration" page; the calibration page is short and unusually good.
- Murphy, *Probabilistic Machine Learning: An Introduction* (free PDF), logistic
  regression chapter — a second derivation angle if Bishop's notation fights you.
