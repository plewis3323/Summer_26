# Week 02 — Reading

## Primary

1. **Bishop, *Pattern Recognition and Machine Learning*.**
   - §3.1 "Linear Basis Function Models" (pp. 137–146). Derivation of normal equations.
   - §3.1.4 "Regularized Least Squares" (pp. 144–146). Ridge and Lasso framing.
   - §4.3.2–4.3.4 (pp. 205–210). Logistic regression and IRLS.
   - §14.3 "Boosting" (pp. 657–663). The original AdaBoost exposition.
   - §14.4 "Tree-based Models" (pp. 663–666). Short but worth reading for the geometric picture.

2. **Guest, Cranmer, Whiteson — *Deep Learning and its Application to LHC Physics* (Annu. Rev. Nucl. Part. Sci. 68, 2018). arXiv:1806.11484.**
   - Read the full paper this week. It situates both BDTs and early DL in the HEP analysis workflow. §2 (tabular analyses) is your frame for this week; §3 (image-based) is Week 04 preview.

3. **Chen & Guestrin — *XGBoost: A Scalable Tree Boosting System* (KDD 2016). https://arxiv.org/abs/1603.02754.**
   - Read §2 and §3 carefully. The objective function with regularization (Eq. 2) and the split-finding algorithm with gradient statistics are the crux of the XGBoost design.

4. **Friedman — *Greedy Function Approximation: A Gradient Boosting Machine* (Annals of Statistics, 2001).**
   - Historical, but the cleanest statement of gradient boosting as function-space gradient descent. Read §1–4. Available at https://jerryfriedman.su.domains/ftp/trebst.pdf.

## Supplementary

5. **Murphy, *Probabilistic Machine Learning: An Introduction*, Ch. 10 "Logistic Regression".** Free at https://probml.github.io/pml-book/. Read §10.1–10.4. More modern notation than Bishop and pairs well.

6. **Radovic et al. — *Machine Learning at the Energy and Intensity Frontiers of Particle Physics* (Nature 560, 41, 2018).**
   - Glossy but useful high-level overview. Worth knowing the canonical paper.

7. **Ke et al. — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 2017).**
   - Skim §3 (GOSS and EFB). You don't need to implement these, just know what they're for.

8. **Kingma & Ba — *Adam: A Method for Stochastic Optimization* (ICLR 2015). arXiv:1412.6980.**
   - Read §2–3. Adam isn't used in classical ML but you'll meet it in Week 03. Read it now, understand the β₁/β₂ debiasing; you'll recognize the intuition.

## Codecademy (Pro)

- **"Build a Machine Learning Model" — modules 3–6.** Linear regression, logistic regression, model evaluation. Complements Bishop; faster-moving.
- **"Feature Engineering".** Full course. Standardization, encoding, feature construction idioms. Watch the categorical-encoding section closely — you'll use it in Week 07 for metadata extraction.

## Videos

- **Andrew Ng — *Machine Learning Specialization, Course 2 Week 3* ("Advice for applying machine learning").** The bias/variance lectures. ~1.5 hours. Dated examples, durable ideas.

- **StatQuest with Josh Starmer — Gradient Boosting playlist.** https://www.youtube.com/watch?v=3CC4N4z3GJc and the four follow-ups. Slow but extremely clear, with worked numerical examples. Watch during slow commute time.

## HEP-ML Living Review

Open https://iml-wg.github.io/HEPML-LivingReview/ and browse these sections:
- "Classification / Boosted Decision Trees" — skim the first page of citations to see the scope.
- "Classification / Multi-class" — note where BDTs show up vs NNs.

## Papers worth *knowing exist* but skipping this week

- Krizhevsky et al. AlexNet (Week 04 reading).
- He et al. ResNet (Week 04 reading).
- The entire "deep sets" and permutation-invariant literature (Week 05).

## Notes to take

- A two-column page: derivation of logistic-regression gradient on the left, step-by-step explanation on the right. Keep this forever.
- A table comparing XGBoost / LightGBM / CatBoost on (speed, best-use-case, categorical handling). Cite the papers.
- One paragraph: in your thesis context, where would a BDT fail that a CNN wouldn't? You'll re-read this in Week 04.
