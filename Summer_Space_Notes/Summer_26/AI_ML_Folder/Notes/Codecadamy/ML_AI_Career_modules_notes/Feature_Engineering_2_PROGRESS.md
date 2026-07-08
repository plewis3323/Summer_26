# Feature Engineering II — Progress Tracker

**Lesson:** Codecademy ML & AI Engineer Path → *Feature Engineering #2*
**Note file:** `Feature_Engineering_2_Lesson.ipynb`
**Started:** 6/26/26

---

## Status: ✅ COMPLETE 7/1/26 — all 4 sublessons folded in (filter + 5 wrapper methods + **§4 Feature Importance**, reconciled against the real Codecademy article text + user's 2 images). TL;DR updated. Notebook runs clean (`nbconvert --execute`, exit 0, verified). NOT yet committed (awaiting go-ahead).

**▶ DONE.** If resuming: only remaining action is the optional `git commit`. (User may still add Q&A for §4 — none supplied yet.)

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded in **from the real source**.

- [x] **Sublesson 1 — Introduction to Feature Selection Methods** ✅ filter / wrapper / embedded overview + at-a-glance table + illustrative `VarianceThreshold` demo (no exercise in this article)
- [x] **Sublesson 2 — Filter Methods (worked example)** ✅ student dataset + variance threshold + Pearson correlation (feature-feature & feature-target, `f_regression`) + mutual information (`LabelEncoder`, `discrete_features`, `SelectKBest`/`partial`) + recap table. Verified runs clean.
  - ⚠️ Note: local MI values differ slightly from Codecademy's on `hours_sleep` (sklearn-version kNN estimator drift); ranking & feature-drop decisions identical. Documented inline.
- [x] **Sublesson 3 — Wrapper Methods** ✅ intro + dataset + all 5 methods (SFS/SBS/SFFS/SBFS/RFE) + exercises + recap folded in. Verified clean.
  - Dataset: **Breast Cancer Coimbra** (UCI #451), saved as `breast_cancer_coimbra.csv` (116×10; 9 features + `Classification` 1=healthy/2=patient). Load cell verified.
  - Installed **mlxtend 0.23.4** (needed for floating selectors SFFS/SBFS; sklearn lacks them).
  - Model setup + **exercise** ✅: `LogisticRegression(max_iter=1000)` baseline on health data → training acc ~0.80 (≈80% correct). Lesson exercise (fit + score on dataR2.csv) matches our baseline cell exactly; lesson itself ships `max_iter=1000`. Noted convergence (scale) + optimistic-score caveats; article's 0.984 was its *fire* example, not ours. Notebook re-verified clean.
  - **SFS concept + exercise** ✅: greedy bottom-up (start empty, add best feature each step until k). Set-logic exercise → set1/set2 = current set + {height} / + {blood_pressure} (the unused features). Runnable cell.
  - **mlxtend SFS background** ✅: `SequentialFeatureSelector` (alias SFS) params (estimator/k_features/forward/floating/scoring/cv) + result attrs (`.subsets_`, `.k_feature_names_`, `.k_feature_idx_`, `.k_score_`). One class → 4 variants via forward/floating flags. Goal: pick 3 features for `lr`.
  - **SFS run (mlxtend)** ✅: `SFS(lr, k_features=3, forward=True, floating=False, scoring='accuracy', cv=0).fit(X,y)`. Selected **Age, Glucose, Insulin**, acc **0.767** (vs 0.802 with all 9). Greedy path: Glucose → +Age → +Insulin. Lesson's "80.2%" baseline matches our max_iter=1000 score. Verified clean, no convergence warnings.
  - **SFS results + exercise** ✅: `.subsets_[3]["feature_names"]` → ('Age','Glucose','Insulin'); `["avg_score"]` → 0.767; `plot_sfs(sfs.get_metric_dict())` accuracy-vs-#features plot. Comparison answered: 76.7% (3 feat) vs 80.2% (9 feat) → ~3.5pt drop for 1/3 the features. Dropped Codecademy's `codecademylib3` shim. Plot cell verified clean. **SFS fully covered.**
  - **SBS** ✅: mirror of SFS (start with all, remove one at a time); same `SFS` class with `forward=False`. Worked 5→3 example. Run on health data → chose **('Age','Glucose','Resistin')** acc 0.741 — *different* subset than SFS (good teaching point: greedy directions disagree, neither optimal). Verified clean. **Exercise** (flip `forward=False`, fit) folded in — matches official solution.
  - **SBS evaluation** ✅: `sbs.subsets_` (keys 9→3) + `plot_sfs(get_metric_dict())`. Our path: acc flat **0.8017 from 9→6 feat** (dropping MCP.1/Adiponectin/HOMA is free!), then 5→0.793, 4→0.767, 3→0.741. Teaching point: pick k where curve plateaus. Verified clean. **Exercise** ✅ answered: SBS 3-set ('Age','Glucose','Resistin') ≠ SFS set (differ Resistin vs Insulin); SBS 0.741 < SFS 0.767 here; trade-off discussion folded in.
  - **SFFS / SBFS (floating)** ✅: floating = sequential + backtrack (after each add, try a remove / after each remove, try an add; loop-guarded). Same `SFS` class, `floating=True`. Ran on health data → SFFS=SFS (0.767), SBFS=SBS (0.741): floating *can* help but didn't change result on this small problem (noted). **Exercise** ✅: SBFS add-back, current {age,height}=0.86, adds → `added_feature = "weight"` (0.95 > 0.92 > 0.86). Verified clean.
  - **RFE** ✅: top-down, ranks by |coef| (needs `StandardScaler` — coef size is scale-sensitive), drops smallest one at a time, refits, repeats. Only 1 model fit per elimination → cheaper than SBS (RFE 2 tests vs SBS 6+5=11 for 6→4). Counting **exercise** ✅ (rfe_test_count=2, sbs_test_count=11). sklearn `RFE(lr, n_features_to_select=3).fit(X,y)` on standardized data → kept **BMI, Glucose, Resistin** acc **0.7328**; `ranking_=[4 1 1 2 3 5 7 1 6]`, elim order Adiponectin→MCP.1→Leptin→Age→HOMA→Insulin. Eval via `.support_`/`.ranking_`/list-comp/`.score`. Both **exercises** ✅ (setup+fit; rfe_features+score vs 0.802 baseline). Verified clean.
  - **SFFS/SBFS mlxtend page** ✅: already covered by existing floating section; added the official code-exercise marker (build sffs/sbfs, print names; CP4 = SFFS=SFS, SBFS=SBS).
  - **Wrapper recap** ✅: 5-method comparison table + results table (only Glucose picked by all 5; RFE the lone BMI-picker; floating changed nothing here; pick k at the plateau).
- [x] **Sublesson 4 — Feature Importance** ✅ folded in from the **real article text** + user's 2 images (hiring Gini-tree diagram; Wisconsin feature-importance bar chart). 7 cells:
  - **Intro** ✅: importance = per-feature association score (embedded family); workflow uses (dim. reduction / filter / wrapper-RFE / model inspection & communication).
  - **Gini concept** ✅: hiring example (experience vs certification); Gini impurity $G=1-\sum p_i^2$ + Gini gain (parent − weighted-child impurity); certification = perfect split → higher gain → more important. `feature_importances_` = normalized total Gini gain.
  - **Article code** ✅: `datasets.load_breast_cancer` (30 feats) + `DecisionTreeClassifier(criterion='gini')` + seaborn bar plot. **Fixed the article's 2 bugs** (`data.feature_names`→`dataset.feature_names`; positional `sns.barplot`→`x=`/`y=`). Used `random_state=6` to reproduce the article's headline → **worst concave points 0.748, worst radius #2**; **19/30** feats ~0 importance; tree test acc 0.881.
  - **Pros/cons** ✅: cheap (free from training); biased to numeric/high-cardinality; ignores correlation (credit dumped on one twin); single tree unstable (top feature is seed-dependent — verified across 12 seeds).
  - **Other measures** ✅ (table): aggregate RF / permutation / standardized coefficients; all still correlation-blind.
  - **Other-methods code** ✅: RF aggregate (worst concave points 0.178, steadier spread) + `permutation_importance` on test (~0.005 tiny → correlation-masking demo'd live).
  - **§4 recap** ✅: robustness ladder (single tree → RF aggregate → permutation → coefficients); run §2 correlation filter first.
  - Note: main worked example switched from my provisional Coimbra/RandomForest draft to the article's **Wisconsin/DecisionTree** — the draft cells were replaced.

## Q&A captured (in notebook `## Q & A` section)
- `fit_transform`, `df.iloc`, `iloc` row×column combining, `LabelEncoder` vs fit_transform.

## How we work this lesson (so Claude stays consistent)
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** → I add it to the **Q & A** section (no need to ask — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant section.
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify `nbconvert --execute` is clean, commit if asked.

(Driven by the user-level `lesson-notes` skill.)

**QA gate (this session):** the `qa-approval-gatekeeper` ("manager") agent signs
off at two checkpoints — (1) the "done" final pass (notebook verified clean via
`nbconvert --execute`) and (2) before any `git commit`. Incremental cell folds
are not individually gated.

**Question routing (this session):** workflow/QA/approval questions go to the
manager agent, not the user. Only genuinely personal preferences the manager
can't decide are surfaced to the user.

## Open questions / to verify
- Confirm Codecademy's exact sublesson titles & order for this module.
