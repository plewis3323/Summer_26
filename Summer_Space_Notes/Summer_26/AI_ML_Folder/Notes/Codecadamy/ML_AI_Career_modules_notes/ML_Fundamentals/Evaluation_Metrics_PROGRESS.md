# Evaluation Metrics — Progress Tracker

**Lesson:** Codecademy MLE Path → ML Fundamentals → *Evaluation Metrics*
**Note file:** `Evaluation_Metrics_Lesson.ipynb`
**Started:** 6/19/26

---

## Status: ✅ COMPLETE (6/20/26)
All five sublessons folded in from the **real Codecademy source** with their real exercises.
Notebook runs top-to-bottom clean (`nbconvert --execute`, exit 0). Verified outputs on lesson data:
accuracy 0.30, recall 0.4286, precision 0.50, F1 0.4615, matrix `[[0 3] [4 3]]` — all agree with
the sklearn `classification_report` cell.

**Fixes made this stretch:** (1) accuracy cell made self-contained (referenced counters defined
later); (2) recovered from a NotebookEdit cell-id aliasing slip that overwrote the recall exercise
— see memory `notebookedit_cell_id_aliasing`.

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded into the notebook **from the real source**.

- [x] **Sublesson 1 — Confusion Matrix** (TP / FP / FN / TN) ✅ matched to source layout + real exercise code
- [x] **Sublesson 2 — Accuracy** ✅ formula + worked example from exercise data
- [x] **Sublesson 3 — Recall** ✅ real source (top-secret-email example) + real exercise (recall = 3/7 ≈ 0.4286) + Q&A (cancer-screening / fishing-net)
- [x] **Sublesson 4 — Precision** ✅ real source (always-"spam" mirror example) + real exercise (precision = 3/6 = 0.5)
- [x] **Sublesson 5 — F1 Score** ✅ real source (harmonic-mean, recall 1 / precision 0.02 worked example) + real exercise (F1 = 0.4615)
- [x] **Review — Choosing a metric** ✅ context-driven choice; spam→precision (real email in spam worse than spam in inbox), cancer→recall; FN→recall / FP→precision / both→F1 rule of thumb
- [x] Q/A captured — 1 Recall Q&A folded in; section open for more on any future pass
- [x] Real Codecademy exercises swapped in for recall / precision / F1

## How we work this lesson (so Claude stays consistent)
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** for that sublesson → I add it to the **Q & A** section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant metric.
4. On **"done"** → final pass, update TL;DR, tick this checklist, commit if asked.

## Open questions / to verify
- Confirm Codecademy's exact sublesson titles & order vs. my scaffold.
- Does this lesson cover **multi-class** averaging (macro/micro/weighted) or just binary? Add if so.
- Any ROC/AUC or threshold-tuning content in this lesson? (Not yet added.)
