# Month 03 — Classical ML

The arc: use Month 02's math to do supervised and unsupervised learning *honestly*.
Week 09 sets the frame — loss, risk, generalization, bias–variance, and the validation
discipline that everything else depends on. Week 10 moves to classification and the
metrics zoo (ROC/PR/calibration — trigger-efficiency thinking, formalized). Week 11
covers trees and ensembles, the methods that actually win on tabular physics data, and
the HEP BDT lineage from TMVA onward. Week 12 adds unsupervised methods (k-means,
GMM+EM derived, PCA/UMAP in practice) and ships Capstone 1: a tabular particle-ID
classifier with nested CV, calibration, tests, and a writeup.

Exercise notebooks are built when each week starts, per `NOTEBOOK_RULES.md`.

**Month-end deliverable:** the Capstone 1 repo — MAGIC gamma/hadron classifier, full
pipeline, `pytest -q` green, nested-CV results table, calibration plot, and a writeup
that includes what failed first. This is also the Phase 1 gate.

**Sign-off:** tag `month-03-complete`, write `retro.md` here, open one issue for the
biggest open question.
