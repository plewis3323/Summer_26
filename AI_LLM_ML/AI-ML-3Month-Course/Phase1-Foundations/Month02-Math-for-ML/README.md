# Month 02 — Math for ML

The arc: rebuild the mathematical core of ML from the ground up, at physicist speed —
fast where your PhD already covers it (Gaussians, Lagrange multipliers), slow where ML
uses tools physics rarely does (SVD as a workhorse, matrix calculus conventions, KL
divergence, stochastic optimizers). Week 05 rebuilds linear algebra as *linear maps and
subspaces*, not just matrix arithmetic. Week 06 earns the SVD and derives PCA twice.
Week 07 rebuilds probability up through MLE/MAP and information theory. Week 08 derives
and implements the optimizer zoo, then races it on physics fits.

Every derivation is done on paper first (photograph or scan into the week folder), then
verified numerically in the exercise notebook, built per `NOTEBOOK_RULES.md`.

**Month-end deliverable:** a from-scratch optimizer module (GD/SGD/momentum/RMSProp/Adam)
plus the Week 08 mini-project — physics-model fits and optimizer races on pathological
surfaces — and a folder of scanned derivations you can redo cold.

**Sign-off:** tag `month-02-complete`, write `retro.md` here, open one issue for the
biggest open question.
