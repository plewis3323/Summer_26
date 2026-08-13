# Month 02 — Math for ML

The arc: rebuild the mathematical core of ML from the ground up. Week 05 builds
calculus from first principles — slope, derivative, gradient, chain rule — and
ends with gradient descent, the algorithm that trains every later model. Week 06
rebuilds linear algebra as *linear maps and subspaces*, not just matrix
arithmetic, and pays the geometric debt Week 05 left (why the gradient points
uphill; how to solve $\nabla L = 0$). Week 07 earns the SVD, derives PCA two
ways, and sets the matrix-calculus conventions Phase 2 will need for backprop.
Week 08 rebuilds probability up through MLE/MAP and information theory, then
derives and races GD / momentum / RMSProp / Adam on surfaces designed to hurt.

Every derivation is done on paper first (photograph or scan into the week
folder), then verified numerically in the exercise notebook, built per
`NOTEBOOK_RULES.md`.

**Month-end deliverable:** a from-scratch optimizer module
(GD/SGD/momentum/RMSProp/Adam) plus the Week 08 mini-project — physics-model
fits and optimizer races on pathological surfaces — and a folder of scanned
derivations you can redo cold.

**Sign-off:** tag `month-02-complete`, write `retro.md` here, open one issue for
the biggest open question.
