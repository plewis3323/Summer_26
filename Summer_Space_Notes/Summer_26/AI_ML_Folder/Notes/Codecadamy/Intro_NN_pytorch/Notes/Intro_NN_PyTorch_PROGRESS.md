# Intro to NN & PyTorch (Codecademy) — Progress Log

_Last updated: 2026-06-21_

## Where we are
Note-taking through Codecademy's **Intro to Neural Networks with PyTorch** course. Notes live in **3
self-contained Jupyter notebooks** in this folder (see Files). **Note-taking marked DONE by user on
2026-06-21**, covering **Lessons 1–11** (tensors → testing/evaluation). All three notebooks were
re-executed clean (0 errors) at finalization.

**If resuming the course:** the next lessons would be **data normalization** and/or the **final
project** — add them to **Part 3 (Training)**. Each notebook is self-contained; run its Setup cell first.

✅ **Split into 3 parts** (2026-06-21) because the single notebook hit the Read-tool token limit. Stale
pre-split monolith was **deleted** at finalization (fully superseded by the parts).

⚠️ **Plot-embedding rule:** inline matplotlib (`plt.show()`) embeds the PNG as base64 → blew Part 3 to
108KB (over Read limit). Fix = **`plt.savefig('images/…png'); plt.close()`** then embed via markdown
`![](images/…png)` (matches the diagram convention). Do this for all future plots.

## Workflow (how we build the notebook)
- I paste course content; Claude folds it into the notebook:
  - **Markdown cells** = reading/concepts, condensed.
  - **Code cells** = exercises (runnable) with **real captured outputs**.
  - **Q&A** = block quotes folded under the relevant section.
- Course exercises that need data we don't have (e.g. `streeteasy.csv`) use a **small fake
  DataFrame** stand-in so the cell runs; the real `pd.read_csv(...)` line is kept as a comment.
- Diagrams saved to **`images/`** and embedded inline.

## Environment (already set up)
- Installed in sandbox: **torch 2.12.1+cu130, numpy 1.26.4, pandas 2.2.2**, jupyter/nbconvert.
- CPU-only (CUDA not available) — fine for this course.
- Notebook executed via: `jupyter nbconvert --to notebook --execute --inplace Intro_NN_PyTorch_Notes.ipynb`

## Lessons completed (in the notebook)
1. **Intro to Tensors** — `torch.tensor()`, dtypes, DataFrame→tensor (`.values`, double-bracket / `.view(-1,1)`). + exercise (int tensor, df→float32 tensor).
2. **Linear Regression Review** — regression vs classification, $y=mx+b$, algebra→NN vocab (weight/bias), multi-feature form. + checkpoint exercise.
3. **Linear Regression with Perceptrons** — perceptron = inputs→single output node; forward pass diagrams. + example + bedrooms exercise. + Q&A "what is a perceptron?"
4. **Activation Functions** — non-linearity, ReLU ($\max(0,x)$), sigmoid, diagram icons. + ReLU illustration + 3-checkpoint exercise.
5. **Multi-Layer Networks** — hidden layers, each node = perceptron-like (weighted sum + bias + activation), "no separate bias node" convention, 4-step training loop (forward → loss → backprop → iterate).
6. **Build a Sequential Neural Network** — `nn.Sequential` container; `nn.Linear(in,out)` (layer + weighted-sum+bias) and `nn.ReLU()`; layer alignment rule; random init. Feedforward via `model(X)`; input tensor convention (capital `X`, 2-D rows=examples/observations, cols=features); `grad_fn=<AddmmBackward0>` foreshadow. Code cell seeds `manual_seed(42)` → reproduces course's exact `-23.0715 / -11.8710`. + Q&A on random-init/seed. + Q&A "what is a Sequential container" / Sequential-vs-subclassing-nn.Module.
   - **Exercise (CP 1–3):** 3→8(ReLU)→1; add 2nd hidden 8→4(Sigmoid)→1; "halving" net 3→16→8→4→1 feedforward on faked streeteasy. Faked CSV uses the **exact 5 apartments the course previews**, so seed-42 feedforward reproduces course's `predicted_rent[:5]` exactly (`-6.9229 …`).
7. **Build a Neural Network Class** — non-sequential nets via OOP (subclass `nn.Module`) for skip/loop/branch logic Sequential can't do. Steps: define `class NN_Regression(nn.Module)`; `__init__` ("gather ingredients" — layers + `relu` as `self.` attrs, `super(NN_Regression,self).__init__()` = older explicit form of `super().__init__()`); `forward(self,x)` ("combine ingredients" — chained `x=self.layerN(x)`); instantiate `model=NN_Regression()`. `model(X)` auto-calls `forward`. Same 3→16→8→4→1 halving net + seed 42 → **reproduces Checkpoint 3's `predicted_rent` exactly**, proving OOP==Sequential here.
   - **Exercise (CP 1–3):** CP1 = the `NN_Regression` cell above (not repeated). CP2 = `OneHidden` 2→4→1, fill in `forward` (empty `forward`=`return x` passes input through unchanged); input `[3,4.5]`→`2.4422`. CP3 = parameterized `OneHidden(numHiddenNodes)` — hidden width set at instantiation; `OneHidden(10)`→`1.2633`, `OneHidden(4)`→`2.4422` (==CP2). Payoff: one class → many architectures via an `__init__` arg.
8. **The Loss Function** — measures error between predictions and targets (`y`, lowercase = 1-D). Plain difference fails (pos/neg cancel → fake 0 loss). **MSE** = mean of squared diffs (`nn.MSELoss()`, call with `(predictions, y)`); huge number → take **RMSE** (`sqrt`) to read on original scale. Example apts: MSE 250000, RMSE 500. **MAE** (`nn.L1Loss`, abs value) less outlier-sensitive than MSE; =500 here. MSE squaring emphasizes big errors → can overfit. Maps to 3B1B Ch.2 cost function.
   - **⚠️ Gotcha captured:** `nn.MSELoss(predictions,y)` (data in constructor) → `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. Must **instantiate then call**: `loss=nn.MSELoss(); loss(pred,y)` (or `nn.MSELoss()(pred,y)`).
   - **Exercise (CP 1–3):** CP1 manual MSE (750,1000 vs 1000,900) → 36250. CP2 `nn.MSELoss()` on untrained net's 5 preds vs real rents → 38,413,624. CP3 RMSE `MSE**(1/2)` → ~6197.87 (≈$6,200 off/apt before training).
9. **Backward Pass & Optimizer Step.** Training lowers loss in 2 moves: **backward pass** `MSE.backward()` computes gradients (downhill = −gradient); **step** `optimizer.step()` updates weights/biases. `.backward()` called on the **computed loss tensor** (not the loss fn) — works because loss carries `grad_fn=<MseLossBackward0>` (the backward function; explains all the `grad_fn=<...>` tags). Gradients **accumulate** → need `optimizer.zero_grad()` each iter. Demo cell: 3 SGD steps (tiny lr=1e-7, raw data unnormalized) lowers loss 38,413,624→38,196,724; grad is `None` before backward. Maps to 3B1B Ch.2–4.
   - **Optimizer exercise (CP 1–3):** CP1 `import torch.optim as optim`. CP2 `optimizer = optim.Adam(model.parameters(), lr=0.001)`. CP3 one full step (fwd→loss→backward→step→new loss) lowers it. **Adam** = adaptive per-param step size → robust to unnormalized data (lr=0.001 works where SGD needed 1e-7). LR trade-off: too small=slow, too big=diverge. ⚠️ Faked 5-row data **can't match course's full-dataset numbers** (29,213,900→29,205,546). ⚠️ Course's 1-D `y` vs `[N,1]` preds → `MSELoss` broadcast warning; we use `y.view(-1,1)`.
10. **Training (the loop).** Turn the 4 steps into a `for epoch in range(num_epochs)` loop; **epoch** = one loop iteration. New line **`optimizer.zero_grad()`** clears gradients each iter (they **accumulate** by default → want a fresh direction). Print every 100 epochs via `if (epoch+1)%100==0` (epoch 0-indexed) using `MSE.item()` (avoids printing `grad_fn`). Demo: 1000 epochs Adam lr=0.001 on faked raw data → loss only crawls (38.19M→38.18M) because **unnormalized features** make lr too small → motivates normalization next. Loop logic verified correct.
    - **Training-loop debugging exercise (CP 1–3):** now **14 features** (bedrooms/bathrooms/size_sqft/min_to_subway/floor/building_age + 8 binary amenity flags), net **`14→128→64→1`** (halving). Captures the **2 classic loop bugs by symptom**: CP1 *loss extremely high* = **missing `zero_grad`** (grads accumulate); CP2 *loss frozen* = **update before grads** (`step`/`zero_grad` before `backward`, or `backward` missing). Rule: `backward()`→`step()` with one `zero_grad()` **not** sandwiched between. CP3 = 1000 epochs, diminishing returns. Demo cell (synthetic 200-row, won't match course's 3.13M→2.63M) shows all 3: correct 1.23M→0.9M, no-zero_grad 33M→27M, frozen 43.9M flat. **All verified.**
11. **Testing & Evaluation.** Training loss only measures fit on data learned from → also **evaluate on new data** + **save** the model. **Train-test split** via `sklearn.model_selection.train_test_split(X, y, train_size=.8, test_size=.2, random_state=2)` → 4 tensors; `random_state` = reproducible split; train on train set, eval on test set. **Eval:** `model.eval()` (eval mode — affects dropout/batchnorm) + `with torch.no_grad():` (no gradient tracking → faster/less mem), then feedforward + MSE on test. Test ≫ train loss = **overfitting**. **Save/load:** `torch.save(model,'model.pth')` / `torch.load(...)`. ⚠️ **torch≥2.6 gotcha:** `torch.load` defaults `weights_only=True` → whole-model load **fails** (`UnpicklingError`); need `weights_only=False` (best practice: save/load `state_dict` instead). Demo verified: split 160/40, train 986K/test 814K, reloaded model identical test MSE. sklearn 1.7.2 + matplotlib 3.9.0 available in sandbox.
    - **Eval exercise (CP 1–4 + viz):** CP1 70/30 split random_state=2. CP2 fix the bug = train on `X_train/y_train` not full `X/y` (data leakage). CP3 `torch.save(model,'model.pth')`. CP4 load `model20k.pth` (course's 20k-epoch model) → eval → course test MSE ≈1,997,977 / RMSE ≈1413. **Viz:** scatter predicted-vs-actual rent (test set) + dashed y=x line → `images/l11_pred_vs_actual.png`. Faked: trained our own 20k-epoch model on synthetic noisy data (RMSE ~271), saved/loaded/eval/plotted; workflow faithful, numbers differ.

## Appendices
- **Python OOP reference** (in **Part 2**, after Lesson 6 exercise) — class/instance/self/attributes/methods/inheritance/`super()`; `__init__` vs `if __name__=="__main__"`; the `nn.Module` subclass pattern + runnable cell proving Sequential == subclassed net give identical output.

## Status / TODO
- [x] **Note-taking DONE (2026-06-21)** — Lessons 1–11 captured across the 3 part notebooks; all execute clean.
- [x] ~~Decide whether to fake/grab `streeteasy.csv`~~ → **kept faking** (small stand-in DataFrame; real `read_csv` as comment).
- [x] ~~Split into per-lesson notebooks~~ → **done**: 3 parts (2026-06-21); stale monolith deleted.
- [ ] *(If course continues)* add **data normalization** / **final project** to **Part 3**.

## Files
- `Intro_NN_PyTorch_Part1_Foundations.ipynb` — Lessons 1–5 (tensors → multi-layer). 22 cells.
- `Intro_NN_PyTorch_Part2_Building_Networks.ipynb` — Lessons 6–7 + OOP reference (Sequential → NN class). 15 cells.
- `Intro_NN_PyTorch_Part3_Training.ipynb` — Lessons 8–11 (loss → optimizer → training loop → testing/eval). 24 cells. **← edit here for Lesson 12+.**
- `images/` — saved course diagrams + plots (perceptron_*, multilayer_network.png, sequential_network.png, l11_pred_vs_actual.png).
- `Intro_NN_PyTorch_PROGRESS.md` — this log.

## Conventions for editing the split notebooks
- Each part is **self-contained**: run its **Setup** cell first (imports torch/nn/optim/numpy/pandas). Part 3's setup also re-defines `NN_Regression` (used by the Lesson 9 demo).
- Part notebooks are small enough to use **NotebookEdit** again (no more direct-JSON appends needed).
- Execute a part with: `jupyter nbconvert --to notebook --execute --inplace <PartFile>.ipynb`.
