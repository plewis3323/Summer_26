# Hyperparameter Tuning — Progress

- **Source:** Codecademy — Deep Learning with TensorFlow path, "Hyperparameter Tuning" lesson
- **Note file:** `Hyperparameter_Tuning_Lesson.ipynb`
- **Started:** 2026-07-21

## Status
Done. Full notebook (Introduction through TL;DR) covers the whole lesson. Per user instruction this session, code cells are commented-out pseudocode (not fully runnable) since model.py/data pieces were never provided. `jupyter nbconvert --to notebook --execute --inplace` ran clean (exit 0) — trivially, since every code cell is comments only, not a real execution check.

## Sublesson checklist
- [x] Introduction to hyperparameter tuning (train/validation/test split, tuning loop, pipeline diagram)
- [x] Using a validation set for hyperparameter tuning (`validation_split` in `.fit()`)
- [x] Exercise: fit with 40 epochs, batch_size=8, verbose=1, validation_split=0.33 (model.py not provided — commented pseudocode only)
- [x] Manual tuning: learning rate (gradient descent, too-large vs too-small tradeoffs, example lr table)
- [x] Exercise: sweep learning_rates=[1e-3, 1e-4, 1e-7], plot train/val loss curves per rate (model.py not provided — commented pseudocode only)
- [x] Manual tuning: batch size (batch/stochastic/mini-batch GD, tradeoffs, lr-batch size trick)
- [x] Exercise: batches=[4,32,64], then raise learning_rate to 0.1 to fix large-batch performance (model.py not provided — commented pseudocode only)
- [x] Manual tuning: epochs and early stopping (overfitting example MAE ~1000 train vs ~3034 val, Keras `EarlyStopping` callback: monitor/mode/patience)
- [x] Exercise: EarlyStopping(patience=20) inside fit_model(), lr=0.1, num_epochs=500, expected stop at epoch 47 (model.py not provided — commented pseudocode only)
- [x] Manual tuning: changing the model (underfitting via one-layer model vs one-hidden-layer model, early stopping at epoch 38, "1 hidden layer / #units = #features" heuristic)
- [x] Exercise: full one_layer_model/more_complex_model pipeline, hidden units 64->8, lr=0.1, batch_size=2, num_epochs=200 (model.py not provided — commented pseudocode only)
- [x] Towards automated tuning: grid and random search (KerasRegressor + GridSearchCV / RandomizedSearchCV, param_grid, greater_is_better=False, n_iter)
- [x] Exercise: full do_grid_search()/do_randomized_search() script, batch_size=[6,64], epochs=[10,50], return_train_score, n_iter=12 (model.py not provided — commented pseudocode only)
- [x] Regularization: dropout (dropout rate, design_model_no_dropout vs design_model_dropout example, why val MAE < train MAE with dropout)
- [x] Exercise: add Dropout(0.3) after the 24-neuron layer, compare no-dropout vs dropout learning curves (model.py not provided — commented pseudocode only)
- [x] Baselines: how to know performance is reasonable (majority-class example, DummyRegressor mean-strategy baseline $9,190 vs tuned ~$3,000)
- [x] Exercise: DummyRegressor strategy="median" instead of "mean" (model.py not provided — commented pseudocode only)

## How we work this lesson
User pastes sublesson text -> concise markdown summary folded into the notebook.
User gives Q&A -> appended to the notebook's `## Q & A` section immediately.
User gives exercise code -> added as a runnable code cell, transcribed faithfully.
On "done" -> final TL;DR pass, run notebook end-to-end, update this tracker.

## Open questions / to verify
- Which learning rate (1e-3, 1e-4, or 1e-7) gave the best performance in the exercise's actual Codecademy run — left as a TODO in the notebook since we didn't execute the code.
- Whether raising learning_rate to 0.1 actually fixed batch=32/64 performance in the batch-size exercise's actual Codecademy run — left as a TODO in the notebook.
- Whether early stopping actually triggers at epoch 47 (and final train/val MAE) in the EarlyStopping exercise's actual Codecademy run — left as a TODO in the notebook.
- At 8 hidden units (down from 64), which epoch early stopping triggers on in the actual Codecademy run — left as a TODO in the notebook.
- Best grid-search combo with batch_size=[6, 64] in the actual Codecademy run (lesson's sample output used a different batch_size range) — left as a TODO in the notebook.
- Median-strategy baseline MAE vs the $9,190 mean-strategy baseline — left as a TODO in the notebook.
