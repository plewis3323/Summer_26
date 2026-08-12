# Implementing Neural Networks — Progress

- **Source:** Codecademy — Deep Learning with TensorFlow path, "Implementing Neural Networks" lesson
- **Note file:** `Implementing_Neural_Networks_Lesson.ipynb`
- **Started:** 2026-07-20

## Status
Done. Full notebook (Introduction through TL;DR) executes end-to-end clean, ~2 min total runtime (two training runs included).

## Sublesson checklist
- [x] Introduction
- [x] Main Components of a Neural Network Pipeline
- [x] Predicting Medical Costs: Loading the Data (insurance.csv, pandas load/split/describe)
- [x] Data Preprocessing: One-Hot Encoding and Standardization (get_dummies, train_test_split, ColumnTransformer + Normalizer)
- [x] Exercise: StandardScaler variant of the ColumnTransformer preprocessing
- [x] Neural Network Model: tf.keras.Sequential (design_model(), empty layers list)
- [x] Neural Network Model: Layers (Dense, weight/bias matrix shapes, lazy shape inference)
- [x] Exercise: varying #samples/#features/#neurons and observing weight/bias shape effects
- [x] Neural Network Model: Input Layer (InputLayer, model.add(), model.summary(), 0 trainable params)
- [x] Neural Network Model: Output Layer (Dense(1) for regression, shape auto-inferred)
- [x] Neural Network Model: Hidden Layers (Dense(64, relu), full input->hidden->output design_model(), 705 params verified)
- [x] Exercise: 128-unit hidden layer variant of design_model() — 1,409 params verified
- [x] Optimizers (Adam, learning rate as hyperparameter, model.compile with mse loss + mae metric)
- [x] Exercise: compiling (Adam + .compile()) inside design_model() itself
- [x] Training and Evaluating the Model (model.fit epochs/batch_size, model.evaluate) — MAE $3800.50 verified, matches lesson's ~$3800-3884 ballpark
- [x] Exercise: seeded full pipeline (set_seed(35), lr=0.1, epochs=40, batch_size=1) — MAE $2695.54 verified, notebook runs clean end-to-end

## How we work this lesson
User pastes sublesson text -> concise markdown summary folded into the notebook.
User gives Q&A -> appended to the notebook's `## Q & A` section immediately.
User gives exercise code -> added as a runnable code cell, transcribed faithfully.
On "done" -> final TL;DR pass, run notebook end-to-end, update this tracker.

## Open questions / to verify
- Lesson text describes 11 input features after one-hot encoding; our `insurance.csv` has `sex`/`smoker` pre-coded as 0/1 integers (not strings), so `get_dummies` doesn't expand them — our actual feature count is 9, confirmed via `layer.weights[0].shape == (9, 3)`.

## Environment notes
- `tensorflow-cpu` was not installed for the notebook's kernel (python3.10, dist-packages) — installed it via `python3 -m pip install tensorflow-cpu` (now 2.21.0) so the Sequential-model cells can execute.
- TF import cells set `TF_CPP_MIN_LOG_LEVEL=3` and `TF_ENABLE_ONEDNN_OPTS=0` before importing tensorflow, to keep oneDNN/absl log spam out of cell outputs.
