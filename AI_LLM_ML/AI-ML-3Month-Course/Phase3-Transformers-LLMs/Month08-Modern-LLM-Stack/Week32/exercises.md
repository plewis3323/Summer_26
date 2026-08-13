# Week 32 — Exercises

This week is mostly the project; the exercises are its milestones, in build order —
each one is a step of `project.md`, and finishing E1–E7 *is* finishing the project
build. Accordingly, most work lives in the repo, not the notebook: the dataset files
(`data/sft_v1.jsonl` and splits), the training script (`train.py`), the eval script
(`eval.py`, with `tests/test_eval.py` covering the scoring functions), per
`NOTEBOOK_RULES.md` §6. The notebook provides setup (model loading, corpus paths,
seeds), drives the scripts, and holds the analysis cells; its checks verify the
artifacts the scripts produce. The fine-tune (E3) and the head-to-head generation
runs (E2, E4) are GPU work — this is a designated Colab/Kaggle week (syllabus §8).

## E1 — The dataset: grow `sft_v0` to ~200

Grow Week 30's `sft_v0.jsonl` (30 examples) to ~200 labeled (abstract → JSON)
examples, saved as `data/sft_v1.jsonl` — fields: collision system, √s_NN,
observable(s), centrality, detector/experiment, physics topic; missing information is
null, never guessed. Dedup, then split 60/20/20 by abstract into
`data/{train,val,test}.jsonl`.
Hint: label in passes (one field at a time across all abstracts) rather than one
abstract at a time — it is faster and the labels come out more consistent. Keep the
variety rule from Week 30 E5: systems, energies, experiments, and theory abstracts
with null detector.
Accept when: all validate against the JSON schema; 60/20/20 split with zero abstract
overlap (dedup check passes).

## E2 — Baselines first: zero-shot and few-shot

Prompt the base instruct model for the extraction task zero-shot (instructions +
schema in the prompt) and few-shot (same, plus 3–5 worked examples from the *train*
split), greedy decoding, on the test split — through `eval.py`, the same harness every
later system uses. Tune prompt wording on the validation split only.
Hint: freeze `canon()` and the schema before running anything on test (lesson §5.3);
few-shot examples come from train, never test. Parse failures score zero on all
fields — that is data, not a bug to hide.
Accept when: field-level precision/recall/F1 per field on the test split, in a table.

## E3 — LoRA fine-tune

Fine-tune the 1–3B model on the train split with the Week 30 recipe (`peft` + TRL
`SFTTrainer`, chat format, loss on response tokens only), tracking validation loss.
Save the adapter.
Hint: start from Week 30's exact `LoraConfig` and LR (2e-4); a few epochs over ~120
train examples is minutes, not hours, on a Colab GPU. If val loss rises from epoch 1,
suspect the chat-template mismatch footgun (Week 29 §5) before touching
hyperparameters.
Accept when: training runs to completion with val-loss curve saved and the adapter
checkpoint committed or linked.

## E4 — Head-to-head on the held-out test split

Run the fine-tuned model through the same harness and produce the benchmark table:
fine-tuned vs zero-shot vs few-shot.
Hint: identical decoding settings and identical `eval.py` for all three — any
difference in apparatus contaminates the comparison. Macro-F1 is the headline;
the per-field rows are the result.
Accept when: one table, per-field and aggregate F1, plus JSON-validity rate for each.

## E5 — Hallucination measurement

For each system, measure the fraction of extracted non-null values not present in (or
entailed by) the source abstract, by field (lesson §4.2). Automate the
present-or-entailed check with `canon()` plus substring/number matching against the
abstract text; hand-adjudicate the residue and record the calls.
Hint: the interesting comparison is fine-tuned vs zero-shot on null-heavy fields —
did training on honest nulls actually teach abstention?
Accept when: rate reported per system and the worst field identified.

## E6 — LLM-as-judge cross-check

Have a judge model (a stronger instruct model than the extractor) score 50 test
extractions as correct/incorrect per field, and compare its verdicts with your
schema-based scores from E4.
Hint: give the judge the abstract, the schema, and the prediction — not the gold
label (that tests copying, not judging). Where judge and harness disagree, the gold
label decides who erred.
Accept when: agreement rate reported and one judge failure dissected in three
sentences.

## E7 — Contamination check

Probe whether your test abstracts are memorized by the base model: prompt with the
verbatim first half of each test abstract, greedy decoding, and measure overlap of
the continuation with the true second half (lesson §2.3), against a
paraphrased-prompt control.
Hint: normalized longest-common-substring or token-level overlap is enough; the
verbatim-vs-paraphrase *gap* is the signal, not the absolute overlap.
Accept when: method + result stated in one short paragraph.

## Review

1. (Week 31) Reproduce the key step of the DPO derivation — why does log Z(x)
   cancel?
2. (Week 10) You are reporting per-field F1. Define precision, recall, and F1 from
   the confusion matrix, and say why accuracy is the wrong metric here.
3. (Week 09) Contamination is a leakage problem. Name the two leakage checks from
   your Capstone 1 pipeline and their analogs here.
4. (Week 28) How does metric choice (exact-match JSON vs per-field score) connect to
   the emergence-mirage argument?
