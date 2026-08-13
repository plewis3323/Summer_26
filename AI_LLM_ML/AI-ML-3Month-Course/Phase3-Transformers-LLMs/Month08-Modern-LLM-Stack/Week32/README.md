# Week 32 — Evaluation + Mini-Project: Abstract Metadata Extractor

LLM evaluation is a systematics-dominated measurement: contamination, judge bias, and
metric choice move the result more than the model often does — treat every benchmark
number the way you treat an uncalibrated detector.

## Objectives

- Explain benchmark contamination and check for it in a concrete case.
- Use LLM-as-judge with eyes open: position bias, verbosity bias, self-preference; and
  validate the judge against your own labels.
- Define and measure hallucination for a structured-output task.
- Ship the mini-project: LoRA fine-tune a 1–3B open model to extract structured
  metadata from domain text (physics abstracts by default), scored against your
  **Week-29 winning prompt** and against naive zero-shot.
- Report results with field-level metrics and honest error analysis, not one headline
  number.

## Core material (~3 hrs)

- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (arXiv 2306.05685) — §§1–3, the bias analyses.
- A benchmark-contamination study of your choice (search "data contamination language
  models"); read one carefully and note its detection method.
- HF `evaluate` docs (skim) and TRL `SFTTrainer` docs (refresher from Week 30).
- Re-read your Week 30 SFT dataset notes before scaling the dataset.

## Exercises (built when the week starts)

This week is mostly the project; exercises are its milestones.

1. Dataset: grow `sft_v0` to ~200 labeled (abstract → JSON) examples — fields:
   collision system, √s_NN, observable(s), centrality, detector/experiment, physics
   topic. Accept when: all validate against the JSON schema; 60/20/20 split with zero
   abstract overlap (dedup check passes).
2. Prompt baseline: rerun your Week-29 winning prompt (and a naive zero-shot) on
   this week's test split. Accept when: field-level precision/recall/F1 per field
   for both, in a table — this is the number LoRA has to beat.
3. LoRA fine-tune (Week 30 recipe) on the train split. Accept when: training runs to
   completion with val-loss curve saved and the adapter checkpoint committed or linked.
4. Head-to-head: fine-tuned vs Week-29 best prompt vs naive zero-shot on the held-out
   test split. Accept when: one table, per-field and aggregate F1, plus JSON-validity
   rate for each.
5. Hallucination measurement: fraction of extracted values not present in (or entailed
   by) the source abstract, by field. Accept when: rate reported per system and the
   worst field identified.
6. LLM-as-judge cross-check: a judge model scores 50 extractions; compare with your
   schema-based scores. Accept when: agreement rate reported and one judge failure
   dissected in three sentences.
7. Contamination check: search your test abstracts for verbatim presence in the base
   model (e.g. completion-overlap probing). Accept when: method + result stated in one
   short paragraph.

## Deliverable

A tested mini-project repo: dataset (with provenance), training script, adapter
checkpoint, eval script producing the benchmark table from one command, and a one-page
writeup including what failed first. This extractor feeds Capstone 3 option (a).

## Review

- Week 31: reproduce the key step of the DPO derivation — why does log Z(x) cancel?
- Week 10: you're reporting per-field F1. Define precision, recall, F1 from the
  confusion matrix, and say why accuracy is the wrong metric here.
- Week 9: contamination is a leakage problem. Name the two leakage checks from your
  Capstone 1 pipeline and their analogs here.
- Week 28: how does metric choice (exact-match JSON vs per-field score) connect to the
  emergence-mirage argument?
