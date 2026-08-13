# Week 32 Project — Abstract Metadata Extractor

## Objective

LoRA-fine-tune a 1–3B open-weight model to read a nuclear physics abstract and emit
its metadata as schema-valid JSON, and *measure* the result honestly: fine-tuned vs
zero-shot vs few-shot prompting, on a held-out test split, reported as per-field
precision/recall/F1 plus JSON validity and hallucination rate. This is the month's
deliverable (see the Month 08 README) and a named component of Capstone 3 option (a),
the physics-literature assistant: an assistant that can answer "which STAR papers
measured v2 in Au+Au at 200 GeV?" needs exactly this extractor behind it. It is also
your first complete LLM *experiment* — dataset, baselines, intervention, controlled
measurement — the shape every later evaluation in the course reuses.

## Background — the task, concretely

**What a nuclear physics abstract looks like.** Your Week 27 corpus is scraped
heavy-ion abstracts from arXiv (nucl-ex, nucl-th, and adjacent hep categories) —
heavy-ion physics collides large nuclei to create and study the quark–gluon plasma,
the state of matter where quarks and gluons are briefly unconfined. A typical
experimental abstract is one dense paragraph reading like:

> "We report measurements of the elliptic flow coefficient $v_2$ of charged hadrons
> in Au+Au collisions at $\sqrt{s_{NN}} = 200$ GeV with the sPHENIX detector, in the
> 0–10% and 10–40% centrality intervals. …"

Theory abstracts discuss the same physics with no detector and often no specific
dataset. The metadata is *in there*, but in prose, notation-heavy, and incomplete by
design — which is what makes extraction a real task rather than regex practice.

**The fields** (fixed since Week 30 E5; the schema is frozen — do not extend it
mid-project):

| field | type | meaning | example |
|---|---|---|---|
| `collision_system` | string/null | which nuclei collide; the projectile+target pair | `"Au+Au"`, `"p+Pb"` |
| `sqrt_s_nn_gev` | number/null | center-of-mass energy per nucleon–nucleon pair, in GeV ($\sqrt{s_{NN}}$: the energy available when one nucleon from each nucleus meets) | `200`, `5020` |
| `observables` | list of strings | the measured/computed quantities — e.g. $v_2$ (elliptic flow, the second Fourier coefficient of particle emission around the beam axis), $R_{AA}$ (nuclear modification factor: yield in nucleus–nucleus vs scaled proton–proton) | `["v2", "R_AA"]` |
| `centrality` | string/null | how head-on the collision was, as a percentile interval (0% = most central) | `"0-10%"` |
| `experiment` | string/null | the detector/collaboration; null for theory papers | `"STAR"`, `"sPHENIX"` |
| `physics_topic` | string | short topic label from your controlled vocabulary | `"collective flow"` |

Missing information is JSON `null` — never guessed. That rule, set in Week 30, is
what makes hallucination measurable (lesson §4) and abstention trainable.

**Gold JSON for the abstract above:**

```json
{"collision_system": "Au+Au", "sqrt_s_nn_gev": 200,
 "observables": ["v2"], "centrality": "0-10%,10-40%",
 "experiment": "sPHENIX", "physics_topic": "collective flow"}
```

**Corpus continuity.** The same abstracts trained your Week 27 nanoGPT, fed Week 29's
perplexity comparison and Week 30's rank sweep, and seed Capstone 3's document
library. By month's end you will have pretrained on, fine-tuned over, and built
evaluation sets from one corpus you know cold — that familiarity is deliberate: you
can spot a hallucinated $\sqrt{s_{NN}}$ at a glance because you know what energies
exist.

**Model.** Any 1–3B open-weight instruct model; resident default
`Qwen/Qwen2.5-1.5B-Instruct` (permissive license, fits Colab's free tier in bf16 with
LoRA). Llama-3.2-1B/3B-class models work identically if you have accepted the
license.

## Data

- **Raw abstracts:** the Week 27 corpus (`abstracts.jsonl`), already scraped and
  deduplicated; all arXiv abstracts, public and free.
- **Labels:** yours. `sft_v0.jsonl` (Week 30 E5, 30 examples) grows to
  `data/sft_v1.jsonl`, ~200 examples, labeled by hand this week (~3–4 hrs; label
  field-by-field in passes, not abstract-by-abstract). Provenance note in the repo:
  which arXiv IDs, who labeled (you), when, and the labeling rules you fixed —
  including the `physics_topic` controlled vocabulary and every judgment call.
- **Splits:** 60/20/20 train/val/test by abstract after dedup. Test is opened once,
  at the end, for all systems at once; all prompt tuning happens on val.

**Compute** (syllabus §8): this is a designated Colab/Kaggle/cloud week. The
fine-tune itself is small — a 1–3B model with LoRA over ~120 examples is minutes per
epoch on a T4-class GPU — but the generation-heavy eval runs (three systems × 40 test
abstracts, plus the judge and contamination probes) also want a GPU. Budget one
session for training, one for evaluation; cache every model output to disk so no
measurement is ever rerun by accident.

## Build steps

Do them in order; each is one of this week's exercises (E1–E7 in `exercises.md`),
and the order is the method: dataset before baselines, baselines before training,
apparatus frozen before the test split is touched.

1. **Dataset** (E1). Grow to ~200 labeled examples; dedup; split 60/20/20 by
   abstract. Accept when: all examples validate against the JSON schema and the
   split has zero abstract overlap (dedup check passes).
2. **Harness + baselines first** (E2). Write `eval.py` — schema validation,
   `canon()`, per-field TP/FP/FN → P/R/F1 (lesson §§5, 7) — with its scoring
   functions under `pytest`. Hand-check it on toy items where you know every count.
   Then run the zero-shot and few-shot baselines through it. Accept when: field-level
   precision/recall/F1 per field on the test split, in a table.
3. **LoRA fine-tune** (E3). Week 30 recipe on the train split; monitor val loss.
   Accept when: training runs to completion with the val-loss curve saved and the
   adapter checkpoint committed or linked.
4. **Head-to-head** (E4). Fine-tuned vs zero-shot vs few-shot, same harness, same
   decoding, one table. Accept when: per-field and aggregate F1 plus JSON-validity
   rate for each system.
5. **Hallucination measurement** (E5). Ungrounded non-null values per field, per
   system. Accept when: rate reported per system and the worst field identified.
6. **Judge cross-check** (E6). A stronger model judges 50 extractions; compare with
   schema-based scores. Accept when: agreement rate reported and one judge failure
   dissected in three sentences.
7. **Contamination check** (E7). Completion-probe the base model on your test
   abstracts, verbatim vs paraphrase control. Accept when: method + result stated in
   one short paragraph.

## Acceptance gate (from `03-Project-Roadmap.md` and the week README)

- **Beat zero-shot prompting on the held-out test set, reported as per-field F1** —
  the roadmap's criterion. Aggregate macro-F1 higher *and* no catastrophic
  regression hidden in the per-field rows.
- Tested and reproducible: `pytest -q` green (scoring functions, schema/dedup
  checks); fresh clone + `uv sync` + one command reruns the full evaluation from
  cached model outputs and reprints the benchmark table.
- The repo ships: dataset with provenance, `train.py`, adapter checkpoint (committed
  or linked), `eval.py`, `results/` with the benchmark table in machine-readable
  form, and the writeup.
- If the fine-tune does *not* beat zero-shot, the gate is not met — but the course's
  honesty policy still applies: write up the negative result and its diagnosis
  (dataset too small? labels inconsistent? task already saturated by prompting?),
  fix the most likely cause, and rerun once. Fine-tuned-vs-zero-shot on a narrow
  structured task with 120 training examples is winnable; a loss usually indicts the
  dataset or the harness, not the method.

## Writeup requirements (one page)

- The task and the headline table (all systems × per-field F1, validity,
  hallucination rate).
- Data provenance: corpus, label count, split sizes, labeling rules and their
  judgment calls.
- Error analysis: best and worst field with one real example of each error mode —
  at minimum one wrong-but-grounded error and one hallucination, and what
  distinguishes them (lesson §4.2).
- Systematics paragraph: matcher strictness, prompt sensitivity of the baselines,
  judge agreement rate, contamination-probe result — the number's error bars, in
  words.
- **What failed first** — written the day it happened, not reconstructed.
- What you would do next with one more week.

## Stretch goals (only after the gate)

- **Constrained decoding:** force schema-valid JSON at generation time (e.g.
  grammar/JSON-schema-constrained generation via `outlines` or llama.cpp grammars)
  for the *zero-shot* baseline — how much of fine-tuning's edge was just format
  compliance?
- **Dataset-size ablation:** retrain at 50 and 100 examples; plot test macro-F1 vs
  dataset size. Where does the curve knee?
- **1B vs 3B:** same recipe on the other end of the size band; does scale or
  fine-tuning buy more on this task?
- **DPO polish pass (Week 31, applied):** build preference pairs from your own
  model's eval mistakes (chosen = corrected JSON, rejected = the model's error) and
  run a short DPO stage; measure the per-field delta and the hallucination delta.
- **Schema-extension cost study:** add one field (e.g. `p_T` range), label 50
  examples, and record how far the extractor gets zero-shot on the new field vs
  after a top-up fine-tune — data for Capstone 3's "what does adding a field cost"
  question.
