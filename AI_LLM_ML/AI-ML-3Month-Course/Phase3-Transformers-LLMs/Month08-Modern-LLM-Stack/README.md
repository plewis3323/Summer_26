# Month 08 — Prompting and the Modern LLM Stack

Month 07 built a transformer; this month operates the ecosystem built on top of them.
The arc: run and sample from open-weight models, and **treat prompting as a tested
program** (Week 29); change behavior cheaply with LoRA/QLoRA (Week 30); understand
how raw predictors become assistants — RLHF and DPO, with the DPO derivation done
for real (Week 31); then distrust every number an LLM eval produces (Week 32).

The weeks chain into the mini-project: Week 29's prompting eval is the baseline
Week 32 must beat, not a vague "zero-shot." LoRA-fine-tune a 1–3B model to extract
structured fields from domain text (physics abstracts *or* a schema you define).
That extractor is a component of Capstone 3.

**Month-end deliverable:** the fine-tuned extractor checkpoint, its SFT dataset, the
Week-29 prompt-eval table, and a benchmark (fine-tuned vs best prompt vs naive
zero-shot) in a tested repo.

**Sign-off:** tag `month-08-complete`, write `retro.md` (250 words) in this folder, and
open one open-question issue. Close last month's.
