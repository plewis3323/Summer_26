# Month 08 — The Modern LLM Stack

Month 07 built a transformer; this month operates the ecosystem built on top of them.
The arc: run and sample from open-weight models properly (Week 29), change their
behavior cheaply with LoRA/QLoRA (Week 30), understand how raw predictors become
assistants — RLHF and DPO, with the DPO derivation done for real (Week 31) — and then
learn to distrust every number an LLM eval produces (Week 32).

The weeks chain into the mini-project: Week 29's model-loading and sampling skills,
Week 30's PEFT tooling and SFT dataset design, and Week 31's understanding of what
tuning does are all spent in Week 32's project — LoRA fine-tuning a 1–3B model to
extract structured metadata from nuclear physics abstracts, scored against a zero-shot
baseline. That extractor is a component of Capstone 3.

**Month-end deliverable:** the fine-tuned extractor checkpoint, its SFT dataset, and a
benchmark table (fine-tuned vs zero-shot, field-level scores) in a tested repo.

**Sign-off:** tag `month-08-complete`, write `retro.md` (250 words) in this folder, and
open one open-question issue. Close last month's.
