# Month 07 — Transformers from Scratch

The arc: derive attention on paper (Week 25), assemble it into a full decoder-only
transformer and read the two founding papers critically (Week 26), build the tokenizer
and train your own nanoGPT-class model on a domain abstracts corpus (Week 27, GPU week; heavy-ion by default), then
step back and ask what happens when you scale it and what is actually going on inside
(Week 28).

Each week feeds the next: the attention module you write in Week 25 becomes a block in
Week 26's GPT; that model plus Week 27's BPE tokenizer is what you train on the arXiv
corpus; Week 28's interpretability exercises probe the very checkpoint you trained.

**Month-end deliverable:** a trained nanoGPT-class checkpoint on your domain corpus,
with the from-scratch model verified against GPT-2 weights, plus a scaling mini-study
and an induction-head hunt in your own model.

**Sign-off:** tag `month-07-complete`, write `retro.md` (250 words) in this folder, and
open one issue for the single biggest thing you don't understand. Close last month's.
