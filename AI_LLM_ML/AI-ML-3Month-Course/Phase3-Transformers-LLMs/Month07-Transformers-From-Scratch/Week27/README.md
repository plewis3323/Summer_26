# Week 27 — Tokenization + nanoGPT (GPU week)

Tokenization is the detector layer of an LLM — a lossy, quirky digitization of text that
every downstream inference inherits, and the source of a disproportionate share of model
failures.

## Objectives

- Implement byte-pair encoding (train, encode, decode) from scratch on raw bytes.
- Predict and demonstrate tokenizer pitfalls: whitespace sensitivity, digits, non-ASCII
  symbols (π⁰, √s_NN), and why "how many r's in strawberry" is hard.
- Scrape and clean a domain abstracts corpus from the arXiv API (default: nucl-ex,
  nucl-th, hep-ex; any other category is allowed — name it) and/or INSPIRE-HEP.
- Train a nanoGPT-class model end to end on that corpus on a free GPU (Colab/Kaggle),
  with checkpointing and a loss curve you can defend.
- Compare against a bigram baseline so "it learned something" is a number, not a vibe.

## Core material (~3 hrs)

- Karpathy, *Let's build the GPT Tokenizer* (video) with the `minbpe` repo as reference
  code — write yours before reading his.
- Karpathy's `nanoGPT` repo — read `train.py` for the training-loop practices (LR
  warmup + cosine decay, grad clipping, AdamW settings, checkpointing).
- GPT-2 paper §2.2 (byte-level BPE rationale).
- arXiv API documentation (bulk metadata / query interface) for the scrape.

## Exercises (built when the week starts)

1. BPE from scratch: `train(text, vocab_size)`, `encode`, `decode`. Accept when:
   `decode(encode(s)) == s` for every string in a test set including Unicode physics text.
2. Merge-order check vs `minbpe` on the same training text. Accept when: first 50 merges
   identical.
3. Tokenizer autopsy: tokenize "√s_NN = 200 GeV Au+Au", numerals, and code with your
   BPE and GPT-2's via `tiktoken`. Accept when: a table of token counts + two identified
   pitfalls, each stated in one line.
4. Corpus scrape: ≥ 5k heavy-ion abstracts, deduplicated, with a saved provenance file
   (query, date, count). Accept when: corpus file + provenance exist and a re-run
   reproduces the count within API drift.
5. Train nanoGPT (~10M params) on the corpus with your tokenizer. Accept when: val loss
   beats the bigram baseline by ≥ 0.5 nats and the loss curve (train+val) is saved.
6. Sample 10 abstracts at temperature 0.8. Accept when: samples saved and three concrete
   failure modes named (fabricated citations, wrong energies, incoherence, …).

## Deliverable

`bpe.py` + tests; corpus + provenance under `data/`; a trained checkpoint + loss curves
+ samples committed (or linked if large); `Week27_Exercises.ipynb` with checks PASS.

## Review

- Week 26: why must loss at init be ≈ ln(vocab_size), and what does your new vocab size
  make that number?
- Week 8: your training uses AdamW with warmup + cosine decay. Write Adam's update rule
  from memory; what do the two moment estimates track?
- Week 4: what does reproducibility require of this GPU run — list the Week 4 checklist
  items that apply (seeds, pinned deps, provenance, one-command run).
- Week 10: the bigram baseline is a giant conditional probability table. What is its
  maximum-likelihood estimate, per Week 7/10?
