# 00 — Syllabus

**Course title:** Applied AI/ML for a Heavy-Ion Physicist
**Duration:** 12 weeks, ~15–20 hrs/week
**Student:** Parker Lewis (sPHENIX / STAR, advisor J. Frantz)
**Format:** Self-study. Reading + exercises + one mini-project per week; monthly capstone; final capstone on a chosen track.

## 1. Learning outcomes

By the end of 12 weeks, you will be able to:

1. **Think in the ML frame.** Distinguish supervised, unsupervised, and self-supervised tasks; identify label leakage and covariate shift; reason about bias–variance and capacity tradeoffs without memorizing a decision chart.
2. **Ship classical models.** Train, evaluate, and ship logistic regression, random forests, and gradient-boosted decision trees on tabular HEP-flavored data. Interpret ROC/PR curves, confusion matrices, calibration plots. Use cross-validation and nested CV correctly.
3. **Build neural networks in PyTorch from first principles.** Derive cross-entropy and softmax gradients on paper; implement a scalar autograd engine (micrograd); build MLPs and CNNs; diagnose training failures (dead ReLUs, vanishing gradients, exploding loss, learning-rate sensitivity).
4. **Understand and train transformers.** Derive self-attention as a differentiable content-based lookup; implement a nanoGPT-style decoder; train it on a domain corpus; understand why scaling works and where it breaks.
5. **Fine-tune modern open-weight LLMs.** Apply LoRA/QLoRA with HuggingFace `peft` on a 1–3B model. Design prompts, build SFT datasets, evaluate with a held-out set, quantify hallucination.
6. **Build retrieval-augmented systems.** Chunk documents, embed with sentence-transformers, store in `chroma` or `lancedb`, retrieve and rerank, integrate into an LLM answer loop.
7. **Build tool-using agents.** Implement function calling, the ReAct loop, multi-step planning, and retries with guardrails. Ship an agent that drives analysis tools.
8. **Speak MCP.** Expose Python functions as MCP tools and consume them from an MCP-aware client.
9. **Hold your own at HEP-ML talks.** Read and critique ParticleNet, Particle Transformer, Exa.TrkX, and CaloChallenge papers.

## 2. Grading rubric (for self-assessment)

There is no grader. But you should hit these gates before moving on:

| Gate | Requirement |
|------|-------------|
| End of Month 1 | Shower-classifier capstone is on GitHub with a README, a test suite that runs `pytest -q`, ROC curve checked in, and a short writeup of what failed first |
| End of Month 2 | A LoRA-fine-tuned small model + a RAG system, both callable from a single Python script; benchmark numbers vs a zero-shot baseline in the writeup |
| End of Month 3 | Final capstone demoed end-to-end; architecture diagram; honest limitations section |

"Done" means reproducible: fresh clone + `uv sync` + one command → same result within stochastic noise.

## 3. Weekly time budget (target)

| Activity | Hrs/week |
|----------|----------|
| Reading + videos | 4–6 |
| Coded exercises | 5–7 |
| Mini-project | 4–6 |
| Notes + reflection | 1 |
| **Total** | **14–20** |

## 4. Week-by-week plan

### Month 1 — Foundations + first ML model

- **Week 01 — Python scientific stack & reproducibility.** NumPy broadcasting, pandas idioms, matplotlib vs the ROOT plotting mindset, `uv`, `ruff`, `pytest`, git hygiene, random-seed discipline.
- **Week 02 — Classical ML.** Linear and logistic regression (with derivations), regularization geometry, decision trees, random forests, gradient boosting, XGBoost vs LightGBM. HEP BDT culture.
- **Week 03 — PyTorch + MLPs + micrograd.** Autograd from scratch; computational graphs; the universal approximation theorem and why it doesn't save you; optimizers; Karpathy's makemore.
- **Week 04 — CNNs and Month-1 capstone.** Convolution as parameter sharing; receptive fields; pooling; batchnorm; dropout; augmentation. Capstone: single-photon vs merged-π⁰ EMCal cluster classifier on simulated images.

### Month 2 — Deep learning, transformers, small LLMs

- **Week 05 — Sequences, attention intuition, GNNs for HEP.** RNNs → why they failed → attention. Message passing on graphs. ParticleNet and Particle Transformer. Exa.TrkX tracking.
- **Week 06 — Build a transformer from scratch.** Token/position embeddings; multi-head self-attention; residual streams; pre-norm vs post-norm; LayerNorm; weight tying. Train nanoGPT on a scraped heavy-ion abstracts corpus.
- **Week 07 — HuggingFace + LoRA/QLoRA.** `transformers`, `datasets`, `peft`, `trl`. Prompt templates, chat formats, tokenizer pitfalls. Fine-tune a 1–3B model (Llama 3.2 / Gemma 2B / Phi-3 mini) to extract structured metadata from abstracts.
- **Week 08 — RAG.** Embeddings, chunking, vector DBs (`chroma`, `lancedb`), hybrid retrieval, reranking. Build a RAG system over your Zotero library + the sPHENIX TDR.

### Month 3 — Agents + capstone

- **Week 09 — Tool use & function calling.** JSON-schema tool specs; single-shot tool calls; multi-tool loops; error recovery.
- **Week 10 — Agent frameworks.** LangGraph, smolagents; ReAct; Anthropic's "Building Effective Agents" patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer).
- **Week 11 — Model Context Protocol.** Spec, transport, servers. Build an MCP server exposing `fit_pi0_peak`, `list_run_files`, `get_calibration`, `generate_fun4all_macro`.
- **Week 12 — Capstone.** Pick ONE of:
  - (a) HEP analysis copilot agent
  - (b) conditional diffusion/VAE for fast EMCal shower generation
  - (c) paper-to-pipeline agent that reads a heavy-ion paper and scaffolds a Fun4All analysis module

## 5. Assessment artifacts

By end of course you should own:

- 12 Week-project repos (or a monorepo with 12 subprojects).
- A fine-tuned model checkpoint on HuggingFace Hub (private is fine).
- A working MCP server in a repo.
- A final capstone with an architecture diagram, a writeup, and — ideally — a short recorded demo.
- A set of notes you can hand to next year's physics-student-learning-ML.

## 6. Policies

- **Copyright.** You will read from Bishop, Goodfellow/Bengio/Courville, Murphy, and various papers. Do not paste large verbatim passages into your notes or repo. Summarize and cite.
- **Honesty.** If a method didn't work, say so in the writeup. Negative results have signal. This is not a LinkedIn post.
- **Tooling drift.** Libraries change. Pin versions in `pyproject.toml`. If a tutorial uses `transformers==4.38` and the current version is `4.52`, expect to read the changelog.
- **Hardware.** Assume CPU + modest local GPU for most weeks. Move to Colab/Kaggle only for Week 04 training at scale, Week 06 nanoGPT training, and Week 07 QLoRA fine-tuning.

## 7. Sign-off protocol

After each month's capstone:
1. Tag the commit `month-N-complete`.
2. Write a 250-word retrospective in `Month#-DeepLearning-LLMs/retro.md` (or equivalent).
3. Open an issue for the single biggest thing you don't understand. Try to close it during the next month.

---
*Syllabus rendered on 2026-04-18. See `01-Reading-List.md` for source material and `03-Project-Roadmap.md` for project specs.*
