# 00 — Syllabus

**Course title:** From Zero to Modern AI — a 12-Month Ground-Up Curriculum
**Duration:** 48 weeks (4 phases × 3 months), ~10 hrs/week, ~500 hrs total
**Student:** a science grad student starting from **no coding background** and
only high-school math. (The resident student is Parker Lewis, 4th-year nuclear
physics PhD, sPHENIX — physics is the *worked example*, always explained, never
assumed. Any STEM reader can start at Week 01.)
**Outcomes this year supports:** using modern AI **in scientific work**, and
building the foundation for an **AI Engineer** or **AI Scientist** role over
the next 12–24 months (`05-Two-Year-Path.md`). This year does not confer those
titles by itself.
**Format:** Self-study. Each week folder contains the materials themselves:
`lesson.md` (the teaching text), `exercises.md` (the exercise page), and `project.md`
on project weeks. Exercise notebooks are generated from `exercises.md` when the week
starts (per `NOTEBOOK_RULES.md`).

**Design goals** (revised 2026-08-12):

- **Assume nothing.** Every concept — including "what is a program," "what is a
  derivative," "what is a matrix," "what is an HTTP request," "what is a prompt" —
  is taught from the ground up the first time it is used. A motivated beginner
  with no coding background can start at Week 01. A reader who already knows a
  week's material skims its lesson and goes straight to the exercises.
- **Materials included.** The teaching text, exercises, and project specs live in
  this repo. External readings (all free) supplement; they are never the only path.
- **Four threads, all year.** (1) *Software engineering* — git, tests, CI, SQL,
  HTTP/JSON, APIs, Docker, so you can ship. (2) *ML and deep learning* — classical
  models through transformers, with full derivations. (3) *Modern generative AI* —
  prompting, LLMs, RAG, agents, evals. (4) *Research habits* — read, reproduce,
  write, evaluate honestly.
- **Science and/or transition.** Every mini-project and capstone has a
  science-domain option (physics fully explained) and a general option, so the
  portfolio works for a lab AI hire *or* an industry AI Engineer / AI Scientist
  application.
- **Full derivations.** Backprop, attention, the VAE ELBO, diffusion, and policy
  gradients are derived on paper before they are typed into a notebook — and the
  lessons build up every piece of math those derivations need, starting from algebra.
- **Teachable.** Notes and notebooks should be handable to the next person learning
  this from scratch.

---

## 1. Learning outcomes

By the end of 48 weeks you will be able to:

1. **Program, and ship.** Go from never having written code to professional
   scientific Python: the language, NumPy, pandas, matplotlib, git, tests,
   environments, CI, logging, and one-command reproduction.
2. **Engineer data and services.** Query data with SQL, speak HTTP/JSON, wrap a
   model or a table in a small REST API, containerize it, and put tests in CI.
   Complexity (why some code is too slow) is in the vocabulary, not a CS degree.
3. **Own the math.** Starting from high-school algebra, build and then derive:
   derivatives and gradients, SVD and PCA, MLE/MAP, KL divergence and entropy,
   gradient descent variants, backpropagation, self-attention, the VAE ELBO, DDPM
   diffusion, and the policy gradient theorem — on paper, without notes.
4. **Ship classical ML correctly.** Linear/logistic regression, trees, forests,
   boosted trees on tabular data with honest validation (nested CV, calibration,
   leakage checks).
5. **Build neural networks from scratch and in PyTorch.** Scalar autograd engine,
   MLPs, CNNs, GNNs; diagnose real training failures; use modern training practice
   (init, normalization, schedules, augmentation, transfer learning).
6. **Build and train transformers.** Implement a decoder-only transformer and a BPE
   tokenizer from scratch; train nanoGPT on a domain corpus; explain scaling laws
   and basic interpretability findings.
7. **Prompt, fine-tune, and evaluate LLMs.** Treat prompts as versioned programs
   with evals; run open-weight models; fine-tune with LoRA/QLoRA; explain RLHF/DPO;
   measure hallucination; build RAG with retrieval evals.
8. **Build agents that are allowed to fail safely.** Tool use, ReAct loops, MCP
   servers, multi-agent orchestration, agent evaluation, and tool-layer defenses
   against prompt injection.
9. **Engineer like it's production.** Experiment tracking, GPU profiling, mixed
   precision, distributed-training concepts, quantized inference, deployment,
   monitoring, drift.
10. **Do research.** Read, critique, and reproduce papers; write clearly; propose
    and execute a capstone aimed at either an engineer-shaped system or a
    scientist-shaped result (`05-Two-Year-Path.md`).

---

## 2. Phase map

| Phase | Months | Theme | Capstone |
|-------|--------|-------|----------|
| 1 — Foundations | 01–03 | Programming from zero, SWE I, math for ML, classical ML | Tabular classifier, full tested pipeline |
| 2 — Deep Learning | 04–06 | NNs from scratch, PyTorch, CNNs, GNNs, VAEs, SWE II (SQL/APIs/Docker) | GNN classifier **or** VAE generator (science or general data) |
| 3 — Transformers & LLMs | 07–09 | Transformers from scratch, prompting, fine-tuning, RAG, diffusion | Domain RAG assistant **or** conditional generator |
| 4 — Agents, Systems, Research | 10–12 | Agents/MCP/security, ML engineering, RL, research skills | Final capstone (engineer track or scientist track) |

---

## 3. Weekly rhythm (~10 hrs)

| Activity | Hrs |
|----------|-----|
| `lesson.md` + external readings/videos | 3 |
| Paper-and-pencil derivations (where scheduled) | 1–2 |
| Exercise notebook (generated from `exercises.md` per `NOTEBOOK_RULES.md`) | 3–4 |
| Spaced review (retrieval questions at the end of `exercises.md`) | 0.5 |
| Notes + reflection | 0.5 |

Mini-project/capstone weeks shift the balance toward building. If a week runs over,
let it — the calendar has slack (see §6). Never skip the review block. If you already
know a week's material, prove it: do the exercises cold; skip the lesson, not the checks.

---

## 4. Week-by-week plan

### Phase 1 — Foundations (Months 01–03)

**Month 01 — Programming from Zero + Software Engineering I**

- **Week 01 — Your first programs.** What a computer actually does; what a program is;
  the terminal; installing Python; variables, numbers, strings; `print` and `input`;
  running scripts and notebooks; reading error messages without fear.
- **Week 02 — Control flow, functions, and data.** `if/else`, `for` and `while` loops,
  writing your own functions, lists and dictionaries, reading and writing files;
  your first real program built from parts.
- **Week 03 — The scientific stack.** Why scientists use arrays: NumPy from scratch
  (arrays, indexing, masks, broadcasting), first plots with matplotlib, first
  DataFrames with pandas.
- **Week 04 — Working like a professional + mini-project.** git and GitHub (including
  pull requests), tests with `pytest`, environments with `uv`, seeds, a first CI
  workflow, sqlite as a results table, and why some code is too slow (Big-O as
  vocabulary). Project: a real particle-physics analysis (CERN Open Data dimuon
  spectrum) as a clean, tested Python package — fully guided; the same skills apply
  to any CSV pipeline.

**Month 02 — Math for ML** (from high-school algebra up)

- **Week 05 — Calculus for ML.** Functions and their graphs; slope → derivative,
  built from first principles; partial derivatives and the gradient; the chain rule;
  finding minima; your first gradient descent — every idea computed in Python
  alongside the paper math.
- **Week 06 — Linear algebra I.** Vectors, matrices as linear maps, matrix
  multiplication from three viewpoints, solving systems, the four fundamental
  subspaces, rank — everything computed in NumPy alongside the theory.
- **Week 07 — Linear algebra II.** Eigenvalues and eigenvectors; SVD derived and
  applied (compression, pseudoinverse); PCA derived from variance maximization *and*
  reconstruction error; the matrix-calculus conventions you'll need for backprop.
- **Week 08 — Probability & statistics + mini-project.** Random variables, common
  distributions, Bayes' theorem, MLE and MAP (derive least squares from a Gaussian
  likelihood), entropy, cross-entropy, KL divergence. Project: optimizer race —
  derive and hand-roll GD, momentum, RMSProp, Adam; race them on pathological
  loss surfaces.

**Month 03 — Classical ML**

- **Week 09 — The ML frame + linear regression.** Supervised learning formally; loss,
  risk, generalization; bias–variance derived; train/val/test discipline;
  cross-validation; linear regression via normal equations *and* GD.
- **Week 10 — Classification.** Logistic regression with gradient derived by hand;
  softmax; decision thresholds; ROC/PR, confusion matrices, calibration; class
  imbalance (the rare-event problem — particle trigger *or* fraud/medical screening).
- **Week 11 — Trees & ensembles.** Decision trees, bagging, random forests, gradient
  boosting derived as functional GD; XGBoost/LightGBM; feature importance and its
  lies; why tabular data still loves BDTs.
- **Week 12 — Unsupervised + Capstone 1.** k-means, GMMs with EM derived, PCA in
  practice, UMAP. **Capstone 1:** tabular classifier with nested CV, calibration,
  tests, writeup — MAGIC gamma/hadron **or** any comparable rare-event tabular set.

### Phase 2 — Deep Learning (Months 04–06)

**Month 04 — Neural Networks from Scratch**

- **Week 13 — MLPs + backprop on paper.** Perceptron → MLP; forward pass as matrix
  products; universal approximation (and why it doesn't save you); derive backprop
  end-to-end for a 2-layer net including softmax/cross-entropy gradients.
- **Week 14 — micrograd.** Build a scalar autograd engine (Karpathy's micrograd,
  yours from scratch); computational graphs; check against finite differences.
- **Week 15 — PyTorch fundamentals.** Tensors, autograd, `nn.Module`, DataLoaders,
  training loops; rebuild Week 14's network in PyTorch; makemore-style char model.
- **Week 16 — Training dynamics + mini-project.** Initialization (Xavier/He derived),
  activation statistics, batchnorm/layernorm, dead ReLUs, LR schedules, gradient
  clipping; a "diagnose the broken training run" exercise set. Project: MLP vs your
  Capstone-1 BDT on the same tabular data.

**Month 05 — CNNs & Sequences**

- **Week 17 — Convolutions.** Convolution as parameter sharing; receptive fields;
  pooling; classic architectures (LeNet → ResNet, skip connections derived as
  gradient highways).
- **Week 18 — Modern vision training.** Augmentation, dropout, transfer learning;
  fine-tune a pretrained ResNet; when to freeze what; CNNs on images (detector
  images *or* a public vision set).
- **Week 19 — Sequences.** RNNs and LSTMs, backprop through time, why long-range
  credit assignment fails; attention as the fix (intuition-level; full derivation in
  Phase 3).
- **Week 20 — Mini-project.** Image classifier with a tabular-features baseline:
  EMCal photon vs merged-π⁰ **or** a public image task with the same protocol.

**Month 06 — Geometric DL, Generative Models, and Software for Services**

- **Week 21 — Graph neural networks.** Message passing derived; permutation
  invariance; point clouds; ParticleNet / citation graphs / molecules as the same
  idea on different data.
- **Week 22 — Autoencoders & VAEs.** Latent variables; derive the ELBO line by line;
  reparameterization trick; posterior collapse; anomaly detection. (Normalizing
  flows: survey + optional full lesson in the Week 23 folder, `optional-flows.md`.)
- **Week 23 — Software engineering II: data and services.** SQL from zero, HTTP/JSON
  and REST, a small FastAPI service over a sqlite table, Docker, logging and env
  vars. This is the week that makes Week 44 a spiral instead of a first contact.
- **Week 24 — Capstone 2.** Choose: (a) GNN classifier vs your CNN, science or
  public graph data, or (b) VAE generator with distributional validation (calorimeter
  showers *or* a public image set). Tested repo + writeup.

### Phase 3 — Transformers & LLMs (Months 07–09)

**Month 07 — Transformers from Scratch**

- **Week 25 — Attention derived.** Attention as differentiable content-based lookup;
  queries/keys/values; scaled dot-product (derive the √d); multi-head; causal masking.
- **Week 26 — The full transformer.** Token + position embeddings, residual stream,
  pre-norm vs post-norm, MLP blocks, weight tying; read *Attention Is All You Need*
  and the GPT-2 paper critically.
- **Week 27 — Tokenization + nanoGPT.** Build BPE from scratch; tokenizer pitfalls;
  train a nanoGPT-class model on a domain corpus you scrape (heavy-ion abstracts
  *or* another field's arXiv category). GPU week.
- **Week 28 — Scaling & interpretability.** Scaling laws (Kaplan, Chinchilla),
  emergence debates, induction heads, superposition; what interpretability can and
  can't tell you.

**Month 08 — Prompting and the Modern LLM Stack**

- **Week 29 — Prompting + HuggingFace.** Load and run open-weight models;
  sampling (temperature, top-p) derived; chat templates; **prompt engineering as
  a tested program** (instructions, few-shot, failure modes, a small eval harness).
- **Week 30 — Fine-tuning.** Full FT vs parameter-efficient; derive the LoRA
  low-rank update; QLoRA and quantization basics; `peft`; SFT dataset design.
- **Week 31 — Alignment.** Instruction tuning → RLHF pipeline; reward models; PPO at
  sketch level; DPO derived from the RLHF objective; safety and refusal behavior.
- **Week 32 — Evaluation + mini-project.** Benchmarks and their contamination;
  LLM-as-judge; measuring hallucination. Project: LoRA-fine-tune a 1–3B model to
  extract structured fields from domain text (physics abstracts *or* another
  schema you define); report vs zero-shot *and* vs your best prompt.

**Month 09 — Retrieval & Generative Models II**

- **Week 33 — Embeddings.** Contrastive training, sentence-transformers, similarity
  metrics, chunking strategies; build and probe an embedding index.
- **Week 34 — RAG.** Retrieval, hybrid search, reranking, answer synthesis; retrieval
  evals (recall@k, faithfulness); build RAG over a document library you assemble.
- **Week 35 — Diffusion models.** Full DDPM derivation (forward noising, reverse
  process, the simplified loss); classifier-free guidance; conditional generation;
  diffusion vs flows vs VAEs (flows at survey depth unless you did `optional-flows.md`).
- **Week 36 — Capstone 3.** Choose: (a) domain-literature assistant (RAG + extractor),
  or (b) conditional diffusion generator vs your Capstone-2 VAE. Science or general
  data; same eval discipline.

### Phase 4 — Agents, Systems, Research (Months 10–12)

**Month 10 — Agents & Tools**

- **Week 37 — Tool use from first principles.** Recap HTTP/JSON from Week 23; the
  Claude API; JSON-schema tool specs; structured output; the agentic loop by hand;
  **prompt injection and tool-layer authorization** (never trust the model's text).
- **Week 38 — Agent patterns.** ReAct; Anthropic's *Building Effective Agents*
  patterns (chaining, routing, parallelization, orchestrator–workers,
  evaluator–optimizer); when agents are the wrong answer.
- **Week 39 — MCP.** The protocol; build an MCP server exposing tools you own
  (science analysis functions *or* general data tools); consume it from a client.
- **Week 40 — Multi-agent + agent evals + mini-project.** Orchestration, handoffs,
  agent benchmarks. Project: a copilot driving your Week-39 tools, with an eval
  table (success rate, cost, trajectory review).

**Month 11 — ML Engineering & RL**

- **Week 41 — Experiment infrastructure.** Tracking (W&B/MLflow), config management,
  sweeps, model registries; retrofit tracking onto a Phase-2/3 project.
- **Week 42 — Performance engineering.** GPU architecture basics, profiling, mixed
  precision, `torch.compile`, DDP concepts, inference optimization (quantization,
  KV cache, batching).
- **Week 43 — Reinforcement learning.** MDPs, returns, value functions; derive the
  policy gradient theorem; REINFORCE → actor-critic → PPO; the RLHF connection;
  RL for control (accelerator/detector *or* a standard control task).
- **Week 44 — Deployment + mini-project.** Spiral from Week 23: model serving
  (FastAPI), containers, monitoring, drift, CI that builds the image; ship one of
  your models as a service with tests and a health check.

**Month 12 — Research Skills & Final Capstone**

- **Week 45 — Reading & reproducing.** A paper-reading workflow; critique one paper
  in writing (HEP-ML *or* a paper in your target role's literature); reproduce a
  small published result from scratch.
- **Week 46 — Writing + proposal.** Writing clearly; draft the capstone proposal
  with an explicit role sentence: AI Engineer or AI Scientist (`05-Two-Year-Path.md`).
- **Weeks 47–48 — Final capstone.** Choose ONE track (see `03-Project-Roadmap.md`):
  engineer-shaped (copilot/service) or scientist-shaped (generative result or
  reproduce-and-extend). Build, validate, write up, demo, mock talk.

---

## 5. Gates (self-assessment; no grader)

| Gate | Requirement |
|------|-------------|
| End Phase 1 | Capstone-1 repo on GitHub: `pytest -q` green, CI green, nested-CV results, calibration plot, writeup incl. what failed first |
| End Phase 2 | Capstone-2 repo: model beats stated baseline or writeup explains why not; backprop + ELBO derivations re-done cold; Week-23 service tests green |
| End Phase 3 | Fine-tuned checkpoint + RAG or generator callable from one script; benchmark table vs the stated baseline (including a prompt-only baseline where relevant); attention derivation re-done cold |
| End Phase 4 | Final capstone demoed end-to-end; architecture diagram; honest limitations; 1 mock talk delivered (record it); `PROPOSAL.md` role sentence still true |

"Done" means reproducible: fresh clone + `uv sync` + one command → same result within
stochastic noise.

---

## 6. Calendar policy

48 content-weeks ≈ 11 months at one week per calendar week. The spare month is
deliberate slack for life, work, conference season, beam time, and thesis crunch.
A true beginner should also expect Months 01–02 to run hot — budget extra hours there
before touching the slack. Slip weeks, not content. If you must cut, cut from:
`optional-flows.md` (already optional), Week 28 (interpretability depth), Week 43
(RL depth) — in that order, and note it in the tracker. **Do not cut Weeks 04, 23,
29, or 37** — those are the SWE / prompting / agent-safety spine.

---

## 7. Spaced review

Every week's `exercises.md` ends with 3–5 retrieval questions drawn from earlier
weeks (answers in the solutions notebook). Monthly: re-derive one flagship result cold
(rotating: PCA → backprop → ELBO → attention → DDPM loss → policy gradient).

---

## 8. Policies

- **Derivations are on paper first.** Scan or photograph them into the week folder.
  Typing LaTeX is optional; understanding is not.
- **No assumed knowledge, ever.** If a lesson uses a term it hasn't defined, that is a
  bug — file it and fix it. Physics examples come with their own explanations.
- **Copyright.** Lessons are original text. External books (Bishop, Goodfellow,
  Murphy, Prince) are summarized and cited; no long verbatim passages.
- **Honesty.** Negative results go in the writeup. This is not a LinkedIn post.
- **Tooling drift.** Pin versions in `pyproject.toml`; expect to read changelogs.
- **Hardware.** Any laptop works through Week 14. CPU + modest local GPU for most
  weeks after. Colab/Kaggle/cloud for Weeks 20, 27, 32, 35–36, 47–48.
- **Notebooks.** Exercise/solution notebooks follow `NOTEBOOK_RULES.md` and are
  generated from `exercises.md` when the week starts, not in advance — libraries and
  models drift too fast. Lesson notes follow the `lesson-notes` format.

---

## 9. Sign-off protocol

After each month:
1. Tag the commit `month-NN-complete`.
2. Write a 250-word retrospective in the month folder (`retro.md`).
3. Open an issue for the single biggest thing you don't understand; close it next month.

---
*Syllabus revised 2026-08-12 (SWE + prompting as first-class threads; dual science /
career-transition tracks; year-1-of-2 framing). Sources in `01-Reading-List.md`;
project specs in `03-Project-Roadmap.md`; two-year path in `05-Two-Year-Path.md`;
environment in `02-Setup-Guide.md`; progress in `04-Progress-Tracker.md`. Previous
courses archived in `archive/`.*
