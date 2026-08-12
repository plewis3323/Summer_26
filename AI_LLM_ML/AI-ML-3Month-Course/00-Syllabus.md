# 00 — Syllabus

**Course title:** From Zero to AI Scientist — a 12-Month Ground-Up Curriculum
**Duration:** 48 weeks (4 phases × 3 months), ~10 hrs/week, ~500 hrs total
**Student:** anyone starting from zero — written for a reader with **no coding
background** and only high-school math. (The resident student is Parker Lewis,
4th-year nuclear physics PhD, sPHENIX — physics shows up everywhere as the
*application*, but it is always explained, never assumed.)
**Format:** Self-study. Each week folder contains the materials themselves:
`lesson.md` (the teaching text), `exercises.md` (the exercise page), and `project.md`
on project weeks. Exercise notebooks are generated from `exercises.md` when the week
starts (per `NOTEBOOK_RULES.md`).

**Design goals** (rebuilt 2026-08-12):

- **Assume nothing.** Every concept — including "what is a program," "what is a
  derivative," "what is a matrix" — is taught from the ground up the first time it is
  used. A motivated beginner with no coding background can start at Week 01 and finish
  as a working AI scientist. A reader who already knows a week's material skims its
  lesson and goes straight to the exercises.
- **Materials included.** The teaching text, exercises, and project specs live in this
  repo. External readings (all free) supplement; they are never the only path.
- **Keep all doors open.** Three threads run throughout: *theory & research skills*
  (AI research scientist path), *AI-for-science* (national-lab path, keeps the nuclear
  physics ties), and *engineering* (applied AI/ML path).
- **Full derivations.** Backprop, attention, the VAE ELBO, diffusion, and policy
  gradients are derived on paper before they are typed into a notebook — and the
  lessons build up every piece of math those derivations need, starting from algebra.
- **Teachable.** Notes and notebooks should be handable to the next person learning
  ML from scratch. Prose stays short; the *why* lives in the lessons and teacher
  explainers.

---

## 1. Learning outcomes

By the end of 48 weeks you will be able to:

1. **Program.** Go from never having written code to fluent scientific Python:
   the language itself, NumPy, pandas, matplotlib, git, tests, and environments.
2. **Own the math.** Starting from high-school algebra, build and then derive:
   derivatives and gradients, SVD and PCA, MLE/MAP, KL divergence and entropy,
   gradient descent variants, backpropagation, self-attention, the VAE ELBO, DDPM
   diffusion, and the policy gradient theorem — on paper, without notes.
3. **Ship classical ML correctly.** Linear/logistic regression, trees, forests, boosted
   trees on tabular data with honest validation (nested CV, calibration, leakage checks).
4. **Build neural networks from scratch and in PyTorch.** Scalar autograd engine,
   MLPs, CNNs, GNNs; diagnose real training failures; use modern training practice
   (init, normalization, schedules, augmentation, transfer learning).
5. **Build and train transformers.** Implement a decoder-only transformer and a BPE
   tokenizer from scratch; train nanoGPT on a domain corpus; explain scaling laws and
   basic interpretability findings.
6. **Operate the modern LLM stack.** Fine-tune open-weight models with LoRA/QLoRA,
   design SFT datasets, explain RLHF/DPO, evaluate rigorously (including hallucination
   measurement), and build RAG systems with retrieval evals.
7. **Build agents.** Tool use, ReAct loops, MCP servers, multi-agent orchestration,
   and agent evaluation — culminating in a physics-analysis copilot.
8. **Engineer like it's production.** Testing, experiment tracking, GPU profiling,
   mixed precision, distributed-training concepts, quantized inference, deployment,
   and monitoring.
9. **Do research.** Read, critique, and reproduce ML papers (including HEP-ML:
   ParticleNet, Particle Transformer, Exa.TrkX, CaloChallenge); write clearly; propose
   and execute an original capstone.

## 2. Phase map

| Phase | Months | Theme | Capstone |
|-------|--------|-------|----------|
| 1 — Foundations | 01–03 | Programming from zero, the scientific stack, math for ML from algebra up, classical ML | Tabular particle-ID classifier, full tested pipeline |
| 2 — Deep Learning | 04–06 | NNs from scratch, PyTorch, CNNs, sequences, GNNs, VAEs/flows | EMCal shower classifier **or** VAE fast-sim |
| 3 — Transformers & LLMs | 07–09 | Transformers from scratch, fine-tuning, alignment, RAG, diffusion | Physics-literature assistant **or** conditional diffusion calo generator |
| 4 — Agents, Systems, Research | 10–12 | Agents/MCP, ML engineering at scale, RL, research skills | Final capstone (choose a track) |

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

## 4. Week-by-week plan

### Phase 1 — Foundations (Months 01–03)

**Month 01 — Programming from Zero to the Scientific Stack**

- **Week 01 — Your first programs.** What a computer actually does; what a program is;
  the terminal; installing Python; variables, numbers, strings; `print` and `input`;
  running scripts and notebooks; reading error messages without fear.
- **Week 02 — Control flow, functions, and data.** `if/else`, `for` and `while` loops,
  writing your own functions, lists and dictionaries, reading and writing files;
  your first real program built from parts.
- **Week 03 — The scientific stack.** Why scientists use arrays: NumPy from scratch
  (arrays, indexing, masks, broadcasting), first plots with matplotlib, first
  DataFrames with pandas.
- **Week 04 — Working like a professional + mini-project.** git and GitHub from zero,
  tests with `pytest`, environments with `uv`, seeds and reproducibility, one-command
  runs. Project: a real particle-physics analysis (CERN Open Data dimuon spectrum)
  as a clean, tested Python package — fully guided.

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
  imbalance (the rare-event problem, with a particle-trigger story).
- **Week 11 — Trees & ensembles.** Decision trees, bagging, random forests, gradient
  boosting derived as functional GD; XGBoost/LightGBM; feature importance and its
  lies; why particle physics loved BDTs.
- **Week 12 — Unsupervised + Capstone 1.** k-means, GMMs with EM derived, PCA in
  practice, UMAP. **Capstone 1:** tabular particle-ID classifier (MAGIC gamma/hadron
  open data) — full pipeline, nested CV, calibration, tests, writeup.

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
  clipping; a "diagnose the broken training run" exercise set. Project: MLP on
  tabular physics data, beating your Capstone-1 BDT or explaining why not.

**Month 05 — CNNs & Sequences**

- **Week 17 — Convolutions.** Convolution as parameter sharing; receptive fields;
  pooling; classic architectures (LeNet → ResNet, skip connections derived as
  gradient highways).
- **Week 18 — Modern vision training.** Augmentation, dropout, transfer learning;
  fine-tune a pretrained ResNet; when to freeze what; CNNs on detector images.
- **Week 19 — Sequences.** RNNs and LSTMs, backprop through time, why long-range
  credit assignment fails; attention as the fix (intuition-level; full derivation in
  Phase 3).
- **Week 20 — Mini-project.** EMCal cluster classifier: single-photon vs merged-π⁰
  on simulated calorimeter images; compare CNN vs your Phase-1 tabular approach.
  (The lesson explains what a calorimeter is and why the two look alike.)

**Month 06 — Deep Learning for Physics + Generative Models I**

- **Week 21 — Graph neural networks.** Message passing derived; permutation
  invariance; point clouds; ParticleNet and Particle Transformer; Exa.TrkX tracking.
- **Week 22 — Autoencoders & VAEs.** Latent variables; derive the ELBO line by line;
  reparameterization trick; posterior collapse; anomaly detection in physics.
- **Week 23 — Normalizing flows + fast simulation.** Change of variables, coupling
  layers; the CaloChallenge and why colliders need learned simulators.
- **Week 24 — Capstone 2.** Choose: (a) GNN jet/cluster classifier benchmarked
  against your CNN, or (b) VAE fast-sim of EMCal showers with physics validation
  (energy response, shower shapes). Tested repo + writeup.

### Phase 3 — Transformers & LLMs (Months 07–09)

**Month 07 — Transformers from Scratch**

- **Week 25 — Attention derived.** Attention as differentiable content-based lookup;
  queries/keys/values; scaled dot-product (derive the √d); multi-head; causal masking.
- **Week 26 — The full transformer.** Token + position embeddings, residual stream,
  pre-norm vs post-norm, MLP blocks, weight tying; read *Attention Is All You Need*
  and the GPT-2 paper critically.
- **Week 27 — Tokenization + nanoGPT.** Build BPE from scratch; tokenizer pitfalls;
  train a nanoGPT-class model on a scraped heavy-ion abstracts corpus (GPU week).
- **Week 28 — Scaling & interpretability.** Scaling laws (Kaplan, Chinchilla),
  emergence debates, induction heads, superposition; what interpretability can and
  can't tell you.

**Month 08 — The Modern LLM Stack**

- **Week 29 — HuggingFace ecosystem.** `transformers`, `datasets`; loading and
  running open-weight models; sampling strategies (temperature, top-p) derived;
  chat templates and their footguns.
- **Week 30 — Fine-tuning.** Full FT vs parameter-efficient; derive the LoRA
  low-rank update; QLoRA and quantization basics; `peft`; SFT dataset design.
- **Week 31 — Alignment.** Instruction tuning → RLHF pipeline; reward models; PPO at
  sketch level; DPO derived from the RLHF objective; safety and refusal behavior.
- **Week 32 — Evaluation + mini-project.** Benchmarks and their contamination;
  LLM-as-judge; measuring hallucination. Project: LoRA-fine-tune a 1–3B model to
  extract structured metadata from nuclear physics abstracts; report vs zero-shot.

**Month 09 — Retrieval & Generative Models II**

- **Week 33 — Embeddings.** Contrastive training, sentence-transformers, similarity
  metrics, chunking strategies; build and probe an embedding index.
- **Week 34 — RAG.** Retrieval, hybrid search, reranking, answer synthesis; retrieval
  evals (recall@k, faithfulness); build RAG over a physics document library.
- **Week 35 — Diffusion models.** Full DDPM derivation (forward noising, reverse
  process, the simplified loss); classifier-free guidance; conditional generation;
  diffusion vs flows vs VAEs for calorimeters.
- **Week 36 — Capstone 3.** Choose: (a) physics-literature assistant (RAG + your
  fine-tuned extractor, evaluated against a held-out Q&A set), or (b) conditional
  diffusion calo-shower generator with physics validation vs your Capstone-2 VAE.

### Phase 4 — Agents, Systems, Research (Months 10–12)

**Month 10 — Agents & Tools**

- **Week 37 — Tool use from first principles.** The Claude API; JSON-schema tool
  specs; structured output; single tool calls → multi-tool loops with error recovery.
- **Week 38 — Agent patterns.** ReAct; Anthropic's *Building Effective Agents*
  patterns (chaining, routing, parallelization, orchestrator–workers,
  evaluator–optimizer); when agents are the wrong answer.
- **Week 39 — MCP.** The protocol; build an MCP server exposing physics tools
  (`fit_pi0_peak`, `list_run_files`, `get_calibration`, …); consume it from a client.
- **Week 40 — Multi-agent + agent evals + mini-project.** Orchestration, handoffs,
  agent benchmarks. Project: prototype analysis-copilot agent driving your MCP tools.

**Month 11 — ML Engineering & RL**

- **Week 41 — Experiment infrastructure.** Tracking (W&B/MLflow), config management,
  sweeps, model registries; retrofit tracking onto a Phase-2/3 project.
- **Week 42 — Performance engineering.** GPU architecture basics, profiling, mixed
  precision, `torch.compile`, DDP concepts, inference optimization (quantization,
  KV cache, batching).
- **Week 43 — Reinforcement learning.** MDPs, returns, value functions; derive the
  policy gradient theorem; REINFORCE → actor-critic → PPO; the RLHF connection;
  RL for accelerator/detector control.
- **Week 44 — Deployment + mini-project.** Model serving (FastAPI), containers,
  monitoring, drift; ship one of your models as a service with tests and a health
  check.

**Month 12 — Research Skills & Final Capstone**

- **Week 45 — Reading & reproducing.** A paper-reading workflow; critique one HEP-ML
  paper in writing; reproduce a small published result from scratch.
- **Week 46 — Research writing + proposal.** Writing clearly for ML venues; draft
  your capstone proposal (problem, baselines, evaluation plan, risks).
- **Weeks 47–48 — Final capstone.** Choose ONE track:
  - (a) **HEP analysis copilot** — agent + MCP + RAG over a real physics
    workflow, end-to-end demo.
  - (b) **Generative fast-sim** — diffusion/flow calo generator with rigorous
    physics validation, aimed at a CaloChallenge-style writeup.
  - (c) **Paper-to-pipeline agent** — reads a heavy-ion paper, scaffolds an analysis.
  - (d) **Reproduce-and-extend** — take a recent HEP-ML paper, reproduce it, add one
    novel twist; aim it at an arXiv note or workshop submission.

## 5. Gates (self-assessment; no grader)

| Gate | Requirement |
|------|-------------|
| End Phase 1 | Capstone-1 repo on GitHub: `pytest -q` green, nested-CV results, calibration plot, writeup incl. what failed first |
| End Phase 2 | Capstone-2 repo: model beats stated baseline or writeup explains why not; backprop + ELBO derivations re-done cold |
| End Phase 3 | Fine-tuned checkpoint + RAG system callable from one script; benchmark table vs zero-shot; attention derivation re-done cold |
| End Phase 4 | Final capstone demoed end-to-end; architecture diagram; honest limitations; 1 mock research talk delivered (record it) |

"Done" means reproducible: fresh clone + `uv sync` + one command → same result within
stochastic noise.

## 6. Calendar policy

48 content-weeks ≈ 11 months at one week per calendar week. The spare month is
deliberate slack for life, work, conference season, beam time, and thesis crunch.
A true beginner should also expect Months 01–02 to run hot — budget extra hours there
before touching the slack. Slip weeks, not content. If you must cut, cut from:
Week 23 (flows), Week 28 (interpretability depth), Week 43 (RL depth) — in that
order, and note it in the tracker.

## 7. Spaced review

Every week's `exercises.md` ends with 3–5 retrieval questions drawn from earlier
weeks (answers in the solutions notebook). Monthly: re-derive one flagship result cold
(rotating: PCA → backprop → ELBO → attention → DDPM loss → policy gradient).

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

## 9. Sign-off protocol

After each month:
1. Tag the commit `month-NN-complete`.
2. Write a 250-word retrospective in the month folder (`retro.md`).
3. Open an issue for the single biggest thing you don't understand; close it next month.

---
*Syllabus rebuilt 2026-08-12 (ground-up edition: no assumed knowledge, materials
included). Sources in `01-Reading-List.md`; project specs in `03-Project-Roadmap.md`;
environment in `02-Setup-Guide.md`; progress in `04-Progress-Tracker.md`. Previous
courses archived in `archive/`.*
