# 00 — Syllabus

**Course title:** From Nuclear Physics to AI Scientist — a 12-Month Curriculum
**Duration:** 48 weeks (4 phases × 3 months), ~10 hrs/week, ~500 hrs total
**Student:** Parker Lewis (4th-year nuclear physics PhD, sPHENIX)
**Format:** Self-study in Jupyter notebooks. Each week: readings + derivations + a built
exercise notebook (per `NOTEBOOK_RULES.md`) + spaced-review questions. Each month ends
with a mini-project or capstone. Each phase ends with a shipped, tested capstone repo.

**Design goals** (from the tailoring interview, 2026-08-12):

- **Keep all doors open.** Balance three threads throughout: *theory & research skills*
  (AI research scientist path), *AI-for-science* (national-lab path, keeps nuclear
  physics ties), and *engineering* (applied AI/ML path).
- **No assumed knowledge.** Everything is (re)built from the ground up — including the
  math — but at physicist speed: review moves fast where your PhD already covers it.
- **Full derivations.** Backprop, attention, the VAE ELBO, diffusion, and policy
  gradients are derived on paper before they are typed into a notebook.
- **Teachable.** Notes and notebooks should be handable to the next physics student
  learning ML. Prose stays short; the *why* lives in teacher explainers.

---

## 1. Learning outcomes

By the end of 48 weeks you will be able to:

1. **Own the math.** Derive and use: SVD and PCA, MLE/MAP, KL divergence and entropy,
   gradient descent variants, backpropagation, self-attention, the VAE ELBO, DDPM
   diffusion, and the policy gradient theorem — on paper, without notes.
2. **Ship classical ML correctly.** Linear/logistic regression, trees, forests, boosted
   trees on tabular physics data with honest validation (nested CV, calibration,
   leakage checks).
3. **Build neural networks from scratch and in PyTorch.** Scalar autograd engine,
   MLPs, CNNs, GNNs; diagnose real training failures; use modern training practice
   (init, normalization, schedules, augmentation, transfer learning).
4. **Build and train transformers.** Implement a decoder-only transformer and a BPE
   tokenizer from scratch; train nanoGPT on a domain corpus; explain scaling laws and
   basic interpretability findings.
5. **Operate the modern LLM stack.** Fine-tune open-weight models with LoRA/QLoRA,
   design SFT datasets, explain RLHF/DPO, evaluate rigorously (including hallucination
   measurement), and build RAG systems with retrieval evals.
6. **Build agents.** Tool use, ReAct loops, MCP servers, multi-agent orchestration,
   and agent evaluation — culminating in a physics-analysis copilot.
7. **Engineer like it's production.** Testing, experiment tracking, GPU profiling,
   mixed precision, distributed-training concepts, quantized inference, deployment,
   and monitoring.
8. **Do research.** Read, critique, and reproduce ML papers (including HEP-ML:
   ParticleNet, Particle Transformer, Exa.TrkX, CaloChallenge); write clearly; propose
   and execute an original capstone.

## 2. Phase map

| Phase | Months | Theme | Capstone |
|-------|--------|-------|----------|
| 1 — Foundations | 01–03 | Scientific Python, software engineering, math for ML, classical ML | Tabular particle-ID classifier, full tested pipeline |
| 2 — Deep Learning | 04–06 | NNs from scratch, PyTorch, CNNs, sequences, GNNs, VAEs/flows | EMCal shower classifier **or** VAE fast-sim |
| 3 — Transformers & LLMs | 07–09 | Transformers from scratch, fine-tuning, alignment, RAG, diffusion | Physics-literature assistant **or** conditional diffusion calo generator |
| 4 — Agents, Systems, Research | 10–12 | Agents/MCP, ML engineering at scale, RL, research skills | Final capstone (choose a track) |

## 3. Weekly rhythm (~10 hrs)

| Activity | Hrs |
|----------|-----|
| Reading + videos | 3 |
| Paper-and-pencil derivations (where scheduled) | 1–2 |
| Exercise notebook (built per `NOTEBOOK_RULES.md` when the week starts) | 3–4 |
| Spaced review (retrieval questions from prior weeks) | 0.5 |
| Notes + reflection | 0.5 |

Mini-project/capstone weeks shift the balance toward building. If a week runs over,
let it — the calendar has slack (see §6). Never skip the review block.

## 4. Week-by-week plan

### Phase 1 — Foundations (Months 01–03)

**Month 01 — Scientific Python & Software Engineering**

- **Week 01 — Python + NumPy.** Python idioms for scientists; NumPy arrays, broadcasting,
  vectorization; leaving the ROOT/C++ event-loop mindset behind.
- **Week 02 — pandas + visualization.** DataFrames, groupby/merge, tidy data;
  matplotlib from first principles; making plots you'd put in a paper.
- **Week 03 — Software engineering I.** git fluency (branches, rebase, PRs), `uv`
  environments, `ruff`, `pytest`, project layout, debugging with `pdb`; why tests are
  the physicist's systematic-uncertainty analysis for code.
- **Week 04 — Reproducibility + mini-project.** Seeds, pinned deps, data provenance,
  one-command runs. Project: rebuild a small ROOT-style analysis (CERN Open Data
  dimuon spectrum) as a clean, tested Python package.

**Month 02 — Math for ML** (ground-up, physicist-fast)

- **Week 05 — Linear algebra I.** Vectors, matrices as linear maps, the four
  fundamental subspaces, rank, eigendecomposition; everything computed in NumPy
  alongside the theory.
- **Week 06 — Linear algebra II.** SVD derived and applied (compression, pseudoinverse,
  PCA derived from variance maximization *and* reconstruction error); matrix calculus
  conventions you'll need for backprop.
- **Week 07 — Probability & statistics.** Random variables, common distributions,
  Bayes' theorem, MLE and MAP (derive least squares from Gaussian likelihood),
  information theory: entropy, cross-entropy, KL divergence.
- **Week 08 — Optimization + mini-project.** Convexity, gradient descent, SGD,
  momentum, RMSProp, Adam — each derived and implemented from scratch. Project: fit
  physics models with hand-rolled optimizers; race them on pathological surfaces.

**Month 03 — Classical ML**

- **Week 09 — The ML frame + linear regression.** Supervised learning formally; loss,
  risk, generalization; bias–variance derived; train/val/test discipline;
  cross-validation; linear regression via normal equations *and* GD.
- **Week 10 — Classification.** Logistic regression with gradient derived by hand;
  softmax; decision thresholds; ROC/PR, confusion matrices, calibration; class
  imbalance (the HEP trigger analogy).
- **Week 11 — Trees & ensembles.** Decision trees, bagging, random forests, gradient
  boosting derived as functional GD; XGBoost/LightGBM; feature importance and its
  lies; why HEP loved BDTs.
- **Week 12 — Unsupervised + Capstone 1.** k-means, GMMs with EM derived, PCA in
  practice, UMAP. **Capstone 1:** tabular particle-ID classifier (MAGIC gamma/hadron
  or similar HEP open data) — full pipeline, nested CV, calibration, tests, writeup.

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
  physics tabular data, beating your Capstone-1 BDT or explaining why not.

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

**Month 06 — Deep Learning for Physics + Generative Models I**

- **Week 21 — Graph neural networks.** Message passing derived; permutation
  invariance; point clouds; ParticleNet and Particle Transformer; Exa.TrkX tracking.
- **Week 22 — Autoencoders & VAEs.** Latent variables; derive the ELBO line by line;
  reparameterization trick; posterior collapse; anomaly detection in HEP.
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
  evals (recall@k, faithfulness); build RAG over the sPHENIX TDR + your Zotero library.
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
  - (a) **HEP analysis copilot** — agent + MCP + RAG over your actual sPHENIX
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
deliberate slack for conference season, beam time, and thesis crunch. Slip weeks, not
content. If you must cut, cut from: Week 23 (flows), Week 28 (interpretability depth),
Week 43 (RL depth) — in that order, and note it in the tracker.

## 7. Spaced review

Every week's exercise notebook ends with 3–5 retrieval questions drawn from earlier
weeks (answers in the solutions notebook). Monthly: re-derive one flagship result cold
(rotating: PCA → backprop → ELBO → attention → DDPM loss → policy gradient).

## 8. Policies

- **Derivations are on paper first.** Scan or photograph them into the week folder.
  Typing LaTeX is optional; understanding is not.
- **Copyright.** Summarize and cite Bishop/Goodfellow/Murphy/Prince; no long verbatim
  passages in notes.
- **Honesty.** Negative results go in the writeup. This is not a LinkedIn post.
- **Tooling drift.** Pin versions in `pyproject.toml`; expect to read changelogs.
- **Hardware.** CPU + modest local GPU for most weeks. Colab/Kaggle/cloud for
  Weeks 20, 27, 32, 35–36, 47–48.
- **Notebooks.** Exercise/solution notebooks follow `NOTEBOOK_RULES.md` and are built
  when the week starts, not in advance. Lesson notes follow the `lesson-notes` format.

## 9. Sign-off protocol

After each month:
1. Tag the commit `month-NN-complete`.
2. Write a 250-word retrospective in the month folder (`retro.md`).
3. Open an issue for the single biggest thing you don't understand; close it next month.

---
*Syllabus rendered 2026-08-12. Sources in `01-Reading-List.md`; project specs in
`03-Project-Roadmap.md`; environment in `02-Setup-Guide.md`; progress in
`04-Progress-Tracker.md`. Previous 12-week course archived in `archive/2026-spring-12week/`.*
