# 01 — Reading List

Sources by phase — **all supplementary**: the primary teaching text for every week is
its `lesson.md` in this repo. **Spine** = the external thread that pairs best with the
lessons; read/watch in order. **Reference** = consult by chapter when a week points at
it. **Papers** = read critically, with the Week 45 workflow once you have it. Titles
only — search by name; URLs rot. Everything in the spine is free.

Books already in `references/`: Bishop, *Pattern Recognition and Machine Learning*;
Russell & Norvig, *AIMA* (background only, not scheduled).

## Phase 1 — Foundations (Months 01–03)

**Spine**
- Severance, *Python for Everybody* (free online book + videos) — Weeks 01–02;
  written for people who have never programmed.
- The official Python Tutorial, sections 1–5 — Weeks 01–02, second pass.
- VanderPlas, *Python Data Science Handbook* (free online), chs. 1–4 — Weeks 03–04.
- Pro Git book (free online), chs. 1–3 — Week 04.
- GitHub Actions "Understanding GitHub Actions" quickstart — Week 04 (the CI file
  is ~20 lines; the doc is so you know what it is doing).
- SQLite tutorial (official "SQLite as an Application File Format" + any SELECT/JOIN
  intro) — Week 04 taste, Week 23 in full.
- 3Blue1Brown, *Essence of Calculus* (all) — Week 05.
- 3Blue1Brown, *Essence of Linear Algebra* (all) — Weeks 06–07.
- Strang, MIT 18.06 lectures (select: subspaces, eigen, SVD) — Weeks 06–07.
- Parr & Howard, *The Matrix Calculus You Need For Deep Learning* — Week 07.
- StatQuest (Bayes, MLE, regularization, trees, boosting playlists) — Weeks 08–11.

**Remedial / catch-up (use as needed, not scheduled)**
- Khan Academy: Algebra II and Precalculus units — if Week 05's first sections feel
  fast, spend a weekend here first; it is normal and costs nothing but time.

**Reference**
- Bishop PRML chs. 1–4 (probability, linear models); Murphy, *Probabilistic Machine
  Learning: An Introduction* (free PDF) as the modern alternative.
- scikit-learn User Guide (cross-validation, calibration, ensembles).
- Boyd & Vandenberghe, *Convex Optimization* ch. 9 only (unconstrained minimization).

**Papers / domain culture**
- One applied-ML paper from *your* field (HEP BDT, medical tabular, etc.) so you
  know the culture you came from.

## Phase 2 — Deep Learning (Months 04–06)

**Spine**
- Karpathy, *Neural Networks: Zero to Hero* (micrograd → makemore) — Weeks 13–16.
- CS231n course notes (conv nets, training) — Weeks 16–18.
- Prince, *Understanding Deep Learning* (free PDF): MLP, training, CNN, GNN, VAE
  chapters as scheduled.
- FastAPI tutorial (first steps + SQL with sqlite) — Week 23.
- Docker getting-started (images vs containers, Dockerfile) — Week 23, reused Week 44.

**Reference**
- Goodfellow, Bengio & Courville, *Deep Learning* (free online) — optimization and
  regularization chapters.
- distill.pub: *A Gentle Introduction to Graph Neural Networks*.
- Lilian Weng: posts on VAEs and (optional) normalizing flows.
- PyTorch tutorials + PyTorch Geometric docs.
- SQLite SELECT/JOIN/GROUP BY docs — Week 23.

**Papers**
- He et al., *Deep Residual Learning* (ResNet).
- Qu & Gouskos, *ParticleNet* **or** Kipf & Welling, *Semi-Supervised Classification
  with Graph Convolutional Networks* — pick the one that matches your Week-21 data.
- Kingma & Welling, *Auto-Encoding Variational Bayes*.
- Dinh et al., *Density estimation using Real NVP* — only if you do `optional-flows.md`.
- *CaloChallenge 2022* summary — only if you take a calorimeter capstone track.

## Phase 3 — Transformers & LLMs (Months 07–09)

**Spine**
- Karpathy, *Let's build GPT from scratch* + *Let's build the GPT Tokenizer* —
  Weeks 25–27.
- 3Blue1Brown, transformer/attention chapters of the neural-net series — Week 25.
- Prince, *UDL* transformer and diffusion chapters — Weeks 26, 35.
- HuggingFace LLM course (transformers, datasets, peft chapters) — Weeks 29–30.
- OpenAI / Anthropic prompting guides (current docs, not a book) — Week 29. Treat
  them as recipes to *test*, not as gospel.

**Reference**
- Lilian Weng: *Attention? Attention!*, *What are Diffusion Models?*.
- HuggingFace docs: transformers, peft, trl, sentence-transformers.
- Anthropic transformer-circuits thread: *A Mathematical Framework for Transformer
  Circuits*, *In-context Learning and Induction Heads*, *Toy Models of
  Superposition* — Week 28, skim-depth.

**Papers**
- Vaswani et al., *Attention Is All You Need*.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Kaplan et al., *Scaling Laws for Neural Language Models*; Hoffmann et al.,
  *Training Compute-Optimal Large Language Models* (Chinchilla).
- Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*.
- Hu et al., *LoRA*; Dettmers et al., *QLoRA*.
- Ouyang et al., *Training language models to follow instructions* (InstructGPT).
- Rafailov et al., *Direct Preference Optimization*.
- Lewis et al., *Retrieval-Augmented Generation*.
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*; Ho & Salimans,
  *Classifier-Free Diffusion Guidance*.

## Phase 4 — Agents, Systems, Research (Months 10–12)

**Spine**
- Anthropic, *Building Effective Agents* essay + Claude API docs (Messages, tool
  use) — Weeks 37–38.
- Simon Willison's prompt-injection writing (any recent roundup) — Week 37.
- Model Context Protocol documentation + Python SDK — Week 39.
- Sutton & Barto, *Reinforcement Learning* (free online) chs. 3, 9, 13 + OpenAI
  *Spinning Up* (policy-gradient sections) — Week 43.
- Huyen, *Designing Machine Learning Systems* — Weeks 41, 44 (selected chapters).

**Reference**
- W&B / MLflow docs; PyTorch performance tuning guide; FastAPI docs (second pass).
- Keshav, *How to Read a Paper* — Week 45.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*.
- Schulman et al., *Proximal Policy Optimization Algorithms*.

**Papers (Week 45 critique pool — pick from your target role)**
- AI Scientist: a recent methods paper in your domain (HEP-ML, CaloChallenge,
  foundation-model-for-science, …).
- AI Engineer: a systems/evals paper (RAG eval, agent benchmarks, LLM-as-judge
  critique).

## Standing subscriptions (all year, ~30 min/week)

- arXiv in *your* field + `cs.LG` cross-listings skim.
- One LLM-news source of your choice (e.g. a weekly newsletter) — enough to know
  what changed, not enough to doomscroll.

## Year 2 (not scheduled; see `05-Two-Year-Path.md`)

- Engineer: Huyen in full; one cloud provider's "deploy a container" tutorial;
  OWASP LLM Top 10 as a checklist, not a course.
- Scientist: Murphy or Prince chapters you actually use; the papers behind your
  workshop/arXiv note.
