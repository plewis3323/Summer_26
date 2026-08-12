# 01 — Reading List

Sources by phase. **Spine** = the primary thread for the phase; read/watch in order.
**Reference** = consult by chapter when a week points at it. **Papers** = read
critically, with the Week 45 workflow once you have it. Titles only — search by name;
URLs rot.

Books already in `references/`: Bishop, *Pattern Recognition and Machine Learning*;
Russell & Norvig, *AIMA* (background only, not scheduled).

## Phase 1 — Foundations (Months 01–03)

**Spine**
- VanderPlas, *Python Data Science Handbook* (free online) — Weeks 01–02.
- Pro Git book (free online), chs. 1–3 + branching — Week 03.
- 3Blue1Brown, *Essence of Linear Algebra* (all) — Weeks 05–06.
- Strang, MIT 18.06 lectures (select: subspaces, eigen, SVD) — Weeks 05–06.
- Parr & Howard, *The Matrix Calculus You Need For Deep Learning* — Week 06.
- StatQuest (Bayes, MLE, regularization, trees, boosting playlists) — Weeks 07–11.

**Reference**
- Bishop PRML chs. 1–4 (probability, linear models); Murphy, *Probabilistic Machine
  Learning: An Introduction* (free PDF) as the modern alternative.
- scikit-learn User Guide (cross-validation, calibration, ensembles).
- Boyd & Vandenberghe, *Convex Optimization* ch. 9 only (unconstrained minimization).

**Papers / HEP culture**
- A TMVA-era HEP BDT paper of your choosing (know the culture you came from).

## Phase 2 — Deep Learning (Months 04–06)

**Spine**
- Karpathy, *Neural Networks: Zero to Hero* (micrograd → makemore) — Weeks 13–16.
- CS231n course notes (conv nets, training) — Weeks 16–18.
- Prince, *Understanding Deep Learning* (free PDF): MLP, training, CNN, GNN, VAE,
  flows chapters as scheduled.

**Reference**
- Goodfellow, Bengio & Courville, *Deep Learning* (free online) — optimization and
  regularization chapters.
- distill.pub: *A Gentle Introduction to Graph Neural Networks*.
- Lilian Weng: posts on VAEs and normalizing flows.
- PyTorch tutorials + PyTorch Geometric docs.

**Papers**
- He et al., *Deep Residual Learning* (ResNet).
- Qu & Gouskos, *ParticleNet: Jet Tagging via Particle Clouds*.
- Qu, Li & Qian, *Particle Transformer for Jet Tagging*.
- Exa.TrkX collaboration, GNN tracking paper.
- Kingma & Welling, *Auto-Encoding Variational Bayes*.
- Dinh et al., *Density estimation using Real NVP*.
- *CaloChallenge 2022* summary paper.

## Phase 3 — Transformers & LLMs (Months 07–09)

**Spine**
- Karpathy, *Let's build GPT from scratch* + *Let's build the GPT Tokenizer* —
  Weeks 25–27.
- 3Blue1Brown, transformer/attention chapters of the neural-net series — Week 25.
- Prince, *UDL* transformer and diffusion chapters — Weeks 26, 35.
- HuggingFace LLM course (transformers, datasets, peft chapters) — Weeks 29–30.

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
- Model Context Protocol documentation + Python SDK — Week 39.
- Sutton & Barto, *Reinforcement Learning* (free online) chs. 3, 9, 13 + OpenAI
  *Spinning Up* (policy-gradient sections) — Week 43.
- Huyen, *Designing Machine Learning Systems* — Weeks 41, 44 (selected chapters).

**Reference**
- W&B / MLflow docs; PyTorch performance tuning guide; FastAPI docs.
- Keshav, *How to Read a Paper* — Week 45.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*.
- Schulman et al., *Proximal Policy Optimization Algorithms*.

**Papers (Week 45 critique pool — pick from)**
- Any recent sPHENIX/ALICE/ATLAS ML paper relevant to your thesis.
- A CaloChallenge entrant paper.
- A recent "foundation model for physics/science" paper.

## Standing subscriptions (all year, ~30 min/week)

- arXiv `hep-ex` + `cs.LG` cross-listings skim; HEP-ML living review
  (*A Living Review of Machine Learning for Particle Physics*).
- One LLM-news source of your choice (e.g. a weekly newsletter) — enough to know
  what changed, not enough to doomscroll.
