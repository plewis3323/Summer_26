# 01 — Reading List

Organized by source type. Each week's `reading.md` points into this list with chapter / page / timestamp specificity. Read what's assigned for the week first; treat everything else as reference.

## A. Textbooks (selected chapters only)

### Bishop — *Pattern Recognition and Machine Learning* (2006)
The foundational Bayesian-flavored ML textbook. Dense, rigorous, readable.

- Ch. 1 — Introduction, polynomial curve fitting, probability theory refresher (Week 01 supplementary)
- Ch. 3 — Linear Models for Regression (Week 02 primary, §3.1–3.3)
- Ch. 4 — Linear Models for Classification (Week 02 primary, §4.1.3, §4.3.2–4.3.4 logistic regression + IRLS)
- Ch. 5 — Neural Networks (Week 03 supplementary, §5.1–5.3 for backprop derivation)
- Ch. 14 — Combining Models (Week 02 primary, §14.3–14.4 boosting and trees)

### Goodfellow, Bengio, Courville — *Deep Learning* (2016)
The canonical deep learning textbook. Some chapters have aged (RNNs, generative models); chapters on optimization and regularization are still essential.

- Ch. 4 — Numerical Computation (Week 03 primary, §4.1–4.3 overflow/underflow, gradient-based opt)
- Ch. 6 — Deep Feedforward Networks (Week 03 primary, §6.1–6.5 covers activation functions and backprop as Jacobian products)
- Ch. 7 — Regularization (Week 04 primary, §7.1–7.5, §7.8, §7.12)
- Ch. 8 — Optimization for Training Deep Models (Week 04 primary, §8.1–8.5)
- Ch. 9 — Convolutional Networks (Week 04 primary, §9.1–9.5, §9.10)
- Ch. 10 — Sequence Modeling: RNNs (Week 05 supplementary, §10.1–10.2 only — dated but useful context)

### Murphy — *Probabilistic Machine Learning: An Introduction* (2022) and *Advanced Topics* (2023)
Modern Bayesian-flavored ML. Freely available at probml.github.io/pml-book/.

- Book 1, Ch. 10 — Logistic Regression (Week 02 supplementary)
- Book 1, Ch. 13 — Neural Networks for Unstructured Data (Week 03 supplementary)
- Book 1, Ch. 14 — Neural Networks for Images (Week 04 supplementary)
- Book 2, Ch. 16 — Attention (Week 05–06 primary)
- Book 2, Ch. 25 — Diffusion Models (Week 12 primary, only if picking track (b))

### Karpathy notes and accompanying repos
- `karpathy/micrograd` (Week 03 primary)
- `karpathy/nanoGPT` (Week 06 primary)
- `karpathy/llm.c` (Week 06 reference only — a good C reading if you miss writing C)

## B. Video lecture series

### Karpathy — *Neural Networks: Zero to Hero* (YouTube playlist)
Primary video resource for the course. Watch in order.

- **Lecture 1 — "The spelled-out intro to neural networks and backpropagation: building micrograd"** (~2h25m). Week 03.
  - 0:00–12:00 calculus refresher — skim if comfortable
  - 12:00–55:00 build the `Value` class with `+`, `*`, `tanh`
  - 55:00–1:25:00 build backward()
  - 1:25:00–end wrap into MLP, train on a toy dataset
- **Lecture 2 — "The spelled-out intro to language modeling: building makemore"** (~1h57m). Week 03/05.
- **Lecture 6 — "Let's build GPT: from scratch, in code, spelled out"** (~1h56m). Week 06.
  - 0:00–20:00 tokenization, batching
  - 20:00–56:00 bigram baseline → self-attention
  - 56:00–1:20:00 multi-head attention, blocks, residual connections
  - 1:20:00–end scale up, training tips
- **Lecture 7 — "Let's build the GPT tokenizer"** (~2h13m). Week 07 (skim before fine-tuning).

### 3Blue1Brown — *Neural Networks* series (YouTube)
For intuition, not implementation. 4 videos, ~1 hour total. Watch before Week 03.

### fast.ai — *Practical Deep Learning for Coders* (2022 version, course.fast.ai)
The counter-weight to the math-first view. Watch lectures 1–4 over Weeks 03–04.

- Lecture 1 (top-down intuition)
- Lecture 3 (fine-tuning, transfer learning)
- Lecture 4 (tabular + ethics)
- Lecture 5 (from-scratch MLPs)

### Andrew Ng — *Machine Learning Specialization* (Coursera, 2022)
Three courses. Treat as **review reference**, not primary reading — you'll outgrow the pace fast. Dip into lectures on regularization and bias/variance (Course 2, Weeks 1–3).

## C. Papers (read in full unless noted)

### HEP-ML foundational

- Guest, Cranmer, Whiteson — *Deep Learning and its Application to LHC Physics* (Annu. Rev. Nucl. Part. Sci. 2018). arXiv:1806.11484. Week 02 primary.
- Radovic et al. — *Machine Learning at the Energy and Intensity Frontiers of Particle Physics* (Nature 2018). Week 02 supplementary.
- Qu, Gouskos — *ParticleNet: Jet Tagging via Particle Clouds* (Phys. Rev. D 2020). arXiv:1902.08570. Week 05 primary.
- Qu et al. — *Particle Transformer for Jet Tagging* (ICML 2022). arXiv:2202.03772. Week 05 primary.
- Ju et al. — *Performance of a geometric deep learning pipeline for HL-LHC particle tracking* (Exa.TrkX collaboration, Eur. Phys. J. C 2021). arXiv:2103.06995. Week 05 supplementary.
- Paganini, de Oliveira, Nachman — *CaloGAN* (Phys. Rev. D 2018). arXiv:1712.10321. Week 12 track (b) reference.
- CaloChallenge 2022 summary paper (arXiv:2410.21611 or the latest). Week 12 track (b) primary.

### Transformer / LLM foundational

- Vaswani et al. — *Attention Is All You Need* (NeurIPS 2017). arXiv:1706.03762. Week 06 primary.
- Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022). arXiv:2106.09685. Week 07 primary.
- Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023). arXiv:2305.14314. Week 07 primary.
- Lewis et al. — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020). arXiv:2005.11401. Week 08 primary.
- Karpukhin et al. — *Dense Passage Retrieval for Open-Domain QA* (EMNLP 2020). arXiv:2004.04906. Week 08 supplementary.

### Agents

- Yao et al. — *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023). arXiv:2210.03629. Week 10 primary.
- Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning* (NeurIPS 2023). arXiv:2303.11366. Week 10 supplementary.
- Schick et al. — *Toolformer* (NeurIPS 2023). arXiv:2302.04761. Week 09 supplementary.
- Anthropic — *Building Effective Agents* (engineering blog, 2024). https://www.anthropic.com/engineering/building-effective-agents. Week 10 primary.
- Anthropic — *Tool use (function calling) with the Messages API* docs. Week 09 primary.
- Model Context Protocol spec — https://modelcontextprotocol.io/specification. Week 11 primary.

### Generative models (for Week 12 track b)

- Kingma, Welling — *Auto-Encoding Variational Bayes* (ICLR 2014). arXiv:1312.6114.
- Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020). arXiv:2006.11239.
- Song, Ermon — *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019). arXiv:1907.05600.

## D. Living references

- **HEP-ML Living Review.** https://iml-wg.github.io/HEPML-LivingReview/ — bibliography curated by the HEP community. Browse it at least once in Week 02 and revisit in Week 05 for GNNs and Week 12 for generative shower models.
- **HuggingFace NLP Course.** https://huggingface.co/learn/nlp-course/ — chapters 1–7 align with Weeks 06–08.
- **HuggingFace PEFT docs.** https://huggingface.co/docs/peft/ — Week 07.
- **PyTorch tutorials.** https://pytorch.org/tutorials/ — reference, especially the 60-minute Blitz.
- **Anthropic cookbook.** https://github.com/anthropics/anthropic-cookbook — tool use and RAG recipes.

## E. Codecademy (you have Pro)

The Codecademy courses are useful for muscle memory and Python syntax drills, not for conceptual depth. Finish them in parallel with the main reading, not instead of it.

- **"Build a Machine Learning Model"** — Weeks 01–02. Pandas and scikit-learn refresher.
- **"Feature Engineering"** — Week 02.
- **"PyTorch for Deep Learning"** — Weeks 03–04. Treat the project segments as additional practice.
- **"Intro to Hugging Face"** — Week 07, first 2 days.
- **"LLM Finetuning with LoRA, QLoRA, and Hugging Face"** — Week 07.
- **"Build Autonomous AI Agents with LangChain"** — Week 10. Note: the course leans on LangChain; you'll rewrite in LangGraph + smolagents after.

## F. Style / craft

- Karpathy — *A Recipe for Training Neural Networks* (blog, 2019). https://karpathy.github.io/2019/04/25/recipe/. Week 04.
- Chip Huyen — *Designing Machine Learning Systems* (O'Reilly, 2022), Ch. 3 and Ch. 6 only. Week 07–08.
- Andrej Karpathy — *State of GPT* (Microsoft Build 2023 keynote). YouTube. Week 07.

## G. Datasets and data portals (referenced throughout)

- **CERN Open Data Portal.** https://opendata.cern.ch/ — CMS and ATLAS-derived datasets; several are small and tabular.
- **HEPMASS.** UCI ML Repository. Tabular jet physics classification. Week 02.
- **Higgs Boson ML Challenge.** Kaggle. Classic HEP BDT exercise. Week 02.
- **JetClass.** HuggingFace / Zenodo. ~100M jets for Particle Transformer. Week 05.
- **Fashion-MNIST.** Standard CNN sanity check. Week 04.
- **Pythia8-generated photon vs π⁰ shower images.** You will simulate / download in Week 04. Fallback: CMS HGCAL sample from CERN Open Data.
- **inspire-hep API + bioRxiv/arXiv scraping.** Week 06.
- **CaloChallenge 2022.** https://calochallenge.github.io/homepage/. Week 12 track (b).

## H. What you are *not* reading and why

You'll notice this list doesn't assign Sutton & Barto (RL), nor Gelman's BDA, nor Rasmussen & Williams (GP). They're worth reading eventually, but within 12 weeks chasing them dilutes the deep-learning + agents core. Come back after.
