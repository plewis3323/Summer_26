# 04 — Progress Tracker

Tick boxes as you complete items. Keep honest — an unchecked box is information. The end-of-week reflection prompts are where learning consolidates.

---

## Month 1 — Foundations

### Week 01 — Python scientific stack & reproducibility
- [ ] Read: Bishop Ch. 1 §1.1–1.3; VanderPlas *PDSH* Ch. 2–3 (skim)
- [ ] Videos: 3Blue1Brown NN series (all 4)
- [ ] Codecademy: "Build a Machine Learning Model" — modules 1–2
- [ ] Exercises 1–8 complete
- [ ] Project: π⁰ peak fit reproduced in numpy/pandas
- [ ] Tests passing; repo pushed
- [ ] Reflection: What felt most/least like ROOT? _________

### Week 02 — Classical ML
- [ ] Read: Bishop §3.1–3.3, §4.1.3, §4.3.2–4.3.4, §14.3–14.4
- [ ] Read: Guest, Cranmer, Whiteson (arXiv:1806.11484) — full
- [ ] Codecademy: "Feature Engineering", "Build a Machine Learning Model" — modules 3–5
- [ ] Andrew Ng ML Spec: Course 2 bias/variance lectures
- [ ] Exercises 1–10 complete
- [ ] Project: BDT vs logistic on HEPMASS, AUC +2% achieved
- [ ] Reflection: Where does the BDT fail? _________

### Week 03 — PyTorch + MLPs + micrograd
- [ ] Read: Goodfellow §4.1–4.3, §6.1–6.5
- [ ] Videos: Karpathy "micrograd" (full); makemore part 1
- [ ] Codecademy: "PyTorch for Deep Learning" — modules 1–3
- [ ] Exercises 1–9 complete
- [ ] Project: micrograd implemented, gradient tests pass, MLP trained
- [ ] Reflection: Which autograd edge case bit you? _________

### Week 04 — CNNs + Month 1 Capstone
- [ ] Read: Goodfellow Ch. 9 (all), §7.1–7.5
- [ ] Read: Karpathy "Recipe for Training Neural Networks"
- [ ] Videos: fast.ai Lecture 1, Lecture 3
- [ ] Codecademy: "PyTorch for Deep Learning" — CNN modules
- [ ] Exercises 1–8 complete
- [ ] Capstone: photon vs π⁰ classifier, AUC ≥ 0.85
- [ ] Writeup names two physical failure modes
- [ ] `git tag month-1-complete`
- [ ] 250-word retrospective in `Month1-Foundations/retro.md`
- [ ] One-issue "biggest thing I don't understand" filed

---

## Month 2 — Deep learning, transformers, small LLMs

### Week 05 — Sequences, attention, GNNs for HEP
- [ ] Read: Murphy vol. 2 Ch. 16
- [ ] Read: Qu & Gouskos ParticleNet (arXiv:1902.08570)
- [ ] Read: Qu et al. Particle Transformer (arXiv:2202.03772)
- [ ] Skim: Exa.TrkX (arXiv:2103.06995)
- [ ] Exercises 1–7 complete
- [ ] Project: GNN beats MLP baseline on jets
- [ ] Reflection: Where is the message-passing prior actually binding? _________

### Week 06 — Transformer from scratch
- [ ] Read: Vaswani et al. "Attention Is All You Need" — full, including appendices
- [ ] Videos: Karpathy "Let's build GPT"
- [ ] Exercises 1–8 complete
- [ ] Project: nanoGPT trained on heavy-ion abstracts corpus
- [ ] Val loss curve monotone; sampling looks physics-flavored
- [ ] Reflection: What surprised you about the training dynamics? _________

### Week 07 — HuggingFace + LoRA/QLoRA fine-tune
- [ ] Read: LoRA paper (arXiv:2106.09685); QLoRA paper (arXiv:2305.14314)
- [ ] HuggingFace NLP Course: chapters 3, 5, 7
- [ ] Codecademy: "Intro to Hugging Face", "LLM Finetuning with LoRA, QLoRA, and Hugging Face"
- [ ] Exercises 1–8 complete
- [ ] Project: fine-tuned metadata extractor, schema-valid ≥95%, beats zero-shot
- [ ] Checkpoint pushed to HF Hub (private)
- [ ] Reflection: What did LoRA actually parameterize? _________

### Week 08 — RAG, embeddings, vector DBs
- [ ] Read: Lewis et al. RAG paper (arXiv:2005.11401)
- [ ] Read: Karpukhin et al. DPR (arXiv:2004.04906)
- [ ] HuggingFace NLP Course: embeddings + semantic search chapter
- [ ] Exercises 1–7 complete
- [ ] Project: RAG over Zotero + sPHENIX TDR, ≥15/20 eval correct + cited
- [ ] `git tag month-2-complete`
- [ ] 250-word retrospective in `Month2-DeepLearning-LLMs/retro.md`

---

## Month 3 — Agents + capstone

### Week 09 — Tool use & function calling
- [ ] Read: Anthropic tool-use docs
- [ ] Read: Toolformer (arXiv:2302.04761) — skim only
- [ ] Exercises 1–6 complete
- [ ] Project: 2-tool agent, arXiv search + summarize
- [ ] Reflection: When does the model hallucinate a tool call? _________

### Week 10 — Agent frameworks (LangGraph, smolagents), ReAct
- [ ] Read: Yao et al. ReAct (arXiv:2210.03629)
- [ ] Read: Anthropic "Building Effective Agents"
- [ ] Codecademy: "Build Autonomous AI Agents with LangChain"
- [ ] Exercises 1–7 complete
- [ ] Project: LangGraph ReAct agent with 4 tools; survives 3 failure modes
- [ ] Reflection: Which pattern from Anthropic's blog did you end up using? _________

### Week 11 — Model Context Protocol
- [ ] Read: MCP spec (full)
- [ ] Read: Anthropic MCP blog posts / examples
- [ ] Exercises 1–7 complete
- [ ] Project: MCP server with 4 HEP tools, working from a real client

### Week 12 — Capstone
- [ ] Selected track: (a) / (b) / (c)
- [ ] Milestone 1: design doc + architecture diagram
- [ ] Milestone 2: end-to-end skeleton runs
- [ ] Milestone 3: evaluation + failure analysis
- [ ] Milestone 4: writeup with honest limitations
- [ ] `git tag course-complete`
- [ ] Final 500-word retrospective

---

## Weekly meta-checks

Each Sunday, answer three questions in a journal (physical or digital):

1. **What did I learn this week that I'll still remember in a year?**
2. **What didn't click, and what's my plan to revisit?**
3. **Where did I cut corners, and is that OK?**

Most ML learning loss happens when people skip step 2 and drift into vibes. Resist.

---

## Calibration table (fill in as you go)

| Week | Expected hours | Actual hours | Biggest unplanned rabbit hole |
|------|----------------|--------------|-------------------------------|
| 01 | 14–18 | | |
| 02 | 14–18 | | |
| 03 | 15–20 | | |
| 04 | 18–22 | | |
| 05 | 14–18 | | |
| 06 | 18–22 | | |
| 07 | 18–22 | | |
| 08 | 14–18 | | |
| 09 | 12–15 | | |
| 10 | 15–18 | | |
| 11 | 15–18 | | |
| 12 | 20–30 | | |
