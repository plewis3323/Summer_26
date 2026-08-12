# 03 — Project Roadmap

Every week has a mini-project. Every month has a capstone. Month 3 has a final capstone with three tracks. Acceptance criteria are concrete and testable.

All projects live under their week folder:
```
Week02/project/
├── pyproject.toml
├── README.md          (problem statement, data, results)
├── src/<pkg>/         (implementation)
├── tests/             (pytest)
├── notebooks/         (exploration, cleared outputs)
├── results/           (figures, metrics.json)
└── data/              (gitignored, README explains how to fetch)
```

---

## Month 1 — Foundations

### Week 01 project — NumPy/pandas fluency drill
**Goal:** replicate a π⁰ invariant-mass-peak plotting workflow you already know in ROOT, but in `numpy` + `pandas` + `matplotlib`.
**Data:** synthetic photon-pair events (you generate) or a CMS open-data dimuon mass dataset (https://opendata.cern.ch/record/301).
**Deliverables:**
- A `Dataset` loading function that returns a `pandas.DataFrame`.
- A fitter that extracts (μ, σ, signal count, background count) via a Gaussian + linear background.
- A publication-quality invariant-mass figure saved to `results/invmass.png`.
- Three pytest tests: shape of the dataframe, sanity on fitted μ, sanity on fit residuals mean ≈ 0.
**Accept when:** `pytest -q` passes and `python -m week01.fit data/events.parquet` regenerates the figure.

### Week 02 project — BDT signal/background classifier
**Goal:** train a BDT and compare to logistic regression on a HEP tabular dataset.
**Data:** HEPMASS (UCI, https://archive.ics.uci.edu/dataset/347/hepmass) or Higgs Boson ML Challenge (Kaggle).
**Deliverables:**
- Preprocessing pipeline (feature standardization, train/val/test split).
- Baseline logistic regression with L2 regularization.
- `XGBClassifier` with tuned `max_depth`, `n_estimators`, `learning_rate` via 5-fold CV.
- ROC curve, PR curve, and calibration plot comparing the two.
- Feature importance plot (gain-based + permutation importance).
- A `metrics.json` with AUC, Brier score, and log-loss for each model.
**Accept when:** BDT AUC beats logistic by >2% on the held-out test set; results are reproducible (pinned seed).

### Week 03 project — micrograd + MLP on synthetic physics data
**Goal:** implement Karpathy's micrograd autograd engine from scratch, then use it to train a 2-layer MLP on a nonlinear regression task.
**Data:** `y = f(x1, x2) + noise` where `f` is a synthetic shower-energy response function (e.g. piecewise-smooth with a saturation tail).
**Deliverables:**
- `Value` class with `+`, `*`, `tanh`, `relu`, `exp`, `log`, `backward()`, fully typed.
- A `pytest` suite that compares gradients against `torch.autograd` on ~20 random expressions.
- An MLP trained with SGD, loss curve plotted.
- A PyTorch re-implementation showing identical learning behavior (±noise).
**Accept when:** all autograd tests pass; MLP reaches target loss ≤ 1.5× irreducible noise floor.

### Week 04 project — Month 1 Capstone: EMCal shower classifier (CNN)
**Goal:** train a CNN to distinguish single-photon EMCal clusters from merged-π⁰ clusters.
**Data options:**
- (A) Simulate with your own toy generator: 2D Gaussian showers for photons, two overlapping for π⁰s, varying separation and energy.
- (B) HGCAL ECAL images from CERN Open Data (https://opendata.cern.ch/).
- (C) Fashion-MNIST as a warm-up; then (A).
**Deliverables:**
- Dataset class with lazy loading and augmentation (rotations, small shifts, Poisson noise).
- A simple CNN (3–4 conv blocks, global average pool, 2-class head).
- A ResNet-18 baseline trained from scratch (or fine-tuned from ImageNet weights).
- Training loop with `wandb` logging, LR scheduler, early stopping.
- Confusion matrix, ROC, calibration plot on held-out test.
- A note on where the model fails: correlate errors with π⁰-separation angle and energy asymmetry.
**Accept when:** ROC AUC ≥ 0.85 on the test set **and** the writeup names two physical failure modes with evidence.

---

## Month 2 — Deep learning, transformers, LLMs

### Week 05 project — GNN on jet constituents
**Goal:** train a simple GNN (or ParticleNet-lite with EdgeConv) to tag jets on a small jet-physics dataset. Priority is understanding the ingredients, not beating SOTA.
**Data:** JetClass (subset), or Top Tagging Reference Dataset (Butter et al. 2019; Zenodo).
**Deliverables:**
- Data loader with per-jet k-NN graph construction.
- EdgeConv block, 2 layers, global pooling, MLP head.
- Benchmark against an MLP-on-flattened-constituents baseline.
- Discussion: when does the graph prior help?
**Accept when:** GNN beats the MLP baseline by ≥3% AUC.

### Week 06 project — nanoGPT on heavy-ion abstracts
**Goal:** train a small (≤20 M param) GPT on a scraped corpus of heavy-ion physics abstracts.
**Data:** scrape arXiv hep-ex / nucl-ex listings with keywords ("quark-gluon plasma", "heavy-ion", "RHIC", "LHC", "π⁰", "J/ψ", "jet quenching"). Aim for 50–200 MB of text after dedup.
**Deliverables:**
- Scraper script (respect arXiv's API guidance; rate-limit).
- BPE or character-level tokenizer; tokenized dataset stored as a `.bin`.
- nanoGPT-style model (you may fork karpathy/nanoGPT).
- Sample generations at epoch 0, epoch 1, final epoch — track qualitative progression.
- Validation loss curve.
**Accept when:** validation loss decreases monotonically (modulo noise) and qualitative samples shift from noise → physics-ish text.

### Week 07 project — LoRA fine-tune for metadata extraction
**Goal:** fine-tune a 1–3B open-weight model with LoRA to extract structured metadata from abstracts: `(observable, sqrt_s, centrality_range, pt_range, cuts)` as JSON.
**Model:** Llama 3.2 1B / Gemma 2B / Phi-3 mini.
**Data:** 200–500 hand-labeled abstracts. Use Claude or GPT-4 to help **draft** labels but hand-verify every one.
**Deliverables:**
- Dataset JSONL with instruction / response pairs.
- QLoRA fine-tune via `peft` + `trl.SFTTrainer`.
- Eval: JSON-schema validity rate, exact-match accuracy per field, a hallucination taxonomy.
- Baseline: zero-shot prompted base model. Fine-tuned should beat it cleanly on at least 3 of 5 fields.
**Accept when:** fine-tuned model's schema-valid rate ≥ 95% on held-out and beats zero-shot on 3+ fields; checkpoint pushed to your HF account (private).

### Week 08 project — RAG over Zotero + sPHENIX TDR
**Goal:** build a RAG system that answers physics questions citing your personal library.
**Stack:** `sentence-transformers` (BGE-small or e5-small), `chroma` or `lancedb`, `anthropic` SDK for answer synthesis.
**Deliverables:**
- PDF-to-text pipeline with section-aware chunking.
- Embedding index with metadata (paper title, section, page).
- Retrieval: top-k with MMR; optional reranker (`bge-reranker-base`).
- Answer-synthesis loop with inline citations.
- Evaluation set of 20 physics questions, hand-graded for (citations correct?, answer correct?, answer supported by retrieved chunks?).
**Accept when:** ≥15/20 answers are both correct and supported, and the system refuses when evidence is absent.

---

## Month 3 — Agents + capstone

### Week 09 project — First 2-tool agent
**Goal:** build an agent with `search_arxiv(query)` and `summarize(url)` tools via Anthropic's Messages API function calling.
**Deliverables:**
- Tool implementations as Python functions with JSON schemas.
- A tool-use loop that handles multi-turn calls and stops gracefully.
- A test notebook: 5 queries, logged traces.
**Accept when:** agent can answer "what are the three most-cited 2025 papers on jet quenching?" by correctly invoking tools.

### Week 10 project — LangGraph ReAct agent with 4 tools
**Goal:** same idea, upgraded framework, richer tools.
**Tools:** `search_arxiv`, `fetch_pdf`, `run_python` (sandboxed), `ask_user`.
**Deliverables:**
- LangGraph state machine with explicit nodes (`plan`, `tool_call`, `observe`, `reflect`).
- Logs persisted as JSONL.
- Failure injection test: what happens if `fetch_pdf` 404s?
**Accept when:** agent recovers gracefully from at least 3 simulated failure modes.

### Week 11 project — MCP server for HEP analysis
**Goal:** expose four tools via an MCP server: `fit_pi0_peak`, `list_run_files`, `get_calibration`, `generate_fun4all_macro`.
**Deliverables:**
- MCP server using the official `mcp` Python SDK.
- Each tool fully typed, with validation and error surfaces.
- A client script (or Claude Desktop config) that connects and exercises each tool.
- README with install, config, and a demo transcript.
**Accept when:** each tool is callable end-to-end from a real client; `generate_fun4all_macro` outputs a macro that compiles.

### Week 12 — Final capstone (pick ONE)

#### Track (a) — HEP analysis copilot agent
**Stack:** Week 11 MCP server + agent framework (LangGraph or Anthropic agent SDK) + your Zotero RAG from Week 08.
**Deliverables:**
- Multi-tool agent that can:
  - Read a question.
  - Retrieve relevant docs (RAG).
  - Call MCP tools to list files, fetch calibration, fit a peak.
  - Produce a Markdown report with a figure.
- Evaluation: 10 realistic analysis questions, scored on (correctness, efficiency, citation quality).
**Accept when:** runs end-to-end on 10/10 questions; at least 7/10 produce correct, cited results.

#### Track (b) — Conditional generator for EMCal showers
**Stack:** PyTorch + conditional VAE or DDPM; CaloChallenge-style data.
**Deliverables:**
- Data loader for a CaloChallenge dataset (the small Dataset 1 is tractable on a T4).
- Model implementing either (i) conditional VAE with energy/η conditioning or (ii) small 2D U-Net DDPM.
- Evaluation: reconstructed shower-shape distributions (energy, shower-width, longitudinal profile) vs Geant4 truth, KL or Wasserstein distances per observable.
- A speed benchmark: generation throughput vs Geant4 reference (seconds per shower).
**Accept when:** for at least 3 shower-shape observables, generated distributions agree with truth within the tolerances used in the CaloChallenge 2022 summary paper.

#### Track (c) — Paper-to-pipeline agent
**Stack:** RAG + agent + tool for Fun4All macro templates.
**Deliverables:**
- Agent that ingests a heavy-ion analysis paper PDF and produces:
  - A structured summary (observable, kinematics, cuts, datasets).
  - A draft Fun4All analysis module (`.C` + `.h`) from a template.
  - A short written plan for how the module differs from existing sPHENIX modules.
- Human evaluation on 3 real papers.
**Accept when:** the generated module compiles against a realistic Fun4All mock-up; plan is coherent enough for your advisor to give you 2 concrete corrections rather than 20.

---

## Cross-cutting expectations

For every project:

- **Reproducibility**: pinned `uv.lock`, seed set, `git_sha` logged in every output artifact.
- **Tests**: at least one non-trivial `pytest` per project.
- **Writeup**: `project/README.md` with: problem, data, approach, results, failure modes, what you'd do next.
- **Honest limitations**: the writeup names at least two things that didn't work and why. This is the single highest-signal skill in applied ML.
