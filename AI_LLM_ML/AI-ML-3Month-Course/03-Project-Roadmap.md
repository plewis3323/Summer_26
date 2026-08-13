# 03 — Project Roadmap

Every month ships something. Mini-projects are 1-week builds inside a month;
capstones close a phase and must pass the gate in `00-Syllabus.md` §5. All projects:
tested (`pytest -q`), reproducible (`uv sync` + one command), honest writeup
(what failed first, what you'd do next).

**Two options on every build after Week 04.** The default worked example is
science (physics, fully explained). The general option exists so a portfolio can
aim at a lab *or* at an AI Engineer / AI Scientist application. Pick once per
project, write it in the repo README, and keep the same eval discipline either way.

## Mini-projects

| Week | Project | Baseline to beat / criterion |
|------|---------|------------------------------|
| 04 | **Dimuon spectrum** — CERN Open Data J/ψ→μμ mass peak as a guided, tested Python package with CI and a sqlite results table | Peak mass within PDG value ± fit σ; one command → plot; CI green |
| 08 | **Optimizer race** — hand-rolled GD/momentum/RMSProp/Adam on pathological loss surfaces | All four converge where they should; Rosenbrock trajectory plot |
| 16 | **MLP vs BDT** — PyTorch MLP on the Capstone-1 tabular data | Match or beat the BDT AUC, or explain why tabular ≠ deep-learning home turf |
| 20 | **Image classifier** — EMCal photon vs merged-π⁰ **or** a public image task, vs a tabular-features baseline | Beat the tabular baseline on the same events/images |
| 23 | **Data API** — FastAPI + sqlite + Docker over a table you own (cut-flow / run metadata **or** any tidy table) | Tests hit the endpoints; container serves the same answers |
| 32 | **Structured extractor** — LoRA fine-tune of a 1–3B model on domain text (physics abstracts **or** a schema you define) | Beat zero-shot *and* beat your Week-29 best prompt on a held-out set |
| 40 | **Copilot prototype** — agent driving your Week-39 MCP tools (science analysis **or** general data tools) | Completes 3 scripted tasks end-to-end unattended; eval table includes cost |
| 44 | **Model service** — one earlier model behind FastAPI in Docker (spiral from Week 23) | Health check + latency budget met; CI builds the image; deploy from clean machine |

## Capstone 1 (Week 12) — Tabular classifier

**Science default:** gamma/hadron separation (MAGIC telescope, UCI) or comparable
HEP open data.
**General option:** any public tabular rare-event set (fraud, medical screening,
credit) with the same protocol.

- Full pipeline: ingest → features → nested CV model selection (logistic vs RF vs
  XGBoost) → calibration → threshold choice for a stated goal (fixed background
  rejection, or a stated precision/recall target).
- **Gate:** repo with green tests *and* CI, ROC + calibration plots checked in,
  leakage audit written up.

## Capstone 2 (Week 24) — choose one

**(a) GNN classifier.** Point-cloud or graph GNN, benchmarked against your CNN
(or a tabular baseline). Ablate: does the graph structure actually help, or is it
the extra parameters?
- Science: Week-20 calorimeter clusters / a public jet dataset.
- General: a public graph dataset (citation, molecules) with the same ablation.

**(b) VAE generator.** Conditional VAE with *distributional* validation, not
eyeballs.
- Science: EMCal showers conditioned on energy (linearity, resolution vs √E,
  shower shapes vs the simulator).
- General: a public image set; report reconstruction FID/MSE *and* at least two
  distributional checks you pre-commit (class-conditional histograms, etc.).

- **Gate:** stated baseline beaten or the failure understood in writing; ELBO/backprop
  re-derived cold and filed.

## Capstone 3 (Week 36) — choose one

**(a) Domain-literature assistant.** RAG over a document library you assemble,
plus your Week-32 extractor for structured lookups. Evaluated on a held-out Q&A
set you write in Week 34 (≥30 questions): retrieval recall@5, answer faithfulness,
hallucination rate vs no-RAG *and* vs prompt-only.
- Science: physics papers / a detector TDR.
- General: any domain library you can legally assemble (your own papers, a public
  corpus, a handbook).

**(b) Conditional diffusion generator.** DDPM with the same validation suite as
Capstone 2b, head-to-head vs your VAE. Report sampling cost honestly.
- Science: calo showers.
- General: the same public image set as 2b, or a new one with a written reason.

- **Gate:** one script runs the system; benchmark table vs the stated baseline.

## Final capstone (Weeks 47–48) — choose one track

Write the role sentence in Week 46 (`05-Two-Year-Path.md`). Engineer-shaped
tracks ship a system someone else can call. Scientist-shaped tracks ship a
result a referee could reproduce.

| Track | What ships | Signals to |
|-------|-----------|------------|
| (a) Domain copilot | Agent + MCP + RAG over a real workflow (science analysis *or* a research-group tool); recorded demo of an actual task | AI Engineer + AI-for-science |
| (b) Generative model with validation | Diffusion/flow/VAE generator, distributional writeup (CaloChallenge-style *or* the general equivalent) | AI Scientist + AI-for-science |
| (c) Paper-to-pipeline agent | Agent that reads a paper and scaffolds an analysis or experiment | AI Engineer |
| (d) Reproduce-and-extend | A recent paper reproduced + one novel twist, written as an arXiv-ready note | AI Scientist |
| (e) Evaluated LLM service | One model/RAG/agent behind the Week-44 service pattern, with CI evals, cost, latency, and injection tests | AI Engineer |

Proposal is drafted in Week 46 (problem, baselines, evaluation plan, risks, compute
budget, **role sentence**) — build in Week 47, validate + writeup + recorded demo
in Week 48.

**Pick the track that feeds your thesis *or* your first AI job application —
ideally both.** If aiming at research-scientist roles, (d) then polishing it into
a workshop submission during the post-course slack month is the strongest single
move. If aiming at AI Engineer roles, (a) or (e) with a colleague who actually
uses the system is the strongest single move.

## After the course

Year 2 is specified in `05-Two-Year-Path.md`, not sketched here. Rough shape:
months 13–18 shared portfolio + real work; months 19–24 the role conversion
(system in production-shaped use, or a paper under review).
