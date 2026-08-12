# 03 — Project Roadmap

Every month ships something. Mini-projects are 1-week builds inside a month;
capstones close a phase and must pass the gate in `00-Syllabus.md` §5. All projects:
tested (`pytest -q`), reproducible (`uv sync` + one command), honest writeup
(what failed first, what you'd do next).

## Mini-projects

| Week | Project | Baseline to beat / criterion |
|------|---------|------------------------------|
| 04 | **Dimuon spectrum** — a real particle-physics analysis (CERN Open Data J/ψ→μμ mass peak) as a guided, tested Python package; the project spec explains all the physics | Peak mass within PDG value ± fit σ; one command → plot |
| 08 | **Optimizer race** — hand-rolled GD/momentum/RMSProp/Adam fitting physics models on pathological loss surfaces | All four converge where they should; Rosenbrock trajectory plot |
| 16 | **MLP vs BDT** — PyTorch MLP on the Capstone-1 tabular data | Match or beat the BDT AUC, or explain why tabular ≠ deep-learning home turf |
| 20 | **EMCal cluster CNN** — single-photon vs merged-π⁰ on simulated calorimeter images | Beat the tabular-features baseline on the same events |
| 32 | **Abstract extractor** — LoRA fine-tune of a 1–3B model for structured metadata from nuclear physics abstracts | Beat zero-shot prompting on a held-out set, report per-field F1 |
| 40 | **Analysis copilot prototype** — agent driving your Week-39 MCP tools | Completes 3 scripted analysis tasks end-to-end unattended |
| 44 | **Model service** — one earlier model behind FastAPI in Docker | Health check + latency budget met; deploy from clean machine |

## Capstone 1 (Week 12) — Tabular particle-ID classifier

Gamma/hadron separation (MAGIC telescope dataset, UCI) or comparable HEP open data.
- Full pipeline: ingest → features → nested CV model selection (logistic vs RF vs
  XGBoost) → calibration → threshold choice for a stated physics goal (e.g. fixed
  background rejection).
- **Gate:** repo with green tests, ROC + calibration plots checked in, leakage
  audit written up.

## Capstone 2 (Week 24) — choose one

**(a) GNN cluster/jet classifier.** Point-cloud GNN on the Week-20 data (or a public
jet dataset), benchmarked against your CNN and BDT. Ablate: does the graph structure
actually help, or is it the extra parameters?

**(b) VAE fast-sim.** Generate EMCal showers conditioned on energy. Validation is
physics, not eyeballs: energy response linearity, resolution vs √E, shower-shape
distributions vs the simulator's.

- **Gate:** stated baseline beaten or the failure understood in writing; ELBO/backprop
  re-derived cold and filed.

## Capstone 3 (Week 36) — choose one

**(a) Physics-literature assistant.** RAG over a physics document library you
assemble (arXiv PDFs from the Week 27/32 corpora; add your own papers — e.g. a
detector TDR or reference-manager library — if you have them), answer synthesis
with citations, your Week-32 extractor for structured lookups.
Evaluated on a held-out Q&A set you write in Week 34 (≥30 questions): retrieval
recall@5, answer faithfulness, hallucination rate vs no-RAG baseline.

**(b) Conditional diffusion calo generator.** DDPM conditioned on energy/angle;
same physics validation suite as Capstone 2b, head-to-head vs your VAE. Report
sampling cost honestly.

- **Gate:** one script runs the system; benchmark table vs zero-shot/VAE baseline.

## Final capstone (Weeks 47–48) — choose one track

| Track | What ships | Signals to |
|-------|-----------|------------|
| (a) HEP analysis copilot | Agent + MCP + RAG over your real sPHENIX workflow; recorded demo of an actual analysis task | AI engineering + AI-for-science |
| (b) Generative fast-sim | Diffusion/flow calo generator, CaloChallenge-style validation writeup | Research + AI-for-science |
| (c) Paper-to-pipeline agent | Agent that reads a heavy-ion paper and scaffolds a Fun4All analysis module | AI engineering |
| (d) Reproduce-and-extend | A recent HEP-ML paper reproduced + one novel twist, written as an arXiv-ready note | Research scientist |

Proposal is drafted in Week 46 (problem, baselines, evaluation plan, risks, compute
budget) — build in Week 47, validate + writeup + recorded demo in Week 48.
**Pick the track that feeds your thesis or your first AI job application — ideally
both.** If aiming at research-scientist roles, (d) then polishing it into a workshop
submission during the post-course slack month is the strongest single move.

## After the course (months 13–36 sketch)

The course ends with artifacts; the 3-year goal needs a record. Rough shape:
year 2 — publish 1–2 ML-for-physics results (thesis-adjacent), contribute to an open
ML tool, start applying the skills inside sPHENIX visibly; year 3 — target role
conversion (lab AI hire, industry research/engineering) with the portfolio: 4 capstone
repos, a fine-tuned model, an MCP server, a publication or workshop paper, and a
thesis that uses ML for real.
