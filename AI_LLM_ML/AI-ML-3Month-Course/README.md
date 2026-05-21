# AI/ML 3-Month Self-Study Course

A senior-mentor-style curriculum taking a physics PhD student from "zero modern ML" to "can train models, fine-tune LLMs, and build tool-using agents on HEP-flavored problems" in 12 weeks.

**Author / Student:** Parker Lewis, 2nd-year physics PhD, Ohio University
**Research context:** sPHENIX / STAR heavy-ion physics, EMCal calibration, π⁰ signal extraction, Fun4All analysis modules
**Prereqs assumed:** Strong C++/ROOT, linear algebra, statistics, numerical methods. Competent-but-not-fluent Python. Zero ML / DL / LLM experience.

## Why this course exists

Most public ML curricula are written either for absolute beginners (and treat math like a tax to be avoided) or for CS grad students (and treat physics context as a curiosity). This one is built for someone who already knows how to propagate a covariance matrix, debug a Fun4All segfault at 3 a.m., and read a probability density function — but has never touched a PyTorch tensor or run a backward pass.

The design goals:

1. **Every technique is tied to a HEP use case** when the topic allows it. Logistic regression becomes a toy π⁰-vs-photon separator. CNNs become EMCal shower classifiers. Transformers become LHC-paper metadata extractors. Agents become analysis copilots.
2. **Math is derived, not hidden.** Cross-entropy, softmax, backprop-through-a-scalar-graph, self-attention, LoRA, and the score-matching form of diffusion all get worked out symbolically at least once.
3. **Every hands-on task is runnable.** Every exercise names a dataset and a setup command. Every project has acceptance criteria.
4. **Modern Python tooling is introduced inline.** `uv`, `ruff`, `pytest`, `pre-commit`, proper `pyproject.toml` projects — not in a separate "DevOps" appendix nobody reads.
5. **Tradeoffs are named honestly.** Where a method fails, where the hype exceeds reality, where "production ML" is mostly glue code.

## Course structure

```
AI-ML-3Month-Course/
├── 00-Syllabus.md / .pdf        ← top-down course map
├── 01-Reading-List.md           ← books, papers, videos, Codecademy units
├── 02-Setup-Guide.md            ← Python env, CUDA, HF token, VS Code
├── 03-Project-Roadmap.md        ← every project with acceptance criteria
├── 04-Progress-Tracker.md       ← week-by-week checkboxes
├── Month1-Foundations/          ← classical ML + first neural nets
│   └── Week01..Week04
├── Month2-DeepLearning-LLMs/    ← transformers, fine-tuning, RAG
│   └── Week05..Week08
└── Month3-Agents-Capstone/      ← tool use, MCP, capstone project
    └── Week09..Week12
```

Each `Week##/` contains:

- **`README.md`** — the module. Learning objectives, concepts, math, worked examples, HEP analogies, pitfalls.
- **`reading.md`** — primary + supplementary readings with exact chapter/page/timestamp references.
- **`exercises.md`** — 5–10 runnable exercises, easy → hard, each with data source and hints.
- **`project.md`** — one mini-project with acceptance criteria.
- **`resources.md`** — curated docs, tutorials, GitHub repos, blog posts.

## Topic map at a glance

| Week | Topic | HEP hook |
|------|-------|----------|
| 01 | Python scientific stack, reproducibility | Replace a ROOT `TH1F` workflow with `numpy` + `matplotlib` |
| 02 | Classical ML, BDTs, XGBoost | Signal/background separation; the workhorse of LHC analyses |
| 03 | PyTorch, MLPs, manual backprop (Karpathy micrograd) | Fitting a shower-energy response curve with a tiny net |
| 04 | CNNs — **Month 1 capstone** | Single-photon vs merged-π⁰ EMCal cluster classifier |
| 05 | Sequences, attention, GNNs | ParticleNet, Particle Transformer, Exa.TrkX tracking |
| 06 | Transformer from scratch (nanoGPT style) | Trained on a scraped heavy-ion abstracts corpus |
| 07 | HuggingFace + LoRA/QLoRA fine-tuning | Extract (observable, √s, centrality, pT, cuts) from abstracts |
| 08 | RAG + vector DBs | Zotero + sPHENIX TDR as a retrieval corpus |
| 09 | Tool use / function calling | First 2-tool agent |
| 10 | Agent frameworks (LangGraph, smolagents), ReAct | Building Effective Agents patterns |
| 11 | Model Context Protocol (MCP) | Expose `fit_pi0_peak`, `list_run_files`, `get_calibration`, `generate_fun4all_macro` |
| 12 | **Final capstone** — pick one | (a) HEP analysis copilot, (b) conditional diffusion/VAE shower generator, (c) paper-to-pipeline agent |

## How to work through it

Budget ~15–20 hours per week. A reasonable weekly rhythm:

- **Mon–Tue:** Read the assigned chapters, watch the videos, write up the concept summary in your own words.
- **Wed–Thu:** Do the exercises. Don't skip ahead; a lot of intuition is in the gradients you hand-compute.
- **Fri:** Ship the mini-project. Push to GitHub with tests and a short writeup.
- **Sat:** Skim a referenced HEP-ML paper and jot a one-paragraph reaction.
- **Sun:** Tick boxes in `04-Progress-Tracker.md`. Short reflection on what clicked and what didn't.

If you fall behind, cut the supplementary readings before you cut the project. The project is where intuition crystallizes.

## Repository hygiene

Treat this folder as a real repo from day 1:

```bash
cd AI-ML-3Month-Course/
git init
git add .
git commit -m "Initial course scaffold"
```

Each week-project lives in its own subdirectory (e.g. `Month1-Foundations/Week02/project/`) with its own `pyproject.toml`. See `02-Setup-Guide.md` for the full environment setup.

## License and attribution

All original text in this course is CC-BY-4.0. Code samples are MIT. Externally-cited textbooks, papers, videos, and Codecademy content remain under their respective licenses; this course summarizes and points at them, never reproduces them at length.

---

Start with `00-Syllabus.md` and `02-Setup-Guide.md`, then begin `Month1-Foundations/Week01/`.
