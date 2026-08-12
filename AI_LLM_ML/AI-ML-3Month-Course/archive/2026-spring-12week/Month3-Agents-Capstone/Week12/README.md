# Week 12 — Capstone: Pick a Track, Ship One Coherent Project

## Where this sits

Eleven weeks in, you've written: a Python ML stack (W1), classical models and BDTs (W2), PyTorch from autograd up (W3), a CNN image classifier (W4), sequence models and attention (W5), a working GPT trained on heavy-ion abstracts (W6), a LoRA-fine-tuned small model with a vLLM server (W7), a RAG system over a corpus of HEP PDFs (W8), a tool-using Claude agent (W9), the same agent reimplemented in three frameworks with tracing (W10), and an MCP server exposing your HEP toolkit to Claude Desktop (W11).

Week 12 is where you turn that pile into one deliverable. You pick a track, spend the week (or two) finishing it, and come out of the course with a thing you'd show at a group meeting — or, more importantly, a thing you'd actually use.

The three tracks are distinct enough that picking forces you to declare a direction; close enough that each reuses five-plus weeks of your prior work. Pick the one that makes you want to work on it.

## Learning objectives

By the end of Week 12 you can:

1. Scope a multi-week project, set acceptance criteria, and hit them.
2. Combine LLM inference, retrieval, tool use, and MCP into one system where each part has a clear role.
3. Ship a polished artifact — documented, tested, installable — that a colleague could run.
4. Demo and defend the system: its design, its failure modes, the tradeoffs you accepted.
5. Write up the work at the level expected in a group-meeting slide deck or internal report.

## The three tracks

### Track A — HEP Copilot (Agent + MCP + RAG)

**One-line pitch.** A Claude Desktop workflow that, given a question about sPHENIX runs or recent heavy-ion papers, answers it accurately by calling your tools, reading your resources, and retrieving from your paper corpus.

**What it combines.** W7 (small model optional), W8 (RAG), W9–W10 (tools + agent loop), W11 (MCP). Most of your code from those weeks is reused.

**What's new.** You wire the pieces together into one MCP server — `hep-copilot` — that exposes:
- Tools: `query_runs_db`, `inspect_root`, `run_python`, `search_papers` (your RAG retriever), `cite_passage`, `plot`.
- Resources: `runs://recent`, `run://{id}`, `paper://{arxiv_id}`, `notes://{note_id}`.
- Prompts: `/run_summary`, `/qc_report`, `/paper_digest`, `/analysis_plan`.

**Deliverables.**
- `hep-copilot` repo, installable via `uv sync`.
- Claude Desktop integration: you (and one colleague) use it for a full week.
- 20-task eval set where you score answers against ground truth.
- 8-slide deck: architecture, demo, eval results, limits.

**Who this is for.** You. Most PhDs spend too many hours on "which runs are usable", "does this paper say X", "plot the z-vertex distribution for these runs". This track makes those 30-second queries in Claude Desktop. You'll use it after the course ends.

**Difficulty.** Medium. All ingredients exist. The work is in integration and polish.

### Track B — Conditional Diffusion / VAE for Detector Simulation

**One-line pitch.** A small generative model that takes a (centrality, species, energy) conditioning vector and produces a sample of physically plausible per-event observables (e.g. n_tracks, mean pT, v2) — fast enough to replace a slow simulation for first-pass exploration.

**What it combines.** W2 (feature engineering, baselines), W3 (PyTorch), W4 (CNN/MLP backbones), W5 (conditioning tricks from sequence models).

**What's new.** Diffusion or VAE from scratch. Conditioning the generator on structured metadata. Evaluating generative models physically (χ² against true distributions, coverage, calibration).

**Deliverables.**
- Dataset: a synthetic or real-ish AuAu minimum-bias sample with ~50k events, ~10 observables per event.
- Model: a conditional VAE (straightforward) or conditional diffusion model (harder, cooler).
- Training: 1–2 days of compute on a GPU. Loss curves, sample quality plots.
- Evaluation: marginal χ² per observable; correlation structure preserved; conditioning actually conditions.
- `hep-gen` repo + 8-slide deck.

**Who this is for.** Anyone who wants more depth in generative modeling, or who is curious about ML-for-physics-simulation. Closer to the "ML for HEP" literature than Track A.

**Difficulty.** Medium-hard. Diffusion training is finicky. VAE is easier but less trendy.

### Track C — Paper to Pipeline (LLM → Code → Artifact)

**One-line pitch.** An agent that reads an arXiv HEP paper, extracts the analysis recipe (dataset, cuts, observables, expected result), and produces a runnable skeleton — a Python module with type-annotated functions, pinned dependencies, a sample `__main__`, and a `tests/` with at least smoke tests.

**What it combines.** W6–W7 (LLM familiarity), W8 (RAG over reference material), W9–W10 (tool use / agent frameworks), W11 (MCP for running it inside Claude Desktop). Focuses on the agent-as-developer archetype.

**What's new.** Structured extraction from unstructured text. Code generation where the model emits files the evaluator compiles/tests. Feedback loop: the agent reads the test output and repairs.

**Deliverables.**
- `paper2pipeline` repo.
- Input: arXiv ID or local PDF.
- Output: a zipped project directory with `pyproject.toml`, `src/`, `tests/`, `README.md`.
- Agent loop: extract → generate → run `pytest` → reflect on failures → repair. Max 5 iterations.
- Eval on 5 real-ish HEP papers (or synthetic toy papers): does the output compile? do smoke tests pass? does the structure match the paper's claimed workflow?
- 8-slide deck.

**Who this is for.** Anyone interested in code-gen agents or in making a "read the paper and do the analysis" tool. Most ambitious of the three.

**Difficulty.** Hard. LLMs generate plausible code that subtly doesn't run. Iteration loops are finicky.

## Pick-one-track heuristic

Answer in 30 seconds:

- Do I want something I'll actually *use* after this course? → **Track A.**
- Do I want to learn more physics-adjacent modeling? → **Track B.**
- Do I want to stretch my agent skills hardest? → **Track C.**

Don't agonize. All three are defensible. Lock it in by end of Day 1.

## Suggested plan (7–14 days)

If you have exactly one week, follow the 7-day plan. If you have two, use the first week for integration and the second for polish / eval / write-up.

### Days 1–2: scope
- Commit to a track. Open an issue-list. List every piece of prior-week code you'll reuse.
- Set acceptance criteria (below). Put them in `README.md`.
- Check all dependencies install; smoke-test every sub-component (W11 server starts, W8 retriever returns results, W7 vLLM server serves).

### Days 3–4: integrate
- For Track A: compose the MCP server; wire in RAG.
- For Track B: finalize dataset; implement model; kick off first training.
- For Track C: implement the extract → generate → run → repair loop on one paper; get it end-to-end before widening.

### Days 5–6: evaluate
- Track A: build the 20-task eval; run it.
- Track B: compute χ² and coverage; plot.
- Track C: run on 5 papers; tabulate pass/fail.

### Day 7 (or 8–14): polish
- Write the 500–1000-word write-up.
- Build the 8-slide deck.
- Clean repo. Update `README.md`. Push. Tag `v0.1`.

## Acceptance criteria (common)

Regardless of track:

1. Repository pushed to GitHub with a non-trivial `README.md`.
2. `uv sync && uv run pytest -q` passes.
3. A 500–1000-word write-up in `docs/report.md` covering: motivation, architecture, results, failure modes, next steps.
4. An 8-slide deck in `docs/deck.pdf` (build from `.key`, `.pptx`, or `reveal.md`).
5. At least one end-to-end demo you'd actually record a 3-minute video for (don't have to; but you should be able to).
6. You can answer three questions about it in your sleep: *What does it do? How does it work? When does it fail?*

## Deliverables checklist

- [ ] Track selected and logged in `docs/report.md`.
- [ ] Acceptance criteria enumerated in `README.md`.
- [ ] Component reuse mapped (which W01–W11 artifacts you port in).
- [ ] Primary artifact (server / model / pipeline) shipped and demonstrably working.
- [ ] Eval set built and run; numbers in the report.
- [ ] Failure modes written up honestly.
- [ ] Write-up + deck + tag.

## How to present this

Don't hand-wave the ML; hand-wave the UI if anything. For a group meeting:

1. **Two sentences on the problem.** "PhD students spend X hours/week doing Y. Can an LLM-driven tool reduce this?"
2. **One slide on the architecture.** Box-and-arrow diagram of your system. No code.
3. **Live demo** (or recorded, if your live demo setup scares you).
4. **Eval numbers.** Even rough ones. "On 20 questions, my system answers 16 correctly, 3 partially, 1 hallucinated."
5. **Failure modes.** Be explicit. "It fails when X. I haven't fixed that because Y."
6. **Next steps.** One slide.

Groups respect honesty about limitations. Don't oversell.

## Evaluation rubric (for self-grading)

| Area | Weak (1) | OK (3) | Strong (5) |
| --- | --- | --- | --- |
| Integration | Pieces don't talk; duct tape | Works end-to-end; occasional seams | Seamless; a colleague could use it |
| Eval | No eval | Small manual eval | Proper eval with ground truth |
| Docs | README stub | README + examples | README + examples + report + deck |
| Honesty | Claims exceed results | Matches results | Preempts questions about limits |
| Reuse | Starts over | Reuses some | Reuses most prior work meaningfully |

Target ≥ 3 on every row, ≥ 4 on two.

## What success feels like

- You open Claude Desktop the day after Week 12 ends and use your tool for something real.
- Or: you open your generative model notebook and poke at conditioning.
- Or: you feed a paper into your pipeline and watch it write code.

The course ends when you have a thing you'd demo to your advisor without apologizing.

## Tooling introduced

- Nothing new. Everything you need is in Weeks 01–11.
- Optional: `pdoc` or `mkdocs` for quick project docs.
- Optional: `just` or `make` for a `just demo` / `make run` recipe.

## Common failure modes

- **"I want to do all three."** No. One. You can revisit the others later.
- **"My eval set is 2 examples."** Push to at least 10. Ideally 20.
- **"It works on my machine."** Make sure `uv sync` + `uv run pytest -q` on a clean clone works.
- **Last-day panic.** Build the MVP end-to-end in Days 1–3. Polish after.
- **Writing the report the hour before.** The report is where you clarify your thinking. Write it Day 5, revise Day 7.

## Sign-off

Ship the thing. Tag `v0.1`. Update `04-Progress-Tracker.md` one last time. Commit.

You're done with the course.

What happens next is up to you. A reasonable short list:

- Extend the capstone to "v0.2" over the next month: more tools, better eval, an internal release.
- Pick one track you didn't do and do it lightweight.
- Start reading the next set of papers with your new vocabulary.
- Apply for a hackathon / fellowship / summer school that uses ML-for-HEP.
- Teach one thing you learned to another grad student.

The course was a trellis. The climbing is still yours.
