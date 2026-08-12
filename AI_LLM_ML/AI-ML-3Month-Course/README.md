# From Nuclear Physics to AI Scientist

A 12-month, 48-week self-study curriculum taking a nuclear physics PhD student
(sPHENIX) from the scientific Python stack to modern AI — classical ML, deep learning,
transformers, LLMs, agents, ML engineering, and research skills — with **no assumed
knowledge** and **full derivations**, keeping three career doors open: AI research
scientist, AI-for-science at a national lab, and applied AI/ML engineer.

## Start here

1. `00-Syllabus.md` — the full 48-week plan, gates, and policies. Read it once, whole.
2. `02-Setup-Guide.md` — get the environment working (WSL2 + uv + Jupyter + GPU/Colab).
3. `Phase1-Foundations/Month01-Scientific-Python/Week01/README.md` — begin.
4. `04-Progress-Tracker.md` — check off as you go.

## Structure

```
00-Syllabus.md            the plan (authoritative)
01-Reading-List.md        books, courses, papers by phase
02-Setup-Guide.md         environment setup
03-Project-Roadmap.md     mini-project + capstone specs
04-Progress-Tracker.md    checkboxes and gate log
NOTEBOOK_RULES.md         how exercise/solution notebooks are written
Phase1-Foundations/       Months 01–03: Python, SWE, math for ML, classical ML
Phase2-DeepLearning/      Months 04–06: NNs from scratch, CNNs, GNNs, VAEs/flows
Phase3-Transformers-LLMs/ Months 07–09: transformers, fine-tuning, RAG, diffusion
Phase4-Agents-Systems-Research/  Months 10–12: agents, MCP, MLOps, RL, capstone
references/               Bishop PRML, Russell & Norvig PDFs
archive/2026-spring-12week/      the previous 12-week course (superseded)
```

Each week folder has a `README.md` spec: objectives, core material, derivations,
exercise outline, deliverable, and spaced-review questions. Exercise notebooks are
**built when a week starts** (per `NOTEBOOK_RULES.md`), never in advance — libraries
and models drift too fast.

## How a week goes

1. Read the week `README.md`.
2. Do the readings/videos (~3 hrs); do scheduled derivations on paper.
3. Build/receive the exercise notebook; work the TODOs.
4. Do the review block (retrieval questions from earlier weeks).
5. Write a few lines of notes; check off the tracker.

## Ground rules

- Derivations on paper before code. Photograph them into the week folder.
- "Done" = fresh clone + `uv sync` + one command reproduces the result.
- Negative results go in writeups.
- Slip weeks, not content — the calendar has a month of slack built in.
