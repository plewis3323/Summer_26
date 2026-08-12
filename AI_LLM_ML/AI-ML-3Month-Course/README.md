# From Zero to AI Scientist

A 12-month, 48-week self-study curriculum that takes you from **no coding background
and high-school math** to a working AI scientist — scientific Python, math for ML,
classical ML, deep learning, transformers, LLMs, agents, ML engineering, and research
skills — with **nothing assumed** and **full derivations**. Nuclear physics runs
through the course as the application thread (the resident student is a physics PhD),
but every physics example explains itself; you don't need to know any physics to
follow it.

The materials are **in this repo**: each week folder contains the teaching text
(`lesson.md`), the exercise page (`exercises.md`), and on project weeks the project
spec (`project.md`).

## Start here

1. `00-Syllabus.md` — the full 48-week plan, gates, and policies. Read it once, whole.
2. `02-Setup-Guide.md` — get a working environment (starts from "what is a terminal").
3. `Phase1-Foundations/Month01-Scientific-Python/Week01/lesson.md` — begin.
4. `04-Progress-Tracker.md` — check off as you go.

Already know some of this? Don't skip weeks — skip **lessons**. Open the week's
`exercises.md`, do it cold; if everything passes, move on.

## Structure

```
00-Syllabus.md            the plan (authoritative)
01-Reading-List.md        free books, courses, papers by phase (supplementary)
02-Setup-Guide.md         environment setup, from zero
03-Project-Roadmap.md     mini-project + capstone specs
04-Progress-Tracker.md    checkboxes and gate log
NOTEBOOK_RULES.md         how exercise/solution notebooks are written
Phase1-Foundations/       Months 01–03: programming from zero, math for ML, classical ML
Phase2-DeepLearning/      Months 04–06: NNs from scratch, CNNs, GNNs, VAEs/flows
Phase3-Transformers-LLMs/ Months 07–09: transformers, fine-tuning, RAG, diffusion
Phase4-Agents-Systems-Research/  Months 10–12: agents, MCP, MLOps, RL, capstone
references/               Bishop PRML, Russell & Norvig PDFs
archive/                  previous course versions (superseded)
```

Each week folder:

```
WeekNN/
  README.md      the week spec: objectives, deliverable, review questions
  lesson.md      the teaching text — ground-up, defines everything it uses
  exercises.md   the exercise page — numbered tasks with "Accept when" criteria
  project.md     (project/capstone weeks only) the full project spec
```

Exercise notebooks are **generated from `exercises.md` when a week starts** (per
`NOTEBOOK_RULES.md`), never in advance — libraries and models drift too fast. The
content is fixed now; the notebook plumbing is built fresh.

## How a week goes

1. Read the week `README.md` (2 minutes — what you'll build and why).
2. Work through `lesson.md`; do the external readings/videos it points to (~3 hrs);
   do scheduled derivations on paper.
3. Generate the exercise notebook from `exercises.md`; work the TODOs until the
   checks pass.
4. Do the review block (retrieval questions at the end of `exercises.md`).
5. Write a few lines of notes; check off the tracker.

## Ground rules

- Derivations on paper before code. Photograph them into the week folder.
- "Done" = fresh clone + `uv sync` + one command reproduces the result.
- Negative results go in writeups.
- Slip weeks, not content — the calendar has a month of slack built in.
- If a lesson uses a term it never defined, that's a bug in the course. Fix it.
