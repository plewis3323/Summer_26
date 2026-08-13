# 05 — The two-year path (AI Engineer or AI Scientist)

This course is **year 1 of 2+**. Twelve months and ~500 hours will not make you
a staff software engineer, a production ML engineer, or a research scientist.
They *will* give a science grad student the foundation those roles actually
hire from: you can program, you can train and evaluate models, you can operate
LLMs and agents, and you have shipped repos a hiring committee can clone.

Year 2 is where the title happens. Pick a role by Month 12 (Week 46 forces the
choice in the capstone proposal). You can change your mind; you cannot skip
the year-2 work and expect the title.

## What year 1 already is

| Thread | What you can point at after Week 48 |
|--------|-------------------------------------|
| **SWE** | git, tests, packages, CI, SQL, HTTP/JSON, FastAPI, Docker, logging, secrets |
| **ML** | honest classical ML, NNs from scratch, PyTorch, CNNs, GNNs, VAEs, diffusion |
| **Gen AI** | prompting with evals, LoRA, RAG, tool-using agents, MCP, LLM evals |
| **Research habits** | paper reading, reproduction, writeups, a recorded talk |

That is a **strong applied foundation**, not four careers.

## Shared work (months 13–18)

Do this regardless of role. Budget ~6–8 hrs/week on top of science/thesis work,
or a dedicated stretch if you have one.

1. **Keep the public portfolio honest.** Four capstone repos, green CI, READMEs
   a stranger can follow. Delete anything that only runs on your laptop.
2. **Use the skills on real work.** One analysis, one paper figure, one internal
   tool, or one open-source PR that is not a course exercise. Course projects
   prove you can learn; real work proves you can deliver.
3. **Read with a purpose.** One paper a week from the role list below. Use the
   Week 45 workflow. File a one-page note.
4. **Software you will be asked about in interviews.** Complexity of the code
   you already wrote (Week 04/23 Big-O); SQL joins on a dataset you own; the
   FastAPI service from Week 44 running on a machine that is not yours
   (a free-tier VM is enough).
5. **Do not collect certificates.** One shipped system beats five courses.

## AI Engineer track (months 13–24)

**The job:** build and operate LLM/ML systems other people depend on — RAG,
agents, evals, cost, latency, security, monitoring.

**Year-2 targets (pick a bar and hit it):**

- One **production-shaped** system: a real user (even if that user is your
  research group), eval harness that runs in CI, cost per request, latency
  envelope, prompt-injection tests, a rollback story.
- Contribute a non-trivial PR to an open tool you actually use (MCP server,
  eval library, serving stack) — or maintain your own for 6+ months.
- Interview loop: system-design a RAG/agent service on a whiteboard; explain
  your Week-32/34/40 evals without notes; write SQL and a small API on the spot.

**What to add that year 1 only tasted:**

- Cloud: one provider, one VM, secrets, a domain, HTTPS. Not "learn AWS."
- Observability: traces of agent runs, token/cost dashboards.
- Data: a warehouse-or-sqlite you query for eval sets; version those sets.
- Security: prompt injection, tool allowlists, PII — treat them as tests.
- Product sense: when *not* to use an agent (Week 38), written as a design doc.

**Skip unless the job posting demands it:** LeetCode grind past easy/medium
arrays and hashes; Kubernetes; Spark. Add them when a specific posting does.

**Science option:** the "user" can be your collaboration. An analysis copilot
that a colleague actually runs is a stronger AI-engineer artifact than a
generic chatbot.

## AI Scientist track (months 13–24)

**The job:** find a question, design an experiment, beat a baseline (or explain
why not), and write it so a referee can reproduce it.

**Year-2 targets:**

- One **workshop or arXiv note** from capstone track (d) or a thesis-adjacent
  ML result. A paper under review beats a perfect unpublished repo.
- A second reproduction of a paper *outside* the course list, with a public
  repo.
- Cold re-derivations still green (the monthly rotation in the tracker).
  Add: a Gaussian-process or Bayesian-update derivation if your work needs it.

**What to add that year 1 only tasted:**

- Depth in *one* family: generative models for your domain, GNNs, or LLMs —
  not all three.
- Math as needed: more of Murphy or Prince in the chapters you actually use;
  information theory if you work on compression/codecs; causal language if
  you work on interventions. Do not start a second degree in theory.
- Compute: one multi-GPU run you profiled (Week 42 skills, larger model).
- Community: present at a group meeting, a workshop, or a journal club.

**Skip unless the lab demands it:** training a foundation model from scratch;
becoming a CUDA kernel author. Read the papers; don't pretend year 2 is a
FAIR/DeepMind pretraining residency.

**Science option:** the strongest AI-scientist conversion from a physics PhD
is a thesis chapter that *is* the ML result, plus a methods paper. Aim the
capstone at that chapter.

## What neither track needs from year 1

- Being all four of SWE / MLE / AI engineer / AI scientist on day 365.
- A second programming language. Python is enough until a job requires
  another.
- Every new framework. The course pins versions; year 2 you read changelogs
  for the stack you ship, not the timeline.

## Decision rule (Week 46)

Write this sentence in `PROPOSAL.md` and do not hedge it:

> After this capstone I am aiming at **[AI Engineer / AI Scientist]**,
> because **[one artifact I will have in month 24]**.

If you want both, the engineer artifact is a system colleagues use; the
scientist artifact is a paper. Sequence them (system in months 13–18, paper
in 19–24, or the reverse). Parallel "I'll do everything" is how neither
happens.
