# Week 40 — Multi-Agent Orchestration, Agent Evals, and the Copilot Prototype

You would never quote a result without a systematic-uncertainty analysis; this week you learn the equivalent discipline for agents — measure them — and then ship a first copilot (science analysis by default; any tool suite you own is allowed).

## Objectives

- Implement orchestrator–worker delegation with explicit handoffs: task briefs out, structured reports back, no shared conversation state assumed.
- Explain the cost model of multi-agent systems (context re-establishment, report round-trips) and when a single loop is cheaper.
- Evaluate an agent quantitatively: task success rate on a fixed task suite, cost per task, latency, plus qualitative trajectory review.
- Build a copilot prototype: one agent loop driving the Week-39 MCP tools against
  real (or realistic) files.
- Write an honest failure analysis from reviewed trajectories.

## Core material (~3 hrs)

- Re-read the orchestrator–workers and evaluator–optimizer sections of Anthropic's *Building Effective Agents*.
- Anthropic engineering blog post on building a multi-agent research system (by title) — focus on how sub-agent briefs are written and why parallelism helped.
- Reading on agent evaluation: the SWE-bench paper's evaluation setup (skim) as a model of task-suite + success-criterion design; note what "resolved" means operationally.
- Your Week-32 notes on LLM-as-judge and its failure modes — they apply verbatim to trajectory grading.

## Exercises (built when the week starts)

1. Handoff schema: define the worker task brief and report as JSON schemas; orchestrator fans out 3 fit tasks and merges. Accept when: all worker reports validate against the schema and the merged table matches single-agent results.
2. Task suite: write 10 analysis tasks with programmatically checkable success criteria (e.g. "reported mass within 2 MeV of reference fit"). Accept when: `run_suite.py` scores any agent 0–10 with no human in the loop.
3. Baseline vs. agent: run the suite against (a) your Week-38 fixed workflow and (b) the agent loop. Accept when: a table reports success rate, mean cost, and mean latency for both.
4. Trajectory review: dump full transcripts for 3 failures; annotate the first wrong step in each. Accept when: `failures.md` names a root cause per failure (bad tool description, missing tool, model error, bad task spec).
5. Copilot mini-project: wire the loop + MCP server + a thin CLI (`copilot "question"`) into one package. Accept when: fresh clone + `uv sync` + one command answers 2 scripted demo questions end-to-end.
6. Eval the copilot: run the task suite on the copilot; state success rate and cost, and one prioritized improvement. Accept when: `EVAL.md` contains the numbers and the improvement is justified by a specific trajectory.

## Deliverable

`week40/copilot/` — the prototype package, task suite, eval results (`EVAL.md`), `failures.md`, and a short recorded demo (screen capture) of the CLI session.

## Review

1. Week 9: define generalization error. What is the analog for an agent evaluated on a 10-task suite it was developed against?
2. Week 32: give two ways an LLM-as-judge can be systematically biased, and how you'd detect each in trajectory grading.
3. Week 11: why can feature importances lie? Connect this to trusting an agent's self-reported reasoning about which tool it used and why.
4. Week 27: what did BPE merges optimize, and why does tokenization affect your cost-per-task numbers?
5. Week 6: state the Eckart–Young theorem in one sentence (best rank-k approximation via SVD).
