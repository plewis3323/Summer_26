# Week 40 — Exercises

Work top to bottom. Setup (imports, `client`, the Week 37 `run_agent`, the
Week 39 `mcp_server` package on the path with its toy data / `calibrations.json`
/ `run_index.json`, hand-computed reference fits for every run file, and the
`ask_logged` / `LEDGER` cost helper) is given by the notebook. Nearly all of
this week's work lives in files under `week40/copilot/`, because the acceptance
criteria are scripts, `pytest`, and a CLI (NOTEBOOK_RULES §6) — the notebook
only launches them and checks outputs. E5 is the mini-project; its full spec,
including the acceptance gate, is `project.md` — read it before starting E5.
Every model call goes through the ledger; these exercises cost real money, so
set a budget as in Week 37 (a few dollars covers the week if your loops are
leashed).

## E1 — Handoff schema (`handoffs.py`)

Define the worker **task brief** and **worker report** as JSON Schemas
(lesson §2). Implement `run_worker(brief)` (Week 37 loop + schema-enforced
report) and an orchestrator that fans out 3 fit tasks — one per run file — and
merges the reports into a table. Compare the merged table against running the
same three fits through a single Week 37 agent conversation.
Hint: validate every report before merging (required keys, types); treat a
non-validating report as a failed task, not a crash. Log both runs through
`ask_logged` and look at where the multi-agent tokens went.
Accept when: all worker reports validate against the schema and the merged
table matches single-agent results.

## E2 — Task suite (`tasks.json` + `run_suite.py`)

Write 10 analysis tasks with programmatically checkable success criteria (e.g.
"reported mass within 2 MeV of reference fit"), spanning easy (count the run
files) to multi-tool (fit the latest *good* run). References come from calling
your tool functions directly. `run_suite.py` takes any answer-producing
function, runs all 10 tasks, applies the checks, and prints per-task pass/fail
plus totals.
Hint: lesson §5's `check_answer` dispatch covers most criteria; if you cannot
write the check, rewrite the task.
Accept when: `run_suite.py` scores any agent 0–10 with no human in the loop.

## E3 — Baseline vs. agent (`compare.py`)

Run the suite against (a) your Week-38 fixed workflow and (b) the Week 37 agent
loop. Collect success rate, mean cost per task (from the ledger and current
prices), and mean latency for both.
Hint: `time.time()` around each task; tasks the workflow cannot attempt count
as failures for it — say so in a footnote row rather than skipping them.
Accept when: a table reports success rate, mean cost, and mean latency for
both.

## E4 — Trajectory review (`failures.md` + `transcripts/`)

Dump full transcripts for 3 failing tasks (from E3 or the copilot; if fewer
than 3 fail, add harder tasks until they do). For each, mark the first wrong
step and assign a root cause from the lesson §7 taxonomy, with the fix it
implies.
Hint: read each transcript as Thought → Action → Observation (Week 38 §7) and
stop at the first action that does not follow from the observations before it.
Accept when: `failures.md` names a root cause per failure (bad tool
description, missing tool, model error, bad task spec).

## E5 — Copilot mini-project (`week40/copilot/`)

Wire the loop + MCP server + a thin CLI (`copilot "question"`) into one
package, per `project.md` — the async glue from lesson §4, the Week 39 server
launched as a subprocess, transcripts saved per run.
Hint: build the glue function first and test it from a plain script before
touching packaging; the CLI is a ten-line wrapper around it.
Accept when: fresh clone + `uv sync` + one command answers 2 scripted demo
questions end-to-end. (The project's full acceptance gate — 3 scripted tasks
unattended — is checked in `project.md`.)

## E6 — Eval the copilot (`EVAL.md`)

Run the E2 task suite on the copilot. Write `EVAL.md`: success rate (as a
fraction, with the small-n caveat), mean cost per task, mean latency, and one
prioritized improvement.
Hint: the improvement must cite a specific trajectory — "t07 failed because the
`get_calibration` description never says runs are integers" — not a vibe.
Accept when: `EVAL.md` contains the numbers and the improvement is justified by
a specific trajectory.

## Review

1. Week 09: define generalization error. What is the analog for an agent
   evaluated on a 10-task suite it was developed against?
2. Week 32: give two ways an LLM-as-judge can be systematically biased, and how
   you'd detect each in trajectory grading.
3. Week 11: why can feature importances lie? Connect this to trusting an
   agent's self-reported reasoning about which tool it used and why.
4. Week 27: what did BPE merges optimize, and why does tokenization affect your
   cost-per-task numbers?
5. Week 06: state the Eckart–Young theorem in one sentence (best rank-k
   approximation via SVD).
