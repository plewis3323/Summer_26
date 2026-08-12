# Week 38 — Agent Patterns

Last week you built the mechanism; this week you learn the architectures — and, like choosing between a cut-based analysis and a BDT, the first question is whether the fancy tool is warranted at all.

## Objectives

- State the workflow-vs-agent distinction: code-orchestrated steps vs. model-directed looping, and the cost/latency/error tradeoffs of each.
- Implement the five patterns from *Building Effective Agents*: prompt chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer.
- Explain ReAct (interleaved Thought → Action → Observation) and use it as a transcript-debugging lens on the Week 37 loop.
- Apply a written decision checklist (complexity, value, viability, cost of error) to decide when NOT to build an agent.
- Instrument any pattern with per-step token/cost accounting.

## Core material (~3.5 hrs)

- `lesson.md` (this folder) — the primary text: all five patterns with runnable sketches on the π⁰ toy data, ReAct, the no-agent checklist, and cost instrumentation.
- Anthropic, *Building Effective Agents* (essay) — read it twice; the second time, list which of your Phase 1–3 pipelines were workflows.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* — the thought/action/observation format sections; skim the benchmarks.
- Optional: Anthropic's *Building Effective Agents* companion cookbook, by title, for reference implementations to compare against your own.

## Exercises

See `exercises.md` (built into the notebook when the week starts). One runnable
script per pattern under `week38/patterns/` — chaining with gates, a measured
router, majority-vote extraction, orchestrator–workers on the run files,
evaluator–optimizer with a real rejection cycle — capped by a workflow-vs-agent
shoot-out with a token/latency/correctness table and a written ship/no-ship verdict.

## Deliverable

`week38/patterns/` — one runnable script per pattern sharing the `ask_logged` cost instrumentation, plus `DECISION.md`: the agent/no-agent checklist applied to three tasks from your own research workflow.

## Review

1. Week 08: momentum vs. Adam — which parts of Adam's update are per-parameter, and why does that matter for sparse gradients?
2. Week 25: derive why attention scores are scaled by 1/√d.
3. The evaluator–optimizer loop is a feedback controller. Week 31: where exactly does the reward model sit in the RLHF pipeline, and what plays the "evaluator" role there?
4. Week 12: in EM for GMMs, what quantity is guaranteed non-decreasing each iteration? Does your evaluator–optimizer loop have any such guarantee?
5. Week 04: what made an analysis "reproducible" in your Month-1 project, and which of this week's patterns is hardest to make reproducible?
