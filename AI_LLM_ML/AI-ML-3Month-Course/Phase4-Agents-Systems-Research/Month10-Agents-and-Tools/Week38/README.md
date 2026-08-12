# Week 38 — Agent Patterns

Last week you built the mechanism; this week you learn the architectures — and, like choosing between a cut-based analysis and a BDT, the first question is whether the fancy tool is warranted at all.

## Objectives

- State the workflow-vs-agent distinction: code-orchestrated steps vs. model-directed looping, and the cost/latency/error tradeoffs of each.
- Implement the five patterns from *Building Effective Agents*: prompt chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer.
- Implement a ReAct loop (interleaved reasoning and tool calls) on top of your Week-37 machinery and explain how it differs from a plain tool loop.
- Apply a written decision checklist (task complexity, value, viability, cost of error) to decide when NOT to build an agent.
- Instrument any pattern with per-step token/cost accounting.

## Core material (~3 hrs)

- Anthropic, *Building Effective Agents* (essay) — the primary text; read it twice, the second time listing which of your Phase 1–3 pipelines were workflows.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (paper) — sections on the thought/action/observation format; skim the benchmarks.
- Anthropic docs: agent-design guidance on tool surfaces (when to promote an action from "bash-like generic tool" to a dedicated typed tool).
- Optional: Anthropic's "Building Effective Agents" companion code/cookbook examples, by title, for reference implementations.

## Exercises (built when the week starts)

1. Prompt chaining: abstract → (extract observables) → (standardize units) → (summary table), three fixed calls with a validation gate between steps. Accept when: 8/10 heavy-ion abstracts produce a table passing all gate checks.
2. Routing: classify incoming questions into {lookup, computation, fit-request} and dispatch to different handlers. Accept when: routing accuracy ≥ 90% on a 20-question labeled set.
3. Parallelization: run the same extraction prompt at 3 variants and majority-vote the field values. Accept when: voted output beats any single variant's field accuracy on your test set.
4. Orchestrator–workers: an orchestrator decomposes "compare pi0 yields across these 4 run files" into worker fit tasks and merges results. Accept when: final table matches a hand-computed reference within stated fit errors.
5. Evaluator–optimizer: a writer drafts a run-summary paragraph, an evaluator scores it against a rubric, loop until pass or 3 iterations. Accept when: logs show ≥1 rejection-then-improvement cycle and a final rubric pass.
6. ReAct vs. workflow: solve exercise 4's task once with the ReAct agent, once with the fixed workflow; compare tokens, wall time, and correctness. Accept when: a short table + 5-sentence verdict states which you'd ship and why.

## Deliverable

`week38/patterns/` — one runnable script per pattern with shared cost instrumentation, plus `DECISION.md`: your agent/no-agent checklist applied to three tasks from your own research workflow.

## Review

1. Week 8: momentum vs. Adam — which parts of Adam's update are per-parameter, and why does that matter for sparse gradients?
2. Week 25: derive why attention scores are scaled by 1/√d.
3. The evaluator–optimizer loop is a feedback controller. Week 31: where exactly does the reward model sit in the RLHF pipeline, and what plays the "evaluator" role there?
4. Week 12: in EM for GMMs, what quantity is guaranteed non-decreasing each iteration? Does your evaluator–optimizer loop have any such guarantee?
5. Week 4: what made an analysis "reproducible" in your Month-1 project, and which of this week's patterns is hardest to make reproducible?
