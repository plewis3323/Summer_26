# Week 37 — Tool Use from First Principles

An LLM that can call your fitting code is a different instrument from one that can only talk about it — this week you wire a model directly to functions you wrote in Phase 1.

## Objectives

- Send and parse raw Messages API requests: roles, content blocks, `max_tokens`, `stop_reason`.
- Define a tool as a JSON-schema `input_schema` and explain how the model decides to call it.
- Implement the full tool loop by hand: `stop_reason == "tool_use"` → execute → return `tool_result` with the matching `tool_use_id` → repeat until `end_turn`.
- Handle failure: return `is_error: true` tool results and watch the model recover or re-plan.
- Get schema-guaranteed structured output (Pydantic-validated JSON) instead of regex-parsing prose.

## Core material (~3 hrs)

- Anthropic docs: "Messages API" overview and the "Tool use" overview page (read the tool-definition best practices and the tool-result format carefully).
- Anthropic docs: "Structured outputs" (JSON schema output) — how it differs from prompting for JSON.
- Skim the equivalent OpenAI "function calling" docs to see that every provider ships the same shape: a JSON-schema tool list, an assistant tool-call turn, a tool-result turn. The loop you write this week is provider-agnostic.
- Anthropic docs: prompt-caching page, sections on why tool definitions render first — enough to understand cost, not to optimize yet.

## Exercises (built when the week starts)

1. Raw call: send one Messages API request with the Python SDK, print every field of the response object. Accept when: script prints `stop_reason`, `usage.input_tokens`, `usage.output_tokens`, and all content-block types for one request.
2. First tool: wrap your Phase-1 Gaussian-peak fitter as a `fit_peak` tool (JSON schema: data file, window edges) and let the model call it once. Accept when: transcript shows a `tool_use` block with schema-valid input and a final answer quoting the fitted mean.
3. The loop: hand-write the while-loop that feeds `tool_result` blocks back until `stop_reason == "end_turn"`; give it two tools (`list_data_files`, `fit_peak`). Accept when: a single question ("fit the pi0 peak in the latest run file") triggers both tools in sequence without manual intervention.
4. Error recovery: make `fit_peak` raise on a bad window; return the message with `is_error: true`. Accept when: the model retries with a corrected window and the loop terminates with a valid fit.
5. Parallel calls: ask a question needing three independent fits; return all `tool_result` blocks in one user message. Accept when: transcript shows ≥2 `tool_use` blocks in a single assistant turn and correct final numbers.
6. Structured output: extract `{particle, mass_mev, width_mev, n_events}` from a paragraph of analysis prose using a schema, not prompting. Accept when: 10/10 test paragraphs parse into the schema with correct fields (checked against hand labels).
7. Cost ledger: log tokens and dollars per exercise run. Accept when: a printed table shows cost per exercise and total week cost under a stated budget.

## Deliverable

`week37/` package: `tools.py` (schemas + implementations), `loop.py` (the agentic loop, tested with a mocked client), transcripts of exercises 3–5, and the cost table.

## Review

1. Derive the softmax temperature's effect on sampling entropy (Week 29) — why does T→0 make tool-call arguments more repeatable?
2. Week 10: you tuned a decision threshold on ROC curves. What is the analog of a "threshold" in deciding whether a model should call a tool vs. answer directly?
3. Write the cross-entropy loss for next-token prediction (Week 13/27) and state what the labels are.
4. Week 34: what did recall@k measure in your RAG evals? Propose the analogous metric for "did the agent call the right tool."
5. Week 3: why do tests on the tool functions matter more than tests on the prompt?
