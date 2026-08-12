# Week 37 — Tool Use from First Principles

An LLM that can call your fitting code is a different instrument from one that can only talk about it — this week you wire a model directly to functions like the ones you wrote in Phase 1. First time calling a remote service from code: the lesson teaches HTTP, JSON, and API-key hygiene from zero.

## Objectives

- Explain what an HTTP API, JSON, and an API key are, and keep the key out of your code.
- Send and parse Messages API requests: roles, content blocks, `max_tokens`, `stop_reason`, `usage`.
- Define a tool as a JSON-Schema `input_schema` and explain how the model decides to call it.
- Implement the full agentic loop by hand: `stop_reason == "tool_use"` → execute → return `tool_result` with the matching `tool_use_id` → repeat until done.
- Handle failure: return `is_error: true` tool results and watch the model recover.
- Get schema-guaranteed structured output instead of regex-parsing prose.
- Track tokens and dollars per call.

## Core material (~4 hrs)

- `lesson.md` (this folder) — the primary text: HTTP/JSON/keys from zero, the tool contract, the loop, structured output, worked two-tool example.
- Anthropic docs: "Messages API" overview and the "Tool use" overview page (read the tool-definition best practices and the tool-result format carefully). The API evolves — the docs win over any lesson snippet.
- Anthropic docs: "Structured outputs" — how schema-enforced output differs from prompting for JSON.
- Skim any OpenAI "function calling" guide to see that every provider ships the same shape: a JSON-schema tool list, a tool-call turn, a tool-result turn. The loop you write this week is provider-agnostic.

## Exercises

See `exercises.md` (built into the notebook when the week starts). Seven exercises,
easy → hard: one raw API call with every field printed, then wrapping your Gaussian
peak fitter as a `fit_pi0_peak` tool, the hand-written agentic loop with two tools,
error recovery via `is_error`, parallel tool calls, schema-enforced extraction, and a
cost ledger totaling the week's spend.

## Deliverable

`week37/` package: `tools.py` (schemas + implementations), `loop.py` (the agentic loop), saved transcripts of E3–E5, and the printed cost table under budget.

## Review

1. Week 29: derive the softmax temperature's effect on sampling entropy — why does T → 0 make tool-call arguments more repeatable?
2. Week 10: you tuned a decision threshold on ROC curves. What is the analog of a "threshold" in deciding whether a model should call a tool vs. answer directly?
3. Write the cross-entropy loss for next-token prediction (Week 27) and state what the labels are.
4. Week 34: what did recall@k measure in your RAG evals? Propose the analogous metric for "did the agent call the right tool."
5. Week 04: why do tests on the tool functions matter more than tests on the prompt?
