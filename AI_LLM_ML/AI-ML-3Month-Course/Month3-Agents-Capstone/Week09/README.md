# Week 09 — Tool Use, Function Calling, Structured Outputs

## Where this sits

You've built RAG, which turns text-in → retrieval → text-out. This week you teach an LLM something fundamentally different: **call code**. The model decides, at generation time, "I need to run this function; here are the arguments." Your harness runs the function, returns the result, and the model continues. Repeat until done.

This capability — tool use, also called function calling — is the single most important building block of agents. It's what separates "chatbot that generates text" from "system that does things". Your ParticleFlow reconstruction, your energy calibration, your event display, your BDT scorer — these are tools. A model that can call them, in sequence, with the right arguments, based on natural language from you, is a productivity multiplier.

By the end of the week you'll have an LLM that, given the question "Plot the η distribution of high-pT jets in this sPHENIX run", writes Python to run against your data and plots it — not by generating standalone code for you to copy-paste, but by actually running tools you expose.

## Learning objectives

By the end of the week you can:

1. Define a tool interface using JSON Schema; understand the Anthropic tool-use protocol (and equivalents in OpenAI, Gemini, local models).
2. Write an agent loop: model call → parse tool call → execute → return result → loop until `stop_reason = "end_turn"`.
3. Define tools for: Python code execution (sandboxed), SQL query (read-only), HTTP fetch, file read, ROOT file inspection.
4. Handle parallel tool calls, tool errors, and retries safely.
5. Enforce structured outputs (JSON mode, structured outputs schema, Pydantic validation).
6. Use Claude / GPT / local models for tool use; understand when local models suffice vs when you need frontier models.
7. Diagnose common failures: tool schema ambiguity, hallucinated function calls, infinite loops, context window exhaustion.

## The mental model

An LLM doesn't actually execute code. It predicts tokens. But if its training data includes many examples of function-call-like JSON, and you expose a grammar for it, it learns to emit that JSON when appropriate.

The loop looks like:

```
1. You call the LLM with: system prompt + user message + list of tools (as JSON schemas).
2. LLM responds with either:
   (a) a plain text answer — done.
   (b) one or more tool_use blocks, each with name + arguments.
3. You dispatch each tool_use to its Python function. You get back a result.
4. You call the LLM again with the prior context + tool_result blocks.
5. Repeat from step 2.
```

The model controls the logic; you control the tools. That's the power and the risk.

## Tool schemas

A tool is described by:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    },
    "required": ["location"]
  }
}
```

### What makes a good tool description

- **Names are nouns or verbs, clear.** `run_python`, `query_runs_db`, `plot_histogram` — not `do_the_thing`.
- **Descriptions describe when to use the tool**, not just what it does. "Use this to run small Python snippets. Do NOT use for long-running computations."
- **Parameters typed strictly.** Enums over free-text for categorical choices. Patterns for regex-constrained strings.
- **Include examples** in descriptions for ambiguous cases.

The model is deciding which tool to call based on descriptions. Bad descriptions = bad decisions. Spend time here.

### Anthropic vs OpenAI protocols

Both use JSON Schema for input. Differences:

- Anthropic: `tool_use` and `tool_result` blocks inside messages. Multiple tool calls per assistant turn possible.
- OpenAI: `tool_calls` field on assistant message; `role: tool` message for results.

Conceptually identical. Semantics: model emits "I want to call tool X with args Y", you run X(Y), you return the result.

## The agent loop (Anthropic flavor)

```python
from anthropic import Anthropic

client = Anthropic()
tools = [...]   # list of tool schemas
messages = [{"role": "user", "content": user_query}]

while True:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system_prompt,
        tools=tools,
        messages=messages,
    )
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn":
        break
    if resp.stop_reason == "tool_use":
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = dispatch(block.name, block.input)
                tool_results.append({"type":"tool_result", "tool_use_id":block.id, "content":result})
        messages.append({"role":"user", "content":tool_results})
        continue
    break  # other stop reasons (max_tokens, refusal, etc.)
```

This is the full agent. Twenty lines. Everything else is tool design.

## Structured outputs

Sometimes you don't want the model to call a tool — you want it to return a specific shape. Two patterns:

### Tool-call as structured output

Define a "tool" that isn't really a tool, just a schema:

```json
{"name": "return_analysis", "input_schema": {...schema for the output you want...}}
```

Call the model with `tool_choice={"type":"tool","name":"return_analysis"}`. The model *must* emit a tool_use matching the schema. Parse the arguments, done.

### Native structured outputs

OpenAI has a `response_format` parameter that enforces a schema (JSON mode, or structured outputs with pydantic). Anthropic has `tool_choice` as above. Either works.

Use Pydantic:
```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float
    evidence_ids: list[str]

# hand the schema to the model via tool_choice
```

Unreliable structured outputs are the #1 source of production pain. This pattern is the fix.

## Tools to build this week

### `run_python(code: str) -> dict`
Execute a Python snippet in a subprocess with a 10-second timeout. Return stdout, stderr, and return code. Sandbox with `subprocess` or, better, a jupyter kernel. Never `exec()` directly — that shares state with your agent process.

### `query_runs_db(sql: str) -> list[dict]`
Read-only SQL against a SQLite database describing run conditions. Whitelist SELECT-only; reject DDL/DML.

### `http_get(url: str) -> str`
Fetch a URL. Whitelist domains. Timeout. Limit response size.

### `inspect_root(path: str) -> dict`
Open a ROOT file (uproot), return list of TTrees, branches, types, entry counts. A single, non-destructive function.

### `plot_histogram(values: list[float], bins: int, title: str) -> str`
Make a histogram plot; return a path to a saved PNG.

Five tools is a full week. Don't over-engineer; design them well and test each one.

## Safety

Tool use is power. Power needs guardrails.

### Whitelist, not blacklist
`run_python` should execute only in a temp dir with no network, no disk outside CWD, no arbitrary imports. Use a subprocess or a jupyter kernel with allowed-imports only.

### Validate inputs before execution
Every tool validates its arguments with Pydantic. If it fails validation, return a structured error; let the model retry.

### Budget per tool call
Each tool has a wall-clock and memory budget. `run_python` gets 10 seconds, 1 GB RAM. Enforced via subprocess limits.

### Audit log
Every tool call + arguments + result is logged with a timestamp. You want to see, after the fact, what the model did.

### Confirmation gates for destructive ops
For anything that writes, deletes, or otherwise affects state beyond the current session, require explicit human confirmation. Anthropic's SDK (and similar) have built-in patterns for this.

## Planning and decomposition

Frontier models (Claude Sonnet, GPT-4-class) are good at planning multi-step tool-use sequences without much prompting. Small open models (7B–13B) often need help:

- **Explicit planning step.** "First, outline the steps you'll take. Then execute them."
- **ReAct-style prompting.** Interleave Thought / Action / Observation. Helpful scaffolding for weaker models.
- **Chain-of-thought hints.** "Think step by step about which tool to call first."

You'll mostly use Claude Sonnet this week for the agent brain; local fine-tuned models struggle with multi-step tool planning.

## Common failure modes

**Model hallucinates a tool that doesn't exist.** Tools list is too long or descriptions are ambiguous. Shorten it or rename.

**Model keeps calling the same tool in a loop.** Your tool's error messages don't give the model enough info to self-correct. Make errors specific and actionable.

**Tool call succeeds but model ignores the result.** The result format is unclear. Return tool results as structured JSON, not free text, when possible.

**Context window exhaustion.** After 20+ tool calls, context is full. Use Claude's extended thinking or summarize intermediate state. Better: scope tasks to fewer steps.

**Two tools with overlapping purposes.** "Call `get_run_info` vs `query_runs_db`." Model picks randomly. Consolidate.

**Tool arguments malformed.** Model emits JSON with wrong key casing. Validate with Pydantic; return a "args invalid: expected 'run_id', got 'runId'" error and let the model retry. Modern models usually self-correct.

**Infinite retries.** Add max step count to your agent loop. 20 is reasonable.

**Prompt injection via tool results.** Untrusted data coming back from `http_get` can contain instructions that the model might obey. Sandbox by prefixing tool results with "BEGIN UNTRUSTED CONTENT" and training/prompting the model to treat it as data, not commands.

## HEP angle

Your actual analysis workflow is tool-use waiting to happen:

- `get_run_conditions(run_id)` → returns calibration constants, detector state.
- `build_skim(runs, cuts)` → launches a condor job, returns job ID.
- `plot_invariant_mass(events, particle_pair)` → returns plot path.
- `fit_peak(histogram)` → returns fit parameters and uncertainties.

An agent wiring these together lets you say "fit the π⁰ peak in run 47000 with default selections, show me the mass and width" and watch it happen.

This week we won't wire the full analysis stack (that's Week 11). We build the tool-use discipline on toy tools.

## Tooling introduced

- `anthropic` Python SDK — primary.
- `openai` Python SDK — alternate protocol for reference.
- Pydantic for tool schemas / result validation.
- `uv` subprocess isolation, `resource` module for limits.
- `jupyter_client` or `ipykernel` for sandboxed code exec (optional, nicer than subprocess).
- `instructor` library — wraps OpenAI/Anthropic for Pydantic-typed outputs.

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | Reading. Simple `get_weather` toy. Understand the loop. |
| 2 | `run_python` with subprocess sandbox. Tests. |
| 3 | `query_runs_db` + schema design. |
| 4 | `http_get` + `inspect_root`. |
| 5 | Compose: multi-tool agent that answers real questions. |
| 6 | Structured outputs with Pydantic. |
| 7 | Harden: error handling, logging, write-up. |

## What "done" looks like

- A small `tools/` package with five well-tested tools.
- An agent loop that runs end-to-end for a queries list, produces traces, and exits cleanly.
- 3 example transcripts: one simple (one tool call), one medium (3–5 tools), one hard (10+ tools).
- A `tool_schemas.json` file you can reuse.
- A 400-word reflection: which tools needed the most iteration? What surprised you?

Week 10 we move from "one tool call at a time" to full agent frameworks with planning, memory, and multi-agent coordination.
