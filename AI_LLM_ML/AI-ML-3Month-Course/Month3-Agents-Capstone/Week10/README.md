# Week 10 — Agent Frameworks: LangGraph, smolagents, and Claude's Agent SDK

## Where this sits

Last week you wrote an agent loop from scratch — you know exactly what the tokens look like, what "parallel tool calls" means, and what it takes to handle errors cleanly. That grounding is valuable. Frameworks are welcome abstractions, not magic; this week you decide *which* abstraction is worth adopting.

You'll look at three frameworks, each with a distinct philosophy:

- **LangGraph.** State-machine graph where nodes are LLM calls or tool calls, edges are conditional. Powerful for complex flows and multi-agent coordination. Opinionated; lots to learn.
- **smolagents** (HuggingFace). Minimalist. Agents write **Python code** to call tools, not JSON. Elegant for tasks where the agent needs to compose tools in ways you didn't pre-specify.
- **Anthropic's Agent SDK (claude-agent-sdk).** Thin wrapper over the raw SDK with primitives for multi-step reasoning, memory, and tool orchestration. Good for "I like the raw SDK but want less boilerplate."

By the end, you'll have re-implemented your Week 09 run-analyst agent in **two** frameworks and a third hand-rolled version, and written a 1000-word comparison to guide your capstone choice in Week 12.

## Learning objectives

By the end of the week you can:

1. Explain the architectural differences between LangGraph's graph model, smolagents' code-execution model, and raw SDK loops.
2. Implement a non-trivial agent in LangGraph using state, nodes, conditional edges, and checkpointing.
3. Implement the same agent in smolagents and explain when the "code-writing agent" wins.
4. Design agent memory: short-term (scratchpad), long-term (vector store), working-memory eviction.
5. Handle multi-agent orchestration: manager + worker agents, handoffs, concurrent execution.
6. Use tracing (LangSmith, Weave, Phoenix) to debug non-deterministic agent runs.
7. Evaluate agents: offline traces, golden tasks, cost and latency, reliability over retries.

## Why use a framework at all?

Frameworks buy you:

- Persistent state across steps (don't re-assemble messages every loop).
- Checkpointing: pause, resume, fork.
- Memory layers: scratchpads, vector stores, conversation history summarization.
- Tracing: what happened, when, how much did it cost.
- Multi-agent patterns: manager/worker, hand-off, parallel branches.
- Integration with model providers, vector DBs, observability tools.

They cost you:

- A learning curve. LangGraph's DSL takes a weekend.
- Magic. Sometimes you can't tell why the agent did X.
- Dependency weight. LangChain alone pulls in a large tree.
- Versioning churn. These libraries move fast.

Rule of thumb: for a one-shot single-tool task, don't reach for a framework. For any durable, multi-step, stateful agent, a framework pays back quickly. Your capstone is the latter.

## LangGraph in a nutshell

LangGraph models an agent as a **state graph**. State is a Python TypedDict (or Pydantic model). Nodes are functions that read state and return state updates. Edges can be conditional.

Canonical shape:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    step: int

def plan_node(state):
    # call LLM to produce a plan
    return {"plan": "..."}

def execute_node(state):
    # dispatch tools based on plan
    return {"messages": [...]}

def should_continue(state):
    return "execute" if state["step"] < len(state["plan"]) else END

g = StateGraph(State)
g.add_node("plan", plan_node)
g.add_node("execute", execute_node)
g.add_edge("plan", "execute")
g.add_conditional_edges("execute", should_continue)
app = g.compile(checkpointer=MemorySaver())
```

Strengths:
- Clear state model. You can inspect state at any step.
- Checkpointing (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) gives you resume.
- Streaming supported.
- Plays well with LangChain's ecosystem of tools, retrievers, memories.

Weaknesses:
- The DSL is a lot to learn.
- Debugging conditional-edge bugs is harder than stepping through a plain loop.

Best for: production agents with non-trivial control flow, multi-agent coordination, human-in-the-loop steps.

## smolagents in a nutshell

HuggingFace's smolagents has a contrarian idea: instead of forcing the LLM to emit JSON for tool calls, let it **write Python code** that uses tools as regular functions. The agent runtime runs the code in a sandboxed Python interpreter.

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def query_runs_db(sql: str) -> list:
    """Run a read-only SELECT on the runs database."""
    ...

@tool
def plot_hist(values: list, bins: int = 50, path: str = "plot.png") -> str:
    """Plot a histogram."""
    ...

agent = CodeAgent(tools=[query_runs_db, plot_hist], model=HfApiModel())
agent.run("Plot a histogram of jet_pt for all good AuAu runs in 2024.")
```

Under the hood, the LLM writes Python that uses `query_runs_db` and `plot_hist` as functions — it can use loops, conditionals, comprehensions, numpy. For tasks that naturally decompose into Python, this is much more expressive than emitting one tool call per step.

Strengths:
- Elegant for data-munging tasks (your domain).
- Fewer tool-call round trips — the LLM can do loops in one pass.
- Fits HuggingFace ecosystem.

Weaknesses:
- Python sandboxing is harder to get right than JSON dispatch.
- Can generate subtly wrong code that silently miscounts.
- Newer; smaller community than LangChain.

Best for: data-analysis agents, physics/scientific computing agents, agents whose task is "compose these primitives in a novel way".

## Anthropic Agent SDK in a nutshell

`claude-agent-sdk` is Anthropic's lightweight agent helper. It provides:
- An opinionated loop with sensible defaults.
- Built-in handling for parallel tool calls.
- Patterns for extended thinking.
- Easy integration with MCP servers (next week).

```python
from claude_agent_sdk import Agent, tool

@tool
def query_runs_db(sql: str) -> list: ...
@tool
def run_python(code: str) -> dict: ...

agent = Agent(
    model="claude-sonnet-4-5",
    system="You are a sPHENIX run analyst...",
    tools=[query_runs_db, run_python],
)
result = agent.run("How many good AuAu runs in 2024?")
```

Strengths:
- Matches the raw Anthropic API; minimal new concepts.
- Native MCP integration (critical next week).
- Less magic than LangGraph; more than your hand-rolled loop.

Weaknesses:
- Less structured than LangGraph for complex flows.
- Claude-only.

Best for: Claude-native agents, MCP-heavy pipelines, production simplicity.

## Memory

Agents need memory beyond the current conversation. Three layers:

### Short-term (scratchpad)
Notes the agent writes during a single task. Included in subsequent prompts. Can be structured (key-value store) or free text.

### Session memory
Persists across multiple task-turns within a session. Stored as a growing list of messages or a summary.

### Long-term
Persists across sessions. Typically a vector store. New memories are embedded + indexed; old ones are retrieved when relevant.

Simple recipe for session memory:
```python
# after each task
summary = client.messages.create(..., messages=[
    {"role": "user", "content": f"Summarize: {full_trace}"},
]).content[0].text

memory_store.append({"timestamp": ..., "summary": summary})
```

At next task start, retrieve recent summaries; include in system prompt.

## Multi-agent patterns

### Manager + workers
A "manager" agent plans, delegates subtasks to specialized "worker" agents (e.g. `data_worker`, `plot_worker`), aggregates their results.

### Hierarchical
Managers of managers. Rare outside research demos.

### Swarm / debate
Multiple agents critique each other's answers. Sometimes improves accuracy on complex reasoning; often expensive.

### Parallel
Multiple agents run independently on different chunks of work, results merged.

Keep it simple. The default should be a single-agent loop with good tools. Reach for multi-agent only when the task genuinely decomposes and no single agent prompt can cover it.

## Tracing

You need to see what the agent did. Options:

- **LangSmith.** Polished UI, integrates with LangChain/LangGraph out of the box. Paid tier for teams.
- **Weave (Weights & Biases).** OSS, integrates with anything; nice for "compare these 10 runs".
- **Phoenix (Arize).** OSS, self-hosted, strong for OpenAI/Anthropic traces.
- **OpenTelemetry.** Framework-agnostic. Send spans to Jaeger, Honeycomb, whatever.

You'll use one this week. Start with Weave — open-source, simple, plays with everything.

## Evaluation

Agent evaluation is harder than eval of any single ML system:

- Outputs are stochastic; rerun and see different traces.
- Long horizons mean errors compound.
- Metrics like "did it succeed" are binary but miss nuance.
- Cost varies per run.

Practical approach:
1. Build a **golden task set.** 20 tasks with known-good outputs (or outputs you can verify programmatically).
2. Run the agent on each N times (N=5 is fine) with different seeds. Measure:
   - Success rate (task passed / total).
   - Mean cost.
   - Mean latency.
   - Mean tool calls.
3. Track over time as you iterate.

## Common failure modes

**Framework fights your tools.** Your tool signatures need to match what the framework expects. LangChain's `@tool` decorator has subtly different semantics from OpenAI's JSON Schema. Read docs carefully.

**State-graph deadlocks.** Conditional edge returns wrong next node; graph loops forever. Log state at every transition; use `max_iterations`.

**Memory leaks.** Every task adds to memory; never evicted. Implement summarization + eviction.

**Agent trying to do the wrong thing "cleverly".** Usually means your system prompt is too permissive. Tighten scope: "You are a summarizer. Do not suggest analysis. Do not call non-summarizer tools."

**Multi-agent coordination overhead dominates value.** Agents spend more time deliberating than doing. Fall back to single agent.

**Trace data exceeds budget.** Turn on sampling; log only first N steps + last N steps of long traces.

## HEP angle

You're building toward a capstone HEP copilot (Track A in Week 12). The skills this week map:

- LangGraph for an analysis pipeline with branches (one branch for run QC, another for physics analysis).
- smolagents for exploratory analysis where you don't know the tool chain in advance.
- Claude Agent SDK for tight, production-simple pipelines.
- Memory: "remember that I'm analyzing runs 46xxx, my cuts are pT > 5 GeV, I only care about central AuAu".
- Multi-agent: one worker per detector subsystem (EMCal, HCal, tracker) if each has specialized tools.

## Tooling introduced

- `langgraph`, `langchain`, `langchain-anthropic`, `langchain-openai`.
- `smolagents`, `huggingface-hub` (you have these already).
- `claude-agent-sdk`.
- `weave` for tracing.
- `langsmith` (optional, requires account).

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | LangGraph tutorial; port Week 09 agent to LangGraph. |
| 2 | LangGraph with state, checkpointing, conditional edges. |
| 3 | smolagents port. |
| 4 | Claude Agent SDK port. |
| 5 | Memory (session + long-term). |
| 6 | Multi-agent experiment. Tracing. |
| 7 | Comparison write-up. |

## What "done" looks like

- Three implementations of the run-analyst: LangGraph, smolagents, Agent SDK. All pass the same 5-question test suite from Week 09.
- Tracing running for at least one of them; linked traces in the write-up.
- A golden task set of 20 tasks; per-framework success rates.
- A 1000-word comparison: framework tradeoffs, when to reach for which.
- One multi-agent experiment: a manager + 2 worker agents. Report whether it helped vs single-agent.
- Memory: session memory implemented with summarization; demonstrated to persist across CLI invocations.

Week 11 we layer MCP on top: your tools become standard MCP servers that any MCP-compatible client (Claude Code, Claude Desktop, Cursor, etc.) can use. That's where your agent starts integrating with your broader workflow.
