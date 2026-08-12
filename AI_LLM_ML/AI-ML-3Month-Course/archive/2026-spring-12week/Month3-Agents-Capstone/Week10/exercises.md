# Week 10 — Exercises

## E1 — LangGraph "hello graph" (easy, 30 min)

Implement a 2-node state graph: `plan_node` → `respond_node`. State has `input: str`, `plan: str`, `answer: str`. Plan node calls the LLM to produce a 3-step plan. Respond node calls the LLM to answer using the plan.

**Accept when:** you can invoke `app.invoke({"input": "What's 2+2?"})` and see both nodes execute in order.

## E2 — LangGraph with tool use (medium, 75 min)

Reimplement your Week 09 agent loop in LangGraph:
- `agent_node`: calls model with tools; returns either a terminal response or a list of tool calls.
- `tools_node`: dispatches tool calls, appends results to messages.
- Conditional edge: if the last message has tool calls → `tools_node`; else → END.
- Use `langgraph.prebuilt.create_react_agent` first to understand the idiom; then build it yourself.

**Accept when:** the graph answers "How many good AuAu runs in 2024?" correctly using your `query_runs_db` tool.

## E3 — Checkpointing and resumption (medium, 45 min)

Add `SqliteSaver` checkpointer:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(conn))
```

Run a task partway, inspect state, resume from a specific checkpoint. Demonstrate that you can branch: run the agent, save, then ask two follow-up questions from the same checkpoint independently.

## E4 — smolagents port (medium, 60 min)

Wrap your Week 09 tools with smolagents' `@tool` decorator. Use `CodeAgent` with a model (HfApi, or via `LiteLLMModel` to point at Claude).

```python
from smolagents import CodeAgent, LiteLLMModel, tool

@tool
def query_runs_db(sql: str) -> list:
    """Run a read-only SELECT on the runs database. Columns: run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events, quality."""
    ...

agent = CodeAgent(tools=[query_runs_db], model=LiteLLMModel("anthropic/claude-sonnet-4-5"))
agent.run("Show mean events per good AuAu run in 2024.")
```

Compare the transcript to your LangGraph run. Notice: smolagents' agent writes Python code that calls `query_runs_db` in a loop/list-comprehension. LangGraph agent calls `query_runs_db` once, reads result, calls `run_python` if needed. Tradeoffs are real.

## E5 — Claude Agent SDK port (medium, 45 min)

```python
from claude_agent_sdk import Agent, tool  # exact import path may differ — check docs

@tool
def query_runs_db(sql: str) -> list: ...

agent = Agent(model="claude-sonnet-4-5", tools=[query_runs_db], system="...")
out = agent.run("How many good AuAu runs in 2024?")
```

Compare against raw SDK loop from Week 09. Count lines, count knobs, count features.

## E6 — Golden task set (medium, 90 min)

Build `goldens.jsonl` with 20 tasks + success criteria. Examples:

```json
{"id": "g1", "input": "Count all good AuAu runs in 2024.", "check": "contains_number: 70..72"}
{"id": "g2", "input": "Plot a histogram of jet_pt from run_47000. Save to hist.png.", "check": "file_exists: hist.png AND file_size > 10kb"}
{"id": "g3", "input": "What's the mean energy across all runs?", "check": "contains_number: 195..205"}
```

Write `eval.py` that runs all 3 framework implementations on the golden set and reports success rate, mean cost, mean latency, mean tool calls.

**Accept when:** you have a table like:
| Framework | Success | Mean cost | Mean latency | Mean calls |
| --- | --- | --- | --- | --- |
| raw SDK | 18/20 | $0.004 | 12s | 4.2 |
| LangGraph | 19/20 | $0.005 | 14s | 4.5 |
| smolagents | 17/20 | $0.003 | 9s | 2.8 |

## E7 — Tracing with Weave (easy, 30 min)

```python
import weave
weave.init("run-analyst-evals")

@weave.op
def run_one(agent, task):
    return agent.run(task["input"])

for t in goldens:
    run_one(agent, t)
```

Check the Weave UI — you should see every call, with inputs, outputs, latencies, and costs.

Write 150 words on what Weave gives you vs. printing trace to stdout.

## E8 — Session memory (medium, 60 min)

Build a session-memory layer:

```python
class SessionMemory:
    def __init__(self, path="memory.json"):
        self.path = Path(path); self.data = []
        if self.path.exists(): self.data = json.loads(self.path.read_text())

    def add(self, task, trace, result):
        summary = llm_summarize(task, trace, result)
        self.data.append({"ts": time.time(), "summary": summary})
        self.path.write_text(json.dumps(self.data))

    def context_for(self, new_task, k=3):
        return "\n".join(m["summary"] for m in self.data[-k:])
```

Integrate into your agent: system prompt includes recent memories. Demonstrate:
- Task 1: "I care about runs 47000-47099. My cuts are jet_pt > 10 GeV."
- Task 2 (new invocation): "Plot jet_pt distribution." — agent applies remembered context.

## E9 — Long-term memory with vector store (medium, 75 min)

Add a long-term layer using your Week 08 infrastructure:
- Write each session summary as a document.
- At task start, embed the task, retrieve top-k relevant past summaries, prepend.

Demonstrate it helps on a task that requires memory from several sessions back.

## E10 — Multi-agent pattern (hard, 90 min)

Build a 3-agent system:
- `manager`: decomposes user task into subtasks.
- `sql_worker`: answers SQL questions only; has `query_runs_db`.
- `code_worker`: runs Python / reads ROOT; has `run_python`, `inspect_root`.

Manager dispatches each subtask to the right worker, collects results, responds.

Use LangGraph's supervisor pattern or build your own. Test on: "For run_47000, how many total events, how many jets with pT > 5 GeV, what fraction in good runs?"

Compare against single-agent implementation. Does multi-agent help? In what ways? Any ways it hurts?

## E11 — Reflexion-style self-critique (medium, 60 min)

After the first answer, add a critic step that asks: "Is this answer grounded? Is it complete? Any numeric errors?" If critic finds issues, agent revises.

Test on 5 tricky tasks. Measure whether critic improves success rate or just adds latency.

## E12 — Framework comparison write-up (medium, 90 min)

Write `framework_comparison.md` (~1000 words) covering:

1. Architecture summary (3-4 sentences per framework).
2. Lines of code for the run-analyst in each.
3. Eval numbers from E6.
4. Developer experience notes: debuggability, docs quality, time-to-first-working-agent.
5. When to pick which.
6. Your recommendation for the Week 12 capstone.

---

## Solution hints

- E1 — if your nodes don't run, you forgot `g.set_entry_point("plan")` or `g.add_edge(START, "plan")`.
- E2 — `create_react_agent(llm, tools)` gives you a working one-liner. Then reverse-engineer what it built.
- E3 — `thread_id` is the checkpoint key. Use different thread_ids for independent branches.
- E4 — smolagents uses LiteLLM under the hood. Point at Claude with `"anthropic/claude-sonnet-4-5"`.
- E5 — the Agent SDK is thin over the raw SDK. If features feel the same, that's by design.
- E6 — use regex/constraint checks for automated grading. LLM-as-judge is fine but slower.
- E7 — Weave captures LLM calls even without decorators if you import `weave` before `anthropic`.
- E8 — summarize prompts need tuning. Too-long summaries blow context; too-short lose detail.
- E9 — don't index every summary blindly; dedupe by cosine similarity.
- E10 — supervisor prompt is 80% of the battle. Spend time on it.
- E11 — critic models are often just expensive text reformulation. Measure, don't assume.
- E12 — be fair. Each framework was built for slightly different target use cases.
