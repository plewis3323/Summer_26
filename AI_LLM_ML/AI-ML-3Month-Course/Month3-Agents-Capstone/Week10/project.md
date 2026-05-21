# Week 10 — Mini-Project: Same Agent, Three Frameworks

## Problem

Re-implement your Week 09 run-analyst agent in (a) LangGraph, (b) smolagents, and (c) Anthropic's Agent SDK. Evaluate all three on a 20-task golden set. Add session memory + long-term memory. Produce a 1000-word framework comparison with eval numbers backing your recommendation for the Week 12 capstone.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/analyst/
│   ├── __init__.py
│   ├── tools/                   # shared tools from Week 09
│   │   ├── python_exec.py
│   │   ├── sql.py
│   │   ├── http.py
│   │   ├── root_inspect.py
│   │   └── plot.py
│   ├── memory/
│   │   ├── session.py           # SQLite-backed session memory
│   │   └── longterm.py          # vector-store memory
│   ├── frameworks/
│   │   ├── raw_sdk.py           # from Week 09, cleaned
│   │   ├── langgraph_impl.py
│   │   ├── smolagents_impl.py
│   │   └── agent_sdk_impl.py
│   ├── eval/
│   │   ├── goldens.jsonl
│   │   ├── runner.py
│   │   └── judge.py             # checkers + LLM-judge fallback
│   └── cli.py
├── tests/
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_langgraph_impl.py
│   ├── test_smolagents_impl.py
│   └── test_agent_sdk_impl.py
├── results/
│   ├── traces/
│   ├── eval_table.md
│   ├── cost_latency_chart.png
│   └── framework_comparison.md
```

## Acceptance criteria

1. `uv run pytest -q` passes all unit tests (tools + memory) and integration tests (each framework on ≥ 3 goldens).
2. All three framework implementations accept the same tool list and pass the same 5 Week-09-style questions.
3. 20-task golden set. Each framework scored. Table in `results/eval_table.md` with success rate, cost, latency, mean tool calls.
4. Session memory: works across CLI invocations. A fixed test scenario demonstrates remembered context.
5. Long-term memory: vector-store-backed; ≥ 10 summaries indexed after running the golden set.
6. One multi-agent experiment (manager + sql_worker + code_worker). Report whether it improved success rate or not.
7. Tracing: at least one framework instrumented with Weave (or LangSmith). Permalinks or screenshots in the write-up.
8. `framework_comparison.md` of ~1000 words with architecture, DX notes, numbers, and a recommendation.
9. README instructs anyone to reproduce: which model, env vars, uv commands, how to regenerate goldens.

## Suggested plan (6–7 days)

- **Day 1.** Import Week 09 tools. Refactor into shared package. Tests.
- **Day 2.** LangGraph implementation. Checkpointing.
- **Day 3.** smolagents implementation. Side-by-side CLI invocation.
- **Day 4.** Agent SDK implementation. Goldens drafted.
- **Day 5.** Eval runner. Weave tracing. First full eval pass.
- **Day 6.** Memory layers. Multi-agent experiment.
- **Day 7.** Comparison write-up. Cleanup.

## Implementation sketches

### LangGraph

```python
# src/analyst/frameworks/langgraph_impl.py
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool as lc_tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

# wrap each of your tools with @lc_tool
@lc_tool
def query_runs_db(sql: str) -> str:
    """Run a read-only SELECT on the runs database. Columns: run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events, quality."""
    from analyst.tools.sql import query_runs_db as impl
    return json.dumps(impl(sql))

# ... others ...

tools = [query_runs_db, run_python, inspect_root, http_get]
llm = ChatAnthropic(model="claude-sonnet-4-5").bind_tools(tools)

def agent(state): return {"messages": [llm.invoke(state["messages"])]}
def route(state) -> Literal["tools", END]:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", route)
g.add_edge("tools", "agent")
app = g.compile()

def run(input_str: str) -> str:
    out = app.invoke({"messages": [("user", input_str)]})
    return out["messages"][-1].content
```

### smolagents

```python
# src/analyst/frameworks/smolagents_impl.py
from smolagents import CodeAgent, LiteLLMModel, tool
from analyst.tools.sql import query_runs_db as _qdb
from analyst.tools.root_inspect import inspect_root as _inspect

@tool
def query_runs_db(sql: str) -> list:
    """SELECT-only on runs table. Columns: run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events, quality."""
    return _qdb(sql)

@tool
def inspect_root(path: str) -> dict:
    """Inspect a ROOT file and list its trees + branches."""
    return _inspect(path)

# note: smolagents' CodeAgent doesn't need a separate run_python — the agent *is* the python runtime.
agent = CodeAgent(
    tools=[query_runs_db, inspect_root],
    model=LiteLLMModel("anthropic/claude-sonnet-4-5"),
    max_iterations=10,
    additional_authorized_imports=["uproot", "numpy", "pandas", "matplotlib.pyplot"],
)

def run(input_str: str) -> str:
    return agent.run(input_str)
```

### Agent SDK

```python
# src/analyst/frameworks/agent_sdk_impl.py
from claude_agent_sdk import Agent, tool as agent_tool
from analyst.tools import sql as _sql, python_exec as _py, http as _http, root_inspect as _root

@agent_tool
def query_runs_db(sql: str) -> list: return _sql.query_runs_db(sql)
@agent_tool
def run_python(code: str) -> dict: return _py.run_python(code)
@agent_tool
def http_get(url: str) -> str: return _http.http_get(url)
@agent_tool
def inspect_root(path: str) -> dict: return _root.inspect_root(path)

agent = Agent(
    model="claude-sonnet-4-5",
    system="You are a sPHENIX run analyst...",
    tools=[query_runs_db, run_python, http_get, inspect_root],
)

def run(input_str: str) -> str:
    return agent.run(input_str)
```

(Adjust imports according to the SDK's actual API, which may move before you read this.)

## Golden set: template

```jsonl
{"id":"g01","input":"Count all good AuAu runs in 2024.","check":{"type":"number_range","min":60,"max":80}}
{"id":"g02","input":"What's the total integrated luminosity for good pp runs?","check":{"type":"number_range","min":50,"max":500}}
{"id":"g03","input":"List the 5 highest-n_events good AuAu runs, newest first.","check":{"type":"contains_all","values":["AuAu"]}}
{"id":"g04","input":"Inspect data/event_files/run_47000.root and list the branches.","check":{"type":"contains_all","values":["jet_pt","jet_eta"]}}
{"id":"g05","input":"Plot a histogram of jet_pt from run_47000 to hist.png.","check":{"type":"file_exists","path":"hist.png","min_size_bytes":5000}}
...
```

Expand to 20 tasks spanning SQL, ROOT, Python, HTTP, and multi-tool compositions.

## Eval runner

```python
# src/analyst/eval/runner.py
import json, time
from pathlib import Path
from analyst.frameworks import raw_sdk, langgraph_impl, smolagents_impl, agent_sdk_impl
from analyst.eval.judge import grade

FRAMEWORKS = {
    "raw_sdk": raw_sdk.run,
    "langgraph": langgraph_impl.run,
    "smolagents": smolagents_impl.run,
    "agent_sdk": agent_sdk_impl.run,
}

def main():
    goldens = [json.loads(l) for l in Path("src/analyst/eval/goldens.jsonl").read_text().splitlines()]
    rows = []
    for name, fn in FRAMEWORKS.items():
        for g in goldens:
            t0 = time.time()
            try:
                out = fn(g["input"])
                passed = grade(out, g["check"])
            except Exception as e:
                out, passed = f"ERROR: {e}", False
            rows.append({"framework": name, "id": g["id"], "passed": passed, "seconds": round(time.time() - t0, 2)})
    # aggregate + print
    ...
```

## Memory

### Session memory

```python
# src/analyst/memory/session.py
import json, sqlite3, time

class SessionMemory:
    def __init__(self, db="memory.sqlite", session_id="default"):
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS mem (session TEXT, ts REAL, summary TEXT)")
        self.session = session_id
    def add(self, summary: str):
        self.con.execute("INSERT INTO mem VALUES (?,?,?)", (self.session, time.time(), summary))
        self.con.commit()
    def last_k(self, k=3):
        return [r[0] for r in self.con.execute(
            "SELECT summary FROM mem WHERE session=? ORDER BY ts DESC LIMIT ?",
            (self.session, k)).fetchall()]
```

### Long-term memory

Reuse your Week 08 LanceDB / Chroma infrastructure. Each summary is embedded on write; query embeds the new task; top-k similar summaries are prepended.

## Write-up outline

`framework_comparison.md`:

1. **What you built.** 1 paragraph.
2. **Architecture differences.** Side-by-side for LangGraph / smolagents / Agent SDK.
3. **Numbers.** Eval table. Chart of cost vs success. Latency distributions.
4. **Developer experience notes.** 4 sentences per framework on docs, debuggability, magic vs explicitness, versioning pain.
5. **Where each wins.** Concrete task types.
6. **Your recommendation for Week 12 capstone.** With reasoning tied to the task.

## Common failure modes

- LangGraph version drift: API changed at 0.2.x vs 0.3.x. Pin.
- smolagents Python sandbox: adding `uproot` sometimes causes permission errors in its sandbox. Use `additional_authorized_imports`.
- Agent SDK may require different auth (env var) than plain `anthropic`. Check.
- Goldens with "contains" checks pass trivially if the agent echoes the prompt. Design checks to be hard to game.

## Extensions (optional)

- Add autogen or crew-ai as a fourth framework.
- Point all agents at a local vLLM-served Qwen 14B or Llama 3.1 70B and re-run goldens. How much worse?
- Add tracing to all three via OpenTelemetry → Phoenix. Consolidate observability.

## Sign-off

Commit, tick Week 10 boxes. Week 11: MCP — we expose your tools as standard servers that Claude Desktop, Claude Code, Cursor, and others can consume.
