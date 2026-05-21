# Week 09 — Mini-Project: A "Run Analyst" Tool-Using Agent

## Problem

Build a small tool-using agent that acts as a junior run-coordinator analyst. Given natural-language questions about a (synthetic) sPHENIX run database, synthetic ROOT files, and a curated subset of arXiv abstracts, it answers them by calling a handful of tools: SQL query, Python exec, HTTP fetch, ROOT inspect, plotting.

This isn't an agent framework yet (Week 10). It's a hand-rolled loop to solidify your understanding of tool use before frameworks abstract it away.

## Scope

```
project/
├── pyproject.toml
├── README.md
├── src/run_analyst/
│   ├── __init__.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── python_exec.py
│   │   ├── sql.py
│   │   ├── http.py
│   │   ├── root_inspect.py
│   │   └── plot.py
│   ├── schemas.py            # JSON Schema / Pydantic models
│   ├── agent.py              # run_agent(client, system, user, tools, dispatch)
│   ├── trace.py              # JSONL logging
│   └── cli.py                # typer
├── tests/
│   ├── test_tools.py         # unit-test each tool
│   ├── test_agent.py         # deterministic tool-mocked agent runs
│   └── test_security.py      # SQL DDL blocked, out-of-whitelist URLs blocked, etc.
├── data/
│   ├── runs.db               # synthetic SQLite
│   ├── event_files/          # synthetic ROOT files
│   └── abstracts/            # a few arXiv abstracts
├── configs/
│   └── run_analyst.yaml
└── results/
    ├── traces/               # transcripts from runs
    └── report.md             # write-up
```

## Acceptance criteria

1. `uv run pytest -q` passes all tool tests, agent tests, and security tests.
2. `run_analyst ask "..."` CLI works end-to-end and writes a trace to `results/traces/{timestamp}.jsonl`.
3. 5 example questions answered correctly:
   - "How many good AuAu runs in 2024?" (SQL)
   - "Compute the mean events/run for pp runs above 100 GeV." (SQL + run_python)
   - "Inspect event_files/run_47000.root; list branches." (root_inspect)
   - "Plot a histogram of jet_pt from run_47000.root; save to plot.png." (root_inspect + run_python + plot)
   - "Fetch arXiv:2202.03772 and tell me its main contribution." (http_get)
4. Security tests pass:
   - `query_runs_db` refuses DROP/UPDATE/DELETE.
   - `http_get` refuses domains not on the whitelist.
   - `run_python` times out correctly at 10s.
5. Trace log includes timestamp, tool name, args, result (or error), elapsed_ms per tool call.
6. At least one prompt-injection test (E12) and a short write-up on what happened.
7. README documents: all tools, their schemas, the agent loop, sample traces, known limitations.

## Suggested plan (5 days)

- **Day 1.** Synthetic SQLite + ROOT generation. Tool scaffolding.
- **Day 2.** run_python + tests. query_runs_db + tests.
- **Day 3.** http_get + inspect_root + plot.
- **Day 4.** Agent loop, schemas, CLI. Run the 5 example questions.
- **Day 5.** Security tests, traces, write-up.

## Generating synthetic data

```python
# scripts/gen_runs.py
import sqlite3, random
random.seed(1)
con = sqlite3.connect("data/runs.db")
con.executescript("""
DROP TABLE IF EXISTS runs;
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    date TEXT, species TEXT, energy_gev REAL,
    integrated_lumi_inv_nb REAL, n_events INTEGER, quality TEXT
);
""")
for run_id in range(47000, 47100):
    species = random.choice(["AuAu", "pp", "pAu"])
    energy = {"AuAu": 200, "pp": 200, "pAu": 200}[species] + random.choice([0, 10, -10])
    lumi = random.uniform(0.1, 50.0)
    nev = random.randint(1_000_000, 100_000_000)
    q = random.choices(["good","bad","ongoing"], [0.7, 0.2, 0.1])[0]
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                (run_id, f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                 species, energy, lumi, nev, q))
con.commit(); con.close()
```

```python
# scripts/gen_root.py
import uproot, numpy as np
from pathlib import Path
Path("data/event_files").mkdir(exist_ok=True, parents=True)
for run_id in [47000, 47001, 47002]:
    n = 50_000
    jet_pt   = np.random.exponential(10.0, n) + 5
    jet_eta  = np.random.uniform(-1.1, 1.1, n)
    jet_phi  = np.random.uniform(-np.pi, np.pi, n)
    n_jets   = np.random.poisson(3, n).astype(np.int32)
    with uproot.recreate(f"data/event_files/run_{run_id}.root") as f:
        f["events"] = {"jet_pt": jet_pt, "jet_eta": jet_eta, "jet_phi": jet_phi, "n_jets": n_jets}
```

## Schemas

```python
# src/run_analyst/schemas.py
TOOLS = [
    {
        "name": "run_python",
        "description": "Execute a Python snippet in a sandboxed subprocess with a 10-second timeout. Use for data munging, math, plotting. Do NOT use for long-running computations. The CWD is a temp dir; read/write files there.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "query_runs_db",
        "description": "Run a read-only SELECT against the runs database (table: runs, columns: run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events, quality). Returns up to 500 rows.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "http_get",
        "description": "Fetch the text content of an arxiv.org or inspirehep.net URL. Returns up to 50000 characters.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "format": "uri"}},
            "required": ["url"],
        },
    },
    {
        "name": "inspect_root",
        "description": "Inspect a ROOT file: list keys, trees, branches, types, entry counts. Does not read values. Use run_python with uproot for actual data access.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "save_plot",
        "description": "Save a matplotlib Figure to a PNG under results/. Returns the saved path. Call after creating a figure in run_python.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "title": {"type": "string"}},
            "required": ["path"],
        },
    },
]
```

(`save_plot` is arguably redundant with `run_python`; included to show tool layering. You can drop it.)

## Agent loop (reference)

```python
# src/run_analyst/agent.py
from anthropic import Anthropic
import time, json, uuid
from pathlib import Path

def run_agent(client, system, user_msg, tools, dispatch, max_steps=15, trace_path=None):
    messages = [{"role": "user", "content": user_msg}]
    trace = []
    for step in range(max_steps):
        resp = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=2048, system=system, tools=tools, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            break
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            t0 = time.time()
            try:
                result = dispatch(block.name, block.input)
                status = "ok"
            except Exception as e:
                result = {"error": str(e)}
                status = "error"
            elapsed = round((time.time() - t0) * 1000)
            trace.append({"step": step, "tool": block.name, "input": block.input,
                          "result_preview": str(result)[:500], "status": status, "elapsed_ms": elapsed})
            tool_results.append({"type":"tool_result","tool_use_id":block.id,
                                 "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)})
        messages.append({"role":"user","content":tool_results})
    if trace_path:
        Path(trace_path).write_text("\n".join(json.dumps(t) for t in trace))
    return messages, trace
```

## CLI

```python
# src/run_analyst/cli.py
import typer, time
from anthropic import Anthropic
from pathlib import Path
from run_analyst.schemas import TOOLS
from run_analyst.agent import run_agent
from run_analyst.tools import dispatch

app = typer.Typer()

SYSTEM = """You are a sPHENIX run-coordination analyst assistant.
You have access to: a SQLite DB of runs, synthetic ROOT event files, Python exec, and HTTP fetch (arxiv/inspire).
When you need data, CALL A TOOL. Do not fabricate numbers.
First plan in <plan> tags. Execute. Then summarize with citations to tool outputs."""

@app.command()
def ask(query: str):
    client = Anthropic()
    ts = time.strftime("%Y%m%d-%H%M%S")
    trace_path = Path("results/traces") / f"{ts}.jsonl"
    trace_path.parent.mkdir(exist_ok=True, parents=True)
    msgs, trace = run_agent(client, SYSTEM, query, TOOLS, dispatch, trace_path=trace_path)
    last = msgs[-1]["content"]
    if isinstance(last, list):
        for b in last:
            if getattr(b, "type", None) == "text":
                print(b.text)
    print(f"\n(trace: {trace_path})")

if __name__ == "__main__":
    app()
```

## Security tests

```python
# tests/test_security.py
import pytest
from run_analyst.tools.sql import query_runs_db
from run_analyst.tools.http import http_get

def test_sql_blocks_drop():
    with pytest.raises(ValueError):
        query_runs_db("DROP TABLE runs")

def test_sql_blocks_update():
    with pytest.raises(ValueError):
        query_runs_db("UPDATE runs SET quality='good'")

def test_http_whitelist_blocks_other_domains():
    with pytest.raises(ValueError):
        http_get("https://evil.example.com/")
```

## Write-up (`results/report.md`)

```markdown
# Run Analyst — Week 09 Mini-Project

## Summary
A tool-using LLM agent over a synthetic sPHENIX run database. Five tools:
run_python, query_runs_db, http_get, inspect_root, save_plot.

## Example traces
[link to `results/traces/20260419-120415.jsonl` etc.]

## Observations
- Claude Sonnet plans 2–4 step workflows cleanly; rarely hallucinates tool names.
- SQL errors trigger one-shot retries without explicit prompting.
- ROOT inspect + run_python combo works reliably for histogramming.
- Prompt injection stress test: agent did not execute injected instructions; occasionally mentioned them.

## Limitations
- Synthetic data — no guarantee real sPHENIX schemas behave the same.
- `run_python` sandbox is minimal; don't expose this on a server without hardening.
- No memory across sessions; each call is independent.

## Reproducibility
Seed-set synthetic data regenerated via `uv run python scripts/gen_runs.py && uv run python scripts/gen_root.py`.
```

## Common failure modes

- `run_python` subprocess hangs on `input()` or infinite loops → enforce timeout.
- `query_runs_db` with a giant `SELECT *` blows the result payload → cap rows.
- Agent loops forever on a task it can't solve → `max_steps=15` bailout.
- Tool descriptions ambiguous → model picks the wrong tool; rephrase.
- Parallel tool results re-ordered by dispatcher → model confused; keep order.

## Extensions (optional)

- Add a `get_run_notes(run_id)` tool that retrieves a markdown blob of human-written notes. Feed into RAG from Week 08.
- Add memory: after each agent call, append a summary to `memory.jsonl`. On next call, include recent memories in the system prompt.
- Make it a cron job: once a day, "summarize yesterday's new good runs" and email the report.
- Switch the model from Claude to a local Qwen-14B via vLLM and measure the quality gap.

## Sign-off

Commit, tick Week 09 boxes. Week 10: agent frameworks abstract away the loop and add planning, memory, and multi-agent coordination.
