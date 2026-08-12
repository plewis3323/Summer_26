# Week 11 — Mini-Project: Ship a Reusable `hep-mcp` Server

## Problem

Turn your Week 09 + Week 10 toolkit into a publishable MCP server that you'd actually install and use. Expose 6+ tools, 3+ resources, 2+ prompts. Integrate with Claude Desktop. Test from a custom client. Write clear install / usage docs. Publish to GitHub. Optionally publish the package to PyPI.

This is a deliverable you could hand to a colleague on Day 1 and have them productive — or leave installed on your own machine and start using daily.

## Scope

```
hep-mcp/
├── pyproject.toml
├── README.md                           # user-facing docs: install, config, usage
├── src/hep_mcp/
│   ├── __init__.py
│   ├── server.py                       # FastMCP entrypoint
│   ├── tools/
│   │   ├── runs_db.py                  # query_runs_db, run_record
│   │   ├── rootio.py                   # inspect_root, read_branch_summary
│   │   ├── python_exec.py              # run_python sandbox
│   │   ├── plotting.py                 # plot_histogram, plot_scatter
│   │   ├── papers.py                   # fetch_arxiv, search_inspire
│   │   └── jobs.py                     # submit_skim (mock)
│   ├── resources/
│   │   ├── recent_runs.py              # runs://recent
│   │   ├── run_record.py               # run://{run_id}
│   │   ├── notes.py                    # notes://{note_id}
│   │   └── schema.py                   # schema://runs.db
│   ├── prompts/
│   │   ├── run_summary.py              # /run_summary {run_id}
│   │   ├── qc_report.py                # /qc_report {start} {end}
│   │   └── analysis_plan.py            # /analysis_plan {topic}
│   └── security.py                     # SQL whitelist, URL whitelist, sandbox utilities
├── tests/
│   ├── test_tools.py
│   ├── test_resources.py
│   ├── test_prompts.py
│   ├── test_integration.py             # spawn server, run via client
│   └── test_security.py
├── examples/
│   ├── claude_desktop_config.example.json
│   ├── client_agent_sdk.py
│   └── client_raw.py
└── docs/
    ├── install.md
    ├── usage.md
    └── security.md
```

## Acceptance criteria

1. `uv run pytest -q` passes — unit tests, integration tests, security tests.
2. Server starts cleanly with `uv run python -m hep_mcp.server` and MCP Inspector lists ≥ 6 tools, ≥ 3 resources, ≥ 2 prompts.
3. `claude_desktop_config.example.json` is tested on your machine; the server appears in Claude Desktop and at least one tool executes successfully from a real Claude Desktop session.
4. Custom client (`examples/client_agent_sdk.py`) successfully queries the server and runs at least one tool.
5. Raw MCP client (`examples/client_raw.py`) also works.
6. Security tests cover: SQL injection / DDL blocks; HTTP whitelist; `run_python` timeout; invalid arguments return structured errors.
7. `README.md` includes: quickstart, full install instructions, example questions/conversations, security notes, limitations, contributing.
8. Repo pushed to GitHub (private or public; doesn't matter for grading).
9. Optional: published to TestPyPI.

## Suggested plan (5 days)

- **Day 1.** Scaffold. Port Week 09 tools. First Inspector run. First Claude Desktop integration.
- **Day 2.** Add resources. Add prompts. Test slash commands.
- **Day 3.** Write custom clients (SDK-based and raw). Integration tests.
- **Day 4.** Security hardening. Logging. Edge-case handling.
- **Day 5.** Docs + README + publish.

## Implementation sketches

### server.py

```python
from mcp.server.fastmcp import FastMCP
from hep_mcp.tools import runs_db, rootio, python_exec, plotting, papers, jobs
from hep_mcp.resources import recent_runs, run_record, notes, schema
from hep_mcp.prompts import run_summary, qc_report, analysis_plan

mcp = FastMCP("hep-tools")

# tools
mcp.tool()(runs_db.query_runs_db)
mcp.tool()(runs_db.run_record_tool)
mcp.tool()(rootio.inspect_root)
mcp.tool()(rootio.read_branch_summary)
mcp.tool()(python_exec.run_python)
mcp.tool()(plotting.plot_histogram)
mcp.tool()(plotting.plot_scatter)
mcp.tool()(papers.fetch_arxiv)
mcp.tool()(papers.search_inspire)
mcp.tool()(jobs.submit_skim)

# resources
mcp.resource("runs://recent")(recent_runs.recent_runs)
mcp.resource("run://{run_id}")(run_record.run_record)
mcp.resource("notes://{note_id}")(notes.get_note)
mcp.resource("notes://")(notes.list_notes)
mcp.resource("schema://runs.db")(schema.runs_schema)

# prompts
mcp.prompt()(run_summary.run_summary)
mcp.prompt()(qc_report.qc_report)
mcp.prompt()(analysis_plan.analysis_plan)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

### Representative tool: `tools/runs_db.py`

```python
import sqlite3, json, re
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "data" / "runs.db"
_ALLOWED_PREFIX = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

def query_runs_db(sql: str) -> list[dict]:
    """Run a READ-ONLY SELECT on the sPHENIX runs database.
    Schema: runs(run_id INTEGER, date TEXT, species TEXT, energy_gev REAL,
    integrated_lumi_inv_nb REAL, n_events INTEGER, quality TEXT).
    Returns up to 500 rows.
    """
    if not _ALLOWED_PREFIX.match(sql):
        raise ValueError("only SELECT queries are permitted")
    con = sqlite3.connect(DB)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(500)]
    finally:
        con.close()
    return rows
```

### Representative resource: `resources/run_record.py`

```python
import sqlite3, json
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "data" / "runs.db"

def run_record(run_id: int) -> str:
    """Full record for a specific run (JSON-formatted)."""
    con = sqlite3.connect(DB)
    try:
        cur = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return json.dumps({"error": f"run {run_id} not found"})
        cols = [d[0] for d in cur.description]
        return json.dumps(dict(zip(cols, row)), indent=2)
    finally:
        con.close()
```

### Representative prompt: `prompts/run_summary.py`

```python
def run_summary(run_id: int) -> str:
    """Write a three-sentence summary of one run, citing the run:// resource."""
    return (
        f"Use resource run://{run_id} to pull the full record. "
        "Then produce a 3-sentence summary covering: species + energy; integrated luminosity and event count; quality."
    )
```

## README (excerpt)

```markdown
# hep-mcp

An MCP server exposing heavy-ion / sPHENIX analysis primitives as tools, resources, and prompts for Claude Desktop, Claude Code, or any MCP-compatible client.

## Install

```bash
git clone https://github.com/plewis3323/hep-mcp
cd hep-mcp
uv sync
```

## Run

```bash
uv run python -m hep_mcp.server
```

## Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hep-tools": {
      "command": "uv",
      "args": ["--directory", "/path/to/hep-mcp", "run", "python", "-m", "hep_mcp.server"]
    }
  }
}
```

Restart Claude Desktop. Ask: *"Using hep-tools, what are the five most recent good AuAu runs?"*

## Capabilities

### Tools
| Name | Purpose |
| --- | --- |
| `query_runs_db` | Read-only SELECT on runs |
| `inspect_root` | List trees/branches in a ROOT file |
| `read_branch_summary` | Statistical summary of a ROOT branch |
| `run_python` | Sandboxed Python execution (10s, 1GB) |
| `plot_histogram` | Save a histogram PNG |
| `plot_scatter` | Save a scatter PNG |
| `fetch_arxiv` | Fetch an arXiv abstract |
| `search_inspire` | Search INSPIRE-HEP |
| `submit_skim` | Mock-queue a skim job |

### Resources
| URI | What it returns |
| --- | --- |
| `runs://recent` | Markdown table of 10 most recent good runs |
| `run://{run_id}` | JSON record of a specific run |
| `notes://{note_id}` | A markdown note |
| `schema://runs.db` | DDL of the runs database |

### Prompts
| Slash | Purpose |
| --- | --- |
| `/run_summary {run_id}` | 3-sentence run summary |
| `/qc_report {start} {end}` | QC report over a date range |
| `/analysis_plan {topic}` | Start an analysis plan |

## Security

- `query_runs_db` enforces SELECT-only; DDL/DML rejected.
- `run_python` runs in a subprocess with a 10s wall-clock limit and 1GB memory limit.
- `fetch_arxiv` and `search_inspire` whitelist exactly `arxiv.org`, `api.arxiv.org`, `inspirehep.net`.

## License

MIT
```

## Integration test (reference)

```python
# tests/test_integration.py
import pytest, asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.mark.asyncio
async def test_list_tools_and_call_query():
    params = StdioServerParameters(
        command="uv", args=["run", "python", "-m", "hep_mcp.server"]
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert "query_runs_db" in names
            result = await session.call_tool(
                "query_runs_db",
                {"sql": "SELECT COUNT(*) AS n FROM runs"},
            )
            # result is a list of content items; at least one should be text
            assert any("n" in str(c) for c in result.content)
```

## Security tests

```python
# tests/test_security.py
from hep_mcp.tools.runs_db import query_runs_db
from hep_mcp.tools.papers import fetch_arxiv
import pytest

def test_sql_blocks_drop():
    with pytest.raises(ValueError):
        query_runs_db("DROP TABLE runs")

def test_sql_blocks_update():
    with pytest.raises(ValueError):
        query_runs_db("UPDATE runs SET quality='good'")

def test_fetch_blocks_non_whitelist():
    with pytest.raises(ValueError):
        fetch_arxiv("http://evil.example.com/")
```

## Common failure modes

- `stdout` pollution (print statements) → protocol garbles. Route everything to stderr or logging.
- Server hangs on a slow tool → client times out. Make tools fast; stream progress for slow ones.
- Claude Desktop caches config aggressively → restart fully after edits.
- Resource URIs with weird characters → URL-encode.
- Version drift in `mcp` package — pin with `mcp==0.x.y`.
- Python tool changes not picked up → Claude Desktop caches server process; fully quit & relaunch.

## Extensions (optional)

- Add a GitHub Actions workflow that runs tests.
- Add a `docker/` with a Dockerfile so colleagues don't need uv.
- Publish to PyPI as `hep-mcp` (if you want the name, grab it now).
- Add a "sampling" handler so your server can itself ask the host LLM for help.
- Add a Postgres-compatible mode for when your runs DB grows.

## Sign-off

Commit. Push. Tick Week 11 boxes.

Week 12 is the capstone. You'll pick one of three tracks and spend the final week (or two) turning everything you've built into a single coherent project that you'd be comfortable showing in a group meeting.
