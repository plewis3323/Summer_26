# Week 11 — Exercises

## E1 — Install and scaffold (easy, 20 min)

```bash
uv init hep-mcp && cd hep-mcp
uv add "mcp[cli]"
```

Create `src/hep_mcp/server.py`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hep-tools")

@mcp.tool()
def ping() -> str:
    """Health check — returns 'pong'."""
    return "pong"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Launch with the Inspector:
```
npx @modelcontextprotocol/inspector uv --directory . run python -m hep_mcp.server
```

**Accept when:** Inspector lists `ping`, and calling it returns `"pong"`.

## E2 — First real tool (easy, 30 min)

Port `query_runs_db` (with SELECT-only guard) from Week 09 into the server. Populate a small `runs.db`. Call it via the Inspector, verify it returns sensible data.

Observe: in MCP, docstrings *are* the tool descriptions for the model. Write them carefully.

## E3 — Port all Week 09 tools (medium, 60 min)

Port: `run_python`, `inspect_root`, `http_get`, `plot`. Each becomes a `@mcp.tool()`.

Keep sandboxing you built in Week 09; MCP doesn't do it for you.

**Accept when:** Inspector lists 5 tools; each works.

## E4 — Register with Claude Desktop (easy, 20 min)

macOS config: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hep-tools": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/hep-mcp", "run", "python", "-m", "hep_mcp.server"]
    }
  }
}
```

Restart Claude Desktop. Go to Settings → Developer → MCP Servers; verify your server is listed.

Ask Claude: "Using hep-tools, how many good AuAu runs are in the database?" Watch it call `query_runs_db`.

Windows: `%APPDATA%\Claude\claude_desktop_config.json`.

## E5 — Resources (medium, 45 min)

Add:

```python
@mcp.resource("runs://recent")
def recent_runs() -> str:
    """Markdown table of the 10 most recently-logged good runs."""
    import sqlite3
    con = sqlite3.connect("runs.db")
    rows = con.execute(
        "SELECT run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events "
        "FROM runs WHERE quality='good' ORDER BY date DESC LIMIT 10"
    ).fetchall()
    lines = ["| run_id | date | species | E (GeV) | L (1/nb) | n_events |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)

@mcp.resource("run://{run_id}")
def run_record(run_id: int) -> str:
    """Full record for a specific run."""
    import sqlite3, json
    con = sqlite3.connect("runs.db")
    row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None: return f"Run {run_id} not found."
    return json.dumps(dict(zip([c[0] for c in con.execute("SELECT * FROM runs LIMIT 1").description], row)), indent=2)
```

In Claude Desktop, click the + icon in the chat, attach `runs://recent`, and watch it inject into the conversation.

## E6 — Prompts (medium, 30 min)

```python
@mcp.prompt()
def run_summary(run_id: int) -> str:
    """Generate a one-paragraph summary of a specific run."""
    return (
        f"Look up run {run_id} via run://{run_id}. "
        "Then give me a 3-sentence summary covering species, energy, integrated luminosity, and quality."
    )

@mcp.prompt()
def qc_report(start_date: str, end_date: str) -> str:
    """Produce a QC report over a date range."""
    return (
        f"Query runs between {start_date} and {end_date}. "
        "For each quality category, count runs and sum integrated luminosity. "
        "Report as a markdown table. Call out any runs marked 'ongoing' or 'bad'."
    )
```

In Claude Desktop, `/run_summary 47000` should appear as a slash command and invoke the prompt.

## E7 — Custom Python client (medium, 60 min)

Use `claude-agent-sdk` to consume your MCP server from a Python script (not Claude Desktop):

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
import asyncio

options = ClaudeAgentOptions(
    mcp_servers={
        "hep-tools": {
            "type": "stdio",
            "command": "uv",
            "args": ["--directory", "/path/to/hep-mcp", "run", "python", "-m", "hep_mcp.server"],
        }
    },
    allowed_tools=["mcp__hep-tools__query_runs_db", "mcp__hep-tools__run_python"],
)

async def main():
    async with ClaudeSDKClient(options=options) as client:
        await client.query("How many good AuAu runs in 2024?")
        async for msg in client.receive_response():
            print(msg)

asyncio.run(main())
```

**Accept when:** the script runs, calls your MCP tool, and prints Claude's answer.

## E8 — Raw MCP client (hard, 60 min)

Write a bare-bones MCP client that speaks to your server without any Anthropic SDK:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    params = StdioServerParameters(command="uv", args=["--directory", "/path/to/hep-mcp", "run", "python", "-m", "hep_mcp.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])
            result = await session.call_tool("query_runs_db", {"sql": "SELECT COUNT(*) FROM runs WHERE quality='good' AND species='AuAu'"})
            print("Result:", result)

asyncio.run(main())
```

Useful for testing, for non-LLM integrations, for writing your own agent hosts.

## E9 — Streamable HTTP transport (medium, 60 min)

Switch server to HTTP:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8765)
```

Client:
```python
from mcp.client.streamable_http import streamablehttp_client
async with streamablehttp_client("http://localhost:8765/mcp") as (r, w, _):
    async with ClientSession(r, w) as session:
        ...
```

Run server in one terminal; client in another. Verify tools list and a tool call work over HTTP.

## E10 — Auth for remote transport (medium, 45 min)

Add a simple bearer-token check:

```python
@mcp.auth(strategy="bearer")
def check_auth(token: str) -> bool:
    return token == os.environ.get("HEP_MCP_TOKEN")
```

(API names may differ; consult current SDK docs.)

Set an env var; confirm client requests without the token get rejected, with the token succeed.

## E11 — Install a community server alongside yours (easy, 30 min)

In Claude Desktop config, also register the `fetch` server:

```json
{
  "mcpServers": {
    "hep-tools": { ... },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

Ask Claude: "Fetch arxiv.org/abs/2202.03772 and summarize the main claim, then look up related runs in our database." Claude should call `fetch` then `query_runs_db`.

**Accept when:** cross-server orchestration works.

## E12 — Tests (medium, 45 min)

Write `tests/test_server.py`:

```python
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.mark.asyncio
async def test_tools_list():
    params = StdioServerParameters(command="uv", args=["run", "python", "-m", "hep_mcp.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert {"query_runs_db", "run_python", "inspect_root"} <= names

@pytest.mark.asyncio
async def test_query_runs_db_blocks_ddl():
    ...
```

**Accept when:** pytest passes; your CI could realistically run these.

## E13 — Write a physics-note resource (medium, 60 min)

Implement a resource that serves a directory of markdown notes:

```python
@mcp.resource("notes://{note_id}")
def get_note(note_id: str) -> str:
    p = Path("notes") / f"{note_id}.md"
    if not p.exists(): return f"Note {note_id} not found."
    return p.read_text()

@mcp.resource("notes://")
def list_notes() -> str:
    return "\n".join(p.stem for p in Path("notes").glob("*.md"))
```

Add 3–5 notes. In Claude Desktop, attach `notes://` to see the list, then attach a specific note URI.

---

## Solution hints

- E1 — if Inspector can't connect, your command and args are wrong; test `uv --directory ... run python -m hep_mcp.server` in terminal first.
- E2 — `FastMCP` infers input_schema from type hints. Use real types (str, int, list[dict]).
- E3 — stdout must stay clean for JSON-RPC. All `print()` calls in your tools go to stderr or are removed.
- E4 — if Claude Desktop doesn't see your server, check `~/Library/Logs/Claude/mcp.log`.
- E5 — resources with `{param}` syntax become templated URIs; clients resolve them.
- E6 — prompts appear as slash commands. Users pick them, then fill in parameters.
- E7 — `claude-agent-sdk` supports MCP; naming conventions for tools are `mcp__<server-name>__<tool-name>`.
- E8 — raw client is low-level but great for tests and custom hosts.
- E9 — HTTP transport behaves the same as stdio from the client's perspective.
- E10 — auth is server-version-dependent; consult current docs.
- E11 — `uvx` is `uv`'s throwaway-install runner. Handy for MCP servers.
- E12 — async tests need `pytest-asyncio`.
- E13 — resources with directory listings enable users to discover context without you hand-curating it.
