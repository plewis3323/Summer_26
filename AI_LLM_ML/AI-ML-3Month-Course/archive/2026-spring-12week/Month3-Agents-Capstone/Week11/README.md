# Week 11 — Model Context Protocol (MCP): Make Your Tools Portable

## Where this sits

You've written tools. You've written an agent loop. You've tried three frameworks. Each framework has its own way of registering tools — a decorator here, a schema there, a dispatcher somewhere else. This is the M×N integration problem: M tools × N agent frameworks = chaos.

The Model Context Protocol (MCP) is Anthropic's answer, published November 2024, now an emerging standard. It's a JSON-RPC protocol that lets tools (servers) and agents (clients) speak a common language. Write your tools once as an MCP server; Claude Desktop, Claude Code, Cursor, Zed, Windsurf, and custom agents can all use them.

This week you learn MCP deeply, build a custom MCP server that exposes HEP analysis primitives (your Week 09 tools + more), consume it from multiple clients, and understand how MCP fits into the agent stack you've been building.

By the end of the week, your sPHENIX run-analyst toolset is a reusable asset: wire it into Claude Desktop and it's in your daily workflow; wire it into an agent framework and it's part of a pipeline; wire it into a colleague's Claude-powered editor and they get the same tools.

## Learning objectives

By the end of the week you can:

1. Explain MCP's architecture: clients, servers, transports (stdio, SSE, streamable HTTP), tools, resources, prompts.
2. Write a Python MCP server with `mcp` (Anthropic's official SDK), exposing tools, resources, and prompts.
3. Register your server with Claude Desktop (or Claude Code) via `claude_desktop_config.json` and use it interactively.
4. Consume an MCP server from a custom Python client using `claude-agent-sdk` or a custom MCP client.
5. Secure an MCP server: authentication, transport choice, tool-access controls.
6. Debug MCP: inspect the protocol traffic, use the MCP Inspector, diagnose schema-mismatch errors.
7. Understand the difference between MCP tools, resources, and prompts, and when to reach for each.

## MCP architecture

Three roles:

- **Server.** Exposes capabilities: *tools* (callable functions), *resources* (readable data, like files or DB records), *prompts* (templated prompts the client can use).
- **Client.** Speaks MCP to a server and relays its capabilities to an *LLM host*.
- **Host.** The UI / agent that runs the LLM (Claude Desktop, Claude Code, Cursor, your custom agent).

Clients and hosts are often the same process; the abstraction matters when you write custom clients.

### Transports

- **stdio.** Server runs as a subprocess of the client; they communicate over stdin/stdout with JSON-RPC framing. Simplest. Local use.
- **SSE (Server-Sent Events).** HTTP-based. Server is a long-running HTTP service; client subscribes for streaming. Good for remote servers. Now deprecated in favor of streamable HTTP for most cases.
- **Streamable HTTP.** Single-endpoint HTTP that supports both request/response and server-streamed notifications. The current default for remote servers.

For your HEP tools, stdio is plenty — you run them locally, and the overhead of HTTP is unnecessary.

### Tools, resources, prompts

- **Tools** are like function calls. The LLM picks one to invoke with args; result is returned. Matches exactly what you built in Week 09.
- **Resources** are more like "URIs the LLM can read when it wants to": `runs://recent`, `file:///path/to/plot.png`. They're listed, and the LLM can decide to read them. Static (or data-returning) reference material.
- **Prompts** are templated prompts the server offers. E.g. "Summarize run N" with a slot for N. The client can show these as slash commands.

You'll use all three.

## The `mcp` SDK (Python)

Official Anthropic package. Minimal ceremony:

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hep-tools")

@mcp.tool()
def query_runs_db(sql: str) -> list[dict]:
    """Run a read-only SELECT on the runs database. Columns: run_id, date, species, energy_gev, integrated_lumi_inv_nb, n_events, quality."""
    ...

@mcp.resource("runs://recent")
def recent_runs() -> str:
    """Markdown table of the 10 most recent good runs."""
    ...

@mcp.prompt()
def summarize_run(run_id: int) -> str:
    """Produce a one-paragraph summary of a specific run."""
    return f"Summarize run {run_id}. Include species, energy, integrated luminosity, event count, and quality."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

That's it. `FastMCP` is the high-level facade; under the hood it's JSON-RPC over stdio.

## Using your server in Claude Desktop

`claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hep-tools": {
      "command": "uv",
      "args": ["--directory", "/Users/parker/projects/hep-mcp", "run", "python", "-m", "hep_mcp.server"]
    }
  }
}
```

Restart Claude Desktop. Your tools show up in the "Attachments" (paperclip) menu or in Settings → Tools. Ask Claude about your runs — it calls your tools directly.

This is the moment that makes MCP make sense: your tools are now part of your daily workflow, not just a scripted agent run.

## Consuming from a custom client

`claude-agent-sdk` supports MCP natively:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    mcp_servers={"hep-tools": {"command": "uv", "args": ["--directory", "...", "run", "python", "-m", "hep_mcp.server"]}},
    allowed_tools=["mcp__hep-tools__query_runs_db", "mcp__hep-tools__run_python"],
)
async with ClaudeSDKClient(options=options) as client:
    await client.query("How many good AuAu runs in 2024?")
    async for msg in client.receive_response():
        print(msg)
```

Alternatively, you can write a raw MCP client in Python using the `mcp` SDK's client API. Useful for testing, integration with non-Claude hosts.

## Resources: when and why

Tools are for actions; resources are for reference. Example uses:

- `run://{run_id}/conditions` — calibration constants, DB record.
- `file:///path/to/papers/*.pdf` — your paper corpus.
- `config://default` — current user settings.
- `schema://runs.db` — DDL text.

Resources let the LLM discover and read context without the overhead of "write a tool to fetch X". They're especially powerful when combined with tools: your tool returns a resource URI; the LLM reads it when needed.

## Prompts

Prompts are canned workflows the server advertises. Users see them as slash commands in Claude Desktop. Useful for:

- Standardized analyses: `/qc_report`, `/run_summary`, `/incident_timeline`.
- Onboarding workflows: `/show_recent_runs`.
- Task templates: `/write_skim` that pulls run/cuts and produces a coffea skim skeleton.

Think of prompts as "menu items" exposing your server's capabilities to users who don't want to type "do X" freehand.

## Security

MCP's trust model deserves respect. An MCP server the LLM can invoke is a server that can read and do things on your behalf.

### Servers you install run on your machine
They inherit your user's permissions. Don't install random MCP servers from strangers. Read the source or trust the publisher.

### Tools are called by an LLM
If the LLM is manipulated (prompt injection), it might call tools it shouldn't. Enforce authorization at the server level. Read-only defaults.

### Transport security for remote servers
If you run an SSE / streamable-HTTP MCP server on a network, authenticate. OAuth is supported in the spec; use it. Don't expose a server on public internet without auth.

### "Destructive ops require confirmation" pattern
Claude Desktop has a "requires confirmation" flag for tools. Use it for anything that writes, submits a condor job, or modifies state.

## Debugging: MCP Inspector

Anthropic ships a tool called the MCP Inspector — a tiny web UI that launches your server, lists its tools/resources/prompts, lets you call them interactively, and shows you the JSON-RPC traffic. Use it before trying to get your server into Claude Desktop.

```
npx @modelcontextprotocol/inspector uv --directory /path/to/project run python -m hep_mcp.server
```

## Ecosystem

A growing marketplace of MCP servers covers:

- **Filesystem** — read/write files in a scoped dir.
- **SQLite / Postgres** — DB access.
- **GitHub** — issues, PRs, code search.
- **Slack** — messages, channels.
- **Google Drive / Workspace** — docs, sheets.
- **Chrome / Puppeteer** — web automation.
- **Memory** — persistent long-term memory.
- **Fetch** — HTTP fetching with scraping helpers.

Many are officially by Anthropic; many are community. See the ecosystem registry at https://github.com/modelcontextprotocol/servers.

## Your HEP MCP server

Extend Week 09's tools into a proper MCP server:

**Tools** (actions):
- `query_runs_db(sql)` — read-only SQL.
- `inspect_root(path)` — list trees/branches.
- `run_python(code)` — sandboxed execution.
- `plot(values, bins, path)` — matplotlib histogram.
- `fetch_arxiv(arxiv_id)` — arXiv abstract.
- `submit_skim(runs, cuts)` — queue a skim job (mock).

**Resources** (context):
- `run://{run_id}` — detailed record of a run.
- `schema://runs.db` — DDL of the database.
- `runs://recent` — last 10 runs.
- `file:///path/to/sphenix_notes.md` — personal notes.

**Prompts** (workflows):
- `/run_summary(run_id)` — one-paragraph summary.
- `/qc_report(date_range)` — run-QC table over a period.
- `/analysis_plan(topic)` — start an analysis plan.

## Common failure modes

**Claude Desktop doesn't see the server.** Config file in wrong location, or JSON invalid. Check macOS `~/Library/Logs/Claude/mcp.log` for errors.

**stdio transport deadlocks.** Server printing to stdout instead of stderr. stdout is reserved for JSON-RPC. All logs go to stderr.

**Tools list loads, but calls fail.** Exception thrown in your tool not being handled. Wrap in try/except; return structured error.

**Schema rejected by client.** `input_schema` must be valid JSON Schema. Avoid `anyOf` with non-trivial branches; some clients choke.

**Long-running tool times out.** Clients have per-tool timeouts (often 30s). Design tools to be fast or stream progress updates.

**Resources not showing up.** Some clients require an explicit "list resources" call; if your server doesn't declare the capability, they don't appear.

## HEP angle

Concretely, after this week you have an MCP server that turns Claude Desktop into an analysis assistant:

- "Which runs from last week had bad HCal quality?" — Claude calls `query_runs_db`, reports.
- "Plot the energy distribution of jets in run 47010." — Claude calls `inspect_root`, then `run_python`, returns plot.
- "Find me three ParticleNet papers." — Claude calls `fetch_arxiv`, summarizes.
- "/run_summary 47042" — slash command invokes the prompt.

This is the shape of the capstone in Week 12 Track A.

## Tooling introduced

- `mcp` (Python SDK, Anthropic). `uv add "mcp[cli]"`.
- MCP Inspector: `npx @modelcontextprotocol/inspector ...`.
- Claude Desktop config editing.
- `claude-agent-sdk` for programmatic MCP consumption (carries over from Week 10).

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | Read MCP spec overview. Install SDK. Hello-world server ("hep-tools" returning "ok"). Inspector. |
| 2 | Port Week 09 tools into MCP tools. Register with Claude Desktop. |
| 3 | Add resources. Add prompts. Test slash commands. |
| 4 | Custom Python MCP client (or claude-agent-sdk consumption). |
| 5 | Auth + security. Streamable HTTP transport (try it). |
| 6 | Extensions: add a "recent arXiv papers" resource that fetches daily. |
| 7 | Write-up, ecosystem survey, cleanup. |

## What "done" looks like

- A published-worthy `hep-mcp` Python package with 6+ tools, 3+ resources, 2+ prompts.
- Integrated with Claude Desktop; you use it for at least one real work task before the week ends.
- At least one integration test: spawn the server, connect a client, call every tool, verify results.
- A 500-word write-up documenting installation, capabilities, security posture, limitations.
- Pushed to GitHub; `README.md` shows the config snippet and a demo screenshot.

Week 12 is the capstone — three tracks, one chosen, 2–3 weeks of work condensed into intense final execution.
