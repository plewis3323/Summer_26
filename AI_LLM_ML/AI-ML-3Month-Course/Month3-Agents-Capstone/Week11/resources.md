# Week 11 — Resources

## Docs

- **Model Context Protocol** — https://modelcontextprotocol.io/
- **MCP specification** — https://modelcontextprotocol.io/specification
- **Python SDK** — https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK** — https://github.com/modelcontextprotocol/typescript-sdk (skim; same concepts)
- **Claude Desktop MCP configuration** — https://modelcontextprotocol.io/quickstart/user
- **claude-agent-sdk MCP section** — https://docs.claude.com

## Libraries

- `mcp[cli]` — Anthropic's Python SDK (includes FastMCP + client).
- `mcp` — same, without CLI extras.
- `claude-agent-sdk` — consumes MCP servers from Python.
- `uv` / `uvx` — run servers; `uvx` spawns throwaway installs.

Install set:
```
uv add "mcp[cli]"
uv add claude-agent-sdk
```

## Reference servers (read the source)

- `filesystem` — scoped file reads/writes. https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
- `postgres` — DB access. https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
- `github` — issues, PRs, code. https://github.com/modelcontextprotocol/servers/tree/main/src/github
- `slack` — messages.
- `memory` — persistent KV / graph memory.
- `fetch` — HTTP fetch with markdown extraction.
- `git` — local git repo.
- `puppeteer` — browser automation.

Install `fetch` alongside yours in Claude Desktop:
```json
"fetch": { "command": "uvx", "args": ["mcp-server-fetch"] }
```

## Hosts

- **Claude Desktop.** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS); `%APPDATA%\Claude\claude_desktop_config.json` (Windows).
- **Claude Code.** `~/.claude.json` or project-level `.mcp.json`.
- **Cursor.** https://docs.cursor.com/
- **Zed.** https://zed.dev/docs/
- **Windsurf.** https://codeium.com/windsurf
- **Continue.dev.** https://docs.continue.dev/

## Tutorials

- **MCP quickstart (server).** https://modelcontextprotocol.io/quickstart/server
- **MCP quickstart (client).** https://modelcontextprotocol.io/quickstart/client
- **Anthropic Academy** — short video walkthroughs.
- **Simon Willison's MCP notes.** https://simonwillison.net/tags/mcp/
- **Matt Pocock on YouTube.** Several MCP videos Q1 2025.
- **Cole Medin on YouTube.** Hands-on MCP builds.

## Debugging / testing

- **MCP Inspector.** `npx @modelcontextprotocol/inspector <cmd...>`. Indispensable.
- **mcp.log.** `~/Library/Logs/Claude/mcp.log` (macOS). Tail while debugging.
- **Protocol traces.** Run server with stderr open in a terminal to see JSON-RPC frames.

## Security resources

- **MCP security considerations.** In modelcontextprotocol.io docs.
- **OAuth 2.1 for remote MCP.** https://modelcontextprotocol.io/specification/draft/basic/authorization
- **Simon Willison — *The lethal trifecta*.** https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- **Anthropic responsible-use guidance for agents.**

## Papers

Not many peer-reviewed papers on MCP itself (too new). Tangential:

- **Any recent survey of LLM tool-use ecosystems.** MCP will appear as a convergence point.
- **Toolformer** — arXiv:2302.04761. Pre-MCP foundation.
- **Gorilla** — arXiv:2305.15334. LLMs calling APIs at scale; informs why standardization matters.

## Blog posts

- **Anthropic — *Introducing the Model Context Protocol*.** https://www.anthropic.com/news/model-context-protocol
- **Anthropic engineering blog** — periodic MCP posts.
- **LangChain blog** — `langchain-mcp-adapters` posts.
- **HuggingFace blog** — smolagents + MCP integration.

## HEP / science-adjacent MCP

Nothing canonical yet (April 2026). Some exploratory servers floating around:

- `mcp-server-arxiv` — community fetcher for arXiv abstracts.
- `mcp-server-inspire` — INSPIRE-HEP search (may or may not exist; build your own if not).
- `mcp-server-jupyter` — exec cells in a running kernel.

If you ship `hep-mcp`, you're early. That's good — easy name, no competition yet.

## Cost awareness

- MCP protocol itself: free; it's local JSON-RPC.
- LLM costs via Claude Desktop: standard per-token (covered by your plan).
- Cost of tool calls: whatever your tools consume (arxiv API is free; your DB is local).

Budget for Week 11: near zero for the protocol work, plus normal Claude Desktop usage.

## Troubleshooting

- **Server doesn't appear in Claude Desktop.** Check JSON validity (`jq . claude_desktop_config.json`). Check `mcp.log`. Restart Claude Desktop fully (quit from the dock, not just close window).
- **Server appears but tools don't list.** Your server crashed on startup. Check stderr: `uv --directory ... run python -m hep_mcp.server` in a terminal and watch for tracebacks.
- **Inspector says "disconnected".** The spawned subprocess printed to stdout. Hunt down `print()` calls; route everything to `logging` with stderr handler.
- **Tool call hangs.** Your tool is slow; increase client timeout or make tool fast.
- **Claude Desktop uses stale code.** Claude Desktop caches the spawned server process. Quit fully and relaunch.
- **Config edits ignored.** Some Claude Desktop versions need a full log-out / log-in; at minimum, quit entirely.
- **Resource URI templates don't resolve.** Make sure the pattern `run://{run_id}` and function signature `def run_record(run_id: int)` match exactly.

## HEP tie-ins

- **sPHENIX runs DB → MCP.** The move this week: your read-only runs query is the first tool; everyone on your subgroup could install it.
- **Personal notes → MCP.** Your markdown notes under `notes/` become a `notes://` resource; Claude Desktop can search and attach them.
- **arXiv daily feed → MCP resource.** A cached `arxiv://hep-ex/today` resource refreshed daily; attach it to your morning workflow.
- **Running condor jobs → MCP tool.** Dangerous, but tractable with confirmation flags. Mock it first.

## Community

- **MCP Discord / GitHub discussions.** https://github.com/modelcontextprotocol/specification/discussions
- **/r/LocalLLaMA.** MCP integration threads.
- **Hacker News.** Frequent MCP posts; comments are mixed-to-useful.

## What to emphasize vs. skim

- **Emphasize.** The SDK API (`FastMCP`), writing tools/resources/prompts correctly, Claude Desktop config, security.
- **Skim.** OAuth (until you run remote), SSE (deprecated), internal JSON-RPC details (SDK handles them).

---

Week 12: the capstone. You pick one of three tracks — HEP copilot, conditional diffusion VAE, or paper-to-pipeline — and turn everything you've built into a single coherent project.
