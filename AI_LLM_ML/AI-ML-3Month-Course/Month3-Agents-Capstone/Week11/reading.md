# Week 11 — Reading

## Primary

1. **Anthropic — *Introducing the Model Context Protocol* (announcement, Nov 2024).** https://www.anthropic.com/news/model-context-protocol
   - The framing piece. Read it end-to-end.

2. **Model Context Protocol — official site.** https://modelcontextprotocol.io/
   - Read all core documentation pages. In order:
     - **Introduction**
     - **Architecture**
     - **Concepts → Tools**
     - **Concepts → Resources**
     - **Concepts → Prompts**
     - **Concepts → Sampling** (optional but interesting)
     - **Concepts → Roots**
   - ~60 minutes.

3. **MCP Specification.** https://modelcontextprotocol.io/specification
   - Read the current revision (e.g. 2025-06 spec). Especially:
     - Lifecycle (initialization, shutdown).
     - Transports (stdio, streamable HTTP).
     - Messages (requests, responses, notifications).

4. **Python SDK docs.** https://github.com/modelcontextprotocol/python-sdk
   - README + the `FastMCP` docs.
   - Examples directory has canonical patterns.

5. **Anthropic — *Build an MCP server* tutorial.** https://modelcontextprotocol.io/quickstart/server
   - Walk through it. ~30 minutes.

6. **Anthropic — *Build an MCP client* tutorial.** https://modelcontextprotocol.io/quickstart/client
   - Walk through it. ~30 minutes.

## Secondary

7. **Anthropic — *MCP Inspector* docs.** https://modelcontextprotocol.io/docs/tools/inspector
   - Short. Learn how to test your server interactively.

8. **Ecosystem — reference implementations.** https://github.com/modelcontextprotocol/servers
   - Browse the 30+ reference servers. Skim at least three: `filesystem`, `postgres`, `github`. Observe patterns.

9. **claude-agent-sdk docs — MCP integration.** Find the MCP section of the SDK docs; read the patterns for declaring MCP servers in `ClaudeAgentOptions`.

10. **Simon Willison — MCP posts.** https://simonwillison.net/tags/mcp/
    - Short, critical, pragmatic takes on MCP's current state.

## Security

11. **Anthropic — *Security considerations for MCP servers*.** Find this in the docs; it's canonical.

12. **Simon Willison — *The lethal trifecta for AI agents*.** https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
    - Re-read with MCP in mind. An MCP server that can read your private files + see untrusted content + have external communication = trifecta.

13. **MCP — *OAuth 2.1 authorization flow*.** https://modelcontextprotocol.io/specification/draft/basic/authorization
    - For remote servers, auth matters. Read when you get to remote transport work.

## Tutorials

- **Anthropic Academy / Claude Code videos** — short walkthroughs of MCP.
- **YouTube "build your own MCP server" tutorials** — several good ones appeared in Q1 2025. Try Cole Medin's or Matt Pocock's.
- **Cursor MCP integration docs.** https://docs.cursor.com/ — if you use Cursor, worth knowing.
- **Zed MCP integration.** https://zed.dev/docs/. Alternate MCP client.

## Community servers worth skimming

- `filesystem` — scoped file access.
- `postgres` — DB adapter.
- `github` — issues, PRs, code.
- `slack` — channels, messages.
- `puppeteer` — browser automation.
- `memory` — persistent memory.
- `fetch` — HTTP fetch with markdown extraction.
- `git` — local git repo interactions.

Read one or two fully. Understand: (a) how they structure tools, (b) how they declare capabilities, (c) how they handle auth.

## Papers

No peer-reviewed papers yet on MCP specifically — it's new. Tangential:
- **Any recent survey of LLM tool use ecosystems** — MCP will be mentioned as a convergence point.

## Blog posts

- **Anthropic engineering blog posts** on MCP adoption and case studies.
- **LangChain blog — MCP integration posts.** LangChain has an adapter: `langchain-mcp-adapters`.
- **HuggingFace — MCP integration notes** (their `smolagents` has or will have MCP support).

## Videos

- **Anthropic's launch video** for MCP.
- **Pocock / Medin / Fireship short explainers** — ~10 min, good for orientation.

## Textbooks

No textbook coverage yet. Too new.

## Notes to take

- A diagram of client/server/host roles.
- A table comparing stdio vs streamable HTTP transports.
- A 1-paragraph security model for your server: "trusted client? trusted network? destructive ops?"
- A short list of community MCP servers you'd actually install (fetch, filesystem, github, ...) for your work.

## Reading plan

| Day | What |
| --- | --- |
| 1 | MCP announcement + modelcontextprotocol.io overview. |
| 2 | Specification (lifecycle, transports, messages). |
| 3 | Python SDK docs + server quickstart. |
| 4 | Client quickstart. Inspector. |
| 5 | Reference servers (2-3). Security docs. |
| 6 | Ecosystem browse. |
| 7 | Reference as you build. |

## What to emphasize vs skim

- **Emphasize.** The architecture, the SDK API (`FastMCP`), the client integration, the security model.
- **Skim.** OAuth flow (until you need remote), sampling (rarely used), roots (useful but not central).
