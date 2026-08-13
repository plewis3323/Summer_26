# Week 39 — Model Context Protocol

MCP is to agent tooling what a DAQ standard is to detectors: agree on the wire format once, and any client can read out any subsystem — this week your analysis code becomes such a subsystem.

## Objectives

- Explain the MCP architecture: hosts, clients, servers; the tools / resources / prompts primitives; stdio vs. streamable-HTTP transports.
- Build an MCP server in Python exposing tools you own (physics-analysis functions
  by default) with typed inputs, useful descriptions, and path/allowlist checks
  from Week 37.
- Connect that server to an off-the-shelf client (Claude Code or another MCP host) and drive it conversationally.
- Debug the protocol layer: inspect JSON-RPC messages, handle malformed tool inputs, log server-side errors without crashing the session.
- Write server-side tests so tool behavior is verified independently of any model.

## Core material (~3 hrs)

- MCP documentation (modelcontextprotocol.io): the specification concepts pages — architecture, tools, resources, transports. Read tools thoroughly; skim resources and prompts.
- MCP Python SDK documentation (the official `mcp` package / FastMCP server API) — the quickstart server and how tool schemas are generated from function signatures.
- Claude Code docs, the MCP section: how a client registers and calls a local stdio server.
- Skim the MCP inspector tool docs (by title) for protocol-level debugging.

## Exercises (built when the week starts)

Server code lives in `src/`, not notebook cells (house rules: decorators are fine there); the notebook drives and checks it.

1. Hello server: a FastMCP server with one tool, `get_calibration(run_number)`, returning canned constants; list its tools from a script. Accept when: a client-side script prints the tool name, description, and schema exactly as defined.
2. Real tools: add `list_run_files(directory)` and `fit_pi0_peak(file, window_lo, window_hi)` wrapping your Week-37 fitter. Accept when: `pytest -q` passes server-side tests calling each tool function directly with good and bad inputs.
3. Resources: expose the run index as an MCP resource (read-only data, not a tool). Accept when: the client can read the resource and its content matches the file on disk.
4. Error contract: malformed windows or missing files return structured tool errors, not server crashes. Accept when: 5 adversarial inputs each yield an error result while the server keeps serving.
5. Macro generator: add `generate_fun4all_macro(analysis_type, run_number)` producing a skeleton macro from a template. Accept when: generated macro for 2 analysis types passes a syntax/structure check in tests.
6. Live client: register the server with Claude Code (or another MCP host) and ask it to find the latest run and fit the pi0 peak. Accept when: a saved transcript shows the client calling ≥2 of your tools and reporting the fitted mass.

## Deliverable

`week39/mcp_server/` — installable package, `pytest` suite, a one-command launch script, and the saved client transcript from exercise 6.

## Review

1. Week 37: why must every `tool_result` carry the `tool_use_id` of the call it answers? What breaks if two are swapped?
2. Week 7: your `fit_pi0_peak` returns a mass and an uncertainty. From MLE theory, where does that uncertainty estimate come from?
3. Week 34: chunking strategy mattered for RAG. What is the analogous design choice for MCP resources holding large run indexes?
4. Week 3: name two things a server-side unit test can catch that a live model conversation never reliably will.
5. Week 26: sketch the transformer block order (pre-norm) from memory — residual, norm, attention, MLP.
