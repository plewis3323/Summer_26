# Week 39 — Exercises

Work top to bottom. Setup (the toy-data generator that writes `data/run_*.csv`,
`calibrations.json`, `run_index.json`, the Fun4All macro template
`macro_template.txt`, and the `check` helper) is given by the notebook. Almost
everything this week lives in files rather than cells: the deliverable is an
installable package, `week39/mcp_server/`, and the acceptance criteria are
`pytest` runs and scripts, so E1–E5 build `server.py`, `tests/test_server.py`,
and small client-side scripts while the notebook only drives and checks them.
Type hints and decorators are required in the server files — the SDK reads them
to generate schemas (lesson §3); the house rules against them apply to notebook
cells, not server code. E1–E5 involve no model and cost nothing; only E6 talks
to a model. Standing warning for the whole week: with stdio, standard output is
the wire — never `print()` inside a tool (lesson §8).

## E1 — Hello server (`server.py` + `client_check.py`)

In `week39/mcp_server/server.py`, create a FastMCP server exposing one tool,
`get_calibration(run_number)`, returning the canned constants from
`calibrations.json` — docstring, type hints, and a `ValueError` naming the known
runs, as in lesson §4. In `client_check.py`, launch the server over stdio and
list its tools (lesson §7), printing each name, description, and input schema.
Hint: the docstring becomes the description and the type hints become the
schema; if the printed schema is missing a property, a hint is missing.
Accept when: a client-side script prints the tool name, description, and schema
exactly as defined.

## E2 — Real tools (`server.py` + `tests/test_server.py`)

Add `list_run_files(directory)` and `fit_pi0_peak(file, window_lo, window_hi)`,
wrapping your Week-37 fitter with the validation guards from lesson §4 and §9.
Write server-side tests that import the functions directly: good inputs (fitted
mean within 0.005 GeV of 0.135) and bad ones (`pytest.raises` with `match=` on
the message text).
Hint: test the error *message*, not just the exception type — the message is
exactly what the model will read when the call fails.
Accept when: `pytest -q` passes server-side tests calling each tool function
directly with good and bad inputs.

## E3 — Resources (`server.py` + `client_check.py`)

Expose the run index as an MCP resource at `runs://index` — read-only data, not
a tool (lesson §5). Extend `client_check.py` to read it back through the client.
Hint: `@mcp.resource("runs://index")` server-side; client-side the session has a
resource-read call (`session.read_resource(...)` — check the current SDK docs
for the exact name).
Accept when: the client can read the resource and its content matches the file
on disk.

## E4 — Error contract (`client_errors.py`)

Write a script that calls your tools through the client with 5 adversarial
inputs — an inverted window, a missing file, an unknown run number, an empty
directory, and a window containing too few events — then makes one good call to
prove the server is still alive.
Hint: no server changes should be needed — if §9's validate-first guards are in
place, each bad call comes back as an error-flagged result, never a crash.
Accept when: 5 adversarial inputs each yield an error result while the server
keeps serving.

## E5 — Macro generator (`server.py` + tests)

Add `generate_fun4all_macro(analysis_type, run_number)`: fill the provided
template to produce a skeleton Fun4All macro. (Fun4All is sPHENIX's C++
analysis framework; a *macro* is the steering script that configures and runs
one analysis job.) Support two analysis types — `"pi0"` and `"calib_check"` —
and reject anything else with a `ValueError` naming the valid ones.
Hint: this is string substitution into the template, nothing more; the tests
check structure (required lines present, braces balanced, the run number
embedded), not C++ semantics.
Accept when: the generated macro for 2 analysis types passes a syntax/structure
check in tests.

## E6 — Live client (transcript)

Register the server with Claude Code (or another MCP host) — lesson §8 — and
ask it to find the latest run and fit the pi0 peak. Save the conversation to
`transcripts/e6.md`.
Hint: `claude mcp add physics -- uv run python server.py` (check the current
docs for the exact syntax). If the server works in the inspector but not in the
host, hunt for a stray `print`.
Accept when: a saved transcript shows the client calling ≥ 2 of your tools and
reporting the fitted mass.

## Review

1. Week 37: why must every `tool_result` carry the `tool_use_id` of the call it
   answers — and what breaks if two are swapped?
2. Week 08: your `fit_pi0_peak` returns a mass and an uncertainty. From MLE
   theory, where does that uncertainty estimate come from?
3. Week 34: chunking strategy mattered for RAG. What is the analogous design
   choice for MCP resources holding large run indexes?
4. Week 04: name two things a server-side unit test can catch that a live model
   conversation never reliably will.
5. Week 26: sketch the pre-norm transformer block order from memory — residual,
   norm, attention, MLP.
