# Week 39 — Model Context Protocol (MCP)

~4 hrs. Before starting you should be able to: define a tool with a JSON Schema and
run the agentic loop (Week 37), write and run `pytest` tests (Week 04), and fit the
π⁰ peak in a toy run file (Week 37's `fit_pi0_peak`).

In Week 37 your tools lived inside one script, wired to one model, in one
conversation loop that you wrote. That does not scale: every new client (a chat app,
an editor, a colleague's agent) would need your functions copy-pasted in and re-wired
by hand. This week you put the same physics tools behind a **standard interface** —
the Model Context Protocol — so that *any* MCP-speaking client can discover and call
them, with no code changes on either side. Your analysis code becomes a service.

Same standing note as all month: **check the current docs — this API evolves.** MCP
is younger than most of what you have learned; the concepts are stable, the SDK
surface shifts.

## 1. What a protocol is, and why MCP exists

A **protocol** is an agreed-upon message format plus rules for who sends what when.
You have used several without ceremony: HTTP (Week 37) is a protocol; so is the
detector world's equivalent — a DAQ standard that lets any readout crate talk to any
subsystem because both sides agreed on the wire format once.

The problem MCP solves is the M×N problem. M applications that host models (chat
apps, coding assistants, your copilot) and N sources of tools and data (your fitter,
a database, GitHub) would naively need M×N custom integrations. A protocol turns
that into M+N: every host speaks MCP once, every tool provider speaks MCP once, and
any pair can connect. MCP (introduced by Anthropic in 2024, since adopted broadly)
is that protocol for connecting models to tools and data.

The architecture has three roles:

- **Host** — the application the user actually runs (Claude Code, a chat app, your
  Week 40 copilot). It owns the model and the conversation.
- **Client** — the connector object *inside* the host that maintains a one-to-one
  connection to a single server. Hosts hold one client per server.
- **Server** — a program (yours, this week) that exposes capabilities. It knows
  nothing about models or conversations; it answers protocol messages.

Under the hood the messages use **JSON-RPC**: a tiny convention for remote procedure
calls as JSON objects — a request has a `method` name, `params`, and an `id`; the
response echoes the `id` with a `result` or an `error`. You saw why matched IDs
matter with `tool_use_id` in Week 37; same idea, one level down. You will rarely
write JSON-RPC by hand, but you will read it when debugging (§8).

Messages have to travel somehow: that is the **transport**. Two matter:

- **stdio** — the host launches your server as a subprocess and they exchange
  JSON-RPC lines over standard input/output. No network, no ports, ideal for local
  tools. This week uses stdio.
- **streamable HTTP** — the server is a web service; remote clients connect over
  HTTP. Same messages, different pipe; you would deploy this to share tools across
  machines.

## 2. The three primitives: tools, resources, prompts

An MCP server can expose three kinds of things:

- **Tools** — functions the *model* decides to call, exactly like Week 37 tools:
  name, description, input schema, result. `fit_pi0_peak` is a tool.
- **Resources** — read-only data identified by a URI, chosen by the *application*
  (or user) rather than the model — think "files to attach as context," not "actions
  to take." A run index listing every run and its status is a resource: nothing to
  execute, just content to read.
- **Prompts** — reusable prompt templates the *user* picks (e.g., a "summarize this
  run" template). Least used of the three; know it exists.

The tool/resource split mirrors an old physics distinction: a resource is the
run database you consult; a tool is the fit you execute. If calling it twice could
give different answers or cost something, it is a tool.

## 3. Two Python constructs the SDK needs

The official SDK generates all the JSON Schema plumbing from your function
signatures — but to do that it relies on two Python features this course has not
needed until now. Both get two paragraphs, which is all they need.

**Type hints.** Python lets you annotate what type each argument and return value
should be: `def get_calibration(run_number: int) -> dict:` says `run_number` should
be an `int` and the function returns a `dict`. Plain Python *ignores* these
annotations at runtime — they are documentation with standard syntax. Libraries,
however, can read them: FastMCP reads `run_number: int` and generates
`{"type": "integer"}` in the tool's schema for you. That is why the hints are
required here after being absent all course.

**Decorators.** The `@something` line above a `def` is a decorator: a function that
takes your function and registers, wraps, or replaces it. `@mcp.tool()` means "hand
the function just defined to the server object, so it becomes a registered tool" —
one line replacing all of Week 37's hand-written definition dicts. You have already
*seen* this shape without the name: it is how pytest fixtures and Flask routes work
in code you may have read. You only need to use decorators, not write them.

## 4. Building the server

Install the official SDK (`uv add "mcp[cli]"`) and create `server.py`. The setup:
a `data/` directory of toy run files as in Week 37, plus a `calibrations.json` the
exercises notebook writes. Here is a complete three-tool server:

```python
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("physics-tools")

CALIB_FILE = "calibrations.json"

@mcp.tool()
def get_calibration(run_number: int) -> dict:
    """Return the energy-scale calibration for one run.

    A calorimeter's raw energy response drifts with time and temperature, so each
    run gets a correction factor determined offline; multiply measured energies by
    it to put the pi0 peak at the true mass. Returns {"energy_scale": float,
    "status": "good"|"bad"} for the requested run.
    """
    with open(CALIB_FILE) as f:
        calib = json.load(f)
    key = str(run_number)
    if key not in calib:
        raise ValueError("no calibration for run " + key +
                         "; known runs: " + ", ".join(sorted(calib)))
    return calib[key]

@mcp.tool()
def list_run_files(directory: str) -> dict:
    """List run data files in a directory, sorted by run number.

    Collider data arrives in numbered runs (contiguous data-taking periods); an
    analysis starts by discovering which runs exist. Returns {"files": [...]}
    with the latest run last.
    """
    files = sorted(glob.glob(directory + "/run_*.csv"))
    if len(files) == 0:
        raise ValueError("no run_*.csv files found in '" + directory + "'")
    return {"files": files}

def gauss_plus_line(x, a, mu, sigma, b0, b1):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x

@mcp.tool()
def fit_pi0_peak(file: str, window_lo: float, window_hi: float) -> dict:
    """Fit the pi0 -> two-photon invariant-mass peak in one run file.

    The pi0 meson decays to two photons; the invariant mass computed from photon
    pairs peaks at 0.135 GeV over a smooth background. Histograms masses between
    window_lo and window_hi (GeV; bracket 0.135, e.g. 0.05-0.25) and fits Gaussian
    plus linear background. Returns mean, sigma, and event count with errors.
    """
    if window_lo >= window_hi:
        raise ValueError("window_lo must be less than window_hi")
    masses = np.loadtxt(file)                      # missing file raises clearly
    masses = masses[(masses > window_lo) & (masses < window_hi)]
    if len(masses) < 100:
        raise ValueError("only " + str(len(masses)) + " events in window; widen it")
    counts, edges = np.histogram(masses, bins=60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p0 = [counts.max(), 0.135, 0.012, counts.min(), 0.0]
    popt, pcov = curve_fit(gauss_plus_line, centers, counts, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return {"mean_gev": float(popt[1]), "mean_err": float(perr[1]),
            "sigma_gev": float(popt[2]), "sigma_err": float(perr[2]),
            "n_events_in_window": int(len(masses))}

if __name__ == "__main__":
    mcp.run()      # stdio transport by default
```

(The `if __name__ == "__main__":` guard runs `mcp.run()` only when the file is
executed directly, not when a test imports it — the same pattern as your Week 04
`run.py`.)

Read what FastMCP did for you: each function's *name* became the tool name, its
*docstring* became the description the model reads, and its *type hints* became the
input schema. All of Week 37's definition dicts, generated. The design lessons carry
over unchanged — the docstring is still prompt engineering, and each one here also
carries its two sentences of physics, because the model reading it has no other way
to know what an energy scale or a run is.

`mcp.run()` blocks, waiting on stdin for a client. Running `python server.py` at a
terminal therefore looks like a hang — that is correct behavior; servers are meant
to be *launched by clients*, not chatted with by humans.

## 5. Resources: the run index

The run index is data to consult, not an action — so it is a resource, addressed by
a URI (a name in `scheme://path` form; the scheme is yours to invent):

```python
@mcp.resource("runs://index")
def run_index() -> str:
    """The run index: one line per run with its status, as JSON."""
    with open("run_index.json") as f:
        return f.read()
```

A client can list available resources and read this one by its URI; a host
application might attach it to the conversation as context before the model ever
speaks. One design note: resources are returned whole. If your run index were a
million lines, you would expose a *search tool* over it instead of a resource —
the same "don't dump the haystack into the context window" judgment you made when
chunking documents for RAG in Week 34.

## 6. Testing the server without a model

Your tools are ordinary Python functions, and that is a superpower: test them with
`pytest`, no model, no network, no tokens. In `tests/test_server.py`:

```python
import pytest
from server import fit_pi0_peak, get_calibration, list_run_files

def test_fit_finds_the_peak():
    result = fit_pi0_peak("data/run_00042.csv", 0.05, 0.25)
    assert abs(result["mean_gev"] - 0.135) < 0.005
    assert result["sigma_gev"] > 0

def test_bad_window_raises_with_useful_message():
    with pytest.raises(ValueError, match="window_lo must be less"):
        fit_pi0_peak("data/run_00042.csv", 0.30, 0.10)

def test_unknown_run_names_the_known_ones():
    with pytest.raises(ValueError, match="known runs"):
        get_calibration(99999)
```

(`pytest.raises` is the standard way to assert "this call must raise"; `match`
checks the error message — which matters here, because the message is exactly what
the model will read when it gets the error back.)

A live model conversation can never reliably tell you what these tests tell you:
that the fit is *numerically* right, and that every failure path produces a message
a model can act on. Test the tools cold; let the model exercise only what is
already known to work.

## 7. Consuming the server from Python

Now the other side of the wire. The SDK's client launches your server as a
subprocess and speaks stdio JSON-RPC to it. One new construct first: the client API
is **asynchronous**. An `async def` function is one that can *pause* at points
marked `await` while it waits for I/O (here: the server's replies), letting Python
do other things meanwhile; `asyncio.run(...)` starts such a function from normal
code, and `async with` is the async version of the `with` blocks you know from
opening files. For this course, treat `async`/`await` as required punctuation for
this library — write the `await`s where the examples put them.

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                 # protocol handshake

            tools = await session.list_tools()
            for t in tools.tools:
                print(t.name, "-", t.description.splitlines()[0])

            result = await session.call_tool(
                "fit_pi0_peak",
                {"file": "data/run_00042.csv",
                 "window_lo": 0.05, "window_hi": 0.25},
            )
            print(result.content[0].text)

asyncio.run(main())
```

`initialize()` is the handshake where client and server exchange versions and
capabilities; `list_tools()` returns exactly the names, descriptions, and schemas
FastMCP generated in §4 — this script is how you *verify* the schema says what you
think it says. `call_tool` returns content blocks (familiar shape), and if your
function raised, the result comes back with an error flag and your message as text
instead of crashing anything — the protocol-level twin of Week 37's `is_error`.

To close the loop model-side, you could feed `session.list_tools()` output into a
Week 37-style agentic loop, translating each `tool_use` block into a
`session.call_tool(...)`. That glue — MCP tools driving the Messages API loop — is
exactly what you will assemble in Week 40's copilot, so keep this script.

## 8. Plugging into a real host, and debugging

**Claude Code as the host.** Any MCP host can now use your server. For Claude Code,
you register the launch command (check the current docs for the exact syntax; at
the time of writing):

```bash
claude mcp add physics -- uv run python server.py
```

Then ask, in a Claude Code session in that project: *"Find the latest run in data/
and fit its pi0 peak."* Watch it discover your tools, call two of them, and quote
your fitted mass — a stock, unmodified client driving analysis code you wrote,
which is the entire point of a protocol.

**MCP Inspector.** For protocol-level debugging there is an official inspector — a
small web UI that connects to your server, shows every JSON-RPC message, and lets
you call tools by hand with arbitrary arguments (`npx @modelcontextprotocol/
inspector python server.py`, or see the current docs). Use it when a tool works in
pytest but misbehaves through a client: the inspector shows you whether the schema,
the arguments, or the result is the thing that is wrong.

**Server-side logging.** One transport gotcha: with stdio, *standard output is the
wire* — a stray `print()` in a tool corrupts the JSON-RPC stream and produces
baffling client-side parse errors. Log to standard error or a file instead
(`print("...", file=sys.stderr)`, or the `logging` module). If your server "works
in the inspector but breaks in Claude Code," a rogue `print` is suspect number one.

## 9. The error contract

Collect the error-handling rules in one place, because Week 40's unattended copilot
lives or dies by them:

1. **Validate first, compute second.** Check windows, files, and run numbers before
   any numerics, and raise `ValueError` with a message that says what was wrong
   *and what would be right* ("bracket 0.135, e.g. 0.05-0.25"). Your error messages
   are model-facing prose.
2. **Never crash the server.** A raised exception inside a tool becomes an error
   *result* to the client; the server keeps serving. What you must avoid is dying
   outside a tool call — corrupt config at startup, a rogue `print` to stdout.
3. **Fail closed on bad data.** The `len(masses) < 100` guard means a nearly-empty
   window returns "widen it" instead of a garbage fit with huge uncertainties that
   a model might quote as physics. An agent cannot apply judgment you did not
   encode; the tool is where "would a physicist accept this?" gets enforced.

## 10. Worked example: end to end

The full assembly, as the exercises build it:

1. Generate the toy data and `calibrations.json` (setup, given).
2. `server.py` from §4–§5: three tools plus the run-index resource. Run
   `pytest -q` — all green, including the failure-path tests.
3. `client_check.py` from §7: lists three tools, prints the first docstring line of
   each, calls `fit_pi0_peak` on run 42, prints `mean_gev = 0.135 ± 0.001`-ish.
4. Register with Claude Code and ask for "the latest run's pi0 mass, using the good
   calibration" — transcript shows `list_run_files`, `get_calibration`, and
   `fit_pi0_peak` calls, then prose quoting your numbers.
5. Break it on purpose: ask for run 99999, watch the "known runs: ..." error come
   back and the model recover by picking a run that exists.

Step 5 is the one to savor. You wrote no recovery logic in the host, no retry
prompt, nothing — a good error message plus the standard protocol was enough. That
is what "your analysis code is now a subsystem" means.

## Check yourself

1. What M×N problem does MCP solve, and how does a protocol turn it into M+N?
2. Host, client, server: which one is Claude Code, and which one did you write?
3. Tool vs. resource: where does `fit_pi0_peak` go, where does the run index go,
   and what is the deciding question?
4. What do FastMCP's generated schemas come from, mechanically? Name both pieces.
5. Why does a stray `print()` inside a tool break a stdio server, and what do you
   use instead?
6. Name two things `pytest` on the tool functions catches that a live model
   conversation never reliably will.
7. Your tool raises `ValueError("bad input")`. What does the client see, and why is
   that message string worth writing carefully?
8. When would you expose a search tool instead of a resource for the same data?

## Answers

1. M hosts × N tool providers would need M×N custom integrations; with a shared
   protocol each side implements it once (M+N) and any pair interoperates.
2. Claude Code is the host (it holds a client object per server); you wrote the
   server.
3. `fit_pi0_peak` is a tool (an action the model chooses, results vary per call);
   the run index is a resource (read-only content the application attaches). The
   question: is it something to *execute* or something to *read*?
4. Type hints (argument names/types → schema properties) and the docstring (→ the
   description the model reads).
5. With stdio, stdout carries the JSON-RPC messages, so extra text corrupts the
   stream mid-message. Log to stderr or a file.
6. Numerical correctness of the fit against known truth, and that every invalid
   input takes the intended failure path with the intended message — a model
   conversation samples paths randomly and can't check the numbers.
7. An error-flagged tool result whose text is your message. The model reads it to
   decide what to do next, so it should state the problem and the fix.
8. When the data is too large to hand over whole — return targeted slices on demand
   instead of dumping the haystack into the context window (the Week 34 chunking
   judgment).

## New terms

- **protocol** — agreed message format plus rules for who sends what when.
- **MCP** — Model Context Protocol; the open standard connecting model hosts to
  tool/data servers.
- **host / client / server** — the user-facing app; its per-server connector; the
  capability provider you write.
- **JSON-RPC** — remote procedure calls as JSON: `method`, `params`, `id` →
  `result` or `error` with matching `id`.
- **transport** — how messages travel: **stdio** (subprocess pipes) or **streamable
  HTTP** (network).
- **tool / resource / prompt (MCP)** — model-invoked function / app-attached
  read-only data at a URI / user-selected template.
- **URI** — a `scheme://path` identifier, e.g. `runs://index`.
- **type hint** — `arg: int` annotations; ignored at runtime, read by libraries to
  generate schemas.
- **decorator** — `@f` above a `def`: passes the function to `f` to register or
  wrap it.
- **`async` / `await` / `asyncio.run`** — functions that pause at I/O waits; the
  marker for pause points; the entry point that runs one.
- **`pytest.raises`** — asserting that a call raises, optionally matching the
  message.
- **MCP Inspector** — official web UI for watching protocol traffic and calling
  tools by hand.
- **calibration constant** — per-run correction for detector response drift,
  applied so measured masses land at truth.
- **run** — a contiguous numbered data-taking period; the unit an analysis
  discovers and loops over.

## Going deeper

- MCP documentation (modelcontextprotocol.io) — the concepts pages: architecture,
  tools, resources, transports. Read tools thoroughly; skim resources and prompts.
- MCP Python SDK documentation (the official `mcp` package) — the FastMCP
  quickstart, and how tool schemas are generated from signatures; the client
  examples §7 is based on.
- Claude Code docs, MCP section — registering and managing local stdio servers in
  the host you already use.
- MCP Inspector docs (by title) — protocol-level debugging; ten minutes here saves
  hours of client-side guessing.
