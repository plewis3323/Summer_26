# Week 37 — Tool Use from First Principles

~4 hrs. Before starting you should be able to: write and run Python scripts and
functions (Week 02), fit a Gaussian peak with `scipy.optimize.curve_fit` (Week 04 /
Week 08), and explain what a token and a language model's next-token prediction are
(Week 27). No web-programming background is assumed — this week teaches it.

Everything you have run so far in this course ran on your own machine. This week your
code talks to a model running on someone else's machine — Anthropic's Claude, over the
internet — and, more importantly, you let that model call *your* functions: the peak
fitter you wrote in Phase 1 becomes a tool the model can decide to use. That loop
(model asks, your code runs, model reads the result) is the core mechanism behind
every "AI agent," and by the end of the week you will have written it by hand.

A standing note for this whole month: **check the current docs — this API evolves.**
The concepts below (requests, JSON, tools, the loop) are stable; exact parameter names
and model names change every few months. When code from this lesson disagrees with
the official documentation, the documentation wins.

## 1. What an HTTP API is

When you type a URL into a browser, the browser sends a *request* across the internet
to a *server* (a program on another machine that waits for requests), and the server
sends back a *response* — usually a web page. That request/response conversation uses
a protocol called **HTTP**: an agreed-upon format for the bytes, so that any client
can talk to any server.

An **API** (Application Programming Interface) is the same idea aimed at programs
instead of people. Instead of a web page meant for human eyes, the server returns
structured data meant for code. A **web API** is just a set of URLs where, if you send
a correctly formatted request, you get back a correctly formatted answer.

An HTTP request has four parts:

- a **method** — what kind of action. `GET` means "give me data"; `POST` means "here
  is data, do something with it." Calling a model is a `POST` (you send a prompt).
- a **URL** — which server, and which *endpoint* (function) on that server.
  For Claude it is `https://api.anthropic.com/v1/messages`.
- **headers** — metadata as key/value pairs: who you are, what format the body is in.
- a **body** — the actual payload (your prompt, settings, tool definitions).

The response has a **status code** (a number: `200` means success, `400` means "your
request was malformed", `401` means "you are not authorized", `429` means "slow down —
too many requests", `500`-series means the server itself had a problem) plus headers
and a body of its own.

You will almost never build these requests by hand — Anthropic publishes a Python
package (an **SDK**, Software Development Kit) that does it for you. But when
something breaks, the error message will be in HTTP vocabulary, so you need the
vocabulary.

## 2. What JSON is

Request and response bodies need a text format both sides understand. The universal
choice is **JSON** (JavaScript Object Notation). You already know it, almost: it looks
like Python dictionaries and lists written down.

```json
{
  "run_number": 42,
  "particle": "pi0",
  "mass_gev": 0.135,
  "good_run": true,
  "photons": [1, 2]
}
```

Differences from Python: JSON uses `true`/`false`/`null` instead of
`True`/`False`/`None`, keys must be double-quoted strings, and there are no comments
or trailing commas. Python's standard `json` module converts both ways:

```python
import json

d = {"run_number": 42, "mass_gev": 0.135}
text = json.dumps(d)        # dict -> JSON string: '{"run_number": 42, ...}'
back = json.loads(text)     # JSON string -> dict
print(back["run_number"])   # 42
```

Every message you send to the model, and everything it sends back — including its
requests to call your tools — travels as JSON. The SDK converts to and from Python
dicts, so in practice you work with dicts and rarely see the raw text.

## 3. API keys, and keeping them safe

Anthropic charges per token processed, so the server must know who is asking. That is
what an **API key** is: a long secret string tied to your account, sent in a request
header. Anyone who has your key can spend your money, which leads to three rules:

1. **Never write the key in your code.** Code gets committed to git, pushed to GitHub,
   pasted into chats. Keys leaked this way get found and abused within minutes —
   automated scanners watch public repositories for exactly this.
2. **Put it in an environment variable.** An environment variable is a named value the
   operating system keeps for the current terminal session, outside any file. Set it
   once per session in your shell, then read it from Python:

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."   # in the terminal, not in a .py file
   ```

   ```python
   import os
   key = os.environ["ANTHROPIC_API_KEY"]   # read it (the SDK does this for you)
   ```

3. **If a key ever touches a committed file, revoke it** in the Anthropic console and
   make a new one. Deleting the line later does not help — git history remembers.

Get a key at the Anthropic console (search for "Anthropic console API keys"; you will
need to add a small amount of credit). Budget note: this month's exercises cost real
money — typically a few dollars total, and exercise E7 has you measure it exactly.
Prices are listed per million tokens and differ by model; the lesson's examples use a
current large model, but nothing below depends on which model you pick, and smaller
models are several times cheaper. Check the current pricing page and decide for
yourself.

## 4. Your first API call

Install the SDK into this week's project:

```bash
uv add anthropic
```

The smallest complete program:

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment

response = client.messages.create(
    model="claude-opus-5",       # check the docs for current model names
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "In one sentence: what is a pi0 meson?"}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

Walk through the pieces:

- `client` wraps the HTTP machinery: it builds the `POST` request to
  `/v1/messages`, attaches your key header, converts your dicts to JSON, and converts
  the JSON response back into a Python object.
- `messages` is the conversation so far: a list of turns, each a dict with a `role`
  (`"user"` — you, or `"assistant"` — the model) and `content`. The API is
  **stateless**: the server remembers nothing between calls, so a multi-turn
  conversation means re-sending the whole list every time, one entry longer.
- `max_tokens` is a hard cap on how much the model may generate — your cost circuit
  breaker.
- `response.content` is a **list of content blocks**, not a plain string. Today it
  holds one block of `type == "text"`. The list matters because tool calls (next
  section) arrive as a different block type in this same list. Always check
  `block.type` before reading `block.text`.

Two response fields you should print in every experiment this week:

- `response.stop_reason` — *why* the model stopped: `"end_turn"` (finished
  naturally), `"max_tokens"` (hit your cap; the answer is truncated), or
  `"tool_use"` (it wants to call one of your tools — the hinge of this whole lesson).
- `response.usage` — token accounting: `usage.input_tokens` and
  `usage.output_tokens`. Multiply by the per-token price and you know exactly what
  the call cost.

## 5. Defining a tool: JSON Schema

A language model generates text. It cannot run code, read your disk, or fit a peak.
**Tool use** is a contract that fakes it convincingly: you describe your functions to
the model, and instead of answering in prose it may reply "please run function *F*
with arguments *A*." Your code runs the function and sends the result back. The model
never executes anything — *your* Python does, which means you keep control of what
can actually happen on your machine.

To describe a function you need a machine-readable way to say "this function takes a
string called `file` and two numbers called `window_lo` and `window_hi`." That
language is **JSON Schema**: a JSON document that describes the shape of other JSON.
A tool definition has three fields:

```python
fit_pi0_peak_tool = {
    "name": "fit_pi0_peak",
    "description": (
        "Fit the pi0 -> two-photon invariant-mass peak in one run file. "
        "Histograms the masses between window_lo and window_hi (GeV) and fits a "
        "Gaussian plus linear background. Returns the fitted mean, width, and "
        "signal count with uncertainties. Use a window that brackets 0.135 GeV, "
        "e.g. 0.05 to 0.25."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Path to the run file, e.g. data/run_00042.csv",
            },
            "window_lo": {
                "type": "number",
                "description": "Lower edge of the fit window in GeV",
            },
            "window_hi": {
                "type": "number",
                "description": "Upper edge of the fit window in GeV",
            },
        },
        "required": ["file", "window_lo", "window_hi"],
    },
}
```

Reading the schema: the input is an `object` (a dict), with three named `properties`,
each with a JSON type (`string`, `number`, `integer`, `boolean`, `array`, `object`)
and a human-readable description; `required` lists which must be present.

Two design lessons that matter more than any syntax:

- **The description is prompt engineering.** The model decides *whether* and *how* to
  call your tool by reading the name and descriptions — they are the only
  documentation it has. Say what the tool does, when to use it, what the arguments
  mean, and what comes back. Vague descriptions produce wrong calls.
- **How does the model "decide" to call a tool?** Nothing mystical: the tool
  definitions are rendered into its context as text, and the model has been trained
  that when a tool would help, it should generate a specially formatted block naming
  the tool and its arguments instead of prose. The API parses that block for you.
  It is still next-token prediction (Week 27) — just prediction of a very
  structured kind.

And the physics behind this particular tool, since it is our running example: a
**π⁰ (pi-zero) meson** is a light unstable particle that decays almost instantly into
two photons. If a detector measures both photons' energies and directions, the
**invariant mass** — a quantity computable from the two measurements that equals the
parent particle's mass regardless of how fast it was moving — piles up in a peak at
the π⁰ mass, 0.135 GeV, on top of a smooth background of photon pairs that did not
come from the same π⁰. Fitting that peak, exactly as you did for the J/ψ in Week 04,
measures the mass and yield; the toy run files this week contain simulated
invariant-mass values shaped like that.

## 6. One tool call, by hand

Pass the tool list via `tools=`, and implement the actual function:

```python
import json
import numpy as np
from scipy.optimize import curve_fit

def gauss_plus_line(x, a, mu, sigma, b0, b1):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x

def fit_pi0_peak(file, window_lo, window_hi):
    masses = np.loadtxt(file)
    masses = masses[(masses > window_lo) & (masses < window_hi)]
    counts, edges = np.histogram(masses, bins=60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p0 = [counts.max(), 0.135, 0.012, counts.min(), 0.0]
    popt, pcov = curve_fit(gauss_plus_line, centers, counts, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return {
        "mean_gev": float(popt[1]), "mean_err": float(perr[1]),
        "sigma_gev": float(popt[2]), "sigma_err": float(perr[2]),
        "n_events_in_window": int(len(masses)),
    }

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=1000,
    tools=[fit_pi0_peak_tool],
    messages=[{"role": "user",
               "content": "Fit the pi0 peak in data/run_00042.csv and report the mass."}],
)
print(response.stop_reason)   # -> "tool_use"
```

With `stop_reason == "tool_use"`, `response.content` contains a block of
`type == "tool_use"` carrying three things: `block.name` (which tool), `block.input`
(the arguments, already parsed into a dict), and `block.id` — a unique ID for *this
particular call*. To answer, you append two turns to the conversation: the model's
own turn (verbatim), then a user turn containing a `tool_result` block whose
`tool_use_id` repeats that ID:

```python
tool_block = None
for block in response.content:
    if block.type == "tool_use":
        tool_block = block

result = fit_pi0_peak(tool_block.input["file"],
                      tool_block.input["window_lo"],
                      tool_block.input["window_hi"])

messages = [
    {"role": "user",
     "content": "Fit the pi0 peak in data/run_00042.csv and report the mass."},
    {"role": "assistant", "content": response.content},        # the model's turn, verbatim
    {"role": "user", "content": [
        {"type": "tool_result",
         "tool_use_id": tool_block.id,                          # must match block.id
         "content": json.dumps(result)},
    ]},
]

final = client.messages.create(model="claude-opus-5", max_tokens=1000,
                               tools=[fit_pi0_peak_tool], messages=messages)
```

The `tool_use_id` is what ties an answer to a question. The model may issue several
calls at once; if two IDs were swapped, the fit result would be silently attributed
to the wrong request and the model would reason from wrong data — the conversation
equivalent of mislabeling two histograms.

## 7. The agentic loop

Section 6 handled exactly one call. The general pattern — the model may need zero,
one, or many tool calls, feeding each result into the next decision — is a `while`
loop. This loop is the entire secret of "agents":

```python
def run_agent(question, tools, tool_functions):
    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(
            model="claude-opus-5", max_tokens=2000,
            tools=tools, messages=messages,
        )
        if response.stop_reason != "tool_use":
            break                                   # model is done talking to tools

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = tool_functions[block.name]
                try:
                    out = fn(**block.input)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(out)})
                except Exception as err:
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": "Error: " + str(err),
                                    "is_error": True})
        messages.append({"role": "user", "content": results})

    for block in response.content:
        if block.type == "text":
            return block.text
```

Three constructs here deserve a note:

- **`fn(**block.input)`** — the `**` unpacks a dict into keyword arguments:
  `fn(**{"file": "x.csv", "window_lo": 0.05})` is `fn(file="x.csv", window_lo=0.05)`.
  New syntax for this course; this is the one place you need it.
- **`try`/`except`** — Python's mechanism for catching errors instead of crashing.
  Code in the `try` block runs normally; if any line raises an exception, execution
  jumps to `except`, with the exception object bound to `err`. We have let exceptions
  crash our programs until now, which is right for analysis code — but here a bad
  tool call from the model should not kill the loop.
- **`is_error: True`** — how you tell the model its call failed. Send the error
  message as the tool result with this flag set, and the model will typically read
  the message, correct its arguments, and retry — error recovery you get for free,
  *if* your error messages say what went wrong ("window_lo must be less than
  window_hi", not just "ValueError").

**Parallel tool calls.** One assistant turn may contain *several* `tool_use` blocks —
the model asking for three independent fits at once. The loop above already handles
it: it collects every result and sends them back in **one** user message. That is the
rule to remember. Sending each result as its own message breaks the turn structure —
and quietly teaches the model to stop batching its requests.

**Safety and cost guards.** A `while True` loop that costs money per iteration wants
a leash: cap iterations (`for _ in range(10)` instead of `while True`), and log
`response.usage` each pass. Also think about what your tools *can do* — a fitter that
reads files it is given is safe; a tool that deletes files or runs shell commands
means the model's mistakes become your mistakes. Start with read-only tools.

## 8. Structured output

A different, humbler use of the same machinery: sometimes you don't want an agent,
you just want the model to return **data in a guaranteed shape** — say, extract
`{particle, mass_gev, width_gev, n_events}` from a paragraph of analysis prose.
Prompting "please respond in JSON" mostly works, until the model adds a chatty
preamble and your `json.loads` crashes at 2 a.m.

The API can instead *constrain* the output to match a schema — same JSON Schema
language as tool inputs:

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=500,
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "particle":  {"type": "string"},
                    "mass_gev":  {"type": "number"},
                    "width_gev": {"type": "number"},
                    "n_events":  {"type": "integer"},
                },
                "required": ["particle", "mass_gev", "width_gev", "n_events"],
                "additionalProperties": False,
            },
        }
    },
    messages=[{"role": "user", "content": "Extract the measurement: " + paragraph}],
)
data = json.loads(response.content[0].text)   # guaranteed to parse and match
```

The SDK also offers a convenience layer, `client.messages.parse(...)`, which takes a
*Pydantic model* — a Python class that declares typed fields and validates data
against them — and hands you back a validated object instead of a dict. Worth
knowing the name; the raw-schema form above is all this course needs. Either way,
the schema is enforced by the API, so the guarantee is structural, not hopeful.
(Constrained output fixes the *shape*, not the *facts* — a wrong number that fits
the schema still parses. Your Week 32 evaluation habits still apply.)

## 9. Worked example: two tools, one question

Everything above, in one runnable script. Setup: a folder `data/` with toy run files
(the exercises notebook generates them — each is one column of simulated diphoton
invariant masses with a π⁰ peak at 0.135 GeV on a smooth background).

```python
import glob
import json
import anthropic
# fit_pi0_peak, gauss_plus_line, fit_pi0_peak_tool as defined in sections 5-6

client = anthropic.Anthropic()

def list_run_files(directory):
    return {"files": sorted(glob.glob(directory + "/run_*.csv"))}

list_run_files_tool = {
    "name": "list_run_files",
    "description": ("List the run data files in a directory. Returns file paths "
                    "sorted by run number, so the last entry is the latest run."),
    "input_schema": {
        "type": "object",
        "properties": {
            "directory": {"type": "string",
                          "description": "Directory to search, e.g. 'data'"},
        },
        "required": ["directory"],
    },
}

tools = [list_run_files_tool, fit_pi0_peak_tool]
tool_functions = {"list_run_files": list_run_files, "fit_pi0_peak": fit_pi0_peak}

answer = run_agent(
    "Find the latest run file in the data directory and fit its pi0 peak. "
    "Report the fitted mass with its uncertainty.",
    tools, tool_functions,
)
print(answer)
```

Run it and read the transcript you logged: the model calls `list_run_files`, reads
the file list from the result, picks the last file, calls `fit_pi0_peak` with a
sensible window (it learned the window advice from the *description* — that sentence
earned its keep), and only then writes prose quoting your fitted numbers. Two tool
round-trips, chosen and sequenced by the model, executed by your code. That is an
agent; everything in Weeks 38–40 is arrangement and discipline on top of this loop.

## Check yourself

1. What are the four parts of an HTTP request, and which part carries your API key?
2. A response comes back with status code 429. What happened, and is retrying
   reasonable?
3. Why must an API key live in an environment variable rather than in your script?
4. The API is stateless. What does your code have to do, because of that, to hold a
   multi-turn conversation?
5. In the agentic loop, what exact condition ends the loop, and what two things must
   your code append to `messages` each iteration it continues?
6. What is `tool_use_id` for, and what would go wrong if two of them were swapped?
7. The model calls `fit_pi0_peak` with `window_lo=0.3, window_hi=0.1` and your
   function raises. What should you send back, and what do you expect the model to
   do next?
8. When should you reach for structured output instead of a tool-using agent?

## Answers

1. Method, URL, headers, body. The key travels in a header.
2. Rate limit: you sent too many requests too quickly. Yes — wait and retry, ideally
   with increasing delays.
3. Anything in a script ends up in git history and can leak publicly; a leaked key
   lets a stranger spend your money. Environment variables live only in your terminal
   session.
4. Re-send the entire `messages` list every call, appending each new user and
   assistant turn — the server remembers nothing.
5. The loop ends when `stop_reason != "tool_use"`. While it continues, append the
   assistant turn (`response.content` verbatim) and then one user turn containing a
   `tool_result` block for every `tool_use` block.
6. It ties each result to the specific call it answers. Swapped IDs silently feed
   each question the other's answer; the model reasons from wrong data with no error
   raised anywhere.
7. A `tool_result` with `is_error: True` and a message that names the problem
   ("window_lo must be less than window_hi"). Expect the model to read it, fix the
   arguments, and call again.
8. When the task is a single transformation into a known shape — extraction,
   classification, reformatting — with no decisions about *which* computation to run.
   No loop, one call, guaranteed parse.

## New terms

- **HTTP** — the request/response protocol of the web; methods like GET and POST.
- **API / endpoint** — a programmatic interface served over URLs; one URL+method
  pair that does one job.
- **status code** — numeric result of a request (200 OK, 400 bad request, 401
  unauthorized, 429 rate-limited, 5xx server error).
- **JSON** — text format for structured data; maps to Python dicts/lists via
  `json.dumps` / `json.loads`.
- **API key** — secret string identifying (and billing) your account; kept in an
  environment variable.
- **environment variable** — a named value held by the shell session, read with
  `os.environ`; keeps secrets out of files.
- **SDK** — a vendor's client library wrapping the raw HTTP API.
- **content block** — one typed element of a message's content list (`text`,
  `tool_use`, `tool_result`).
- **stop_reason** — why generation ended: `end_turn`, `max_tokens`, `tool_use`.
- **tool / tool use** — a function you describe to the model; it requests calls,
  your code executes them.
- **JSON Schema** — JSON that describes the required shape of other JSON; used for
  tool inputs and structured output.
- **agentic loop** — model → tool call → result → model, repeated until the model
  stops requesting tools.
- **`is_error`** — flag on a `tool_result` marking a failed call so the model can
  recover.
- **structured output** — API-enforced guarantee that the response text matches a
  schema.
- **`try`/`except`** — catching an exception instead of crashing.
- **`**kwargs` unpacking** — `fn(**d)` passes dict `d` as keyword arguments.
- **invariant mass / π⁰** — mass reconstructed from decay products' measured
  energies and angles; the π⁰ decays to two photons and peaks at 0.135 GeV.

## Going deeper

- Anthropic docs, "Messages API" and "Tool use" overview pages — the authoritative,
  current version of everything here; read the tool-definition best practices.
- Anthropic docs, "Structured outputs" — the full schema-support list and the
  `parse()` helper.
- Any OpenAI "function calling" guide — skim to see that every provider ships the
  same shape (schema list, tool-call turn, tool-result turn); your loop is
  provider-agnostic.
- Anthropic docs, prompt caching page — why tool definitions render first in the
  prompt and what that means for cost; skim now, use in Week 40.
