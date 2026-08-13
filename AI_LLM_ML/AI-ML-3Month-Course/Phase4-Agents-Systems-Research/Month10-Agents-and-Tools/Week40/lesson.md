# Week 40 — Multi-Agent Orchestration, Agent Evals, and the Copilot Prototype

~3 hrs of core material; the balance of the week goes to the mini-project
(`project.md`). Before starting you should be able to: run the hand-written
agentic loop (Week 37), implement orchestrator–workers and apply the
when-not-to-build-an-agent checklist (Week 38), serve and test your physics
tools over MCP (Week 39), and recall the LLM-as-judge failure modes (Week 32).

This week closes the month with the two things still missing between "a loop
that works when I watch it" and "a tool I would hand a colleague": teams of
agents with disciplined handoffs, and — more important — *measurement*. You
would never quote a physics result without an uncertainty analysis; the agent
equivalent is a task suite, a success rate, a cost figure, and a reviewed
failure. By Friday you will have all four, attached to a working
analysis-copilot prototype driving your Week-39 MCP server.

Same standing note as all month: **check the current docs — this API evolves.**
Concepts here are stable; exact parameter and attribute names drift.

## 1. From one loop to a team

Week 38's orchestrator–workers pattern had a model *plan* subtasks while plain
Python executed them. A **multi-agent system** takes the next step: the workers
are themselves agents — each with its own conversation, its own tool access, its
own loop. The **orchestrator** is now an agent whose "tools" include *delegating
to other agents*.

Why would you ever want this, given that one loop already works?

- **Context isolation.** A worker fitting run 42 does not need the three
  thousand tokens of conversation about runs 40, 41, and 43. Giving each worker
  a fresh, small context keeps every conversation short — and Week 38 §9 showed
  you why short conversations matter: the stateless API re-sends everything,
  every iteration.
- **Parallelism.** Independent workers can run concurrently. Four fits that
  take thirty seconds each finish in thirty seconds, not two minutes.
- **Specialization.** Each worker gets only the tools and instructions its
  subtask needs. A fit worker with two tools and a five-line brief makes fewer
  wrong calls than a generalist with six tools and a page of context.

And the cost, stated now and quantified in §3: every delegation is a
*conversation boundary*. Nothing crosses it except what you explicitly send.
That is both the feature (isolation) and the bill (re-briefing).

The Week 38 checklist still governs. Most tasks that fit in one loop should
stay in one loop; a team of agents is the *most* agentic design on the ladder,
and it has to out-measure the single loop before you ship it — which is what
this week's evaluation machinery is for.

## 2. Handoffs: briefs out, reports back

A **handoff** is the transfer of work across a conversation boundary. The
discipline that makes multi-agent systems debuggable is to make both directions
*structured*:

- The **task brief** is what the orchestrator sends a worker: everything the
  worker needs, because the worker starts blank. It has no memory of the parent
  conversation, no access to its transcript, no idea why the task exists unless
  the brief says so.
- The **worker report** is what comes back: a fixed schema, not free prose, so
  the orchestrator's merge step is code reading fields — Week 37's structured
  output, doing exactly the job it was built for.

Both directions get JSON Schemas, the same language you have used since
Week 37:

```python
import json

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id":  {"type": "string"},
        "status":   {"type": "string"},   # "ok" or "failed"
        "mean_gev": {"type": "number"},
        "mean_err": {"type": "number"},
        "notes":    {"type": "string"},
    },
    "required": ["task_id", "status", "mean_gev", "mean_err", "notes"],
    "additionalProperties": False,
}

def run_worker(brief):
    question = (
        "You are a fit worker. Your entire job is the task below. Use the "
        "tools to complete it.\n\nTask brief:\n" + json.dumps(brief)
    )
    answer = run_agent(question, tools, tool_functions)     # Week 37 loop
    report = client.messages.create(
        model="claude-opus-5",
        max_tokens=500,
        output_config={"format": {"type": "json_schema", "schema": REPORT_SCHEMA}},
        messages=[{"role": "user",
                   "content": "Convert this result into the report format. "
                              "Task id: " + brief["task_id"] + "\n\n" + answer}],
    )
    return json.loads(report.content[0].text)

briefs = [
    {"task_id": "fit-" + f,
     "objective": "Fit the pi0 peak and report the mean with its uncertainty.",
     "file": f, "window_lo": 0.05, "window_hi": 0.25}
    for f in list_run_files("data")["files"]
]
reports = [run_worker(b) for b in briefs]
```

(The report-formatting call is a second, cheap request per worker; you can also
build the schema into the loop's final turn — the exercises accept either.)

Here the fan-out is code — one brief per file, a workflow shape. Swap in a
planning call that *writes* the briefs (Week 38 §5) and the orchestrator
becomes a model; either way the two rules of handoffs hold:

1. **The brief is self-contained.** If the worker would need to ask "which
   window?", the brief failed. Write briefs the way you would write instructions
   for a new student who missed every previous meeting — because that is
   literally the worker's situation.
2. **The report is validated before it is merged.** Check required fields and
   types (or lean on schema-enforced output) and treat a non-validating report
   as a failed task, not a crash. Worker output is model output; Week 38's
   parse-defensively rule applies at every boundary.

## 3. What a handoff costs

Multi-agent systems have a specific cost anatomy, and you should be able to
sketch it before building one:

- **Context re-establishment.** Every worker pays tokens to be told things the
  orchestrator already knew: the brief, the tool definitions, any standing
  instructions. N workers ⇒ N copies of that overhead.
- **Report round-trips.** Results come back as tokens the orchestrator must
  then *re-read* as input when it merges. Data crosses the boundary twice.
- **The offsetting saving.** Each worker's loop is short, and short loops are
  where the quadratic re-send cost (Week 38 §9) is small. One long conversation
  of 6 tool calls re-sends far more than three short conversations of 2.

So the crossover is about *shape*: multi-agent wins when the task splits into
independent, tool-heavy chunks whose intermediate detail the parent never needs
(the four-file fit comparison — the orchestrator wants five numbers back, not
four fitting transcripts). It loses when steps are coupled and every decision
needs the full history — there, re-briefing costs more than the isolation
saves, and one loop is cheaper *and* more coherent. E1's ledger comparison
makes you measure this rather than take it on faith.

One more cost that does not show up in the token ledger: **error opacity**.
When a single loop goes wrong, the whole story is in one transcript. When a
team goes wrong, the bug may live in a brief, a worker, a report, or the merge
— four places to look. Structured handoffs are what keep that searchable; they
are to multi-agent systems what run logs are to a detector.

## 4. The glue: MCP tools in the agentic loop

Week 39 §7 ended with a promise: feed `session.list_tools()` into the Week 37
loop and translate each `tool_use` into a `session.call_tool(...)`. That glue
is the heart of the copilot, and it is about twenty lines.

Two translations. First, MCP tool listings → Messages API tool definitions
(same information, slightly different field names):

```python
def mcp_tools_as_api_tools(listing):
    return [{"name": t.name,
             "description": t.description,
             "input_schema": t.inputSchema}
            for t in listing.tools]
```

Second, `tool_use` blocks → protocol calls. Because the MCP client is async
(Week 39 §7), the whole loop moves inside an `async def`:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_copilot(question):
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = mcp_tools_as_api_tools(await session.list_tools())

            messages = [{"role": "user", "content": question}]
            for _ in range(10):                       # leash, as in Week 37
                response = client.messages.create(
                    model="claude-opus-5", max_tokens=2000,
                    tools=tools, messages=messages,
                )
                if response.stop_reason != "tool_use":
                    break
                messages.append({"role": "assistant",
                                 "content": response.content})
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await session.call_tool(block.name, block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.content[0].text,
                            "is_error": bool(result.isError),
                        })
                messages.append({"role": "user", "content": results})

            for block in response.content:
                if block.type == "text":
                    return block.text

print(asyncio.run(run_copilot("Fit the pi0 peak in the latest run.")))
```

Read what changed from Week 37, because it is only three things: the tool
definitions now come from the server instead of hand-written dicts, executing a
tool is `await session.call_tool(...)` instead of `fn(**block.input)`, and the
protocol's error flag (`result.isError` — attribute names per the current SDK
docs) maps onto the API's `is_error`. The Week 39 error contract is what makes
this composition safe: every bad call comes back as a message the model can
act on, so the loop needs no special cases. Everything else in this week's
project — the CLI, the packaging, the evaluation — is arranged around this one
function.

## 5. Evaluating an agent: the task suite

"I ran it a few times and it seemed fine" is the agent world's equivalent of
eyeballing a fit and calling it good. The professional discipline — the reason
this section exists — is the **agent eval**: run the system against a fixed set
of tasks with machine-checkable answers, and report numbers.

A **task suite** is a list of task specifications. Each has a question exactly
as the agent will receive it, and a **success criterion** a program can check
without a human in the loop:

```python
TASKS = [
    {"task_id": "t01",
     "question": "How many run files are in data/?",
     "check": {"type": "contains_int", "value": 4}},
    {"task_id": "t02",
     "question": "Fit the pi0 peak in data/run_00042.csv and report the "
                 "mass in GeV.",
     "check": {"type": "number_within", "reference": 0.1352, "tol": 0.002}},
    {"task_id": "t03",
     "question": "Is run 41's calibration good or bad?",
     "check": {"type": "contains_word", "value": "bad"}},
    # ... ten in all
]
```

Where do the references come from? From running the tools *directly* — your own
`fit_pi0_peak` on the same file is the reference analysis, exactly the habit of
checking a new pipeline against a hand computation before trusting it. The
checker is then dumb code:

```python
import re

def check_answer(answer_text, check):
    if check["type"] == "number_within":
        found = [float(m) for m in re.findall(r"\d+\.\d+", answer_text)]
        return any(abs(x - check["reference"]) < check["tol"] for x in found)
    if check["type"] == "contains_int":
        return str(check["value"]) in answer_text
    if check["type"] == "contains_word":
        return check["value"] in answer_text.lower()
    return False
```

(`re.findall` is the standard library's pattern matcher — here, "every decimal
number in the text." That leniency is deliberate: you are grading whether the
*right value* appears, not the phrasing around it.)

The design rule: **if you cannot write the check, rewrite the task.** "Analyze
run 41" is not evaluable; "report run 41's fitted mass in GeV" is. This is the
same move as Week 38's evaluator rubrics — vague criteria produce noisy
signals — except here the check must be code, because the suite runs unattended
and a model-graded check would smuggle the noise back in.

The reading this week includes SWE-bench, the standard benchmark for coding
agents, precisely because its criterion is a model of this discipline: each
task is a real GitHub issue, and "resolved" means *the repository's own
held-back tests pass after the agent's patch is applied*. Not "the patch looks
right" — an operational, programmatic, argument-free check. Your ten-task suite
should meet the same bar at toy scale.

## 6. The three numbers

Running the suite yields the measurement triple you will report for every agent
from now on:

- **Success rate** — tasks passed / tasks attempted. The headline number, and
  the one to treat most carefully: 7/10 is a measurement with an uncertainty of
  roughly ±1.4 tasks (binomial, √(np(1−p))), the same reason you would not
  quote an efficiency from ten events to three digits. Report the fraction, not
  a false-precision percentage.
- **Cost per task** — dollars, from summing `response.usage` over every call
  the task triggered, at current per-million-token prices (your Week 37 E7
  ledger, promoted to standard equipment). Agents with equal success rates can
  differ by 10× here.
- **Latency** — wall-clock seconds per task (`time.time()` around the run).
  Matters because the copilot's user is you, waiting.

Two systems, one table — that is E3: your Week 38 fixed workflow and the agent
loop, same suite, side by side. Expect the Week 38 §10 result to generalize:
the workflow wins on the scripted tasks and cannot attempt the open-ended
ones. Numbers, not vibes, decide what ships — and the honest conclusion is
often "the workflow, for these tasks."

One caution before you believe your own success rate: you built the suite, and
you debugged the agent *against* it. That makes the suite a development set,
and your measured rate the analog of training accuracy — Week 09's
generalization gap, wearing a new costume. The fix is the same as it was then:
freeze the system, then evaluate on tasks it has never influenced (write two
fresh ones at the end, or better, have a colleague write them). Public
benchmarks have the same disease at field scale — models trained on the
internet have often *seen* benchmark tasks, which is why Week 32 taught you to
distrust headline numbers; "contamination" is the term of art.

## 7. Trajectory review: finding the first wrong step

Aggregate numbers say *whether* the agent fails; they never say *why*. For
that you read **trajectories** — the full transcript of one task attempt:
every model turn, every tool call with its arguments, every result. You built
the habit in Week 38 §7: read the transcript as Thought → Action → Observation
and find the **first wrong step**, the earliest point where an action stops
following from the observations before it. Everything downstream of that step
is noise; diagnose there.

Then name the root cause. Four categories cover nearly everything, and each
points at a different fix:

| Root cause | What it looks like in the trajectory | The fix |
|---|---|---|
| **Bad tool description** | A plausible call with wrong arguments (window in MeV, wrong directory) — the model did what the description let it believe | Rewrite the description; it is prompt engineering (Week 37 §5) |
| **Missing tool** | The model improvises with the wrong tool, or asks the user for something a tool should provide | Add or split a tool on the server |
| **Model error** | Right information in context, wrong reasoning or arithmetic on top of it | Tighten the question or system prompt; add a code gate; consider whether the task passes Week 38's viability test at all |
| **Bad task spec** | The agent answered a reasonable reading of an ambiguous question | Fix the task, not the agent |

The categories matter because they route the work: three of the four fixes are
*yours* (descriptions, tools, specs), not the model's. Reviewing three failing
trajectories and mislabeling a bad description as "model error" sends you
tuning prompts when one docstring edit would have fixed it. E4 has you write
this analysis down in `failures.md`; the mini-project requires it because an
unreviewed failure count is a number without an error budget.

## 8. LLM-as-judge, reused with care

Some qualities resist programmatic checks: is the copilot's final summary
*clear*? Did it *explain* the bad calibration or just mention it? For these you
can use a model as grader — **LLM-as-judge**, exactly as in Week 32, and every
failure mode you catalogued there transfers verbatim:

- **Position bias** — in A/B comparisons, judges favor one slot. Detect by
  swapping the order and checking the verdict flips only when it should.
- **Verbosity bias** — longer answers score higher at equal correctness.
  Detect by grading padded vs. tight versions of the same content.
- **Self-preference** — models rate their own family's style highest; grade
  with a different model than the one being graded when you can.

The placement rule this week adds: **judges grade style, code grades facts.**
Whether the mass is right, whether the tools were called, whether the file
exists — code. Whether the prose is a good run-log summary — a judge, with a
rubric as concrete as Week 38 §6's, and spot-checked by you. A success rate
that rests on a judge alone is a measurement whose calibration you never
checked.

## 9. The copilot: what you are assembling

The mini-project (`project.md` — read it now) is the month's deliverable: an
**analysis copilot**, one command that takes a physics question and drives your
MCP tools to an answer with no human in the middle. "Copilot" is not a new
mechanism — it is §4's glue function wearing an interface — and that is the
point: after four weeks, the pieces snap together.

- The **Week 39 server** supplies the tools, already tested and speaking a
  standard protocol. The copilot adds zero physics code.
- The **Week 37 loop** (async flavor, §4) supplies the mechanism.
- The **Week 38 checklist** justifies the design: the demo tasks genuinely
  vary in path (which runs exist, which are good, what needs refitting), the
  atomic steps are proven viable, tools are read-only or write-to-new-file,
  and errors are cheap because every number is re-derivable.
- **This week** supplies the interface (a thin CLI), the discipline
  (**unattended** operation: nothing between command and answer but the agent
  — possible precisely because Week 39's error contract lets the model recover
  from its own bad calls), and the evaluation: the suite, the three numbers,
  the trajectory review.

The project spec defines three scripted analysis tasks the copilot must
complete end-to-end unattended — that is the month's acceptance gate. Build
the glue first, keep the CLI thin, and spend the saved time where this week
says value lives: on the eval and the failure analysis. A modest copilot with
honest numbers is worth more — to you in Week 47, and to anyone reading your
repo — than a slick demo with none.

## Check yourself

1. State the two rules of a disciplined handoff, and what a worker knows about
   the parent conversation at the moment it starts.
2. Name the two costs and the one saving in the multi-agent cost anatomy. What
   task *shape* makes the saving win?
3. In the MCP glue loop, exactly three things changed from the Week 37 loop.
   Name them.
4. Why must a task-suite success criterion be checkable by code rather than by
   a model?
5. What does "resolved" mean operationally in SWE-bench, and why is that a good
   template for your criteria?
6. Your copilot scores 7/10 on the suite you developed it against. Give two
   reasons not to report "70% success" and what you would do about each.
7. A failing trajectory shows a fit called with `window_lo=105, window_hi=250`.
   Which root cause do you suspect first, and what is the fix?
8. Where in this week's evaluation is an LLM judge acceptable, and which two
   biases must you control before trusting it?

## Answers

1. The brief is self-contained, and the report is validated before merging. The
   worker knows *nothing* of the parent conversation — only what the brief
   says; it starts from an empty context.
2. Costs: context re-establishment (every worker re-briefed and re-given
   tools) and report round-trips (results cross the boundary as output, then
   again as input). Saving: each worker's loop is short, so the quadratic
   re-send cost stays small. Independent, tool-heavy chunks whose intermediate
   detail the parent never needs make the saving win.
3. Tool definitions come from `session.list_tools()` instead of hand-written
   dicts; execution is `await session.call_tool(...)` instead of
   `fn(**block.input)`; the protocol's error flag maps onto the API's
   `is_error`.
4. The suite must run unattended and repeatably; a model-graded check
   reintroduces exactly the noise and bias the suite exists to measure, and
   costs tokens per check besides.
5. The repository's own held-back tests pass after the agent's patch is
   applied — operational, programmatic, no judgment call. Your criteria should
   have the same "a program says pass or fail, end of discussion" character.
6. Small n: 7/10 carries a binomial uncertainty of roughly ±1.4 tasks, so
   quote the fraction, not a percentage implying precision. Development-set
   bias: you debugged against these tasks, so the rate is training accuracy —
   evaluate frozen on tasks the agent never influenced (fresh or
   colleague-written).
7. Bad tool description — the arguments look like MeV against a GeV schema,
   a units confusion the description failed to prevent. Fix the docstring
   (state units, give an example window); it reaches the model on every call.
8. For qualities code cannot check — clarity and completeness of the final
   prose summary — graded against a concrete rubric. Control position bias
   (swap A/B order) and verbosity bias (check length does not drive score);
   prefer a different model family than the one being graded.

## New terms

- **multi-agent system** — agents delegating to agents; workers with their own
  loops, contexts, and tools.
- **orchestrator / worker (agent form)** — the delegating agent; the agent
  executing one delegated subtask.
- **handoff** — transfer of work across a conversation boundary: brief out,
  report back.
- **task brief** — the self-contained instruction packet a worker starts from.
- **worker report** — the schema-validated structured result a worker returns.
- **context re-establishment** — tokens spent re-telling a fresh worker what
  the parent already knew.
- **glue layer** — the translation between MCP (list/call tools) and the
  Messages API loop (tool definitions, `tool_use` → `call_tool`).
- **agent eval / task suite** — a fixed set of task specs with programmatic
  success criteria, run unattended.
- **success criterion** — the code-checkable pass/fail rule attached to one
  task.
- **success rate / cost per task / latency** — the measurement triple reported
  for any agent.
- **benchmark / contamination** — a shared public task suite; the leakage of
  its tasks into training data or development.
- **trajectory** — the full transcript of one task attempt, tool calls and
  results included.
- **first wrong step** — the earliest trajectory point where an action stops
  following from prior observations; where diagnosis happens.
- **root-cause taxonomy** — bad tool description / missing tool / model error /
  bad task spec; each routes to a different fix.
- **LLM-as-judge (revisited)** — model-graded evaluation; style only, never
  facts, biases controlled.
- **copilot** — a conversational agent driving domain tools end-to-end on a
  user's question.
- **unattended run** — no human input between command and answer; enabled by
  the tool-side error contract.

## Going deeper

- Anthropic, *Building Effective Agents* — re-read the orchestrator–workers and
  evaluator–optimizer sections now that your workers are agents; the essay's
  cost warnings are this week's §3 in fewer words.
- Anthropic engineering blog, the post on building a multi-agent research
  system (search by title) — read for how the sub-agent briefs are written and
  why parallelism helped; their brief-writing lessons are §2's rules at
  production scale.
- Jimenez et al., *SWE-bench* — skim the evaluation setup only: how tasks were
  mined, and what "resolved" means operationally. A model of task-suite design
  worth imitating at any scale.
- Your own Week-32 notes on LLM-as-judge — they apply verbatim to trajectory
  grading; re-read before writing any judge rubric.
