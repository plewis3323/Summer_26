# Week 38 — Agent Patterns

~3.5 hrs. Before starting you should be able to: run the hand-written agentic loop —
model → tool call → result → model (Week 37), define a tool with a JSON Schema
(Week 37), and read `stop_reason` and `usage` off a response (Week 37).

Last week you built the mechanism. This week is about *architecture*: the handful of
ways people arrange LLM calls into systems, when each one earns its complexity — and,
just as important, when the honest answer is "don't build an agent at all." The
patterns come from Anthropic's essay *Building Effective Agents*, which is this
week's external spine; the lesson gives each pattern a definition, a small runnable
sketch on our π⁰ toy data, and a cost note.

One shared helper keeps the sketches short. Assume it, plus last week's `client`,
`run_agent`, `fit_pi0_peak`, and `list_run_files`, in everything below:

```python
def ask(prompt, system=None):
    kwargs = {"model": "claude-opus-5", "max_tokens": 1000,
              "messages": [{"role": "user", "content": prompt}]}
    if system is not None:
        kwargs["system"] = system          # standing instructions, outside the turn
    response = client.messages.create(**kwargs)
    for block in response.content:
        if block.type == "text":
            return block.text
```

(`system` is a new parameter: a **system prompt** is standing instructions the model
treats as coming from the application rather than the user — "You are a router.
Respond with exactly one word." It sets behavior for the whole conversation.)

## 1. Workflows vs. agents

The distinction that organizes everything else:

- A **workflow** is a system where *your code* decides the control flow. The LLM is
  called at fixed points, like any other function. You know at coding time how many
  calls will happen and in what order.
- An **agent** is a system where *the model* decides the control flow — which tools
  to call, in what order, when to stop. Your Week 37 loop is an agent: you could not
  say in advance whether a question would take one fit or five.

Workflows are predictable, testable, cheap, and debuggable — a fixed pipeline of
calls fails in fixed places. Agents are flexible — they handle inputs you did not
enumerate — and pay for it in cost (loops burn tokens), latency (each round-trip is
seconds), and a failure surface that moves. The design rule from the essay is blunt:
**find the simplest system that meets the need**, and only add agency when the task
genuinely cannot be scripted. Most "agent" products in the wild are workflows, and
should be.

You have been here before: this is the cut-based analysis vs. BDT decision from
Phase 1. Reach for the learned, flexible thing only when the simple, inspectable
thing measurably falls short.

## 2. Pattern: prompt chaining

**What:** decompose a task into a fixed sequence of calls, each consuming the
previous output — ideally with a cheap programmatic *gate* between steps that checks
the intermediate result before you pay for the next call.

**When:** the task decomposes cleanly and the decomposition never changes. You trade
one hard call for several easy ones, each individually more reliable.

```python
abstract = open("abstract.txt").read()

# step 1: extract raw measurements
raw = ask("List every measured quantity in this abstract as 'name: value unit', "
          "one per line, nothing else:\n\n" + abstract)

# gate: cheap code check before paying for step 2
lines = [l for l in raw.splitlines() if ":" in l]
if len(lines) == 0:
    raise ValueError("gate failed: no measurements extracted")

# step 2: standardize units
std = ask("Convert every value to GeV where it is an energy or mass. "
          "Keep the same one-per-line format:\n\n" + raw)

# step 3: format the final table
table = ask("Format these as a markdown table with columns "
            "quantity | value | unit:\n\n" + std)
print(table)
```

Three calls, fixed order, one `if` between them. The gate is the underrated part:
it catches failures at the cheapest possible point, using code rather than another
model call.

## 3. Pattern: routing

**What:** a first call classifies the input; your code dispatches to a specialized
handler — a different prompt, a different tool set, even a different (cheaper) model.

**When:** inputs come in a few distinct kinds that want different treatment, and
mixing all the instructions into one mega-prompt makes every kind worse.

```python
def route(question):
    label = ask(
        question,
        system=("Classify the user's request into exactly one word: "
                "'lookup' (asks for a stored fact, e.g. a calibration constant), "
                "'computation' (asks for arithmetic on given numbers), or "
                "'fit' (asks for a peak fit on a run file). "
                "Respond with only that word."),
    ).strip().lower()

    if label == "fit":
        return run_agent(question, tools, tool_functions)   # Week 37 loop
    if label == "computation":
        return ask(question, system="Show your arithmetic step by step.")
    return ask(question, system="Answer from the provided context only.")
```

The router itself is a one-word classification — a task so easy you can (and in the
exercises, will) measure its accuracy on a labeled set like any Week 09 classifier.
Note the shape: *the model classifies, your code branches.* Control flow stays in
Python, so this is still a workflow.

## 4. Pattern: parallelization

**What:** several independent calls whose outputs your code merges. Two flavors:
*sectioning* (split the input — e.g., one call per run file) and *voting* (same input
several times, aggregate — the ensembling idea from Week 11, applied to prompts).

**When:** subtasks are independent (no call needs another's output), or one noisy
judgment gets more reliable as a majority vote.

```python
# voting: extract a value three ways, take the majority
prompts = [
    "What pi0 mass in GeV does this paragraph report? Reply with the number only.",
    "Extract the reported pi0 peak position in GeV. Number only.",
    "Find the fitted pi0 mass in GeV in the text. Number only.",
]
votes = [float(ask(p + "\n\n" + paragraph)) for p in prompts]
votes.sort()
answer = votes[1]          # median of three
```

Cost note: parallelization multiplies token spend by the number of branches — it buys
reliability or latency (the calls can run concurrently), never economy. Vote only on
fields that are actually noisy; measure whether the vote beats the best single prompt
before shipping three calls where one would do (exercise E3 does exactly this).

## 5. Pattern: orchestrator–workers

**What:** an *orchestrator* call breaks a task into subtasks it invents at run time;
your code runs a *worker* call (or Week 37 agent) per subtask; a final call merges
the workers' reports. Unlike chaining, the number and content of the subtasks is
decided by the model, per input.

**When:** the task decomposes, but the decomposition depends on the input — "compare
the π⁰ yield across whatever run files are in this directory" has as many subtasks
as there are files.

```python
import json

files = list_run_files("data")["files"]

plan = ask(
    "You are planning fits. For each file in this list, output one JSON object per "
    "line: {\"file\": ..., \"window_lo\": 0.05, \"window_hi\": 0.25}. "
    "Output nothing else.\n\n" + json.dumps(files)
)

reports = []
for line in plan.splitlines():
    task = json.loads(line)
    result = fit_pi0_peak(task["file"], task["window_lo"], task["window_hi"])
    reports.append({"file": task["file"], "result": result})

summary = ask("Write a five-line comparison of these pi0 fits. Flag any file whose "
              "mean is more than 3 sigma from 0.135 GeV:\n\n" + json.dumps(reports))
```

Here the workers are plain function calls; in E4 and in Week 40 they become model
calls, and the orchestrator's plan/merge quality becomes the thing you engineer.
The failure mode to watch: the orchestrator's plan is model output, so *parse it
defensively* — the `json.loads` line is where this sketch breaks first.

## 6. Pattern: evaluator–optimizer

**What:** one call drafts, a second call grades the draft against explicit criteria,
and the loop repeats — feed the critique back to the drafter — until the grade passes
or a retry cap hits.

**When:** you can write down what "good" means precisely enough for a model to check
it, and a critique genuinely helps the redraft. Text with hard requirements (a run
summary that must quote numbers with uncertainties) fits; "make it better" does not.

```python
rubric = ("PASS only if the summary: (1) quotes the fitted mass with uncertainty, "
          "(2) quotes the yield, (3) is at most 4 sentences, (4) states whether "
          "the run is consistent with 0.135 GeV. Reply 'PASS' or 'FAIL: <reason>'.")

draft = ask("Summarize this fit result for a run log:\n\n" + json.dumps(report))
for attempt in range(3):
    verdict = ask("Rubric: " + rubric + "\n\nSummary:\n" + draft)
    if verdict.strip().startswith("PASS"):
        break
    draft = ask("Rewrite this summary to fix the problem.\nProblem: " + verdict +
                "\nSummary:\n" + draft)
print(draft)
```

This is a feedback controller, and everything you know about them applies: it only
converges if the evaluator's signal is accurate. An evaluator that grades on vibes
produces confident, useless iterations. (You will meet the same caveat as
"LLM-as-judge bias" when this pattern reappears in agent evaluation, Week 40 —
and note the contrast with EM from Week 12: that loop provably improved its
objective every iteration; this one carries no such guarantee.)

## 7. ReAct: the agent pattern, named

The 2022 ReAct paper ("Reasoning and Acting") formalized the loop you built in
Week 37, with one addition: the model interleaves explicit written *reasoning* with
its *actions*, in a Thought → Action → Observation cycle:

```
Thought: I need to know which run files exist before I can fit anything.
Action: list_run_files(directory="data")
Observation: {"files": ["data/run_00040.csv", ..., "data/run_00042.csv"]}
Thought: run_00042 is the latest. Now fit its pi0 peak.
Action: fit_pi0_peak(file="data/run_00042.csv", window_lo=0.05, window_hi=0.25)
Observation: {"mean_gev": 0.1352, ...}
Thought: I have the mass. I can answer now.
```

The paper's finding: forcing the reasoning step reduces hallucinated actions and
helps the model recover from surprising observations, because each Thought re-grounds
the plan in what was actually observed. Modern tool-use APIs bake this in — the model
emits its thinking as text blocks alongside `tool_use` blocks, so your Week 37 loop
*is* a ReAct loop whenever the model chooses to think out loud. What remains useful
is the diagnostic habit: when an agent misbehaves, read the transcript as
Thought/Action/Observation and find the first step where a Thought stopped following
from the Observations. That is your bug — in the prompt, a tool description, or a
tool result that did not say enough.

## 8. When NOT to build an agent

The most valuable section of this week. Before building an agent, put the task
through four questions:

1. **Complexity** — is the path genuinely unknowable in advance? If you can write the
   flowchart, write the flowchart: it is a workflow, and every box you script is a
   failure mode you removed.
2. **Value** — is the answer worth the cost? An agent run costs tens of thousands of
   tokens across loop iterations. A task worth a fraction of a cent cannot justify
   it; batch classification wants one structured-output call per item, not a loop.
3. **Viability** — can the model actually do the atomic steps? Prototype the single
   hardest step alone first. If the model can't reliably choose a fit window from a
   histogram description, an agent wrapped around that inability just fails slower
   and more expensively.
4. **Cost of error** — what happens when it is wrong, and will you notice? Agent
   errors compound silently across steps. Read-only tools and checkable outputs
   (numbers you can re-derive) keep mistakes cheap; tools with side effects make
   the model's mistakes yours. Gate anything irreversible behind a human.

If any answer says no, step down the ladder: agent → orchestrator–workers →
routing/chaining → one structured-output call → plain code. The best system is the
*least* agentic one that meets the need. Physics instinct transfers directly: you
never fit a six-parameter function when a two-parameter one describes the data.

## 9. Instrumentation: cost per step

Every pattern this week gets the same accounting habit, because the pattern choice
*is* a cost choice. Wrap the API call once:

```python
LEDGER = []

def ask_logged(prompt, step, system=None):
    kwargs = {"model": "claude-opus-5", "max_tokens": 1000,
              "messages": [{"role": "user", "content": prompt}]}
    if system is not None:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    LEDGER.append({"step": step,
                   "in": response.usage.input_tokens,
                   "out": response.usage.output_tokens})
    for block in response.content:
        if block.type == "text":
            return block.text
```

At the end of any run, `LEDGER` tells you where the tokens went — which is how you
discover, say, that the evaluator in section 6 costs more than the drafts it grades,
or that a chain re-sends the full abstract three times when step 2 only needed
step 1's output. In agentic loops, remember the stateless-API fact from Week 37:
the whole conversation is re-sent every iteration, so input tokens grow roughly
quadratically with loop length. Long agent runs are expensive *by construction*;
that, not any single price, is the real argument for the simplest viable pattern.

## 10. Worked example: same task, workflow vs. agent

The question: "Compare the π⁰ yields across the four run files in `data/` and flag
outliers." Two implementations:

**Workflow (orchestrator–workers, section 5):** code lists the files, one planning
call emits fit windows, code runs the four fits, one summary call writes the
comparison. Six model-free fits, two model calls, fixed shape. Token cost: two
prompts, each a few hundred tokens.

**Agent (Week 37 loop):** hand `run_agent` the question with `list_run_files` and
`fit_pi0_peak` and let it drive. Typical transcript: one listing call, four fit
calls, a final summary — six round-trips, each re-sending the growing conversation.

Run both (E6 does) and compare three numbers — total tokens, wall time, correctness
against a hand-computed reference. On this task the workflow typically wins all
three, *and that is the lesson*: the agent's flexibility bought nothing here because
the task's structure was known. Change the task to "figure out why run 41 looks
weird" — no known decomposition, needs iterative poking — and the agent earns its
keep. Match the architecture to how much of the path you can see in advance.

## Check yourself

1. State the workflow/agent distinction in one sentence each. Which was your Week 34
   RAG pipeline?
2. What is a gate in a prompt chain, and why is it code rather than a model call?
3. Your router hits 85% accuracy on a labeled set. What are your two cheapest levers
   before adding a bigger model?
4. Why does parallelization-by-voting cost 3× but chaining cost roughly 1× (per
   step) relative to a single call?
5. In orchestrator–workers, which two places is the system most likely to break, and
   what defends each?
6. The evaluator–optimizer loop has no convergence guarantee. What property of the
   evaluator makes it converge in practice?
7. Name the four when-not-to-build-an-agent questions and give a physics-workflow
   task that fails each one.
8. Why do input tokens grow roughly quadratically with agent-loop length?

## Answers

1. Workflow: code decides the control flow, the model fills fixed slots. Agent: the
   model decides which actions to take and when to stop. Week 34 RAG was a workflow —
   retrieve, rerank, synthesize, in an order you wrote.
2. A cheap programmatic check between steps (a regex, a count, a parse) that stops
   the chain before the next paid call. Code, because it must be reliable and free —
   a model-graded gate re-introduces the noise it exists to catch.
3. Improve the routing prompt (clearer label definitions, examples of each class) and
   fix the label set itself (ambiguous categories cause most routing errors). Both
   are free; a bigger model is neither.
4. Voting sends the same input N times by design; chaining sends each step mostly its
   predecessor's (smaller) output.
5. Parsing the orchestrator's plan (model output — parse defensively, validate
   against a schema) and merging worker reports (workers can disagree or fail —
   merge code must tolerate missing/flagged entries).
6. That its verdicts correlate with the real rubric — an accurate, specific evaluator
   ("FAIL: no uncertainty quoted") moves each redraft toward passing; a vague one
   random-walks.
7. Complexity ("histogram these files" — flowchartable), value (labeling 10k
   abstracts — per-item value too low for loops), viability (choosing physics cuts
   the model can't validate), cost of error (a tool that deletes calibration files —
   unbounded downside). Each fails a different question.
8. The API is stateless, so iteration k re-sends all k−1 prior turns; summing
   1+2+…+n turns gives ~n²/2 turn-sends.

## New terms

- **workflow** — LLM calls arranged by code-owned control flow; call count and order
  known in advance.
- **agent** — system where the model chooses actions, order, and stopping.
- **system prompt** — standing application-level instructions, sent via `system=`,
  separate from user turns.
- **prompt chaining** — fixed sequence of calls, each consuming the previous output.
- **gate** — cheap programmatic check between chain steps that stops early failures.
- **routing** — classify first, then dispatch to a specialized handler.
- **parallelization** — independent calls merged by code: *sectioning* (split input)
  or *voting* (aggregate repeated judgments).
- **orchestrator–workers** — a model call plans input-dependent subtasks; workers
  execute; a merge call combines reports.
- **evaluator–optimizer** — draft/grade/redraft loop against an explicit rubric.
- **ReAct** — the Thought → Action → Observation loop; explicit reasoning interleaved
  with tool calls.
- **agent ladder** — prefer the least agentic design that meets the need.

## Going deeper

- Anthropic, *Building Effective Agents* — the primary text. Read it twice; the
  second time, list which of your Phase 1–3 pipelines were workflows (answer: nearly
  all, and rightly so).
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (arXiv
  2210.03629) — read the sections on the Thought/Action/Observation format; skim the
  benchmarks.
- Anthropic's *Building Effective Agents* cookbook — reference implementations of all
  five patterns; search for it by name and compare with your own sketches after E1–E5.
- Anthropic docs, tool-use best practices — guidance on when to promote an action
  from a generic tool to a dedicated typed tool; foreshadows Week 39.
