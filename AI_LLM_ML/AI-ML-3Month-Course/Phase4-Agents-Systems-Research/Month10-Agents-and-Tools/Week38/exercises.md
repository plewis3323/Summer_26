# Week 38 — Exercises

Work top to bottom. Setup (imports, `client`, the Week 37 `run_agent` /
`fit_pi0_peak` / `list_run_files`, the toy-data generator, ten heavy-ion abstracts
with hand labels, a 20-question labeled routing set, and the shared `ask_logged` /
`LEDGER` cost helper) is given by the notebook. Each pattern lives in its own file
under `week38/patterns/` (one runnable script per pattern, per this week's
deliverable); the notebook runs them and checks outputs. Every exercise must log its
calls through `ask_logged` — E6 reads the ledger.

## E1 — Prompt chaining (`chain.py`)

Implement the three-step chain from lesson.md §2: abstract → extract observables →
standardize units → markdown table, with a programmatic gate after each of the first
two steps (non-empty, parseable, expected line format).
Hint: make each step's prompt state the exact output format; the gates then have
something crisp to check.
Accept when: 8/10 provided abstracts produce a table passing all gate checks, and a
deliberately empty abstract is stopped by gate 1 (no step-2 call is made).

## E2 — Routing (`router.py`)

Implement the {lookup, computation, fit} router from lesson.md §3 and dispatch to
three handlers (canned-fact lookup, arithmetic prompt, Week 37 agent). Score it on
the 20 labeled questions.
Hint: force a one-word reply in the system prompt and `.strip().lower()` it; most
routing bugs are formatting, not classification.
Accept when: routing accuracy ≥ 90% (18/20), and every misroute is listed with the
label it should have had.

## E3 — Parallelization (`vote.py`)

Run the extraction of `mass_gev` from each abstract with 3 differently-phrased
prompts and take the median. Compare per-field accuracy of the vote vs. each single
prompt on the 10 labeled abstracts.
Hint: `sorted(votes)[1]` is the median of three; convert to `float` inside
`try/except` and treat a parse failure as a wrong vote.
Accept when: a printed table shows accuracy for prompt A, B, C, and the vote — and
the vote is ≥ the best single prompt.

## E4 — Orchestrator–workers (`orchestrate.py`)

An orchestrator call decomposes "compare pi0 yields across the 4 run files in
data/" into one fit task per file (JSON, one object per line); code executes each
fit; a merge call writes the comparison table. Validate every parsed task dict
before running it.
Hint: defend the `json.loads` — skip and report malformed lines rather than crashing.
Accept when: the final table's means and yields match your hand-run
`fit_pi0_peak` reference for all 4 files within the fit uncertainties.

## E5 — Evaluator–optimizer (`evaluate.py`)

A writer call drafts a run-summary paragraph from a fit report; an evaluator call
grades it against the 4-point rubric from lesson.md §6; loop until PASS or 3
attempts. Log every draft and verdict.
Hint: seed the first draft with an instruction that *misses* one rubric point (e.g.
omit the uncertainty) so you can watch a real rejection.
Accept when: the log shows ≥ 1 FAIL-then-improved-redraft cycle and the final draft
gets PASS, with all four rubric items verifiably present.

## E6 — ReAct vs. workflow shoot-out

Solve E4's task twice: once with the fixed workflow (your `orchestrate.py`), once by
handing the same question to the Week 37 agent loop. From `LEDGER` and timing,
build a comparison table: total input tokens, output tokens, dollars, wall time,
correctness vs. reference.
Hint: `time.time()` before and after each run is enough.
Accept when: the table prints both rows, plus a ≤ 5-sentence written verdict naming
which one you would ship for this task and why — justified by the numbers, not
taste.

## Review

1. Week 08: which parts of Adam's update are per-parameter, and why does that matter
   for sparse gradients?
2. Week 25: derive why attention scores are scaled by 1/√d before the softmax.
3. Week 31: where does the reward model sit in the RLHF pipeline, and what plays the
   "evaluator" role there?
4. Week 12: in EM for GMMs, what quantity is guaranteed non-decreasing each
   iteration — and why does your E5 loop have no such guarantee?
5. Week 04: what made an analysis "reproducible" in your Month-1 project? Which of
   this week's patterns is hardest to make reproducible, and why?
