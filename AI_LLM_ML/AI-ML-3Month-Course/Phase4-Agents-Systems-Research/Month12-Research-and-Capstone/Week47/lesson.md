# Week 47 — Running a One-Week Build Sprint

~1 hr reading — this week's time goes into the build (`project.md`). Before
starting you should have: a final, red-teamed `PROPOSAL.md` (Week 46), the
engineering habits of tests + tracked runs + one-command repos (Weeks 04, 41,
44), and a chosen capstone track. This short lesson is about *how to spend the
week*: the sprint shape, descoping, and the lab log.

## 1. What a sprint is, and why the plan is already written

A **sprint** is a fixed time box with a fixed goal: the time does not move, so
when reality disagrees with the plan, *scope* moves instead. Physics runs this
way too — an approved beam-time request at an accelerator gets its scheduled
days on the machine whether or not the detector is fully calibrated, and good
experimenters plan for that: a must-have measurement, then nice-to-haves in
priority order.

Your beam-time request is `PROPOSAL.md`. This changes your job for the week.
You are not deciding *what* to build — Week 46 decided, while you were still
impartial. You are executing, and the only decisions left are the ones the
proposal deliberately left open: which cut lines to invoke, and when. Treat an
urge to redesign mid-week as a signal to re-read the proposal, not the code.

## 2. Skeleton first: end-to-end before deep

The single highest-value rule of a short build: **make the whole pipeline run,
badly, on day 1** — then deepen. A **walking skeleton** is the thinnest
end-to-end version of the system: toy input, trivial model or canned response,
real output format, one command. For the four tracks that means, respectively:
an agent that answers one hardcoded question through the real tool interface; a
generator that emits noise in the correct shower format and runs the validation
suite on it; an extraction agent that turns one paragraph into one stub config;
a training loop that runs the target paper's pipeline on 100 events.

Why this order wins:

- **Integration risk dies first.** The failures that kill week-long builds are
  at the seams — data formats, tool interfaces, evaluation plumbing — not in
  the clever component. The skeleton forces every seam on day 1, when there is
  still time.
- **You always have a shippable system.** Every day after day 1 improves a
  working thing. If Thursday goes badly, Week 48 validates Wednesday's system.
  The alternative — a deep component with no pipeline around it — ships
  nothing until everything works, which is a coin flip on a deadline.
- **The evaluation harness exists from the start**, so every improvement is
  measured the day it lands, against the proposal's own metrics (and the
  baselines get evaluated *before* the main system is tuned — otherwise the
  comparison quietly becomes "tuned system vs default baseline", the exact sin
  Week 45 taught you to catch in other people's papers).

The day-by-day shape this implies is written out as milestones in
`project.md`; roughly: day 1 skeleton, day 2 real data locked, day 3
baselines, days 4–5 the core system, day 6 first full evaluation, day 7 slack
— and if you think you will not need the slack day, that belief is itself a
risk-register entry.

## 3. Descoping is a skill, not a failure

Mid-sprint, behind schedule, you will want to push through — one more evening,
one more clever fix. Occasionally right; usually **sunk-cost reasoning** (the
effort already spent is gone either way; the only question is where the
*remaining* time does the most good). The proposal's cut lines exist because
Week-46-you was a better judge of that question than tired-Thursday-you.

The discipline:

- **Check triggers daily.** Each risk in the register has a trigger ("if X is
  not working by Wednesday..."). Every morning, read the register and check
  the triggers like you check a pressure gauge — deliberately, not when the
  alarm sounds.
- **Invoke cut lines in writing.** When a trigger fires, write it in the scope
  ledger (§4) — what is cut, why, what is delivered instead — and move on
  without renegotiating. The renegotiation already happened, in Week 46.
- **Cut scope, never rigor.** Drop a feature, shrink a dataset, reduce a
  sweep: fine — the deliverable narrows but stays true. Skip the tests, stop
  tracking runs, hardcode a path "for now": not fine — that converts visible
  schedule slip into invisible technical debt, which Week 48 (validation and
  writeup) pays back with interest. "Rough is fine; unmeasured is not."

## 4. The lab log

You already keep one for the detector; the build gets one too. `BUILDLOG.md`,
one entry per day, written before you stop working, ~10 lines:

- **Did / result** — what was attempted, what happened, with numbers ("baseline
  BDT: AUC 0.943, run `bl-03`" — future-you needs the run id, Week 41).
- **Decisions** — every choice not already in the proposal, with the
  one-sentence why. These become the writeup's method section.
- **Surprises and failures** — what broke, what was weird, including the dead
  ends. Week 48's report requires "what failed first"; this is where it comes
  from. Physics analysis has the same rule: the analysis note records the cuts
  that *didn't* work, or the review asks why not.
- **Scope ledger** (running section at the top of the file) — every deviation
  from the proposal: what, why, cost. An empty ledger with missed milestones
  means the ledger is a lie, not that the week went to plan.
- **Tomorrow** — the next entry point, so the morning starts with zero
  "where was I".

Ten minutes a day. Week 48's report is then *assembly* — the log has the
decisions, the numbers have run ids, the failures are already written down —
instead of a week of archaeology on your own repo.

## Check yourself

1. In a fixed time box, what is the variable that absorbs schedule slip, and
   what document pre-negotiated how it moves?
2. Your generative model is brilliant but the validation suite won't be
   wired up until Friday. What sprint rule did you break, and what should
   day 1 have produced instead?
3. Why must baselines be evaluated before the main system is tuned?
4. A cut-line trigger fires on Wednesday. List the three things you write in
   the scope ledger, and the one thing you do not do.
5. Name two "cuts" that are actually rigor cuts in disguise, and what each
   costs in Week 48.
6. What makes a `BUILDLOG.md` entry useful six days later — name two concrete
   properties.

## Answers

1. Scope. `PROPOSAL.md` — its milestones say what "on plan" means and its cut
   lines say what moves, under which trigger, when the plan slips.
2. Skeleton-first: end-to-end shallow before deep components. Day 1 should
   have produced the full pipeline on toy input — noise samples in the right
   format flowing through the *real* validation suite — so the model's quality
   was measurable from the day it existed.
3. Because tuning attention is a budget: if the system is tuned first, the
   baselines get whatever budget is left (usually none), and the comparison
   becomes "tuned vs untuned" — the unfair-baseline failure from Week 45's
   checklist, now committed by you.
4. Write: what is cut, why (which trigger fired), and what is delivered
   instead. Do not: reopen the decision — the renegotiation happened in Week
   46, while you were impartial.
5. Skipping tests on load-bearing paths (Week 48's validation then can't
   distinguish bugs from findings) and not tracking runs (Week 48's results
   table can't be regenerated, so the "one command" gate fails). Both convert
   visible slip into invisible debt.
6. Numbers with run ids (a result you can find again beats a result you
   remember), and decisions with their why (the writeup's method section needs
   the reason, not just the choice). Honorable mention: a "tomorrow" line that
   makes the next morning start instantly.

## New terms

- **sprint / time box** — fixed duration, fixed goal; slip moves scope, not
  the deadline.
- **walking skeleton** — thinnest end-to-end version of the system, running on
  day 1; deepened thereafter.
- **sunk-cost reasoning** — weighing already-spent effort in a decision that
  only remaining time can affect.
- **scope ledger** — the running written record of every deviation from the
  proposal: what, why, cost.
- **lab log / build log** — daily record of results (with run ids), decisions,
  failures, and the next entry point.

## Going deeper

- Re-read your own `PROPOSAL.md` — it is this week's actual spine text.
- Hunt & Thomas, *The Pragmatic Programmer*, "tracer bullets" section — the
  walking-skeleton idea under its other common name; optional, library copy.
- Anthropic, *Building Effective Agents* (free) — for tracks (a)/(c), re-skim
  the "start simple, add complexity only when needed" thread before day 1; it
  is the same skeleton-first argument for agent systems.
