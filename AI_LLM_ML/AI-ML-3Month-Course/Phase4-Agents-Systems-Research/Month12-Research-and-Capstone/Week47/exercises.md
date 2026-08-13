# Week 47 — Exercises

Work top to bottom, one per day. These are milestones, not conventional
exercises — the six daily milestones from `project.md`, instantiated from
`PROPOSAL.md` (fill in your proposal's own metrics, datasets, and thresholds
before day 1). All the work lives in the capstone repo per NOTEBOOK_RULES §6;
the notebook hosts only the acceptance checks (file-exists, results-table
completeness) and a display cell for E5's results table. Day 7 is slack — if
it goes unused, spend it on Week 48's stress-test list, not on new features.

## E1 — Day-1 walking skeleton

Make the full pipeline run on toy input — trivial model / canned agent
response / 100-event sample — via one command, producing output in the real
format.
Hint: the skeleton per track (lesson §2): an agent answering one hardcoded
question through the real tool interface (a); a generator emitting noise in
the correct shower format, flowing through the real validation suite (b); one
paragraph turned into one stub config (c); the target paper's pipeline on 100
events (d). Toy input, real seams.
Accept when: one command produces a (bad) end-to-end result on day 1.

## E2 — Data and environment locked

Download, stage, and version the real inputs; put the loading code under test.
Hint: the integrity check is counts, hashes, and schema — write it as a
`pytest` test so CI re-verifies it forever; track (d) verifies the download
today, not day 3 (`project.md`, Data).
Accept when: a data-integrity check (counts, hashes, schema) passes in CI.

## E3 — Baselines first

Run every baseline named in the evaluation plan through the *same* evaluation
harness and log its metrics.
Hint: baselines before the main system is tuned, or the comparison quietly
becomes "tuned system vs default baseline" — the unfair-baseline sin from
Week 45's checklist, committed by you (lesson §2).
Accept when: the results table exists with all baseline rows filled, before
the main system is tuned.

## E4 — Core system to proposal spec

Bring the track's central component — agent loop + tools / generative model /
extraction agent / reproduced method — to its mid-week milestone.
Hint: read the risk register's triggers every morning like a pressure gauge
(lesson §3); if one fires, invoke the cut line in the scope ledger, in
writing, and keep moving — the renegotiation already happened in Week 46.
Accept when: the milestone's pre-committed metric or capability check from
`PROPOSAL.md` passes.

## E5 — First full evaluation

Run the complete evaluation plan once against the real system, however
unflattering the numbers.
Hint: do not tune first and evaluate later — an ugly measured number today
beats a flattering unmeasured one; Week 48 validates the frozen system, and
"rough is fine; unmeasured is not."
Accept when: `results/` contains machine-readable metrics for system and
baselines from the same harness.

## E6 — Scope ledger current

Record every deviation from the proposal — what, why, cost — and keep the
build log daily.
Hint: write the day's ~10-line entry before you stop working (lesson §4):
results with run ids, decisions with the one-sentence why, failures as they
happen. A week-old entry is archaeology.
Accept when: `BUILDLOG.md` has daily entries and the ledger; an empty ledger
alongside missed milestones is a fail.

## Review

Keep it light this week — 15 minutes, closed book:

1. Re-derive backprop through one linear+ReLU layer (Phase 2 flagship) —
   gradient shapes included.
2. (Week 38) State the agent/no-agent checklist's four questions and apply
   them to your own capstone in one sentence each.
3. (Week 41) What belongs in a run's config vs. its code, and what goes in
   the registry?
4. (Week 44) What does your `/health` endpoint actually verify, and what does
   it miss?
