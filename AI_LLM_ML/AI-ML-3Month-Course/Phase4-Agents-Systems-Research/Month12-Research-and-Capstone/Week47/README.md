# Week 47 — Final Capstone: Build

This is the build week: execute the proposal like an approved beam-time request — the plan is fixed, the clock is running, and scope changes get written down, not improvised.

## Objectives

- Stand up the full skeleton of your chosen track early (end-to-end but shallow), then deepen — never a deep component with no pipeline around it.
- Hit the proposal's Week-47 milestones or invoke a pre-planned cut line, in writing.
- Keep engineering discipline under time pressure: tests on the load-bearing paths, tracked runs (Week 41), one-command reproducibility.
- Produce baseline results for every comparison the evaluation plan promises.
- Keep a daily build log that Week 48's writeup can be assembled from.

## Core material (~3 hrs)

Reading this week is track-specific reference, not new study:

- Track (a) Domain copilot: your Week-39/40 code and MCP + *Building Effective Agents* notes; the workflow docs you're wrapping.
- Track (b) Generative model: your Week-35 DDPM derivation and Capstone-2/3 validation suites; CaloChallenge docs if on the physics track.
- Track (c) Paper-to-pipeline agent: your Week-38 pattern code; the target paper read at pass-3 depth.
- Track (d) Reproduce-and-extend: the target paper at pass-3 depth (Week 45 method); its public code/data documentation.
- Track (e) Evaluated LLM service: your Week-44 service, Week-32/34/40 evals, Week-37 injection tests.

## Exercises (built when the week starts)

Milestones, not exercises — instantiated from `PROPOSAL.md` when the week starts:

1. Day-1 skeleton: end-to-end pipeline runs on toy input (trivial model / canned response / 100-event sample). Accept when: one command produces a (bad) end-to-end output on day 1.
2. Data/environment locked: real inputs staged, versioned, and loading under test. Accept when: a data-integrity check (counts, hashes, schema) passes in CI.
3. Baselines first: every baseline in the evaluation plan runs and logs its metric. Accept when: the results table exists with all baseline rows filled before the main system is tuned.
4. Core system at proposal spec: the track's central component (agent loop + tools / generative model / extraction agent / reproduced method / evaluated service) meets its mid-week milestone. Accept when: the milestone's pre-committed metric or capability check from `PROPOSAL.md` passes.
5. First full evaluation: the complete evaluation plan runs once on the real system, however unflattering. Accept when: `results/` contains machine-readable metrics for system and baselines from the same harness.
6. Scope ledger: every deviation from the proposal recorded (what, why, cost). Accept when: `BUILDLOG.md` has daily entries and the ledger — an empty ledger with missed milestones is a fail.

## Deliverable

The capstone repo at end-of-week state: skeleton-to-real pipeline, baselines evaluated, first full eval results, `BUILDLOG.md`. Rough is fine; unmeasured is not.

## Review

Keep it light this week — 15 minutes, closed book:

1. Re-derive backprop through one linear+ReLU layer (Phase 2 flagship) — gradient shapes included.
2. Week 38: state the agent/no-agent checklist's four questions and apply them to your own capstone in one sentence each.
3. Week 41: what belongs in a run's config vs. its code, and what goes in the registry?
4. Week 44: what does your `/health` endpoint actually verify, and what does it miss?
