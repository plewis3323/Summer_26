# Week 45 — Reading & Reproducing Papers

Papers are arguments, not lessons — this week you install a systematic workflow for
reading them (Keshav's three passes), a claims-vs-evidence checklist calibrated to ML's
failure modes (weak baselines, unreported variance, benchmark contamination), and then
reproduce a small published result to see how much a paper leaves unsaid.

## Objectives

- Apply a three-pass reading workflow: pass 1 (the five-C triage), pass 2 (figures,
  claims, evidence), pass 3 (virtual re-implementation).
- Write a structured critique: claims vs. evidence, baseline fairness, ablations
  present/missing, statistical rigor of reported gains.
- Reproduce a small published result from the paper's text alone and quantify the gap
  between your numbers and the paper's.
- Maintain a reading log that makes papers retrievable months later (claims, convincing
  evidence, doubts — not summaries).

## Core material (~3 hrs)

- `lesson.md` (this folder) — the three-pass method, the critique checklist with its
  HEP-ML additions, how to pick and run a reproduction, the reading log; worked example
  on ParticleNet.
- S. Keshav, *How to Read a Paper* — the three-pass method; adopt it verbatim this week.
- One HEP-ML paper for deep critique — pick from the Week 45 pool in
  `01-Reading-List.md`, or a Phase-2 classic you have NOT already read closely
  (ParticleNet, Particle Transformer, an Exa.TrkX tracking paper, the CaloChallenge
  summary paper).
- Skim *ML Reproducibility Challenge* reports (any year, by title) — note the recurring
  reasons reproductions fail.
- Your reproduction target's paper (see `exercises.md`) — passes 1 and 2 this block,
  pass 3 while implementing.

## Exercises

See `exercises.md` — six exercises building from a timed three-pass triage drill,
through a written deep critique and a from-text reproduction of one published number,
to the gap analysis and the start of your permanent reading log.

## Deliverable

`week45/` — `critique.md`, the reproduction repo (runnable, with `repro_plan.md` and a
results-vs-paper table), and the reading log started.

## Review

1. Week 09: a paper reports 0.3% accuracy gain over baseline, single seed. What would
   you need to see to believe it? Connect to your nested-CV work in Capstone 1.
2. Week 32: define benchmark contamination and give one way a paper could detect it in
   its own eval.
3. Week 21: what property of message passing makes GNNs suited to particle clouds, and
   which paper this week relied on it?
4. Week 35: write the simplified DDPM training loss from memory and name what each
   symbol is.
5. Week 11: why is "our method beats an untuned baseline" the ML equivalent of
   comparing to a detector simulation with the wrong calibration?
