# Week 45 — Reading & Reproducing Papers

You already read physics papers professionally; this week you calibrate that skill for ML papers — where the "detector effects" are benchmark contamination, weak baselines, and unreported variance — and then reproduce a result to see how much a paper hides.

## Objectives

- Apply a three-pass reading workflow: pass 1 (category, context, correctness-at-a-glance), pass 2 (figures, claims, evidence), pass 3 (virtual re-implementation).
- Write a structured critique: claims vs. evidence, baseline fairness, ablations present/missing, statistical rigor of reported gains.
- Reproduce a small published result from scratch and quantify the gap between your numbers and the paper's.
- Maintain a reading log that makes papers retrievable months later (claims, methods, doubts — not summaries).

## Core material (~3 hrs)

- S. Keshav, *How to Read a Paper* — the three-pass method; adopt it verbatim this week.
- One HEP-ML paper for deep critique — pick from: ParticleNet (jet tagging as point clouds), the Particle Transformer paper, an Exa.TrkX tracking paper, or the CaloChallenge summary paper. Choose one you have NOT already read closely.
- Skim *ML Reproducibility Challenge* reports (any year, by title) — note the recurring reasons reproductions fail.
- Your reproduction target's paper (see exercises) — passes 1 and 2 this block, pass 3 while implementing.

## Exercises (built when the week starts)

1. Three-pass drill: apply passes 1–2 to three candidate papers in ≤90 minutes total, with the pass-1 five-C notes (category, context, correctness, contributions, clarity) for each. Accept when: three structured notes exist and one paper is selected for deep reading with a stated reason.
2. Deep critique: full three-pass read of the chosen HEP-ML paper; write a 1–2 page critique covering claims/evidence, baselines, ablations, and what experiment you'd demand before believing the headline number. Accept when: `critique.md` contains ≥3 specific technical criticisms citing section/figure numbers.
3. Pick a reproduction target: a single figure or table row reproducible on your hardware in <1 GPU-day (e.g. a baseline MLP/BDT row from a HEP-ML benchmark paper, or a small model's curve from an ML paper). Accept when: `repro_plan.md` states the exact number(s) targeted, data source, and compute estimate.
4. Reproduce: implement from the paper's description alone (no reference code until stuck; log when/why you peek). Accept when: your number is within the paper's reported uncertainty — or the gap is quantified and diagnosed in writing.
5. Gap analysis: list every implementation decision the paper left unspecified and which you had to guess. Accept when: the list has ≥5 entries, each with your guess and its observed effect (or "untested").
6. Reading log: set up a permanent log (one file or Zotero notes) and backfill entries for this week's papers. Accept when: each entry answers "what did they claim, what convinced me, what didn't" in ≤10 lines.

## Deliverable

`week45/` — `critique.md`, the reproduction repo (runnable, with `repro_plan.md` and results vs. paper table), and the reading log started.

## Review

1. Week 9: a paper reports 0.3% accuracy gain over baseline, single seed. What would you need to see to believe it? Connect to your nested-CV work in Capstone 1.
2. Week 32: define benchmark contamination and give one way a paper could detect it in its own eval.
3. Week 21: what property of message passing makes GNNs suited to particle clouds, and which paper this week relied on it?
4. Week 35: write the simplified DDPM training loss from memory and name what each symbol is.
5. Week 11: why is "our method beats an untuned baseline" the ML equivalent of comparing to a detector simulation with the wrong calibration?
