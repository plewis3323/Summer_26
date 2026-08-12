# Week 46 — Research Writing & the Capstone Proposal

Writing is where you find out what you actually showed — this week teaches the ML
paper's structure from zero (what each section owes the reader, what makes a figure
honest, how to calibrate claims to evidence), then points the skeleton forward: you
write the capstone proposal that will govern your final two weeks.

## Objectives

- Outline a standard ML paper (abstract, intro with contributions list, related work,
  method, experiments, limitations) and explain what each section owes the reader.
- Write claims that are exactly as strong as the evidence — no stronger — and revise
  for one-idea-per-sentence clarity.
- Design figures that carry the argument: one message per figure, honest axes,
  uncertainty shown.
- Produce a complete capstone proposal: problem statement, baselines, evaluation plan
  with pre-committed metrics, risk register, and compute budget.
- Scope two weeks of work into day-level milestones with explicit cut lines.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the paper skeleton section by section, claim
  calibration, honest figures, the full proposal template, and a worked mini-proposal.
- Simon Peyton Jones, *How to write a great research paper* (talk + notes, free) —
  the week's external spine; watch or read fully.
- Re-read the best-written of the Week 45 papers purely for craft: how the intro
  states contributions, how figures pair with claims, how limitations are handled.
- Skim two abstracts you consider bad and diagnose why (vague claims, buried
  contribution, unfalsifiable language).
- Review your own Capstone-2 and Capstone-3 writeups — mark every claim that outran
  its evidence.

## Exercises

See `exercises.md` — a reverse outline of a published paper, a rewrite drill on your
own worst paragraph, figure surgery on an earlier capstone plot, track selection, and
the week's deliverable: `PROPOSAL.md`, drafted then red-teamed.

## Deliverable

`week46/PROPOSAL.md` (final, red-teamed) plus the writing exercises — this document is
the contract for Weeks 47–48.

## Review

1. Week 12/24/36: for each prior capstone, quote its headline claim from memory and
   say whether the evidence supported it.
2. Week 08: a result is "2.1σ above baseline." What distributional assumptions hide
   inside that sentence?
3. Week 34: what pre-committed metrics did your RAG evaluation use, and why does
   committing before running matter more for LLM systems than for a chi-square fit?
4. Week 03: name three properties of a publication-quality matplotlib figure you
   standardized in Month 1.
5. Week 40: what were the columns of your agent evaluation table, and which would a
   referee attack first?
