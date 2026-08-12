# Week 46 — Research Writing & the Capstone Proposal

A thesis chapter and an ML paper sell evidence differently — this week you learn the ML genre's structure, then use it to write the proposal that will govern your final two weeks.

## Objectives

- Outline a standard ML paper (abstract, intro with contributions list, related work, method, experiments, limitations) and explain what each section owes the reader.
- Write claims that are exactly as strong as the evidence — no stronger — and revise for one-idea-per-sentence clarity.
- Design figures that carry the argument: one message per figure, honest axes, uncertainty shown.
- Produce a complete capstone proposal: problem statement, baselines, evaluation plan with pre-committed metrics, risk register, and compute budget.
- Scope two weeks of work into milestones with explicit cut lines.

## Core material (~3 hrs)

- One writing guide, read fully: the widely-circulated "How to write a great research paper" talk/notes (Simon Peyton Jones), or an equivalent ML-writing guide by title.
- Re-read the best-written of the HEP-ML papers from Week 45 purely for craft: how the intro states contributions, how figures pair with claims, how limitations are handled.
- Skim two abstracts you consider bad and diagnose why (vague claims, buried contribution, unfalsifiable language).
- Review your own Capstone-2 and Capstone-3 writeups — mark every claim that outran its evidence.

## Exercises (built when the week starts)

1. Reverse outline: reduce the Week-45 critiqued paper to one sentence per paragraph; mark where the argument jumps. Accept when: the outline exists and identifies ≥2 structural weaknesses or strengths.
2. Rewrite drill: take the worst paragraph from your Capstone-3 writeup and produce three revisions (shorter, plainer, claim-calibrated). Accept when: side-by-side versions exist with a one-line rationale each.
3. Figure surgery: remake one figure from an earlier capstone to carry exactly one message with uncertainties shown. Accept when: before/after figures plus a caption that states the takeaway in the first sentence.
4. Track selection: for each of the four capstone tracks, write a five-line feasibility sketch (data in hand? compute? riskiest step?). Accept when: one track is chosen and the runner-up documented with the tiebreaker reason.
5. Proposal draft: full proposal — problem, why it matters, baselines, evaluation plan (metrics and thresholds fixed now), risk register with mitigations, compute budget in GPU-hours and dollars, week-by-week milestones with cut lines. Accept when: `PROPOSAL.md` is ≤4 pages and every metric has a number you'd accept as success before running anything.
6. Red-team: attack your own proposal as a hostile referee — three ways it fails, and the proposal's answer to each. Accept when: the risk register absorbs all three attacks or the proposal is revised.

## Deliverable

`week46/PROPOSAL.md` (final, red-teamed) plus the writing exercises — this document is the contract for Weeks 47–48.

## Review

1. Week 12/24/36: for each prior capstone, quote its headline claim from memory and say whether the evidence supported it.
2. Week 7: a result is "2.1σ above baseline." What distributional assumptions hide inside that sentence?
3. Week 34: what pre-committed metrics did your RAG evaluation use, and why does committing before running matter more for LLM systems than for a chi-square fit?
4. Week 2: name three properties of a publication-quality matplotlib figure you standardized in Month 1.
5. Week 40: what were the three columns of your agent evaluation table, and which would a referee attack first?
