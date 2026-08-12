# Week 45 — Exercises

Work top to bottom. Setup (imports, data loading, constants) is given by the
notebook; you write only the lines each exercise asks for. This week most work
lives in files, not the notebook: E1, E2, E5, and E6 produce markdown documents,
and E4 is a small repo of its own; the notebook hosts only E4's analysis cells
(results table, comparison plot) per NOTEBOOK_RULES §6.

Paper pool for E1–E2 — pick from the Week 45 critique pool in `01-Reading-List.md`
(a recent sPHENIX/ALICE/ATLAS ML paper, a CaloChallenge entrant, or a
"foundation model for physics" paper) or from the Phase-2 classics you have not
yet read closely (ParticleNet, Particle Transformer, the Exa.TrkX tracking paper,
the CaloChallenge summary).

## E1 — Three-pass triage drill

Apply passes 1–2 of the three-pass method to **three** candidate papers in at
most 90 minutes total (budget: ~10 min pass 1 each, ~20 min pass 2 on whichever
survive). For each paper write the pass-1 five-C card (category, context,
correctness, contributions, clarity) in `triage.md`, then select ONE paper for
the deep critique and state the reason in one sentence.
Hint: set a timer per pass and stop when it rings — overrunning the timebox is
the failure mode this drill trains out.
Accept when: `triage.md` contains three five-C cards, each with all five Cs
filled, plus a one-sentence selection rationale.

## E2 — Deep critique

Full three-pass read of the paper selected in E1. Write `critique.md` (1–2
pages) working through the §2 checklist: headline claim quoted verbatim, what
was measured, baseline fairness, variance, ablations present and missing,
contamination risk, compute matching, what is conspicuously absent — ending with
the one experiment you would demand before believing the headline number.
Hint: build the claims/evidence table first (one row per claim, one column for
the figure/table that supports it); the prose then writes itself.
Accept when: `critique.md` contains at least 3 specific technical criticisms,
each citing a section, figure, or table number.

## E3 — Pick a reproduction target

Choose a single number or curve to reproduce: a table row or figure from any
paper, reproducible on your hardware in under 1 GPU-day, with public data. A
baseline row (MLP, BDT, small CNN) from a HEP-ML benchmark paper is the intended
difficulty. Write `repro_plan.md`: the exact number(s) targeted with the paper's
stated uncertainty, the dataset and where to get it, your compute estimate, and
your pre-committed success tolerance.
Hint: apply the four criteria from lesson §3; if the paper states no
uncertainty, commit to your own tolerance now and justify it.
Accept when: `repro_plan.md` states target number(s), data source, compute
estimate, and a numeric success tolerance — all before any code exists.

## E4 — Reproduce it

Implement the target from the paper's description alone, as a small tracked repo
(seeded, `uv sync` + one command, per Week 41 practice). No reference code until
you are genuinely stuck; if you peek, log when and why in `BUILDNOTES.md`. Then
fill the notebook's comparison cells: your metric (≥2 seeds if compute allows),
the paper's value, and the gap.
Hint: get the *metric computation* verified on a small subsample first —
mismatched metric conventions cause most false failures.
Accept when: your number is within the E3 tolerance, OR the gap is quantified
(your value ± your seed spread vs the paper's value ± its uncertainty) and a
written diagnosis names the leading suspect.

## E5 — Gap analysis

List every implementation decision the paper left unspecified and that you had
to guess, in `gaps.md`: the decision, your guess, why you guessed it, and its
observed effect on the result (or "untested").
Hint: walk the pipeline in order — data selection, preprocessing, architecture
details, loss, optimizer, schedule, stopping rule, evaluation convention — and
interrogate each stage.
Accept when: `gaps.md` has at least 5 entries, each with a guess and an
effect/"untested" field.

## E6 — Start the reading log

Create your permanent reading log (one markdown file, or Zotero notes — your
choice, but pick the one you will still use in a year). Backfill entries for
every paper touched this week (the three from E1, the reproduction paper).
Each entry: what did they claim, what convinced me, what didn't — ≤10 lines —
plus retrieval tags.
Hint: write each entry the same day you read the paper; a week-old entry is
already a re-read.
Accept when: the log exists with ≥4 entries, each answering all three questions
in ≤10 lines.

## Review

1. (Week 09) Define bias and variance of an estimator, and state which one
   cross-validation is trying to measure honestly.
2. (Week 32) You suspect a benchmark is contaminated for a given LLM. Describe
   one concrete test the paper itself could have run to detect it.
3. (Week 41) A run's config file vs its code vs the experiment registry: name
   one thing that belongs in each, and why moving it elsewhere hurts
   reproducibility.
4. (Week 21) Why does an unordered set of particles call for a
   permutation-invariant architecture, and what goes wrong if you feed it to an
   MLP as a fixed-order vector?
5. (Week 35) Write the simplified DDPM training loss from memory and name every
   symbol.
