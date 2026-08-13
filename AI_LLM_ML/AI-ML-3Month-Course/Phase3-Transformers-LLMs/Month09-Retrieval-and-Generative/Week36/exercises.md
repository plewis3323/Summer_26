# Week 36 — Exercises

This week is Capstone 3; the exercises are milestones, not conventional
exercises — finishing them *is* finishing the capstone. Read `project.md`
before Monday. Choose **one** track by day 1, in writing, with one sentence
on why, and stay on it: (a) domain-literature assistant (physics default), or (b)
conditional diffusion generator (calo default; public images allowed). Do E1–E4 for your track, then E5–E6 (both tracks).
Most work lives in the capstone repo, not the notebook, per
`NOTEBOOK_RULES.md` §6; the notebook hosts acceptance checks and displays
the benchmark table. One scheduled week, two weeks' worth of building —
syllabus §6 slack exists for this; take the second calendar week if you need
it and log it.

The Phase 3 gate derivation (scaled dot-product attention + √d_k, cold) is
scheduled for a *fresh* morning and scanned into this folder — it is in
`project.md`'s acceptance gate, not a seventh exercise. Monthly rotation
also wants 2-layer backprop; do it the same morning if the attention pass is
clean.

## Track (a) — domain-literature assistant

### E1 — Architecture

Week 34 RAG + Week 32 extractor behind one interface: free-text answers cite
sources; metadata queries route to the extractor. One script answers both a
free-text and a structured query end to end.
Hint: do not rebuild either component — wrap them. A thin router (question
type → RAG synthesis vs extractor JSON) plus the Week 34 citation prompt is
the architecture; lesson §2. The extractor schema stays frozen (Week 32).
Accept when: one script answers both a free-text and a structured query end
to end.

### E2 — Held-out eval

Write 30 *new* Q&A pairs (not from Week 34's set), frozen before integration
finishes. Each row: question, gold source span / chunk id, gold answer.
Commit with a hash-dated provenance line.
Hint: Week 34 E2's held-out set was written to stay closed until this week —
that is a *different* file; these 30 are new. Write questions from the
documents (point at the passage first). Once this file is committed, it is
the test set; extra items you write after seeing failures go in a dev split,
not in the table (lesson §3).
Accept when: file committed with gold sources/answers and a hash-dated
provenance line.

### E3 — Benchmark

Assistant vs plain RAG vs no-retrieval LLM on the held-out set — answer
quality (judge + spot-check), groundedness, retrieval recall@5. One table,
three systems, all metrics.
Hint: identical judge, identical spot-check budget, identical recall@5
definition. Plain RAG is Week 34 without the extractor; no-retrieval is the
same instruct model, no library. "The answers looked good" is not a row
(lesson §4). Macro answer-quality vs the better baseline is a reasonable
E1-of-the-project success number — name it in the Monday proposal.
Accept when: one table, three systems, all metrics.

### E4 — Failure analysis

Dissect the 5 worst answers — retrieval miss vs synthesis failure vs
extractor error. Each failure attributed to a stage with evidence.
Hint: evidence is a span. If the gold chunk was not in the top 5, it is a
retrieval miss even if the answer is fluent. If the chunk was there and the
answer invented a √s_NN, it is synthesis (or extractor, if the question was
structured). If you cannot say which stage failed, you have a demo, not a
system.
Accept when: each failure attributed to a stage with evidence.

## Track (b) — conditional diffusion calo generator

### E1 — Scale the warm-up

Scale the Week 35 E5 warm-up to Capstone 2's shower dataset, conditioned on
incident energy (and angle if available). Training converges with loss
curves and fixed-seed sample grids saved.
Hint: start from the Week 35 checkpoint if you have it; same L_simple, same
condition-dropout for CFG. Freeze the energy grid and the list of physics
observables *before* you sample onto them (lesson §3) — that is this track's
held-out set. GPU week (syllabus §8).
Accept when: training converges with loss curves and fixed-seed sample grids
saved.

### E2 — Physics validation vs truth

Energy response + resolution vs condition, longitudinal and radial shower
profiles (or the Week-20/24 shower-shape stand-ins, if the toy has no
depth), at ≥ 3 energies. Overlay plots with a quantitative distance
(χ²/ndf or KS) per observable.
Hint: sample grids prove nothing — the suite is Capstone 2's, now required
at ≥ 3 energies (lesson §4). Plot resolution vs 1/√E. If you only have
transverse 8×8 images, "longitudinal" is not available; say so and use
second-moment width + leading-tower fractions, the Week-20 shapes, at ≥ 3
energies. That substitution is honest; skipping the distance is not.
Accept when: overlay plots with a quantitative distance (χ²/ndf or KS) per
observable.

### E3 — Head-to-head vs the Capstone 2 VAE

Same observables, same test set, plus sampling time per shower. One
comparison table and a verdict paragraph.
Hint: re-eval (or retrain) the VAE *here* on the same showers — quoting last
month's table is not a comparison (Week 24 §3, still in force). Sampling
cost is wall-clock per shower, same hardware, eval mode; a table without
this row is marketing (Week 35 §10). Verdict: which you would ship, given
quality *and* cost.
Accept when: one comparison table and a verdict paragraph.

### E4 — Guidance / ablation

Sweep CFG weight (or one architecture choice). Plot its effect on ≥ 1
physics observable.
Hint: high w that nails the mean energy response and kills the shower-shape
*width* is a failed generator that looks successful on the wrong plot
(Week 35 §9). Sweep w ∈ {0, 0.5, 1, 2} on a physics width, not on an
eyeball grid. If you skip CFG, ablate sampling steps T or the energy-
embedding instead — one axis, plotted.
Accept when: its effect on ≥ 1 physics observable is plotted.

## Both tracks

### E5 — Repo hygiene

Tests green, pinned deps, README with one-command repro. Fresh-clone run
reproduces the benchmark table within stochastic noise.
Hint: "one command" is `uv sync` + the script named in the README, which
prints the table from cached generations / cached model outputs if you do
not want to retrain on a stranger's laptop. Training from scratch can be a
second documented command. `pytest -q` covers the load-bearing paths
(retrieval metrics, schema, physics-histogram helpers — not "the GPU
training loop").
Accept when: fresh-clone run reproduces the benchmark table within
stochastic noise.

### E6 — Writeup

~2 pages: problem, system, eval, results, limitations, what failed first.
Committed as `writeup.md` with the benchmark table embedded.
Hint: the first paragraph after the table states whether the Monday success
metric was met. An understood negative result passes; an unexplained win
does not. Write "what failed first" the day it happened. Then close the
phase: tag `month-09-complete`, write `retro.md`, open the issue.
Accept when: committed as `writeup.md` with the benchmark table embedded.

## Review

1. Week 25: the gate item — derive scaled dot-product attention, multi-head
   shapes, and the √d_k argument cold, on paper. No notes.
2. Week 13: monthly rotation — re-derive backprop for a 2-layer net with
   softmax + cross-entropy.
3. Week 24: what physics validation did Capstone 2 use, and which observable
   was hardest to match? Why is that observable hard for a latent-variable
   model?
4. Week 32: state your extractor's headline F1 and its worst field — from
   memory, then check.
