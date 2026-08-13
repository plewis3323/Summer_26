# Week 36 — Capstone 3 (GPU week)

Ship one system end to end: either a domain-literature assistant (language track;
physics papers by default) or a conditional diffusion generator (simulation track;
calorimeter showers by default) — a defended result with an eval, not a demo.
A general document library or public image set with the same protocol is allowed.

## Objectives

- Choose a track by day 1, in writing, with one sentence on why.
- Integrate components you already built (Weeks 27–35) rather than building new ones.
- Define the evaluation before finishing the system; freeze the held-out set first.
- Ship a reproducible repo: fresh clone + `uv sync` + one command → the benchmark table.
- Pass the Phase 3 gate, including re-deriving attention cold.

## Core material (~3 hrs)

No new reading. Re-read your own: Week 32 writeup and Week 34 metrics table (track a),
or Week 24 Capstone 2 writeup and Week 35 derivations (track b). Budget the freed hours
into building.

## Exercises (built when the week starts)

Milestones, not exercises. Track (a) — domain-literature assistant (physics default):

1. Architecture: Week 34 RAG + Week 32 extractor behind one interface; answers cite
   sources; metadata queries route to the extractor. Accept when: one script answers
   both a free-text and a structured query end to end.
2. Held-out eval: 30 new Q&A pairs (not from Week 34's set), frozen before integration
   finishes. Accept when: file committed with gold sources/answers and a hash-dated
   provenance line.
3. Benchmark: assistant vs plain RAG vs no-retrieval LLM on the held-out set —
   answer quality (judge + spot-check), groundedness, retrieval recall@5. Accept when:
   one table, three systems, all metrics.
4. Failure analysis: 5 worst answers dissected — retrieval miss vs synthesis failure vs
   extractor error. Accept when: each failure attributed to a stage with evidence.

Track (b) — conditional diffusion calo generator:

1. Scale the Week 35 warm-up to Capstone 2's shower dataset, conditioned on incident
   energy (and angle if available). Accept when: training converges with loss curves
   and fixed-seed sample grids saved.
2. Physics validation vs Geant4-level truth: energy response + resolution vs condition,
   longitudinal and radial shower profiles, at ≥ 3 energies. Accept when: overlay plots
   with a quantitative distance (χ²/ndf or KS) per observable.
3. Head-to-head vs the Capstone 2 VAE: same observables, same test set, plus sampling
   time per shower. Accept when: one comparison table and a verdict paragraph.
4. Guidance/ablation: CFG weight (or one architecture choice) swept. Accept when: its
   effect on ≥ 1 physics observable is plotted.

Both tracks:

5. Repo hygiene: tests green, pinned deps, README with one-command repro. Accept when:
   fresh-clone run reproduces the benchmark table within stochastic noise.
6. Writeup (~2 pages): problem, system, eval, results, limitations, what failed first.
   Accept when: committed as `writeup.md` with the benchmark table embedded.

## Deliverable

The Capstone 3 repo, tagged. Phase 3 gate (Syllabus §5): fine-tuned checkpoint + RAG
system callable from one script (track a) or validated generator (track b), benchmark
table vs baseline, and the attention derivation re-done cold — scan it into this folder.

## Review

- Week 25: the gate item — derive scaled dot-product attention, multi-head shapes, and
  the √d_k argument cold, on paper. No notes.
- Week 13: monthly rotation — re-derive backprop for a 2-layer net with softmax +
  cross-entropy.
- Week 24: what physics validation did Capstone 2 use, and which observable was hardest
  to match? Why is that observable hard for a latent-variable model?
- Week 32: state your extractor's headline F1 and its worst field — from memory, then
  check.
