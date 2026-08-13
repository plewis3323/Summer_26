# Week 36 Project — Capstone 3

## Objective

Ship one Phase-3 system end to end: either (a) a domain-literature assistant
(RAG over a document library you assembled, with the Week-32 extractor for
structured lookups, evaluated on a held-out Q&A set) or (b) a conditional
diffusion calorimeter generator (DDPM conditioned on energy/angle, physics-
validated, head-to-head vs your Capstone-2 VAE). Choose **one** track. Science-domain data is the default; a general document
library or image set with the same eval discipline is allowed (say so Monday).
The gate (from `03-Project-Roadmap.md` and syllabus §5): **one script runs the
system; benchmark table vs the stated baseline** (zero-shot / no-RAG for
(a); VAE for (b)), and **scaled dot-product attention is re-derived cold**
and filed. This closes Phase 3.

## Background — the two tracks, and the physics they wrap

Both tracks compose things you already built. The choice is the first
committed line of the project (build step 1); make it Monday from evidence
you own — Week-32 F1 and Week-34 recall@5 for (a), Week-24 writeup and
Week-35 loss curves for (b) — not from which demo would look better in a
talk.

**(a) Physics-literature assistant.** Heavy-ion physics (the course's
resident context) collides large nuclei to study the **quark–gluon plasma**,
the briefly deconfined state of quarks and gluons. The literature — arXiv
nucl-ex / nucl-th abstracts you scraped in Week 27, plus PDFs you ingested
in Week 34 — is dense with the same few fields over and over: collision
system (Au+Au, p+Pb), center-of-mass energy √s_NN, observables (v₂, R_AA),
centrality, experiment (STAR, sPHENIX). A working assistant has to do two
different jobs without mixing them up:

- **Free-text synthesis** — "what did sPHENIX measure for v₂ in 0–10%
  Au+Au at 200 GeV?" — needs **RAG** (Week 34): retrieve passages, generate
  an answer that *cites* them, and refuse to answer from parametric memory
  alone. Evaluated on **recall@5** (was the gold chunk in the top 5?),
  **groundedness / faithfulness** (do the claims sit in the retrieved
  spans?), and answer quality (judge + a human spot-check).
- **Structured lookup** — "what √s_NN does this abstract report?" — needs
  the Week-32 **extractor** (LoRA-fine-tuned 1–3B model → schema-valid
  JSON). Routing metadata questions to the extractor is the point of
  composing the two; plain RAG will hallucinate a 200 when the abstract
  said 5020.

The no-retrieval baseline (same instruct model, no library) isolates what
RAG is adding. The plain-RAG baseline (no extractor) isolates what the
fine-tune is adding. Three systems, one frozen Q&A set, one table.

**(b) Conditional diffusion calo generator.** A **calorimeter** stops
particles and measures dumped energy as a shower on a tower grid (Week 20
Background, restated). Full simulation (Geant4) is too slow; **fast-sim**
learns the map from incident energy (and angle) to the deposited-energy
image. Capstone 2 (b) did this with a VAE — a one-pass decoder, a bound on
the likelihood, a known failure mode (blur, posterior collapse, widths too
narrow). This track does it with a **DDPM** (Week 35): destroy the shower
with a fixed Gaussian noising process, learn to predict the noise, integrate
back. Conditioning on incident energy is what makes it a simulator.
**Classifier-free guidance** (CFG) sharpens that condition at sampling time
and will happily buy a pretty mean by killing fluctuations — which is why
the CFG sweep is required, not a stretch.

Physics validation is the Capstone-2 suite, now also vs the VAE, at ≥ 3
energies:

- energy response (mean of E_gen / E_true vs E — linearity);
- energy resolution (width of that ratio vs E, read against 1/√E);
- shower shapes (lateral / radial / longitudinal profiles, or the Week-20
  moment and fraction stand-ins on the toy), each with χ² or KS;
- **sampling cost** per shower, same hardware — the axis on which diffusion
  pays for quality. A table without this row is marketing.

**Freeze the eval first (both tracks).** The held-out Q&A (a) or the energy
grid and reference histograms (b) are committed before the system is
finished. Baselines run on that frozen set, under this repo's protocol.
Quoting Week 34's table or Capstone 2's VAE numbers is not a comparison.
The success metric is named Monday; deciding it after the results are in is
the same sin with a delay.

## Data

Named in the Monday proposal; typical, all public and free:

- **Track (a):** the Week 34 document library (arXiv PDFs on the Week 27/32
  corpus topics; add a detector TDR or your own papers if you have them),
  the Week 32 extractor checkpoint and schema, the Week 34 index. The eval
  set is **new**: ≥ 30 Q&A pairs, not recycled from Week 34, each with gold
  answer and gold source, plus a hash-dated provenance line. Week 34's
  held-out file, if you kept it closed, can be a *second* test; it does not
  replace the 30.
- **Track (b):** Capstone 2's single-photon shower dataset (Week-20
  generator, or CaloChallenge if you already moved). Same test showers for
  DDPM and VAE. Freeze the condition grid (energies, and angles if you have
  them) and the reference histograms before sampling the diffusion model.

**Compute** (syllabus §8): designated GPU week. Track (a) is generation-
heavy eval (three systems × 30 questions, plus the judge) more than
training. Track (b) is training + T-step sampling; budget wall-clock
honestly and cache every sample grid to disk so no measurement is rerun by
accident. Take the second calendar week if you need it.

## Build steps

Do them in order; each is one of this week's exercises. The ordering *is*
lesson §3 made mechanical.

**Monday, both tracks:**

0. **Proposal.** Track, one sentence of why (evidence you already own), the
   single success number, the frozen-eval plan, risks (a: writing the Q&A
   after seeing answers; b: over-guidance and sampling cost). Accept when:
   it names the single number that decides success.

**Track (a):**

1. **Architecture** (E1). RAG + extractor behind one interface. Accept when:
   one script answers both a free-text and a structured query end to end.
2. **Held-out eval** (E2). ≥ 30 new Q&A, committed, hashed, before
   integration finishes. Accept when: file committed with gold
   sources/answers and a hash-dated provenance line.
3. **Benchmark** (E3). Assistant vs plain RAG vs no-retrieval — quality,
   groundedness, recall@5. Accept when: one table, three systems, all
   metrics.
4. **Failure analysis** (E4). Five worst answers, each attributed to a
   stage with evidence. Accept when: each failure names a stage.

**Track (b):**

1. **Scale the warm-up** (E1). Week 35 DDPM on Capstone-2 showers,
   conditioned on energy. Accept when: loss curves and fixed-seed sample
   grids saved.
2. **Physics validation** (E2). Response, resolution, profiles/shapes, ≥ 3
   energies, χ² or KS per observable. Accept when: overlay plots with a
   quantitative distance per observable.
3. **Head-to-head vs VAE** (E3). Same observables, same test set, sampling
   time per shower. Accept when: one comparison table and a verdict
   paragraph.
4. **CFG / ablation** (E4). Sweep w (or one architecture choice) against
   ≥ 1 physics observable. Accept when: the effect is plotted.

**Both tracks:**

5. **Repo hygiene** (E5). `pytest -q` green, pinned deps, one-command repro
   of the table. Accept when: fresh clone + `uv sync` + one command
   reproduces the benchmark table within stochastic noise.
6. **Writeup** (E6). ~2 pages, table embedded. Accept when: `writeup.md`
   committed with the table.
7. **Gate check.** Scaled dot-product attention, multi-head shapes, √d_k,
   cold, on paper, scanned into this folder. Looked-up steps flagged for
   the month's open-question issue. (Monthly rotation: 2-layer backprop the
   same morning if attention is clean.)

## Acceptance gate (end of Week 36 — from `03-Project-Roadmap.md` and syllabus §5)

- **One script runs the system** from a fresh clone (`uv sync` + one
  command → the benchmark table, within stochastic noise). Track (a): the
  assistant (fine-tuned checkpoint + RAG, syllabus §5's wording). Track
  (b): the validated generator. Either track closes the phase.
- **Benchmark table vs the stated baseline.** Track (a): vs plain RAG and
  vs no-retrieval (roadmap: "vs zero-shot"). Track (b): vs the Capstone-2
  VAE, including sampling cost. A clean, understood loss passes; an
  unexamined win does not (syllabus §8).
- **Attention derivation re-done cold** and filed.
- Tested: `pytest -q` green on the load-bearing paths. The repo ships:
  proposal, frozen eval set, baselines run under this protocol, system,
  table, failure analysis (a) or CFG plot (b), writeup, derivation scan.

Then close the month and the phase: tag `month-09-complete`, write
`retro.md`, open the open-question issue.

## Writeup requirements (~2 pages)

- Track chosen and why (one sentence of evidence you already owned).
- The system, in one paragraph naming which week built each piece.
- The headline table, and whether the Monday success metric was met — first
  paragraph after the table.
- Track (a): five failures attributed to a stage; extractor's worst field
  from memory, then checked. Track (b): hardest observable, CFG's effect on
  a width, sampling-ms row.
- **What failed first** — written the day it happened, not reconstructed.
- Limitations, and what you would do with a month — including whether that
  month is the Week 47 track this capstone is a rehearsal for ((a) → final
  (a) or (c); (b) → final (b)).

## Stretch goals (only after the gate)

- **Track (a):** constrained decoding on the extractor path; or score the
  frozen Week-34 held-out set as a second, pre-registered test; or add one
  field to the schema and measure zero-shot vs top-up fine-tune (Week 32
  stretch, now with a user in the loop).
- **Track (b):** reduced-step sampler (DDIM) with the quality–cost curve
  plotted; or condition on angle as well as energy; or a CaloChallenge
  dataset if you have been on the toy the whole time.
- **Both:** one extra baseline that makes the table stronger (a BM25-only
  row for (a); your Week-23 flow for (b), with the NLL-vs-bound caveat).
