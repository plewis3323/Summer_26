# Week 47 Project — Final Capstone, Build Phase

## Objective

Execute the build phase of your final capstone: by the end of this week, the system
described in `PROPOSAL.md` exists end-to-end, every baseline in the evaluation plan
has been run, and one full (unflattering is fine) evaluation of the real system has
been recorded. This is the project the whole course has been pointing at — it is the
repo you will demo in Week 48, put at the top of your portfolio, and show to
employers, labs, or your thesis committee. The week runs as a sprint (see
`lesson.md`): the proposal is the plan, milestones are checked daily, deviations go
in the scope ledger.

## Background — choosing your track (decided in Week 46, confirmed here)

You chose a track in Week 46's E4; this section is the fuller guidance that choice
was based on, kept here because it is also the frame for the build. All five tracks
are legitimate; they differ in what they *signal* (see `03-Project-Roadmap.md` and
`05-Two-Year-Path.md`). Science-domain and general-domain data are both allowed.

**(a) HEP analysis copilot** — an agent (Weeks 37–40) driving your Week-39 MCP
tools, with RAG (Week 34) over analysis documentation, that carries out real
physics-analysis tasks end-to-end. The course's resident example wraps an sPHENIX
workflow: sPHENIX is a detector at RHIC, the Relativistic Heavy Ion Collider at
Brookhaven National Lab, which collides heavy nuclei to study the quark–gluon
plasma — the state of matter where quarks and gluons are briefly unconfined. If you
have access to a different real analysis workflow (any research code you actually
use), wrap that instead; "real" is the load-bearing word. *Ships:* agent + MCP +
RAG, recorded demo of an actual analysis task. *Signals:* AI engineering +
AI-for-science — the strongest track for lab "AI engineer" roles and for making
yourself visibly useful to your collaboration. *Riskiest step:* the real workflow's
mess (auth, file layouts, undocumented conventions) eating the week.

**(b) Generative fast-sim** — a diffusion or VAE model (Weeks 22, 35; flows
optional) generating
calorimeter showers (the energy cascades particles produce in an energy-measuring
detector layer), conditioned on incident energy/angle, validated the way physics
demands: energy response linearity, resolution vs √E, shower-shape distributions
against the full simulator — head-to-head vs your Capstone-2 VAE and/or Capstone-3
diffusion model, aimed at a CaloChallenge-style writeup (the community benchmark
for learned calorimeter simulation). *Ships:* generator + validation writeup.
*Signals:* research + AI-for-science — the strongest track for "ML for simulation"
lab roles and thesis-adjacent publication. *Riskiest step:* training cost; the
compute budget must be real.

**(c) Paper-to-pipeline agent** — an agent that reads a heavy-ion physics paper
(heavy-ion: collisions of large nuclei, as at RHIC) and scaffolds a runnable
analysis module — the resident example targets Fun4All, sPHENIX's C++ analysis
framework, generating module skeletons and steering macros from the paper's
described event selection and observables. The agent orchestrates your Week-38
patterns and your Week-45 reading workflow, mechanized. *Ships:* the agent, plus
≥2 papers turned into scaffolds, one verified runnable. *Signals:* AI engineering —
the most novel demo of the four; also the highest variance. *Riskiest step:*
defining "correct scaffold" tightly enough to evaluate (the evaluation plan carries
this project or kills it).

**(d) Reproduce-and-extend** — take a recent HEP-ML paper, reproduce its central
result (Week 45's method, at full scale), then add one novel twist: a new ablation,
a new dataset, a robustness study, an architectural variant — written as an
arXiv-ready note. *Ships:* reproduction + extension + the note. *Signals:* research
scientist — per `03-Project-Roadmap.md`, if you aim at research roles, this track
polished into a workshop submission during the post-course slack month is the
strongest single move. *Riskiest step:* the reproduction stalling and starving the
extension; the cut line here is "reproduce within tolerance by day 3 or narrow the
extension."

**(e) Evaluated LLM service** — one model, RAG system, or agent behind the
Week-44 FastAPI+Docker pattern, with evals in CI, a cost and latency envelope,
and the Week-37 injection tests green. Science or general domain. *Ships:*
service repo a stranger can `docker run`, `SERVICE.md`, eval table. *Signals:*
AI Engineer — the strongest track if the job you want is "ship LLM systems."
*Riskiest step:* pretending a notebook demo is a service; the cut line is
"TestClient + container + one CI eval job by day 3."

Pick the track that feeds your thesis or your first AI-job application — ideally
both. Write the role sentence in Week 46 and do not hedge it.

## Data

Named in your `PROPOSAL.md`; typical per track — all public and free:

- (a) Your own analysis files/docs plus the Week-39 MCP tool suite's data; where a
  collaboration's data cannot be shared, structure the repo so a CERN Open Data
  (opendata.cern.ch) sample drives the public demo.
- (b) The CaloChallenge datasets (search "CaloChallenge datasets", Zenodo-hosted;
  1–10s of GB by dataset tier) or your Capstone-2/3 shower data for continuity.
- (c) Open-access heavy-ion papers from arXiv; Fun4All tutorial repos (public on
  GitHub) as scaffold references.
- (d) Whatever the target paper used — public availability was a Week 46 selection
  criterion; verify the download *on day 1*, not day 3.
- (e) The model checkpoint and eval set from Weeks 32/34/40; synthetic injection
  strings in `tests/test_security.py`.

## Build steps (daily milestones)

Instantiate these from `PROPOSAL.md`'s milestone section; check one off per day,
in order. They are the same six the week README lists.

1. **Day-1 walking skeleton.** The full pipeline runs on toy input — trivial
   model / canned agent response / 100-event sample — via one command, producing
   output in the real format. Accept when: one command produces a (bad) end-to-end
   result on day 1.
2. **Data and environment locked.** Real inputs downloaded, staged, versioned;
   loading code under test. Accept when: a data-integrity check (counts, hashes,
   schema) passes in CI.
3. **Baselines first.** Every baseline named in the evaluation plan runs through
   the *same* evaluation harness and logs its metrics. Accept when: the results
   table exists with all baseline rows filled, before the main system is tuned.
4. **Core system to proposal spec.** The track's central component — agent loop +
   tools / generative model / extraction agent / reproduced method — reaches its
   mid-week milestone. Accept when: the milestone's pre-committed metric or
   capability check from `PROPOSAL.md` passes.
5. **First full evaluation.** The complete evaluation plan runs once against the
   real system, however unflattering the numbers. Accept when: `results/` contains
   machine-readable metrics for system and baselines from the same harness.
6. **Scope ledger current.** Every deviation from the proposal recorded — what,
   why, cost — and the build log has an entry for every working day. Accept when:
   `BUILDLOG.md` has daily entries and the ledger; an empty ledger alongside
   missed milestones is a fail.

Day 7 is slack. If it goes unused, spend it on Week 48's stress-test list, not on
new features.

## Acceptance gate (end of Week 47 — from `03-Project-Roadmap.md` and the syllabus)

- Tested (`pytest -q` green on the load-bearing paths) and reproducible: fresh
  clone + `uv sync` + one command reruns the evaluation.
- Skeleton-to-real pipeline complete; all baseline rows filled; first full
  evaluation results in `results/` in machine-readable form.
- `BUILDLOG.md` with daily entries and the scope ledger.
- Rough is fine; unmeasured is not. (The polish gate — demo, writeup, talk — is
  Week 48's, not this week's.)

## Writeup requirements (this week: raw material only)

No report is due this week — Week 48 writes it — but the report's raw material is:
the daily `BUILDLOG.md` entries (results with run ids, decisions with reasons),
the scope ledger, and a "what failed first" note: the first substantive thing that
broke this week and what it cost, written the day it happens, not reconstructed.

## Stretch goals (only from the slack day, only if milestones 1–6 are done)

- Start Week 48 early: draft the stress-test list, or rough out the architecture
  diagram while the design is fresh.
- Add one extra baseline that makes the Week-48 comparison table stronger (e.g. a
  zero-shot/no-RAG ablation for tracks a/c; your Capstone-2 VAE row for track b).
- Track (d) only: begin the extension's second seed sweep so Week 48 has variance
  to report.
