# Week 48 Project — Final Capstone, Validate/Write/Present Phase

## Objective

Turn Week 47's working system into a defensible result. Freeze the system, run
the complete pre-committed evaluation plan with measured variance, stress it
past the happy path, ablate its core component, and produce the artifacts that
make the result real: a 6–10 page report with architecture diagram and honest
limitations, a clean-clone demo recording, a recorded mock research talk, and
the gate check that closes the course. The week's rules are `lesson.md` §1: the
proposal's thresholds are the grading key, the system is frozen from the first
evaluation run, and every number in the report regenerates from one command.

Suggested shape (instantiate from `PROPOSAL.md`'s Week-48 milestones; the
milestones are E1–E7 in `exercises.md`): day 1 freeze + final evaluation (E1);
day 2 stress suite (E2) and ablation (E3); days 3–4 report (E4); day 5 demo
(E5); day 6 talk (E6); day 7 gate check and close-out (E7).

## Validation by track

Same harness for every row; per-track specifics for the evaluation, stress
suite, and ablation:

**(a) HEP analysis copilot.** Re-run the Week-40 agent-eval harness on the
frozen system: the scripted analysis tasks, completion and correctness rates,
trajectory review (Week 40's method) on the failures. Stress: ambiguous
questions, adversarial questions, questions whose correct answer is "I can't
do that with these tools," tool timeouts/failures mid-trajectory, prompts that
tempt fabrication (report the hallucination rate, Week 32). Ablation:
retrieval off, or the tool set cut to the trivial subset — what does the
agent's machinery actually buy over the bare model?

**(b) Generative fast-sim.** The physics validation suite, CaloChallenge-style
(the community benchmark's own metric definitions): energy response linearity,
resolution vs √E, shower-shape distributions against the full simulator, and a
classifier-based two-sample score; head-to-head rows for your Capstone-2 VAE
and/or Capstone-3 diffusion baselines; sampling cost reported honestly.
Stress: conditioning values (energy/angle) at and beyond the training range,
rare shower topologies, sensitivity of the physics metrics to sampling
settings. Ablation: conditioning removed or the model shrunk — architecture or
parameter count?

**(c) Paper-to-pipeline agent.** The proposal's pre-committed "correct
scaffold" rubric applied to every generated scaffold; ≥2 papers processed, one
scaffold verified runnable; trajectory review (Week 40) on the misses. Stress:
a paper never touched during development, a paper with a missing or
unconventional section, an underspecified event selection — does the agent
flag the gap or invent one (log it as fabrication if it invents)? Ablation:
the structured reading stage replaced by a single naive pass — does the
mechanized Week-45 workflow earn its complexity?

**(d) Reproduce-and-extend.** The target paper's own evaluation protocol, run
on both the reproduction and the extension, with seeds on each; the paper's
published numbers as the baseline row. Stress: sensitivity of the extension's
claim to the gap-list guesses (Week 45) and to data subsampling — the
systematic-uncertainty study the arXiv note needs anyway. Ablation: the twist
removed is by construction the reproduction itself; report the twist's
measured gain against its cost in compute and complexity.

## Deliverables

- **Results table** — every pre-committed metric with mean ± spread across
  seeds/runs; all baseline rows; the ablation row; regenerated from `results/`
  by one command. Missed thresholds appear with their diagnosis, at equal
  prominence.
- **`stress.md`** — ≥5 stress tests: what was done, what happened, verdict —
  failures included.
- **`REPORT.md` (or PDF)** — 6–10 pages per the Writeup requirements below,
  with the architecture diagram.
- **Demo recording** — ≤10 minutes, fresh clone → `uv sync` → one command →
  the headline capability, narrated; the clean-clone run on camera.
- **Talk recording** — 15–20 minutes, slides, mixed HEP/ML audience; includes
  the limitations slide and the "what I'd do next" slide.
- **`GATE.md`** — the gate checklist, every item linked to its evidence, plus
  the scanned cold flagship derivation.
- **Close-out** — tag `month-12-complete`, 250-word `retro.md` in the month
  folder, and the final open-question issue (syllabus §9).

## Acceptance gate (End Phase 4 — from `00-Syllabus.md` §5)

- Final capstone demoed end-to-end; architecture diagram; honest limitations;
  one mock research talk delivered and recorded.
- The course-wide reproducibility bar: fresh clone + `uv sync` + one command →
  same result within stochastic noise — proven on camera by the demo.
- Tested: `pytest -q` green on the load-bearing paths (unchanged from Week
  47's gate; validation must be able to tell bugs from findings).
- No post-freeze tuning: any fix after E1 is logged in `BUILDLOG.md` and the
  full evaluation re-run.
- Every pre-committed threshold reported as met, or missed with a written
  diagnosis. Missed-and-diagnosed passes the gate; missed-and-undiagnosed, a
  quietly redefined threshold, or a report number the harness cannot
  regenerate fails it.

## Writeup requirements

`REPORT.md`, 6–10 pages, Week 46's skeleton with results filled in:

- Abstract with the headline numbers; intro whose contributions list is the
  proposal's promised claims, each marked kept or not.
- Method with the architecture diagram (drawn from the code, boxes mapped to
  the repo); deviations from the proposal summarized from the scope ledger.
- Results vs. every baseline the proposal named, plus the ablation row; claims
  at calibration level 4 (scope, number, variance, matching); every claim
  cites a table/figure and every table/figure supports a claim.
- Limitations: ≥3 real weaknesses a referee would find anyway — specific and
  consequential, sourced from `stress.md` and Week 45's checklist run against
  your own draft.
- "What failed first," assembled from `BUILDLOG.md` — written the day it
  happened, not reconstructed. Negative results go in.
- Future work: concrete next experiments, not aspirations — the final
  open-question issue starts here.

## After the course

- The post-course slack month is real budget (`00-Syllabus.md` §6). The
  strongest single use, especially for track (d): polish the capstone into a
  workshop submission (`03-Project-Roadmap.md`) — it converts a course
  artifact into a community artifact.
- The months 13–36 sketch (`03-Project-Roadmap.md`, "After the course"):
  year 2 — publish 1–2 ML-for-physics results, contribute to an open ML tool,
  apply the skills visibly inside sPHENIX; year 3 — role conversion with the
  portfolio: four capstone repos, a fine-tuned model, an MCP server, a
  publication or workshop paper, and a thesis that uses ML for real.
- Keep the reading log and the weekly build-and-measure rhythm at maintenance
  dose; keep one named open question at all times — that is what the final
  issue is for.
