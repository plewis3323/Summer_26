# Week 36 — Capstone 3: Shipping a Phase-3 Result

~1 hr reading — this week's time goes into the build (`project.md`). Before
starting you should have: the Week-32 extractor with a per-field F1 you can
quote from memory, the Week-34 RAG library and metrics table, the Week-35 DDPM
warm-up (and its sampling-cost note), and the Capstone-2 repo with its physics
validation suite. This short lesson is about *how to run the week*: choosing a
track from evidence you already own, freezing the eval before the system is
clever, and getting through the Phase 3 gate.

## 1. What this capstone proves

Capstone 2 proved you could take a deep-learning idea to a defensible physics
result. Capstone 3 proves the Phase 3 claim: you can *compose* the transformer
stack — retrieval, a fine-tune, a generative model — into one system a stranger
can run, and evaluate it against a baseline under a protocol you froze while
you were still impartial. The artifact is the repo; the gate (§5 of the
syllabus) also demands the *understanding*: scaled dot-product attention,
including the $\sqrt{d_k}$, re-derived cold, because that derivation is the
load-bearing math of everything you built this phase.

One scheduled week, two weeks' worth of building — same as Week 24, and the
syllabus §6 slack exists for the same reason. Take the second calendar week if
you need it and log it in the tracker. What you may not do is shrink the rigor
to fit the week: cut scope (fewer Q&A items, a smaller energy grid, one fewer
ablation), never cut tests, baselines, or the writeup's honesty.

## 2. Choosing your track

Both tracks are legitimate and both are staffed by things you already built.
The choice is the first committed line of the project (`project.md`, step 1),
so make it deliberately, Monday, using evidence you already own — not a paper
you have not read and not which demo would look better in a talk.

**(a) Physics-literature assistant** — Week 34 RAG plus the Week 32 LoRA
extractor, behind one interface. Free-text questions go through retrieval and
cited synthesis; structured metadata questions ("what $\sqrt{s_{NN}}$ did this
abstract report?") route to the extractor. Evaluated on a *new* held-out Q&A
set of ≥ 30 items, against two baselines: plain RAG (no extractor) and
no-retrieval (the instruct model alone). Choose (a) if the language thread is
your thesis-relevant one, if your Week-34 recall@5 left you with a specific
failure you want to kill, or if your Week-32 macro-F1 is a number you would
actually cite. Riskiest step: the comparison, not the wiring — an eval set
written after you have seen the system's answers is not an eval set.

**(b) Conditional diffusion calo generator** — the Week 35 DDPM warm-up, grown
up onto Capstone 2's shower dataset, conditioned on incident energy (and angle
if you have it). Validated the way Week 24 §4 and Week 35 §10 already required:
physics observables against generator truth, head-to-head with your Capstone-2
VAE, sampling cost in the same table. Choose (b) if generative models for simulation is your thread (it is the
CaloChallenge line into the final capstone), if the Week-35 warm-up actually
converged, or if Capstone 2 left you
with an observable the VAE lost and you want a rematch. Riskiest step: sampling
cost plus over-guidance silently buying pretty means and too-narrow widths
(Week 35 §9) — which is why the CFG sweep is a required ablation, not a
stretch.

Tie-breakers, in order: which result your thesis or target job would actually
cite; which failure you would learn more from; which dataset and checkpoint you
trust on Monday morning. Your Week-34 metrics table is evidence for (a)'s
feasibility; your Week-24 writeup plus Week-35 loss curves are evidence for
(b)'s. Do not choose by which architecture is trendier — the writeup, not the
stack, is what a reader keeps.

## 3. Freeze the eval first

The single most common way this week goes wrong: the system is integrated,
then you write questions it happens to answer, or you pick energies at which
the showers look good. That is the identical-protocol failure from Week 24,
translated into whichever track you picked.

The discipline, non-negotiable and encoded in the build steps:

- **The held-out set is committed before integration finishes.** Track (a):
  ≥ 30 new Q&A pairs, not recycled from Week 34, each with a gold answer and
  gold source span, plus a hash-dated provenance line. Track (b): the energy
  (and angle) grid and the list of physics observables, frozen as reference
  histograms from the generator, before you sample the diffusion model onto
  them. Once your system's numbers exist, every item you add is contaminated
  by knowing them.
- **Baselines run under this repo's protocol, on that frozen set.** Track (a):
  plain RAG and no-retrieval, same judge, same spot-check budget, same
  recall@5 definition. Track (b): the Capstone-2 VAE retrained or re-evaled
  here on the same test showers — quoting last month's table is not a
  comparison (Week 24 §3, again).
- **The success metric is named in the Monday proposal.** One number that
  decides success (macro answer-quality vs the better baseline; KS or χ² on
  the hardest shower-shape, plus a sampling-ms budget). Deciding after the
  results are in what "success" means is the same sin with a delay.

## 4. What "evaluated" means on each track

ML metrics answer "is the loss down?" The capstone answers "would a user, or
an analysis, be misled?" Different question; you answer both.

For track (a): a fluent answer with a citation is not a result. The table is
three systems × three families of number — **answer quality** (LLM-as-judge
plus a human spot-check on a declared subset; you know the judge's biases
from Week 32), **groundedness / faithfulness** (does the answer's content
sit in the retrieved spans?), **retrieval recall@5** (was the gold chunk in
the top 5 at all?). Then five worst answers, each attributed to a *stage*
with evidence: retrieval miss vs synthesis failure vs extractor error. If
you cannot say which stage failed, you do not yet have a system, you have a
demo.

For track (b): sample grids prove nothing (Week 23 §9, Week 24 §4, Week 35
§10 — the course has said this three times). The suite is the Capstone-2
suite, now also vs the VAE:

- **Energy response** — mean of $E_{\text{gen}}/E_{\text{true}}$ vs true
  energy (linearity, not just an average ratio).
- **Energy resolution** — width of that ratio vs energy, read against the
  generator's own $1/\sqrt{E}$ trend.
- **Shower shapes** — ≥ 2 distributions a photon-ID analysis would cut on,
  generated vs truth, each with χ² or KS, at ≥ 3 energies.
- **Sampling cost** — wall-clock per shower, same hardware, VAE vs DDPM, in
  the table. A comparison without this row is marketing (Week 35 §10).

CFG weight (or one architecture choice) is swept against at least one of
those observables. High $w$ that nails the mean and kills the width is a
failed generator that looks like a successful one on the wrong plot.

## 5. The gate, and how the week runs

The Phase 3 gate has two halves. The repo half: one script runs the system
from a fresh clone (`uv sync` + one command → the benchmark table, within
stochastic noise). Track (a) that is the assistant; track (b) the validated
generator. The table is vs the stated baselines — a clean, understood loss
passes; an unexamined win does not. (Syllabus §5 is written in track-(a)
language, "fine-tuned checkpoint + RAG"; the equivalent for (b) is the
conditional generator plus the VAE table. Either track closes the phase.)

The derivation half: scaled dot-product attention, multi-head shapes, and
the $\sqrt{d_k}$ argument (Week 25), cold, on paper, scanned into the repo.
Schedule it for a *fresh* morning, not the last hour of the last day — it is
the review block and the gate simultaneously, and rushing it defeats both.
Anything you had to look up gets flagged into the month's open-question
issue. Monthly rotation also wants a 2-layer backprop (Week 13); do it the
same morning if the attention pass is clean.

The build itself is specified in `project.md`: proposal → frozen eval →
baselines → integrated system → table and failure analysis → writeup → gate
check, each with its accept criterion. Work it in order; the ordering is
§3 made mechanical. Then close the phase: tag `month-09-complete`, write
`retro.md`, open the issue.

## Check yourself

1. The gate accepts a system that *loses* to its baseline. Under what
   condition, and why is that the right policy?
2. Why must the held-out Q&A (track a) or the energy grid and reference
   histograms (track b) be committed before the system is finished?
3. Track (a): name the three systems in the table and the three metric
   families. Why is "the answers looked good" not a row?
4. Track (b): why is a grid of convincing showers insufficient, and which
   row of the table is the one Week 35 said you must not omit?
5. You write five extra Q&A items on Thursday after seeing Wednesday's
   failure modes. What did you just do to the eval?
6. Which derivation closes the Phase 3 gate, and what happens to a step you
   had to look up?

## Answers

1. When the writeup explains the failure — what was tried, what the evidence
   says about *why* (retrieval? extractor field? posterior of the VAE?
   sampling steps? protocol?), and what would change the outcome. The gate
   tests defensible knowledge; an understood negative result is that, an
   unexplained win is not (syllabus §8).
2. Once the system's numbers exist, every new item or energy you add is
   chosen with those numbers in mind — implicit tuning of the eval against
   the comparison. Freezing the target first keeps you impartial, the same
   rule as Week 24's baseline-first commit.
3. Assistant vs plain RAG vs no-retrieval. Quality (judge + spot-check),
   groundedness, recall@5. Eyeballs miss retrieval misses that a fluent
   model papered over — the failure Week 34 was built to catch.
4. Eyeballs miss tails, conditional miscalibration, and width-vs-energy.
   Replace with response, resolution, and shower-shape distances, at ≥ 3
   energies, vs truth *and* vs the VAE. The omitted-at-your-peril row is
   sampling cost per shower.
5. Contaminated the held-out set: the new items are a training set for your
   own debugging, not a test set. They can go in a development split; they
   cannot go in the table.
6. Scaled dot-product attention with the $\sqrt{d_k}$ argument, re-derived
   cold and scanned in. A looked-up step gets flagged in the month's
   open-question issue and re-derived when the issue closes.

## New terms

- **identical-protocol rule** — baselines and candidate share the frozen
  eval set, the metric definitions, and the compute/spot-check budget;
  anything else compares effort, not systems. (Week 24, still in force.)
- **groundedness / faithfulness** — whether the generated answer's claims
  sit in the retrieved evidence, as opposed to being fluent invention.
- **recall@5** — fraction of questions for which the gold chunk is among
  the five retrieved; a retrieval metric, not a generation metric.
- **no-retrieval baseline** — the same instruct model, no library, same
  questions; isolates what RAG is actually adding.
- **sampling cost** — wall-clock (or FLOPs) per generated shower; the axis
  on which diffusion pays for quality (Week 35).
- **understood negative result** — a baseline loss plus a writeup that
  diagnoses it; passes the gate where an unexplained win would not.

## Going deeper

No new reading — this week runs on your own artifacts. Re-read before
starting: your Week-32 writeup and per-field F1 (track a), Week-34 § on
evals and your metrics table (track a), your Week-24 writeup and the
hardest observable (track b), Week-35 §§9–10 and your warm-up curves
(track b), and the Capstone 3 section of `03-Project-Roadmap.md`. If you
must look at one external thing, make it the RAGAS faithfulness metric
definitions (track a) or the CaloChallenge evaluation observables
(track b) — as models of what "evaluated properly" looks like in the
field.
