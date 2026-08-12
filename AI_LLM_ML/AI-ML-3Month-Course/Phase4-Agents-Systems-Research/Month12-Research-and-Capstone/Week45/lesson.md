# Week 45 — Reading & Reproducing Papers

~3 hrs reading + the exercises. Before starting you should be able to: train and
honestly evaluate a model end-to-end (Weeks 09–12, 16), explain benchmark
contamination and why single-seed comparisons mislead (Week 32), and run a tracked,
reproducible experiment (Week 41). You have also already read at least two papers
closely — *Attention Is All You Need* and GPT-2 in Week 26 — but without a system.
This week installs the system.

## 1. Why reading papers is a skill, not a chore

Everything you have learned so far came from a lesson whose job was to teach you.
A paper's job is different: it is an *argument* that a piece of work deserves your
attention, written by people who benefit if you believe it. That does not make
authors dishonest — most are careful — but it means a paper is structured to
persuade, and the evidence that would weaken the claim is the easiest thing to
leave out. Reading a paper well means recovering what was actually shown, which is
usually narrower than what the abstract says.

Two facts make this manageable:

1. **Papers are formulaic.** Almost every ML paper has the same skeleton (abstract,
   intro, related work, method, experiments, conclusion). Once you know the
   skeleton, you know where each kind of information hides, and you can read
   non-linearly.
2. **Most papers do not deserve a full read.** A working scientist skims dozens of
   papers for every one they study. The skill is deciding *how deeply* to read,
   fast — and having a fixed procedure for each depth.

The procedure this course adopts is S. Keshav's **three-pass method**, from his
short note *How to Read a Paper*. Read that note first (it is three pages, free —
search the title); the rest of this section restates it with ML-specific additions.

### Pass 1 — the five-minute triage (5–10 min)

Read only: the title, the abstract, the introduction, every section and subsection
heading, and the conclusion. Glance at the figures. Skim the reference list and
mark what you have already read.

At the end of pass 1 you write the **five Cs**:

- **Category** — what kind of paper is it? (New method? Benchmark/measurement?
  Analysis of an existing method? Position paper? Reproduction?)
- **Context** — what other work does it build on or compare against? What is the
  theoretical or empirical base?
- **Correctness** — do the assumptions look plausible, at a glance?
- **Contributions** — what does the paper claim to contribute? (Most intros list
  these explicitly; copy them verbatim.)
- **Clarity** — is it well enough written to be worth more of your time?

Then decide: stop here, or go to pass 2. Most papers stop here, and that is the
method working, not failing.

### Pass 2 — the careful read (~1 hour)

Read the whole paper, but skip proofs and derivation details. Your attention goes
to the **figures and tables** — in an ML paper, that is where the evidence lives.
For each figure: are the axes labeled and sensible? Are there error bars or any
statement of variance? Does the caption's claim match what the plot shows? Mark
every unread reference that seems load-bearing.

At the end of pass 2 you can summarize the paper, with evidence, to someone else.
This is the depth you use for papers relevant to your work but not central to it.

### Pass 3 — the virtual re-implementation (hours)

Attempt to re-derive and re-build the paper in your head (or on paper): given only
the text, could you write the code and reproduce the experiments? Every place you
get stuck is either something you need to learn or something the paper failed to
specify — and telling those apart is the point. Pass 3 is the depth you use for
papers you will build on. This week, your reproduction target gets a literal
pass 3: you actually write the code.

## 2. Claims vs evidence — the ML critique checklist

Physics has a culture of hunting systematic errors: no measurement is believed
until the ways it could be wrong have been enumerated and bounded. ML papers need
the same treatment, but the failure modes are different. Here is the checklist
this course uses. For any paper you critique, answer each item in writing.

**1. Quote the headline claim verbatim.** Not your paraphrase — the paper's own
sentence, usually from the abstract. Everything else is checked against this.

**2. What exactly was measured?** Which metric, on which dataset, on which split?
A claim of "state of the art" on one benchmark is a claim about that benchmark,
full stop. Recall from Week 32 that metric choice is itself a decision that can
flatter a method (accuracy on imbalanced data, ROC-AUC when the physics goal
lives at one working point, and so on — Week 10).

**3. Are the baselines fair?** The most common way an ML paper misleads is by
comparing a heavily tuned new method against untuned or outdated baselines. Ask:
did the baselines get the same hyperparameter search budget, the same data
augmentation, the same training compute? "Our method beats the baseline" means
little if the baseline was run at default settings. You saw in Weeks 11 and 16
how much tuning moves a BDT or an MLP; the paper's baseline got the same
sensitivity.

**4. Is there any statement of variance?** Deep learning results vary across
random seeds — often by more than the reported improvement. A single-seed gain of
0.3% is noise until shown otherwise (Week 32). Look for: multiple seeds, error
bars, standard deviations, or significance tests. Their absence is not proof of
fraud; it is proof the claim is weaker than it sounds.

**5. Are there ablations?** An **ablation** is an experiment that removes or
degrades one component of the method and re-measures, to show which parts of the
system actually cause the improvement. A method paper without ablations is
claiming "the whole package works" while showing nothing about *why*. Ask which
ablation you would demand before believing the headline number.

**6. Could the evaluation data have leaked?** **Benchmark contamination** (Week
32) — evaluation examples appearing in the training data — inflates results.
For classical benchmarks, check whether the splits are standard and whether any
preprocessing peeked at the test set (the leakage audit you ran in Capstone 1).
For LLM-adjacent work, ask whether the benchmark predates the model's training
data.

**7. Are comparisons compute-matched?** A bigger model trained longer usually
wins (Week 28, scaling laws). If the new method used 10× the compute of the
baseline, the interesting question — "is the *idea* better, per unit compute?" —
was not answered.

**8. What is missing?** Datasets where the method was presumably tried and not
reported. Metrics conspicuously absent. The limitations section (if there is
one — its absence is itself a finding).

None of these items requires believing the authors did anything wrong. The
checklist measures the *gap between claim and evidence*, and a good critique
states that gap precisely: "the paper claims X; the experiments support the
weaker claim Y; to support X you would additionally need experiment Z."

### HEP-ML failure modes, specifically

Papers applying ML to high-energy physics (HEP — the physics of particle
collisions, done at accelerators like the LHC and RHIC) add domain-specific traps:

- **Simulation-only evaluation.** Most HEP-ML papers train *and* evaluate on
  simulated collisions, because simulation provides perfect labels. But
  simulators are imperfect models of the real detector, so "99% accuracy on
  simulation" says nothing about performance on real data until the
  simulation-to-data gap is measured. Ask whether the paper addresses it.
- **Unphysical metrics.** Image-generation papers use perceptual scores; a
  calorimeter simulation must instead reproduce *physics* distributions — energy
  response, resolution, shower shapes (Weeks 23–24). A generative fast-sim
  evaluated only by eye or by a generic ML metric has not been validated.
- **Benchmark monoculture.** Much of jet tagging is evaluated on one or two
  public simulated datasets. Results can overfit the quirks of a specific
  simulator version and event selection.

(Quick re-grounding, since these terms last appeared weeks ago: a **jet** is the
collimated spray of particles produced when a quark or gluon flies out of a
collision; **jet tagging** is classifying which particle started the jet. A
**calorimeter** is a detector layer that stops particles and measures their
energy; the cascade a particle produces inside it is a **shower**. **Fast
simulation** replaces the slow physics-accurate simulation of showers with a
cheap learned generator — the motivation for Weeks 23 and 35.)

## 3. Choosing a reproduction target

The fastest way to learn what a paper actually contains is to reproduce a piece
of it. Not the whole paper — one number: a single table row or a single curve.
Choosing that target well is most of the battle. Criteria:

1. **Small.** Reproducible on your hardware in under one GPU-day, ideally under
   an hour. A baseline row (an MLP, a BDT, a small CNN) from a benchmark table is
   often perfect — you know how to build all of those from scratch.
2. **Public data.** The exact dataset must be downloadable (CERN Open Data,
   Zenodo-hosted benchmark sets like the top-tagging landscape dataset or the
   CaloChallenge datasets, UCI, HuggingFace). If the paper's data is private,
   pick a different paper.
3. **A concrete number to hit.** "We achieve AUC 0.9820 ± 0.0004" is a target;
   "our method works well" is not. Prefer numbers with a stated uncertainty, so
   "success" has a definition.
4. **Enough description to implement from.** Skim the method section: are the
   architecture, loss, optimizer, and training schedule stated? Gaps are
   expected — cataloguing them is part of the exercise — but a paper that
   specifies almost nothing makes a frustrating first target.

Write the choice down *before* coding, in a short `repro_plan.md`: the exact
number(s) targeted, the data source, your compute estimate, and what counts as
success (within the paper's stated uncertainty; or within a tolerance you commit
to now).

## 4. The reproduction workflow

Reproduce from the **paper's description alone**. Do not open the authors'
reference code at the start — the entire value of the exercise is discovering
what the text underdetermines. The workflow:

1. **Pass 3 on the method section.** On paper, write the full pipeline the way
   Week 41 taught you to think about experiments: data → preprocessing →
   architecture → loss → optimizer → schedule → evaluation. Every box you cannot
   fill from the text goes on the **gap list**.
2. **For each gap, make a documented guess.** "Paper does not state the batch
   size; guessing 256 because the dataset card suggests it and it fits memory."
   The gap list with your guesses *is* a research artifact — reproducibility
   studies find that these unstated choices often move results more than the
   headline method does.
3. **Build the smallest version first.** Subsample the data, train for a few
   epochs, verify the pipeline runs end-to-end and the metric is computed the
   same way the paper computes it. Metric mismatches (different ROC convention,
   different working point, different split) cause more false "failed
   reproductions" than modeling errors do.
4. **Run the real thing, tracked.** Seeds fixed, config recorded, results logged
   (Week 41). Run more than one seed if compute allows — you need your own
   variance to interpret the gap.
5. **Compare honestly.** Your number, the paper's number, the paper's stated
   uncertainty, your observed seed-to-seed spread. Then diagnose: if you are
   outside tolerance, which gap-list guess is the leading suspect? Test the
   suspect if cheap; write "untested" if not.
6. **Only now, peek.** If stuck after a real attempt, read the reference code —
   and log *when* you peeked and *what* the text had failed to tell you. That log
   entry is a finding about the paper, not a confession.

A reproduction that lands outside the paper's error bar, with the gap quantified
and diagnosed in writing, is a **successful exercise**. The deliverable is the
comparison and the diagnosis, not the matching number.

## 5. The reading log

You will read hundreds of papers over a career. A **reading log** makes them
retrievable: one entry per paper, written the day you read it, answering three
questions in at most ten lines —

- **What did they claim?** (verbatim headline claim + your one-line restatement)
- **What convinced me?** (the specific figure/table and why it carries weight)
- **What didn't?** (checklist items that failed, missing experiments, doubts)

Plus tags for retrieval (topic, dataset, method family). Plain markdown in one
file is fine; a reference manager (Zotero is free) works too. The test of a good
entry: six months from now, it answers "should I re-read this paper?" without
opening the PDF. Summaries fail that test; claims-and-doubts pass it.

## 6. Worked example — three passes over ParticleNet

Here is the method applied to a paper you met in Week 21: Qu & Gouskos,
*ParticleNet: Jet Tagging via Particle Clouds*. (If you pick ParticleNet for your
own deep critique this week, this section is a head start; verify every detail
against the actual paper — part of the point is that you should not trust a
secondhand account, including this one.)

**Pass 1 card (five Cs):**

- *Category*: new-method paper — an architecture for jet tagging.
- *Context*: builds on point-cloud deep learning from computer vision (Dynamic
  Graph CNN / EdgeConv) and on the prior generation of jet taggers (jet images +
  CNNs, particle sequences + RNNs, particle sets + Deep Sets-style models).
  Treats the jet's constituent particles as an unordered set of points — a
  "particle cloud" — which matches the physics: the particles have no natural
  order (Week 21, permutation invariance).
- *Correctness (at a glance)*: plausible — permutation-invariant architectures
  are the right symmetry class for sets of particles, argued in Week 21.
- *Contributions*: the particle-cloud representation; the ParticleNet
  architecture (EdgeConv blocks adapted to particle features); benchmark results
  on public top-tagging and quark–gluon datasets claiming improvement over prior
  taggers.
- *Clarity*: well-structured; standard benchmark tables; worth a pass 2 for
  anyone working on jet tagging. → proceed.

**Pass 2 focus — the claims/evidence table.** The evidence is a small number of
benchmark tables comparing ParticleNet against earlier taggers on public
simulated datasets. Checklist highlights you would write down:

- *Baseline fairness*: the baselines are prior published models — trained by
  other groups, under other budgets. Standard practice, but it means the
  comparison inherits every group's tuning choices; ask what an equal-budget
  re-tuning would show.
- *Variance*: check whether the tables report an uncertainty on each entry and
  what it covers (multiple trainings? one training, bootstrap eval?). Whatever
  it covers, ask if the margin over the runner-up exceeds it.
- *Ablations*: ask what is ablated — does the paper separate "point-cloud
  representation helps" from "this particular architecture helps" from "more
  parameters help"? (This is exactly the ablation your Capstone-2a asked of you.)
- *Simulation-only*: both benchmarks are simulated; the paper's claims are about
  simulation. Note it.

**Pass 3 sketch — could you rebuild it?** From Week 21 you can: k-nearest-neighbor
graph in feature space, EdgeConv message passing, global pooling, an MLP head.
The gap list starts immediately: exact input features per particle and their
preprocessing/normalization, k for the neighbor graphs, learning-rate schedule,
number of epochs, early-stopping rule, how the reported number's uncertainty was
computed. Each gap is a line in `repro_plan.md` if you choose a ParticleNet row
as your target — though for a first reproduction, the *baseline* rows of the
same tables (a simple deep-sets model or a CNN) are smaller game and teach the
same lessons.

## Check yourself

1. In pass 1, you read the conclusion but skip the method section. Why is that
   the right order?
2. A paper reports a 0.4% accuracy improvement over the baseline, one seed,
   no error bars. Name two checklist items this fails and state what evidence
   would repair each.
3. What is an ablation, and what specific question does it answer that the
   headline benchmark table cannot?
4. Why is "evaluated on simulation only" a caveat for a HEP-ML paper even when
   the evaluation is flawless?
5. Your reproduction lands at AUC 0.971 against the paper's 0.9820 ± 0.0004.
   Your three seeds span 0.969–0.973. What do you conclude, and what do you do
   next?
6. Give two reasons to reproduce from the paper's text before opening the
   authors' code, even when the code is public.
7. What three questions must every reading-log entry answer, and why is "a good
   summary" not the goal?

## Answers

1. The conclusion states what the authors believe they showed, in their own
   ranking of importance — the fastest way to learn the paper's claims. The
   method section is only worth your time after you have decided the claims
   matter, and it only makes sense once you know what it is trying to achieve.
2. Fails *variance* (no seeds/error bars — repaired by multiple seeds showing
   the gain exceeds seed-to-seed spread) and *baseline fairness* (repaired by
   showing the baseline received an equal tuning budget). With single-seed deep
   learning runs, 0.4% is routinely within seed noise.
3. An ablation removes or degrades one component and re-measures. It answers
   "which part of the system causes the improvement?" — the benchmark table only
   shows that the *whole* system, with all choices bundled, scores well.
4. Because the deployment target is real detector data, and the simulator is an
   imperfect model of it. Perfect simulation performance bounds nothing about
   the simulation-to-data gap; a method can exploit simulator artifacts that do
   not exist in data.
5. The gap (0.011) is far outside both the paper's uncertainty and your seed
   spread (~0.002), so it is systematic, not noise. Next: go to the gap list,
   identify the most suspicious guessed decision (preprocessing and metric
   conventions first — they are the usual culprits), test the leading suspect,
   and write the diagnosis down whether or not you find it.
6. (a) The text-only attempt reveals what the paper underdetermines — the gap
   list is the exercise's main scientific product, and reading the code first
   destroys it. (b) Reference code often contains unstated fixes and settings
   that silently differ from the paper; implementing from the text tests the
   *paper*, which is what the community actually consumes.
7. What did they claim; what convinced me; what didn't. A summary restates the
   paper — it goes stale and forces a re-read. Claims-and-doubts record your
   *judgment*, which is the part you cannot recover from the PDF later.

## New terms

- **three-pass method** — Keshav's reading procedure: triage (pass 1), careful
  read of figures and claims (pass 2), virtual re-implementation (pass 3).
- **five Cs** — pass-1 output: category, context, correctness, contributions,
  clarity.
- **ablation** — removing/degrading one component of a method and re-measuring,
  to attribute the improvement.
- **benchmark contamination** — evaluation data leaking into training data,
  inflating results (first defined Week 32; central here).
- **compute-matched comparison** — comparing methods at equal training compute,
  so the idea is tested rather than the budget.
- **reproduction target** — the single number/figure you commit to reproducing,
  chosen before coding.
- **gap list** — the catalogue of decisions a paper leaves unspecified, with
  your documented guesses and their observed effects.
- **reading log** — per-paper record of claims, convincing evidence, and doubts,
  written for retrieval months later.
- **HEP** — high-energy physics: the study of fundamental particles via
  accelerator collisions; the course's application domain.
- **jet / jet tagging** — a collimated particle spray from a quark or gluon;
  classifying which particle started it.
- **calorimeter / shower** — an energy-measuring detector layer; the particle
  cascade inside it.
- **fast simulation** — a learned generator replacing slow physics-accurate
  detector simulation.

## Going deeper

- S. Keshav, *How to Read a Paper* — the three-pass method; three pages; adopt it
  verbatim. This week's spine.
- *ML Reproducibility Challenge* reports (any year; search the title) — skim a
  few; note how often reproductions fail on unstated details rather than on the
  core idea.
- *A Living Review of Machine Learning for Particle Physics* — the standing
  HEP-ML bibliography; where you find critique-pool and reproduction candidates
  beyond the reading list.
- Papers With Code (site) — links papers to code and benchmark leaderboards;
  useful for choosing a reproduction target with public data, and for the
  post-reproduction peek.
- Pineau et al., *Improving Reproducibility in Machine Learning Research* (the
  NeurIPS reproducibility program report) — what a whole field learned when it
  tried to enforce reproducibility; the checklist in §2 echoes its findings.
