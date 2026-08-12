# Week 46 — Research Writing & the Capstone Proposal

~3 hrs reading + the exercises (the main exercise is writing your capstone
proposal). Before starting you should be able to: critique a paper with the
claims-vs-evidence checklist (Week 45), state what your three earlier capstones
claimed and showed (Weeks 12, 24, 36), and estimate a training run's cost from
experience with your own tracked runs (Weeks 41–42).

## 1. Writing is part of the research, not the paperwork after it

The naive model of research is: do the work, then write it up. It fails in
practice because writing is where you discover what you actually showed. Vague
understanding survives in your head and in your code; it does not survive being
forced into sentences with subjects and verbs. Simon Peyton Jones's advice
(this week's spine talk, *How to write a great research paper*) goes further:
start writing *before* the work is done, because the writing tells you which
experiments you actually need.

This week uses that principle twice. First you learn the genre — how an ML paper
is built, sentence by sentence and figure by figure. Then you apply it
*forward*: instead of writing up finished work, you write the **proposal** for
work not yet done. The proposal is the same skeleton as a paper with the results
section replaced by an evaluation *plan* — and it becomes the contract that
governs Weeks 47–48.

## 2. The shape of an ML paper

Nearly every ML paper has the same sections, and each one owes the reader
something specific. Knowing the debts makes both reading (Week 45) and writing
mechanical.

**Abstract** (~150–250 words). A miniature of the whole paper: the problem, why
it is hard, the idea, the headline evidence *with numbers*, and the scope. A
reader decides from the abstract alone whether to spend pass 1 on you. The
classic failure is the vague abstract ("we explore...", "promising results") —
it hides the contribution and reads as having nothing to hide it for.

**Introduction** (~1 page). Peyton Jones's formula: state the problem, say why
it is interesting, say why it is hard, then state your **contributions as an
explicit list** — "We show X (Section 4); we prove Y (Section 3); we release
Z." Each contribution must be a *falsifiable claim* the paper then supports,
not an activity report ("we studied...", "we investigated..." — those are
things you did, not things the reader learns). The contributions list is the
paper's promise; the experiments section is where the promise is kept.

**Related work.** What existed before, and precisely how this work differs.
Two debts: honesty (do not weaken competitors to look better — Week 45's
checklist catches it anyway) and *placement* — many strong papers put related
work late, after the method, so the reader can understand the comparison. The
debt is a fair map, not a literature dump.

**Method.** Enough detail that a competent reader could reimplement — Week 45's
pass 3 is the test. Every symbol defined at first use; every design choice
either justified or flagged as a choice. Your Week 45 gap list is a catalogue of
method-section failures; do not write those failures yourself.

**Experiments.** The evidence for each claimed contribution, in the same order
as the contributions list. Setup stated fully (datasets, splits, baselines and
their tuning budget, seeds, compute). Every claim in the paper should trace to
a table or figure here; every table or figure should support some claim.
Orphans in either direction are bugs.

**Limitations / discussion.** What the evidence does *not* show. Written well,
this section builds trust and pre-empts referees; written as false modesty
("our method may not work on all datasets") it does nothing. A believable
limitation is specific and consequential: "all evaluation is on simulation;
the simulation-to-data gap is unmeasured" (Week 45).

**Conclusion.** The claims, restated in light of the evidence, plus what is
next. No new claims may appear here.

### The reverse outline

The genre's main diagnostic tool: take a finished draft (yours or a published
paper), and write **one sentence per paragraph** stating what that paragraph
does. Read the sentence list alone. If the list is a coherent argument, the
structure works; where it jumps, repeats, or wanders, the draft needs surgery
at the paragraph level — no amount of sentence-polishing fixes a structural
hole. You will run this on a published paper in the exercises.

## 3. Claims exactly as strong as the evidence

Week 45 taught you to measure the gap between a paper's claims and its
evidence. Writing well means making that gap zero — in both directions.
Overclaiming gets caught (by referees, by Week-45-style readers, eventually by
reality). Underclaiming buries your contribution. The skill is **calibration**,
and it lives at the sentence level. A ladder of the same result at five
strengths:

1. "Our method solves jet tagging." — unfalsifiable and false.
2. "Our method outperforms existing taggers." — overclaims: all taggers? all
   datasets? by how much?
3. "Our method outperforms ParticleNet on the top-tagging landscape dataset."
   — better; still hides the margin and the uncertainty.
4. "Our method improves AUC on the top-tagging landscape dataset from 0.9820
   to 0.9837 ± 0.0005 (5 seeds), at equal parameter count." — calibrated: the
   claim, the number, the variance, the matching.
5. "In one run, our method's AUC was slightly higher." — underclaims if you in
   fact have the seeds and the margin.

Rule of thumb: every empirical claim carries its scope (which dataset, which
metric, which conditions) and its uncertainty, either inline or by citing the
table that has them. Words like "significantly" are banned unless you mean a
statistical test and name it (Week 08).

Three sentence-level habits that do most of the work:

- **One idea per sentence.** If a sentence needs "and", "which", and two
  commas, split it. Short declarative sentences read as confidence; tangled
  ones read as hedging even when they are not.
- **Concrete subject, active verb.** "The model overfits after epoch 10"
  beats "overfitting behavior was observed to occur." The passive voice hides
  the actor, and in methods sections the actor (you? the optimizer? the
  authors of the baseline?) often matters.
- **Cut ruthlessly.** "It is worth noting that", "in order to", "very" — delete
  and nothing is lost. First drafts are for finding the idea; revision is for
  removing everything that is not the idea.

## 4. Figures that carry the argument

In an ML paper the figures *are* the evidence (Week 45, pass 2 — you read
figures first; so does everyone). Standards for your own figures:

1. **One message per figure.** Before making the plot, write the sentence the
   figure exists to prove — that sentence becomes the first sentence of the
   caption. If a figure needs two sentences, it is two figures.
2. **Honest axes.** Axis ranges chosen to show the effect at its true size —
   a y-axis zoomed to [0.980, 0.984] makes a 0.4% difference look enormous;
   sometimes that zoom is *right* (the differences at stake are small and the
   reader needs to see them) but then the axis labels and caption must make
   the scale unmissable. Log scales where the physics lives on decades (Week
   03). Never truncate a bar chart's baseline away from zero.
3. **Uncertainty shown.** Error bars, bands, or multiple-seed scatter on every
   quantity that has variance. A curve without a band claims the variance is
   negligible — only make that claim on purpose.
4. **Readable in grayscale, labeled directly.** Distinguish lines by style and
   marker, not color alone; label lines on the plot or in a legend that does
   not cover data; fonts legible at final print size (your Month-1 matplotlib
   standards, Week 02–03).
5. **The caption is self-contained.** Takeaway first, then what is plotted,
   then the conditions (dataset, model, seeds). A reader flipping through
   figures — which is how papers are read — gets the argument from captions
   alone.

The same standards apply to tables: bold the best number only if "best" is
outside the uncertainties; report the uncertainty column; state what ± means
(std over seeds? bootstrap? something else?) in the caption.

## 5. The capstone proposal

Now the genre gets used forward. A **proposal** answers, before any code is
written: what will be built, how success will be judged, what could go wrong,
and what it will cost. Research groups, funding agencies, and experiment
collaborations all run on these; a beam-time request at an accelerator is
exactly this document. Yours governs Weeks 47–48 and will be graded — by you,
in Week 48 — against its own success criteria. The template, section by
section (this is the required structure of `PROPOSAL.md`):

**1. Problem statement** (2–3 paragraphs). What you will build/show, for whom,
and why it matters — with the physics or systems context explained the way
this course's lessons explain things: from the ground up. If a reader from
outside your subfield cannot tell what the deliverable is, rewrite. Ends with
the deliverable in one sentence.

**2. Track and prior work.** Which capstone track (Week 47's `project.md`
covers choosing), and what you are building on: your own earlier capstones and
the 2–4 external works closest to this (Week 45 reading log entries earn their
keep here).

**3. Baselines.** Every comparison the final report will make, named now. A
system without a baseline cannot claim improvement over anything — at minimum:
the trivial/naive baseline, and your own strongest prior work where it applies
(the Capstone-2 VAE for a fast-sim track; zero-shot prompting for an agent
track; the paper's published numbers for reproduce-and-extend).

**4. Evaluation plan — metrics and thresholds, pre-committed.** For each claim
you hope to make: the metric, the dataset/split, and the *number that counts
as success*, fixed now. Pre-commitment (Week 32, Week 34) is what makes the
Week-48 report honest — deciding what "good" means after seeing the results
is how self-deception works, in ML and in physics alike. Include how variance
will be measured (seeds, bootstraps) and what the stress tests will be.

**5. Risk register.** The 3–5 most likely ways the project fails, each with:
probability (low/med/high), impact, a mitigation, and a trigger ("if X is not
working by Wednesday of Week 47, switch to Y"). Risks named in advance get
managed; risks discovered on Thursday get panicked about.

**6. Compute budget.** GPU-hours and dollars, estimated from *your own*
tracked runs (Weeks 41–42), with the arithmetic shown: (time per training
run) × (number of runs: seeds × configurations × the retries you always
forget) + evaluation + slack. A budget of "should be fine" is a risk-register
entry pretending to be a plan. Include where the compute comes from (local
GPU, Colab/Kaggle, cloud credit) and the fallback if it runs out.

**7. Milestones and cut lines.** Day-level milestones for Week 47 and Week 48
(the `project.md` specs give the skeleton), and for each stretch component a
**cut line**: the pre-agreed condition under which it gets dropped, and what
the reduced deliverable is. Cutting scope by plan is project management;
cutting it in a 2 a.m. panic is how repos end up half-broken.

Length: ≤4 pages. Every metric has a number. Every risk has a trigger. Every
milestone has a date.

### Red-teaming your own proposal

Before the proposal is final, attack it as a hostile referee (Week 45 made you
one): find the three most damaging objections — the unfair baseline, the
metric that can be gamed, the milestone that silently assumes a dataset you
have never actually loaded. Then either revise the proposal or add the
objection to the risk register with a real mitigation. The exercises make this
a required step.

## 6. Worked example — a mini-proposal, filled in

A deliberately small example (a weekend project, not a capstone) so the
template is visible end to end. Notice every number is committed up front.

> **Problem.** Week 45's reproduction found our from-text ParticleNet-baseline
> gap was dominated by unstated preprocessing. Deliverable: a measured answer
> to "how much does input normalization move a deep-sets jet tagger?" — an
> ablation grid over 3 normalization schemes, written as a 2-page note.
> **Prior work:** our Week-45 repro repo; the ParticleNet paper's appendix.
> **Baselines.** The paper's published baseline row (AUC 0.982 as stated);
> our Week-45 reproduction (AUC 0.971 ± 0.002 over 3 seeds).
> **Evaluation plan.** Metric: AUC on the top-tagging landscape test split,
> 3 seeds per scheme, reported as mean ± std. Success: at least one scheme
> closes ≥ half the 0.011 gap; a null result (no scheme moves AUC by > 0.002)
> is reportable as "normalization is not the culprit."
> **Risks.** (med) Training 9 runs exceeds the GPU budget → mitigation: train
> on a 50% subsample, verified in Week 45 to preserve ranking; trigger: >2 h
> per run. (low) Test-split mismatch vs paper → mitigation: re-verify split
> hashes first, before any training.
> **Compute.** 9 runs × 40 GPU-min (measured last week) ≈ 6 GPU-h + 1 h eval
> + 50% slack ≈ 10 GPU-h. Free tier covers it; fallback: subsample.
> **Milestones.** Sat noon: grid launched. Sat night: all runs logged. Sun
> noon: note drafted. Cut line: if <6 runs finish by Sun 9 a.m., drop scheme 3
> and report a 2-way comparison.

Note what pre-commitment bought: the null result has a *planned meaning*, so
the weekend cannot end in "it didn't work, nothing to report."

## Check yourself

1. What does the introduction's contributions list owe the experiments
   section, and what is wrong with "we investigate the effect of X" as a
   contribution?
2. Rewrite at calibration level 4 (§3's ladder): "Our fast-sim is much more
   accurate than the VAE."
3. A figure's caption reads "Results of the ablation study." Name two ways
   this caption fails the §4 standards.
4. Why must evaluation metrics and thresholds be fixed *before* running the
   system? Connect to benchmark contamination's cousin from Week 32.
5. What is a cut line, and why is it written into the proposal rather than
   decided during the build week?
6. Your compute budget says 40 GPU-hours and you have 60 available. A reviewer
   still flags the budget as risky. What is missing?
7. What is a reverse outline, and what class of writing problem does it find
   that sentence-level editing cannot?

## Answers

1. Every listed contribution must be supported by a specific experiment,
   table, or figure — the list is a set of promissory notes the experiments
   section redeems, ideally in the same order. "We investigate X" is an
   activity, not a claim: it cannot be false, so it cannot be supported, so
   the reader learns nothing from it.
2. Something like: "On held-out showers, our diffusion fast-sim improves
   energy-resolution agreement with the full simulator from 12% to 4%
   (mean over 5 seeds, std 1%), compared to our Capstone-2 VAE at equal
   sampling budget." Scope, metric, numbers, variance, matching.
3. It states no takeaway (the first sentence should be the message the figure
   proves), and it is not self-contained (no dataset, model, or seed/variance
   conditions — a reader flipping through figures learns nothing).
4. Choosing thresholds after seeing results lets you (unconsciously) pick the
   definition of success the results already meet — the evaluator's version
   of test-set peeking/leakage from Week 32: information flows from the
   evaluation into the criteria, and the criterion stops measuring anything.
5. A cut line is a pre-agreed condition under which a component is dropped,
   plus the reduced deliverable. It is written in advance because mid-sprint
   you are the worst-positioned person to make scope decisions: sunk cost,
   fatigue, and optimism all bias toward "one more day" — the proposal makes
   the decision while you are still impartial.
6. The arithmetic. A total without (time per run) × (runs, including seeds and
   retries) + eval + slack cannot be checked, and unshown arithmetic usually
   omits the retries and the evaluation passes. The 1.5× headroom is only
   meaningful if the 40 is itself credible.
7. One sentence per paragraph stating what the paragraph does; read the list
   alone. It finds structural problems — missing steps, repeated points,
   arguments in the wrong order — which are invisible when you polish
   sentence by sentence, because each sentence can be fine while the argument
   is broken.

## New terms

- **contributions list** — the introduction's explicit, falsifiable claims,
  each redeemed by a specific experiment.
- **reverse outline** — one sentence per paragraph of a draft, read as a list,
  to diagnose structure.
- **claim calibration** — writing each claim exactly as strong as its
  evidence: scope, numbers, uncertainty attached.
- **pre-committed metric** — a metric and success threshold fixed before the
  system is run, so results cannot redefine success.
- **proposal** — the paper skeleton pointed forward: problem, baselines,
  evaluation plan, risks, compute budget, milestones.
- **risk register** — enumerated failure modes with probability, impact,
  mitigation, and trigger.
- **compute budget** — GPU-hours/dollars estimated with shown arithmetic from
  measured run times, plus slack and a fallback.
- **cut line** — a pre-agreed condition for dropping scope, with the reduced
  deliverable named.
- **red-teaming** — attacking your own plan as a hostile referee before
  committing to it.

## Going deeper

- Simon Peyton Jones, *How to write a great research paper* (talk + slides,
  free) — this week's spine; the contributions-list discipline and
  "writing drives research" come from here.
- Gopen & Swan, *The Science of Scientific Writing* (American Scientist, free
  online) — why readers misread structurally sound-looking prose; the best
  short piece ever written on sentence-level information flow.
- Zinsser, *On Writing Well* — general nonfiction craft; the "cut ruthlessly"
  habit, book-length. Library/used copy; optional.
- Rougier, Droettboom & Bourne, *Ten Simple Rules for Better Figures* (PLoS
  Comput Biol, free) — a checklist that matches and extends §4.
- Re-read the best-written paper from your Week 45 pool purely for craft: how
  the intro lists contributions, how each figure pairs with a claim, how
  limitations are phrased. Reading as a writer is a different pass 2.
