# Week 48 — Final Capstone: Validate, Write, Present

~3 hrs reading + the validation week (`project.md`). Before starting you should
have: the Week-47 repo at its gate state (skeleton-to-real pipeline complete,
all baseline rows filled, one full evaluation recorded, `BUILDLOG.md` current),
your final `PROPOSAL.md`, Week 45's critique checklist, and Week 46's writing
standards. This is the course's last lesson. It teaches the part of research
that happens after the system works: proving it, writing it, saying it out
loud, and stopping honestly.

## 1. A result does not exist until it survives review

In a physics collaboration, an analysis is not a result when the plot first
looks right. It becomes a result after the **analysis note** — the internal
document recording every cut, every systematic check, every dead end — has
survived **internal review**: colleagues whose job is to break it before the
outside world sees it. Nobody enjoys review week. But the culture exists
because the alternative — believing your own first plot — is how fields fill
up with results that evaporate on contact.

You are self-studying, so there is no committee. This week you are both
analyst and review committee, and the split is enforced by *artifacts*: the
evaluation plan was pre-committed in Week 46 (analyst-you cannot move the
goalposts), the stress suite and ablation are required (reviewer-you gets to
attack), and the report, demo, and talk are recorded (the evidence outlives
your memory of it). Week 45 trained you to measure the gap between a paper's
claims and its evidence. This week the paper is yours, and the job is to make
that gap zero before anyone else measures it.

One rule governs everything below: **the system is frozen.** From the moment
the final evaluation starts, no tuning, no "quick improvements," no prompt
tweaks. Genuine bugs — the code does not do what the report will say it
does — may be fixed, but each fix is logged in `BUILDLOG.md` and the entire
evaluation reruns from scratch afterward. Why so strict? Because an
evaluate–tweak–evaluate loop is optimization against the test set (Week 32's
leakage, in slow motion): every peek-then-adjust cycle transfers information
from the evaluation into the system, and the final number stops measuring
generalization and starts measuring persistence. Physics has the same
discipline under the name **blinding** — analyses fix their cuts before
looking at the signal region, precisely because smart, honest people fool
themselves otherwise. You blinded yourself in Week 46, when you pre-committed
the metrics. Stay blind.

## 2. The final evaluation: freeze, seed, one command

The first deliverable is the complete evaluation plan from `PROPOSAL.md`, run
against the frozen system, every number reported. Mechanics:

**Run everything through one harness.** System, baselines, and (later) the
ablation all go through the same evaluation code path — the one Week 47's
milestone 3 built. Different harnesses for different rows is how subtle
metric mismatches (Week 45 §4: the top cause of false comparisons) sneak into
your own table.

**Measure your own variance.** Wherever the system is stochastic — training
seeds, generative sampling, LLM decoding, agent trajectories — run it more
than once and report mean ± spread. This is Week 45's checklist item 4 turned
on yourself: a single-seed gain is noise until shown otherwise, *including
yours*. Practical notes per source of randomness:

- Training stochasticity: retrain with ≥3 seeds if the compute budget allows;
  if it does not, say so in the report and bound what you can (e.g. seed the
  cheap components, report the one expensive run as n=1).
- Sampling/decoding stochasticity: cheap to measure — repeat the evaluation
  pass several times on the same weights. For LLM-backed tracks, repeat the
  runs even at temperature 0: served models are not perfectly deterministic,
  and agent trajectories compound small differences. The trial-to-trial
  spread *is* a property of your system; report it.
- Interpretation: a margin over baseline smaller than your own spread is not
  a margin. Write it that way.

**One command regenerates the table.** The results table in the report is
generated from `results/` by a script, not assembled by hand — hand-copied
numbers drift, and the Week-48 gate requires that a fresh clone can rebuild
the table. If a number in the report cannot be regenerated, it does not go in
the report.

**Grade against the proposal.** For each pre-committed threshold there are
exactly three outcomes: **met**; **missed, with a diagnosis** (which
component, which assumption, which risk-register entry fired); or **missed,
undiagnosed**. Only the third is a failure of this week. A missed threshold
with a written diagnosis is a normal research outcome — Week 46's proposal
was designed so that even the null result has a planned meaning. The table
reports all three kinds with equal prominence.

## 3. Stress tests: beyond the happy path

The evaluation plan measures the system on the distribution it was built
for. A **stress test** probes it where that distribution does not: inputs at
the edge of or outside the training/design range, inputs crafted to be
difficult, and sensitivity of the result to choices you guessed rather than
measured. Physics calls the third kind a **systematic-uncertainty study**:
vary everything you do not fully trust, and see whether the result moves. The
question a stress suite answers is not "does it work?" (§2 answered that) but
"where does it stop working, and how does it fail when it does?" — and for a
system you intend to demo to employers or a collaboration, the second
question is the one they will ask.

Design the suite from four categories (the track specifics are in
`project.md`; ≥5 tests total):

1. **Out-of-distribution inputs.** Conditions the system never saw:
   energies/angles at and beyond the training range for a fast-sim; a paper
   from a different subfield for a paper-to-pipeline agent; an analysis task
   type absent from the scripted set for a copilot.
2. **Adversarial and ambiguous inputs.** Inputs designed to elicit the worst
   behavior: questions whose correct answer is "I can't do that with these
   tools"; questions that tempt fabrication (Week 32's hallucination
   measurements, now aimed at your own agent); malformed or underspecified
   requests.
3. **Sensitivity to guessed choices.** Every unmeasured decision in the
   scope ledger and (for track d) the gap list: perturb it and re-run the
   headline metric. If a guess you cannot justify moves the result more than
   your margin over baseline, the report must say so.
4. **Failure behavior.** Kill a dependency mid-run — a tool times out, a file
   is missing, the retriever returns nothing. Does the system fail loudly and
   honestly, or fabricate, hang, or half-complete? (Week 44's health-check
   thinking, applied above the service layer.)

Each test gets a line in `stress.md`: what was done, what happened, verdict.
**Report the failures.** A stress suite that reports only passes was not
stressing — reviewer-you should treat an all-pass suite the way Week 45
taught you to treat a paper with no limitations section: as a finding about
the authors, not the system. The failures found this week are not
embarrassments; they are the limitations section of the report, discovered on
your schedule instead of the audience's.

## 4. The ablation: your own medicine

Week 45's checklist item 5 demanded ablations of other people's papers: which
part of the system actually causes the improvement? Now you owe one. Remove
or degrade one core component of your system, re-run the headline metric
through the same harness, and put the row in the results table:

- (a) Copilot: retrieval off (agent answers from the model alone), or the
  tool set reduced to the trivial subset.
- (b) Fast-sim: conditioning removed or the model shrunk — is it the
  architecture or the parameter count? (The exact question your Capstone-2a
  gate asked.)
- (c) Paper-to-pipeline: the structured reading stage replaced by a single
  naive pass — does the mechanized Week-45 workflow earn its complexity?
- (d) Reproduce-and-extend: the extension's twist removed — which is, by
  construction, your reproduction baseline; here the ablation is about the
  twist's *cost* (compute, complexity) against its measured gain.

One sentence in the report states what the ablation shows the component
contributes. Two honest possible endings: "removing X costs Y on the headline
metric — X is load-bearing," or "removing X changes nothing measurable — the
improvement lives elsewhere." The second is a real finding and goes in the
report at full volume. You spent Week 45 learning to distrust papers that
bundle ten changes and credit their favorite; do not write one.

## 5. The report, written to be believed

The report is 6–10 pages, and it is Week 46's paper skeleton with the
results filled in — re-read that lesson's §2–§4 now; this section adds only
what is specific to *this* document.

**The structure is the proposal's.** Abstract with the headline numbers;
intro whose contributions list is the proposal's promised claims, each marked
kept or not; method with the architecture diagram; results against every
baseline the proposal named; limitations; future work. The symmetry is the
point: Week 46's document said "here is what would convince me"; this one
says "here is what happened." A reader with both documents can check you.
Where the build deviated from the plan, the scope ledger entries become one
honest paragraph — plans that survived contact unchanged are rare and
reviewers know it.

**Every claim cites a table or figure; every table and figure supports a
claim.** Orphans in either direction are bugs (Week 46 §2). Claims sit at
calibration level 4 — scope, number, variance, matching — and "significantly"
still means a named statistical test or nothing (Week 08).

**The architecture diagram** is one page: boxes are components that actually
exist in the repo, arrows are data that actually flows, each box labeled with
where it lives in the code. Draw it from the code, not from the aspiration —
the reader will use it to navigate the repo, and a diagram showing the system
you meant to build is worse than none. (This is the gate's diagram; the talk
reuses it.)

**The limitations section is written to be believed.** At least three real
weaknesses — the ones a referee would find anyway, which is exactly the test:
run Week 45's checklist and your own `stress.md` against the report, and
whatever they catch goes in. A believable limitation is specific and
consequential ("all evaluation is on simulation; the simulation-to-data gap
is unmeasured"), not false modesty ("may not generalize to all settings").
Stating a weakness precisely, with its expected impact and what closing it
would take, reads as command of the work. Vagueness reads as concealment —
and after 45 weeks, you can hear the difference.

**"What failed first" is a required subsection.** The syllabus has demanded
it since Capstone 1; the raw material is in `BUILDLOG.md`, written the day it
happened. Negative results go in the writeup. This is not a LinkedIn post.

## 6. The demo and the talk

Two recordings, two different burdens of proof.

**The demo (≤10 minutes)** proves the reproducibility bar *on camera*: fresh
clone, `uv sync`, one command, the headline capability, narrated. The cold
environment is the entire content of the proof — a demo from a warm
environment proves "it works on my machine," which is the claim nobody
doubts and nobody cares about. Record the terminal from the `git clone`
onward; narrate what the viewer is seeing and what they should notice; when
the headline output appears, connect it to the report's headline number. If
the clean-clone run fails on camera, that is not a wasted recording — it is
the reproducibility bar failing, which you need to know today. Fix it, log
it, re-record.

**The talk (15–20 minutes, slides, recorded)** proves you can carry the
argument to people. The imagined audience is a mixed HEP/ML lab seminar —
which means, in this course's house style, each half needs the other half's
terms defined in one line as they appear: the physicists get "ablation" and
"AUC," the ML people get "calorimeter" and "systematic." Structure is the
paper's argument compressed: the problem, why it is hard, the idea, the
evidence, the limitations, what is next.

Craft rules that do most of the work:

- **One point per slide, and the title is the point.** "Retrieval doubles
  task completion on ambiguous questions" is a slide title; "Results" is not.
  A reader flipping the deck gets the argument from titles alone — the same
  standard Week 46 set for figure captions, because it is the same skill.
- **The figures are the report's figures.** They were built to carry the
  argument (Week 46 §4); the caption's first sentence becomes the sentence
  you say when the slide appears.
- **End with limitations, then "what I'd do next."** Both slides are required
  by the gate, and ending there is not humility theater — it is how careful
  scientists signal that they know where the edge of their own evidence is.
  The strongest question period is the one you started yourself.
- **Rehearse once out loud, then record the real one.** Out loud is
  non-negotiable: prose that reads fine is routinely unsayable, and timing
  only exists out loud. Before recording, write down the two hardest
  questions the audience could ask (Week 46's red-team, now aimed at the
  podium) and know your answers.

Why record, with no audience to show it to? Three reasons. The recording is
the gate's evidence. Watching yourself is the only talk feedback available in
self-study — one viewing finds the filler words and the slide you rushed.
And the recording is a portfolio artifact: "here is me presenting my work to
a technical audience" is evidence a resume line cannot fake.

## 7. Closing the course honestly

The course ends the way each phase did: with a gate, checked in writing, by
you.

**The End-Phase-4 gate** (`00-Syllabus.md` §5): final capstone demoed
end-to-end; architecture diagram; honest limitations; one mock research talk
delivered and recorded. Plus the course-wide bar that has defined "done"
since Week 04: fresh clone + `uv sync` + one command → same result within
stochastic noise. `GATE.md` is the checklist made concrete — every item
checked with a *link to the evidence* (the results table, the diagram, the
recordings, the report section), so that checking a box requires the artifact
to exist. An unlinked check is an unchecked check.

**One flagship derivation, cold.** The monthly rotation (PCA → backprop →
ELBO → attention → DDPM loss → policy gradient) lands on one of them; do it
on paper, closed book, and scan it into the week folder. This is the course's
closing argument about what you now own: not familiarity — the ability to
rebuild the result from nothing. If it does not come out cold, that is the
retro's first line, not a reason to skip the scan.

**The sign-off** (`00-Syllabus.md` §9): tag `month-12-complete`, write the
250-word `retro.md` — for this month, honestly assess the whole course: what
the gates actually caught, where you cut corners, which weeks you would
redo — and open the final issue: the question this course leaves you wanting
to answer. That issue is deliberate. A course can end; the habit of always
having one named open question is the part that should not.

**What comes next.** The course ends with artifacts; the three-year goal
needs a *record* (`03-Project-Roadmap.md`, "After the course"). The sketch:
year 2 — publish one or two ML-for-physics results (thesis-adjacent),
contribute to an open ML tool, apply the skills visibly inside sPHENIX;
year 3 — target role conversion (lab AI hire, or industry research/
engineering) carrying the portfolio this course built: four capstone repos, a
fine-tuned model, an MCP server, a publication or workshop paper, and a
thesis that uses ML for real. The single strongest immediate move, if you aim
at research roles, is spending the post-course slack month polishing the
capstone — track (d) especially — into a workshop submission: it converts a
course artifact into a community artifact, which is the currency that counts.
And keep two habits at maintenance dose: the reading log (papers keep
coming), and the weekly rhythm of building something measured. The course
installed the system; the record comes from running it.

## Check yourself

1. Why must the system be frozen before the final evaluation begins, and
   what earlier concept (Week 32) does an evaluate–tweak–evaluate loop
   reenact?
2. Your system beats the strongest baseline by 1.2% on the headline metric;
   your five evaluation runs on the frozen system span ±0.9%. What does the
   report say?
3. Your stress suite comes back 8 passes, 0 failures. What should
   reviewer-you conclude, and what does Week 45 say about the analogous
   pattern in papers?
4. What question does the ablation row answer that the rest of the results
   table cannot, and what are the two honest possible endings?
5. Why must the demo recording start from `git clone` in a cold environment?
6. Why does the talk end with a limitations slide — give the strategic
   reason, not the virtue reason.
7. A pre-committed threshold was missed, and the report says so with a
   diagnosis pointing at a named risk-register entry. Did the capstone fail
   the week? What *would* fail it?

## Answers

1. Because every peek-then-adjust cycle moves information from the evaluation
   into the system: the final number stops measuring generalization and
   starts measuring how long you kept tweaking. It is test-set
   leakage/contamination (Week 32) in slow motion — the same reason physics
   blinds analyses before opening the signal region.
2. That the margin is smaller than the system's own trial-to-trial spread,
   so the comparison is unresolved at current statistics: report
   mean ± spread for both rows, state that the difference is within the
   measured variance, and (if it matters) what additional runs would resolve
   it. Claiming the 1.2% as a gain is the single-seed sin from Week 45's
   checklist, self-inflicted.
3. That the suite was too gentle — the tests were drawn from the happy path's
   neighborhood rather than the system's edges. Week 45's analogue: a paper
   with no limitations section (or only false-modesty limitations) — the
   absence is a finding about the authors, not evidence of strength. Add
   harder tests until something breaks; the breaks are the limitations
   section.
4. Attribution: whether the core component *causes* the headline performance,
   which the full-system rows cannot show because they bundle every choice.
   Honest endings: "removing X costs Y — load-bearing," or "removing X
   changes nothing measurable — the improvement lives elsewhere." Both are
   reportable; only silence is not.
5. Because the demo's job is to prove the reproducibility bar — fresh clone +
   `uv sync` + one command → the result — and a warm environment proves only
   "works on my machine," a claim that carries no information. The cold start
   *is* the evidence; a failure on camera is the bar failing, which is worth
   knowing today.
6. Because it seizes control of the question period: the audience's hardest
   questions get asked and answered by you, on your framing, with your
   evidence — and it signals you know where the edge of your evidence is,
   which makes every claim before it more credible. (The virtue reason is
   also true; it is just not why it works.)
7. No — a missed threshold with a written diagnosis is a planned outcome of a
   pre-committed evaluation; Week 46 designed the proposal so even null
   results have meaning. What fails the week: a missed threshold with no
   diagnosis, a threshold quietly redefined after the results existed, or a
   number in the report that the one-command harness cannot regenerate.

## New terms

- **analysis note / internal review** — physics' discipline of documenting
  every check and dead end, then having colleagues attack the result before
  the world sees it; this week's model.
- **freeze** — the point after which the system may not be tuned; bug fixes
  are logged and force a full re-evaluation.
- **blinding** — fixing the analysis (cuts, metrics, thresholds) before
  looking at the answer, so the answer cannot shape the analysis.
- **stress test** — a probe of the system off the evaluation plan's
  distribution: out-of-distribution, adversarial, sensitivity, or
  failure-behavior.
- **systematic-uncertainty study** — physics' name for sensitivity stress
  tests: vary what you do not trust; see if the result moves.
- **ablation (owed)** — the component-removal experiment you demanded of
  papers in Week 45, now performed on your own system.
- **assertion-evidence slide** — a slide whose title states the point and
  whose body is the evidence; the deck's titles alone carry the argument.
- **gate check / `GATE.md`** — the end-of-phase checklist with every item
  linked to its evidence artifact; an unlinked check is an unchecked check.

## Going deeper

- Simon Peyton Jones, *How to give a great research talk* (talk + notes,
  free) — the companion to Week 46's writing spine; the one-point-per-slide
  and "rehearse out loud" disciplines come from here.
- The CaloChallenge summary paper (Week 45 pool) — for track (b), the
  community's own validation-metric definitions; your `stress.md` should
  speak its language.
- Re-read your own `PROPOSAL.md` and Week 46 §5 side by side with the
  finished report — the proposal-vs-report symmetry is this week's actual
  spine text.
- Pineau et al., *Improving Reproducibility in Machine Learning Research*
  (Week 45's Going deeper) — reread the checklist now that you are the
  authors; it is this week's gate from the field's point of view.
- `03-Project-Roadmap.md`, "After the course" — the months 13–36 sketch;
  read it once more before writing the retro, and pick the slack month's
  project deliberately.
