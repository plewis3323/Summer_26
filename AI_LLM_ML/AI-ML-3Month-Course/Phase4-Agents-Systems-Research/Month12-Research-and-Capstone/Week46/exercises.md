# Week 46 — Exercises

Work top to bottom. Setup (imports, data loading, constants) is given by the
notebook; you write only the lines each exercise asks for. This week is a
writing week: E1, E2, E4, E5, and E6 live in markdown files; only E3 (figure
surgery) uses the notebook, which provides the plotting scaffold per
NOTEBOOK_RULES §6. E5 is the week's deliverable — everything else feeds it.

## E1 — Reverse outline a published paper

Take the paper you critiqued in Week 45 and write its reverse outline: one
sentence per paragraph, in order, stating what that paragraph does (not what it
says). Read the sentence list alone and mark where the argument jumps, repeats,
or stalls.
Hint: if you cannot write the one sentence, that is itself a finding — mark the
paragraph as doing nothing identifiable.
Accept when: `reverse_outline.md` covers every paragraph of the paper's intro,
method, and experiments sections and identifies ≥2 structural weaknesses or
strengths with paragraph references.

## E2 — Rewrite drill

Find the worst paragraph in your Capstone-3 writeup (Week 36). Produce three
revisions: (a) shorter — same content, ≤60% of the words; (b) plainer — one
idea per sentence, concrete subjects, active verbs; (c) claim-calibrated —
every empirical claim carries scope and uncertainty per lesson §3's ladder.
Hint: do (a) first; most of what (b) and (c) need to fix is easier to see once
the filler is gone.
Accept when: `rewrite.md` shows the original plus all three versions
side-by-side, each with a one-line rationale for what changed.

## E3 — Figure surgery

Pick one figure from an earlier capstone that carries a real result. In the
notebook, remake it to lesson §4 standards: one message (stated as the
caption's first sentence), honest axes, uncertainty shown, readable without
color. Render before and after side by side.
Hint: write the caption's first sentence *before* touching the plot code — it
decides what stays on the figure.
Accept when: the notebook shows before/after, the new figure displays
uncertainty (band, bars, or seed scatter), and the caption states the takeaway
in its first sentence.

## E4 — Track selection

For each of the five capstone tracks in `03-Project-Roadmap.md` (copilot,
generative validation, paper-to-pipeline, reproduce-and-extend, evaluated LLM
service), write a five-line feasibility sketch: data in hand? compute within
budget? riskiest step? what it signals (AI Engineer vs AI Scientist)?
what you would reuse from earlier capstones? Then choose one track, write the
role sentence from `05-Two-Year-Path.md`, and document the runner-up with the
tiebreaker reason.
Hint: the honest tiebreakers are "which feeds my thesis or job target" and
"which riskiest step do I already know how to de-risk" — not "which sounds
most impressive."
Accept when: `tracks.md` has five sketches with all five questions answered,
one chosen track, a role sentence, and a named runner-up with a one-sentence
tiebreaker.

## E5 — The proposal

Write `PROPOSAL.md` for the chosen track, following lesson §5's template
exactly: role sentence, problem statement, track and prior work, baselines, evaluation plan
with pre-committed metrics and thresholds, risk register (≥3 risks with
probability, impact, mitigation, trigger), compute budget with shown
arithmetic, and day-level milestones for Weeks 47–48 with cut lines.
Hint: write the evaluation plan and budget first — they constrain everything
else, and writing the problem statement last keeps it honest about scope.
Accept when: `PROPOSAL.md` is ≤4 pages, the role sentence is unhedged, every metric has a numeric success
threshold committed before any code runs, every risk has a trigger, and every
milestone has a date and a cut line where applicable.

## E6 — Red-team your proposal

Attack E5 as a hostile referee: write the three most damaging objections (an
unfair baseline, a gameable metric, a hidden assumption, an untested data
dependency...). For each, either revise the proposal or absorb the objection
into the risk register with a real mitigation and trigger.
Hint: run Week 45's critique checklist against your own evaluation plan — it
was built for exactly this.
Accept when: `redteam.md` lists three concrete attacks and, for each, points
to the proposal revision or risk-register entry that answers it; `PROPOSAL.md`
is updated accordingly and marked final.

## Review

1. (Week 12, 24, 36) For each prior capstone, quote its headline claim from
   memory, then check the writeup: did the evidence support the claim at
   lesson §3's calibration level 4?
2. (Week 08) A result is "2.1σ above baseline." What distributional
   assumptions hide inside that sentence?
3. (Week 34) What pre-committed metrics did your RAG evaluation use, and why
   does committing before running matter more for LLM systems than for a
   chi-square fit?
4. (Week 03) Name three properties of a publication-quality matplotlib figure
   you standardized in Month 1, and which lesson §4 standard each maps to.
5. (Week 40) What were the columns of your agent evaluation table, and which
   would a referee attack first?
