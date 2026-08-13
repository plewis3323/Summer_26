# Week 32 — Evaluation + Mini-Project: Abstract Metadata Extractor

~10 hrs, most of it the mini-project (see `project.md`). Before starting you should
be able to: LoRA-fine-tune with `peft`/TRL and design an SFT dataset with loss
masking and dedup discipline (Week 30); apply chat templates and sample at controlled
temperature (Week 29); explain what preference optimization changes and why every
alignment objective is a proxy (Week 31); define precision, recall, and F1 from a
confusion matrix (Week 10); state the train/val/test and leakage discipline (Week 09).

You have spent three weeks operating the modern LLM stack: loading models, tuning
them, aligning them. The closing question of the month is the one every experiment
ends with: *how do you know it worked?* For LLMs the honest answer is uncomfortable —
the standard measurement instruments are themselves miscalibrated in known ways, and
the errors they introduce are often larger than the differences being measured. This
lesson teaches the three big instruments and their failure modes — benchmarks (and
contamination), LLM-as-judge (and its biases), and hallucination measurement — and
then narrows to the metric family your mini-project actually needs: per-field
precision/recall/F1 for structured extraction. Everything in this lesson is
apparatus for the project: a LoRA fine-tune of a 1–3B model that reads nuclear
physics abstracts and emits structured metadata, scored honestly against a zero-shot
baseline.

## 1. Evaluation as a measurement problem

Physics distinguishes two kinds of uncertainty, and the distinction is the right
frame for everything this week:

- **Statistical uncertainty** is the scatter from having finitely many samples. It
  shrinks as $1/\sqrt{N}$: measure more events (here: more test items) and it goes
  away.
- **Systematic uncertainty** is bias built into the apparatus — a miscalibrated
  energy scale, a trigger that silently prefers one event class. No amount of extra
  data shrinks it; you have to find it, measure it, and correct or report it.

LLM evaluation is **systematics-dominated**. With a few hundred test items the
statistical error bar on an accuracy is a couple of percent — small. The systematics
are not:

- **Contamination** (§2): the test items were in the training data, so the score
  measures memory, not capability.
- **Judge bias** (§3): the grader has preferences (position, verbosity, itself) that
  move scores independently of quality.
- **Metric choice** (§5): a strict all-or-nothing metric and a partial-credit metric
  can order two models *differently* on the same outputs.
- **Prompt sensitivity**: rephrasing the task prompt, reordering few-shot examples,
  or changing the output format can move benchmark scores by more than the gap
  between the models being compared. If you didn't hold the prompt fixed across
  systems, you measured the prompts.

The week's README says it straight: treat every benchmark number the way you treat
an uncalibrated detector. The professional posture is not cynicism — it is listing
your systematics next to your number, exactly as a physics measurement does. Your
project's writeup will do literally that.

## 2. Benchmarks and their contamination

### 2.1 What a benchmark is

A **benchmark** is a fixed dataset of tasks plus a fixed scoring rule, so that
different models can be compared on the same footing. The names you will see
constantly, each defined in one line:

- **MMLU** — multiple-choice questions across 57 academic subjects; score = accuracy.
- **GSM8K** — grade-school math word problems; score = exact match on the final
  number.
- **HumanEval** — programming problems; score = fraction whose generated code passes
  hidden unit tests.
- **HellaSwag** — pick the most plausible continuation of an everyday scene.

Even "score = accuracy" hides apparatus. Multiple-choice can be scored by *generating*
a letter or by comparing the model's *log-likelihood* (Week 08) of each option and
taking the argmax — the two disagree, and harness details (prompt format, number of
few-shot examples, whether options are lettered) move MMLU scores by whole percentage
points. Two leaderboard numbers are comparable only if the same harness produced
both. When you read "model X: 71.3 on MMLU", the honest parse is "71.3 under one
specific apparatus, ±harness."

### 2.2 Contamination: leakage at internet scale

**Contamination** is the presence of evaluation data in the training corpus. You met
this failure class in Week 09 as **leakage** — information from the test set reaching
training — and you spent Capstone 1 auditing for it. The LLM version is worse in
three ways: the training corpus is a scrape of much of the internet, and benchmarks
*live on the internet* (GitHub repos, papers that quote items, blog posts that
discuss answers); the corpus is too large to inspect by eye; and for closed models
you cannot inspect it at all.

The consequence: a benchmark delta between two models may measure which one's crawl
happened to include the test set, not which one is more capable. Scores on a popular
benchmark tend to drift upward across model generations partly for this reason — the
benchmark ages into the training distribution.

### 2.3 Detecting it

You cannot prove absence of contamination, but you can hunt for its fingerprints.
Three method families, from most to least access:

1. **N-gram overlap** (needs the training corpus). An **n-gram** is any run of $n$
   consecutive tokens. Slide over each test item and search the training corpus for
   long shared runs (13-grams is a common choice); flag items with matches. This is
   the method training-data curators use to *decontaminate* before training.
2. **Completion probing** (black-box — works on any model you can sample from).
   Prompt the model with the verbatim first half of a test item, temperature 0, and
   measure how well the continuation reproduces the second half (exact or
   near-exact overlap). An uncontaminated model completes *plausibly*; a
   contaminated one completes *verbatim* — it has no other way to know the exact
   wording. Comparing reproduction rates against paraphrased versions of the same
   items sharpens the test.
3. **Memorization probes**: unusually low perplexity (Week 29) on test items
   relative to matched fresh text, or the model reproducing dataset-specific
   formatting, IDs, or ordering it could only know from the file itself.

What to do about it, in order of strength: evaluate on data created *after* the
model's training cutoff; evaluate on private data the crawl cannot have seen; or —
the project's route — build your own labeled evaluation set. Note the precision of
that last claim: your test *abstracts* are public arXiv text and very likely are in
the base model's pretraining corpus (your project's contamination check measures
exactly this), but your *labels* — the JSON you hand-wrote in Weeks 30–32 — have
never existed anywhere. The extraction task cannot be contaminated even when its
inputs are, though a memorized abstract could still make extraction artificially easy
for those items. Measure, then caveat: that is the whole method.

Inside your own project the discipline is Week 09's, restated for LLMs: dedup before
splitting, split by document (never by row), and let nothing from the test split —
including near-duplicates — touch training or prompt engineering.

## 3. LLM-as-judge

### 3.1 The idea

Many tasks have no programmatic scoring rule: "is this summary faithful?", "which
answer is more helpful?". Human grading is the gold standard and does not scale.
**LLM-as-judge** uses a strong LLM as the grader: show it the task and the response
(pointwise: grade 1–10), or two responses (pairwise: pick A or B), optionally with a
reference answer to compare against, and take its verdict as the score.

Zheng et al. (2023) made this respectable by *measuring* it: on MT-Bench (a suite of
multi-turn questions), GPT-4's verdicts agreed with human majority preference about
as often as individual humans agreed with each other (~80%+). That is the honest
pitch: a judge can be *as good as one noisy human rater*, at 1000× the throughput.

### 3.2 The biases

The same paper documented systematic errors — memorize these three:

- **Position bias.** In pairwise comparison, judges favor one position (usually the
  first answer). Mitigation: run each comparison twice with the order swapped, and
  count only consistent verdicts (call the rest ties).
- **Verbosity bias.** Longer, more elaborate answers score higher at equal
  correctness. This is Week 31's label-bias story wearing judge's robes: human
  raters reward length and confidence, models trained on human preferences inherit
  it, and a judge built from such a model re-applies it to everything it grades.
- **Self-preference (self-enhancement) bias.** A judge tends to favor outputs whose
  style resembles its own — including, when grading a pool that contains its own
  generations, literally preferring itself. Never let a model be scored by itself or
  a close sibling if you can avoid it.

Add a capability ceiling: a judge cannot reliably grade what it cannot do (subtle
math, niche physics), and it can be confidently wrong — a hallucinated verdict about
a hallucination.

### 3.3 Using a judge with eyes open

The rule that turns a judge from vibes into an instrument: **calibrate it against
your own labels before you trust it**. Take a sample of items, label them yourself,
run the judge, and report the agreement rate — exactly the way a detector gets
calibrated against a known source before you point it at the unknown. If judge–you
agreement is 92%, the judge is a usable proxy with a stated error; if it is 70%, it
is a coin with opinions. And Goodhart's law (Week 31) still applies: a judge is a
learned proxy, and anything optimized *against* it (prompt tweaks, style changes)
will find its gaps.

For your project the judge is deliberately the *secondary* instrument: structured
extraction has a programmatic scoring rule (§5), which is strictly better where it
applies. The project's judge exercise runs the judge anyway on a sample and measures
its agreement with the schema-based scores — you get a calibrated feel for judge
error on a task where, for once, you know the truth.

## 4. Measuring hallucination

### 4.1 Definitions first

**Hallucination** is fluent output not supported by the relevant ground truth. Two
distinct flavors — conflating them causes most confusion in the literature:

- **Faithfulness** errors: the output contradicts, or is unsupported by, a *provided
  source* (the document being summarized; the abstract being extracted from).
- **Factuality** errors: the output is wrong about the *world*, source or no source.

For open-ended generation, measuring either is hard: you need to decompose the
output into atomic claims and check each against a source or a knowledge base —
expensive, subjective at the edges, and itself often done with an LLM judge (with
§3's caveats compounding).

This is why the project's task is chosen the way it is. For **structured extraction**
hallucination becomes crisply *faithfulness-flavored and measurable*:

> An extracted value is **hallucinated** if it is neither present in nor entailed by
> the source abstract.

**Entailed** means: derivable from what the text says without new information.
An abstract that says "gold–gold collisions at 200 GeV" entails
`collision_system: "Au+Au"` and `sqrt_s_nn_gev: 200` — the words differ, the content
doesn't. Your normalization rules (§5.3) operationalize entailment; whatever they
don't cover, you adjudicate by hand and write down.

### 4.2 The metric, and what it is not

$$\text{hallucination rate (per field)} =
\frac{\#\{\text{predicted non-null values not grounded in the source}\}}
     {\#\{\text{predicted non-null values}\}}.$$

Grounded = present or entailed. Note what this metric deliberately excludes: a value
that *is* in the abstract but is the answer to the wrong question — e.g. the model
extracts 62.4 GeV as the collision energy because the abstract compares 200 GeV
results *with* 62.4 GeV results — is **wrong but grounded**. It is an error (F1
catches it); it is not a fabrication. Meanwhile `sqrt_s_nn_gev: 5020` on an abstract
that never mentions any LHC energy is a fabrication — the model imported a plausible
number from its pretraining prior instead of reading. The split matters because the
two errors have different fixes: wrong-but-grounded errors call for better task
understanding (more/better SFT examples); fabrications call for better *abstention*.

**Abstention** is the model's ability to output "not there" — in your schema, the
JSON value `null`. Week 30's dataset rule ("missing information is null — never
guessed") was anti-hallucination training before you had the word for it: every
theory abstract labeled `experiment: null` is a demonstration that the correct
response to absent information is silence, pushing against the pretraining prior
that every field slot deserves a plausible filler. Expect the fine-tune to beat
zero-shot prompting *most* dramatically on hallucination rate, and check whether it
does.

Report hallucination per system and per field. The worst field tells you where the
model is pattern-completing instead of reading.

## 5. Scoring structured extraction: per-field precision, recall, F1

### 5.1 The setup

Each test item is (abstract, gold JSON); each system produces a predicted JSON. The
zeroth metric is the **JSON validity rate**: the fraction of outputs that parse as
JSON and validate against the schema at all. An invalid output scores zero on every
field — and validity alone often separates fine-tuned from zero-shot models. Report
it first.

Then score field by field. For one field on one item, with gold value $g$ and
predicted value $p$ (either may be null):

| case | counts as |
|---|---|
| $g \ne$ null, $p \ne$ null, match($p, g$) | **TP** (true positive) |
| $p \ne$ null, and ($g =$ null or no match) | **FP** (false positive) |
| $g \ne$ null, and ($p =$ null or no match) | **FN** (false negative) |
| $g =$ null and $p =$ null | correct rejection (counted in neither) |

A non-null prediction that misses a non-null gold counts as *both* FP and FN — the
model asserted something wrong *and* failed to produce the right value. Then, per
field, exactly Week 10's definitions:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad
\text{recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}
           {\text{precision} + \text{recall}}.$$

Precision reads "when the model asserts this field, how often is it right"; recall
reads "of the values that were there to find, how many did it find"; F1 is their
harmonic mean, punishing imbalance.

### 5.2 Why not accuracy, and why not exact match

**Accuracy fails on null-heavy fields.** In a corpus with many theory abstracts,
`experiment` is null for a third of items and `centrality` for more. A model that
*always* outputs null scores high accuracy on those fields while finding nothing —
the rare-event problem from Week 10's trigger story, returned. The TP/FP/FN scheme
above excludes null–null agreements from both numerator and denominator, so the
score measures finding things, not agreeing about absence.

**Whole-JSON exact match fails by discontinuity.** Scoring an item 1 only if *every*
field matches makes the metric a product of per-field successes: a model that
improves from 60% to 80% per field moves from $0.6^6 \approx 5\%$ to
$0.8^6 \approx 26\%$ on six-field exact match — smooth per-field progress dressed up
as a sudden leap. This is precisely the emergence-mirage argument from Week 28:
harsh discontinuous metrics can manufacture apparent phase transitions out of steady
underlying improvement. Report per-field, aggregate transparently.

Two aggregation conventions, name yours: **micro-averaging** pools TP/FP/FN across
fields before computing one F1 (weights fields by how often they occur);
**macro-averaging** computes per-field F1 and takes the unweighted mean (every field
counts equally, rare ones included). Your headline number is macro-F1; the per-field
table is the real result.

### 5.3 match(): the metric inside the metric

What counts as a match is itself apparatus, and it must be *fixed before you look at
test results* and *identical for every system*. Concretely:

- **Canonicalization.** "Au+Au", "Au–Au", "gold–gold", "AuAu" → `Au+Au`. "0.2 TeV"
  → `200` (GeV). Case-fold experiment names. Build one normalization function; both
  the gold labels and predictions pass through it before comparison.
- **List-valued fields** (`observables`): compare as *sets* after canonicalization —
  per-item TP = predicted ∩ gold, FP = predicted − gold, FN = gold − predicted —
  then pool into the field's counts. Order must not matter.
- **Free-text fields** (`physics_topic`): the fuzziest field; use canonicalized
  exact match against a small controlled vocabulary you define when labeling, accept
  the lower F1, and say so. Do not quietly loosen the matcher until the number looks
  better — that is tuning the apparatus on the result, the classic self-inflicted
  systematic.

A loose matcher inflates every system's score; a tight one deflates them. Either is
fine *if it is the same for all systems and stated in the writeup* — relative
comparisons survive calibration offsets; sneaky ones don't.

## 6. The mini-project, framed

Everything this month built a component; the project spends them all:

| week | contribution |
|---|---|
| 27 | the heavy-ion abstracts corpus (the raw material) |
| 29 | model loading, chat templates, deterministic generation for eval |
| 30 | LoRA/`peft`/TRL recipe; `sft_v0.jsonl` (30 seed labels); dataset discipline |
| 31 | what tuning does to behavior; why abstention must be trained |
| 32 | the measurement apparatus of §§2–5 |

The experiment, in one paragraph: grow the labeled dataset to ~200 (abstract → JSON)
examples; split 60/20/20 by document with dedup; run the *baselines first* —
zero-shot and few-shot prompting of the base instruct model, through the same
scoring harness; LoRA-fine-tune on the train split; then the head-to-head on the
held-out test split: per-field and aggregate F1, JSON validity, hallucination rate,
for all three systems in one table. A judge cross-check (§3.3) and a contamination
probe (§2.3) close the loop. The acceptance gate, from the roadmap: **beat zero-shot
prompting on the held-out set, reported as per-field F1.**

Baselines-first is a course habit worth naming as method: the zero-shot number is
the "does this need fine-tuning at all?" control, and it must be measured before you
tune anything — including your prompts — on test data. Prompt engineering for the
baseline happens on the *validation* split only; the test split is opened once, at
the end, for every system simultaneously. Full spec, data details, and build order:
`project.md`.

## 7. Worked example: the scoring harness, end to end

The core of your eval script, on three toy items — small enough to check by hand,
real enough to reuse. Gold and predicted pairs for two fields:

```python
def canon(field, value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if field == "collision_system":
        v = v.replace("--", "+").replace("-", "+").replace(" ", "")
        v = v.replace("gold+gold", "au+au")
    if field == "sqrt_s_nn_gev":
        v = v.replace("gev", "").strip()
        if "tev" in v:
            v = str(float(v.replace("tev", "").strip()) * 1000)
        v = str(float(v))
    return v

def score_field(field, golds, preds):
    tp, fp, fn = 0, 0, 0
    for g, p in zip(golds, preds):
        g, p = canon(field, g), canon(field, p)
        if p is not None and g is not None and p == g:
            tp += 1
        else:
            if p is not None:
                fp += 1
            if g is not None:
                fn += 1
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

golds_sys = ["Au+Au", "Pb+Pb", None]          # item 3: a theory abstract, no system
preds_sys = ["gold-gold", "p+Pb", "Au+Au"]    # 1 right, 1 wrong, 1 hallucinated
golds_e   = [200, 5020, None]
preds_e   = ["0.2 TeV", None, None]           # 1 right (entailed form), 1 missed

print(score_field("collision_system", golds_sys, preds_sys))
print(score_field("sqrt_s_nn_gev", golds_e, preds_e))
```

Walk the arithmetic before running it. `collision_system`: item 1 is a TP
(canonicalization maps "gold-gold" → "au+au"); item 2 is wrong → FP *and* FN; item 3
asserts a system where gold is null → FP. So TP=1, FP=2, FN=1: precision 1/3, recall
1/2, F1 = 0.4. `sqrt_s_nn_gev`: item 1 TP ("0.2 TeV" → 200.0 — entailment via
normalization); item 2 predicted null on a non-null gold → FN; item 3 null–null →
excluded. TP=1, FP=0, FN=1: precision 1.0, recall 0.5, F1 ≈ 0.667. The code should
agree; if it doesn't, the harness — not the model — is what you debug. Item 3's
`collision_system` is also one for the hallucination count: a non-null assertion with
nothing in the source to ground it.

That habit — hand-check the metric on a case where you know every answer before
pointing it at 40 test items — is the difference between an eval script and a random
number generator with a table formatter.

## Check yourself

1. Statistical vs systematic uncertainty: define both, and name three systematics of
   LLM evaluation.
2. Your test abstracts are on arXiv and almost certainly in the base model's
   pretraining data. Why can your *extraction benchmark* still be uncontaminated,
   and what residual caveat survives?
3. Describe completion probing in two sentences. Why does it work black-box?
4. Name the three judge biases and one mitigation for position bias.
5. Give an example (from this task) of a wrong-but-grounded extraction and of a
   hallucinated one. Which metric catches each?
6. A field is null in 40% of gold items. Why does the TP/FP/FN scheme not reward a
   model that always outputs null, while accuracy would?
7. Per-field scores improve smoothly from 60% to 80%, yet six-field exact match
   jumps from ~5% to ~26%. Which Week 28 argument is this, and what should you
   report instead?
8. Why must the `match()` normalization be frozen before you look at test results?

## Answers

1. Statistical: scatter from finite samples, shrinks as $1/\sqrt{N}$. Systematic:
   bias in the apparatus, unaffected by more data. LLM-eval systematics include
   contamination, judge biases (position/verbosity/self-preference), metric choice,
   and prompt sensitivity.
2. The labels are yours and never existed publicly, so (abstract → JSON) as a *task*
   cannot have been memorized. Caveat: a verbatim-memorized abstract may be easier
   to extract from than a genuinely unseen text, so measure memorization (completion
   probing) and report it alongside.
3. Prompt the model with the verbatim first half of a test item at temperature 0 and
   measure how exactly the continuation reproduces the true second half; verbatim
   reproduction (especially vs a paraphrase control) indicates the text was in
   training. It needs only the ability to sample the model — no access to weights or
   corpus.
4. Position bias (favoring one answer slot — mitigate by judging both orders and
   keeping only consistent verdicts), verbosity bias (longer scores higher at equal
   quality), self-preference (favoring its own style/outputs).
5. Wrong-but-grounded: extracting 62.4 GeV as the collision energy because the
   abstract mentions it as a comparison energy — caught by F1 (it is an FP+FN), not
   by the hallucination rate (it is grounded). Hallucinated: emitting
   `experiment: "STAR"` on a theory abstract that names no experiment — caught by
   both F1 and the hallucination rate.
6. Null–null agreements are counted in neither numerator nor denominator, so
   always-null yields TP=0 and recall 0 on that field; accuracy would credit every
   null–null item, rewarding blanket abstention.
7. The emergence-mirage argument: a harsh discontinuous metric (exact match ≈
   product of per-field successes) manufactures an apparent jump from smooth
   underlying progress. Report per-field F1 with a transparent (macro) aggregate.
8. Loosening or tightening the matcher after seeing results is tuning the apparatus
   on the answer — a self-inflicted systematic. Fixed before test-time and identical
   across systems, its calibration offset cancels in the comparison.

## New terms

- **statistical / systematic uncertainty** — scatter that shrinks with $N$ vs bias
  in the apparatus that doesn't.
- **benchmark** — fixed task set + fixed scoring rule for comparing models.
- **evaluation harness** — the concrete prompting/scoring apparatus producing a
  benchmark number.
- **contamination** — evaluation data present in the training corpus; leakage at
  internet scale.
- **n-gram overlap** — contamination detection by shared long token runs with the
  training corpus.
- **completion probing** — black-box contamination test: verbatim-continue a test
  item's first half.
- **LLM-as-judge** — using a strong LLM as grader (pointwise, pairwise, or
  reference-guided).
- **position / verbosity / self-preference bias** — the judge's systematic errors.
- **judge calibration** — measuring judge agreement against your own labels before
  trusting it.
- **hallucination** — fluent output not supported by the relevant ground truth.
- **faithfulness vs factuality** — unsupported by the provided source vs wrong about
  the world.
- **grounded / entailed** — present in the source / derivable from it without new
  information.
- **abstention** — outputting null when the information is absent; trained behavior.
- **JSON validity rate** — fraction of outputs that parse and validate against the
  schema.
- **per-field F1** — precision/recall/F1 computed field by field over (gold,
  predicted) values.
- **canonicalization** — normalizing surface forms before matching; part of the
  metric.
- **micro / macro averaging** — pooling counts across fields vs averaging per-field
  scores.
- **exact match** — all-or-nothing whole-output scoring; discontinuous, mirage-prone.

## Going deeper

- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (arXiv 2306.05685), §§1–3 — the judge-bias measurements behind §3.
- One contamination study read carefully — e.g. Golchin & Surdeanu, *Time Travel in
  LLMs* (arXiv 2308.08493) for guided/completion probing, or any recent "data
  contamination in language models" survey; note the detection method it uses.
- HF `evaluate` docs (skim) — prebuilt metrics and their exact definitions; useful
  for cross-checking your harness, not replacing it.
- TRL `SFTTrainer` docs — refresher for the project's training step (Week 30's
  recipe unchanged).
