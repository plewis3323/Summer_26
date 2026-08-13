# Week 31 — Exercises

Work top to bottom, derivation first: E2 and E3 implement the DPO loss you derived on
paper — file the derivation scan before you start them. Setup (imports, a small
instruct model plus a frozen reference copy of it, TRL, LoRA config, seeds) is given
by the notebook; you write only the lines each exercise asks for. Everything this week
is notebook cells; E5 and E6 save their outputs (before/after samples, the probe
table) under `results/`. E5 is the one GPU-hungry cell — if no local GPU, run it on
Colab/Kaggle.

## E1 — Toy reward model: Bradley–Terry with known truth

Build a toy response space (~200 random feature vectors) with a fixed ground-truth
linear reward r*. Generate ~500 preference pairs by drawing two responses and labeling
the winner with probability σ(r*(y_a) − r*(y_b)). Fit a small reward model on the
Bradley–Terry negative log-likelihood (lesson §2) and compare its ranking of all
responses against the truth.
Hint: sample the labels from the Bradley–Terry probability — don't assign them
deterministically. Human preferences are noisy; your synthetic ones should be too.
Accept when: the learned reward's ranking correlates with truth (Spearman > 0.9).

## E2 — The DPO loss, from your derivation

Write `dpo_loss(logp_w, logp_l, logp_w_ref, logp_l_ref, beta)` implementing the boxed
loss of lesson §5, fed by the `response_logprob` pattern of §8. Run one fixed batch of
preference pairs through the LoRA-wrapped model and its frozen reference, and compare
your loss value and gradient against TRL's.
Hint: you don't need a full training run to compare — TRL's `DPOTrainer` exposes its
loss computation (`dpo_loss`) on exactly these four log-prob tensors. For the gradient,
call `backward()` on each loss from the same starting point and compare the adapter
gradients.
Accept when: value and gradient match TRL's `DPOTrainer` loss on a fixed batch within
1e-5.

## E3 — Gradient-direction check

On one (y_w, y_l) pair, take a single optimizer step of your DPO loss and verify that
log π(y_w) went up and log π(y_l) went down. Repeat with 10 different seeds
(re-initialize the LoRA adapter each time).
Hint: measure the two log-probs before and after with gradients off; use a small LR —
you are testing the gradient's direction, not surviving an overshoot.
Accept when: both signs correct across 10 seeds.

## E4 — The PPO clip objective, plotted

Plot L^CLIP as a function of the probability ratio ρ ∈ [0, 2] at ε = 0.2, one curve
for advantage A = +1 and one for A = −1. Annotate every flat region with why it is
flat.
Hint: the objective is min(ρA, clip(ρ, 1−ε, 1+ε)A) — two lines of NumPy; the
annotations are the actual deliverable. Ask where the incentive to keep pushing ρ
disappears, and in which direction clipping never binds.
Accept when: the plot reproduces the flat-clip regions and you annotate why each
exists.

## E5 — Tiny DPO run with TRL

Build a small pairwise dataset (~100 pairs) where the concise answer wins: for
prompts like "summarize this abstract in one sentence" over your Week 27 corpus,
sample two continuations at T = 1 and label the shorter-but-complete one chosen, the
verbose one rejected (or use any small public preference set from the Hub). DPO-train
the ~1B model with LoRA via `DPOTrainer`; log the implicit-reward margin; generate
from 5 fixed prompts before and after and save both sets side by side.
Hint: the margin β[(log π − log π_ref)_w − (log π − log π_ref)_l] is DPOTrainer's
`rewards/margins` metric — lesson §6 says it should climb. A few hundred steps at
β ≈ 0.1 is plenty; this is a direction check, not a production run.
Accept when: the implicit-reward margin increases over training and before/after
samples are saved.

## E6 — Refusal probe

Write 30 prompts: 10 clearly harmful (things a model should refuse), 10 clearly
benign, and 10 dual-use physics (detector radiation-dose estimates, shielding
thicknesses, activation calculations — legitimate Tuesday work that pattern-matches
to harm). Run all 30 against one open instruct model, classify each response by hand
as refuse / comply / partial-hedge, and tabulate.
Hint: keep the harmful prompts generic enough that you'd be comfortable with the
refusals succeeding; the interesting cells are the dual-use row. Quote the worst
physics over-refusal verbatim in your note.
Accept when: a 3×N table of behaviors with a one-paragraph note on over-refusal.

## Review

1. (Week 08) The optimal RLHF policy is π_ref · e^{r/β}/Z(x). Which distribution from
   statistical mechanics is this, and what plays the role of temperature?
2. (Week 22) The ELBO also contains a KL-to-a-reference term. Re-state the ELBO from
   memory and compare the role of its KL with RLHF's.
3. (Week 10) The reward-model loss is logistic regression on score differences. Write
   the logistic loss gradient from Week 10 and point to where its "wrongness-weighted"
   shape reappears in the DPO gradient.
4. (Week 29) Why is sampling at T > 0 essential for generating diverse preference-pair
   candidates in the first place?
