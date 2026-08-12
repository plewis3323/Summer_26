# Week 31 — Alignment: Instruction Tuning, RLHF, DPO

How a next-token predictor becomes an assistant: supervised instruction tuning, then
optimization against a learned reward — and DPO's observation that the RL loop can be
integrated out exactly, a closed-form trick worth deriving in full.

## Objectives

- Draw the full pipeline: pretraining → SFT → preference data → reward model → RLHF
  (or DPO), and say what each stage changes about behavior.
- Explain the Bradley–Terry preference model and train a toy reward model.
- Sketch PPO for RLHF: the clipped objective, the KL-to-reference penalty, and why the
  reference model is there.
- Reproduce the DPO derivation from the RLHF objective, on paper, without notes.
- Discuss safety training and refusals as engineered behavior with failure modes
  (over-refusal, jailbreaks, sycophancy), not as magic.

## Core material (~3 hrs)

- Ouyang et al., *Training language models to follow instructions with human feedback*
  (InstructGPT, arXiv 2203.02155) — §§1–3, Fig. 2 pipeline.
- Rafailov et al., *Direct Preference Optimization* (arXiv 2305.18290) — §§1–4 and
  Appendix A; this is the source of this week's derivation.
- Lilian Weng's RLHF blog post (lilianweng.github.io) for the PPO-level view.
- Skim Bai et al., *Constitutional AI* (Anthropic) for how refusal behavior is trained
  without per-example human labels.

## Derivations (paper first)

- Bradley–Terry: P(y_w ≻ y_l) = σ(r(x, y_w) − r(x, y_l)); write the reward-model
  negative log-likelihood.
- RLHF objective: max_π E[r(x, y)] − β KL(π ‖ π_ref). Show the optimal policy is
  π*(y|x) = π_ref(y|x) e^{r(x,y)/β} / Z(x) (this is a Gibbs/Boltzmann distribution —
  note it).
- Invert for r: r(x, y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x). Substitute into
  Bradley–Terry, watch Z(x) cancel in the difference, and arrive at the DPO loss:
  −log σ(β[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]).
- From the DPO gradient, show it upweights y_w and downweights y_l with a weight that
  grows when the implicit reward ordering is wrong.

## Exercises (built when the week starts)

1. Toy reward model: Bradley–Terry on synthetic preferences with a known ground-truth
   reward. Accept when: learned reward's ranking correlates with truth (Spearman > 0.9).
2. DPO loss implemented from your derivation. Accept when: value and gradient match
   TRL's `DPOTrainer` loss on a fixed batch within 1e-5.
3. Gradient-direction check: on one (y_w, y_l) pair, verify a DPO step raises
   log π(y_w) and lowers log π(y_l). Accept when: both signs correct across 10 seeds.
4. PPO clip objective: plot L^CLIP vs probability ratio for both advantage signs.
   Accept when: plot reproduces the flat-clip regions and you annotate why each exists.
5. Tiny DPO run with TRL on a small pairwise dataset (e.g. verbose vs concise answers)
   on a ~1B model. Accept when: implicit-reward margin increases over training and
   before/after samples are saved.
6. Refusal probe: 10 clearly-harmful, 10 clearly-benign, 10 dual-use physics prompts
   (e.g. detector radiation calculations) against one open instruct model. Accept when:
   a 3×N table of behaviors with a one-paragraph note on over-refusal.

## Deliverable

The full DPO derivation scanned (this is a flagship derivation — filed for cold-redo
rotation); `Week31_Exercises.ipynb` with checks PASS; DPO before/after samples.

## Review

- Week 7: the optimal RLHF policy is π_ref · e^{r/β}/Z. Which distribution from
  statistical mechanics is this, and what plays the role of temperature?
- Week 22: the ELBO also contains a KL-to-a-reference term. Re-state the ELBO from
  memory and compare the role of its KL with RLHF's.
- Week 10: the reward-model loss is logistic regression on score differences. Write
  the logistic loss gradient from Week 10.
- Week 29: why is sampling at T > 0 essential for generating diverse preference-pair
  candidates in the first place?
