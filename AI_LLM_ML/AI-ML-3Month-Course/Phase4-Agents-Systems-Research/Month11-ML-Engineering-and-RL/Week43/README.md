# Week 43 — Reinforcement Learning

RL is control theory for systems you can only sample — the same framing labs use for automated accelerator tuning and detector control, and the machinery underneath Week 31's RLHF.

## Objectives

- Formalize a problem as an MDP: states, actions, transition kernel, reward, discount; define return, value function V, and action-value Q.
- Derive the policy gradient theorem on paper and explain the log-derivative trick and why a baseline reduces variance without adding bias.
- Implement REINFORCE with a learned baseline, extend it to an advantage actor–critic, and train both to solve a classic control environment.
- Write the PPO clipped surrogate objective, explain what the clip does to the incentive landscape, and state how PPO-for-RLHF differs (reward model + KL-to-reference penalty).
- Diagnose RL training pathologies: reward curves that collapse, high-variance gradients, entropy collapse.

## Core material (~3 hrs)

- Sutton & Barto, *Reinforcement Learning: An Introduction* (free online): Ch. 3 (finite MDPs) and Ch. 13 (policy gradient methods); skim Ch. 6 intro for TD intuition.
- OpenAI Spinning Up: "Part 1: Key Concepts in RL", "Part 3: Intro to Policy Optimization", and the PPO documentation page.
- Schulman et al., *Proximal Policy Optimization Algorithms* — read for the objective (Eq. with the clip), not the benchmarks.
- Physics hook (skim one, by title): a paper on RL for accelerator control (e.g. RL-based tuning at a light source or CERN beam control) — identify its state, action, and reward choices.

## Derivations (paper first)

- The policy gradient theorem: from J(θ) = E[R(τ)] to ∇J = E[Σ ∇log π(a|s) · G_t], via the log-derivative trick and the trajectory-probability factorization.
- Show E[∇log π(a|s) · b(s)] = 0 for a state-dependent baseline; define the advantage A = Q − V.
- Write the PPO clipped objective and show, case by case in the sign of A, which probability-ratio movements it stops rewarding.
- Photograph all three into `week43/derivations/`.

## Exercises (built when the week starts)

1. MDP formalization: cast "keep the beam centered under drift" as an MDP in markdown (states, actions, reward, discount) and defend each choice. Accept when: the writeup names one deliberate simplification and its consequence.
2. Random-baseline: run CartPole (gymnasium) with a random policy for 100 episodes. Accept when: mean and max episode return are printed as the baseline row.
3. REINFORCE: implement it with a small policy net in PyTorch. Accept when: 10-seed-averaged return reaches the environment's solve threshold (CartPole: ≥475 mean over 100 eval episodes) at least once in 3 training runs.
4. Baseline ablation: add a learned value baseline; compare gradient-estimate variance and learning curves against exercise 3. Accept when: plot shows visibly faster/steadier learning and a sentence links it to the derivation.
5. Actor–critic: bootstrap with TD advantage instead of Monte-Carlo returns. Accept when: solves the environment in fewer episodes than exercise 4 on the same seeds, or the writeup explains why not.
6. PPO objective unit: implement the clipped loss as a pure function; test it on hand-built (ratio, advantage) cases. Accept when: `pytest` checks confirm zero gradient outside the clip region in the correct cases.
7. RLHF mapping: one-page table mapping (policy, action, reward, baseline, trust region) from CartPole PPO onto Week-31's RLHF pipeline. Accept when: every row is filled, including what plays the role of "environment" for an LLM.

## Deliverable

`week43/` — derivation scans, training scripts with tracked runs (Week-41 infrastructure), learning-curve plots, and the RLHF mapping page.

## Review

1. Week 8: REINFORCE is SGD on what objective? Why is the gradient estimate unbiased but high-variance, and what did minibatching do for variance in supervised learning?
2. Week 31: derive (or sketch) how DPO eliminates the explicit reward model from the RLHF objective.
3. Week 7: the log-derivative trick uses ∇log p = ∇p / p. Where else did the log-likelihood's gradient structure appear in this course?
4. Week 22: the reparameterization trick and the score-function (REINFORCE) estimator solve the same problem. State the problem and when each applies.
5. Week 19: why did long-range credit assignment fail in RNNs, and what is the RL analog of that problem?
