# Week 43 — Exercises

Work top to bottom, derivations first: the policy gradient theorem, the
baseline lemma, and the PPO clip case analysis go on paper before any code,
and the scans go in `week43/derivations/` (README "Derivations" section).

Where the work lives (per NOTEBOOK_RULES §6): E2 is notebook cells. E1 and E7
are markdown files the notebook only checks exist. E3–E5 are training scripts
in `week43/` — the deliverable is "training scripts with tracked runs," so
they are files the notebook launches and whose logged results it checks. E6
is a module plus a pytest file. Instrument every training run with your
Week-41 MLflow setup (params, per-episode return, entropy); untracked RL runs
are unreadable later. `uv add gymnasium torch mlflow` before starting. Seeds
matter more this week than any other — the acceptance criteria say how many.

## E1 — MDP formalization

Cast "keep the beam centered under drift" as an MDP in `week43/MDP.md`:
states, actions, reward, discount, one paragraph defending each choice
(lesson §10 has the hardware vocabulary — BPMs, correctors).
Hint: the Markov property is the interesting design pressure — a drifting
system's instantaneous position is not a Markov state; say what you add.
Accept when: the writeup names one deliberate simplification and its
consequence.

## E2 — Random baseline

Run CartPole (`gymnasium.make("CartPole-v1")`) with a uniformly random
policy for 100 episodes; record every episode return.
Hint: an episode return in CartPole is just its length — +1 per surviving
step; loop `env.step(env.action_space.sample())` until terminated or
truncated.
Accept when: mean and max episode return are printed as the baseline row.

## E3 — REINFORCE

Implement REINFORCE in `week43/reinforce.py`: small policy net (4 → 64 → 2),
sampled actions via `torch.distributions.Categorical`, reward-to-go weights,
the surrogate loss from lesson §6, MLflow tracking of per-episode return.
Evaluate by freezing the policy and running 100 greedy episodes.
Hint: `loss = -(logps * returns.detach()).sum()`; normalize the returns
batch to zero mean, unit variance before the multiply — it is a free
variance reducer and usually the difference between solving and stalling.
Accept when: 10-seed-averaged return reaches the environment's solve
threshold (CartPole: ≥475 mean over 100 eval episodes) at least once in 3
training runs.

## E4 — Baseline ablation

Add a learned state-value baseline to `week43/reinforce.py` behind a
`--baseline` flag: a second small net trained by regression on the observed
returns, its prediction subtracted from G_t in the policy loss. On the same
seeds as E3, compare (a) the empirical variance of the per-batch gradient
estimate and (b) seed-averaged learning curves, with and without.
Hint: measure gradient variance directly — compute the loss gradient on
several independent episode batches at a fixed policy checkpoint and take
the variance across batches of one layer's flattened gradient.
Accept when: plot shows visibly faster/steadier learning and a sentence
links it to the derivation.

## E5 — Actor–critic

Write `week43/actor_critic.py`: replace Monte-Carlo reward-to-go with the
one-step TD advantage δ = r + γV̂(s′) − V̂(s) (lesson §7), updating actor
and critic from short interaction batches instead of full episodes.
Hint: detach δ in the actor loss; the critic's target r + γV̂(s′) is also
detached — only V̂(s) carries gradient in the critic loss. Zero the γV̂(s′)
term on terminal steps.
Accept when: solves the environment in fewer episodes than exercise 4 on
the same seeds, or the writeup explains why not.

## E6 — PPO objective unit

Implement the clipped surrogate as a pure function in `week43/ppo_loss.py`
— `ppo_loss(ratio, advantage, eps)` returning the per-element objective —
and test it in `tests/test_ppo_loss.py` on hand-built (ratio, advantage)
cases from your paper case analysis: inside the clip both signs, outside
the clip both directions for both signs, and the pessimistic-min cases.
Hint: for the zero-gradient checks, make `ratio` a tensor with
`requires_grad=True`, backward the summed loss, and assert the gradient
entry is exactly 0 where your case analysis says the clip is active.
Accept when: `pytest` checks confirm zero gradient outside the clip region
in the correct cases.

## E7 — RLHF mapping

Write `week43/RLHF_MAP.md`: a one-page table mapping (policy, action,
reward, baseline, trust region) from your CartPole PPO onto Week-31's RLHF
pipeline, with one sentence per row on what changes in the LLM setting.
Hint: lesson §9 is the frame, but the table must be yours — in particular,
be precise about *when* reward arrives in each column and what that does to
credit assignment.
Accept when: every row is filled, including what plays the role of
"environment" for an LLM.

## Review

1. Week 08: REINFORCE is SGD on what objective? Why is the gradient
   estimate unbiased but high-variance, and what did minibatching do for
   variance in supervised learning?
2. Week 31: derive (or sketch) how DPO eliminates the explicit reward model
   from the RLHF objective.
3. Week 08: the log-derivative trick uses ∇log p = ∇p / p. Where else did
   the log-likelihood's gradient structure appear in this course?
4. Week 22: the reparameterization trick and the score-function (REINFORCE)
   estimator solve the same problem. State the problem and when each
   applies.
5. Week 19: why did long-range credit assignment fail in RNNs, and what is
   the RL analog of that problem?
