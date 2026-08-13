# Week 43 — Reinforcement Learning

~5 hrs reading + the paper derivation. Before starting you should be able to:
apply the chain rule and explain why the derivative of log f is f′/f (Week 05),
work with expectations, conditional expectations, and maximum likelihood
(Week 08), write a PyTorch training loop with a sampled loss (Week 15), state
the reparameterization trick and what problem it solves (Week 22), and sketch
the RLHF pipeline with its reward model and KL-to-reference penalty (Week 31).
This week contains the course's last flagship derivation — the policy gradient
theorem — and it goes on paper first, per the README.

## 1. A third kind of learning problem

Everything you have trained so far learned from a dataset that sat still.
Supervised learning (Week 09): inputs with right answers attached. Unsupervised
(Week 12): inputs, find structure. In both, your model's outputs never changed
what data came next.

**Reinforcement learning (RL)** drops that assumption. An **agent** chooses
**actions**; an **environment** responds with a new situation and a single
number, the **reward**; and the agent's goal is to choose actions that make
the total reward large. Nobody ever shows it the right action — it only ever
learns "that worked out well" or "that didn't," and usually long after the
action that mattered. Worse, its own choices determine what it gets to see
next: a policy that never explores the left half of the room never learns
what's there. RL is control theory for systems you can only sample.

Why a physicist should care, concretely: a particle accelerator is a machine
that uses electric fields to speed charged particles up and magnets to steer
and focus the resulting beam. Hundreds of magnet currents are knobs; the
machine drifts with temperature and ground motion; and "is the beam good" is a
measured number, not a differentiable function of the knobs. Operators retune
constantly, by experience. That is exactly an RL problem — state you can
measure, actions you can take, a reward you can only observe — and labs
(CERN, light sources) now run RL agents on real machines (§10). It is also,
via RLHF, the machinery that turned next-token predictors into assistants:
Week 31 sketched PPO in one paragraph; this week you build it from the ground
up, so the sketch becomes something you could implement.

## 2. The MDP: the formal object

The standard formalization is the **Markov decision process (MDP)**. Time
runs in discrete steps. At step t:

1. The agent observes the **state** s_t ∈ S — everything it gets to know
   about the situation (cart position and velocity; beam-monitor readings;
   the prompt plus tokens generated so far).
2. It picks an **action** a_t ∈ A (push left/right; nudge a magnet current;
   emit one token).
3. The environment draws the next state from the **transition kernel**
   P(s_{t+1} | s_t, a_t) — a conditional probability distribution, possibly
   deterministic, generally unknown to us — and hands back a scalar
   **reward** r_{t+1}.

The **Markov property** gives the object its name: the next state and reward
depend only on the current state and action, not on the longer history.
That's a modeling *choice*, not a law — you make it true by putting enough
into the state. (A beam position alone isn't Markov if the beam is drifting;
position plus recent trend is closer.)

An **episode** is one run from a start state to a terminal state (pole falls,
game ends, completion finishes); its record
τ = (s₀, a₀, r₁, s₁, a₁, r₂, …, s_T) is a **trajectory**.

The agent's goal is not the next reward but the long run. Define the
**return** from step t as the discounted sum of everything that comes after:

$$ G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{} \gamma^{k}\, r_{t+k+1} $$

where γ ∈ [0, 1] is the **discount factor**. It earns its place three times
over: it makes the infinite sum converge for never-ending tasks (a geometric
series, provided rewards are bounded and γ < 1); it encodes that sooner
reward is worth more than later; and it acts as a soft horizon — rewards
beyond roughly 1/(1−γ) steps barely count, so γ = 0.99 means "care about the
next few hundred steps." For short episodes you can set γ = 1 and the return
is just the sum.

## 3. Policies and value functions

The agent's behavior is a **policy** π(a | s): a probability distribution
over actions given the state. We use *stochastic* policies deliberately:
randomness is how the agent explores actions it wouldn't currently pick, and
— crucially for this week — a smooth parameterized distribution π_θ(a | s)
(say, a neural net producing softmax probabilities, Week 10) is something we
can differentiate. "Learning" will mean gradient ascent on θ.

Two functions summarize how good things are under a policy:

- The **state-value function** V^π(s) = E[G_t | s_t = s] — the expected
  return from state s if you follow π thereafter.
- The **action-value function** Q^π(s, a) = E[G_t | s_t = s, a_t = a] — the
  same, but with the first action pinned to a.

Their difference, A^π(s, a) = Q^π(s, a) − V^π(s), is the **advantage**: how
much better than π's average it is to take action a in state s. Positive
advantage = better than what you'd typically do; negative = worse. Hold onto
it — it becomes the central quantity of §5–§8.

One identity to file now (it powers §7): the return telescopes,
G_t = r_{t+1} + γ G_{t+1}, so taking expectations gives the **Bellman
consistency** equation V^π(s) = E[r_{t+1} + γ V^π(s_{t+1}) | s_t = s]. A
value function that violates it is wrong — and the violation is trainable
signal.

Finally the objective. With a parameterized policy π_θ, define

$$ J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[ R(\tau) \right], \qquad R(\tau) = \sum_{t=0}^{T-1} \gamma^t\, r_{t+1} $$

the expected return of a whole episode, where p_θ is the distribution over
trajectories induced by running π_θ in the environment. We want ∇_θ J and
gradient *ascent*.

## 4. Why you can't just backprop

In every loss so far, autograd could follow a differentiable path from the
parameters to the number you minimized. Here that path is cut twice:

1. **The environment is a black box.** The reward comes out of a simulator, a
   physical machine, or a learned reward model applied after generation. You
   can't differentiate through a pole falling over or a magnet power supply.
   You only get samples: run the policy, observe what happened.
2. **The distribution depends on the parameters.** J(θ) is an expectation
   *over trajectories whose probability depends on θ*. Changing θ changes
   which data you see. There is no fixed dataset to sum a loss over.

You met exactly this structure in Week 22. There, the gradient of an
expectation E_{z∼q_θ}[f(z)] had two known escapes: the **reparameterization
trick** — rewrite the sample as a deterministic, differentiable function of θ
and noise (z = μ + σε) — which needs f differentiable and the distribution
reparameterizable; and the **score-function estimator**, which needs neither.
The environment being a non-differentiable black box rules out
reparameterization. So RL's fundamental tool is the score-function route, and
deriving it carefully is the point of the next section.

## 5. The policy gradient theorem, line by line

This is the flagship. Do it on paper alongside the text; every step below
states its justification. Setup: episodic MDP with horizon T, trajectory
τ = (s₀, a₀, s₁, a₁, …, s_T), start-state distribution ρ(s₀). To keep the
algebra clean we write R(τ) without discount weights inside the derivation
(set γ = 1; nothing below changes if you carry γ through — see the honesty
note at the end).

**Step 0 — the probability of a trajectory.** By the chain rule of
probability (Week 08: any joint factorizes into a product of conditionals,
here in time order) plus the Markov property (each conditional needs only the
latest state/action) plus the fact that the policy looks only at the current
state:

$$ p_\theta(\tau) = \rho(s_0)\, \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\; P(s_{t+1} \mid s_t, a_t) $$

Note which factors carry θ: only the policy's. The start distribution and
the dynamics P are the environment's business.

**Step 1 — differentiate the objective.**

$$ \nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau)\, R(\tau)\, d\tau = \int \nabla_\theta\, p_\theta(\tau)\, R(\tau)\, d\tau $$

Justification: J is by definition the integral (a finite sum, for finite
MDPs) of p_θ(τ) R(τ) over all trajectories; R(τ) doesn't depend on θ (the
rewards are the environment's); and gradient and sum swap — exactly true for
finite sums, true under mild regularity conditions for integrals (the same
swap you used deriving MLE gradients in Week 08).

The problem: the right-hand side is *not an expectation* — ∇p_θ is not a
probability distribution (its entries can be negative; they sum to
∇∫p = ∇1 = 0). An expectation we could estimate by sampling trajectories
and averaging. This, we can't. The entire trick is to make it one again.

**Step 2 — the log-derivative trick.** From Week 05: the derivative of
log f(θ) is f′(θ)/f(θ) (chain rule with (log x)′ = 1/x). Read it backwards:

$$ \nabla_\theta\, p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau) $$

valid wherever p_θ(τ) > 0. This is the same object as the **score function**
from maximum likelihood (Week 08) — the gradient of a log-probability —
which is no accident, as we'll see.

**Step 3 — substitute, and recover an expectation.**

$$ \nabla_\theta J = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau = \mathbb{E}_{\tau \sim p_\theta}\!\left[ \nabla_\theta \log p_\theta(\tau)\, R(\tau) \right] $$

Justification: substitution of Step 2 into Step 1; then the definition of
expectation (an integral of "probability × quantity" *is* E[quantity]). This
is the payoff: run the current policy, collect trajectories, average
∇log p_θ(τ) R(τ) over them — that sample average is an **unbiased** estimate
of the true gradient (its expectation is exactly ∇J, by construction).

**Step 4 — the environment drops out.** Take the log of Step 0's product
(log of a product is a sum of logs, Week 05):

$$ \log p_\theta(\tau) = \log \rho(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t) $$

Apply ∇_θ: the first and third groups contain no θ, so their gradients are
zero. Hence

$$ \nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) $$

This is the quiet miracle of the theorem: **the unknown dynamics P appear
nowhere in the gradient.** You never need a model of the environment — only
the ability to *act in it* and differentiate your own policy. That is what
"model-free RL" means.

**Step 5 — first form of the theorem.** Combining Steps 3 and 4:

$$ \nabla_\theta J = \mathbb{E}_{\tau}\!\left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right] $$

Readable already: each action's log-probability gets pushed up in proportion
to the *whole episode's* return. Good episode → every action in it becomes
more likely; bad episode → less. It works, but crudely: an action is credited
with rewards that happened *before it was taken*. Fixing that needs a lemma.

**The zero-mean score lemma.** For any state s:

$$ \mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s) \right] = \sum_a \pi_\theta(a \mid s)\, \frac{\nabla_\theta\, \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} = \sum_a \nabla_\theta\, \pi_\theta(a \mid s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta\, 1 = 0 $$

Justifications, one per equals sign: definition of expectation; log-derivative
trick (Step 2, in reverse); the π's cancel; gradient and finite sum swap;
probabilities over all actions sum to 1 for every θ; the derivative of a
constant is zero. (Same lemma as "the MLE score has zero mean" in Week 08.)

**Corollary (baselines are free).** Multiply the score by *any* quantity b
that does not depend on the action a — a constant, or a function b(s) of the
state: E_a[∇log π_θ(a|s) · b(s)] = b(s) · E_a[∇log π_θ(a|s)] = b(s) · 0 = 0.
The factor pulls out of the expectation over a precisely because it doesn't
vary with a. We now use this twice.

**Step 6 — causality: only the future counts.** Split R(τ) inside Step 5's
expectation, term by term. The piece multiplying ∇log π_θ(a_t | s_t) is
Σ_{t′} r_{t′+1}, which splits into rewards *before* t and rewards *from* t
on. For a reward r_{t′+1} with t′ < t: condition the expectation on the
trajectory prefix (s₀, a₀, …, s_t) — legitimate by the tower property of
expectation, E[X] = E[E[X | prefix]] (Week 08). Given the prefix, that past
reward is a fixed number, and the inner expectation over a_t ∼ π_θ(·|s_t) of
"score × fixed number" is zero by the corollary. So every past-reward term
vanishes *in expectation*, and dropping them from the estimator changes its
mean not at all — while removing genuinely random terms, which can only
shrink its variance. What survives multiplying each score is the
**reward-to-go** — the return G_t from that step:

$$ \nabla_\theta J = \mathbb{E}_{\tau}\!\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t \right] $$

An action is now judged only by what came after it. This is credit
assignment, done by algebra.

**Step 7 — subtract a baseline.** Apply the corollary again, this time
deliberately: for any state-dependent **baseline** b(s_t),
E[∇log π_θ(a_t|s_t) b(s_t)] = 0 (condition on reaching s_t, then the inner
expectation over a_t is zero). So

$$ \nabla_\theta J = \mathbb{E}_{\tau}\!\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \big( G_t - b(s_t) \big) \right] $$

for *any* b(s) — still exactly unbiased. Why bother? Variance. In CartPole
every surviving step pays +1, so every G_t is positive: the raw estimator
pushes up the probability of *every* action it happened to sample, good or
bad, and only the slow accumulation of averages sorts them out. Subtract
"how well you typically do from here" and the multiplier becomes signed:
better-than-usual outcomes push their actions up, worse-than-usual push
down, *within a single episode*. The near-optimal choice (you'll verify the
intuition in exercise E4) is b(s) = V^π(s) — and G_t − V^π(s_t) is exactly a
sample of the advantage from §3.

**Step 8 — the theorem, stated.** Conditioning G_t on (s_t, a_t) (tower
property once more) turns it into its mean, Q^π(s_t, a_t), giving the
textbook statement of the **policy gradient theorem**:

$$ \nabla_\theta J = \mathbb{E}\!\left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; Q^{\pi}(s_t, a_t) \right] = \mathbb{E}\!\left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; A^{\pi}(s_t, a_t) \right] $$

with the second equality by Step 7 with b = V^π. Read it as **weighted
maximum likelihood**: ∇log π is the same score you ascend when fitting a
model to data (Week 08), but here each data point (s_t, a_t) is weighted by
how much better than average it turned out. Supervised learning treats every
label as worth imitating; RL lets the weights be earned — and possibly
negative.

*Honesty note on γ:* carried through strictly, the discounted theorem puts a
γ^t weight on the whole term at time t. Nearly every implementation drops
that outer γ^t (while keeping γ inside G_t), which is a known, accepted bias
— the estimator optimizes something slightly different from discounted J.
File it; don't lose sleep.

## 6. REINFORCE: the theorem as an algorithm

Replace the expectation with a sample average and you have **REINFORCE**
(Williams, 1992):

1. Run the current policy for one (or a few) episodes; store states,
   actions, rewards.
2. Compute the reward-to-go G_t at every step.
3. Ascend: θ ← θ + α Σ_t ∇log π_θ(a_t|s_t) (G_t − b(s_t)).
4. Throw the data away (it came from the *old* policy) and repeat.

Autograd builds the estimator for you via a **surrogate loss**: minimize
−Σ_t log π_θ(a_t|s_t) · Ĝ_t with the weights Ĝ_t treated as constants
(`.detach()` in PyTorch, so no gradient flows into them) — its gradient is
exactly the negative of Step 7's estimator. The core of a CartPole policy:

```python
import torch
import torch.nn as nn

policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))

def run_episode(env):
    logps, rewards = [], []
    s, info = env.reset()
    done = False
    while not done:
        logits = policy(torch.as_tensor(s, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        s, r, terminated, truncated, info = env.step(a.item())
        rewards.append(r)
        done = terminated or truncated
    return logps, rewards

def rewards_to_go(rewards, gamma):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return torch.tensor(out)
```

then `loss = -(torch.stack(logps) * returns.detach()).sum()` and an Adam
step. Two practical notes you'll rediscover in the exercises: normalizing
the batch of returns to zero mean and unit variance is a cheap baseline-like
variance reducer (the mean subtraction is a constant baseline; the std
division just rescales the LR); and REINFORCE's curves are *noisy* —
seed-to-seed spread is enormous, which is why the acceptance criteria this
week average over 10 seeds. A single lucky curve is not a result, in RL more
than anywhere else in this course.

## 7. Actor–critic: learn the baseline, bootstrap the return

Step 7 wants b(s) ≈ V^π(s), which we don't know — so fit it. Train a second
network V̂_φ(s) (the **critic**) by regression on observed returns, while
the policy (the **actor**) trains on advantages. And once you trust the
critic a little, use it *inside* the target too, via §3's Bellman identity:
since G_t = r_{t+1} + γG_{t+1}, a one-step estimate of the advantage is the
**TD error** (temporal-difference error)

$$ \delta_t = r_{t+1} + \gamma\, \hat V_\phi(s_{t+1}) - \hat V_\phi(s_t) $$

— "what one real step plus the critic's forecast says, minus what the critic
promised." Using δ_t in place of G_t − b(s_t) is **bootstrapping**: the
target leans on the current estimate of V̂. The trade is the fundamental one
of this week: the Monte-Carlo return G_t is unbiased but high-variance (it
sums an episode's worth of randomness); δ_t is low-variance (one step of
randomness) but *biased* whenever the critic is wrong — which is always,
early on. The **advantage actor–critic** loop:

1. Act for a batch of steps with π_θ; store (s, a, r, s′).
2. Critic: minimize (r + γV̂_φ(s′) − V̂_φ(s))² — push V̂ toward Bellman
   consistency.
3. Actor: ascend Σ ∇log π_θ(a|s) · δ, with δ detached.

(The standard middle ground, **GAE** — generalized advantage estimation —
blends the k-step estimators with a decay λ, giving a variance–bias dial.
Know the name; PPO implementations all use it.)

## 8. PPO: taking bigger steps without falling over

Two REINFORCE-family pains remain. First, **sample efficiency**: the
estimator is only valid for data from the *current* policy, so every batch
is used for one gradient step and discarded — painful when a sample is a
real robot movement or an LLM generation with a reward-model call. Second,
**fragility**: one too-large step changes the policy, which changes the data
distribution, which can collapse performance with no way back (the data that
would fix you is no longer visited). Supervised learning forgives bad steps
— the dataset waits patiently; RL punishes them twice.

To reuse a batch collected under an old policy π_old for several updates of
π_θ, correct with an importance ratio (the same importance-sampling move as
Week 08's expectation identities):

$$ r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}, \qquad L^{\text{IS}}(\theta) = \mathbb{E}\!\left[ r_t(\theta)\, \hat A_t \right] $$

whose gradient at θ = θ_old is exactly the policy gradient. But maximizing
L^IS freely is catastrophic: the surrogate happily inflates r_t to chase a
positive advantage far beyond where the old data says anything. **PPO**
(proximal policy optimization) clips the incentive instead:

$$ L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[ \min\!\big( r_t(\theta)\, \hat A_t,\;\; \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, \hat A_t \big) \right] $$

with ε ≈ 0.1–0.2. Case analysis — do it on paper, it's the third derivation:

- **Â_t > 0** (action was good): the objective grows with r_t, but the clip
  freezes it once r_t > 1+ε — the gradient there is exactly zero. You may
  make a good action more likely, but only ~ε more likely per data batch.
- **Â_t < 0** (action was bad): the objective improves as r_t shrinks, until
  r_t < 1−ε, where it flattens — same cap on how hard you push away.
- The **min** makes the clip one-sided pessimistic: when the *unclipped*
  term is worse (e.g. r_t drifted the wrong way for the advantage's sign),
  the min selects it, so you always feel the penalty of a bad move; you just
  can't over-collect the reward of a good one.

The clip builds a **trust region** — "don't move far from the policy that
generated this data" — out of nothing but `min` and `clamp`, which is why
PPO displaced its more mathematically elaborate ancestor (TRPO, which
enforced a KL constraint with second-order machinery). The full recipe:
collect a few thousand steps with π_old; compute GAE advantages with the
critic; run a few epochs of minibatch SGD on
L^CLIP − c₁·(value loss) + c₂·(entropy bonus); repeat. The **entropy bonus**
pays the policy to stay stochastic a while longer — insurance against §11's
entropy collapse.

## 9. The RLHF connection, closed

Week 31 asked you to take on faith that "PPO fine-tunes the policy against
the reward model." You can now expand every word. The mapping, which
exercise E7 has you complete:

| RL quantity | RLHF instantiation |
|---|---|
| state s_t | the prompt + tokens generated so far |
| action a_t | the next token (a ~50k-way discrete choice) |
| policy π_θ | the LLM itself (its softmax over the vocabulary) |
| episode | one full completion |
| reward | reward-model score, paid once at the final token — minus a per-token β·KL penalty against the frozen reference model |
| baseline / critic | a value head bolted onto the LLM, predicting the final score from the tokens so far |
| trust region | the PPO clip *and* the KL-to-reference term |

Points worth noticing. The reward is *sparse* — one number after hundreds of
actions — which is precisely the long-range credit-assignment regime where
the critic earns its keep. The KL penalty and the clip solve different
problems at different ranges: the clip keeps each update near the *sampling*
policy (an optimization safeguard); the KL-to-reference keeps the whole
process near the *original model* (a don't-forget-English,
don't-hack-the-reward-model safeguard). And the environment — the thing
whose "dynamics" the theorem let us ignore — is nearly trivial here:
appending your own token to the context is a deterministic transition. The
hard part is entirely the reward. Week 31's DPO derivation showed that for
this particular MDP the RL loop can be integrated out exactly; now you can
appreciate how much machinery that shortcut removes — and what you give up
(on-policy exploration against the reward model) by taking it.

## 10. RL for accelerator and detector control

The AI-for-science thread, with the physics self-contained.

**Accelerator control.** An accelerator steers beams of charged particles
with magnets: **dipoles** bend the beam, **quadrupoles** focus it (like
lenses), and small **corrector** magnets fine-tune the orbit. **Beam
position monitors (BPMs)** measure where the beam actually is at points
along the ring or line. The machine drifts — tunnel temperature, ground
motion, power-supply ripple — so the "correct" magnet settings move over
hours. As an MDP: state = the vector of BPM readings (plus recent history,
to restore Markovness under drift); actions = small changes to corrector
currents; reward = negative orbit error (or delivered beam intensity, or
−beam loss). Published results — RL steering beams at CERN (AWAKE, LINAC4)
and reward-driven tuning at light sources — follow exactly this pattern,
and the paper you skim this week should be read by extracting that
state/action/reward triple, per the README.

**Detector control** is the same shape one floor down: keeping a
calorimeter's photosensor gains stable as temperature moves (state:
temperatures, currents, monitoring-LED response; action: high-voltage
trims; reward: gain uniformity), or tuning trigger thresholds against a
rate budget.

What makes lab RL genuinely different from CartPole — three constraints:

1. **Samples cost beam time.** Millions of environment steps are free in a
   simulator and unthinkable on a real machine. Hence: pretrain in
   simulation and transfer (and then fight the **sim-to-real gap** — the
   simulator is never the machine), or use extremely sample-efficient
   methods.
2. **Exploration can break things.** A random-exploring policy will happily
   steer the beam into the pipe. Real deployments bound actions, wrap the
   agent in interlocks, and explore within safe envelopes.
3. **Sometimes RL is the wrong tool.** If the problem is "find the best
   static setting of a handful of knobs" with no sequential structure,
   Bayesian optimization (a sample-efficient black-box optimizer) usually
   wins. RL earns its complexity when the problem is genuinely sequential —
   when *this* correction changes the state the *next* correction faces.
   Knowing which problem you have is the actual expertise.

## 11. Reading an RL training run

RL curves fail differently from supervised ones (Week 16's debugging eye
needs recalibrating). Track these with your Week-41 infrastructure — mean
return, policy entropy, value loss, KL(π_new‖π_old) per update — and learn
the pathologies:

- **High variance / seed lottery.** Identical hyperparameters, wildly
  different curves. Not a bug; the estimator is noisy and the process
  nonstationary. Report seed-averaged curves with spread (the exercises
  demand 10 seeds), never a single run.
- **Collapse after progress.** Return climbs, then craters and doesn't
  recover: a too-large update moved the policy somewhere bad, and the data
  that would fix it is no longer collected. This failure — impossible in
  supervised learning, where the dataset waits — is the one PPO's clip
  exists to prevent. Check the per-update KL: spikes precede collapse.
- **Entropy collapse.** Policy entropy falls to ~0 early: the policy went
  deterministic before it found anything good, exploration ends, learning
  flatlines at a mediocre policy. Levers: entropy bonus, smaller steps.
- **Reward hacking.** The return goes *up* while the behavior you wanted
  gets worse: the agent optimized the reward you wrote, not the one you
  meant (Goodhart's law). In RLHF this appears as over-optimization against
  the reward model — gibberish the RM happens to score highly — and is why
  the KL leash exists. The fix is never "more training"; it's a better
  reward or a tighter leash.

## Check yourself

1. Define state, action, transition kernel, reward, and discount for
   "keep the beam centered under drift" — and say what the Markov property
   demands of your state choice.
2. Why must the policy be stochastic for the policy gradient derivation to
   even start?
3. Write the log-derivative trick and state exactly where it enters the
   derivation and what problem it solves there.
4. Which terms of log p_θ(τ) die when ∇_θ hits them, and what practical
   superpower does that give the method?
5. Prove in three lines that a state-dependent baseline adds no bias.
6. Your friend's REINFORCE pushes up the probability of every action in
   every CartPole episode. Why does it still eventually learn — and what
   does the baseline change about *how* it learns?
7. In PPO with Â > 0, what is the gradient of the clipped objective when
   r_t = 1.5 and ε = 0.2 — and why is that the design goal?
8. In RLHF, the environment transition is deterministic (append the token).
   So which part of the MDP carries all the difficulty, and which two
   leashes constrain the policy's movement?

## Answers

1. State: BPM readings (plus recent history/trend); action: corrector-magnet
   current changes; transitions: beam physics + drift, unknown to the agent;
   reward: negative orbit deviation; discount: <1, horizon of the control
   task. Markov demands the state carry everything predictive — under drift,
   instantaneous position alone isn't enough, so include history.
2. Two reasons from the derivation: π_θ(a|s) must be a differentiable
   probability so ∇log π exists, and the log-derivative trick divides by
   π(a|s) — the sampled actions must have nonzero probability. (Plus
   exploration: a deterministic policy never gathers evidence about other
   actions.)
3. ∇p = p∇log p. It enters converting ∫∇p_θ(τ)R(τ)dτ — not an expectation,
   so not estimable from samples — into E[∇log p_θ(τ) R(τ)], which a Monte
   Carlo average over sampled trajectories estimates without bias.
4. log ρ(s₀) and every log P(s_{t+1}|s_t,a_t): no θ inside, gradient zero.
   Superpower: the gradient never references the dynamics — you can improve
   a policy on a system you cannot model, only interact with (model-free RL).
5. E[∇log π(a|s)b(s)] = b(s)E_a[∇log π(a|s)] = b(s)·Σ_a∇π(a|s) =
   b(s)·∇Σ_aπ = b(s)·∇1 = 0. The factor pulls out because b doesn't depend
   on a; the score's mean is zero because probabilities sum to one.
6. All G_t > 0, so every sampled action is reinforced — but *good* actions
   appear in higher-return episodes and get bigger weights, so on average
   they win; learning rides on differences between large positive weights,
   slowly. With b ≈ V the weights become signed advantages: bad actions are
   pushed down within a single episode — same expectation, far less variance.
7. Zero. r_t = 1.5 > 1+ε = 1.2, so the clipped term is the active one under
   the min (for Â > 0), and it's constant in θ there. That's the point: a
   good action can gain at most ~ε probability ratio per batch, bounding
   how far one update strays from the data-generating policy.
8. The reward: one sparse reward-model score per completion (hence the
   value head for credit assignment) — and the RM is imperfect, hence
   hackable. Leashes: the PPO clip (stay near the sampling policy per
   update) and the KL-to-reference penalty (stay near the original model
   overall).

## New terms

- **reinforcement learning (RL)** — learning to act from reward signals in
  an environment your own actions influence.
- **agent / environment / reward** — the actor; the system it acts in; the
  scalar feedback per step.
- **MDP** — Markov decision process: states, actions, transition kernel,
  reward, discount.
- **Markov property** — next state/reward depend only on current state and
  action; a property you engineer into the state.
- **trajectory / episode** — the recorded sequence of one run; one run from
  start to terminal state.
- **return G_t** — discounted sum of future rewards from step t.
- **discount factor γ** — convergence + impatience + soft horizon ≈ 1/(1−γ).
- **policy π(a|s)** — distribution over actions given state.
- **V, Q, advantage A = Q − V** — expected return from a state; from a
  state–action pair; how much better an action is than the policy's average.
- **Bellman consistency** — V^π(s) = E[r + γV^π(s′)].
- **score function** — gradient of a log-probability; zero mean under its
  own distribution.
- **log-derivative trick** — ∇p = p∇log p; turns gradients of expectations
  into expectations of gradients.
- **policy gradient theorem** — ∇J = E[Σ ∇log π(a_t|s_t) A^π(s_t,a_t)].
- **reward-to-go** — the return from step t on; past rewards drop by the
  zero-mean lemma.
- **baseline** — any b(s) subtracted from the return weight; bias-free,
  variance-reducing; best choice ≈ V^π.
- **REINFORCE** — Monte-Carlo policy gradient with the surrogate-loss
  implementation.
- **actor–critic** — policy (actor) trained on advantages estimated with a
  learned value function (critic).
- **TD error δ** — r + γV̂(s′) − V̂(s); a one-step, low-variance, biased
  advantage estimate.
- **bootstrapping** — building targets from your own current value
  estimates.
- **GAE** — λ-blend of k-step advantage estimators; the standard
  bias–variance dial.
- **importance ratio r_t(θ)** — π_θ/π_old; corrects for reusing old data.
- **PPO / clipped surrogate** — min(rÂ, clip(r, 1±ε)Â); a trust region
  built from clamp.
- **trust region** — a bound on how far one update may move the policy.
- **entropy bonus / entropy collapse** — reward for staying stochastic; the
  failure where the policy goes deterministic before finding anything good.
- **reward hacking** — optimizing the written reward against its intent
  (Goodhart); RLHF's over-optimization of the reward model.
- **sim-to-real gap** — the transfer penalty from simulator-trained policies
  meeting the real machine.
- **dipole / quadrupole / corrector / BPM** — bending, focusing, and
  fine-tuning magnets; beam position monitors — the accelerator MDP's
  action and state hardware.

## Going deeper

- Sutton & Barto, *Reinforcement Learning: An Introduction* (free online) —
  Ch. 3 for MDPs done properly, Ch. 13 for policy gradients; the Ch. 6 intro
  pages for TD intuition. The field's canonical text.
- OpenAI Spinning Up — "Key Concepts in RL" and "Intro to Policy
  Optimization" re-derive this lesson's §5 with different notation (do the
  translation; it's a good check), and the PPO page documents the exact
  practical recipe.
- Schulman et al., *Proximal Policy Optimization Algorithms* — read for the
  clipped objective and its rationale, not the benchmark tables.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* — where the GAE dial comes from, if §7 left you
  wanting the full family.
- One accelerator-RL paper by title (e.g. Kain et al., "Sample-efficient
  reinforcement learning for CERN accelerator control," PRAB 2020, or any
  light-source RL tuning paper) — skim for the state/action/reward table,
  per the README.
