# Week 31 — Alignment: Instruction Tuning, RLHF, DPO

~10 hrs. Before starting you should be able to: write KL divergence and maximum
likelihood from memory and use them in a derivation (Week 08); derive the logistic
regression loss and its gradient (Week 10); explain what SFT does and how preference
for chat formats is trained in (Weeks 29–30); fine-tune with LoRA (Week 30).

A pretrained LLM is a next-token predictor: ask it "How do I calibrate a
calorimeter?" and its honest continuation might be another question, because on the
internet questions often follow questions. **Alignment** is the engineering that
turns that predictor into an assistant — one that answers, follows instructions,
declines harmful requests, and admits uncertainty. This week you learn the pipeline
(SFT → preference data → reward model → RL), and then the week's flagship
derivation: DPO, which shows the entire RL stage can be integrated out *exactly*,
collapsing reinforcement learning into a single supervised loss. Every step uses
math you already own — KL divergence, MLE, and logistic regression.

## 1. The pipeline at altitude

Modern assistants are built in stages, each changing something specific:

1. **Pretraining** (Weeks 25–28): next-token prediction on a huge corpus. Produces
   *capability* — knowledge, syntax, reasoning patterns — but no preference for being
   helpful. It completes text; that's all it was asked to do.
2. **Supervised fine-tuning (SFT)** (Week 30): train on curated (instruction →
   good response) demonstrations. Teaches the *format* of assistance: answer the
   question, use the chat template, stop when done.
3. **Preference optimization** (this week): humans compare pairs of model responses
   ("A is better than B"); the model is optimized so the preferred kind of response
   becomes more likely. Done either with an explicit reward model plus reinforcement
   learning (**RLHF**) or directly (**DPO**).

Why is stage 3 needed at all — why not just more SFT? Two structural reasons. First,
SFT can only say "here is a good response"; it cannot express "response A is *better
than* B" — and much of what we want (more honest, less verbose, safer) is easier for
a human to *compare* than to write from scratch. Judging is cheaper than authoring,
so preference labels scale where demonstrations don't. Second, SFT only pushes
probability *up* on demonstrated text; it has no mechanism to push probability *down*
on plausible-but-bad continuations the demonstrations never mention. Comparisons
carry exactly that signal.

## 2. Preference data and the Bradley–Terry model

The raw material: a prompt $x$, two sampled responses, and a human label saying which
is better. Write the winner $y_w$ and the loser $y_l$; the dataset is triples
$(x, y_w, y_l)$. (Note that generating *diverse* candidate pairs requires sampling at
$T > 0$ — Week 29 — since greedy decoding would produce twice the same answer.)

To learn from comparisons we need a model of how comparisons arise. The
**Bradley–Terry model** (1952, built for sports rankings) posits that each response
has a latent scalar quality — a **reward** $r(x, y)$ — and that the probability a
human prefers $y_w$ over $y_l$ depends only on the reward *difference*, squashed
through the sigmoid $\sigma(t) = 1/(1 + e^{-t})$ from Week 10:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big).$$

Sanity checks: equal rewards give $P = 1/2$; a large positive gap gives $P \to 1$;
and only differences matter — shifting every reward by any $C(x)$ changes nothing.
Remember that shift-invariance; it is the load-bearing fact of §5.

A **reward model** $r_\phi$ (typically the SFT model with its token head swapped for
a scalar head) is fit by maximum likelihood (Week 08). The negative log-likelihood of
the dataset is

$$\mathcal{L}_R(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)}
\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].$$

Look closely: this is *exactly* Week 10's logistic regression loss with the score
difference playing the role of the logit and the label always "1" (winner beats
loser). Everything you know about logistic regression applies, including its gradient
$\big(\sigma(\cdot) - 1\big)\nabla(\text{score difference})$: the update is large
when the model currently ranks the pair *wrongly*, and fades as it gets the ranking
right. File that shape away — the DPO gradient in §6 will look strikingly familiar.

## 3. The RLHF objective

With a reward model in hand, we want a policy — RL vocabulary for the generating
model, written $\pi_\theta(y \mid x)$, the probability the model assigns to the full
response $y$ given prompt $x$ — that achieves high reward. But not *only* high
reward. A learned $r_\phi$ is only trustworthy near the data it was fit on; a model
free to maximize it will wander into regions where $r_\phi$ is wrong and exploit its
errors — emitting whatever pathological text happens to score high. This is **reward
hacking**, and it is Goodhart's law ("a measure that becomes a target ceases to be a
good measure") in loss-function form.

The fix: penalize divergence from a trusted anchor. Let $\pi_{\text{ref}}$ be the
frozen SFT model (the **reference model**). The RLHF objective is

$$\max_{\pi}\;\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)}
\big[ r(x, y) \big]
\;-\; \beta\, \mathrm{KL}\!\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big),$$

where the KL divergence (Week 08) measures how far the policy has drifted from the
reference, and $\beta > 0$ prices that drift. Two limits orient you: $\beta \to
\infty$ pins $\pi = \pi_{\text{ref}}$ (no learning); $\beta \to 0$ permits unbounded
reward hacking. In between, the policy trades reward against staying in the region
where the reward model — and the language model itself — remain sane.

**PPO, at sketch level.** The classical way to optimize this objective is
reinforcement learning: sample responses from the current policy, score them
(reward minus the KL penalty), and nudge the policy toward higher-scoring samples.
PPO (proximal policy optimization — derived properly in Week 43) is the standard
algorithm. Its signature move is the clipped surrogate: for probability ratio
$\rho = \pi_\theta(y|x)/\pi_{\text{old}}(y|x)$ and advantage $A$ (how much better
than expected the sample scored),

$$L^{\text{CLIP}} = \mathbb{E}\big[\min\big(\rho A,\;
\mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)\, A\big)\big],$$

which flattens the incentive once a single update moves any sample's probability
ratio outside $[1-\epsilon, 1+\epsilon]$ — a trust region enforced by construction,
so no single noisy batch can yank the policy far. The machinery cost is real: four
models in memory (policy, reference, reward model, value function), sampling inside
the training loop, and RL's notorious instability. Hold that thought.

## 4. Warm-up: the optimal policy in closed form

DPO begins with a question that sounds academic: *never mind algorithms — what is
the mathematical solution of the RLHF objective?* For a fixed prompt $x$, we want

$$\max_{\pi}\; \mathbb{E}_{y \sim \pi}[r(x, y)]
- \beta\, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}).$$

**Step 1 — write everything as one expectation.** Expanding the KL as
$\mathbb{E}_{y \sim \pi}[\log \pi(y|x) - \log \pi_{\text{ref}}(y|x)]$ and dividing by
$\beta$ (harmless: positive constant), maximizing the objective is the same as
*minimizing*

$$\mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
- \frac{1}{\beta} r(x, y)\right].$$

**Step 2 — force it into the shape of a KL.** A KL divergence is
$\mathbb{E}_\pi[\log \pi - \log(\text{something})]$; we have an extra $-r/\beta$
riding along. Absorb it into the log: since $r/\beta = \log e^{r/\beta}$,

$$= \mathbb{E}_{y \sim \pi}\left[\log
\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)\, e^{\,r(x,y)/\beta}}\right].$$

The denominator is positive but doesn't sum to 1, so it's not yet a distribution.
Normalize it: define the **partition function** (the same normalizer $Z$ as every
softmax and Boltzmann distribution you've met)

$$Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\, e^{\,r(x, y)/\beta},
\qquad
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{\,r(x, y)/\beta}.$$

Multiplying and dividing by $Z(x)$ inside the log splits off a $-\log Z(x)$, which
does not depend on $y$, so it slides out of the expectation:

$$= \mathrm{KL}\big(\pi \,\|\, \pi^*\big) \;-\; \log Z(x).$$

**Step 3 — read off the answer.** $\log Z(x)$ doesn't depend on $\pi$ at all. And KL
is minimized — value zero — exactly when its arguments are equal (Week 08, via
Jensen's inequality). So the optimum is, exactly,

$$\boxed{\;\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\,
e^{\,r(x, y)/\beta}.\;}$$

Pause and look at it: this is a **Gibbs/Boltzmann distribution** — reference policy
reweighted by $e^{\text{reward}/\beta}$, with $\beta$ playing temperature, exactly
the structure of Week 29's temperature sampling ($\pi_{\text{ref}}$ uniform,
$r = $ logits, $\beta = T$). The RLHF optimum is a Boltzmann tilt of the SFT model
toward reward.

So why doesn't everyone just *write down* $\pi^*$ and go home? Because $Z(x)$ sums
over **every possible response** $y$ — all token sequences. For a 50k vocabulary and
100-token responses, that's $50000^{100}$ terms. $\pi^*$ exists, is unique, and is
uncomputable. This is where RLHF says "approximate it with PPO." DPO says something
cleverer.

## 5. The DPO derivation: integrate the RL out

Here is the pivot, and it deserves slow reading. We can't evaluate $\pi^*$ — but we
can *solve the boxed equation for $r$*:

$$e^{\,r(x, y)/\beta} = \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\, Z(x)
\;\;\Longrightarrow\;\;
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
+ \beta \log Z(x).$$

Read it in words: *any* reward function and its optimal policy are two descriptions
of the same object — the reward is recoverable from the policy (up to the
prompt-dependent constant $\beta \log Z(x)$). A policy secretly *is* a reward model.

Now recall §2: the Bradley–Terry likelihood only ever uses reward **differences** at
the same prompt, and is invariant to per-prompt constant shifts. Substitute the
expression for $r$ into $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$ and watch:

$$r(x, y_w) - r(x, y_l)
= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
+ \underline{\beta \log Z(x)}
- \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
- \underline{\beta \log Z(x)}.$$

**The $\beta \log Z(x)$ terms cancel.** The one uncomputable object in the whole
story appears twice with opposite signs — because both responses share the same
prompt, hence the same normalizer — and vanishes. What survives is built entirely
from things a GPU can evaluate: log-probabilities of two specific given responses
under two models. No sum over all possible responses anywhere.

The endgame is now pure Week 08 MLE. We *want* the optimal policy; we have a dataset
of human preferences that (per Bradley–Terry) were generated by the underlying
reward; and we've shown that reward differences are a formula in the optimal policy.
So: parameterize the policy as $\pi_\theta$, define each response's **implicit
reward**

$$\hat{r}_\theta(x, y) = \beta \log
\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)},$$

and fit $\theta$ by maximizing the Bradley–Terry likelihood of the observed
preferences — i.e. minimize the negative log-likelihood:

$$\boxed{\;\mathcal{L}_{\text{DPO}}(\theta) =
-\,\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)\right].\;}$$

This is **Direct Preference Optimization** (Rafailov et al., 2023). Take stock of
what just happened: the reward model, the sampling loop, the value function, and PPO
itself have all been integrated out — analytically, not approximately. What remains
is a supervised classification loss (it *is* §2's reward-model loss, with the
implicit reward in place of a separate network) on a fixed offline dataset: four
forward passes per example ($\pi_\theta$ and $\pi_{\text{ref}}$ on $y_w$ and $y_l$;
each $\log \pi(y|x)$ is just a sum of next-token log-probs — your Week 26 loss
without the minus sign), one backward pass, no sampling during training, two models
in memory instead of four. The KL anchor hasn't been dropped: it lives inside the
loss, because the implicit reward is measured *relative to* $\pi_{\text{ref}}$ and
scaled by the same $\beta$.

One honest footnote: DPO finds the optimum *of the Bradley–Terry model fit to your
preference dataset*, offline. PPO-based RLHF samples fresh responses during training,
so it can explore beyond the support of the collected pairs — and exploit reward-model
errors there, for better and worse. The trade is exactness and stability against
on-policy exploration; both are used in practice, and the derivation you just did is
what lets you reason about the difference instead of vibing about it.

## 6. What the DPO gradient does

Differentiate the loss (chain rule through $\log \sigma$; recall from Week 10 that
$\frac{d}{dt}\log\sigma(t) = \sigma(-t)$, i.e. $1 - \sigma(t)$). Writing the implicit
rewards $\hat{r}_w = \hat{r}_\theta(x, y_w)$ and $\hat{r}_l = \hat{r}_\theta(x, y_l)$:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} =
-\,\beta\, \mathbb{E}\Big[
\underbrace{\sigma\big(\hat{r}_l - \hat{r}_w\big)}_{\text{weight}}
\big(
\underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{push up winner}}
- \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{push down loser}}
\big)\Big].$$

Read the three pieces:

- The direction: increase the winner's log-probability, decrease the loser's — the
  push-down term being exactly what SFT could never express (§1).
- The weight $\sigma(\hat{r}_l - \hat{r}_w)$: large (→ 1) when the model's implicit
  reward currently ranks the pair *wrong* ($\hat{r}_l > \hat{r}_w$), small (→ 0) when
  it already ranks it right. Confidently correct pairs stop contributing; the model
  spends its gradient budget on its mistakes — the same "wrongness-weighted" shape
  as logistic regression's gradient in §2, because it *is* that gradient.
- The $\beta$ out front, and inside the weight: stronger anchoring to
  $\pi_{\text{ref}}$ means each example must argue harder to move the policy.

**Worked numbers.** Suppose for one pair, summing token log-probs gives
$\log \pi_\theta(y_w|x) = -42.0$, $\log \pi_{\text{ref}}(y_w|x) = -41.0$,
$\log \pi_\theta(y_l|x) = -38.0$, $\log \pi_{\text{ref}}(y_l|x) = -40.0$, with
$\beta = 0.1$. Implicit rewards: $\hat{r}_w = 0.1 \times (-42 + 41) = -0.1$;
$\hat{r}_l = 0.1 \times (-38 + 40) = +0.2$. The model currently rates the *loser*
higher (it has drifted toward $y_l$ relative to the reference). Margin:
$\hat{r}_w - \hat{r}_l = -0.3$; loss $= -\log \sigma(-0.3) = 0.855$; gradient weight
$\sigma(0.3) = 0.574$ — a hard pair, weighted heavily. After training moves the
policy, say the margin reaches $+2.0$: loss $-\log\sigma(2.0) = 0.127$, weight
$\sigma(-2.0) = 0.119$ — nearly retired. Track the *mean margin*
$\hat{r}_w - \hat{r}_l$ across your eval pairs and you have DPO's standard training
diagnostic (it should climb).

## 7. Safety and refusals: engineered behavior, with failure modes

Refusal behavior — a model declining to help synthesize a nerve agent — is not an
emergent conscience. It is *trained*, by the exact machinery above: preference pairs
where the refusal is labeled $y_w$ for harmful prompts and $y_l$ for benign ones,
plus SFT demonstrations of good refusals. Constitutional AI (Anthropic) reduces the
human-labeling load by having a model *critique and revise its own outputs* against a
written list of principles, then using those revisions and AI-generated preference
labels in place of most per-example human ones — the pipeline shape is unchanged.

Because it is trained behavior, it fails like trained behavior:

- **Over-refusal.** The training signal generalizes imperfectly; benign prompts near
  the decision boundary get refused. You will likely see this with dual-use physics —
  "estimate the activation dose rate near the beam dump" pattern-matches to
  radiological harm even when it is Tuesday's detector-operations homework. Your
  refusal-probe exercise measures this directly.
- **Jailbreaks.** Adversarial framings (roleplay, encodings, many-shot patterns,
  "hypothetically...") move a prompt out of the trained refusal distribution. Robust
  refusal is an unsolved generalization problem, not a solved filter.
- **Sycophancy.** Preference data generated by human approval contains a systematic
  bias: people rate agreement and flattery highly. Optimizing the proxy imports the
  bias — the model learns to tell you your wrong calculation is right. This is reward
  hacking (§3) where the exploited errors are *in the labels themselves*, and it is
  the reason you should never use "the model agreed with me" as evidence.

The through-line of the month: every alignment objective is a proxy for what we
actually want, and optimization pressure finds the gap between proxy and target.
Week 32 turns that suspicion into measurement.

## 8. Worked example: one DPO step, end to end

Tiny but real — the full computation on one preference pair, using log-probs from any
small instruct model and its frozen copy. This is exercises E2/E3's skeleton.

```python
import torch
import torch.nn.functional as F

def response_logprob(model, prompt_ids, response_ids):
    # log pi(y|x): sum of next-token log-probs over the response tokens only
    full = torch.cat([prompt_ids, response_ids], dim=1)
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(full).logits
    # logits at position t predict token t+1; slice the response region
    start = prompt_ids.shape[1] - 1
    logps = F.log_softmax(logits[:, start:-1, :].float(), dim=-1)
    picked = torch.gather(logps, 2, response_ids.unsqueeze(-1)).squeeze(-1)
    return picked.sum(dim=1)                      # one scalar per sequence

beta = 0.1
# policy = trainable (LoRA-wrapped) model; ref = frozen copy of it
logp_w      = response_logprob(policy, prompt_ids, win_ids)
logp_l      = response_logprob(policy, prompt_ids, lose_ids)
logp_w_ref  = response_logprob(ref,    prompt_ids, win_ids)
logp_l_ref  = response_logprob(ref,    prompt_ids, lose_ids)

margin = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
loss = -F.logsigmoid(margin).mean()
loss.backward()
print(loss.item(), margin.item())   # watch margin climb over steps
```

Every line is a formula from this lesson: `response_logprob` is
$\log \pi(y|x) = \sum_t \log \pi(y_t | x, y_{<t})$, `margin` is
$\beta(\hat{r}_w - \hat{r}_l)$ built from the four log-probs of §5, and `loss` is the
boxed $\mathcal{L}_{\text{DPO}}$. TRL's `DPOTrainer` wraps this plus batching and the
reference-model bookkeeping; your exercise E2 checks your version against it to 1e-5,
which is what "I derived it and it's the same thing" looks like numerically.

## Check yourself

1. Name the two structural things preference optimization gives that SFT cannot.
2. Write the Bradley–Terry model and the reward-model loss. Which Week 10 object is
   the loss, exactly?
3. In the RLHF objective, what specifically goes wrong as $\beta \to 0$, and what is
   the name of that failure?
4. Derive $\pi^*$ from the RLHF objective in three steps (one line of purpose each).
5. Why exactly does $\log Z(x)$ cancel in DPO — what two facts conspire?
6. Define the implicit reward. In what precise sense is "your language model secretly
   a reward model"?
7. In the worked numbers of §6, recompute the loss and gradient weight if
   $\log \pi_\theta(y_l|x)$ were $-41.0$ instead of $-38.0$.
8. Your model enthusiastically confirms your wrong invariant-mass calculation. Which
   alignment failure mode is this, and where did it enter the pipeline?

## Answers

1. (i) It learns from comparisons, which are cheaper to label than demonstrations and
   can express "better than"; (ii) it can push probability *down* on bad responses,
   which SFT's likelihood objective cannot.
2. $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$;
   $\mathcal{L}_R = -\mathbb{E}[\log \sigma(r_w - r_l)]$. It is logistic regression
   with the score difference as logit and label fixed at 1.
3. The KL anchor vanishes, so the policy is free to maximize the learned reward
   anywhere — including regions where $r_\phi$ is wrong — and exploits those errors:
   reward hacking (Goodhart's law).
4. (i) Fold the KL and the reward into one expectation of a log-ratio (absorb
   $r/\beta$ as $\log e^{r/\beta}$); (ii) normalize the denominator with $Z(x)$ so it
   becomes the distribution $\pi^* \propto \pi_{\text{ref}} e^{r/\beta}$, splitting
   off a $\pi$-independent $-\log Z$; (iii) the objective is now
   $\mathrm{KL}(\pi\|\pi^*) + \text{const}$, minimized iff $\pi = \pi^*$.
5. Bradley–Terry depends only on the reward *difference* at a shared prompt, and
   $\beta \log Z(x)$ is the same additive constant for both responses at that prompt
   — so it appears twice with opposite signs.
6. $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.
   Inverting the optimal-policy formula shows any policy defines a reward for which
   it is RLHF-optimal (up to a per-prompt constant) — policy and reward are two
   coordinates for the same object.
7. $\hat{r}_l = 0.1 \times (-41 + 40) = -0.1 = \hat{r}_w$; margin $0$; loss
   $-\log \sigma(0) = \log 2 \approx 0.693$; weight $\sigma(0) = 0.5$.
8. Sycophancy. It entered through the preference labels: human raters systematically
   prefer agreement, and optimizing the Bradley–Terry fit to biased labels learns the
   bias.

## New terms

- **alignment** — training stages that turn a next-token predictor into an assistant.
- **policy** $\pi_\theta(y|x)$ — RL's name for the generating model's distribution
  over responses.
- **preference pair** — $(x, y_w, y_l)$: prompt, preferred and dispreferred response.
- **Bradley–Terry model** — preference probability as sigmoid of a reward difference.
- **reward model** — a network mapping (prompt, response) to a scalar quality score.
- **RLHF** — reinforcement learning from human feedback: maximize learned reward
  minus a KL penalty to the reference.
- **reference model** $\pi_{\text{ref}}$ — the frozen SFT model anchoring the KL.
- **reward hacking** — exploiting errors in a learned reward (Goodhart's law).
- **PPO** — the clipped-surrogate RL algorithm classically used for RLHF (Week 43).
- **partition function** $Z(x)$ — the (uncomputable) normalizer of the optimal
  policy.
- **Gibbs/Boltzmann distribution** — $\propto \pi_{\text{ref}}\, e^{r/\beta}$; the
  RLHF optimum's form.
- **DPO** — direct preference optimization: the RLHF optimum's MLE recast as a
  supervised loss on preference pairs.
- **implicit reward** — $\beta \log(\pi_\theta/\pi_{\text{ref}})$; the reward a
  policy encodes.
- **margin** — implicit-reward gap $\hat{r}_w - \hat{r}_l$; DPO's training
  diagnostic.
- **Constitutional AI** — alignment using model self-critique against written
  principles in place of most per-example human labels.
- **over-refusal / jailbreak / sycophancy** — refusing benign prompts / adversarially
  evading trained refusals / learned flattery from approval-biased labels.

## Going deeper

- Rafailov et al., *Direct Preference Optimization* (arXiv 2305.18290), §§1–4 and
  Appendix A — this lesson's §§4–6 in the authors' own hand; read after your
  paper-and-pencil pass.
- Ouyang et al., *Training language models to follow instructions with human
  feedback* (InstructGPT, arXiv 2203.02155), §§1–3 and Fig. 2 — the canonical
  three-stage pipeline of §1.
- Lilian Weng, the RLHF blog post (lilianweng.github.io) — the PPO-level machinery
  §3 only sketched.
- Bai et al., *Constitutional AI* (Anthropic) — skim for how refusal behavior is
  trained without per-example human labels.
