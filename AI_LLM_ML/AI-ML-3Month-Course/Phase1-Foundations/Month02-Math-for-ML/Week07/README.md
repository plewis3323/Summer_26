# Week 07 — Probability & Statistics

You already do MLE every time you fit a peak; this week makes the machinery explicit and
adds the information-theoretic vocabulary (entropy, cross-entropy, KL) that ML losses
are written in.

## Objectives

- Manipulate joint/marginal/conditional distributions and apply Bayes' theorem cleanly.
- Derive MLE and MAP estimators and articulate exactly how a prior changes the answer.
- Derive least squares from a Gaussian likelihood (and name the assumption that breaks it).
- Compute entropy, cross-entropy, and KL divergence, and state what each measures.
- Connect: minimizing cross-entropy ⇔ minimizing KL ⇔ maximizing likelihood.

## Core material (~3 hrs)

- Bishop, *PRML* (in `references/`): §1.2 (probability theory) and §2.1–2.3
  (binary/multinomial variables, the Gaussian). Physicist-fast — much is review.
- Murphy, *Probabilistic Machine Learning: An Introduction* (free PDF): the probability
  and information-theory chapters as a second angle; skim, don't grind.
- 3Blue1Brown: the Bayes' theorem video, for the geometric picture.
- StatQuest: "Maximum Likelihood" and the entropy/cross-entropy videos if the
  definitions don't stick from the texts alone.

## Derivations (paper first)

- Bayes' theorem from the product rule; work one full example with a non-flat prior.
- MLE for the mean and variance of a Gaussian (and show the variance MLE is biased).
- Least squares from maximizing a Gaussian likelihood with iid noise; note where the
  σ = const assumption enters.
- MAP with a Gaussian prior on the parameters → L2-regularized least squares (ridge).
- KL(p‖q) for two Gaussians (closed form); show KL ≥ 0 via Jensen; show
  cross-entropy = entropy + KL.

## Exercises (built when the week starts)

1. **Bayesian coin.** Beta-Bernoulli updating on simulated flips; plot the posterior
   evolving; compare MLE, MAP, posterior mean.
   Accept when: posterior after N flips matches the closed-form Beta(α+h, β+t) and all three estimators are marked on the plot.
2. **Detector Bayes.** Given particle-ID efficiency and fake rate plus a prior
   abundance, compute P(true kaon | tagged kaon); scan vs prior.
   Accept when: the curve matches the hand-derived formula and the low-prior regime is flagged in one line.
3. **Least squares from likelihood.** Simulate straight-line data with Gaussian noise;
   maximize the log-likelihood numerically; compare with the analytic least-squares fit.
   Accept when: numerically-maximized parameters match the lstsq solution to 1e-6.
4. **Heteroscedastic twist.** Repeat with σᵢ varying per point; show plain least squares
   is now the wrong MLE and weighted least squares is right.
   Accept when: weighted fit recovers true parameters within errors over 1000 pseudo-experiments while unweighted shows the predicted excess variance.
5. **KL numerically.** Compute KL between two Gaussians by numerical integration and
   from the closed form; then KL(data-histogram ‖ model) for a mis-modeled peak.
   Accept when: numeric vs closed form agree to 1e-6 and KL correctly ranks two candidate models.
6. **Entropy of distributions.** Entropy of uniform/Gaussian/peaked histograms from
   samples; watch the binning dependence.
   Accept when: sampled Gaussian entropy approaches ½log(2πeσ²) as N grows, deviation < 2% at N = 10⁶.

## Deliverable

Completed notebook plus scanned derivations. The MAP→ridge derivation gets referenced
again in Week 09; file it where you can find it.

## Review

1. (Wk 06) PCA maximizes variance — this week reframes covariance as a Gaussian's shape
   parameter. What distribution is PCA implicitly fitting?
2. (Wk 06) Write ∇ₓ‖Ax − b‖² from memory; which derivation this week reused it?
3. (Wk 05) The projection matrix P and the least-squares/likelihood fit: what is the
   geometric relationship?
4. (Wk 04) Your pseudo-experiment loops need reproducible randomness — which NumPy API,
   and why not the global seed?
