# Month 06 — Deep Learning for Physics + Generative Models I

The arc: the month where the course turns squarely toward HEP-ML. Week 21 drops the
fixed-grid assumption — message-passing GNNs on point clouds, with ParticleNet, Particle
Transformer, and Exa.TrkX as the literature anchors. Weeks 22–23 build Generative Models
I: autoencoders and VAEs with the ELBO derived line by line, then normalizing flows via
change of variables and coupling layers, framed by the CaloChallenge and the case for
learned fast simulation. Week 24 is Capstone 2: either a GNN cluster/jet classifier
benchmarked against the Week-20 CNN, or a VAE fast-sim of EMCal showers validated on
physics observables (energy response, shower shapes).

The build-up is direct: Week 21's models and Week 22–23's generators both feed Week 24,
and the Week-20 project supplies the dataset and the baseline either capstone track
must beat or explain.

**Month-end deliverable:** Capstone 2 — tested repo + writeup; the Phase 2 gate also
requires re-doing the backprop and ELBO derivations cold.

**Sign-off:** tag `month-06-complete`, write `retro.md` (~250 words), open one issue for
the biggest thing you still don't understand.
