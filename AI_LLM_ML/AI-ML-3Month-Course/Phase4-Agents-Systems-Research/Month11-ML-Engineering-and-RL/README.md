# Month 11 — ML Engineering & RL

This month professionalizes how you train, speed up, and ship models — and adds the last
major theory block of the course. Week 41 puts experiment infrastructure under a Phase-2/3
project: tracking, config management, sweeps, and a model registry, so results stop living
in notebook scrollback. Week 42 is performance engineering: GPU architecture and roofline
intuition, profiling, mixed precision, `torch.compile`, DDP concepts, and inference
optimization. Week 43 is reinforcement learning done properly — MDPs through the policy
gradient theorem (derived on paper) to PPO — closing the loop with Week 31's RLHF and
pointing at control tasks (accelerator/detector *or* a standard control problem). Week 44 ships: one earlier model wrapped in a
tested FastAPI service in a container (spiral from Week 23), with health checks, monitoring, and CI that builds the image.

**Month-end deliverable:** a deployed (locally containerized) model service with green CI,
a tracked-and-swept retraining run behind it, and a cold re-derivation of the policy
gradient theorem in the week folder.

**Sign-off:** tag the commit `month-11-complete`, write a 250-word `retro.md` in this
folder, and open one issue for the biggest thing you don't yet understand.
