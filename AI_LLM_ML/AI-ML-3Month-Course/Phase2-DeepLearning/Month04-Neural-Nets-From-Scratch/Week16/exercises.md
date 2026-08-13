# Week 16 — Exercises

Work top to bottom. Setup (imports, seeds, the Week-15 char-MLP data, the
three provided broken-run scripts, and the `sys.path` line that imports your
Week 15 training loop) is given by the notebook; you write only the lines
each exercise asks for. E1–E6 live in the notebook. E7 *is* the mini-project
— `project.md` guides it; that work lives in your `mlp_vs_bdt/` repo (per
`NOTEBOOK_RULES.md` §6). Classes stay in play from Week 14 onward
(`nn.Module` is one); keep them as plain as the lesson's.

Paper first: Xavier (forward + fan-avg), He (where the 2 comes from), and
batchnorm's scale-invariance of the loss. File the pages in this folder.

## E1 — Init sweep

On a 5-layer tanh MLP, initialize three copies — $\mathcal{N}(0,1)$,
Xavier, He — and plot per-layer activation histograms at step 0 (same
x-range, one panel per layer, three rows). Do not train.
Hint: lesson §2; `nn.init.xavier_normal_` and `nn.init.kaiming_normal_`
(the latter is He; using it on tanh is the "expected wrong" row). Forward
one batch; histogram `tanh(z)` per `nn.Linear`.
Accept when: plots reproduce the saturate / stable / expected-too-wide
pattern and each row gets a one-line reading.

## E2 — Dead ReLU counter

On a ReLU MLP, train for one full epoch at each learning rate in the
setup's sweep and record the fraction of hidden units whose activation was
exactly 0 on every example. Plot that death rate vs LR.
Hint: lesson §4.3; a unit is dead if `(a == 0).all(dim=0)` over the epoch's
concatenated activations (or a running `alive` mask). The mechanism is one
large step pushing $z<0$ permanently, after which $g'=0$.
Accept when: the plot shows the death rate rising with LR and the
mechanism is stated in one line.

## E3 — Diagnose broken run A

Setup provides a training script whose loss goes NaN (or explodes) within
a few steps: silent `lr=10` with gradient explosion. Name the diagnosis,
apply a one-line fix, and confirm with the loss curve after the fix.
Hint: lesson §4.3 "Exploding gradients"; check update-to-weight ratios
before you change anything — they will not be $\sim 10^{-3}$. Lowering the
LR is the fix; clipping is a seatbelt, not a substitute.
Accept when: named diagnosis + one-line fix, confirmed by the loss curve
after the fix.

## E4 — Diagnose broken run B

Setup provides a second script: unscaled inputs + bad init → saturated
sigmoids (or tanh). Same criterion as E3.
Hint: lesson §4.3 "Saturated tanh"; histograms piled at the rails and
early-layer gradient norms shrinking are the signature. Xavier plus
scaling the inputs (or switching the activation) are distinct fixes —
pick one and show it.
Accept when: named diagnosis + one-line fix, confirmed by the loss curve
after the fix.

## E5 — Diagnose broken run C

Setup provides a third script: `zero_grad()` missing. Same criterion as
E3 — this is the trap Week 15's review asked about.
Hint: `.grad` accumulates (Week 14 `+=`); gradient norms grow roughly
linearly with step at a modest LR, and nothing raises. The fix is one
call, in the right place in the loop.
Accept when: named diagnosis + one-line fix, confirmed by the loss curve
after the fix.

## E6 — Batchnorm vs LR range

Add batchnorm to the Week-15 character MLP (`BatchNorm1d` on the hidden
pre-activations). Sweep the same LR grid with and without BN; plot
validation loss (or the fraction of runs that train) vs LR for both.
Hint: lesson §5; `model.train()` while training, `model.eval()` at val —
forgetting either is its own silent bug. Do not BN the output logits.
Accept when: the plot shows BN widening the range of workable LRs.

## E7 — Mini-project: MLP vs BDT

This one goes in files. Work `project.md` in order: reuse the Capstone 1
MAGIC splits, train a PyTorch MLP (init / LR / normalization tuned, early
stopping), score it against the Capstone 1 BDT under the **identical CV
protocol**, and write the short comparison.
Hint: tabular data is BDT home turf (Week 11); a loss is a finding.
Scale the Hillas features; the E1–E6 diagnostics belong in the writeup
as evidence the net *trained*. Do not hunt for a win by giving the MLP
features the BDT was denied.
Accept when: AUC vs the BDT under the identical CV protocol, with a win
or a defensible why-not.

## Review

- (Week 13) Re-derive the backward recursion $\delta^{(l)}$ — cold. This is the month's flagship.
- (Week 08) Batchnorm standardizes to zero mean, unit variance. Which Week-08 moments
  are estimated on-the-fly, and why does small batch size make them noisy?
- (Week 11) Why do trees not care about feature scaling while your MLP badly does?
