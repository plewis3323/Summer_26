# Week 42 — Exercises

Work top to bottom. Setup (imports, device detection, model/data loading,
timing helper with warmup + synchronize) is given by the notebook; you write
only the lines each exercise asks for. E2–E5 operate on your Week-41 project's
training loop; results for E2–E7 also go into `week42/PERF.md` (a plain
markdown file of before/after tables — that file is the deliverable). A GPU
(local or Colab/Kaggle) is needed for E2–E5; E1, E6, E7 run anywhere.

## E1 — Arithmetic intensity, paper then benchmark

On paper: FLOPs, bytes, and arithmetic intensity for (a) a d×d fp32 matmul
and (b) elementwise add of d×d tensors. Look up (or estimate) your GPU's peak
FLOP/s and bandwidth, find the roofline corner, and predict which operation is
memory-bound at d = 4096. Then benchmark both and compute achieved TFLOP/s
and GB/s.
Hint: matmul 2d³ FLOPs over 12d² bytes; add n FLOPs over 12n bytes; corner =
peak FLOP/s ÷ bandwidth.
Accept when: measured ordering (achieved TFLOP/s high for matmul, GB/s high
for add) matches the prediction, and a one-line roofline argument is written.

## E2 — Profile the training loop

Run the PyTorch profiler (CPU + CUDA activities) over ~10 steps of your
Week-41 training loop. Produce the top-10 ops table sorted by CUDA time and
export a chrome trace.
Hint: look at the GPU row of the trace first — busy solid, or gappy?
Accept when: a written answer names the single largest time sink and states
whether the GPU sits idle waiting on data (with the trace region that shows
it).

## E3 — Fix the input pipeline

Guided by E2's trace, tune `num_workers`, `pin_memory`, and batch size.
Measure samples/sec before and after each change (one change at a time).
Hint: if E2 showed the GPU 100% busy, the pipeline is already saturated —
proving that with a measurement is a full pass on this exercise.
Accept when: measured samples/sec improves ≥20% over E2's baseline, or a
measurement demonstrates the pipeline was already saturated.

## E4 — Mixed precision

Train with bf16 autocast (or fp16 + GradScaler if your GPU lacks bf16).
Record throughput (samples/sec), peak GPU memory
(`torch.cuda.max_memory_allocated()`), and final val metric for fp32 vs
mixed.
Hint: reset the peak-memory counter between runs; same seed, same epochs.
Accept when: the table shows throughput and peak memory for both, with the
val metric within run-to-run noise.

## E5 — torch.compile

Compile the model and measure steady-state step time, excluding warmup, vs
eager. Note the first-step compilation overhead separately.
Hint: time steps 20–50, not 0–30; recompilation triggers on new input shapes.
Accept when: the table reports compiled vs eager steady-state step time and
the one-time compilation cost.

## E6 — KV cache by hand

For your Week-27 nanoGPT-class model, generate 200 tokens greedily twice:
(a) recomputing the full forward pass over the whole sequence each token, and
(b) caching K/V per layer and feeding only the new token. Verify identical
outputs; record per-token time vs position for both.
Hint: with the cache, causal masking is free — the new token may attend to
everything cached. Watch the positional-embedding index for the single token.
Accept when: outputs match token-for-token, and the plot shows per-token time
roughly flat with cache vs growing without.

## E7 — Quantize

Dynamic-quantize (or int8-load) one of your models. Measure file size, CPU
inference latency, and the val metric, vs fp32.
Hint: `torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear},
dtype=torch.qint8)` covers Linear-heavy models; batch=1 latency is the honest
serving number.
Accept when: the table reports size, latency, and metric for fp32 vs
quantized.

## Review

1. Week 13: in backprop for a linear layer, which stored tensors force memory
   use, and why does that motivate activation checkpointing?
2. Week 25: what are the time and memory complexities of attention in
   sequence length, and which one does KV-caching change at generation time?
3. Week 05: matrix multiply is O(n³) FLOPs on O(n²) data. Use that ratio to
   explain why big matmuls are compute-bound and small ones aren't.
4. Week 28: which quantity do scaling laws trade against compute, and why
   does that make training FLOPs a budget question?
5. Week 17: why are convolutions more parameter-efficient than dense layers,
   and does that make them faster per FLOP? Why not necessarily?
