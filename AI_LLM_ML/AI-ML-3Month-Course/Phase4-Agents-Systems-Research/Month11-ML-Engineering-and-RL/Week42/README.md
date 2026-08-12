# Week 42 — Performance Engineering

Beam time and GPU time are the same resource: scarce, scheduled, and wasted by the unprofiled — this week you learn where the cycles actually go.

## Objectives

- Describe GPU architecture at working depth: SMs, warps, HBM bandwidth vs. FLOPs, and place a kernel on a roofline plot (memory-bound vs. compute-bound).
- Profile a real training loop with the PyTorch profiler and attribute wall time to data loading, host-device transfer, kernels, and Python overhead.
- Apply mixed precision (fp16/bf16 autocast + GradScaler where needed) and `torch.compile`, and measure — not assume — the speedup.
- Explain DistributedDataParallel conceptually: replicas, gradient all-reduce, overlap of communication with backward; contrast with model/tensor parallelism.
- Explain the main inference levers: quantization (int8/int4), KV-cache and why it makes attention linear per generated token, and batching (static vs. continuous).

## Core material (~3 hrs)

- PyTorch documentation: performance tuning guide, profiler recipe, and the automatic mixed precision (AMP) docs.
- PyTorch docs: `torch.compile` introduction and the DDP overview/notes page (concepts, not launcher details).
- A GPU-architecture explainer of your choice — e.g. the "Making Deep Learning Go Brrrr From First Principles" essay (Horace He) for the memory-vs-compute framing, plus any roofline-model introduction.
- Skim: a KV-cache explainer and the vLLM paper's continuous-batching idea (PagedAttention) at abstract level, by title.

## Exercises (built when the week starts)

1. Arithmetic intensity on paper-then-notebook: compute FLOPs and bytes for a d×d matmul and for elementwise add; predict which is memory-bound on your GPU's specs. Accept when: benchmark timings agree with the prediction's ordering and a one-line roofline argument is written.
2. Profile: run the PyTorch profiler on your Week-41 training loop; produce a trace and a top-10 ops table. Accept when: written answer names the single largest time sink and whether the GPU is idle waiting on data.
3. Fix the input pipeline: tune `num_workers` / `pin_memory` / batch size guided by the profile. Accept when: measured samples/sec improves ≥20% over exercise 2's baseline or a measurement shows the pipeline was already saturated.
4. Mixed precision: train with bf16 (or fp16+GradScaler); compare speed, memory, and final val metric. Accept when: table shows throughput and peak memory for fp32 vs. mixed, with val metric within noise.
5. `torch.compile`: compile the model, measure steady-state step time (excluding warmup). Accept when: table reports compiled vs. eager step time and notes compilation overhead.
6. KV-cache by hand: for your Week-27 nanoGPT-class model, generate 200 tokens with and without caching K/V; verify identical outputs. Accept when: outputs match token-for-token and measured per-token time is roughly flat with cache vs. growing without.
7. Quantize: dynamic-quantize (or int8-load) one model and measure size and CPU inference latency vs. accuracy. Accept when: table reports size, latency, metric for fp32 vs. quantized.

## Deliverable

`week42/PERF.md` — the before/after tables for exercises 2–7 with one paragraph per optimization saying why it did or didn't help, plus saved profiler traces.

## Review

1. Week 13: in backprop for a linear layer, which stored tensors force memory use, and why does that motivate activation checkpointing?
2. Week 25: what are the time and memory complexities of attention in sequence length, and which one does KV-caching change at generation time?
3. Week 5: matrix multiply is O(n³) FLOPs on O(n²) data. Use that ratio to explain why big matmuls are compute-bound and small ones aren't.
4. Week 28: which quantity do scaling laws trade against compute, and why does that make training FLOPs a budget question?
5. Week 17: why are convolutions more parameter-efficient than dense layers, and does that make them faster per FLOP? Why not necessarily?
