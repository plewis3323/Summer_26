# Week 42 — Performance Engineering

Beam time and GPU time are the same resource: scarce, scheduled, and wasted by the unprofiled — this week you learn where the cycles actually go.

## Objectives

- Describe GPU architecture at working depth: SMs, warps, HBM bandwidth vs. FLOPs, and place a kernel on a roofline plot (memory-bound vs. compute-bound).
- Profile a real training loop with the PyTorch profiler and attribute wall time to data loading, host-device transfer, kernels, and Python overhead.
- Apply mixed precision (fp16/bf16 autocast + GradScaler where needed) and `torch.compile`, and measure — not assume — the speedup.
- Explain DistributedDataParallel conceptually: replicas, gradient all-reduce, overlap of communication with backward; contrast with model/tensor parallelism.
- Explain the main inference levers: quantization (int8/int4), KV-cache and why it makes attention linear per generated token, and batching (static vs. continuous).

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text: GPU architecture from "what is a core" up, the roofline model with worked numbers, profiling method, mixed precision (fp16/bf16 bit layouts), `torch.compile`, DDP concepts, and KV-cache arithmetic.
- PyTorch documentation: performance tuning guide, profiler recipe, and the automatic mixed precision (AMP) docs.
- PyTorch docs: `torch.compile` introduction and the DDP overview/notes page (concepts, not launcher details).
- Horace He, "Making Deep Learning Go Brrrr From First Principles" — the memory-vs-compute framing; plus any roofline-model introduction.
- Skim: a KV-cache explainer and the vLLM paper's continuous-batching idea (PagedAttention) at abstract level, by title.

## Exercises

See `exercises.md` (notebook generated from it when the week starts, per `NOTEBOOK_RULES.md`). Seven exercises: roofline prediction vs. benchmark, profiling the Week-41 loop, input-pipeline tuning, mixed precision, `torch.compile`, a hand-built KV cache for the Week-27 model, and quantization — each producing a before/after table for `PERF.md`.

## Deliverable

`week42/PERF.md` — the before/after tables for the profiling and optimization exercises, with one paragraph per optimization saying why it did or didn't help, plus saved profiler traces.

## Review

1. Week 13: in backprop for a linear layer, which stored tensors force memory use, and why does that motivate activation checkpointing?
2. Week 25: what are the time and memory complexities of attention in sequence length, and which one does KV-caching change at generation time?
3. Week 5: matrix multiply is O(n³) FLOPs on O(n²) data. Use that ratio to explain why big matmuls are compute-bound and small ones aren't.
4. Week 28: which quantity do scaling laws trade against compute, and why does that make training FLOPs a budget question?
5. Week 17: why are convolutions more parameter-efficient than dense layers, and does that make them faster per FLOP? Why not necessarily?
