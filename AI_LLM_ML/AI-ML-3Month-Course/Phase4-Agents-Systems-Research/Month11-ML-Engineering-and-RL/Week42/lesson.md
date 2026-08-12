# Week 42 — Performance Engineering

~5 hrs. Before starting you should be able to: write and time a PyTorch
training loop (Week 15), state what backprop stores between forward and
backward (Week 13), write attention's cost in sequence length (Week 25), and
say roughly what fp16 and int8 quantization are (Week 30). A GPU helps for the
exercises (local, Colab, or Kaggle) but every idea here can be understood — and
most code run — on CPU.

## 1. Why speed is a scientific problem

Beam time and GPU time are the same resource: scarce, scheduled, and easy to
waste invisibly. A training run that takes 10 hours instead of 2 doesn't just
cost 8 hours — it costs the four other experiments you'd have run in them. And
the waste is usually invisible: the run *finishes*, so nothing looks broken.

The discipline this week is: **measure, don't guess.** Almost everyone guesses
wrong about where their program's time goes, including people who wrote the
program. The tools — a mental model of the hardware, a profiler, and a handful
of standard levers — turn "it's slow" into "the GPU is idle 40% of each step
waiting for data, and here is the one-line fix."

## 2. What a GPU actually is, from "what is a core"

A **processor core** is a circuit that executes instructions one after another:
fetch an instruction ("add these two numbers"), execute it, move to the next.
Your laptop's **CPU** (central processing unit) has a handful of cores — say 8
— each individually very fast and very flexible: good at running *different*
complicated things at once (your editor, the OS, a Python interpreter).

A **GPU** (graphics processing unit) makes the opposite bet. Instead of 8
clever cores, it has thousands of simple ones — a modern card has 5,000–15,000
— each slow and dumb alone, but all able to work at the same time. The catch:
they're organized to do the *same operation* on *different data* in lockstep.
That style is called **SIMT** (single instruction, multiple threads): one
instruction — "multiply and add" — issued to a whole group of threads at once,
each thread handling its own array element.

This bet was originally made for drawing graphics (every pixel gets the same
shading math) and happens to be exactly what neural networks need: a matrix
multiplication is millions of independent multiply-adds. That coincidence is
why deep learning runs on repurposed gaming hardware.

The organizational units, from small to large:

- A **thread** computes one element (one entry of an output matrix, say).
- Threads are grouped into **warps** of 32 that execute in strict lockstep —
  all 32 run the same instruction each cycle. (If your code makes half a warp
  take an `if` branch and half the `else`, the hardware runs both branches
  serially and masks lanes off — branchy code wastes a GPU.)
- Warps live on a **streaming multiprocessor (SM)** — the GPU's real "core",
  a block containing execution units, a chunk of fast local memory, and a
  scheduler juggling many warps at once. A mid-range GPU has 30–60 SMs; a
  data-center GPU has over 100.
- A **kernel** is one GPU program launched over many threads — "one matmul"
  or "one elementwise add" is typically one kernel launch. Your PyTorch line
  `c = a @ b` becomes a kernel launch under the hood.

Memory is a hierarchy, and this is where performance actually lives:

- **HBM / GPU DRAM** — the "16 GB" on the spec sheet. Big, but relatively
  slow to reach: hundreds of GB/s up to a few TB/s of **bandwidth** (bytes
  deliverable per second).
- **On-chip SRAM** (caches and per-SM "shared memory") — tiny (megabytes
  total) but an order of magnitude faster.
- **Registers** — per-thread scratch space, fastest of all.

Two spec-sheet numbers summarize a GPU:

- **FLOP/s**: floating-point operations per second the execution units can do
  (a **FLOP** is one add or multiply). Mid-range card: ~15 TFLOP/s at fp32,
  several times that on **tensor cores** — dedicated matrix-multiply units —
  at lower precision.
- **Memory bandwidth**: bytes/s HBM can deliver. Mid-range card: ~400 GB/s.

Notice the imbalance: 15e12 FLOP/s ÷ 400e9 B/s ≈ 37 FLOPs *per byte*. The
machine can compute ~37 operations in the time it takes to fetch one byte from
memory. Most operations don't have 37 FLOPs of work to do per byte — so most
operations spend their life **waiting on memory**, with the arithmetic units
idle. That single ratio explains most GPU performance behavior you will ever
see.

## 3. The roofline: memory-bound vs compute-bound

Formalize it. For any operation, define its **arithmetic intensity**:

$$ \text{AI} = \frac{\text{FLOPs performed}}{\text{bytes moved to/from memory}} $$

The runtime is limited by whichever budget runs out last:

$$ \text{time} \approx \max\!\left( \frac{\text{FLOPs}}{\text{FLOP/s}},\ \frac{\text{bytes}}{\text{bandwidth}} \right) $$

Plotting achievable FLOP/s against AI gives the **roofline**: a rising line
(bandwidth × AI) that caps at a flat roof (peak FLOP/s). The corner sits at
AI = peak FLOP/s ÷ bandwidth — our ~37 FLOPs/byte. Left of the corner an
operation is **memory-bound** (the memory bus is the bottleneck; extra compute
units can't help); right of it, **compute-bound** (the arithmetic is the
bottleneck; faster memory can't help).

Two examples, worked with 4-byte fp32 numbers:

**Elementwise add**, `c = a + b`, n elements: n FLOPs; bytes = read a, read b,
write c = 12n. AI = n / 12n = **1/12 ≈ 0.08**. Hundreds of times left of the
corner: hopelessly memory-bound. The GPU spends ~99.8% of the operation's time
moving data and ~0.2% adding. Every elementwise op — add, multiply, ReLU,
dropout mask, bias add — lives here.

**Matrix multiply**, C = AB with all matrices d×d: each of the d² outputs is a
dot product of length d ≈ 2d FLOPs → **2d³ FLOPs total**, but only 3d² × 4 =
12d² bytes of matrices. AI = 2d³ / 12d² = **d/6**. Intensity *grows with
size*: at d = 222, d/6 ≈ 37 — the corner. Big matmuls (d in the thousands)
are deep into compute-bound territory; small ones (d < ~200) are memory-bound
like everything else. This is why "make the batch bigger / fuse small ops into
big matmuls" is the universal GPU advice: it moves work rightward on the
roofline.

The deep-learning summary (Horace He's framing, from this week's reading):
your program's time is spent in one of three ways — **compute** (the matmuls;
only these use the expensive units well), **memory** (all the elementwise ops
between matmuls), and **overhead** (Python and kernel-launch bookkeeping when
the ops are too small to hide it). Every optimization this week attacks one of
the three. Identifying *which one is yours* is the profiler's job.

## 4. Profiling: finding where the time goes

One trap before any measurement: GPU work is **asynchronous**. When Python
executes `c = a @ b`, it *queues* the kernel and returns immediately; the GPU
does the work later. So this is wrong:

```python
t0 = time.time()
c = a @ b
t1 = time.time()          # measures queueing time, not the matmul!
```

You must wait for the GPU to finish before reading the clock:

```python
torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
t1 = time.time()
```

Also always discard the first few iterations (**warmup**): first calls pay
one-time costs (memory allocator, kernel selection, compilation).

For whole training steps, use the **PyTorch profiler**, which records every
op on CPU and GPU with timestamps:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for step in range(10):
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

The table names the top ops by GPU time. The **trace** is richer: open
`trace.json` in a Chrome-family browser at `chrome://tracing` (or
`https://ui.perfetto.dev`) to see a timeline with one row per CPU thread and
GPU stream. The single most valuable thing to look for: **gaps in the GPU
row**. A GPU that is busy 100% of the step is limited by §3's rooflines; a GPU
with gaps is limited by whatever fills the gaps on the CPU rows — usually one
of:

- **Data loading**: the `DataLoader` can't produce batches fast enough.
  Levers: `num_workers=N` (N background processes preparing batches),
  `pin_memory=True` (page-locked host memory, enabling faster async
  host→GPU copies), bigger batches, cheaper transforms, pre-decoded data.
- **Host–device transfer**: shipping tensors CPU↔GPU mid-step. Move data to
  the GPU once, keep it there; never call `.item()` / `.cpu()` on things you
  don't need every step (each is a synchronization point).
- **Python overhead**: thousands of tiny kernel launches, each microseconds
  of bookkeeping around microseconds of work. Lever: fewer, bigger ops —
  vectorize, batch, or `torch.compile` (§6).

Profile → read the trace → name the bottleneck → apply one lever → **measure
again**. Never stack three optimizations and declare victory; you won't know
which one worked, and one may have done nothing (Week 41's tracker is a good
place for before/after numbers).

## 5. Mixed precision: fp16 and bf16

Week 30 introduced the idea that numbers in a computer have a size. Here's the
structure. A floating-point number is stored in binary scientific notation —
sign × mantissa × 2^exponent — in a fixed number of bits:

| format | bits | sign | exponent | mantissa | max value | decimal digits |
|--------|------|------|----------|----------|-----------|----------------|
| fp32   | 32   | 1    | 8        | 23       | ~3×10³⁸   | ~7             |
| fp16   | 16   | 1    | 5        | 10       | 65504     | ~3             |
| bf16   | 16   | 1    | 8        | 7        | ~3×10³⁸   | ~2             |

The **exponent bits set the range** (how big/small a number can get); the
**mantissa bits set the precision** (how many significant digits). fp16 spent
its bits on precision, so its range is tiny: anything past 65504 becomes `inf`
(**overflow**), and small gradients (below ~6×10⁻⁵ for normal numbers) round
to zero (**underflow**). bf16 ("brain float") keeps fp32's 8 exponent bits and
sacrifices mantissa instead: same range, less precision.

Why bother with 16 bits? Three multiplicative wins: half the memory (bigger
batches/models fit), half the bytes over the memory bus (every memory-bound op
in §3 gets ~2× faster), and tensor cores run matmuls at several times their
fp32 rate at 16-bit precision. Typical end-to-end training speedups: 1.5–3×.

**Mixed precision** means: do the bulk work (matmuls, convolutions) in 16-bit,
keep the numerically fragile parts (loss, softmax reductions, the optimizer's
master copy of the weights) in fp32. PyTorch automates the choice per op with
**autocast**:

```python
scaler = torch.amp.GradScaler()                       # fp16 only
for x, y in loader:
    opt.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

The **GradScaler** exists because of fp16's underflow: real gradient values
are often ~10⁻⁵–10⁻⁷, which fp16 flushes to zero. The scaler multiplies the
loss by a large factor (say 2¹⁶) before backward — by the chain rule
(Week 05) every gradient is scaled by the same factor, lifting them into
fp16's representable range — then divides it back out before the optimizer
step, and skips steps where `inf`/`nan` shows up while it re-tunes the factor.

With **bf16** the range problem doesn't exist, so no scaler is needed:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward()
opt.step()
```

Rule of thumb: on hardware that supports bf16 (most GPUs since ~2020), use
bf16 and skip the scaler; use fp16+GradScaler on older cards. Either way,
**verify the final validation metric is unchanged within noise** — speed that
changes the science isn't an optimization, it's a bug.

## 6. torch.compile: fusing the small stuff

Eager PyTorch runs your model one op at a time: each line launches its own
kernel, and each elementwise op does a full round-trip through HBM — write the
result, read it back for the next op. For the memory-bound middle of a network
(bias add → activation → dropout → norm …) this is maximally wasteful.

A **compiler** here is a program that reads your model's sequence of ops as a
whole (a **graph**), optimizes it, and emits fewer, better kernels. The
headline optimization is **fusion**: merge chains of elementwise ops into one
kernel that reads inputs from HBM once, does all the arithmetic in registers,
and writes once. Three memory round-trips become one — a direct attack on the
memory-bound region of the roofline.

Using it is one line:

```python
model = torch.compile(model)
```

The first call (and the first call at each new input shape) is slow — seconds
to minutes — because compilation happens then. So benchmark **steady state**:
run several iterations first, then time. Typical wins are 1.2–2× on top of
mixed precision, biggest on models with lots of small ops between matmuls.
When it fails or falls back (dynamic shapes, exotic ops), it usually degrades
gracefully to eager speed — measure, as always.

## 7. Training on many GPUs: DDP concepts

One GPU eventually isn't enough — the dataset is huge or the model is. The
standard first tool is **DistributedDataParallel (DDP)**, and the concept
matters more than the launcher incantations:

- Run N processes, each with a full **replica** of the model on its own GPU.
- Each step, each replica gets a *different* shard of the batch and computes
  forward + backward locally.
- Before the optimizer step, the replicas **all-reduce** their gradients: a
  collective operation where every participant ends up with the sum (here,
  mean) of everyone's tensors. In efficient implementations (ring all-reduce)
  each GPU sends/receives only ~2× the gradient bytes regardless of N.
- Every replica then applies the identical averaged gradient, so weights stay
  bit-identical across replicas forever.

The mathematics: averaging gradients over N shards of size B is exactly the
gradient of one batch of size N×B. DDP is *large-batch SGD*, nothing more —
which is why LR often needs retuning when you scale out (bigger effective
batch, Week 16's dynamics).

The efficiency trick that makes DDP fast: **overlap**. Gradients for the last
layers are ready at the *start* of backward (backprop runs output → input,
Week 13), so DDP begins all-reducing those buckets while earlier layers are
still computing. Communication hides behind computation; scaling stays near
linear until the network can't keep up.

DDP requires the whole model to fit on one GPU. When it doesn't, you split
the *model* instead of the data: **model parallelism** (different layers on
different GPUs, a pipeline of stages) or **tensor parallelism** (single big
matmuls split across GPUs, each holding a slice of each weight matrix — how
frontier LLMs train). File the taxonomy: data parallel = replicate model,
split data; model/tensor parallel = split model. Combinations of all three
are standard at scale.

## 8. Inference: quantization, KV cache, batching

Training efficiency is about throughput; **inference** (serving a trained
model, Week 44's topic) is about latency, memory, and cost. Three standard
levers.

**Quantization** (Week 30, now from the systems side): store weights as 8- or
4-bit integers plus a scale factor instead of 16/32-bit floats. 2–4× less
memory and less bandwidth — and since single-request inference is heavily
memory-bound (see below), less bandwidth is directly less latency. Accuracy
usually drops little at int8, more at int4; the table of size/latency/metric
is the deliverable, never the assumption.

**The KV cache.** Recall Week 25: attention at position t computes queries,
keys, values, and each token attends to all previous tokens. Generation is
autoregressive — one token at a time. Done naively, generating token t means
re-running the full forward pass over all t tokens so far: the keys and values
for tokens 1..t−1 get recomputed *from scratch for every new token*. Total
work for T tokens scales like 1+2+⋯+T ≈ T²/2 forward-pass-equivalents of
redundant compute.

The fix is embarrassingly simple: keys and values for past tokens never
change, so **cache them**. Each step, compute Q/K/V for the one new token
only, append K and V to the cache, and attend against the cache. Per-token
compute drops from "forward pass over t tokens" to "forward pass over 1 token
plus t dot products" — per-token time goes from growing linearly (quadratic
total) to nearly flat.

The price is memory, and you can now do the arithmetic. Per token, per layer,
the cache stores one K and one V vector, each of size d_model:

$$ \text{cache bytes} = 2 \times n_\text{layers} \times d_\text{model} \times \text{bytes/value} \times n_\text{tokens} $$

- GPT-2-small scale (12 layers, d=768, fp16): 2×12×768×2 = 36 KB/token →
  a 1024-token context costs ~37 MB. Trivial.
- 7B-scale (32 layers, d=4096, fp16): 2×32×4096×2 = 512 KB/token → a
  4096-token context costs **2 GB per sequence**. Serve 8 users concurrently
  and the cache outweighs the 14 GB of weights.

This is why long context windows are expensive, why serving frameworks obsess
over cache memory (vLLM's PagedAttention manages cache in small pages, like
an operating system manages RAM, so fragments aren't wasted), and why
"tokens/sec" quotes always depend on batch size and context length.

**Batching.** With a cache, each generation step reads *all the weights* to
produce *one token* per sequence — arithmetic intensity ≈ 1 FLOP per weight
byte... deeply memory-bound. Serving B sequences at once reuses each weight
fetch B times: B× the throughput for nearly the same latency, until compute
saturates. **Static batching** waits to fill a batch and holds it until every
sequence finishes (short requests wait on long ones). **Continuous batching**
admits and retires sequences every step, keeping the batch full — the standard
in modern LLM servers and most of their throughput advantage.

## 9. Worked example: predict, then measure

The full loop of this week's method on the smallest possible case: use the
roofline to *predict* how two operations will behave, then benchmark to check.
Runs on any CUDA GPU (Colab's free tier is fine); on CPU the code runs but the
prediction below is GPU-specific.

Prediction, from §3: a 4096×4096 fp32 matmul has AI = d/6 ≈ 683 ≫ 37 →
compute-bound, time ≈ 2d³/peak-FLOP/s. An elementwise add on the same-size
tensors has AI = 1/12 ≪ 37 → memory-bound, time ≈ 12n bytes/bandwidth. On a
~15 TFLOP/s, ~400 GB/s card: matmul ≈ 2·4096³/15e12 ≈ 9.2 ms; add ≈
12·4096²/400e9 ≈ 0.5 µs — and the add's *achieved* FLOP/s will be pathetic
(~0.03 TFLOP/s) while the matmul's approaches the peak.

```python
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
d = 4096
a = torch.randn(d, d, device=device)
b = torch.randn(d, d, device=device)

def bench(fn, n_warmup, n_timed):
    for i in range(n_warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n_timed):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n_timed

def do_matmul():
    return a @ b

def do_add():
    return a + b

t_mm = bench(do_matmul, 5, 20)
t_add = bench(do_add, 5, 20)

flops_mm = 2 * d**3
bytes_add = 3 * d * d * 4
print(f"matmul: {t_mm*1e3:8.3f} ms   achieved {flops_mm/t_mm/1e12:6.2f} TFLOP/s")
print(f"add   : {t_add*1e6:8.1f} us   achieved {bytes_add/t_add/1e9:6.1f} GB/s")
print(f"matmul does {flops_mm/(d*d):.0f}x the FLOPs of add "
      f"but takes only {t_mm/t_add:.0f}x as long")
```

Read the output against the prediction: the matmul should achieve a healthy
fraction of your GPU's peak TFLOP/s (look up the spec); the add should achieve
a healthy fraction of the *bandwidth* spec while its TFLOP/s is a rounding
error. The last line is the roofline in one sentence: the matmul does ~8000×
more arithmetic for a far smaller time ratio, because one op pays for compute
and the other pays for memory. When prediction and measurement agree, you own
the model; when they disagree, the profiler (§4) tells you which of compute /
memory / overhead you mispredicted — that disagreement is where all the
learning is.

## Check yourself

1. A GPU has thousands of cores but your branchy per-element Python-style
   code runs terribly on it. Which two architectural facts explain this?
2. Compute the arithmetic intensity of ReLU applied to n fp32 elements
   (read x, write y, 1 FLOP each). Memory- or compute-bound at 37 FLOPs/byte?
3. Why must you call `torch.cuda.synchronize()` before reading the clock?
4. fp16 and bf16 are both 16 bits. What did each spend its bits on, and which
   consequence forces GradScaler for fp16 training?
5. What does kernel fusion save, concretely, for a chain of three elementwise
   ops — and which roofline regime does that help?
6. In DDP, why can gradient communication overlap with the backward pass
   instead of waiting for it to finish?
7. For a 24-layer model with d_model = 2048 serving fp16, how many bytes of
   KV cache per token? Per 2048-token sequence?
8. Why does batching multiply throughput for cached LLM generation, in
   roofline terms?

## Answers

1. SIMT lockstep — warps of 32 threads execute the same instruction, so
   divergent branches serialize; and the FLOP/byte imbalance — scattered,
   low-intensity memory access leaves the arithmetic units starved.
2. AI = n FLOPs / 8n bytes = 0.125. Far below 37 → memory-bound (like every
   elementwise op).
3. Kernel launches are asynchronous: Python queues work and returns
   immediately, so without a sync you time the queueing, not the computation.
4. fp16: 10 mantissa bits (precision) but 5 exponent bits (range to 65504 and
   underflow near 6×10⁻⁵) — small gradients flush to zero, so the loss is
   scaled up to lift them into range. bf16: 8 exponent bits (fp32's range)
   with 7 mantissa bits — no underflow problem, no scaler.
5. Two full HBM round-trips: instead of write+read between op 1→2 and op 2→3,
   the fused kernel reads inputs once, computes in registers, writes once.
   It helps the memory-bound regime — bytes moved drop ~3×, FLOPs unchanged.
6. Backprop produces gradients output-layer-first (Week 13), so the last
   layers' gradients are final while earlier layers are still computing;
   DDP all-reduces those finished buckets concurrently with the remaining
   backward computation.
7. 2 × 24 × 2048 × 2 = 196,608 B ≈ 192 KB/token; × 2048 tokens ≈ 384 MB per
   sequence.
8. Each generation step is memory-bound on reading the weights (~1 FLOP per
   weight byte for one token). A batch of B reuses each weight fetch for B
   tokens — B× the FLOPs per byte moved — climbing the roofline toward the
   compute corner at nearly constant step time.

## New terms

- **CPU core / GPU core** — instruction-executing circuit; CPUs have few fast
  flexible ones, GPUs thousands of simple lockstep ones.
- **SIMT** — single instruction, multiple threads: one instruction issued to
  many threads over different data.
- **thread / warp / SM** — one element's worker; a lockstep group of 32; the
  GPU's core-like block that schedules many warps.
- **kernel** — one GPU program launch (e.g. one matmul).
- **HBM / bandwidth** — the GPU's main memory / bytes-per-second it delivers.
- **FLOP, FLOP/s** — one floating-point add/multiply; how many per second.
- **tensor cores** — dedicated matrix-multiply units, fastest at 16-bit.
- **arithmetic intensity** — FLOPs per byte moved; an operation's position on
  the roofline.
- **roofline model** — achievable FLOP/s vs intensity: bandwidth-sloped line
  capped by peak compute; memory-bound left of the corner, compute-bound
  right.
- **profiler / trace** — tool recording per-op CPU/GPU timing; its timeline
  view.
- **warmup** — discarded first iterations that pay one-time costs.
- **pin_memory / num_workers** — DataLoader levers: page-locked host memory
  for fast async copies; background batch-preparing processes.
- **fp32 / fp16 / bf16** — float formats; exponent bits set range, mantissa
  bits set precision.
- **overflow / underflow** — value too large for the format (→ inf) / too
  small (→ 0).
- **mixed precision / autocast / GradScaler** — 16-bit bulk math with fp32
  fragile parts; PyTorch's per-op automation; fp16's loss-scaling workaround
  for gradient underflow.
- **torch.compile / fusion / graph** — PyTorch's compiler; merging ops into
  one kernel to skip memory round-trips; the whole-model op sequence it
  optimizes.
- **DDP / replica / all-reduce / overlap** — data-parallel training; a full
  model copy per GPU; the collective that averages gradients everywhere;
  communicating finished gradient buckets during backward.
- **model / tensor parallelism** — splitting the model by layers / splitting
  individual weight matrices across GPUs.
- **KV cache** — stored keys/values of past tokens so generation computes
  each token once; memory grows linearly in context length.
- **static / continuous batching** — fixed batch held to completion vs
  admitting/retiring sequences every step.
- **PagedAttention** — vLLM's page-based KV-cache memory management.

## Going deeper

- Horace He, *Making Deep Learning Go Brrrr From First Principles* — the
  compute/memory/overhead trichotomy this lesson leans on, with better jokes.
- PyTorch docs: performance tuning guide, profiler recipe, and the automatic
  mixed precision (AMP) pages — the current APIs for §4–5.
- PyTorch docs: `torch.compile` introduction and the DDP overview/notes —
  concepts to launcher details, in that order.
- Any roofline-model introduction (the original Williams–Waterman–Patterson
  paper is readable) — the general form of §3.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* (the vLLM paper) — abstract + figures for continuous
  batching and cache paging.
