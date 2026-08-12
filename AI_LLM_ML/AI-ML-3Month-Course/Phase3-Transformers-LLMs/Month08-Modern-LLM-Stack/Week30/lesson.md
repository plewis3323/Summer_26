# Week 30 — Fine-Tuning: Full FT, LoRA, QLoRA

~10 hrs. Before starting you should be able to: load an open-weight model and reason
about its memory footprint (Week 29); name every weight matrix in a transformer block
(Week 26); state what rank means and what the SVD tells you about a matrix (Weeks
06–07); run a supervised training loop in PyTorch and fine-tune a pretrained network
(Weeks 15, 18).

A pretrained LLM knows language; it doesn't know your task. **Fine-tuning** continues
training on your data so it does. This week answers three questions with arithmetic
rather than folklore: why full fine-tuning is memory-brutal, why a low-rank update
(LoRA) captures most of the benefit at a tiny fraction of the cost, and how
quantization (QLoRA) shrinks the frozen base model itself. The quantization story
requires knowing what a floating-point number actually is, so we build that from the
bits up.

## 1. Full fine-tuning and its memory bill

Fine-tuning is Week 18's transfer learning applied to a language model: initialize at
the pretrained weights, train on task data with a small learning rate. Nothing new
conceptually. What is new is the bill. Training with Adam (Week 08) must hold, per
parameter:

| item | dtype (mixed precision) | bytes |
|---|---|---|
| weight | bf16 | 2 |
| gradient | bf16 | 2 |
| master copy of weight | fp32 | 4 |
| Adam first moment $m$ | fp32 | 4 |
| Adam second moment $v$ | fp32 | 4 |
| **total** | | **16** |

(The fp32 master copy exists because optimizer updates are tiny; applied directly in
bf16 they round to zero — Week 16's story about small numbers vanishing, now at the
storage level. §2 makes this precise.)

So full fine-tuning a 1B model costs roughly $10^9 \times 16 = 16$ GB *before*
activations, which grow with batch size and sequence length. A 7B model needs ~112 GB
— multiple GPUs for a task that might only need to nudge the model slightly. That gap
between "cost of training machinery" and "size of the change we actually need" is the
opening LoRA walks through. But first, the bits.

## 2. What a floating-point number actually is

A computer stores numbers in fixed-width chunks of **bits** (binary digits, 0 or 1).
With 32 bits you get $2^{32} \approx 4.3$ billion distinct patterns — so any 32-bit
format can represent at most that many distinct numbers, and the design question is
*which* numbers.

**Fixed point** would space them evenly, like a ruler. But model weights, gradients,
and activations span wildly different magnitudes ($10^{-8}$ to $10^{3}$ in one
network), and a ruler fine enough for $10^{-8}$ can't reach $10^3$ with 32 bits.
**Floating point** solves this the way scientific notation does: store a number as

$$x = (-1)^{s} \times 1.f \times 2^{\,e - \text{bias}},$$

with three fields: a **sign bit** $s$; an **exponent** $e$ (sets the magnitude, the
"$\times 10^n$" of scientific notation, but base 2); and a **mantissa** (or fraction)
$f$ (the significant digits after the leading 1, which is implicit and free).

**A worked encoding.** Take $x = 5.75$. In binary, $5.75 = 4 + 1 + 0.5 + 0.25 =
101.11_2$. Normalize: $101.11_2 = 1.0111_2 \times 2^2$. So the mantissa bits are
$0111$ (the leading 1 is implicit) and the exponent must encode 2. In float32 the
exponent field is 8 bits with bias 127, so $e = 2 + 127 = 129 = 10000001_2$. Sign 0.
Done — that bit pattern *is* 5.75.

The three formats you met in Week 29, now with their anatomy:

| format | sign | exponent bits | mantissa bits | max value | decimal digits of precision |
|---|---|---|---|---|---|
| float32 | 1 | 8 | 23 | $\sim 3.4\times10^{38}$ | ~7 |
| float16 | 1 | 5 | 10 | 65504 | ~3 |
| bfloat16 | 1 | 8 | 7 | $\sim 3.4\times10^{38}$ | ~2 |

Two consequences worth engraving:

- **Exponent bits buy range; mantissa bits buy precision.** float16 spent its bits on
  precision and pays with range: 65504 is easy to overflow with an attention score or
  a gradient, giving `inf` → `nan` cascades. bfloat16 keeps float32's full range by
  keeping its 8 exponent bits and sacrifices precision instead — the trade deep
  learning wants, because training tolerates noise but not overflow. That is the whole
  reason bf16 exists and why it is the modern default.
- **Precision is relative, not absolute.** The representable numbers are dense near 0
  and sparse far away: gaps between consecutive float32 values near $1$ are about
  $10^{-7}$, near $10^6$ about $0.06$. This is why the fp32 master weights exist in
  §1's table: a bf16 weight near 1.0 has gaps of about $0.008$, and a gradient-descent
  update of $10^{-4}$ falls *between* representable values and rounds away to nothing.
  In fp32 it survives.

## 3. Quantization: int8 and nf4

Inference (and QLoRA's frozen base) doesn't need training's dynamic range. If the
weights just sit there being multiplied, we can store each in 8 or even 4 bits and
convert back on the fly. **Quantization** is that: map each float to a small integer
code plus shared scaling information, accepting a little rounding error for a 4–8×
memory win.

### 3.1 int8: absmax scaling, worked

The simplest scheme, **absmax quantization**, maps a block of floats onto the 255
integers $\{-127, \ldots, 127\}$:

$$\text{scale} = \frac{127}{\max_i |w_i|}, \qquad
q_i = \mathrm{round}(\text{scale} \cdot w_i), \qquad
\hat{w}_i = q_i / \text{scale}.$$

Worked example on a 4-weight block $w = (0.40,\ -0.90,\ 0.05,\ 0.31)$:

- absmax $= 0.90$, so scale $= 127/0.90 = 141.1$.
- scaled: $(56.4,\ -127.0,\ 7.06,\ 43.7)$ → rounded codes $q = (56, -127, 7, 44)$.
- dequantized: $\hat{w} = (0.397,\ -0.900,\ 0.0496,\ 0.312)$.

Largest error here: $0.003$ — under 1% of the weight scale. Each weight now costs 1
byte instead of 4, plus one shared fp scale per block.

The catch: **outliers**. One weight of $8.0$ in a block of $\sim0.3$'s drags the scale
down to $127/8 = 15.9$, and every ordinary weight gets rounded onto a handful of
integer levels — its precision is destroyed by a neighbor. Real LLM activations have
exactly such outlier features. Mitigations: quantize in small blocks so an outlier
poisons only its own block, use per-row/per-channel scales, or (LLM.int8(),
Dettmers et al. 2022) pull outlier dimensions out and process them in fp16.

### 3.2 nf4: quantile bins for Gaussian weights

Four bits give only $2^4 = 16$ levels — placement of those levels now matters
enormously. Int-style quantization spaces levels *uniformly*. But trained network
weights are empirically close to a Gaussian, $w \sim \mathcal{N}(0, \sigma^2)$: dense
near zero, thin in the tails. Uniform levels waste codes out where almost no weights
live and starve the crowded region near zero.

**NF4 (NormalFloat-4)**, from the QLoRA paper, places the 16 levels at
**equal-probability-mass quantiles** of the standard normal: each of the 16 bins is
sized so that a Gaussian weight is equally likely to land in any of them — narrow
bins packed near zero, wide bins in the tails. (Quantiles are Week 08: the $q$-th
quantile is the value below which a fraction $q$ of the probability mass lies.) Every
code is then used equally often, which is the information-theoretically efficient
choice for this source — the same reason (Week 08 again) a code is optimal when its
symbols are equiprobable.

In practice: weights are quantized in blocks of 64; each block stores its absmax as a
scale (so the block is normalized into $[-1, 1]$, matching the normalized quantile
levels, and an outlier only poisons its own 64 weights); and QLoRA quantizes those
per-block scales *themselves* to 8 bits (**double quantization**), shaving another
~0.4 bits per parameter. Net: a 1B-parameter base model in nf4 is ~0.55 GB instead of
2 GB in bf16.

You cannot *train* nf4 weights (16 levels can't express a $10^{-4}$ nudge — the update
rounds away, catastrophically now). QLoRA's resolution: don't train them. Freeze the
quantized base, train something small in bf16 on top. That something is LoRA.

## 4. LoRA: the low-rank update, derived

### 4.1 The ansatz

Fine-tuning changes each weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times
d_{\text{in}}}$ to $W' = W + \Delta W$. LoRA (Hu et al., 2021) is one falsifiable
claim: *the useful $\Delta W$ has low rank*. Rank (Week 06) is the number of
independent directions a matrix actually uses; a rank-$r$ matrix maps everything into
an $r$-dimensional subspace. The claim is motivated by the observation that
fine-tuning barely changes model behavior in most directions — adapting a broad
pretrained model to one narrow task is a small, structured correction, not a rewrite —
and by measurements (Aghajanyan et al., 2020) that fine-tuning succeeds even when
confined to surprisingly low-dimensional parameter subspaces. It is an ansatz in the
physics sense: assume the solution has a restricted form, solve cheaply within that
form, and *check* (your rank-sweep exercise is the check).

So parameterize the update as a product of two thin matrices:

$$\Delta W = \frac{\alpha}{r}\, B A, \qquad
B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad
A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad r \ll d,$$

which by construction has rank at most $r$ (Week 06: the product's rank can't exceed
either factor's). $\alpha$ is a fixed scaling hyperparameter; dividing by $r$
decouples the update's overall scale from the choice of rank, so you can sweep $r$
without retuning the learning rate.

### 4.2 The forward pass and the zero init

The adapted layer computes

$$h = W'x = Wx + \frac{\alpha}{r} B (A x).$$

Read the parenthesization: $Ax$ first squeezes $x \in \mathbb{R}^{d_{\text{in}}}$
down to $r$ numbers, then $B$ expands back to $d_{\text{out}}$ — one extra low-rank
bottleneck riding alongside the frozen $W$, at cost $r(d_{\text{in}} +
d_{\text{out}})$ multiplies instead of $d_{\text{in}} d_{\text{out}}$.

Initialization: $A$ is random (small Gaussian), $B = \mathbf{0}$. Then $BA =
\mathbf{0}$ and at step 0 the adapted model is *exactly* the base model — training
starts from the pretrained function, not from a randomly perturbed one, and the
learning rate alone controls how far it strays. (Why not $A = 0$ *and* $B = 0$?
Compute the gradients: $\partial \mathcal{L}/\partial B \propto (\cdot)\,A^\top$ and
$\partial \mathcal{L}/\partial A \propto B^\top(\cdot)$. With both zero, both
gradients vanish and the adapter never moves — the same symmetry-breaking argument as
Week 13's "why not init a network at all zeros". One random factor breaks it.)

After training you can **merge**: compute $W' = W + \frac{\alpha}{r}BA$ once and store
it as an ordinary dense matrix. Inference then costs exactly the base model — zero
added latency — which is LoRA's quiet killer feature. Or keep adapters separate: many
tasks, one shared 2 GB base, a few MB per task.

### 4.3 The parameter arithmetic

Trainable parameters per adapted matrix:

$$\underbrace{r \, (d_{\text{in}} + d_{\text{out}})}_{\text{LoRA}}
\quad\text{vs}\quad
\underbrace{d_{\text{in}} \, d_{\text{out}}}_{\text{full}}.$$

For a square $2048 \times 2048$ projection at $r = 8$:

- full: $2048 \times 2048 = 4{,}194{,}304$;
- LoRA: $8 \times (2048 + 2048) = 32{,}768$;
- ratio: $4{,}194{,}304 / 32{,}768 = 128$. LoRA trains **0.78%** of the matrix.

In general the ratio for a square $d \times d$ matrix is $d/2r$ — it *improves* as
models grow, which is why the trick matters more at 70B than at 1B.

Whole-model count, concretely. Take a Llama-3.2-1B-class model: $L = 16$ layers,
hidden size $d = 2048$, and adapt the four attention projections $W_Q, W_K, W_V, W_O$
at $r = 8$. Treating all four as $2048 \times 2048$:

$$16 \text{ layers} \times 4 \text{ matrices} \times 32{,}768
= 2{,}097{,}152 \approx 2.1\text{M trainable},$$

against $\sim 1.24$B total: **0.17%**. One wrinkle to expect when you verify this
with `peft` (exercise E2): modern models use grouped-query attention — several query
heads share each key/value head, so $W_K$ and $W_V$ are thin rectangles
($2048 \times 512$ here), not squares. LoRA on a $2048 \times 512$ matrix at $r=8$
costs $8 \times (512 + 2048) = 20{,}480$. Redo the sum: $16 \times (2 \times 32{,}768
+ 2 \times 20{,}480) = 1{,}703{,}936 \approx 1.7$M. Your hand count must match
`print_trainable_parameters` *exactly*, so read the model's config, not the folklore.

The memory bill collapses accordingly: §1's 16 bytes/param now applies only to ~2M
adapter parameters (~32 MB of training state), while the 1B frozen base sits in bf16
(2 GB) or nf4 (~0.55 GB) with no gradients and no optimizer states at all. That is
how a 7B fine-tune fits on one consumer GPU.

### 4.4 Why rank r works — and how you'd falsify it

If the "true" full-FT update $\Delta W^\ast$ were an arbitrary matrix, a rank-8
approximation would be hopeless. The empirical facts say otherwise: compute the SVD
(Week 07) of a trained full-FT $\Delta W^\ast$ and its singular values crash after the
first handful — most of the update's energy lives in a few directions, i.e. its
**effective rank** is tiny even though its nominal rank is full. Intuition: the
pretrained $W$ already contains the features; adapting to a task mostly *re-weights
and re-mixes* existing directions (amplify "this is physics notation", suppress
"casual chat register") rather than manufacturing thousands of new independent ones.
LoRA hard-codes that geometry into the parameterization.

This is falsifiable two ways, and you'll do both: (1) sweep $r$ — if the ansatz holds
for your task, validation loss saturates at small $r$ (the LoRA paper found $r = 1$–$4$
already competitive on many tasks); (2) SVD a trained adapter's $BA$ and look at the
singular-value decay. When the ansatz *fails* — tasks demanding genuinely new
knowledge or heavy re-structuring, e.g. a new language or heavy continued pretraining
— the sweep shows it: loss keeps improving with $r$, and full FT keeps a stubborn
edge. Low rank is a hypothesis about the task, not a law.

## 5. QLoRA: the assembled system

Now the pieces click together. **QLoRA** (Dettmers et al., 2023) is:

1. Base weights quantized to **nf4** (§3.2), block size 64, double-quantized scales —
   and *frozen*.
2. **LoRA adapters** in bf16 on top (§4) — the only trainable parameters.
3. Compute in bf16: each layer dequantizes its weight block on the fly, multiplies,
   moves on. Gradients flow *through* the frozen quantized weights into $A$ and $B$
   (backprop needs $\partial h/\partial x = W^\top$, which dequantization provides;
   it does not need $\partial \mathcal{L}/\partial W$, which is the part 4-bit
   storage couldn't survive).
4. **Paged optimizer states**: Adam's $m, v$ for the adapters can spill to CPU RAM
   when the GPU spikes, borrowing the OS's virtual-memory trick.

Cost of the compromise: dequantization makes each step somewhat slower than bf16
LoRA, and the quantized base is a slightly blurred copy of the original. The QLoRA
paper's headline result is that 4-bit-base fine-tuning matches 16-bit fine-tuning
quality on their benchmarks — you rent the memory savings nearly free.

## 6. SFT dataset design

**Supervised fine-tuning (SFT)** is fine-tuning on (prompt → desired response) pairs.
The mechanics are ordinary next-token training; the *dataset* is where projects
succeed or die. Four design axes:

- **Format.** Chat format (role-tagged messages rendered through the model's chat
  template — Week 29 §5) for instruct models; plain completion format for base
  models. The template used to *format training data* must match the one used at
  *inference* exactly. This is Week 29's footgun 2 wearing a new hat, and it is the
  most common silent SFT bug.
- **Loss masking.** You want the model to learn to produce *responses*, not to
  parrot prompts, so the loss is computed only on response tokens (prompt-token
  labels set to `-100`, PyTorch's ignore index). TRL's `SFTTrainer` does this for
  you if configured; verify it once by eye on a decoded batch.
- **Diversity beats bulk.** A few hundred varied, carefully-checked examples
  typically beat tens of thousands of near-duplicates; duplicates teach the model to
  memorize phrasing, and systematic label errors are learned *faithfully*. Budget
  your hours toward checking labels, not scraping more.
- **Contamination discipline.** Nothing from your evaluation split may appear in
  training — including near-duplicates (same abstract, trivially different
  whitespace). Dedup before you split, and split by document, never by row. This is
  Week 09's leakage discipline; the stakes return in Week 32.

A minimal, correct SFT pipeline with `peft` + TRL:

```python
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,                       # alpha/r = 2, a common default
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()            # check against your hand count

ds = load_dataset("json", data_files="sft_v0.jsonl", split="train")
# each row: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(output_dir="out", num_train_epochs=3,
                   per_device_train_batch_size=2, learning_rate=2e-4),
)
trainer.train()

merged = model.merge_and_unload()             # W + (alpha/r) B A, baked in
merged.save_pretrained("out/merged")
```

For QLoRA, add `quantization_config=BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` (from
`transformers`) to `from_pretrained` — the rest is unchanged. Note the LoRA learning
rate ($2 \times 10^{-4}$) is ~100× larger than a full-FT rate: you are training a
tiny bottleneck from $B=0$, not nudging a trillion-step-old pretrained matrix.

## 7. Worked example: LoRA by hand on one linear layer

Everything in §4, in twenty lines, on one layer — this is exercise E1's core.

```python
import torch

torch.manual_seed(30)
d_in, d_out, r, alpha = 64, 64, 4, 8

W = torch.randn(d_out, d_in) / d_in**0.5      # "pretrained", frozen
A = torch.randn(r, d_in) * 0.01               # random init
B = torch.zeros(d_out, r)                     # zero init
A.requires_grad_(True); B.requires_grad_(True)

x = torch.randn(32, d_in)

base = x @ W.T
adapted = x @ W.T + (alpha / r) * (x @ A.T) @ B.T
print(torch.allclose(base, adapted))          # True: B=0 => exactly the base model

# one training step toward a made-up target
target = torch.randn(32, d_out)
loss = ((adapted - target) ** 2).mean()
loss.backward()
print(A.grad.abs().max() > 0, B.grad.abs().max() > 0)  # both flow: A.grad via B? no —
# B.grad ∝ (...) @ (x @ A.T): nonzero since A is random.
# A.grad ∝ B.T @ (...): ZERO at the first step since B = 0. B moves first; then A.

with torch.no_grad():
    B -= 0.1 * B.grad
W_merged = W + (alpha / r) * B @ A
print(torch.allclose(x @ W_merged.T,
                     x @ W.T + (alpha / r) * (x @ A.T) @ B.T, atol=1e-6))
```

Run it and watch the three claims of §4 verify themselves: exact base-model output at
init, symmetry breaking via the one random factor (note the wrinkle the comments
flag: at the *very first* step only $B$ has nonzero gradient; $A$ starts moving the
step after), and merge-equals-adapter.

## Check yourself

1. Why 16 bytes per parameter for Adam mixed-precision training? Itemize.
2. float16 and bfloat16 both use 16 bits. State the bit-field split of each and which
   failure mode each choice invites.
3. A bf16 weight sits near 1.0; the update is $10^{-4}$. What happens without an fp32
   master copy, and why?
4. Quantize $w = (0.2, -0.5, 0.1)$ to int8 with absmax scaling: give scale, codes,
   and dequantized values.
5. Why does nf4 place its 16 levels at equal-mass Gaussian quantiles instead of
   uniformly?
6. LoRA on a $4096 \times 4096$ matrix at $r = 16$: trainable parameters, and the
   ratio vs full?
7. Why is $B$ initialized to zero but $A$ random, and not the reverse (or both zero)?
8. Your rank sweep shows validation loss still improving sharply from $r = 16$ to
   $r = 64$. What does that say about the low-rank ansatz for this task?

## Answers

1. bf16 weight (2) + bf16 gradient (2) + fp32 master weight (4) + fp32 Adam $m$ (4) +
   fp32 Adam $v$ (4) $= 16$.
2. fp16: 1/5/10 (sign/exponent/mantissa) — more precision, range capped at 65504, so
   it invites overflow (`inf`/`nan`). bf16: 1/8/7 — fp32's range, ~2 decimal digits of
   precision, so it invites rounding noise instead. Training prefers noise to
   overflow.
3. Consecutive bf16 values near 1.0 are ~$0.008$ apart (7 mantissa bits:
   $2^{-7} \approx 0.008$); $1.0 + 10^{-4}$ rounds back to $1.0$, so the update is
   lost. The fp32 master copy (gaps ~$10^{-7}$) accumulates it.
4. absmax $= 0.5$; scale $= 127/0.5 = 254$; codes $= (51, -127, 25)$ (from $50.8,
   -127, 25.4$); dequantized $\approx (0.201, -0.500, 0.0984)$.
5. Trained weights are approximately Gaussian: dense near 0, thin tails. Equal-mass
   quantile bins make all 16 codes equally likely to be used, spending resolution
   where the weights actually are; uniform bins waste codes in the empty tails.
6. $16 \times (4096 + 4096) = 131{,}072$ vs $4096^2 = 16{,}777{,}216$; ratio $128$
   ($= d/2r$), i.e. 0.78%.
7. $B = 0$ guarantees $\Delta W = 0$ at init (exact base model at step 0). If both
   were zero, both gradients ($\propto$ the other factor) would vanish and the adapter
   would never train. $A=0, B$ random would equally give the exact base model at
   init and also trains — but then the first gradient step moves only $A$; the
   standard choice is the one from the paper, and either way one factor must be
   nonzero.
8. The useful update for this task is *not* well-approximated at low rank — the
   ansatz is failing; consider larger $r$, adapting more matrices (MLP projections
   too), or full fine-tuning.

## New terms

- **fine-tuning** — continuing training of a pretrained model on task data.
- **bit / mantissa / exponent / sign bit** — the anatomy of a floating-point number.
- **bfloat16** — 16-bit float keeping float32's exponent range at reduced precision.
- **master weights** — fp32 copies that accumulate updates too small for bf16.
- **quantization** — storing weights as small integer codes plus shared scales.
- **absmax quantization** — scale by $127/\max|w|$, round to int8.
- **outlier problem** — one large weight ruins its block's scale/precision.
- **NF4** — 4-bit format with levels at equal-mass quantiles of a Gaussian.
- **double quantization** — quantizing the per-block scales themselves.
- **PEFT** — parameter-efficient fine-tuning; train few parameters, freeze the rest.
- **LoRA** — the low-rank ansatz $\Delta W = (\alpha/r)BA$ with frozen $W$.
- **low-rank ansatz** — the testable assumption that the useful update has small rank.
- **effective rank** — how many singular values carry most of a matrix's energy.
- **merging (adapters)** — baking $W + (\alpha/r)BA$ into one dense matrix.
- **QLoRA** — LoRA adapters in bf16 over a frozen nf4-quantized base.
- **paged optimizer** — optimizer states that spill to CPU RAM under memory pressure.
- **SFT** — supervised fine-tuning on prompt→response pairs.
- **loss masking** — computing the loss only on response tokens (`-100` labels).

## Going deeper

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
  (arXiv 2106.09685), §§1–4 and 7 — the source; §7 has the rank experiments behind
  §4.4.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*
  (arXiv 2305.14314) — nf4, double quantization, paged optimizers; skim the evals.
- HF `peft` docs (LoRA quickstart, `LoraConfig`, merging) and TRL `SFTTrainer` docs
  (dataset formats, loss masking) — the operational reference for §6.
- HuggingFace LLM course, fine-tuning/`peft` chapters — a guided second pass over
  this lesson's code path.
