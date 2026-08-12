# Week 06 — Transformer From Scratch (nanoGPT-style)

## Where this sits

You've read Vaswani. You've implemented scaled dot-product attention. Now you build a full decoder-only transformer — token embeddings, positional encodings, stacked self-attention + FFN blocks, training loop, sampling loop — from primitives. The reference is Karpathy's nanoGPT. The dataset is a heavy-ion abstracts corpus you'll scrape from arXiv. By the end of the week you will have trained a small GPT, watched loss curves flatten, and generated plausibly physics-sounding nonsense.

This is the most consequential week of the course. Every modern LLM is a variant of what you'll build here. Fine-tuning (Week 07), RAG (Week 08), tool use (Week 09), and agents (Weeks 10–11) all sit on top of this architecture. If you internalize one thing across twelve weeks, make it this.

## Learning objectives

By the end of the week you can:

1. Explain, at a whiteboard, the data flow through a decoder-only transformer — from token IDs to logits — without notes.
2. Implement causal multi-head self-attention, residual streams, pre-LayerNorm, and a position-wise FFN from `torch.nn` primitives.
3. Train a small (<20M parameter) GPT on a custom corpus; reach training loss that tracks the validation loss within 10%.
4. Write a sampling loop with temperature, top-k, and top-p (nucleus) decoding.
5. Compute approximate FLOPs per training step, and identify whether your run is compute-bound, memory-bound, or I/O-bound.
6. Connect transformer components to Week 05 concepts — scaling laws, KV caching, and why pre-LN won.

## The big picture

A decoder-only transformer is a stack of identical blocks that transform a sequence of token embeddings into contextualized embeddings. The block is:

```
x_in  → LN → MHA → + → LN → FFN → +  → x_out
        residual skip over MHA    residual skip over FFN
```

At the top you apply a final LayerNorm, project to vocabulary size, and softmax to get next-token probabilities. You train by minimizing the cross-entropy between the predicted distribution and the true next token, for every position in parallel (this is the superpower — you get T training signals per sequence instead of one).

That's it. Everything else is engineering.

## The pieces, in detail

### Token embeddings

A lookup table: `nn.Embedding(vocab_size, d_model)`. Row i is the d_model-dimensional embedding of token i. Learned end-to-end. Weight-tied with the output projection in nanoGPT (`lm_head.weight = wte.weight`) — saves parameters and improves generalization.

### Positional embeddings

Transformers have no notion of order without them. Three flavors:

1. **Learned absolute.** `nn.Embedding(block_size, d_model)`. Added to token embeddings. Simple, works. Caps at `block_size`.
2. **Sinusoidal (Vaswani).** Fixed, infinite-length-generalizable in principle. In practice, rarely used in modern LLMs.
3. **Rotary (RoPE).** Applied inside attention to Q and K; rotates pairs of dimensions by an angle proportional to position. Used in LLaMA, Mistral, most modern frontier models. Generalizes better to longer contexts.

You'll use **learned absolute** this week (nanoGPT's default). RoPE is a 2-hour extension if you finish early.

### Multi-head self-attention

For each head h = 1..H:
- `q_h = x @ W_Q^h`, `k_h = x @ W_K^h`, `v_h = x @ W_V^h`, each with output dim `d_head = d_model / H`.
- `a_h = softmax(q_h @ k_h^T / √d_head + causal_mask) @ v_h`.

Concatenate heads along the feature axis, project with `W_O` of shape `(d_model, d_model)`.

In practice you do this with one batched matmul:
```python
qkv = self.c_attn(x)          # (B, T, 3*d_model)
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
# same for k, v
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash on supported GPUs
y = y.transpose(1, 2).contiguous().view(B, T, d_model)
y = self.c_proj(y)
```

`F.scaled_dot_product_attention` with `is_causal=True` is fused and ~2–3× faster than naive. Use it.

### Position-wise FFN

```
h = GELU(x @ W_1 + b_1)       # expand: d_model → 4 * d_model
y = h @ W_2 + b_2              # contract: 4 * d_model → d_model
```

The 4× expansion is canonical. GELU is the default nonlinearity (slightly better than ReLU for LLMs empirically). Modern variants: SwiGLU (used by LLaMA), GeGLU. Stick with GELU this week.

### LayerNorm

Normalizes across the feature dimension per token:
```
LN(x)_i = γ · (x_i - μ) / √(σ² + ε) + β
```
where μ, σ² are computed over the `d_model` features for that token. γ, β are learnable per-feature parameters.

Normalizes activations; reduces internal covariate shift (the hand-wavy motivation; the real reason it helps is still debated). Crucially different from BatchNorm in that it doesn't couple batch elements — essential for variable-length sequences and inference with batch=1.

### Pre-LN vs Post-LN

Vaswani put LayerNorm **after** the residual add ("Post-LN"). Modern transformers put it **before** the MHA/FFN block ("Pre-LN"). Pre-LN trains more stably at depth; Post-LN needs careful warmup. Use Pre-LN. Every production LLM does.

Pre-LN block:
```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

### Residual streams

The x that gets skipped around is the "residual stream". Each block reads from it and writes back. Think of it as a slowly-refined vector that aggregates information across layers. Anthropic's mechanistic interpretability work (for your own curiosity: the transformer-circuits thread) treats the residual stream as the central object.

### Output head

```python
logits = x @ self.lm_head.weight.T   # (B, T, vocab_size)
```

If you weight-tie, `lm_head.weight = wte.weight`. One less parameter tensor; slightly better perplexity.

## Training

### Loss

Cross-entropy between logits `(B, T, V)` and targets `(B, T)`. `F.cross_entropy(logits.view(-1, V), targets.view(-1))`. The targets are the input sequence shifted by one.

### Optimizer

AdamW. Weight decay 0.1 on all parameters except biases and LayerNorm (nanoGPT's "decayed vs undecayed params" split). Betas (0.9, 0.95) are more common than (0.9, 0.999) for LLMs — slightly faster convergence.

### Schedule

Linear warmup (say 2000 steps) then cosine decay to 10% of peak LR. Peak LR ~ 3e-4 for small models. Lower for larger models; scaling laws exist but you don't need them this week.

### Gradient clipping

Clip to 1.0 norm. One-liner: `torch.nn.utils.clip_grad_norm_(params, 1.0)`. Prevents the occasional exploding-gradient spike.

### Mixed precision

`torch.autocast(device_type='cuda', dtype=torch.bfloat16)` wraps the forward pass. bfloat16 on A100+/RTX 30+/RTX 40+. On older GPUs use float16 with a GradScaler. Cuts memory ~2× and speeds training 1.5–2×.

### `torch.compile`

`model = torch.compile(model)`. 20–50% speedup on modern PyTorch. First call compiles (slow), subsequent calls fast. Verify correctness by running one step compiled and one uncompiled; losses should match within numerical precision.

## Tokenization

You need to turn text into token IDs. Options:

### Character-level
`vocab = sorted(set(text))`. Each character is one token. ~100 tokens. Simple, works for toy corpora. Poor efficiency: "neutron" is 7 tokens.

### Word-level
`vocab = text.split()`. Unbounded vocab; OOV tokens everywhere. Don't.

### BPE (Byte-Pair Encoding)
Start with characters; iteratively merge the most common adjacent pair. GPT-2's tokenizer has 50257 tokens. `tiktoken.get_encoding("gpt2")` gives you this for free.

**Use BPE (via tiktoken) for the project.** Character-level is fine for the toy MVP in exercise E5.

## Datasets

### Shakespeare (tiny, for smoke-testing)
~1MB. `input.txt` from nanoGPT repo. Trains in minutes on a CPU. Use this for your first end-to-end run.

### Your heavy-ion abstracts corpus
Scrape arXiv `hep-ex` + `nucl-ex` + `nucl-th` abstracts from 2015–2025 tagged with "heavy-ion" or "QGP". ~50k abstracts, ~50MB. Big enough that a 10M-parameter GPT can underfit; small enough that training runs finish in an evening on an RTX 3060.

See `project.md` for the scraper.

## Training dynamics to expect

- **First 100 steps:** loss drops fast from ~10 (uniform over vocab ~ log(50257) ≈ 10.8) to ~4. Model learning unigram statistics.
- **Steps 100–2000:** loss drops from ~4 to ~3. Bigrams, common fragments.
- **Steps 2000–10000:** loss drops slowly from ~3 to ~2.5. Longer-range structure. This is where most of the "wow" gains happen.
- **Steps 10000+:** loss drops very slowly. Diminishing returns.

If your loss plateaus at ~10 and never drops: bug. Usually learning rate too high (NaN), or you forgot to shift targets, or the mask is wrong.

If your loss drops below training loss: bug. Validation should always track training.

If training loss diverges: lower LR, check gradient clipping, check for shape errors.

## Sampling

### Greedy
`next_id = logits.argmax(-1)`. Deterministic, boring. Collapses to loops.

### Temperature
`logits /= T`. T=1 is as-trained; T<1 sharpens (more deterministic); T>1 flattens (more random). Typical: 0.7–1.0.

### Top-k
Zero out all but top-k logits before softmax. k=40 is common.

### Top-p (nucleus)
Zero out logits outside the smallest set whose softmax mass ≥ p. p=0.9 is common. Usually better than top-k for creative generation.

Combine: temperature + (top-k OR top-p). nanoGPT provides both.

## FLOPs and tokens

For a decoder-only transformer with N non-embedding parameters, training FLOPs per token is approximately **6N** (factor of 2 for forward + backward; factor of 3 for matmul flop counting). So a 10M-param model trained on 100M tokens uses ~6 × 10^15 FLOPs = 6 PFLOPs. An A100 does ~300 TFLOPs/s at FP16. So training should take ~20 seconds of pure math; reality will be 5–10 minutes due to I/O, kernel launch overhead, memory bandwidth.

Chinchilla (Hoffmann et al. 2022) says for optimal compute allocation, tokens ≈ 20 × parameters. A 10M-param model wants ~200M training tokens. Your 50MB corpus is ~10M tokens. You're undertrained — that's fine for pedagogy; you'll see overfitting, which is instructive.

## HEP angle

Why does any of this matter for your sPHENIX work? A few things:

1. Tokenizing physics text teaches you what embeddings can represent. When we get to fine-tuning (Week 07), you'll fine-tune a pre-trained model on your domain — but the intuitions come from building one yourself.
2. Attention matters for jet constituents (Particle Transformer) and tracking (coming up). Seeing it run end-to-end here demystifies the variants.
3. An honest understanding of what LLMs can and can't do — scaling laws, memorization vs reasoning, limits of context windows — is needed before trusting one to read your papers or drive analysis.

## Pitfalls and debugging

**Loss stuck at log(V).** Your targets are not shifted, or you're averaging over the wrong dimension, or your vocab has a different size than the embedding table. Print shapes and one batch of (input, target, logits).

**Loss drops then spikes to NaN.** Learning rate too high, or no gradient clipping, or mixed-precision overflow. Clip to 1.0, reduce peak LR by 3×, add `GradScaler` if fp16.

**Generated text is gibberish even after long training.** Either undertrained or tokenizer mismatch (training on GPT-2 tokens but decoding with char-level, or vice versa). Double-check `encode/decode` round-trip.

**Causal mask wrong.** You see future tokens leaking into past predictions. Run the perturbation test from Week 05 E3.

**`torch.compile` breaks on first call.** PyTorch version < 2.0 or a weird op. Drop compile; reproducibility first, speed second.

**GPU memory blows up.** Reduce `batch_size × block_size`. You can fit ~65k tokens per forward pass on a 12GB GPU for a 10M-param model; on a 24GB GPU, ~130k. Gradient accumulation is your friend for effective larger batch sizes.

**Validation loss diverges from training.** Overfitting. Your corpus is too small or model too large. Shrink the model or get more data. Regularize with dropout=0.1.

## Tooling introduced

- **tiktoken.** OpenAI's BPE library. `pip install tiktoken`. Fast, Rust-backed. The de facto standard.
- **`torch.compile`.** Worth turning on by default in 2025+ PyTorch.
- **`torch.autocast`.** Mixed precision context manager.
- **wandb.** Log training curves. You should have wandb from Week 04; use it here too.
- **Reading somebody else's code.** nanoGPT is ~300 lines; read it top-to-bottom before you write your own. This week is partly exegesis.

## Time budget (7 days)

| Day | Focus |
| --- | --- |
| 1 | Read Vaswani again. Read nanoGPT source top-to-bottom. E1–E2. |
| 2 | Implement MHA, FFN, block. E3. |
| 3 | Stitch into full GPT class. Overfit on a single batch. |
| 4 | Shakespeare end-to-end. Sampling loop. |
| 5 | Scrape heavy-ion abstracts. Tokenize. |
| 6 | Train full run on heavy-ion corpus. |
| 7 | Evaluation, sampling experiments, write-up. |

## What "done" looks like

- A `gpt.py` file implementing `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` — roughly mirroring nanoGPT's shape.
- Trained a ~10M-parameter model on heavy-ion abstracts; train/val loss curves in wandb.
- Generation samples: give it a prompt like "The elliptic flow of" and produce 200 tokens. Qualitatively the output should look physics-like even if wrong.
- A 400-word reflection: "What surprised me, what didn't, what I'd change."

Week 07 we take a pre-trained HuggingFace model and fine-tune it with LoRA. That's where this really starts earning its keep for your research.
