# Week 06 — Exercises

## E1 — Read nanoGPT end-to-end (no code, 60 min)

Clone `karpathy/nanoGPT`. Open `model.py`. In a notebook, for each class (`CausalSelfAttention`, `MLP`, `Block`, `GPT`), write 3–5 sentences explaining what each method does and why each design choice was made. The file is ~300 lines; the exercise is ~40 minutes of reading + 20 minutes of writing.

**Accept when:** you can answer, without looking: (a) why the QKV projection is one linear, (b) why weight-tying is applied, (c) what `torch.nn.init` initializations are used for the linear weights and embeddings, (d) what the `configure_optimizers` method's split between decayed and undecayed parameters is doing.

## E2 — Causal self-attention from primitives (medium, 75 min)

Reimplement `CausalSelfAttention` in your own file. Use `F.scaled_dot_product_attention(q, k, v, is_causal=True)` for the core op.

Test against a minimal reference (your Week 05 manual attention with a causal mask). For random `x` of shape `(2, 16, 128)`, outputs must match to 1e-5.

Then:
- Disable `F.scaled_dot_product_attention` and fall back to a manual `QK^T / √d + mask → softmax → V`. Verify outputs match.
- Benchmark the two paths with `%timeit`. Report the speedup. On modern GPUs, SDPA wins by 2–3×.

## E3 — Full GPT in under 150 lines (medium, 120 min)

Write `gpt.py` implementing:

```python
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304   # round up to multiple of 64 for kernels
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

class GPT(nn.Module):
    def __init__(self, config): ...
    def forward(self, idx, targets=None): ...      # returns (logits, loss)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None): ...
```

Parameter count for the config above should be ~10M. Verify with `sum(p.numel() for p in model.parameters())`.

**Accept when:** `forward` produces a (B, T, V) tensor; `generate` produces new tokens without error; param count is within 5% of nanoGPT's reference.

## E4 — Overfit on one batch (easy, 15 min)

Grab a single batch of 64 sequences of length 128 from Shakespeare. Train your GPT on just this batch, no other data. Loss should drop to near-zero in a few hundred steps.

```python
for step in range(500):
    logits, loss = model(xb, yb)
    loss.backward()
    opt.step(); opt.zero_grad()
    if step % 50 == 0:
        print(step, loss.item())
```

**Accept when:** loss drops below 0.1. If it doesn't, your forward/backward has a bug. This is the most important 15-minute test in the week.

## E5 — Shakespeare end-to-end (medium, 90 min)

Download `input.txt` from nanoGPT's repo. Character-level tokenize. Train a small (~1M-param) GPT for 5000 steps. Generate 500 tokens from the prompt "ROMEO:".

**Accept when:** output is recognizably Shakespeare-shaped English (proper capitalization, some rhythm, character names). Train loss should dip below 1.5.

This is your smoke test. If this works, your implementation is correct; if it doesn't, stop and debug before scaling up.

## E6 — BPE tokenization with tiktoken (easy, 30 min)

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Elliptic flow in Au+Au collisions at √s_NN = 200 GeV")
print(ids)
print(enc.decode(ids))
```

Compare: how many tokens does "elliptic flow" use in BPE vs character-level? How about "QGP"? How about "Kaon"?

Write a 100-word note: what does BPE give you that character-level doesn't, and where does that matter for your corpus?

## E7 — Positional encoding comparison (medium, 45 min)

Train three versions of your GPT on Shakespeare (or a 200k-token slice of your heavy-ion corpus):
1. Learned absolute position embeddings (nanoGPT default).
2. Sinusoidal (Vaswani formula, non-learned).
3. No position encoding at all.

Report final val loss. (3) should be dramatically worse — a transformer with no position info is a set-function; language is not a set. (1) and (2) should be close.

Write 150 words on what you observed and why.

## E8 — Sampling with temperature, top-k, top-p (easy, 30 min)

Extend your `generate` to support `top_p`:
```python
def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_idx = logits.sort(descending=True)
    probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -float('inf')
    logits_filtered = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits_filtered, dim=-1), 1)
```

Generate from the same prompt with (greedy / T=0.7 / T=0.7 top-k=40 / T=0.7 top-p=0.9). Report qualitative differences in a short table. Flag which version you'd use for a physics summarizer and why.

## E9 — Heavy-ion abstracts corpus (medium, 60 min)

Using the arXiv API (see `project.md` scraper skeleton), download ~5000 abstracts from 2020–2025 tagged hep-ex, nucl-ex, nucl-th, filter for any of: "heavy-ion", "QGP", "Au+Au", "Pb+Pb", "sPHENIX", "ALICE", "STAR", "RHIC".

Save to `heavy_ion.txt`, one abstract per line (or joined, your choice). Tokenize with BPE. Report total tokens. Report top-20 most frequent BPE tokens in your corpus vs GPT-2's general distribution — do physics-specific tokens jump out?

## E10 — Train your physics GPT (hard, 4–8 hours wall-clock)

Train your 10M-parameter GPT on the heavy-ion corpus. Target train/val loss curves. 30k steps, batch_size 64, block_size 256 — adjust for your GPU.

Generate samples from prompts:
- "The elliptic flow coefficient v2"
- "Quark-gluon plasma is characterized by"
- "At sPHENIX, the EMCal detector"

Log to wandb. Save checkpoints. Post one prompt/sample pair that is convincing (or funniest) to the final README.

## E11 — Efficient attention with `torch.compile` (medium, 30 min)

Wrap your model with `torch.compile(model)`. Verify losses match the uncompiled version on one batch. Benchmark: how much speedup? On A100, expect 30–60% on attention+FFN passes.

**Accept when:** wandb curves overlay between compiled and uncompiled runs; you can report a speedup number.

## E12 — KV cache for inference (hard, 60 min)

Generation without a KV cache is O(T²) per token. With a KV cache it's O(T). Implement one.

In your `CausalSelfAttention.forward`, accept optional `past_kv`, concatenate with new k, v, and return new past_kv. In `generate`, maintain past_kv across tokens.

Benchmark: generate 512 tokens with and without KV cache. Report the speedup. Typical: 10–30× for long outputs.

## E13 — Scaling sketch (easy, 20 min)

For your 10M-param model with 256 block size and batch 64, estimate:
1. FLOPs per forward pass (approximately `6 × N × T` per sequence, per step for forward+backward).
2. Memory for activations (hint: dominated by attention scores: `B × H × T × T × 4 bytes` per layer in fp32).
3. Tokens processed per training run (steps × batch_size × block_size).
4. Is your run compute-bound or memory-bound? (Check `nvidia-smi` during training for GPU util.)

Write the numbers in a short `scaling.md`.

---

## Solution hints

- E1 — the decayed/undecayed split is: bias and LayerNorm get weight_decay=0; everything else gets 0.1. The motivation is that decaying biases or norms hurts.
- E2 — the single QKV linear saves a matmul and makes the code shorter. Output is identical.
- E3 — if your param count is way off, check: (a) weight tying between `wte` and `lm_head`, (b) whether biases are enabled (`bias=False` matters), (c) FFN expansion ratio (4× by default).
- E4 — if loss doesn't drop on a single batch, your model isn't learning *anything*. Print gradients; check `opt.step()` is being called; check `loss.backward()` is being called; check the loss computation.
- E5 — if output looks nothing like Shakespeare, either your tokenizer is wrong (encode/decode round-trip check) or you're training too few steps.
- E6 — BPE nails common sequences as single tokens. "QGP" is probably one token; "quark-gluon plasma" is likely 4–5. BPE efficiency means more information per position.
- E7 — (3) should still train to something — the causal mask gives a tiny bit of position info. But the gap is dramatic.
- E8 — top-p is almost always preferred for generation you'd publish.
- E9 — dedupe by arXiv ID. Drop abstracts where key fields are empty.
- E10 — the "convincing" samples will be short phrases. Entire paragraphs quickly drift into nonsense on a 10M-param model. That's fine; it's still instructive.
- E11 — `torch.compile` has startup overhead (~20s). Don't panic on the first iteration.
- E12 — the KV cache is the single biggest deployment optimization for LLM inference. LLaMA.cpp, vLLM, etc., all implement it. Understanding it here pays off in Week 11.
- E13 — A 10M-param model is almost always memory-bandwidth-bound on consumer GPUs, compute-bound on A100+.
