# Week 29 — The HuggingFace Ecosystem

~10 hrs. Before starting you should be able to: train a decoder-only transformer and
explain every tensor in its forward pass (Weeks 25–27); build a BPE tokenizer and
name its pitfalls (Week 27); write cross-entropy and entropy from memory and state
their relationship (Week 08); run a PyTorch training loop with `nn.Module` and
DataLoaders (Week 15).

You spent Month 07 building a language model with your own hands. This week you learn
the shared facility almost everyone actually uses: the HuggingFace ecosystem. The
payoff for having built your own model is that nothing here is a black box — a
"checkpoint on the Hub" is exactly the `state_dict` you saved in Week 27, and
`generate()` is exactly your sampling loop with more options. The one genuinely new
piece of theory this week is sampling: temperature, top-k, and top-p, derived from the
softmax you already know.

## 1. The Hub, checkpoints, and model cards

A trained model is two things: a **config** (the architecture hyperparameters — layers,
heads, hidden size, vocab size) and the **weights** (the tensors). HuggingFace's
**Hub** (huggingface.co) is a giant git-backed file server holding both, plus the
tokenizer files, for hundreds of thousands of models. The `transformers` library reads
those files and rebuilds the model as a PyTorch `nn.Module`.

Three habits before you load anything:

1. **Read the model card** — the README on the model's Hub page. It states what the
   model was trained on, what the chat format is, and what the license allows. Licenses
   differ: some models (Qwen, most Mistral, Gemma with conditions) are permissive;
   Llama models require accepting a community license before download.
2. **Note the parameter count and do the memory arithmetic** (§2). A "1B model" is not
   a small file.
3. **Know where the cache lives.** Downloads go to `~/.cache/huggingface/` by default.
   Multi-GB checkpoints accumulate; `HF_HOME` moves the cache, and
   `huggingface-cli delete-cache` (or plain `rm`) cleans it. This is the Week 03/04
   disk-hygiene habit at LLM scale.

Install the stack with `uv add torch transformers datasets accelerate`.

## 2. Loading a model deliberately: dtype, device, memory

Every parameter is a floating-point number stored in some **dtype** (data type). The
three you will meet constantly:

| dtype | bytes/param | note |
|---|---|---|
| `float32` | 4 | PyTorch's default; full precision |
| `float16` | 2 | half precision; limited range |
| `bfloat16` | 2 | half precision, float32's range; the modern default for LLMs |

(Week 30 opens these formats up bit by bit; for now, bytes-per-parameter is what
matters.) Memory for the weights alone is

$$\text{memory} \approx N_{\text{params}} \times \text{bytes per param}.$$

A 1B-parameter model is ~4 GB in float32 but ~2 GB in bfloat16 — the difference
between fitting on your GPU or not. Inference also needs the **KV cache** (the keys
and values you learned to cache conceptually in Week 26 grow with sequence length),
so leave headroom.

Loading, explicitly and deliberately:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"   # small, permissive license, runs anywhere

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,     # 2 bytes/param
    device_map="auto",        # GPU if available, else CPU
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(n_params, "parameters")
print(n_params * 2 / 1e9, "GB in bf16")
```

`AutoModelForCausalLM` reads the config, picks the matching architecture class, and
loads the weights — the same thing you did by hand with `torch.load` in Week 27.
Print the model. You will recognize every line: embedding, a stack of decoder blocks
(attention + MLP with pre-norm), a final norm, an output head. This is your Week 26
architecture with different variable names.

Generation, the built-in way:

```python
prompt = "The quark-gluon plasma is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

(`**inputs` unpacks a dictionary into keyword arguments — `inputs` holds `input_ids`
and `attention_mask`, and this passes both at once. You will see this idiom in every
HF example, so learn it here: `f(**d)` is `f(input_ids=..., attention_mask=...)`.)

What do `temperature` and `top_p` actually do? That is §3, and it is the heart of the
week.

## 3. Sampling strategies, derived

Your model's forward pass ends in a vector of **logits** $z \in \mathbb{R}^{|V|}$ —
one raw score per vocabulary token. Softmax turns them into a probability
distribution:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.$$

Generation is: sample a token from $p$, append it, run the forward pass again, repeat.
Everything in this section is a different rule for turning $z$ into the distribution
you actually sample from.

### 3.1 A running example

Take a five-token vocabulary and one context's logits:

| token | "the" | "a" | "quark" | "gluon" | "banana" |
|---|---|---|---|---|---|
| logit $z_i$ | 2.0 | 1.0 | 0.5 | 0.2 | −1.0 |

Softmax at face value: $e^{2.0}=7.389$, $e^{1.0}=2.718$, $e^{0.5}=1.649$,
$e^{0.2}=1.221$, $e^{-1.0}=0.368$; the sum is $13.345$, so

$$p = (0.554,\ 0.204,\ 0.124,\ 0.092,\ 0.028).$$

Keep these numbers in view; every strategy below reuses them.

### 3.2 Temperature

**Temperature** $T > 0$ divides the logits before the softmax:

$$p_i(T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.$$

This is *exactly* the Boltzmann distribution from statistical mechanics — the
probability of a physical system occupying a state of energy $E_i$ at temperature $T$
is $\propto e^{-E_i/kT}$. Here $-z_i$ plays the role of energy: high-logit tokens are
low-energy states, and $T$ controls how willing the system is to occupy the unlikely
ones. If you have done statistical mechanics, every intuition transfers. If you
haven't, the two limits below *are* the intuition.

**Limit $T \to 0$.** Write $z_{\max}$ for the largest logit and factor it out:

$$p_i(T) = \frac{e^{(z_i - z_{\max})/T}}{\sum_j e^{(z_j - z_{\max})/T}}.$$

Every exponent $(z_i - z_{\max})/T$ is $\le 0$. As $T \to 0$, each strictly negative
exponent goes to $-\infty$ and its term to 0; only the max-logit token survives with
$e^0 = 1$. So $T \to 0$ is **greedy decoding**: always pick the argmax.

**Limit $T \to \infty$.** Every $z_i/T \to 0$, every $e^{z_i/T} \to 1$, so
$p_i \to 1/|V|$: the uniform distribution. The model's opinion is erased.

**Numbers.** At $T = 0.5$ the scaled logits are $(4, 2, 1, 0.4, -2)$, giving

$$p(0.5) = (0.823,\ 0.111,\ 0.041,\ 0.022,\ 0.002),$$

and at $T = 2$ the scaled logits are $(1, 0.5, 0.25, 0.1, -0.5)$, giving

$$p(2) = (0.369,\ 0.224,\ 0.174,\ 0.150,\ 0.082).$$

Low temperature sharpens; high temperature flattens. "banana" goes from a
1-in-500 event at $T=0.5$ to a 1-in-12 event at $T=2$.

**Entropy rises monotonically with $T$ — proved.** Recall entropy from Week 08:
$H(p) = -\sum_i p_i \ln p_i$. It is cleanest in the inverse temperature
$\beta = 1/T$. Define the normalizer $Z(\beta) = \sum_j e^{\beta z_j}$, so
$p_i = e^{\beta z_i}/Z$. Then

$$H = -\sum_i p_i(\beta z_i - \ln Z) = \ln Z - \beta \langle z \rangle,$$

where $\langle z \rangle = \sum_i p_i z_i$ is the mean logit under $p$. Two standard
facts (differentiate $Z$ and check them — both are one-line calculations):
$\frac{d \ln Z}{d\beta} = \langle z \rangle$ and
$\frac{d\langle z \rangle}{d\beta} = \mathrm{Var}(z)$, the variance of the logits
under $p$. So

$$\frac{dH}{d\beta} = \langle z \rangle - \langle z \rangle - \beta\,\mathrm{Var}(z)
= -\beta\,\mathrm{Var}(z) \le 0,$$

and by the chain rule ($\beta = 1/T$, $d\beta/dT = -1/T^2$),

$$\frac{dH}{dT} = \frac{\mathrm{Var}(z)}{T^3} \ge 0.$$

Entropy never decreases as $T$ rises, with equality only when all logits are equal.
Check against the numbers: $H(p(0.5)) \approx 0.63$ nats,
$H(p(1)) \approx 1.23$ nats, $H(p(2)) \approx 1.50$ nats, ceiling $\ln 5 \approx
1.61$ nats. (Compute one of these by hand; it is the Week 08 formula and nothing
else.)

### 3.3 Top-k: truncate by count

**Top-k sampling** keeps only the $k$ highest-probability tokens, sets the rest to
zero, and renormalizes:

$$p_i^{(k)} = \frac{p_i}{\sum_{j \in \text{top-}k} p_j} \text{ if } i \in \text{top-}k,
\qquad 0 \text{ otherwise}.$$

With $k = 2$ on the running example: keep $(0.554, 0.204)$, whose sum is $0.758$, so

$$p^{(2)} = (0.731,\ 0.269,\ 0,\ 0,\ 0).$$

Note that $k = 1$ is greedy decoding again — so greedy $=$ top-k(1) $=$ the $T \to 0$
limit. Three roads to the same place.

Why truncate at all? Holtzman et al. (2019) showed that the low-probability tail is
where degenerate text comes from: each tail token is individually unlikely, but there
are tens of thousands of them, so their *total* mass is large, and sampling the full
distribution keeps rolling that die until something incoherent lands. Truncation
deletes the tail.

### 3.4 Top-p (nucleus): truncate by mass

Top-k has a flaw: $k$ is fixed, but the *shape* of the next-token distribution is not.
After "The capital of France is", the distribution is sharply peaked — one token
deserves nearly all the mass, and $k = 50$ admits 49 bad options. Mid-sentence in open
prose, the distribution is flat — hundreds of tokens are reasonable, and $k = 50$
cuts off good ones.

**Top-p (nucleus) sampling** truncates by cumulative probability instead: keep the
smallest set of tokens, taken in descending probability order, whose total mass
reaches $p$, then renormalize over that set.

With $p = 0.9$ at $T = 1$: cumulative sums are $0.554, 0.758, 0.882, 0.974$. After
three tokens we have $0.882 < 0.9$, so the fourth is included; the nucleus is
{"the", "a", "quark", "gluon"} with mass $0.974$, and

$$p^{(0.9)} = (0.569,\ 0.209,\ 0.127,\ 0.094,\ 0).$$

Now the adaptivity: at $T = 0.5$ the distribution was $(0.823, 0.111, \ldots)$, and
the cumulative sums are $0.823, 0.934$ — the same $p = 0.9$ keeps only **two** tokens.
A peaked (low-entropy) context gets a small nucleus, a flat (high-entropy) context a
large one, automatically. That is what "top-p adapts where top-k cannot" means, and
you just watched it happen in four rows of arithmetic.

### 3.5 Composing them, and implementing them

In practice the pipeline is: logits → temperature scale → top-k mask → top-p mask →
softmax → sample. Masking is done in logit space by setting excluded logits to
$-\infty$ (their $e^{-\infty} = 0$ handles the renormalization for free):

```python
import torch

def sample_next(logits, temperature, top_k, top_p, generator):
    logits = logits / temperature
    if top_k > 0:
        kth_best = torch.topk(logits, top_k).values[-1]
        logits[logits < kth_best] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cut = cumulative - probs >= top_p   # True where mass ALREADY >= p before this token
        sorted_logits[cut] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits[sorted_idx] = sorted_logits
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)
```

The `cumulative - probs >= top_p` line is the classic off-by-one trap: the nucleus
must *include* the token that crosses the threshold, so a token is excluded only if
the mass *before* it already reached $p$. Get this wrong and you will disagree with
HF `generate` on exactly the borderline tokens — which is why Exercise E2's acceptance
criterion is exact token-id agreement.

## 4. `datasets`: data that doesn't fit in RAM

The `datasets` library is pandas-flavored tooling for ML corpora, with two features
pandas lacks: memory-mapped storage (a million abstracts on disk, read lazily) and
**streaming** (iterate over a Hub dataset without downloading it all).

```python
from datasets import load_dataset, Dataset

# From your own files (the Week 27 abstracts corpus):
ds = load_dataset("json", data_files="abstracts.jsonl", split="train")

# Transform every row; batched=True processes chunks, not single rows:
def tok_fn(batch):
    return tokenizer(batch["abstract"], truncation=True, max_length=512)

ds_tok = ds.map(tok_fn, batched=True)

# Filter:
ds_long = ds_tok.filter(lambda row: len(row["input_ids"]) > 50)
print(len(ds), len(ds_tok), len(ds_long))

# Streaming a Hub dataset — nothing is downloaded up front:
stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
first = next(iter(stream))
```

(`lambda row: ...` is a one-line unnamed function — shorthand for a one-line `def`.
`datasets` expects functions as arguments so often that the shorthand is idiomatic;
use a named `def` whenever the body is more than a comparison.) The habit to build:
print row counts after every `map`/`filter` stage. Silent row loss is the
data-pipeline equivalent of the leakage bugs you hunted in Week 09.

## 5. Chat templates and their footguns

An instruct model was fine-tuned on conversations serialized in one **exact** textual
format — special tokens marking who is speaking, e.g. (Qwen's ChatML format):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the QGP?<|im_end|>
<|im_start|>assistant
```

The **chat template** is a small program, shipped inside the tokenizer, that renders a
list of `{"role": ..., "content": ...}` dictionaries into that exact string:

```python
messages = [
    {"role": "system", "content": "You are a concise physics assistant."},
    {"role": "user", "content": "What is the quark-gluon plasma?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
```

The model is *brittle* about this format — it saw millions of examples of it and
essentially none of anything else. Hence the footguns:

**Footgun 1 — raw prompts to instruct models.** Send `"What is the QGP?"` with no
template and the model is completing an out-of-distribution string, not answering a
question. Typical symptoms: it continues your question, answers as if mid-document,
or emits role tokens itself. Base models want raw text; instruct models want the
template. Always know which you loaded.

**Footgun 2 — the missing generation prompt.** `add_generation_prompt=True` appends
the header that cues the model that *the assistant speaks next* (the trailing
`<|im_start|>assistant` above). Omit it and the model may continue *your* turn or
open a new user turn. The related training-time trap: when you format SFT data in
Week 30, the template must match what you use at inference *exactly* — a stray
newline or a default system prompt inserted by one code path and not the other is a
real, silent performance bug.

Templates differ per model family (ChatML, Llama's header tags, Gemma's turns).
`apply_chat_template` exists so you never hand-write them. Use it, and diff its
output against the model card once per new model — trust, but verify.

## 6. Perplexity: one number for "how surprised is the model"

You already own the pieces. Cross-entropy of a model $q$ on text $x_{1:N}$
(Week 08, Week 26 — it is your training loss) is the average negative log-likelihood
per token:

$$H = -\frac{1}{N}\sum_{t=1}^{N} \ln q(x_t \mid x_{<t}).$$

**Perplexity** exponentiates it:

$$\mathrm{PPL} = e^{H}.$$

Interpretation: the model is, on average, as uncertain as if it were choosing
uniformly among $\mathrm{PPL}$ tokens. A perplexity of 12 on physics abstracts means
"an effective 12-way choice per token." Two caveats that make it an honest tool
rather than a leaderboard number:

- **It is tokenizer-dependent.** Different vocabularies chop the same text into
  different numbers of tokens; per-token perplexities across models with different
  tokenizers are not comparable. (Your Week 27 tokenizer-autopsy findings — digits and
  symbols like $\sqrt{s_{NN}}$ splitting into many tokens — bite here.)
- **It measures fit to the text distribution, not usefulness.** A model can have
  great perplexity and still be a poor assistant, and instruction tuning typically
  *worsens* perplexity on raw corpus text while improving everything you care about.

Computing it honestly on long text uses a **sliding window**: the model has a finite
context (say 2048 tokens), so you score long documents in overlapping windows with a
stride, only counting the loss of tokens whose full left context is present. State
your window and stride when you report the number; they change it.

## 7. Worked example: sampler vs `generate`, end to end

The loop that ties the week together — your own sampler driving a Hub model, verified
against the library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
model.eval()

messages = [{"role": "user", "content": "Name one signature of the quark-gluon plasma."}]
ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)

g = torch.Generator().manual_seed(29)
generated = ids
with torch.no_grad():
    for step in range(40):
        logits = model(generated).logits[0, -1, :].float()
        next_id = sample_next(logits, 0.7, 50, 0.9, g)   # from section 3.5
        generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True))
```

Every line is something you built in Month 07: `apply_chat_template` writes the prompt
format, the forward pass gives logits, `sample_next` is §3, and the loop is
autoregression. `model.generate(...)` is this loop plus batching, KV caching, and
stopping criteria. You have now seen both sides of the abstraction — which is the
whole point of having built the instrument before using the facility.

## Check yourself

1. A 3B-parameter model in bfloat16: roughly how much memory for the weights alone?
2. Show in two lines why $T \to 0$ makes softmax-with-temperature pick the argmax.
3. In the running example (§3.1), what does top-p $= 0.9$ keep at $T = 1$, and what
   does it keep at $T = 0.5$? What general property does the difference illustrate?
4. What is $\frac{dH}{dT}$ for the temperature-scaled softmax, and what does its sign
   tell you?
5. Greedy decoding, top-k(1), and $T \to 0$: what is the relationship?
6. You send a bare question string to an instruct model and it replies with another
   question. Which footgun is this, and what is the fix?
7. Why can't you compare per-token perplexities between two models with different
   tokenizers?
8. Perplexity 15 on a corpus — say in one sentence what that means operationally.

## Answers

1. $3\times10^9 \times 2$ bytes $= 6$ GB, weights only; the KV cache and activations
   come on top.
2. $p_i(T) = e^{(z_i - z_{\max})/T} / \sum_j e^{(z_j - z_{\max})/T}$; for
   $z_i < z_{\max}$ the exponent $\to -\infty$ so the term $\to 0$; only the argmax
   term survives ($e^0 = 1$).
3. At $T=1$: four tokens (cumulative $0.882 < 0.9$ after three, so the fourth joins).
   At $T=0.5$: two tokens (cumulative $0.934 \ge 0.9$). The nucleus size adapts to
   the entropy of the distribution — small when peaked, large when flat — which fixed
   $k$ cannot do.
4. $dH/dT = \mathrm{Var}_p(z)/T^3 \ge 0$: raising temperature never decreases the
   entropy of the sampling distribution.
5. They are identical: top-k with $k=1$ keeps only the argmax, which is exactly the
   $T \to 0$ limit, which is greedy decoding.
6. Footgun 1 (missing chat template) — or footgun 2 if the template was applied
   without `add_generation_prompt=True`. Fix: `apply_chat_template` with the
   generation prompt on.
7. The tokenizers split the same text into different numbers of tokens, so
   "per token" refers to different units; the averages are over different
   denominators and are not comparable.
8. On average the model's next-token uncertainty is equivalent to a uniform choice
   among 15 tokens.

## New terms

- **Hub** — HuggingFace's repository of model checkpoints, tokenizers, and datasets.
- **model card** — the model's documentation page: training data, format, license.
- **dtype** — the storage format of each number; float32/float16/bfloat16 here.
- **logits** — the raw pre-softmax scores over the vocabulary.
- **temperature** — divisor applied to logits before softmax; controls entropy.
- **greedy decoding** — always take the argmax token; the $T \to 0$ limit.
- **top-k sampling** — keep the $k$ most probable tokens, renormalize.
- **top-p (nucleus) sampling** — keep the smallest token set with cumulative mass
  $\ge p$, renormalize.
- **chat template** — tokenizer-shipped renderer from role/content messages to the
  exact string format the instruct model was trained on.
- **generation prompt** — the trailing assistant-turn header that cues the model to
  respond.
- **instruct model / base model** — fine-tuned-for-dialogue vs raw next-token
  predictor.
- **streaming** — iterating a dataset from remote storage without downloading it all.
- **perplexity** — $e^{\text{cross-entropy}}$; effective branching factor per token.
- **sliding window (perplexity)** — scoring long text in overlapping context windows.
- **KV cache** — stored keys/values from previous positions, reused each decode step.

## Going deeper

- HuggingFace LLM course, the `transformers` and `datasets` chapters — the guided tour
  of everything in §§1, 2, 4 (free, huggingface.co/learn).
- HF docs, *Generation strategies* and *Chat templates* — the authoritative reference
  for §§3 and 5's knobs and formats.
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (arXiv 1904.09751),
  §§1 and 3 — the paper that diagnosed the tail problem and proposed nucleus sampling.
- One model card read end to end (e.g. Llama 3.2 1B or Qwen2.5-0.5B-Instruct) — the
  habit matters more than the specific card.
