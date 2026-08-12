# Week 29 — Exercises

Work top to bottom. Setup (imports, model/tokenizer loading, the abstracts corpus from
Week 27, seeds) is given by the notebook; you write only the lines each exercise asks
for. E2's sampler lives in a file, `src/sampling.py`, because Weeks 30–32 reuse it;
everything else is notebook cells.

## E1 — Load a model like you mean it

Load a ~0.5–1B instruct model (default: `Qwen/Qwen2.5-0.5B-Instruct`) three ways:
float32, bfloat16, and float16. For each, generate 100 tokens from the same fixed
prompt and record peak memory and tokens/sec in a table.
Hint: `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`
on GPU; on CPU, compare `sum(p.numel() * p.element_size() for p in model.parameters())`
instead and time with `time.time()`.
Accept when: the 3-row table exists and the bf16/fp32 memory ratio is within 10% of
the 0.5 predicted by bytes-per-parameter.

## E2 — Your own sampler (`src/sampling.py`)

Write `sample_next(logits, temperature, top_k, top_p, generator)` implementing
temperature, top-k, and top-p, composable, as derived in `lesson.md` §3. Then drive a
full generation loop with it.
Hint: mask in logit space with `-inf`; the nucleus must *include* the token that
crosses the threshold — exclude a token only if the cumulative mass before it already
reached `top_p`.
Accept when: with the same seed and settings (T=0.7, k=50, p=0.9), your loop's output
token ids match HF `generate` exactly for 40 tokens.

## E3 — Temperature sweep and entropy

For one physics prompt, sample 5 continuations at each T in {0.1, 0.7, 1.0, 1.5}
(k and p off). At each step also record the entropy of the sampling distribution;
report the mean per T.
Hint: entropy in nats is `-(p * p.log()).sum()` over nonzero entries; average over
steps and samples.
Accept when: mean entropy is strictly increasing in T (the lesson's dH/dT ≥ 0, live)
and the 20 samples are saved side by side in one text file.

## E4 — Chat-template footguns, demonstrated

Send the same question to the instruct model three ways: (a) raw string, no template;
(b) `apply_chat_template` with `add_generation_prompt=True`; (c) the template with
`add_generation_prompt=False`. Save all three outputs.
Hint: print the *rendered strings* too — the bug is visible before generation.
Accept when: (b) answers the question, (a) and (c) visibly misbehave, and each footgun
is named in one line next to its output.

## E5 — A `datasets` pipeline over your corpus

Load the Week 27 abstracts JSONL as a `Dataset`; tokenize with `.map(batched=True)`;
filter to abstracts with > 50 tokens; then open any large Hub text dataset with
`streaming=True` and pull 5 rows without downloading it.
Hint: print `len(ds)` after every stage — silent row loss is the bug class here.
Accept when: the pipeline runs end to end and row counts at each stage are printed.

## E6 — Perplexity: physics vs news

Compute the base (non-instruct) model's sliding-window perplexity on 100 held-out
heavy-ion abstracts and on 100 generic news paragraphs (any public news dataset on
the Hub works). Report both, with window and stride stated.
Hint: score only tokens whose full left context is in the window; `stride = window//2`
is a fine default. Cross-entropy per token first, then `exp`.
Accept when: both perplexities are reported with the window/stride stated, plus one
sentence on which corpus surprises the model more and a guess at why (think Week 27
tokenizer autopsy).

## Review

1. (Week 25) The temperature in sampling rescales logits. Where did a $1/\sqrt{d_k}$
   rescaling appear inside the transformer itself, and what problem did it solve?
2. (Week 08) Write cross-entropy $H(p, q)$ and state its decomposition into entropy
   plus KL divergence.
3. (Week 27) Name two tokenizer pitfalls you demonstrated, and predict which one
   distorts the physics-vs-news perplexity comparison in E6.
4. (Weeks 03–04) You are pulling multi-GB checkpoints this week. Which environment
   and disk-hygiene practices apply (cache location, pinned deps, provenance)?
