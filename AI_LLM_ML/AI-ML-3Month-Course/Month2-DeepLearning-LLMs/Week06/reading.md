# Week 06 — Reading

## Primary

1. **Vaswani et al. — *Attention Is All You Need* (NeurIPS 2017). arXiv:1706.03762.**
   - Re-read. Focus this time on §3.1 (encoder-decoder, though you're building decoder-only), §3.2.1 (scaled dot-product), §5 (training details — optimizer, schedule, label smoothing, warmup). The engineering sections are what you'll implement.

2. **Karpathy — *Let's build GPT: from scratch, in code, spelled out* (YouTube, Lecture 6, ~1h56m).**
   - Watch **all of it**, start to finish. Code along. This is the single most valuable video for this week. You can skim at 1.5x after minute 90 if tight on time.
   - URL: https://www.youtube.com/watch?v=kCc8FmEb1nY

3. **nanoGPT repository (Karpathy).** https://github.com/karpathy/nanoGPT.
   - Read **`model.py`** top to bottom. ~300 lines. Understand every line. It is the reference you will check your implementation against.
   - Skim `train.py` — the training loop you will mostly mirror.
   - Skim `sample.py` — reference for your generation loop.

4. **Radford et al. — *Language Models are Unsupervised Multitask Learners* (GPT-2 paper, 2019).**
   - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
   - §2 (architecture), §3 (zero-shot on NLP benchmarks). The architecture §2 is what you're building. Ignore the capabilities claims — they're dated.

5. **Radford et al. — *Improving Language Understanding by Generative Pre-Training* (GPT-1 paper, 2018).**
   - Short. The architecture is the canonical starting point.

## Secondary

6. **The Annotated Transformer (2022 update).** http://nlp.seas.harvard.edu/annotated-transformer/
   - Vaswani paper interleaved with PyTorch implementation. An alternative walkthrough. Useful if Karpathy's video doesn't click — sometimes reading it vs watching it hits different.

7. **Elhage et al. (Anthropic) — *A Mathematical Framework for Transformer Circuits* (2021).** https://transformer-circuits.pub/2021/framework/index.html
   - Residual stream intuition. Read §0–§2. Optional but deepens understanding.

8. **Hoffmann et al. — *Training Compute-Optimal Large Language Models* (Chinchilla, 2022). arXiv:2203.15556.**
   - §3 (the scaling law). You'll see why your 10M-param model on 10M tokens is undertrained.

9. **Kaplan et al. — *Scaling Laws for Neural Language Models* (2020). arXiv:2001.08361.**
   - The predecessor to Chinchilla. §3 is the key result. Skim.

10. **Su et al. — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021). arXiv:2104.09864.**
    - RoPE. §3.1–3.3. Extension material.

11. **Press, Smith, Lewis — *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi, 2022). arXiv:2108.12409.**
    - ALiBi. Short paper. Skim for the idea.

## Tertiary / videos

- **3Blue1Brown — *Attention in transformers, visually explained*** (2024). ~20 min. Visual intuition. Complements Karpathy.
- **Kilcher — *LLaMA: Open and Efficient Foundation Language Models* paper review.** ~45 min. What production variants of your architecture look like.
- **State of GPT — Karpathy at Microsoft Build 2023.** ~40 min. https://build.microsoft.com/ or YouTube. The strategic picture — pretraining → SFT → RLHF → deployment. You're on pretraining.

## Textbooks

- **Murphy, *Probabilistic Machine Learning: Advanced Topics*, Ch. 16 "Attention" §16.2–§16.4.** Transformer architecture, positional encodings, efficiency tricks. Dense but rigorous.
- **Goodfellow, Bengio, Courville — *Deep Learning*, Ch. 10 Sequence Modeling.** Dated for LLMs (pre-transformer focus) but the foundations on recurrence, vanishing gradients, and teacher forcing are timeless.

## Codecademy

Nothing directly. PyTorch sequence is relevant background but you're past it.

## arXiv to skim (pick 2)

- **PaLM** — Chowdhery et al. 2022. arXiv:2204.02311. §2 (architecture) — see the engineering of a frontier-scale transformer.
- **LLaMA** — Touvron et al. 2023. arXiv:2302.13971. §2.2–2.4 (architecture, optimizer, hardware). Pre-norm, RoPE, SwiGLU, RMSNorm — production choices.
- **GPT-NeoX** — Black et al. 2022. arXiv:2204.06745. Training recipe at 20B scale, fully open.

## HEP

- **HEP-ML Living Review** — "Classification / Transformers" section. See who's using transformers on jet / event data.
- **sPHENIX tdrs + recent theses.** The abstracts corpus you scrape will include sPHENIX and RHIC-era papers; having skimmed ~5 of them helps you judge whether your model's generations are plausibly physics-shaped.

## Notes to take

- One-page annotated sketch of nanoGPT's `GPT.__init__` — which module holds which parameters, total param count derivation.
- A small table: GPT-2 vs LLaMA-7B vs your model — `d_model`, `n_layers`, `n_heads`, `vocab_size`, context length, activation, norm type, position encoding.
- 150-word summary of why pre-LN replaced post-LN.
- A paragraph: "If I were to train this on sPHENIX run-summary files instead of arXiv abstracts, how would the tokenizer change and why?"

## Reading plan

| Day | What |
| --- | --- |
| 1 | Karpathy video 0:00–1:00. Annotate. |
| 2 | Karpathy video 1:00–end. nanoGPT `model.py`. |
| 3 | Vaswani re-read §3, §5. |
| 4 | GPT-2 paper §2. Chinchilla §3. |
| 5–7 | Reference reading as needed while coding. |

The video + nanoGPT is the core 80%. Everything else is depth.
