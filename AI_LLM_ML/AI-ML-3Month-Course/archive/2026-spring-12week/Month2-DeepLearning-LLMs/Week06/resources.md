# Week 06 — Resources

## Docs

- **PyTorch `nn.Transformer` / `MultiheadAttention`.** https://pytorch.org/docs/stable/nn.html#transformer-layers. Reference implementations; you won't use them directly this week but knowing they exist matters.
- **`F.scaled_dot_product_attention`.** https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html. The function you actually use.
- **`torch.compile`.** https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html.
- **`torch.autocast`.** https://pytorch.org/docs/stable/amp.html.
- **tiktoken.** https://github.com/openai/tiktoken. BPE.
- **arXiv API.** https://arxiv.org/help/api. Use the `arxiv` Python package (https://pypi.org/project/arxiv/).

## Code to read

- **nanoGPT (Karpathy).** https://github.com/karpathy/nanoGPT. `model.py` and `train.py`. Your gold reference.
- **minGPT (Karpathy).** https://github.com/karpathy/minGPT. Older, more pedagogical. nanoGPT is the cleaner version, but minGPT has more comments.
- **x-transformers (Lucidrains).** https://github.com/lucidrains/x-transformers. Menagerie of attention variants; useful reference for RoPE, ALiBi, etc.
- **LLaMA-official.** https://github.com/meta-llama/llama. Production-grade decoder-only transformer. Read after you've built your own.
- **mistral-src.** https://github.com/mistralai/mistral-src. Extremely clean reference for a modern frontier model.

## Tutorials / videos

- **Karpathy "Let's build GPT: from scratch".** https://www.youtube.com/watch?v=kCc8FmEb1nY. Primary reference.
- **Karpathy "State of GPT" (Microsoft Build 2023).** https://www.youtube.com/watch?v=bZQun8Y4L2A. Orientation, ~40 min.
- **3Blue1Brown transformer series.** https://www.youtube.com/watch?v=wjZofJX0v4M. Visual. Watch Ch. 5–6 for attention.
- **Annotated Transformer.** http://nlp.seas.harvard.edu/annotated-transformer/. Alternate walkthrough.

## Papers

- Vaswani — arXiv:1706.03762
- GPT-1 (Radford 2018) — OpenAI report, linked on their research page
- GPT-2 (Radford 2019) — OpenAI report
- GPT-3 (Brown 2020) — arXiv:2005.14165
- LLaMA (Touvron 2023) — arXiv:2302.13971
- LLaMA 2 — arXiv:2307.09288
- Chinchilla — arXiv:2203.15556
- Kaplan (scaling laws) — arXiv:2001.08361
- RoPE — arXiv:2104.09864
- ALiBi — arXiv:2108.12409
- FlashAttention — arXiv:2205.14135
- FlashAttention-2 — arXiv:2307.08691

## Datasets

- **Tiny Shakespeare.** https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/. ~1 MB. For smoke tests.
- **Tiny Stories.** https://huggingface.co/datasets/roneneldan/TinyStories. ~2 GB. Good next-step if you want more data.
- **OpenWebText.** https://huggingface.co/datasets/Skylion007/openwebtext. GPT-2 reproduction corpus, ~40 GB.
- **arXiv abstracts (official dump).** https://www.kaggle.com/datasets/Cornell-University/arxiv. Bulk alternative to scraping the API. ~3 GB.
- **HEP-ML Living Review bibliography.** https://iml-wg.github.io/HEPML-LivingReview/. Not a training corpus, but a well-organized list of titles.

## GitHub repos

- **`karpathy/nanoGPT`** — the thing.
- **`karpathy/minbpe`** — if you want to understand BPE deeply, this is 300 lines that teach it.
- **`bytedance/flash-attention`** — fused attention kernels.
- **`NVIDIA/Megatron-LM`** — reference for large-scale training. Overkill, but the README is educational.
- **`EleutherAI/lm-evaluation-harness`** — benchmarks for LLMs. You won't run it this week; you will in Week 07.

## Tooling

- **wandb.** Already installed. Use it — watching val loss curve descend live is half the fun.
- **`rich` or `tqdm`** for progress bars. `rich` renders beautifully; `tqdm` is ubiquitous.
- **`hydra-core`** for config management. Overkill for this project; worth knowing for Week 11.

## When you get stuck

- **Karpathy's Discord / Zero-to-Hero community.** Very active for nanoGPT debugging.
- **PyTorch forums.** General.
- **HuggingFace forums.** Less relevant this week, relevant next.

## HEP tie-ins

- **Event generator logs as training data** — ambitious extension. Herwig / Pythia / Sherpa logs have structured output; a GPT trained on them could predict next-line structure. Not this week; a potential capstone.
- **ATLAS/CMS papers** — NASA ADS bulk download is a friendlier API than arXiv for large harvesting.

## Notes on correctness

Your biggest risk this week is a subtle bug — mask leaking future info, off-by-one in targets, wrong shape in reshape — that allows loss to drop but produces models that generate garbage. Two guards:

1. The single-batch overfit (E4). If the model can't overfit one batch to near-zero loss, it has a bug.
2. A causal-mask perturbation test (E3 of Week 05). Run it again on your full GPT.

Do both before launching the long training run.

---

Week 07: take a 100M–1B parameter HuggingFace model and fine-tune it with LoRA on a physics summarization task. Everything gets real.
