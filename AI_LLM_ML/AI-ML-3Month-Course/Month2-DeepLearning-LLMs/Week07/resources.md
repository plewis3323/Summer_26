# Week 07 — Resources

## Docs

- **`transformers`** — https://huggingface.co/docs/transformers
- **`peft`** — https://huggingface.co/docs/peft
- **`trl`** — https://huggingface.co/docs/trl
- **`datasets`** — https://huggingface.co/docs/datasets
- **`accelerate`** — https://huggingface.co/docs/accelerate
- **`bitsandbytes`** — https://github.com/bitsandbytes-foundation/bitsandbytes
- **`evaluate`** — https://huggingface.co/docs/evaluate

## Models (open weights)

- **Llama 3.2** — 1B and 3B instruct models. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- **Llama 3.1 8B Instruct** — for QLoRA. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **Qwen 2.5** — 0.5B, 1.5B, 3B, 7B. https://huggingface.co/Qwen
- **Phi-3-mini** — 3.8B. https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **Mistral-7B-Instruct-v0.3** — classic mid-size. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- **Gemma-2-2B** — Google's small. https://huggingface.co/google/gemma-2-2b-it
- **SmolLM2-1.7B-Instruct** — HF's own; very small, surprisingly capable. https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct

Pick Llama-3.2-1B-Instruct as default. If it's too small for quality, bump to Llama-3.2-3B.

## Serving

- **vLLM** — https://github.com/vllm-project/vllm. Production inference.
- **Text Generation Inference (TGI)** — https://github.com/huggingface/text-generation-inference. HF's server.
- **Ollama** — https://ollama.ai. Local deployment via CLI.
- **llama.cpp** — https://github.com/ggerganov/llama.cpp. CPU / Mac / GGUF.

## Tutorials

- **HF LoRA tutorial** — https://huggingface.co/docs/peft/main/en/tutorial/lora_based_methods
- **HF QLoRA blog** — https://huggingface.co/blog/4bit-transformers-bitsandbytes
- **Raschka LoRA practical guide** — https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- **HF SFTTrainer tutorial** — https://huggingface.co/docs/trl/main/en/sft_trainer

## Frameworks (higher-level, optional)

- **Axolotl** — https://github.com/axolotl-ai-cloud/axolotl. Config-file training. Good for batch jobs.
- **Unsloth** — https://github.com/unslothai/unsloth. 2–5× faster LoRA via custom kernels. Free tier has model restrictions but works on Llama/Qwen/Mistral.

## Papers

- LoRA — arXiv:2106.09685
- QLoRA — arXiv:2305.14314
- Adapter tuning — arXiv:1902.00751
- Prefix tuning — arXiv:2101.00190
- P-tuning v2 — arXiv:2110.07602
- LoRA+ — arXiv:2402.12354 (variant that scales B's LR differently)
- DoRA — arXiv:2402.09353 (weight-decomposed LoRA, sometimes better)
- PEFT survey — arXiv:2303.15647

## Datasets (reference)

- **Alpaca** — https://huggingface.co/datasets/tatsu-lab/alpaca. General instruction tuning.
- **Dolly-15k** — https://huggingface.co/datasets/databricks/databricks-dolly-15k. Human-generated Qs.
- **UltraChat** — https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k. Multi-turn.
- **OpenOrca** — https://huggingface.co/datasets/Open-Orca/OpenOrca. Chain-of-thought style.
- **No-Robots** — https://huggingface.co/datasets/HuggingFaceH4/no_robots. 10k human-only prompts.

You're not using these directly — you're generating your own physics dataset. These are for reference patterns.

## Tooling

- **wandb** — already installed. Log LoRA runs.
- **HuggingFace CLI** — `huggingface-cli login`, `huggingface-cli download`, `huggingface-cli upload`.
- **`tensorboard`** — alternative logger. Less slick than wandb.

## When you get stuck

- **HuggingFace Discord and Forums.** Active; most questions answered within hours.
- **PEFT GitHub Issues.** Well-curated.
- **r/LocalLLaMA (Reddit).** Community discussion; quality varies but signal is there.
- **Anthropic's prompting docs** — https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering. Relevant for your teacher pipeline.

## HEP tie-ins

- **Papers using LLMs in HEP** — search arXiv for "large language model" + "particle physics" in titles. Small but growing list. Mikuni, Dolan, and others have touched it.
- **HEP-ML Living Review** — https://iml-wg.github.io/HEPML-LivingReview/. No dedicated LLM section yet; worth scanning classification and NLP-adjacent work.

## GitHub repos worth cloning

- **`huggingface/peft`** — reference implementation.
- **`artidoro/qlora`** — Dettmers' original QLoRA repo. Still an educational read.
- **`tloen/alpaca-lora`** — the canonical "fine-tune a small model on a small instruction dataset" repo. Outdated code but clear structure.
- **`h2oai/h2o-llmstudio`** — GUI over HF training. Worth knowing exists.

## Cost awareness

- Claude API for teacher summaries: ~$0.01 per abstract summary. 500 summaries = ~$5.
- Model-hosted inference on Hub: free for community models.
- Hub storage: free up to ~5 GB per repo.
- Your compute: local. Negligible marginal cost.

Set a daily API cap in your Anthropic console (`$10/day` is more than enough).

## Notes on data cleanliness

Small-dataset fine-tuning is data-cleanliness-limited. Budget:

- 30% of your project time on data curation.
- 40% on training / eval loop engineering.
- 20% on evaluation + analysis.
- 10% on write-up.

First-time LoRA fine-tuners spend 80% on training and 10% on data and wonder why their models are mediocre. Don't be that person.

---

Week 08: RAG. We plug your model into a retrieval layer so it stops guessing and starts citing.
