# Week 07 — Reading

## Primary

1. **Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022). arXiv:2106.09685.**
   - Read §1–4 fully. §4.1 (which weights to adapt) and §4.3 (LoRA rank vs. full FT) are the engineering heart. Skim §5–6 experiments.

2. **Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023). arXiv:2305.14314.**
   - Read §3 (4-bit NormalFloat), §4 (double quantization, paged optimizers). §5 evaluations.

3. **HuggingFace `transformers` docs — Causal LM tutorial.** https://huggingface.co/docs/transformers/tasks/language_modeling
   - Read the causal LM section. Complements with code what the papers say.

4. **HuggingFace `peft` docs — LoRA.** https://huggingface.co/docs/peft/conceptual_guides/lora
   - Conceptual overview + quickstart. ~15 minutes.

5. **HuggingFace `trl` docs — `SFTTrainer`.** https://huggingface.co/docs/trl/main/en/sft_trainer
   - Skim the full page. This is the API you'll use.

6. **Raschka — *Understanding Parameter-Efficient Fine-tuning* (blog, 2023).** https://magazine.sebastianraschka.com/p/understanding-parameter-efficient
   - Excellent practitioner overview of LoRA and variants. ~30-minute read.

## Secondary

7. **Houlsby et al. — *Parameter-Efficient Transfer Learning for NLP* (ICML 2019). arXiv:1902.00751.**
   - The original adapter paper. LoRA is a descendant. Read §3 for historical context.

8. **Lialin et al. — *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning* (2023). arXiv:2303.15647.**
   - Survey. §3 taxonomy is worth skimming.

9. **Liu et al. — *P-Tuning v2* (ACL 2022). arXiv:2110.07602.**
   - Prompt tuning as an alternative to LoRA. Skim the idea.

10. **Zhao et al. — *Instruction Tuning for Large Language Models: A Survey* (2023). arXiv:2308.10792.**
    - §2, §5. SFT landscape and common datasets.

11. **Ding et al. — *Parameter-efficient fine-tuning of large-scale pre-trained language models* (Nature MI 2023).**
    - Peer-reviewed survey with a useful taxonomy diagram.

## Videos

- **Sebastian Raschka — *Fine-tuning Llama 3.* ~45 min.** Practical walkthrough of the SFTTrainer pipeline. Highly recommended.
- **Andrej Karpathy — *Intro to Large Language Models*. ~1h.** Watch the back half (pretraining → SFT → RLHF) to orient.
- **HuggingFace Deep RL Course — not this week.**
- **Stanford CS324 — LLM systems.** Lecture on fine-tuning and alignment. ~1h.

## Textbooks

Nothing in the canonical ones covers LoRA yet — it's too recent. Murphy PML2 Ch. 17 (*LLMs*) touches on scaling and fine-tuning, but LoRA isn't there yet.

## HEP

- **"Domain-Specific Language Models for High-Energy Physics" — Mikuni et al. (2023).** arXiv search for recent physics LLM efforts; the field is small but growing. Look at what corpus sizes they use, what tasks they fine-tune on.
- **"Chipmunk" (Zenodo) — physics QA benchmark.** If your fine-tune targets physics QA, this is one of the few benchmark datasets.

## Codecademy

"Fine-Tuning Transformers with Hugging Face" module (if available in your Pro catalog). Optional, slightly redundant with the HF docs but good if you learn by interactive exercise.

## GitHub references

- **`huggingface/transformers`** — source is excellent reading. `modeling_llama.py` pairs nicely with your Week 06 nanoGPT.
- **`huggingface/peft`** — LoRA implementation. `peft/tuners/lora.py` is the meat.
- **`unslothai/unsloth`** — 2–5× faster LoRA fine-tuning via Triton kernels. Not required, but worth knowing.
- **`axolotl-ai-cloud/axolotl`** — config-file-driven LoRA training framework. Community standard for batch jobs.

## Blog posts worth reading

- **Tim Dettmers — *8-bit Methods for Efficient Deep Learning*.** https://timdettmers.com
- **Eugene Yan — *Patterns for Building LLM-based Systems*.** https://eugeneyan.com/writing/llm-patterns/
- **Lilian Weng — *Prompt Engineering*.** https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

## Notes to take

- A 1-page table: LoRA vs QLoRA vs full FT on (memory, speed, accuracy, artifact size, composability).
- A diagram of one LoRA-wrapped linear layer showing W (frozen), B, A, and the forward computation.
- A short calculation: for Llama-3.2-1B, target all linear layers with r=16. How many trainable params? (Hint: look up `num_parameters` of each module type.)
- A paragraph: "What fine-tune task on physics abstracts is a better proof-of-work than summarization, and why?"

## Reading plan

| Day | What |
| --- | --- |
| 1 | LoRA paper §1–4. peft conceptual guide. |
| 2 | QLoRA paper §3–4. bitsandbytes docs. |
| 3 | SFTTrainer docs. Raschka blog. |
| 4–5 | Reference as you code. |
| 6 | HEP-specific LLM papers (skim). |
| 7 | Survey reading if time permits. |

## What you don't need to read yet

- RLHF, DPO, PPO — Week 09+ territory.
- MoE, Mamba, retrieval-augmented training — out of scope.
- Quantization deep-dive (GPTQ, AWQ) — you'll use bitsandbytes, which is sufficient.
