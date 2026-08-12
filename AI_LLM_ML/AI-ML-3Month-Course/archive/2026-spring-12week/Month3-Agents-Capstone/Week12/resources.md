# Week 12 — Resources

Track-specific only. For foundational references see weeks 6–11.

## Everyone

### Project-hygiene docs
- **uv** — https://docs.astral.sh/uv/
- **ruff** — https://docs.astral.sh/ruff/
- **pytest** — https://docs.pytest.org/
- **pre-commit** — https://pre-commit.com/
- **pyproject.toml reference** — https://packaging.python.org/en/latest/specifications/pyproject-toml/

### Deck-authoring
- **Marp** (Markdown → slides) — https://marp.app/
- **Quarto** — https://quarto.org/ (slides + docs; excellent)
- **reveal.js** — https://revealjs.com/
- **PowerPoint / Keynote** — fine too. Ship, don't shop.

### Writing / communication
- **Eugene Yan — *How to write better blog posts*.** https://eugeneyan.com/writing/writing-better/
- **Patrick Winston — *How to speak*.** https://www.youtube.com/watch?v=Unzc731iCUY. 60 minutes; watch this before your demo.
- **Harvard's technical writing checklist** — find via search; any will do.

### Eval methodology
- **Hamel Husain — *Your AI product needs evals*.**
- **Eugene Yan — *Evaluation & hallucination detection for abstractive summaries*.**
- **Anthropic — *How we build evals*.** (docs.claude.com)

---

## Track A — HEP Copilot

### Docs
- **MCP specification.** https://modelcontextprotocol.io/specification
- **MCP Python SDK.** https://github.com/modelcontextprotocol/python-sdk
- **claude-agent-sdk.** https://docs.claude.com (SDK section)
- **Claude Desktop config reference.** https://modelcontextprotocol.io/quickstart/user

### Libraries
```
uv add "mcp[cli]" claude-agent-sdk
uv add lancedb sentence-transformers  # from W08
uv add uproot numpy matplotlib pandas
```

### Exemplars to read
- `modelcontextprotocol/servers/src/filesystem`
- `modelcontextprotocol/servers/src/postgres`
- Any community `mcp-server-*` that mirrors your tool structure.

### HEP + RAG flavor
- **Contextual retrieval.** https://www.anthropic.com/news/contextual-retrieval
- **ragas** — https://docs.ragas.io/

### Tooling for demo recording
- **asciinema** (terminal demos) — https://asciinema.org/
- **QuickTime / OBS** (video demos)
- **gif.ski** (make small gifs from video)

---

## Track B — Generative models

### Docs + libraries
- **PyTorch** — https://pytorch.org/
- **diffusers (HuggingFace).** https://huggingface.co/docs/diffusers — reference for clean diffusion code.
- **lightning** or raw loops — your call; raw is fine for this scale.
- **wandb** or **weave** — track training.

```
uv add torch einops wandb
uv add diffusers accelerate      # optional, for reference and utilities
```

### Papers (repeat from reading.md; here with direct arXiv links)
- **DDPM** — https://arxiv.org/abs/2006.11239
- **Score-SDE** — https://arxiv.org/abs/2011.13456
- **EDM (Karras)** — https://arxiv.org/abs/2206.00364
- **VAE (Kingma)** — https://arxiv.org/abs/1312.6114
- **CaloGAN** — https://arxiv.org/abs/1705.02355
- **CaloFlow** — https://arxiv.org/abs/2106.05285

### Tutorials
- **The Annotated Diffusion Model.** HuggingFace blog. https://huggingface.co/blog/annotated-diffusion
- **Tony Duan — *Diffusion models from scratch*.** https://www.tonyduan.com/diffusion/index.html
- **Lilian Weng — *What are diffusion models*.** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- **DeepMind EMA — *A tutorial on energy-based models*.** Background if curious.

### Physics-eval tooling
- **scipy.stats** for χ², Wasserstein.
- **POT (Python Optimal Transport).** https://pythonot.github.io/ — EMD for multivariate.

---

## Track C — Paper-to-pipeline

### Docs + libraries
- **Instructor** — https://python.useinstructor.com/ — structured outputs with Pydantic.
- **Pydantic.** https://docs.pydantic.dev/
- **marker-pdf** (from W08). https://github.com/VikParuchuri/marker
- **pytest** (tests of generated code).
- **subprocess / docker** for sandboxed runs.

```
uv add instructor anthropic pydantic
uv add marker-pdf
```

### Papers / prior art
- **AlphaCode** — https://arxiv.org/abs/2203.07814
- **Codex (HumanEval)** — https://arxiv.org/abs/2107.03374
- **SWE-Bench** — https://arxiv.org/abs/2310.06770
- **SWE-Agent** — https://arxiv.org/abs/2405.15793
- **Reflexion** — https://arxiv.org/abs/2303.11366
- **Voyager** — https://arxiv.org/abs/2305.16291
- **Self-Refine** — https://arxiv.org/abs/2303.17651

### Code-gen eval
- **HumanEval** dataset — https://github.com/openai/human-eval
- **MBPP** — https://github.com/google-research/google-research/tree/master/mbpp
- **BigCodeBench** — https://github.com/bigcode-project/bigcodebench
- (You won't eval against these directly — your eval is on HEP papers — but the methodology is transferable.)

### Sandboxing
- **e2b.dev** — hosted code sandboxes. https://e2b.dev/ — free tier covers capstone scale.
- **Docker Python SDK** — if you prefer local.
- **runsc (gVisor).** https://gvisor.dev/ — lightweight VM-ish container; overkill for capstone but know it exists.

---

## Cost awareness

| Track | Primary cost | Rough capstone budget |
| --- | --- | --- |
| A | Claude API for eval harness (20 tasks × 5 trials × $0.02) | ~ $5 |
| B | GPU time (Colab free or local); no API | $0–$5 |
| C | Claude API for extraction + generation + repair (5 papers × 3 rounds × $0.05) | ~ $10–$20 |

If you're on Colab free tier for Track B, 50k × 10-obs training fits in an A100 session easily; on a T4 slower but still fine.

## Security

- **Track A.** MCP tools inherit your user perms. Default read-only. Audit every tool for side effects.
- **Track B.** Mostly not a security concern; just don't commit checkpoints with personal data.
- **Track C.** Running LLM-generated code is the whole game. Use a sandbox (e2b or Docker). Never `subprocess.run(generated_code)` on your host without isolation.

## Troubleshooting

- **Track A.** "Claude Desktop doesn't see new tools after edits." Fully quit (⌘Q) and relaunch; it caches server processes.
- **Track A.** "Tool returns but Claude never mentions result." Tool's return value isn't serializable; Claude sees an error it doesn't surface. Log to stderr and inspect.
- **Track B.** "VAE collapses (σ→0, posterior ignored)." Use β-warm-up; check your reconstruction term isn't dwarfing KL.
- **Track B.** "Diffusion loss plateau at random noise." Sampling loop bug (schedule mismatch between train and sample). Re-read Karras.
- **Track C.** "Every paper extraction looks the same." Your prompt's too generic. Give the model a concrete paper-schema example few-shot.
- **Track C.** "Repair loop spins forever fixing the same error." Cap iterations. Also: check you're actually passing in the *new* error, not the old one.

## After the course — what to read next

In rough order of growing ambition:

1. **MLSys** short courses — Chip Huyen's *Designing Machine Learning Systems*.
2. **Prince — *Understanding Deep Learning*.** Best comprehensive modern textbook.
3. **Karpathy's *Zero to Hero*** second half (nanoGPT derivative series).
4. **Gemm / FlashAttention / systems papers** — if GPU performance interests you.
5. **Any current heavy-ion ML paper** — now you have the vocabulary.

## Sign-off

This is the last `resources.md`. Week 12 ends the course. Ship `v0.1`. Walk away. Use the thing. Come back next quarter for `v0.2`.
