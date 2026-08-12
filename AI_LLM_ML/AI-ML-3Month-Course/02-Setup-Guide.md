# 02 — Setup Guide

Environment: WSL2 Ubuntu under Windows, Edge as browser, Jupyter configured to launch
Edge (`~/.jupyter/jupyter_server_config.py`, `use_redirect_file = False`). Set up once
in Week 01; later phases add pieces when first needed — don't pre-install Phase 3
tooling in Month 1.

## Week 01 — base environment

```bash
# uv (Python environment + package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# per-project environment (repeat per project/capstone repo)
uv init && uv add numpy pandas matplotlib jupyter ipykernel
uv add --dev ruff pytest

# run things
uv run jupyter lab
uv run pytest -q
uv run ruff check .
```

Git identity, GitHub SSH key, and a `course-work` GitHub repo (or one repo per
capstone — decide in Week 03 and stick with it).

## Phase-by-phase additions

| When | Add | Notes |
|------|-----|-------|
| Week 09 | `scikit-learn` | plus `xgboost`, `lightgbm` in Week 11 |
| Week 12 | `umap-learn` | |
| Week 15 | `torch` | CUDA build if local NVIDIA GPU; else CPU + Colab |
| Week 21 | `torch_geometric` | follow its install matrix for your torch/CUDA combo |
| Week 27 | Colab/Kaggle account warm | first serious GPU training week |
| Week 29 | `transformers`, `datasets`, `accelerate` | HF account + token (`huggingface-cli login`) |
| Week 30 | `peft`, `bitsandbytes`, `trl` | bitsandbytes needs CUDA; else Colab |
| Week 33 | `sentence-transformers` | |
| Week 34 | `chromadb` or `lancedb` | pick one, don't run both |
| Week 37 | `anthropic` | `ANTHROPIC_API_KEY` in `~/.bashrc`; set a spend cap in the console |
| Week 39 | `mcp` (official Python SDK) | |
| Week 41 | `wandb` or `mlflow` | pick one |
| Week 44 | `fastapi`, `uvicorn`, Docker Desktop (WSL2 backend) | |

## GPU policy

- **Local GPU (if present):** everything through Phase 2 except Week 20 at full scale.
- **Colab/Kaggle free tier:** Weeks 20, 27, 32, 35–36 training runs.
- **Paid burst (optional, ~$20–50 total):** only if a capstone needs it; Lambda/RunPod
  hourly beats a subscription at this usage.

Check CUDA visibility from WSL2: `python -c "import torch; print(torch.cuda.is_available())"`.
If false with an NVIDIA card present, update the Windows NVIDIA driver — WSL2 uses the
Windows driver, not a Linux one.

## Secrets & spend hygiene

- API keys live in environment variables, never in notebooks or git.
- `.gitignore` from day one: `data/`, `*.ckpt`, `wandb/`, `.env`, `__pycache__/`.
- Set billing alerts on every service that has a card attached.

## Reproducibility checklist (applies to every project)

1. `pyproject.toml` with pinned versions (`uv lock` committed).
2. Seeds set for `random`, `numpy`, `torch` (and note where nondeterminism remains —
   GPU ops, dataloader workers).
3. Data downloaded by a script with a checksum, never by hand.
4. One command runs the whole thing: `uv run python run.py`.
5. `pytest -q` green before any tag.
