# 02 — Setup Guide

One-time environment setup for the whole course. Budget ~2 hours. Do this before Week 01.

## 0. What you're installing and why

| Tool | Role | Why this one |
|------|------|--------------|
| `uv` | Python package + project manager | 10–100× faster than pip; writes reproducible `uv.lock`; handles Python itself |
| `conda` / `micromamba` | Alternative env manager for CUDA-sensitive stacks | Some ML libraries (older PyTorch wheels, ROOT) ship easier via conda-forge |
| `ruff` | Linter + formatter | Replaces `black` + `isort` + `flake8`, written in Rust, near-instant |
| `pytest` | Test runner | Standard; you will write tests for each project |
| `pre-commit` | Git hook runner | Prevents "forgot to format" commits |
| `jupyterlab` | Notebooks | For exploratory work only; projects live in `.py` files |
| `git` + GitHub CLI (`gh`) | Version control | You already know git; `gh` makes repo creation from CLI trivial |
| CUDA drivers + cuDNN | GPU compute | For Weeks 04, 06, 07, 12 |
| HuggingFace account + token | Model hub auth | For Weeks 07–08, 11, 12 |
| Anthropic API key | Agent development | For Weeks 09–12 |
| OpenAI or OpenRouter (optional) | Cross-model testing | Comparative agent runs |

You do **not** need Docker for this course. It adds overhead you don't need yet.

## 1. Install `uv`

`uv` is the fastest way to manage Python versions and per-project virtual environments. Install it once, globally.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
# uv 0.4.x or later
```

Install a Python version under `uv`'s control (so it doesn't fight with system Python):
```bash
uv python install 3.11
```

Python 3.11 is the current sweet spot for ML libraries — 3.12 works, 3.13 is still catching up on some wheels as of early 2026.

## 2. Course-wide directory layout

```
AI-ML-3Month-Course/
├── .venv/                    ← course-wide venv (optional; or per-project)
├── pyproject.toml            ← course-wide baseline deps (optional)
├── Month1-Foundations/
│   └── Week02/
│       └── project/
│           ├── pyproject.toml   ← per-project
│           └── src/
└── ...
```

Two options:

**Option A (recommended): one venv per week-project.** More reproducible; you'll end up with 12 `pyproject.toml` files. Each project's README tells you what's in it.

**Option B: one venv for the whole course.** Faster to set up; risk of dependency collisions when some week pins `transformers==4.38` and another wants `4.52`.

We'll use Option A. Each week's `project/` has its own `pyproject.toml`.

## 3. Create your first environment (sanity check)

```bash
cd AI-ML-3Month-Course/
mkdir -p scratch && cd scratch
uv init --python 3.11
uv add numpy pandas matplotlib scikit-learn jupyterlab ipykernel pytest ruff
uv run python -c "import numpy, pandas, sklearn; print('ok')"
```

If that prints `ok`, your Python stack works.

## 4. CUDA / PyTorch

### Check what you have
```bash
nvidia-smi
```
Note the "CUDA Version" in the top right. That's the *driver* CUDA capability, not the CUDA toolkit version you need to match. You can install PyTorch built for CUDA 12.1 on a driver reporting 12.4, for instance.

### Install PyTorch (in a project venv)
PyTorch has its own wheel index. Pick the right line from https://pytorch.org/get-started/locally/.

Typical 2026 invocation for a Linux box with an NVIDIA card and CUDA 12.1 drivers or newer:

```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only fallback (works on a Mac or a laptop without a dGPU):
```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify CUDA from PyTorch
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
```

**Pitfall:** if `is_available()` is `False` but `nvidia-smi` shows your card, you probably installed a CPU wheel. Run `pip show torch` and inspect `Location` and the wheel name — it'll have `+cpu` in it if it's wrong.

### Apple Silicon
If you're on an M-series Mac, PyTorch has MPS backend. It works for small nets (Weeks 03–04) and is slower than CUDA for large ones. You cannot do QLoRA on MPS as of 2026-Q1 — Week 07 must go to Colab/Kaggle.

```python
import torch
torch.backends.mps.is_available()  # True on M-series
```

## 5. Colab and Kaggle

You will need cloud GPUs for Weeks 04 (optional), 06 (helpful), 07 (required), 12 (likely).

### Colab
- Free tier: T4 GPU, ~12h runtime, ~15 GB RAM. Session dies if idle.
- Colab Pro ($10/mo): more priority on T4/A100, longer sessions. Worth it for Weeks 06–07.
- Setup: mount Google Drive, install packages per-notebook, use HF token via `getpass`.

### Kaggle Notebooks
- Free: 30h/week of GPU (P100 or T4 ×2).
- Strict internet policy; enable internet in notebook settings for HF downloads.
- Good for Week 04 and Week 07 if you avoid Colab.

**Template for cloud notebooks:** commit the same notebook to your repo under `Weekxx/notebooks/` so the work is versioned.

## 6. HuggingFace

Create an account at https://huggingface.co.

1. Settings → Access Tokens → New token. Name it `ai-ml-3month-course`, scope **Write** (you'll upload models).
2. Accept terms for **Llama 3.2** (gated) at https://huggingface.co/meta-llama/Llama-3.2-1B. This can take hours.
3. Store the token:
   ```bash
   huggingface-cli login
   ```
   or, in a notebook:
   ```python
   from huggingface_hub import login
   login()  # paste token
   ```
4. Alternative: put it in `~/.huggingface/token` or a `.env` you never commit.

**Safety:** your HF token should not live in git. Add `.env` to `.gitignore` in every project (`uv init` does this by default).

## 7. Anthropic API

For Weeks 09–12. Create an account at https://console.anthropic.com, get a key, set:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Put it in your shell rc (`~/.zshrc` or `~/.bashrc`). Never commit it.

Budget a few dollars for capstones. At 2026 prices, Claude Sonnet 4.6 is the workhorse for agents; Haiku 4.5 is fine for routing.

## 8. Code quality tooling

In every project:

```bash
uv add --dev ruff pytest pytest-cov pre-commit ipykernel
```

### `.ruff.toml` (starter)
```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "I", "UP", "B", "SIM", "C90"]
ignore = ["E501"]  # we manage line length via formatter

[format]
quote-style = "double"
```

### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

Then:
```bash
pre-commit install
```

### Running tests
```bash
uv run pytest -q
```
`-q` is quiet mode. Good default.

## 9. Git hygiene

```bash
cd AI-ML-3Month-Course
git init
git add .
git commit -m "scaffold course"
gh repo create ai-ml-3month --private --source=. --push
```

`.gitignore` at course root (minimum):
```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
wandb/
lightning_logs/
*.ckpt
*.pt
*.bin
*.safetensors
data/raw/
data/interim/
```

**What to commit:** source code, notebooks (cleaned outputs), configs, small data (<10 MB), figures.
**What to ignore:** model checkpoints, large datasets, API keys, caches.

### Jupyter outputs in git
Clear outputs before committing:
```bash
jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb
```
Or add this to `.pre-commit-config.yaml` via `nbstripout`.

## 10. Reproducibility discipline

You already know this from physics. Restate for ML:

- **Seed everything.** Every script starts with:
  ```python
  import random, numpy as np, torch
  SEED = 42
  random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
  Caveat: full determinism on GPU costs ~10–30% performance. For Week 04+ you can relax this.
- **Pin dependencies.** `uv.lock` does this for you. Commit it.
- **Log the git SHA and config into every experiment output.** In JSON, `git_sha` and `wall_clock`.
- **Experiment tracking.** Week 04 onward, adopt `wandb` (free academic tier) or `mlflow`. `wandb` is lighter to start.

## 11. Hardware quick sheet

| Task | Fits on | Notes |
|------|---------|-------|
| Weeks 01–03 | CPU laptop | numpy, sklearn, micrograd, tiny MLPs |
| Week 04 CNN train | Local GPU OK; ~15 min on T4 | Use Colab if local is slower |
| Week 06 nanoGPT | Local GPU slow; T4 ~30 min; A100 <5 min | Don't overtrain; 1–2 epochs of ~10M tokens is plenty |
| Week 07 QLoRA 1B | T4 borderline OOM; A100 fine | 4-bit quantization + LoRA is key |
| Week 08 embeddings | CPU works | BGE-small is ~33 M params |
| Weeks 09–11 agents | CPU; API-bound | GPU only if using local LLMs for a tool |
| Week 12 (a) or (c) | CPU; API-bound | — |
| Week 12 (b) diffusion | Local GPU insufficient for full train | Use CaloChallenge small subset on Colab Pro |

## 12. VS Code (or your editor)

Extensions worth installing:
- Python (Microsoft)
- Ruff
- Jupyter
- GitLens
- GitHub Copilot (optional — off if you want to actually learn)
- Even Better TOML

`settings.json` snippets:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  }
}
```

## 13. Smoke-test script

Create `scripts/smoke_test.py` in the scratch project:

```python
"""Smoke test the course environment."""
from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    import numpy as np
    print(f"numpy: {np.__version__}")
    x = np.random.randn(1000, 1000)
    print(f"  100x100 matmul norm: {np.linalg.norm(x @ x):.3f}")

    import pandas as pd
    print(f"pandas: {pd.__version__}")

    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")

    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("  MPS available")
    except ImportError:
        print("torch: NOT INSTALLED")


if __name__ == "__main__":
    main()
```

Run:
```bash
uv run python scripts/smoke_test.py
```

If everything prints and CUDA/MPS is detected, you're ready for Week 01.

## 14. Common setup failures

- **`ModuleNotFoundError: No module named 'torch'` inside a notebook** — you launched Jupyter from the wrong Python. Run `python -m ipykernel install --user --name course-week02` once per env and pick that kernel.
- **`libcuda.so.1: cannot open shared object file`** — wrong CUDA wheel. Reinstall with the matching index URL.
- **`OOM: CUDA out of memory` during fine-tuning** — reduce batch size; enable gradient accumulation; for Week 07, enable 4-bit quantization via `bitsandbytes`; last resort, move to Colab A100.
- **HF gated model 403** — you didn't accept the license or your token scope is `Read` not `Write`.
- **Slow Jupyter on Windows** — use WSL2 if possible. Native Windows Python works but the tooling around CUDA is fussier.

---

When this file is green, move to `Month1-Foundations/Week01/README.md`.
