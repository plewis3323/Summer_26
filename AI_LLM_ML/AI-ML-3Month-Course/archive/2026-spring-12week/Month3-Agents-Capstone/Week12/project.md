# Week 12 — Capstone Project: One Track, Shipped

## Summary

Pick one of three tracks. Spend one to two weeks turning your course-long work into a coherent, documented, tested, installable artifact. Demonstrate it. Evaluate it. Write it up.

## Common requirements (all tracks)

### Repository layout

```
capstone-{track}/
├── pyproject.toml
├── README.md                  # install + quickstart + demo gif
├── src/capstone/
├── tests/
├── eval/
│   ├── tasks.jsonl            # or similar — your eval set
│   └── run_eval.py
├── docs/
│   ├── report.md              # 800-1200 word write-up
│   ├── deck/deck.pdf          # 8 slides
│   ├── architecture.png       # box diagram
│   └── failure_modes.md       # ≥5 modes enumerated
├── scripts/
│   └── demo.py                # one-command demo
└── .github/workflows/ci.yml   # nice-to-have
```

### Acceptance criteria (universal)

1. `uv sync` works on a clean clone (on your laptop, not just your dev dir).
2. `uv run pytest -q` passes.
3. A single command reproduces a demo: `uv run python scripts/demo.py`.
4. `docs/report.md` exists with ≥800 words covering motivation, architecture, results, limits.
5. 8-slide deck exists in `docs/deck/`.
6. `docs/failure_modes.md` lists ≥ 5 explicit failure modes.
7. `eval/` contains a ≥ 10-item eval set with a `run_eval.py` that produces a one-line summary.
8. Git tag `v0.1`.

---

## Track A — HEP Copilot

### What you build

An MCP server `hep-copilot` that Claude Desktop (and anyone speaking MCP) can use to answer questions about runs and papers. Built on W09 (tools) + W10 (frameworks — optional) + W11 (MCP) + W08 (RAG).

### Capabilities

**Tools.**

| Name | Notes |
| --- | --- |
| `query_runs_db(sql)` | SELECT-only, 500-row cap. |
| `inspect_root(path)` | uproot summary of a ROOT file. |
| `read_branch_summary(path, tree, branch)` | stats on one branch. |
| `run_python(code, deps=[])` | sandboxed, 10s / 1GB. |
| `plot_histogram(values, bins, title)` | save PNG under `outputs/`. |
| `search_papers(query, k=5)` | W08 retriever wrapped. |
| `cite_passage(passage_id)` | full chunk by ID. |
| `fetch_arxiv(arxiv_id)` | abstract + basic metadata. |

**Resources.**

| URI | Returns |
| --- | --- |
| `runs://recent` | markdown table |
| `run://{run_id}` | JSON record |
| `paper://{arxiv_id}` | structured metadata + abstract |
| `notes://{note_id}` | user markdown notes |
| `schema://runs.db` | DDL |

**Prompts.**

| Slash | Purpose |
| --- | --- |
| `/run_summary {run_id}` | 3-sentence summary |
| `/qc_report {start} {end}` | date-range QC table |
| `/paper_digest {arxiv_id}` | 150-word paper digest |
| `/analysis_plan {topic}` | starter outline for a new analysis |

### Acceptance criteria (track-specific)

- Server starts and Claude Desktop lists it with ≥ 8 tools.
- 20-task eval in `eval/tasks.jsonl`; `run_eval.py` reports pass@1 ≥ 70%.
- At least one real task during the build week: solved a real question with hep-copilot, documented in `docs/report.md`.
- `fetch_arxiv` / any URL-fetching tool enforces domain whitelist and has a security test.
- A screenshot of Claude Desktop using the tool in `docs/screenshot.png`.

### Minimal server.py

```python
from mcp.server.fastmcp import FastMCP
from hep_copilot.tools import (
    runs_db, rootio, python_exec, plotting, papers, citations
)
from hep_copilot.resources import runs, papers as paper_res, notes, schema
from hep_copilot.prompts import run_summary, qc_report, paper_digest, analysis_plan

mcp = FastMCP("hep-copilot")

for t in (
    runs_db.query_runs_db, rootio.inspect_root, rootio.read_branch_summary,
    python_exec.run_python, plotting.plot_histogram,
    papers.search_papers, papers.fetch_arxiv, citations.cite_passage,
):
    mcp.tool()(t)

mcp.resource("runs://recent")(runs.recent)
mcp.resource("run://{run_id}")(runs.record)
mcp.resource("paper://{arxiv_id}")(paper_res.record)
mcp.resource("notes://{note_id}")(notes.get)
mcp.resource("schema://runs.db")(schema.ddl)

mcp.prompt()(run_summary.run_summary)
mcp.prompt()(qc_report.qc_report)
mcp.prompt()(paper_digest.paper_digest)
mcp.prompt()(analysis_plan.analysis_plan)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

### Eval harness sketch

```python
# eval/run_eval.py
import json, asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def score_one(task):
    opts = ClaudeAgentOptions(
        mcp_servers={"hep-copilot": {"type": "stdio", "command": "uv",
                                     "args": ["--directory", ".", "run", "python", "-m", "hep_copilot.server"]}},
        allowed_tools=[f"mcp__hep-copilot__{t}" for t in task["allowed_tools"]],
    )
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(task["q"])
        text = ""
        tools_called = []
        async for msg in client.receive_response():
            text += getattr(msg, "text", "")
            tools_called.extend(getattr(msg, "tool_calls", []))
    pass_keywords = all(k.lower() in text.lower() for k in task["expected_keywords"])
    pass_tools = set(task["expected_tools"]).issubset({tc["name"] for tc in tools_called})
    return {"pass": pass_keywords and pass_tools, "q": task["q"]}

async def main():
    tasks = [json.loads(l) for l in open("eval/tasks.jsonl")]
    results = [await score_one(t) for t in tasks]
    n = len(results); k = sum(r["pass"] for r in results)
    print(f"pass@1: {k}/{n} ({100*k/n:.1f}%)")

asyncio.run(main())
```

### Sample eval tasks

```jsonl
{"q":"List the five most recent good AuAu runs.","allowed_tools":["query_runs_db"],"expected_tools":["query_runs_db"],"expected_keywords":["run_id","AuAu"]}
{"q":"Summarize the abstract of arXiv 2402.10475.","allowed_tools":["fetch_arxiv"],"expected_tools":["fetch_arxiv"],"expected_keywords":["abstract","2402.10475"]}
{"q":"Plot the energy_gev histogram for all good runs in 2025.","allowed_tools":["query_runs_db","run_python","plot_histogram"],"expected_tools":["plot_histogram"],"expected_keywords":["histogram","energy"]}
```

---

## Track B — Conditional Generative Model for Detector Surrogates

### What you build

`hep-gen`: a conditional generator (VAE or diffusion) for per-event observables. Conditioning: `(species, sqrt_s_nn, centrality_bin)`. Targets: 8–12 per-event scalars (n_tracks, mean_pT, v2, v3, <pT^2>, epsilon2, ...).

### Dataset

Synthetic but physics-flavored:

```python
# scripts/make_data.py
import numpy as np, pandas as pd

N = 50_000
rng = np.random.default_rng(0)

species = rng.choice(["AuAu","pAu","pp"], size=N, p=[0.6,0.2,0.2])
sqrts = np.where(species=="pp", 200., np.where(species=="pAu", 200., 200.))
centrality = rng.integers(0, 10, N)   # 0=central, 9=peripheral

# physics-ish dependencies (toy)
base_ntracks = {"AuAu":1500, "pAu":200, "pp":30}
lam = np.array([base_ntracks[s] for s in species]) * (1 - 0.08*centrality)
n_tracks = rng.poisson(lam)

mean_pT = 0.5 + 0.15*rng.standard_normal(N) + 0.05*(n_tracks/1000)
v2 = np.clip(0.02 + 0.01*centrality + 0.01*rng.standard_normal(N), 0, 0.2)
v3 = np.clip(0.01 + 0.005*rng.standard_normal(N), 0, 0.1)
eps2 = v2/0.25 + 0.01*rng.standard_normal(N)
mean_pT2 = mean_pT**2 + 0.05*rng.standard_normal(N)

df = pd.DataFrame({
    "species":species, "sqrts":sqrts, "centrality":centrality,
    "n_tracks":n_tracks, "mean_pT":mean_pT, "mean_pT2":mean_pT2,
    "v2":v2, "v3":v3, "eps2":eps2,
})
df.to_parquet("data/events.parquet")
```

### Model options

**VAE (easier).** Encoder `q(z|x,c)` = 2×256 MLP → (μ, logσ). Decoder `p(x|z,c)` = same shape. ELBO with β-warm-up from 0 → 1 over 10k steps.

**Diffusion (harder, cooler).** 1D DDPM with T=200 steps. Denoiser is an MLP conditioned on noise level `t` (sinusoidal embed) and metadata `c`. Karras EDM schedule if you want smoother training.

### Acceptance criteria (track-specific)

- Dataset saved as `data/events.parquet` with ≥ 50k rows.
- Trained model checkpoint in `checkpoints/`.
- Sample script `scripts/sample.py --cond '{"species":"AuAu","centrality":0}' --n 10000` produces samples.
- Per-observable χ² vs truth reported; ≤ 1.5× baseline GMM χ².
- Pairwise correlation Frobenius error < baseline.
- Conditioning test: samples for `centrality=0` and `centrality=9` differ in mean n_tracks by ≥ 5σ.
- 3 qualitative figures in `docs/`: marginals overlay, 2D scatter (n_tracks vs mean_pT, data vs samples), conditioning response.

### Eval sketch

```python
# eval/run_eval.py
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency

real = pd.read_parquet("data/events.parquet")
gen  = pd.read_parquet("outputs/samples.parquet")

cols = ["n_tracks","mean_pT","v2","v3","eps2"]
for c in cols:
    hist_r, edges = np.histogram(real[c], bins=40)
    hist_g, _     = np.histogram(gen[c],  bins=edges)
    eps = 1e-6
    chi2 = ((hist_r - hist_g)**2 / (hist_r + hist_g + eps)).sum()
    print(f"{c:10s} chi2={chi2:.1f}")
```

---

## Track C — Paper-to-Pipeline

### What you build

`paper2pipeline`: input arXiv ID or local PDF, output a runnable Python project directory with pinned deps, a `main.py`, tests, and a README.

### Pipeline

1. **Extract.** Read paper; call Claude with a Pydantic `AnalysisSpec` schema; get structured spec back.
2. **Plan.** Translate spec into a file plan (a list of files with docstrings).
3. **Generate.** For each file, call Claude with the spec + plan + neighbor files; get code.
4. **Assemble.** Write files to temp dir; run `uv sync` (if pyproject); run `pytest`.
5. **Repair.** If tests fail, send error log + failing file back to Claude; apply patch. Up to 5 iterations.

### Schema sketches

```python
from pydantic import BaseModel, Field
from typing import Literal

class Cut(BaseModel):
    variable: str
    op: Literal["<", "<=", ">", ">=", "==", "!=", "in"]
    value: str | float | list

class Observable(BaseModel):
    name: str
    formula: str                 # e.g. "sum(pT)/N" or "mean(eta)"
    unit: str

class AnalysisSpec(BaseModel):
    title: str
    dataset: str                 # e.g. "AuAu sqrt(s)=200 GeV, 2024 run"
    cuts: list[Cut]
    observables: list[Observable]
    expected_trend: str = Field(description="qualitative, one sentence")

class FilePlan(BaseModel):
    path: str                    # relative
    docstring: str
    imports: list[str]
    public_fns: list[str]
```

### Acceptance criteria (track-specific)

- `capstone run arxiv:<id>` produces a valid project directory with `uv sync && uv run pytest -q` passing.
- Repair loop runs ≥ once on at least one test case.
- Eval on 5 papers: tabulate (extraction quality 1–5 manual, compiles Y/N, smoke passes Y/N, structure-match 1–5 manual).
- Target: ≥ 3/5 compile and pass smoke tests.
- Guardrails: model can only emit Python into a restricted set of paths (no `/etc`, no network writes).

### Eval sketch

```python
# eval/run_eval.py
import subprocess, json, tempfile
PAPERS = ["2402.10475", "2403.09080", "2404.02892", "2405.11819", "2406.00192"]
results = []
for pid in PAPERS:
    with tempfile.TemporaryDirectory() as td:
        ok = subprocess.run(["paper2pipeline", f"arxiv:{pid}", "-o", td]).returncode == 0
        pytest_ok = subprocess.run(["uv","run","pytest","-q"], cwd=td).returncode == 0 if ok else False
        results.append({"paper": pid, "generated": ok, "smoke": pytest_ok})
print(json.dumps(results, indent=2))
```

---

## Write-up template (`docs/report.md`)

```markdown
# Capstone: {title}

## Motivation
(1 paragraph — what problem, who cares, why now)

## Architecture
(1 paragraph + 1 diagram)

## Implementation
(what you built, which W01–W11 components you reused, where you deviated from the plan)

## Evaluation
(methodology + numbers)

## Failure modes
(3-5 explicit ones with repros)

## Limitations
(what this isn't; what I'd need to do v0.2)

## Lessons
(what surprised me; what I'd do differently)

## Next steps
(4-6 bullets)
```

## Deck template (`docs/deck/`)

| Slide | Content |
| --- | --- |
| 1 | Title + 1-sentence pitch |
| 2 | Motivation (why this matters to me / the group) |
| 3 | Architecture diagram |
| 4 | Demo (screenshot / gif / embedded video) |
| 5 | Eval setup |
| 6 | Eval results table |
| 7 | Failure modes + next steps |
| 8 | Backup: technical detail or math |

## Sign-off

Tag `v0.1`. Update `04-Progress-Tracker.md`. Commit. Push.

Congratulations — you're done with the 12-week course.

Now use the thing.
