# Week 12 — Reading

The capstone is mostly *doing*, not reading. This list is therefore short and track-specific. Re-read Weeks 08–11 reading lists if you need a refresher.

## Everyone

1. **Re-read your own `docs/report.md` draft, daily.** Treat it as the contract with yourself.
2. **Anthropic — *Building Effective Agents*.** If your capstone touches agents (all three do, loosely), one more pass is worth it.
3. **Eugene Yan — *Prompting guide for writing better LLM evals*.** https://eugeneyan.com/writing/evals/
4. **Hamel Husain — *Your AI product needs evals*.** https://hamel.dev/blog/posts/evals/
5. **Jason Liu — shipping LLM projects.** https://jxnl.co/writing/

## Track A — HEP Copilot

6. **MCP spec revisited.** https://modelcontextprotocol.io/specification
   - Skim again with an integrator's eye. Anything you glossed over in W11 comes back now.
7. **Anthropic — *How Claude uses tools well*.** (part of the tool use docs tree at docs.claude.com)
8. **One real production MCP server.** Reference implementations `filesystem`, `github`, or `postgres` — pick the one closest to what you're building. Read the source, not just the README.
9. **Contextual retrieval reminder.** https://www.anthropic.com/news/contextual-retrieval
10. **Your W10 `docs/framework-comparison.md`.** The decision you made there is the decision you live with.

## Track B — Generative models

6. **Ho et al. — *Denoising Diffusion Probabilistic Models*.** arXiv:2006.11239.
   - The original DDPM. ~10 pages of careful math.
7. **Song et al. — *Score-Based Generative Modeling through Stochastic Differential Equations*.** arXiv:2011.13456.
   - Generalizes DDPM; read intro + algorithm boxes.
8. **Karras et al. — *Elucidating the Design Space of Diffusion-Based Generative Models*.** arXiv:2206.00364.
   - "EDM". The paper that made diffusion sampling much simpler. Algorithmic pseudocode is what you'll translate.
9. **Kingma + Welling — *Auto-Encoding Variational Bayes*.** arXiv:1312.6114.
   - If going VAE route, this is the foundation.
10. **Rezende + Mohamed — *Variational Inference with Normalizing Flows*.** arXiv:1505.05770.
    - Optional; useful if you want to compare against NFs.

### HEP generative-model papers

11. **Paganini, de Oliveira, Nachman — CaloGAN.** arXiv:1705.02355.
12. **Amram, Nachman, Thaler et al. — *JetGPT / particle transformers for event generation*.** Recent; search arXiv.
13. **Butter et al. — *How to GAN higher jet resolution*.** arXiv:2012.11944. The style of eval is what you want to borrow.
14. **Krause + Shih — *CaloFlow* series.** Normalizing-flow-based calorimeter simulation.

## Track C — Paper-to-pipeline

6. **Li et al. — *Competition-level code generation with AlphaCode*.** arXiv:2203.07814. How the big boys think about code gen.
7. **Chen et al. — *Evaluating Large Language Models Trained on Code* (Codex).** arXiv:2107.03374. HumanEval.
8. **Jimenez et al. — *SWE-Bench*.** arXiv:2310.06770. Agent-driven codebase modification; the closest public benchmark to your task.
9. **Yang et al. — *SWE-Agent*.** arXiv:2405.15793. Agent scaffolding around SWE-Bench.
10. **Shinn et al. — *Reflexion*.** arXiv:2303.11366. The self-critique loop you'll implement.
11. **Wang et al. — *Voyager*.** arXiv:2305.16291. Skill-library pattern, useful if your agent builds reusable helpers.

## Secondary (everyone, skim)

12. **Eugene Yan — *How to interview LLM systems*.** Short. Think of your defense as an interview.
13. **Chip Huyen — *ML test score*.** Vaguely applicable; the culture of testing generalizes.
14. **Simon Willison blog — last 30 days.** Scan for anything that happens to be relevant this week.

## Videos (all optional)

- Karpathy — *Let's build the GPT tokenizer* (for Track C, if you write your own structured-output parser).
- Welch Labs — *But what is a neural network?* (for relaxation when your code is wrong and you need to remember why this is fun).
- Any 3Blue1Brown video you haven't seen.

## Your own notes

Re-read:

- Your Week 06 training log.
- Your Week 08 RAG metrics.
- Your Week 10 framework comparison.

These are the anchors. You're not starting from scratch.

## Reading plan

| Day | Focus |
| --- | --- |
| 1 | Re-read own W10 comparison + track-specific primary paper. |
| 2 | Track-specific secondary. |
| 3–6 | No new reading. Execute. |
| 7 | Read your own report before writing the final pass. |

## What to emphasize vs skim

- **Emphasize.** Eval methodology posts (Yan, Husain). They shape your work directly.
- **Skim.** Theory-heavy papers (e.g., SDE diffusion) unless you're building that specifically.
