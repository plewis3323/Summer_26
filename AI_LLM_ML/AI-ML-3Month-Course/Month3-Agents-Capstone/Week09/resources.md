# Week 09 — Resources

## Docs

- **Anthropic SDK** — https://docs.anthropic.com/en/api/client-sdks
- **Anthropic tool use** — https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
- **Anthropic cookbook (tool use)** — https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
- **Building effective agents** (Anthropic) — https://www.anthropic.com/research/building-effective-agents
- **OpenAI function calling** — https://platform.openai.com/docs/guides/function-calling
- **OpenAI structured outputs** — https://platform.openai.com/docs/guides/structured-outputs
- **Instructor** — https://python.useinstructor.com/

## Libraries

- **anthropic** — `pip install anthropic`
- **openai** — `pip install openai`
- **instructor** — `pip install instructor`
- **pydantic** — `pip install pydantic`
- **uproot** — `pip install uproot`
- **typer** — `pip install typer[all]`
- **rich** — `pip install rich`
- **requests** / **httpx** — standard.

## Papers

- ReAct — arXiv:2210.03629
- Toolformer — arXiv:2302.04761
- Chain-of-Thought — arXiv:2201.11903
- Reflexion — arXiv:2303.11366
- Tree-of-Thoughts — arXiv:2305.10601
- Self-Consistency — arXiv:2203.11171
- LLM-Agent Survey (Wang) — arXiv:2308.11432
- LLM-Agent Survey (Xi) — arXiv:2309.07864
- Cognitive Architectures for Language Agents — arXiv:2309.02427
- Generative Agents (Park) — arXiv:2304.03442
- AutoGPT — see its repo README, more of a product than a paper.

## Security

- **Prompt Injection Attacks against GPT-4** — arXiv:2302.12173
- **Simon Willison's prompt injection tag** — https://simonwillison.net/tags/prompt-injection/
- **"Lethal trifecta" essay** — https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- **OWASP Top 10 for LLMs** — https://owasp.org/www-project-top-10-for-large-language-model-applications/

## Blog posts

- **Anthropic — *Building effective agents* (2024).** Mandatory reading.
- **Lilian Weng — *LLM-powered autonomous agents*.** https://lilianweng.github.io/posts/2023-06-23-agent/
- **Eugene Yan — patterns and tooling posts.** https://eugeneyan.com/
- **Hamel Husain — *Your AI Product Needs Evals*.** https://hamel.dev/blog/posts/evals/

## Videos

- **Anthropic — various agent tutorials on YouTube.**
- **Harrison Chase (LangChain) — agent design talks.**
- **Simon Willison — periodic talks on practical LLM tooling.**
- **LlamaIndex / LangChain office hours** — YouTube streams.

## Sandboxing

- **`subprocess` + `resource.setrlimit`** — minimal, Linux/Mac.
- **Docker** — strongest. `docker run --rm --network=none --memory=1g --cpus=1 python:3.12 python -c "..."`.
- **gVisor** — container isolation.
- **Wasmtime / WASI** — for WASM-compiled Python (Pyodide). Strong sandbox.
- **Jupyter kernel over ZeroMQ** — if you want stateful exec like in Anthropic's computer use tools.
- **E2B** (https://e2b.dev) — hosted code-sandbox API; good if you want to outsource safety.

## Structured output

- **Instructor** — Pydantic wrapper. Simplest.
- **Outlines** — https://github.com/outlines-dev/outlines. Token-level constraint.
- **LM-Format-Enforcer** — alternative constrained decoding.

## Testing and eval

- **pytest** — of course.
- **Deepeval** — https://github.com/confident-ai/deepeval. LLM test framework.
- **Weights & Biases Weave** — tracing and eval for LLMs.
- **LangSmith** — tracing from LangChain folks.
- **Phoenix** (Arize) — OSS tracing.

## HEP-specific

- **INSPIRE-HEP REST API** — https://inspirehep.net/api
- **ArXiv API** — https://arxiv.org/help/api
- **uproot** — https://uproot.readthedocs.io/. Pure-Python ROOT reader.
- **coffea** — https://coffea-hep.readthedocs.io/. Analysis framework on uproot. Worth skimming; you may expose coffea primitives as tools in Week 11.

## Cost awareness

- Claude Sonnet per agent turn: ~$0.003 for a 5-tool trace. Budget $1 = ~300 traces.
- Rate limits apply; Anthropic's tier 1 is plenty for this week.

## Anti-patterns

- **Too many tools.** >15 tools and Claude sometimes picks wrong one. Keep it tight.
- **Overlapping tool purposes.** Consolidate.
- **Tools that fail silently.** Always return a structured error, not None.
- **Tools that return too much.** Truncate. Model context is finite.
- **Agents for one-shot tasks.** A plain LLM call is fine for simple tasks. Reserve agents for genuinely multi-step work.

---

Week 10: agent frameworks. We take what you built from scratch and compare with LangGraph, smolagents, and Claude's built-in agent loop. The exercise clarifies what frameworks add and what they hide.
