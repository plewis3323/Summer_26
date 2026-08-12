# Week 10 — Resources

## Docs

- **LangGraph** — https://langchain-ai.github.io/langgraph/
- **LangChain** — https://python.langchain.com/docs/
- **smolagents** — https://huggingface.co/docs/smolagents
- **claude-agent-sdk** — https://docs.claude.com (navigate to SDK / agents)
- **Weave** — https://weave-docs.wandb.ai/
- **LangSmith** — https://docs.smith.langchain.com/
- **Phoenix (Arize)** — https://docs.arize.com/phoenix

## Libraries

- `langgraph`, `langchain`, `langchain-anthropic`, `langchain-openai`, `langchain-community`
- `smolagents`
- `claude-agent-sdk` (Anthropic)
- `weave`, `langsmith`, `arize-phoenix` (pick one for tracing)
- `crewai`, `autogen-agentchat` (alternates you might sample)

Install set:
```
uv add langgraph langchain langchain-anthropic langgraph-checkpoint-sqlite
uv add smolagents
uv add claude-agent-sdk
uv add weave
```

## Tutorials / courses

- **LangChain Academy — *Introduction to LangGraph*.** Free YouTube course. ~2 hours.
- **LangChain Academy — *Long-Term Agentic Memory*.** Continuation.
- **DeepLearning.ai — *AI Agents in LangGraph*.** Harrison Chase, free.
- **HuggingFace Agents Course.** https://huggingface.co/learn/agents-course/. Free; uses smolagents.
- **Anthropic cookbook.** https://github.com/anthropics/anthropic-cookbook/tree/main/agents

## Papers

- ReAct — arXiv:2210.03629
- Reflexion — arXiv:2303.11366
- Tree-of-Thoughts — arXiv:2305.10601
- Self-Consistency — arXiv:2203.11171
- AutoGen — arXiv:2308.08155
- MetaGPT — arXiv:2308.00352
- Generative Agents — arXiv:2304.03442
- MemGPT — arXiv:2310.08560
- Voyager — arXiv:2305.16291
- Self-Discover — arXiv:2402.03620
- AgentBench — arXiv:2308.03688 (benchmark suite)
- TheAgentCompany — arXiv:2412.14161 (recent, CMU benchmark)

## Blog posts

- **Anthropic — *Building Effective Agents*.**
- **LangChain blog** — https://blog.langchain.com/. Frequent posts on agent design patterns.
- **Hamel Husain — eval posts.** https://hamel.dev/blog/posts/evals/
- **Eugene Yan — patterns posts.** https://eugeneyan.com/writing/llm-patterns/
- **Simon Willison — agents tag.** https://simonwillison.net/tags/agents/

## Videos

- **Harrison Chase talks.** YouTube.
- **Matthew Berman** (YouTuber) — cursory but often has hands-on with new frameworks.
- **Cole Medin** — practical builds with LangGraph and others.
- **Anthropic's "How I'd Explain It To You" series.**

## Tracing / observability

- **Weave** — free, OSS, simple. `weave.init("project")` + decorators.
- **LangSmith** — free tier available, LangChain-native.
- **Phoenix** — OSS, self-hosted, vendor-neutral.
- **OpenTelemetry + Jaeger/Honeycomb** — if you want vendor-neutral tracing in a broader observability stack.

## Datasets / benchmarks

- **AgentBench** (THU) — multi-environment benchmark.
- **GAIA** — Meta's real-world agent benchmark.
- **SWE-Bench** — software-engineering agent benchmark (capstone-adjacent).
- **WebArena / VisualWebArena** — browser agents.

Not required reading this week; mentioned for capstone context.

## Other agent frameworks worth sampling

- **CrewAI.** Role-based multi-agent.
- **AutoGen / AutoGen-AgentChat.** Microsoft's.
- **Swarm** (OpenAI). Minimalist multi-agent pattern.
- **Letta** (formerly MemGPT). Memory-focused.
- **ControlFlow** (prefect.io). Workflow + agent hybrid.
- **dspy.** Different paradigm — "programming, not prompting".

Sample *one* extra; the point is perspective, not comprehensiveness.

## Cost awareness

- LangGraph + Claude: same per-call cost as raw SDK. No framework overhead.
- smolagents CodeAgent: often fewer LLM calls per task (the agent writes Python loops once), so sometimes *cheaper*.
- Agent SDK: same pricing as raw SDK.
- Tracing (Weave/LangSmith): can add tokens for summarization; usually negligible.

Budget $5–10 for a full Week 10 run with 20 goldens × 4 frameworks × ~$0.004 = ~$0.32, plus iteration.

## Security

- LangGraph's `ToolNode` dispatches whatever the model emits. Your tool implementations must still be hardened.
- smolagents' CodeAgent runs arbitrary Python. Restrict `additional_authorized_imports` carefully. For untrusted use, run inside Docker.
- Agent SDK is Anthropic's sandboxing abstraction; use it as intended.

## Troubleshooting

- **LangGraph state not updating.** You're not returning the right key in the node. Nodes return dicts merged into state.
- **smolagents won't use a tool.** Function is missing a docstring or has unusual types. Check with `agent.tools` to see what it actually knows.
- **Agent SDK behaves differently across versions.** Pin with `uv add claude-agent-sdk==x.y.z`.
- **Weave not capturing calls.** `weave.init()` before importing your LLM client.

---

Week 11: MCP. You expose your tools as standard servers. Claude Desktop, Cursor, Zed — anyone speaking MCP — can then use them. The HEP analysis tools you'd use daily become plug-and-play.
