# Week 10 — Reading

## Primary

1. **LangGraph docs.** https://langchain-ai.github.io/langgraph/
   - Read: **Concepts → Overview**, **How-to guides → Graphs**, **How-to guides → Persistence**, **Tutorials → Agentic Patterns**. ~90 minutes total.
   - The conceptual framing (nodes, edges, state, checkpointing) is what you need before coding.

2. **smolagents docs.** https://huggingface.co/docs/smolagents
   - Read the full *Overview*, *Tutorial*, *Guided Tour*, and *CodeAgent* sections. ~30 minutes.
   - The paradigm is fundamentally different from LangGraph — take it seriously.

3. **Anthropic — *Claude Agent SDK* docs.** https://docs.claude.com/en/docs/claude-code/sdk (naming varies by version; look for agent-sdk)
   - Read the SDK's quickstart and concepts.
   - API reference for `Agent` / `Tool` / `Workflow`.

4. **Anthropic — *Building Effective Agents* (2024).** https://www.anthropic.com/research/building-effective-agents
   - Re-read. With Week 09 and this week's material fresh, you'll see the distinctions (workflows vs agents, composition patterns) more clearly.

## Secondary

5. **LangGraph — *Multi-agent systems* guide.** https://langchain-ai.github.io/langgraph/concepts/multi_agent/
   - Patterns: supervisor, hierarchical, swarm, network. §"When to use multi-agent" is the most important paragraph.

6. **Andrew Ng — *The Agentic Design Patterns* newsletter series (DeepLearning.ai, 2024).** https://www.deeplearning.ai/the-batch/
   - Short pieces on planning, reflection, tool use, multi-agent. Read all four.

7. **Park et al. — *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023). arXiv:2304.03442.**
   - Skim §3–4 for a long-horizon memory design. Useful for capstone.

8. **Shinn et al. — *Reflexion* (NeurIPS 2023). arXiv:2303.11366.**
   - Self-critique loop. One way to add reliability to an agent.

9. **Wu et al. — *AutoGen: Enabling Next-Gen LLM Applications* (2023). arXiv:2308.08155.**
   - Microsoft's multi-agent framework. Read §2–3 for the conversation-based paradigm. Not what you'll use, but the comparison sharpens your view.

10. **Shinn et al. — *OpenDevin* / **Cognition Labs — *Devin* blog.**
    - Examples of what "agent as engineer" looks like when taken seriously. Not required reading but illuminating.

## Tutorials

- **LangChain Academy — LangGraph short courses** (free, on YouTube). "Introduction to LangGraph" and "Long-Term Agentic Memory with LangGraph". ~2 hours total.
- **DeepLearning.ai — *AI Agents in LangGraph*** (short course, free). ~1 hour. Harrison Chase instructs.
- **Anthropic cookbook — agentic examples.** https://github.com/anthropics/anthropic-cookbook/tree/main/agents

## Videos

- **Harrison Chase (LangChain CEO) — various talks on "what is an agent".** YouTube. Skim one 30-minute one.
- **Chip Huyen — *Evaluating LLM Applications*. ~45 min.** Evaluation specifically for agentic systems.

## Blog posts

- **Anthropic — *Building Effective Agents*.** (Listed above; cannot overemphasize.)
- **LangChain blog — *What is a cognitive architecture?*** Definition worth internalizing.
- **Hamel Husain — *Your AI Product Needs Evals*.** On building golden sets and iterating systematically.
- **Simon Willison — *Coding agents*.** https://simonwillison.net/tags/agents/. Short, sharp takes.
- **Lilian Weng — *Agent* post.** https://lilianweng.github.io/posts/2023-06-23-agent/. Revisit; you'll notice layers you missed the first time.

## Papers on memory and planning

- **MemGPT** — arXiv:2310.08560. OS-like memory for LLMs. Skim §3.
- **HuggingGPT** — arXiv:2303.17580. Planning + dispatch. Skim.
- **Voyager** — arXiv:2305.16291. Agent that learns new skills in Minecraft. The "skill library" pattern is interesting.
- **Reflexion** — arXiv:2303.11366. Already listed. Canonical self-correction paper.
- **Self-Discover** — arXiv:2402.03620. Planning by selection from reasoning-primitive library.

## Eval-specific

- **Cassidy Williams — *How to eval agents*.** Short blog posts, recent.
- **LangSmith docs — *Evaluations*.** https://docs.smith.langchain.com/evaluation

## Textbooks

- **Huyen, *Designing Machine Learning Systems* (O'Reilly 2022).** Chapters on production ML generalize to production agent systems.
- **Huyen, *AI Engineering* (O'Reilly 2024).** Newer, agent-era. Chapters on inference, evals, prompting are relevant.

## HEP

Nothing agent-specific. But:
- **Anthropic's computer use documentation.** Hints at the agent UX pattern you might want in your capstone (a physicist pointing an agent at their screen).

## Notes to take

- A decision tree: when should I reach for LangGraph vs smolagents vs raw SDK?
- A diagram of a multi-agent pattern that maps to your HEP workflow.
- A 1-page list: "Memory types I need for an HEP analysis copilot" (session, persistent, user prefs, run metadata index, tool-call history).
- One sketch of how you'd evaluate the agent: what's a golden task? What's "success"?

## Reading plan

| Day | What |
| --- | --- |
| 1 | LangGraph concepts + basic tutorial. |
| 2 | LangGraph agents + multi-agent patterns. |
| 3 | smolagents full docs. |
| 4 | Claude Agent SDK. |
| 5 | Memory papers (MemGPT, Generative Agents). |
| 6 | Evaluation posts and papers. |
| 7 | Reference. |
