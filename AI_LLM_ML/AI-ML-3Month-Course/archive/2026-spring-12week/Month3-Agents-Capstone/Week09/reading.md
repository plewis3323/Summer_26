# Week 09 — Reading

## Primary

1. **Anthropic — *Tool use (function calling) overview*.** https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
   - Read all of it. The canonical reference for how Anthropic's tool use works. Roughly 30 minutes.

2. **Anthropic — *Build a tool-use chatbot* cookbook.** https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
   - Multiple worked examples. Read `01_tool_use_overview.ipynb` and `03_agentic_workflows.ipynb`. Source-of-truth for idiomatic Claude tool use.

3. **OpenAI — *Function calling* guide.** https://platform.openai.com/docs/guides/function-calling
   - Read alongside Anthropic's docs; the patterns are identical, the API shape differs. Knowing both lets you port agents across providers.

4. **Yao et al. — *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023). arXiv:2210.03629.**
   - The paper that kicked off tool-use agents. Read §1–3; the Thought/Action/Observation pattern is still the mental model 2+ years later.

5. **Schick et al. — *Toolformer: Language Models Can Teach Themselves to Use Tools* (NeurIPS 2023). arXiv:2302.04761.**
   - Alternative framing: train the model to insert tool calls in natural language. Read §1–3. You won't implement it, but it's an important counterpoint to the "prompt + schema" approach you'll use.

6. **Wei et al. — *Chain of Thought Prompting Elicits Reasoning in Large Language Models* (NeurIPS 2022). arXiv:2201.11903.**
   - CoT is the underpinning of most agent reasoning. Read §3 and §5. Know what CoT is and when it helps (complex multi-step reasoning) and when it hurts (short, single-shot tasks).

## Secondary

7. **Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning* (NeurIPS 2023). arXiv:2303.11366.**
   - A nice pattern: self-critique loop. Read §1–4.

8. **Wang et al. — *A Survey on Large Language Model based Autonomous Agents* (2023). arXiv:2308.11432.**
   - Taxonomy of agent designs. Skim §3 (components: profile, memory, planning, action).

9. **Xi et al. — *The Rise and Potential of Large Language Model Based Agents* (2023). arXiv:2309.07864.**
   - Alternative survey, more philosophical. Skim if you want the "why agents" motivation written out.

10. **Sumers et al. — *Cognitive Architectures for Language Agents* (2023). arXiv:2309.02427.**
    - Heavier. Useful if you're interested in agent design abstractions. Skim §3.

11. **Park et al. — *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023). arXiv:2304.03442.**
    - Not tool use per se, but the most famous early demonstration of multi-agent systems. Good orientation.

## Anthropic-specific

- **Anthropic — *Building effective agents*.** https://www.anthropic.com/research/building-effective-agents
  - A 2024 post laying out practical patterns. **Must-read.** ~20 minutes. The distinction between "workflows" and "agents" is important.

- **Anthropic — *Extended thinking* docs.** https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
  - Claude's scratchpad reasoning. When to enable it for agent work.

- **Anthropic — *Prompt engineering overview*.** https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
  - Read the full section. Tool use lives inside the prompt ecosystem.

## Tutorials

- **Instructor library docs.** https://python.useinstructor.com/. Turns Anthropic / OpenAI calls into Pydantic-validated structured outputs with retries. Read the quickstart.

- **LangChain tool-calling docs.** https://python.langchain.com/docs/how_to/tool_calling/. If you end up using LangChain (you might in Week 10), this is required reading.

## Videos

- **Harrison Chase (LangChain founder) — agent pattern talks.** Several on YouTube. ~30 min each.
- **Andrej Karpathy — *Let's build a Claude agent from scratch* (if released).** Haven't seen one by the cutoff date; check.
- **Simon Willison — *Tools I use daily*.** Ongoing blog series on tool-using LLMs. https://simonwillison.net/

## Blog posts

- **Simon Willison — *The lethal trifecta for AI agents* (2024).** https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/. Short, sobering piece on agent security: private data + untrusted content + external communication = danger. Read it.
- **Eugene Yan — *Patterns for Building LLM-based Systems*.** The agents section specifically.
- **Lilian Weng — *LLM Powered Autonomous Agents*.** https://lilianweng.github.io/posts/2023-06-23-agent/. Encyclopedia-level overview. ~45-minute read.
- **The Instructor library author blog** — on structured outputs in practice.

## Textbooks

No textbook has caught up on agents yet. Closest: Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) has some relevant chapters on system design that generalize.

## Codecademy

"Agents with Claude" (if present) — worth doing. Otherwise skip.

## Security / safety reading

- **Greshake et al. — *Prompt Injection Attacks against GPT-4* (2023).** arXiv:2302.12173. Learn what's possible.
- **Simon Willison — *Prompt injection explained* series.** Readable, practical.

## Notes to take

- A table of tool schemas for 5 canonical tools — what parameters, when used, what errors they can return.
- A flowchart of your agent loop, including error paths.
- A 200-word reflection on "when would an agent actually beat a non-agent workflow?" — agents should not be the first answer; know when they are.
- A security-threats shortlist: what could go wrong if your agent were exposed to untrusted input? Prompt injection, data exfiltration, cost blow-up, model misuse.

## Reading plan

| Day | What |
| --- | --- |
| 1 | Anthropic tool use overview. Cookbook notebooks. |
| 2 | ReAct paper. Building Effective Agents blog. |
| 3 | Instructor docs. Structured-outputs patterns. |
| 4 | Simon Willison on agent security. Prompt injection reading. |
| 5 | Chain-of-thought paper. Reflexion paper. |
| 6 | Surveys (skim). |
| 7 | Reference as you build. |
