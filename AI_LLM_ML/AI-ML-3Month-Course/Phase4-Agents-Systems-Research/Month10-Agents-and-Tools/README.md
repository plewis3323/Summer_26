# Month 10 — Agents & Tools

The month goes from a single API call to a working agent, with security in the
tool layer from day one. Week 37 recaps HTTP/JSON from Week 23, builds tool use
from first principles, and treats prompt injection as a test, not a hope.
Week 38 steps back to architecture — ReAct and Anthropic's workflow patterns —
including when an agent is the wrong answer. Week 39 standardizes the tool
layer: an MCP server exposing tools you own (science analysis *or* general data
tools). Week 40 combines everything: multi-agent orchestration, agent
evaluation, and a copilot prototype.

**Month-end deliverable:** a repo containing the MCP server (tested, including
an injection case), the agent loop, and a recorded demo of the copilot completing
a real task end-to-end, with an evaluation table (task success rate, cost per
task, one trajectory review).

**Sign-off:** tag the commit `month-10-complete`, write a 250-word `retro.md` in this
folder, and open one issue for the biggest thing you don't yet understand.
