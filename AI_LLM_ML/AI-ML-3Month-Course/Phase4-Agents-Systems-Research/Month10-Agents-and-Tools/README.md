# Month 10 — Agents & Tools

The month goes from a single API call to a working physics-analysis agent. Week 37 builds
tool use from first principles on the Claude Messages API: JSON-schema tool definitions,
structured output, and a hand-written multi-tool loop with error recovery. Week 38 steps
back to architecture — ReAct and the workflow patterns from Anthropic's *Building
Effective Agents* — including when an agent is the wrong answer. Week 39 standardizes the
tool layer: an MCP server exposing your own physics analysis functions
(`fit_pi0_peak`, `list_run_files`, `get_calibration`, ...), consumed from a real client.
Week 40 combines everything: multi-agent orchestration, agent evaluation, and a
mini-project — an analysis-copilot prototype driving the Week-39 MCP tools.

**Month-end deliverable:** a repo containing the MCP server (tested), the agent loop, and
a recorded demo of the copilot answering a real analysis question end-to-end, with an
evaluation table (task success rate, cost per task, one trajectory review).

**Sign-off:** tag the commit `month-10-complete`, write a 250-word `retro.md` in this
folder, and open one issue for the biggest thing you don't yet understand.
