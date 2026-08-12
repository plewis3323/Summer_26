# Week 37 — Exercises

Work top to bottom. Setup (imports, the toy-data generator that writes
`data/run_*.csv`, the `check` helper, and a `usage_log` list for cost tracking) is
given by the notebook; you write only the lines each exercise asks for.
E2–E5 live in files — `week37/tools.py` and `week37/loop.py` — because their
acceptance criteria are scripts and saved transcripts; E1, E6, and E7 are notebook
cells. Set `ANTHROPIC_API_KEY` in your terminal before starting, and remember these
exercises cost real money — E7 totals it up.

## E1 — Raw call, every field

Send one `client.messages.create` request asking a one-line physics question. Print
`stop_reason`, `usage.input_tokens`, `usage.output_tokens`, and the `type` of every
content block.
Hint: loop over `response.content`; each block has a `.type`.
Accept when: the printout shows `stop_reason == "end_turn"`, both token counts > 0,
and at least one block of type `text`.

## E2 — First tool (`tools.py`)

In `week37/tools.py`, implement `fit_pi0_peak(file, window_lo, window_hi)` (Gaussian +
linear background via `curve_fit`, as in lesson.md §6) and its tool definition dict
`FIT_PI0_PEAK_TOOL`. Send one request with `tools=[FIT_PI0_PEAK_TOOL]` asking for a
fit of `data/run_00042.csv`, and print the `tool_use` block.
Hint: the description should tell the model the peak is near 0.135 GeV and suggest a
window — that sentence is what makes the call come back sensible.
Accept when: `stop_reason == "tool_use"`, the block's `input` contains all three
required keys, and running your own `fit_pi0_peak` on that input returns a mean
within 0.005 GeV of 0.135.

## E3 — The loop (`loop.py`)

In `week37/loop.py`, write `run_agent(question, tools, tool_functions)` — the full
while-loop from lesson.md §7, plus: append every request/response pair to a
transcript list and save it as JSON when the loop ends. Give it two tools
(`list_run_files`, `fit_pi0_peak`) and ask: "Fit the pi0 peak in the latest run file
in data/."
Hint: cap the loop at 10 iterations so a confused model can't spend all afternoon.
Accept when: one question triggers both tools in sequence with no manual
intervention, the final text quotes a mass within 0.005 GeV of your direct fit, and
`transcripts/e3.json` exists and contains both `tool_use` blocks.

## E4 — Error recovery

Make `fit_pi0_peak` raise `ValueError("window_lo must be less than window_hi")` on a
bad window. Ask the agent a question phrased to invite a mistake ("fit the peak using
a window from 0.30 down to 0.10 GeV") and let the loop's `is_error` path do its job.
Hint: don't special-case anything — if E3's loop is correct, recovery is automatic.
Accept when: the transcript shows a `tool_result` with `is_error: true` followed by a
retried `tool_use` with a valid window, and the loop terminates with a successful fit.

## E5 — Parallel calls

Ask one question needing three independent fits ("fit the pi0 peak in each of these
three run files and compare the means"). Return all `tool_result` blocks in a single
user message, as the loop already does.
Accept when: the transcript shows an assistant turn containing ≥ 2 `tool_use` blocks,
the following user turn carries one `tool_result` per call with matching
`tool_use_id`s, and the final answer quotes all three means correctly.

## E6 — Structured output

The notebook gives 10 short paragraphs of analysis prose with hand-labeled values.
Extract `{particle, mass_gev, width_gev, n_events}` from each using
`output_config={"format": {"type": "json_schema", ...}}` — no "please reply in JSON"
prompting.
Hint: one call per paragraph in a `for` loop; `json.loads` each response text.
Accept when: 10/10 responses parse with `json.loads` and all four fields match the
hand labels (numbers within 1%).

## E7 — Cost ledger

After each of E1–E6, the notebook appended `response.usage` to `usage_log`. Convert
the log into a printed table: exercise, input tokens, output tokens, dollars (use the
current per-million-token prices from the pricing page — paste them into the two
constants the notebook provides).
Hint: cost = in_tokens/1e6 * price_in + out_tokens/1e6 * price_out.
Accept when: the table prints one row per exercise plus a total, and the total is
under your stated budget for the week (set it before you start; $5 is plenty).

## Review

1. Week 27: write the cross-entropy loss for next-token prediction. What plays the
   role of the label?
2. Week 29: how does sampling temperature change the output distribution, and why
   would T → 0 make tool-call arguments more repeatable?
3. Week 10: you chose a decision threshold on a ROC curve. What is the analogous
   trade-off in a model's decision to call a tool versus answer directly?
4. Week 34: what did recall@k measure in your RAG evaluation? Propose the analogous
   metric for "did the agent call the right tool."
5. Week 04: why do unit tests on `fit_pi0_peak` itself matter more than any amount of
   prompt-tweaking?
