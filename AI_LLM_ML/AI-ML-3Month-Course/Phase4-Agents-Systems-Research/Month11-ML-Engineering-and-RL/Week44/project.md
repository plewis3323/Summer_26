# Week 44 Project — Model Service

## Objective

Wrap one earlier model (Capstone-1 BDT, Week-20 CNN, or Week-32 extractor)
in a FastAPI service, containerize it, monitor it, put it in CI, and
measure its latency envelope. The gate (from `03-Project-Roadmap.md`):
**health check + latency budget met; deploy from a clean machine.** This is
Month 11's shipped artifact — a process anyone (or last week's agent) can
call without knowing your kernel state.

## Background — from a notebook to a crate

A Jupyter notebook is a conversation with yourself: no public contract, no
lifecycle, the model is whatever happened to be in memory. A **service** is
a long-running process with a stated contract — schema, endpoints,
lifecycle, isolation — the way a detector's DAQ crate accepts a formatted
event, returns a formatted result, and has a heartbeat so the shift crew
knows it is alive. Week 39's MCP server was this idea for *tools*. This
week is the same idea for *models*, over HTTP.

**The model you are wrapping is already a physics object.** Pick one you
trust, and keep its meaning in the response:

- **Capstone-1 BDT** — gamma/hadron separation on MAGIC telescope
  (or your HEP tabular PID). The service returns a score (and, if you
  apply it, a class). Week 10's **decision threshold** — the cut that
  turns a score into a keep/reject at a stated efficiency — lives *in the
  service*, versioned with the model. If each client applies its own cut,
  every consumer silently forks the operating point and you can no longer
  say which threshold produced a reported rate.
- **Week-20 CNN** — single-photon vs merged-π⁰ on EMCal tower images. A
  **calorimeter** stops particles and measures dumped energy on a grid of
  **towers**; a **π⁰ → γγ** decay at high pT merges into one cluster that
  fakes a direct photon (Week 20 Background). The payload is an 8×8 patch
  (or the flattened 64); the score is photon-ness. Same threshold rule.
- **Week-32 extractor** — a 1–3B model that reads a nuclear-physics
  abstract and emits schema-valid JSON (collision system, √s_NN,
  observables, …). Missing fields are `null`, never guessed. The service
  returns that JSON plus a model version; hallucination rate does not
  become someone else's problem just because the call went over HTTP.

**HTTP, schema, two endpoints.** A client sends `POST /predict` plus a
JSON body; the server replies with a status code and a body. The contract
is a **schema** (Pydantic), not a docstring: malformed input is **422
Unprocessable Entity** at the boundary, not a numpy cast twenty frames
down. `GET /health` is the heartbeat: process up *and* model loaded. A
listener with a missing checkpoint will 500 on the first real call;
orchestrators (and you, at 2 a.m.) poll health so they do not have to send
a fake event to find out. Load the model **at startup** from the Week-41
registry (`models:/name/version`), never per request. Return the **model
version** on every prediction so Tuesday's 0.91 and Thursday's 0.89 are
attributable.

**Containers.** It runs on your laptop. That is not a deployment. An
**image** is the immutable snapshot (the release), built from a
**Dockerfile**; a **container** is one running instance (the job). Pin the
deps. Weights are a separate decision: **baked in** (the image *is* the
model version — right for a small BDT/CNN) or **mounted** (thinner image,
swap without rebuilding, and a new failure mode `/health` must catch).
Treat it like calibrations: compiled-in vs conditions database; both
valid, mixing them silently not.

**Monitoring and load.** Latency is quoted as **p50 / p95**, not a mean —
the tail is where triggers drop. **Input drift** (covariate shift: P(x)
moved, P(y|x) did not) is detectable with a rolling-window mean/std of one
feature against training statistics. **Concept drift** (P(y|x) moved —
recalibration, a changed label) is not; a green monitor is not a correct
model. A **load test** ramps concurrency and reports the envelope plus the
**saturation point** (where you are measuring queueing, not inference).
The roadmap's "latency budget" is a number *you* state in `SERVICE.md`
from that envelope (e.g. p95 < 50 ms at concurrency 8 on the stated
hardware) and then meet — not a number this spec invents for a BDT and a
1.5B extractor alike.

## Data / artifacts

No new physics data. The service wraps a checkpoint you already have:

- Week-41 registered model (`models:/name/version`) — load at startup.
- A fixture batch of inputs with offline scores, for the E2 spot-check
  and as a regression guard (Week 41's pytest pattern, now behind HTTP).
- Training-set mean/std of the one feature the drift monitor watches,
  written next to the checkpoint so the container does not scrape a
  notebook.

If the registry is inconvenient inside Docker, bake a copy of that exact
version and still stamp its registry id on `/predict`. The version in the
response must match what `/health` reports.

## Build steps

Do them in order; each is one of this week's exercises (E1–E6 in
`exercises.md`). Stage 1 with a stub is required.

Suggested layout:

```
week44/service/
  pyproject.toml
  Dockerfile
  SERVICE.md
  src/service/
    app.py           FastAPI app, lifespan load, /predict, /health
    schema.py        Pydantic request/response
    monitor.py       request log + drift check
  tests/
    test_app.py      TestClient: 200, 422, health loaded/unloaded
    test_infer.py    100 sequential vs offline fixture
    test_drift.py    shifted stream trips; unshifted does not
  .github/workflows/ci.yml
  models/            baked weights, or a mount point documented in SERVICE.md
```

1. **Service skeleton** (E1). Typed `/predict` and `/health`, TestClient
   including 422. Stub model allowed. Accept when: those tests pass.
2. **Real model** (E2). Registry load at startup; 100 sequential requests
   match offline inference; no reload. Accept when: that check passes.
3. **Containerize** (E3). Dockerfile; wiped-cache (or clean-machine) build
   and run passes the E1 tests against the container. Accept when: that
   deploy works.
4. **Monitoring** (E4). Per-request log (latency, input summary); drift
   check on one feature. Accept when: shifted replay trips, unshifted does
   not.
5. **CI** (E5). GitHub Actions: `pytest` and `docker build` on push.
   Accept when: badge green on the final commit; a broken test on a branch
   turns it red.
6. **Load test** (E6). Async client, ramp concurrency, p50/p95, max
   sustained req/s, saturation point, written to `SERVICE.md`. Accept when:
   the envelope and saturation point are reported — and the p95 at the
   stated operating concurrency meets the latency budget you wrote down
   before looking at the curve (or you revise the budget in writing, with
   the hardware conditions, and meet the revision).

## Acceptance gate (from `03-Project-Roadmap.md` and the week README)

- **Health check + latency budget met.** `/health` reports model-loaded
  (not just port-open). `SERVICE.md` states a p95 latency budget under
  named conditions (hardware, concurrency, warm/cold) and the load-test
  curve shows it is met at a concurrency below saturation.
- **Deploy from a clean machine.** Fresh clone (or wiped image cache) +
  `docker build` + `docker run` serves the same predictions the TestClient
  tests assert. `uv sync` + `pytest -q` green on the host is necessary and
  not sufficient — the gate is the container.
- Tested and reproducible: TestClient coverage of `/predict`, `/health`,
  422, no-reload, drift trip/silence; CI badge green with a demonstrated
  red-on-break.
- The repo ships: `week44/service/` with app, tests, Dockerfile, CI
  config, and `SERVICE.md` (API spec, run instructions, bake-vs-mount,
  monitoring notes, load-test envelope, latency budget).

Then close the month: tag `month-11-complete`, write `retro.md`, open the
open-question issue. Scan the policy-gradient cold redo into the week
folder the same week — Month 11's understanding, next to the artifact.

## Writeup requirements (`SERVICE.md`)

Not a paper — an operator's page:

- **What it is and how to run it** — which model, the two endpoints, the
  one `docker run` line, one example request and response (including
  `model_version`).
- **API spec** — request/response fields, 422 cases, what `/health`
  actually verifies and what it misses (concept drift, for one).
- **Weights** — baked vs mounted, and the registry id stamped on
  `/predict`.
- **Threshold** — where Week 10's cut lives, and the operating point it
  implements (if the model is a classifier).
- **Monitoring** — log location, which feature is watched, the trip
  threshold, the shifted-stream evidence.
- **Load-test envelope** — p50/p95 vs concurrency, saturation point,
  hardware, the latency budget and whether it was met.
- **What failed first** — written the day it happened (a 422 you forgot,
  a baked path that was `.gitignore`'d, a monitor that always fired).

## Stretch goals (only after the gate)

- **Auth / rate limit:** a shared-secret header and a per-client cap —
  enough that last week's copilot cannot melt the box.
- **Batch endpoint:** `POST /predict_batch` and a second envelope; report
  whether Week 42's batching lever moved p95.
- **MCP wrapper:** expose `/predict` as one more Week-39-style tool so the
  Week-40 copilot can call the model without learning HTTP. (Final-
  capstone track (a) will want this.)
- **Second model:** same service, second registry id, `/predict` dispatches
  on a `model_version` field — the first rehearsal of not forking the
  operating point across consumers.
