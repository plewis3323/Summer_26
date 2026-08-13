# Week 44 — Deployment: A Model as a Service

~2 hrs reading — this week's time goes into the build (`project.md`). Before
starting you should be able to: load a registered checkpoint from Week 41's
model registry; quote a p50/p95 from a timing loop (Week 42); write `pytest`
tests that fail on purpose (Week 04); state distribution shift as "the test
$P$ is not the train $P$" (Week 09's true-risk vs empirical-risk gap, now in
production); and write the policy gradient theorem from memory (Week 43 —
this month's cold re-derivation, scheduled below).

A model that only runs in your notebook is a detector that only works when
you are in the counting house. This week one earlier model becomes a
**service**: a process someone else (or last week's agent) can call without
knowing your kernel state. Pick one artifact you already trust — the
Capstone-1 BDT, the Week-20 CNN, or the Week-32 extractor — and wrap it.
The six build stages live in `project.md`; this lesson is the *why* of each
stage, not a second copy of the accept criteria.

## 1. Why a notebook is not a service

A Jupyter notebook is a conversation with yourself. It has no public
contract: the input is "whatever cells you ran," the output is "whatever
the last cell printed," the model is whatever happened to be in memory, and
the only client is you. That is the counting house: you can look at events,
you cannot take data.

A **service** is a long-running process with a stated contract:

- a **schema** — which fields come in, which go out, with types;
- **endpoints** — named URLs that do one job each;
- a **lifecycle** — the model is loaded once, then many requests share it;
- **isolation** — it runs the same way on your laptop, in CI, and on a
  machine you have never logged into.

The physics analog is the DAQ: a crate does not "open a notebook and think
about the waveform." It accepts a formatted event, returns a formatted
result, and has a heartbeat so the shift crew knows it is alive. Week 39's
MCP server was this idea for *tools* an agent can call. This week is the
same idea for *models*, over HTTP. FastAPI is the library; the concept is
the contract.

## 2. Request/response, `/health` vs `/predict`

**HTTP** (Hypertext Transfer Protocol) is a request/response protocol: a
client sends a method plus a path (`POST /predict`) plus a body; the server
replies with a status code and a body. You used it as a client in Week 37;
this week you are the server.

The contract is a **schema**, not a docstring. In FastAPI that is a Pydantic
model: a class whose fields *are* the types. A request that is missing
`n_hits` or that sends a string where a float belongs is rejected with
**422 Unprocessable Entity** before your model code runs. That is the point
— fail at the boundary, with a machine-readable error, not inside a numpy
cast twenty frames down.

Two endpoints, different jobs:

- **`POST /predict`** — the physics. Body in, prediction out, plus the
  **model version** (registry id from Week 41, or a git hash plus a
  checkpoint name). The version belongs in the response so a client can
  tell that Tuesday's 0.91 and Thursday's 0.89 were not the same artifact.
  Week 10's decision threshold — the cut that turns a score into a class —
  lives *in the service*, versioned with the model, not in each client.
  Otherwise every consumer silently forks the operating point.
- **`GET /health`** — the heartbeat. Returns whether the process is up
  *and* whether the model is actually loaded. A process that is listening
  but will 500 on the first predict is not healthy. Orchestrators (and you,
  at 2 a.m.) poll this; they should not have to send a fake event to find
  out the weights file is missing.

Load the model **at startup**, not per request. Per-request load is a
latency cliff (hundreds of ms to seconds, every call) and a correctness
bug (two requests can see two different files if someone swaps the
checkpoint mid-flight). Startup load is the DAQ pattern: configure once,
then take data. Week 41's registry call (`models:/name/version`) belongs
in the startup hook; the request path only runs inference.

A sketch of the shape — not the project, the shape. (Current FastAPI prefers a
`lifespan` context manager over `@app.on_event`; both mean "load once, then
serve.")

```python
from fastapi import FastAPI
from pydantic import BaseModel

class PredictIn(BaseModel):
    features: list[float]

class PredictOut(BaseModel):
    score: float
    model_version: str

app = FastAPI()
model = None
VERSION = "unset"

@app.on_event("startup")
def load():
    global model, VERSION
    model, VERSION = load_registered("pid-bdt/1")  # once

@app.get("/health")
def health():
    return {"ok": model is not None, "model_version": VERSION}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    return PredictOut(score=float(model.predict([body.features])[0]),
                      model_version=VERSION)
```

`TestClient` tests both paths, including a 422 on a malformed body. A test
for the *model* (Week 10, Week 20) checks that the scores are right on a
fixture batch. A test for the *service* checks the contract: schema
rejection, health when unloaded vs loaded, version stamp, no reload across
100 sequential calls. Those are different claims; you need both.

## 3. Containers: image vs container, bake vs mount

It runs on your laptop. That is not a deployment. Your laptop has your
`conda` accidents, your CUDA version, and a weights file in
`~/Downloads`. A **container** is a process with its *own* filesystem,
libraries, and network, built from a recipe so a clean machine can
reproduce it.

Two words that get swapped and should not:

- An **image** is the immutable snapshot — the class, the software
  release. Built from a **Dockerfile**, tagged (`week44-svc:0.1.0`).
- A **container** is one running instance of that image — the object, the
  job. `docker run` starts a container from an image; you can run three
  containers from one image.

The Dockerfile is the recipe: a base image, pinned dependencies, a copy of
the app, a command that starts the server. **Pin the deps** (lock file or
exact versions). "Whatever `pip install fastapi` means today" is how
Thursday's green build becomes Friday's mystery. Layer order matters for
cache (put the rarely-changing `pyproject.toml` copy before the app copy)
but correctness matters more than cache: a floating `FROM python:3.12` is
a moving floor.

Weights are a separate decision, because they are large and they version
independently of the code:

- **Baked in** — `COPY` the checkpoint into the image. The image *is* the
  model version; simple, reproducible, fat. Right for a small BDT or a
  small CNN you will not swap daily.
- **Mounted** — the image contains only code; weights arrive as a volume
  or are pulled from the Week 41 registry at startup. Thinner images,
  possible to swap weights without rebuilding, and a new failure mode (the
  container is "healthy" as a process and unhealthy as a model if the
  mount is empty — which is why `/health` checks the load, not just the
  port).

Treat this like calibrations. A reconstruction release with constants
compiled in vs a release that reads a conditions database: both are
valid; mixing them silently is not. `SERVICE.md` states which you did.

## 4. Monitoring: latency tails and input drift

A service that is "up" can still be wrong or slow. You measure both.

**Latency** is the wall-clock of one `/predict`. The mean is a vanity
metric. Report **percentiles**: p50 (median — what a typical call costs)
and p95 (the tail — what one in twenty costs). Detector timing
resolution is quoted as a width, not a mean, for the same reason: the
tail is where triggers drop and users leave. Week 42's lesson applies
directly: if p95 is huge and p50 is fine, you are looking at tail
effects (GC, a hitch on first-token compile, a disk hit on an unbaked
weight), not at "the model is slow." The first lever is usually the one
Week 42 named for your architecture (batch, precision, not doing a
Python loop per feature) — pick it from the load-test numbers, not from
taste.

**Drift** is Week 09's generalization problem with the clock running.
Week 09 assumed train and test were draws from the same $P(x,y)$. In
production that assumption dies in two different ways, and you must not
use one word for both:

- **Covariate shift / input drift** — $P(x)$ moves, $P(y \mid x)$ does
  not. The beam energy changed; the abstract corpus picked up a new
  experiment's notation; a camera's pedestal jumped. The *decision
  surface* is still right, but it is being queried off the manifold it
  was trained on. A rolling-window mean and width of an input feature,
  compared to the training set's mean and width, can detect this. That
  is the monitor you will build.
- **Concept drift** — $P(y \mid x)$ moves. The detector's energy scale
  was recalibrated, so the same shower image now has a different true
  label; the collaboration changed what "signal" means. Feature
  statistics will *not* catch this. You need labels, or a proxy that
  actually depends on $y$ (a downstream physics plot, a spot-check).
  Saying "the drift monitor is green" is not saying "the model is still
  correct."

The Week 09 names, used carefully: a feature-statistics monitor is a
test of $P(x)$, i.e. of whether the *empirical risk you can still
compute* is even on the right distribution. It is not a test of true
risk. Replay a deliberately shifted stream and show the alarm; replay
an unshifted stream and show silence. An alarm that always fires is a
pager, not a monitor.

## 5. CI and load-test honesty

**CI** (continuous integration) is a machine that runs your tests on
every push so you do not have to remember to. This week that means two
jobs: `pytest` (contract + a regression guard on the registered model —
Week 41) and `docker build` (the image still builds). A badge that is
green because you did not add the workflow does not count; a
deliberately broken test on a branch must turn it red, or the badge is
a decoration.

A **load test** asks what the service does under concurrent clients, not
what it does in a `for` loop in the same process that serves it. Ramp
concurrency; record p50, p95, and the maximum sustained requests/s
*before* latency falls apart. That last number is the **saturation
point** — past it you are measuring queueing, not inference. Report the
envelope and the saturation point in `SERVICE.md`. Dishonest patterns,
so you do not commit them:

- quoting a localhost number as if it were a deployed number;
- warming every cache, then reporting the warm p50 as "latency";
- a single client in a tight loop, which never exercises concurrency;
- omitting the saturation point, so a reader thinks the p50 holds at
  10× the measured rate.

Honest load tests look like beam-rate scans: you state the conditions
(hardware, batch, warm or cold, concurrency) and you show the curve
until it breaks.

## 6. How the week runs, and this month's cold redo

The build is six stages in `project.md`, in order: (1) service skeleton
with typed `/predict` and `/health` and TestClient coverage including
422; (2) real model loaded at startup from the Week 41 registry, 100
sequential requests matching offline inference; (3) Dockerfile, rebuild
from a wiped cache, same tests against the container; (4) request logs
plus a drift check that trips on a shifted stream and not on an
unshifted one; (5) GitHub Actions running `pytest` and `docker build`,
proven red on a broken branch; (6) load test, envelope written to
`SERVICE.md`. Work them in order. Stage 1 with a stub model is the
thinnest end-to-end path — real HTTP, fake model, tests green — and you
do not skip it to "just load the CNN."

This month's flagship re-derivation is the **policy gradient theorem**
(Week 43):

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^{\pi}(s_t, a_t)\right].$$

Cold, on paper, no notes, scanned into the week folder. Schedule a
fresh morning. A looked-up step goes in the month's open-question
issue. The service you ship is the month's artifact; the derivation is
the month's understanding. Neither substitutes for the other.

## Check yourself

1. Name two claims a service test can fail on that a model test on a
   fixture batch cannot see.
2. Why does `/health` have to report *model-loaded*, not just "the
   process accepted a TCP connection"?
3. Where does Week 10's decision threshold live, and what goes wrong if
   each client applies its own?
4. Image vs container, in one sentence each. Give one reason to bake
   weights into the image and one reason to mount them.
5. Your feature-mean monitor is green for a week, then the physics
   analysis that consumes `/predict` goes wrong. Which kind of drift
   can the monitor not see, and why?
6. A load test reports p50 = 8 ms at concurrency 1 on your laptop.
   List two reasons that number is not an envelope.
7. State the policy gradient theorem (this month's cold redo) and name
   the Week 08 object $\nabla \log \pi$ already was.

## Answers

1. Schema rejection (422 on a missing field); health when the model
   failed to load; the version stamp; no per-request reload under 100
   sequential calls. The model test never opens a socket.
2. Because a listener with a missing checkpoint will 500 on the first
   real call. Orchestrators and humans poll health to decide whether to
   send traffic; "port open" is not that decision.
3. In the service, versioned with the model. If clients each apply a
   cut, every consumer silently forks the operating point and you can
   no longer say which threshold produced a reported rate.
4. Image: the immutable snapshot (the release). Container: one running
   instance of it (the job). Bake: the image *is* the model version,
   simple reproducibility. Mount: swap weights without rebuilding;
   thinner images — at the cost of a new failure mode `/health` must
   catch.
5. Concept drift: $P(y \mid x)$ moved (recalibration, a changed label
   definition) while $P(x)$ stayed put. A monitor on input mean/std
   watches $P(x)$ only.
6. Concurrency 1 never saturates the server; localhost is not the
   deploy hardware; a warm single-process loop hides cold-start and
   queueing. An envelope is a curve vs concurrency, with a saturation
   point, under stated conditions.
7. $\nabla J = \mathbb{E}[\sum_t \nabla\log\pi_\theta(a_t\mid s_t)\,
   A^\pi(s_t,a_t)]$. $\nabla\log\pi$ is the score function from
   maximum-likelihood (Week 08); here each action is weighted by
   advantage instead of being treated as a label to imitate.

## New terms

- **service** — long-running process with a typed contract, endpoints,
  and a lifecycle; as opposed to a notebook.
- **schema (request/response)** — the typed fields of the body; enforced
  at the boundary (Pydantic / 422).
- **`/health` vs `/predict`** — liveness-plus-model-loaded vs inference.
- **model version** — registry id or commit-plus-checkpoint, returned
  with every prediction so results are attributable.
- **image / container / Dockerfile** — snapshot; running instance;
  recipe that builds the snapshot.
- **baked vs mounted weights** — checkpoint copied into the image vs
  supplied at runtime.
- **percentile latency (p50, p95)** — median and tail of per-request
  wall-clock; the tail is the operating number.
- **covariate shift / input drift** — $P(x)$ moves, $P(y \mid x)$ does
  not; feature-statistic monitors can see it.
- **concept drift** — $P(y \mid x)$ moves; feature statistics cannot
  see it; needs labels or a downstream proxy.
- **CI** — automated `pytest` + `docker build` on push.
- **saturation point** — concurrency (or rate) at which latency
  collapses into queueing; the end of the honest envelope.

## Going deeper

- FastAPI docs: first steps, request body / response model, and
  `TestClient` — enough to write stage 1 without a tutorial maze.
- Docker getting-started: images vs containers, Dockerfile, layer
  caching; skim multi-stage builds (optional here, useful later).
- Chip Huyen, *Designing Machine Learning Systems* — the deployment
  chapter and the data-distribution-shifts / monitoring chapter; focus
  on what a production monitor can actually see.
- GitHub Actions quickstart — `pytest` plus `docker build` on push;
  that is the whole CI bar this week.
- Re-derive Week 43 §§5–8 (policy gradient through the advantage form)
  the morning you scan it; the PPO sketch in §9 is optional review, not
  the gate.
