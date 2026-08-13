# Week 44 — Exercises

This week is the mini-project; the exercises are stages of one build, in
order — each one is a step of `project.md`, and finishing E1–E6 *is*
finishing the project. Read `project.md` before starting E1. Pick one
earlier model you already trust (Capstone-1 BDT, Week-20 CNN, or the
Week-32 extractor) and wrap it; do not train a new one. Most work lives in
`week44/service/`, per `NOTEBOOK_RULES.md` §6 — the acceptance criteria are
`pytest`, a Docker image, a CI badge, and `SERVICE.md`. The notebook
launches the tests and checks outputs. Stage 1 with a stub model is the
thinnest end-to-end path; do not skip it to "just load the CNN."

This month's cold re-derivation is the policy gradient theorem (Week 43) —
schedule a fresh morning, scan it into the week folder, flag looked-up
steps. The service is the month's artifact; the derivation is the month's
understanding. Neither substitutes for the other.

## E1 — Service skeleton

FastAPI app with `POST /predict` (typed payload in, prediction + model
version out) and `GET /health` (returns model-loaded status). TestClient
tests for both endpoints, including a 422 on malformed input.
Hint: lesson §2 — Pydantic models *are* the schema; a missing field or a
string where a float belongs must 422 before your model code runs. `/health`
reports whether the model is actually loaded, not just that the process
accepted a TCP connection. A stub model (return 0.5, version `"stub"`) is
correct for this stage; the real checkpoint is E2. `TestClient` lives in
`fastapi.testclient`.
Accept when: TestClient tests for both endpoints pass, including a 422 on
malformed input.

## E2 — Real model

Load the Week-41 registered model at startup; never at request time. 100
sequential requests return correct predictions (spot-checked against
offline inference) with no reload.
Hint: the request path only runs inference. Per-request load is a latency
cliff and a correctness bug (two requests can see two files if someone
swaps the checkpoint). Week 41's `models:/name/version` (or the equivalent
path you registered) belongs in the startup / lifespan hook. Spot-check:
save a fixture batch of inputs with offline scores, assert the service
matches to a tight tolerance, and assert `model_version` is constant across
the 100.
Accept when: 100 sequential requests return correct predictions
(spot-checked against offline inference) with no reload.

## E3 — Containerize

Dockerfile + `docker run -p ...` serves the same predictions. Fresh machine
(or wiped image cache) build-and-run passes the E1 tests against the
container.
Hint: pin the base image and the deps (lock file or exact versions). State
in `SERVICE.md` whether weights are **baked in** (`COPY` the checkpoint) or
**mounted** (volume / registry pull at startup) — lesson §3; mixing them
silently is the failure mode `/health` exists to catch. Wipe the cache with
`docker build --no-cache` (or a fresh VM / Codespace) so "it built on my
laptop" is not the test. Hit the container with the same TestClient tests
via the published port, or `docker run` plus a client script.
Accept when: fresh machine (or wiped image cache) build-and-run passes the
exercise-1 tests against the container.

## E4 — Monitoring

Log every request (latency, input summary stats) to a file. Add a drift
check comparing a rolling window of one input feature's mean/std to
training values. Replaying a deliberately shifted input stream trips the
drift warning; the unshifted stream does not.
Hint: lesson §4 — this monitor sees **covariate shift** (P(x) moved), not
**concept drift** (P(y|x) moved). Pick one feature with a well-defined
training mean (a shower width, a MAGIC Hmax, an abstract token-count). A
window of ~50–100 requests and a threshold of a few training-σ on the
window mean is enough; tune it so the unshifted replay is silent. An alarm
that always fires is a pager, not a monitor.
Accept when: replaying a deliberately shifted input stream trips the drift
warning; the unshifted stream doesn't.

## E5 — CI

GitHub Actions workflow running `pytest` and `docker build` on push. The
badge is green on the final commit and a deliberately broken test turns it
red on a branch.
Hint: two jobs, or one job with two steps. A badge that is green because
you did not add the workflow does not count. Prove the red: open a branch,
break one assert, push, screenshot or link the failed run, revert. The
broken-test branch is evidence, not a deliverable you leave merged.
Accept when: the badge is green on the final commit and a deliberately
broken test turns it red on a branch.

## E6 — Load test

Hammer `/predict` with a simple async client at increasing concurrency;
record p50/p95 latency and max sustained req/s. `SERVICE.md` reports the
envelope and the saturation point.
Hint: lesson §5 — the mean is a vanity metric; the envelope is a *curve*
vs concurrency, under stated conditions (hardware, warm or cold, batch).
The **saturation point** is where latency falls apart into queueing; past
it you are not measuring inference. Do not quote a localhost concurrency-1
p50 as "the latency." Dishonest patterns to avoid are listed in the lesson;
the honest analog is a beam-rate scan.
Accept when: `SERVICE.md` reports the envelope and the saturation point.

## Review

1. Week 10: your classifier had a tuned decision threshold. Where does that
   threshold now live — client or service — and why does it matter for
   versioning?
2. Week 42: which inference optimizations from last week apply to this
   service, and which is the single best first lever given your load-test
   numbers?
3. Week 9: distribution shift vs. concept drift — define both; which can
   your feature-statistics monitor detect?
4. Week 3: what does a test for the service check that a test for the model
   doesn't? Give two concrete examples from this week.
5. Week 43: state the policy gradient theorem from memory (this month's
   cold re-derivation target).
