# Week 44 — Deployment

A model that only runs in your notebook is a detector that only works when you're in the counting house — this week one of your models becomes a service anyone (or any agent) can call.

## Objectives

- Wrap a trained model in a FastAPI service with a typed request/response schema, an inference endpoint, and a `/health` check.
- Containerize it with Docker: small image, pinned deps, model weights handled deliberately (baked in or mounted), one-command run.
- Add monitoring: request logging, latency percentiles, and a simple input-drift check against training-data statistics.
- Set up CI that runs the test suite and builds the image on every push.
- Load-test the service and state its throughput/latency envelope honestly.

## Core material (~3 hrs)

- FastAPI documentation: first steps, request body/response model, and testing (TestClient) sections.
- Docker getting-started guide: images vs. containers, Dockerfile basics, layer caching; skim multi-stage builds.
- Chip Huyen, *Designing Machine Learning Systems*: the deployment chapter and the data-distribution-shifts/monitoring chapter — focus on what drift is detectable in production and what isn't.
- GitHub Actions quickstart docs — enough to run pytest and `docker build` in CI.

## Exercises (built when the week starts)

Mini-project week: the exercises are stages of one build. Pick one earlier model (Capstone-1 BDT, Week-20 CNN, or the Week-32 extractor).

1. Service skeleton: FastAPI app with `/predict` (typed payload in, prediction + model version out) and `/health` (returns model-loaded status). Accept when: TestClient tests for both endpoints pass, including a 422 on malformed input.
2. Real model: load the Week-41 registered model at startup; never at request time. Accept when: 100 sequential requests return correct predictions (spot-checked against offline inference) with no reload.
3. Containerize: Dockerfile + `docker run -p ...` serves the same predictions. Accept when: fresh machine (or wiped image cache) build-and-run passes the exercise-1 tests against the container.
4. Monitoring: log every request (latency, input summary stats) to a file; add a drift check comparing a rolling window of one input feature's mean/std to training values. Accept when: replaying a deliberately shifted input stream trips the drift warning; the unshifted stream doesn't.
5. CI: GitHub Actions workflow running `pytest` and `docker build` on push. Accept when: the badge is green on the final commit and a deliberately broken test turns it red on a branch.
6. Load test: hammer `/predict` with a simple async client at increasing concurrency; record p50/p95 latency and max sustained req/s. Accept when: `SERVICE.md` reports the envelope and the saturation point.

## Deliverable

`week44/service/` — repo with app, tests, Dockerfile, CI config, and `SERVICE.md` (API spec, run instructions, monitoring notes, load-test results). This is the month's shipped artifact.

## Review

1. Week 10: your classifier had a tuned decision threshold. Where does that threshold now live — client or service — and why does it matter for versioning?
2. Week 42: which inference optimizations from last week apply to this service, and which is the single best first lever given your load-test numbers?
3. Week 9: distribution shift vs. concept drift — define both; which can your feature-statistics monitor detect?
4. Week 3: what does a test for the service check that a test for the model doesn't? Give two concrete examples from this week.
5. Week 43: state the policy gradient theorem from memory (this month's cold re-derivation target).
