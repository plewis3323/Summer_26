# Week 23 Project — Data API (SWE II)

## Objective

Ship a small service that answers questions about a table you own: sqlite +
SQL + FastAPI + Docker, with tests and a `SERVICE.md` a stranger can follow.
This is the Month 06 software deliverable and the shape Week 44 will put a
*model* behind. The gate (from `03-Project-Roadmap.md`): **tests hit the
endpoints; the container serves the same answers.**

Default table: Week 04's dimuon cut-flow / fit results (rebuild with
`build_db.py` if you no longer have `results.db`). Allowed substitute: any
tidy table with ≥20 rows and two joinable relations (say so in `SERVICE.md`).
Physics is not required; the skills are.

## Background — from zero, restated

A **service** is a program that waits for HTTP requests and returns JSON.
**SQL** is how you ask a database questions. **sqlite** is a database that is
one file. **FastAPI** turns Python functions into endpoints. **Docker** packs
the service so it runs the same on another machine. `lesson.md` teaches each
of these; this spec is the build order.

## Build steps

The exercises in `exercises.md` *are* the stages. In order:

1. Rebuild `data/results.db` from known inputs; four SQL queries with tests.
2. Time a JOIN with and without an index; record both times in `SERVICE.md`.
3. FastAPI: `GET /health`, `GET /cutflow` (or `/rows`), `GET /.../{id}` with
   404. Pydantic schemas. `TestClient` tests green.
4. `RESULTS_DB` from the environment; logs must not contain a fake `API_KEY`.
5. `Dockerfile`; `docker run` serves the same `/health` JSON as TestClient.
6. `SERVICE.md`: rebuild db, run tests, run container, example JSON.

## Acceptance gate

- `uv run pytest -q` green.
- Container `GET /health` returns `{"ok": true}` from the host.
- `SERVICE.md` is sufficient without the lesson.
- No secrets in git or logs.

Then continue to Week 24 (Capstone 2). The Phase 2 gate also wants this
week's tests green (`00-Syllabus.md` §5).

## Writeup

`SERVICE.md` is the writeup: commands, example response, JOIN timing, and
one paragraph on what failed first (wrong listen address, missing index,
CI vs local db path — write the real one).
