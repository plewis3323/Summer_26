# Week 23 — Software Engineering II: Data and Services

A model that only lives in a notebook is a measurement that only exists on one
laptop. This week you learn the four things every AI Engineer (and every
scientist who ships a tool) actually uses: **SQL** to ask questions of tables,
**HTTP/JSON** to talk between programs, **FastAPI** to wrap a table or a model
in a small service, and **Docker** so the service runs the same on another
machine. No web-development background is assumed.

Normalizing flows (the previous occupant of this week) live in `optional-flows.md`
in this folder — do them if your Capstone 2/3 is a generative-sim track; otherwise
Week 35's survey is enough.

## Objectives

- Write SQL `SELECT` / `WHERE` / `JOIN` / `GROUP BY` against sqlite, and say
  why a JOIN can go O(n²) without an index.
- Explain HTTP methods, status codes, and JSON as the language programs use
  over the network.
- Wrap a sqlite table in a FastAPI service with typed request/response and tests.
- Containerize that service with a Dockerfile and run it with one command.
- Keep secrets in environment variables; log requests without logging secrets.

## Core material (~3 hrs)

- `lesson.md` (this folder) — the primary text, from zero.
- SQLite docs: *SELECT*, *JOIN*, *GROUP BY* (the language, not the C API).
- FastAPI tutorial: first steps, path parameters, request body, TestClient.
- Docker getting-started: images vs containers, a minimal Dockerfile.
- Week 04's sqlite results file is the default table; any tidy table you own is
  an allowed substitute (say so in the repo README).

## Exercises

See `exercises.md`. Six exercises: SQL queries on a known table, a JOIN with and
without an index, a FastAPI read API with tests, an env-var secret that must not
appear in logs, a Dockerfile, and a one-command run from a clean image. The
exercises *are* the mini-project; `project.md` is the spec.

## Deliverable

`week23/service/` — sqlite db (rebuilt from a script), FastAPI app, tests green,
Dockerfile, `SERVICE.md` with the three commands to run it. This is the skill
Week 44 will spiral on.

## Review

- (Week 04) What does CI do that `git push` alone does not? This week's container
  is the same idea for *running* the service, not just testing it.
- (Week 04) Why were `?` placeholders required in sqlite inserts?
- (Week 03) A pandas merge is a JOIN. Which join type drops rows silently, and
  what is the SQL name for it?
- (Week 16) If this service wrapped your MLP, what would you put in `/health`
  that a `/predict` test would not catch?
