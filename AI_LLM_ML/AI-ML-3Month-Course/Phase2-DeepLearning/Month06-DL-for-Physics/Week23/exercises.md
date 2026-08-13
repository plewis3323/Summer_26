# Week 23 — Exercises

Work top to bottom. The notebook may prototype queries; the service lives in
`week23/service/` and is checked by pytest and `docker run`, not by notebook
cells (per `NOTEBOOK_RULES.md` §6). If you do not have Week 04's dimuon
results, `build_db.py` may load any tidy CSV with at least 20 rows and two
joinable tables — say so in `SERVICE.md`.

## E1 — SQL on a known table

Rebuild (or create) `data/results.db` with `cutflow(step, n)` and
`fits(name, value, error)` from known numbers. Write four queries in
`queries.sql` (or as functions in `src/db.py`): all rows; `WHERE n > 40000`;
`GROUP BY` on a `runs` table you add with at least two energies; a `JOIN` of
`runs` to `fits`.
Hint: `sqlite3.connect` + `.fetchall()`; print the rows.
Accept when: each query's result is asserted in `tests/test_queries.py`
against numbers you computed by hand.

## E2 — JOIN with and without an index

Time the JOIN in E1 on a *scaled-up* table: duplicate `runs` and `fits` to
at least 50,000 rows each (a loop writing fake run numbers is fine). Run the
JOIN once with no index, once after `CREATE INDEX` on the join key. Record
both wall times.
Hint: `time.perf_counter()` around `fetchall()`. On some machines the
indexed JOIN is "only" a few times faster at 50k — still report the ratio.
Accept when: both times are in `SERVICE.md` and the indexed query is faster
(or you explain, with `EXPLAIN QUERY PLAN`, why sqlite already had a plan).

## E3 — FastAPI read API

`app.py` with `GET /health`, `GET /cutflow` (list), `GET /cutflow/{step}`
(one row or 404). Schemas via Pydantic. `TestClient` tests for 200, 404, and
JSON shape.
Hint: lesson.md §4 is the skeleton; do not add features it does not name.
Accept when: `uv run pytest -q` is green and `/docs` loads when the server
is running.

## E4 — Env var + logs without secrets

The db path comes from `RESULTS_DB` (default `data/results.db`). A fake
`API_KEY` env var is read at startup. One request is logged at INFO with
path and status. A test fails the week if the log line contains the key
value.
Hint: `caplog` in pytest, or read a `StringIO` handler. Set `API_KEY=secret-test`
in the test; assert `"secret-test"` not in the log.
Accept when: the test is green, and unsetting `RESULTS_DB` still finds the
default path.

## E5 — Dockerfile

A `Dockerfile` that builds the service. `docker build -t week23-service .`
then `docker run --rm -p 8000:8000` serves the same `/health` JSON as
TestClient.
Hint: listen on `0.0.0.0`; copy the db *or* run `build_db.py` in the image.
Accept when: `curl` (or `httpx`) from the host gets `{"ok": true}` from the
container.

## E6 — One-command story

`SERVICE.md` lists, in order: how to rebuild the db, how to run tests, how
to build and run the container, and one example `GET /cutflow` response.
Hint: three command blocks, one JSON block. No screenshots.
Accept when: a classmate (or you tomorrow) can follow only `SERVICE.md` and
hit `/health` without reading the lesson.

## Review

1. (Week 04) What does CI do that `git push` alone does not?
2. (Week 04) Why `?` placeholders instead of f-strings in SQL?
3. (Week 03) Inner join vs left join: which drops unmatched rows?
4. (Week 16) If `/predict` wrapped your MLP, what belongs in `/health`?
5. (Week 08) A query returns a mean mass. What else must you store if you
   want to quote an uncertainty later?
