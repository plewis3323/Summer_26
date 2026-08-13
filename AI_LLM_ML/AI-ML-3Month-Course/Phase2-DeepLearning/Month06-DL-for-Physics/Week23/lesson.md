# Week 23 — Software Engineering II: Data and Services

~3 hrs reading plus the build. Before starting you should be able to: write
functions and tests (Weeks 02, 04); load a table with pandas (Week 03); put
results in sqlite (Week 04 §9); state Big-O as vocabulary (Week 04 §10). No
web-programming background is assumed.

A **service** is a program that waits for requests and answers them. Every
model you will ever deploy, every agent tool you will ever expose, and every
internal plot-server a collaboration actually uses is a service. This week
builds a tiny one over a table you already own, so Week 44 can put a *model*
behind the same shape instead of teaching HTTP from scratch.

If you are on a generative-simulation capstone track, the old Week 23
(normalizing flows) is `optional-flows.md` in this folder — do it in slack
time or as a weekend. Week 35 still compares flows vs VAEs vs diffusion at
survey depth.

## 1. Why pandas is not a database

pandas is a calculator for tables that fit in memory. A **database** is a
program (or a file, in sqlite's case) designed to *answer questions* about
tables that may be larger than RAM, with guarantees: the data is still there
after a crash, two programs can read it, and a query can use an **index**
instead of scanning every row.

You already have a sqlite file from Week 04 (`data/results.db`) or can rebuild
one. This week you stop treating it as a write-only notebook and start
**querying** it.

## 2. SQL from zero

**SQL** (Structured Query Language) is the sentence structure for those
questions. sqlite speaks it; so do Postgres, BigQuery, and almost every
warehouse you will meet in an AI Engineer interview. Four clauses cover 90%
of what you need.

Assume a table `cutflow(step TEXT, n INTEGER)` and a table
`runs(run INTEGER, energy_gev REAL, n_events INTEGER, good INTEGER)`:

```sql
SELECT step, n FROM cutflow;
SELECT step, n FROM cutflow WHERE n > 40000;
SELECT energy_gev, SUM(n_events) FROM runs WHERE good = 1 GROUP BY energy_gev;
```

- **SELECT** names the columns you want (`*` means all — useful while
  exploring, sloppy in a service).
- **WHERE** filters rows. This is a mask, like pandas.
- **GROUP BY** collapses rows that share a key and lets you `SUM` / `COUNT` /
  `AVG` per group. This is a pandas `groupby`.

A **JOIN** combines two tables on a shared key. If `fits(run INTEGER, mass_gev REAL)`
holds one fit per run:

```sql
SELECT runs.run, runs.energy_gev, fits.mass_gev
FROM runs
JOIN fits ON runs.run = fits.run
WHERE runs.good = 1;
```

That is the same idea as `pd.merge(runs, fits, on="run")`. The silent-drop
join from Week 02/03 is SQL `INNER JOIN` (the default `JOIN`): rows whose key
is missing on the other side vanish. `LEFT JOIN` keeps them and fills the
missing side with `NULL` (pandas `NaN`).

**Indexes and O(n²).** Without an index, matching every row of A to every row
of B is nested loops — O(|A| × |B|). An **index** is a lookup structure on a
column (usually a B-tree; you do not need the internals) so each row of A
finds its partner in roughly O(log |B|). Create one with
`CREATE INDEX idx_fits_run ON fits(run);` when a JOIN is slow. Week 04's
rule still holds: ask "am I doing n² work?" before buying a bigger machine.

**Never build SQL by gluing strings.** This is wrong:

```python
conn.execute("SELECT * FROM runs WHERE run = " + user_text)  # don't
```

If `user_text` is `1; DROP TABLE runs;` you have just deleted the table. That
is **SQL injection**. Always pass values as `?` placeholders, the way Week 04
did:

```python
conn.execute("SELECT * FROM runs WHERE run = ?", (run_number,))
```

Python's `sqlite3` module:

```python
import sqlite3
conn = sqlite3.connect("data/results.db")
conn.row_factory = sqlite3.Row          # rows behave like dicts
rows = conn.execute("SELECT step, n FROM cutflow").fetchall()
for row in rows:
    print(row["step"], row["n"])
conn.close()
```

## 3. HTTP, JSON, and REST — programs talking

When a browser loads a page, it sends a **request** to a **server** and gets a
**response**. The agreed format for those bytes is **HTTP**. An **API**
(Application Programming Interface) is the same conversation aimed at
*programs*: instead of HTML for human eyes, the server returns structured
data.

A request has four parts:

- **method** — `GET` means "give me data" (must not change anything);
  `POST` means "here is data, do something with it."
- **URL** — which machine, and which **endpoint** (function) on it, e.g.
  `http://127.0.0.1:8000/cutflow`.
- **headers** — metadata: content type, authorization.
- **body** — the payload, if any.

The response has a **status code**: `200` success, `400` your request was
malformed, `401` you are not authorized, `404` that URL does not exist,
`422` the body failed validation, `500` the server itself broke.

The body language is almost always **JSON** (JavaScript Object Notation) —
Python dicts and lists as text. Differences from Python: `true`/`false`/`null`
instead of `True`/`False`/`None`, keys always double-quoted, no comments.

```python
import json
text = json.dumps({"step": "both global", "n": 59485})
back = json.loads(text)
```

**REST** is a convention, not a law: URLs name *resources* (`/runs/42`),
`GET` reads them, `POST` creates them, JSON in and out. You will see the word
on every job posting. This week's service is a REST API with two `GET`s.

## 4. FastAPI: a service in a file

**FastAPI** is a Python library that turns functions into HTTP endpoints.
**Uvicorn** is the program that waits for requests and calls those functions.
Together they are the default small-Python-service stack.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI()

class CutflowRow(BaseModel):
    step: str
    n: int

def connect():
    conn = sqlite3.connect("data/results.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/cutflow", response_model=list[CutflowRow])
def cutflow():
    conn = connect()
    rows = conn.execute("SELECT step, n FROM cutflow").fetchall()
    conn.close()
    return [CutflowRow(step=r["step"], n=r["n"]) for r in rows]

@app.get("/cutflow/{step}", response_model=CutflowRow)
def one_step(step: str):
    conn = connect()
    row = conn.execute(
        "SELECT step, n FROM cutflow WHERE step = ?", (step,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown step")
    return CutflowRow(step=row["step"], n=row["n"])
```

`BaseModel` is a **schema**: it names the fields and their types so a bad
payload becomes a `422` instead of a mysterious crash. `@app.get` attaches a
function to a URL. `{step}` in the path is a **path parameter**.

Run it:

```
uv add fastapi uvicorn
uv run uvicorn app:app --reload --port 8000
```

Then visit `http://127.0.0.1:8000/docs` — FastAPI generates a live catalog of
your endpoints. Tests do not use a browser; they use `TestClient`:

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
```

## 5. Secrets, env vars, logging

An **environment variable** is a name→string mapping the operating system
hands your process. API keys, database paths, and anything that must not be
committed live there — never in source, never in notebooks (Week 04
`.gitignore` already lists `.env`).

```python
import os
db_path = os.environ.get("RESULTS_DB", "data/results.db")
```

**Logging** is print statements with a level and a timestamp, meant to be
grepped later:

```python
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("service")
log.info("cutflow requested")
```

Log the *shape* of a request (path, status, milliseconds). Do not log the
Authorization header, the API key, or raw user text that might contain one.

## 6. Docker: the same machine, somewhere else

A **container** is a packaged process: your code, its libraries, and a
minimal Linux, isolated from the host. An **image** is the recipe's result;
a **container** is one running instance. **Docker** is the common tool.

A `Dockerfile` is the recipe:

```
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY app.py ./
COPY data/results.db ./data/results.db
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`FROM` starts from a known base. `COPY` brings your files in. `RUN` executes
build steps. `CMD` is what the container runs. `0.0.0.0` means "accept
requests from outside the container" — `127.0.0.1` inside a container is only
the container itself, a classic first-day footgun.

```
docker build -t week23-service .
docker run --rm -p 8000:8000 week23-service
```

`-p 8000:8000` maps host port 8000 to container port 8000. `--rm` deletes the
container when it stops. Week 44 will add health checks, model weights, and
CI that *builds* this image. This week, "it serves the same JSON as
TestClient" is the bar.

## 7. Worked example: the cut-flow service

End-to-end, the week is one repo:

1. A `build_db.py` that recreates `data/results.db` from known inputs (your
   Week-04 cut-flow, or a tiny `runs.csv` you commit as a fixture).
2. `app.py` as in §4, with `RESULTS_DB` from the environment.
3. `tests/test_app.py` covering `/health`, `/cutflow`, a 404, and a 422 if
   you add a `POST`.
4. `Dockerfile` as in §6.
5. `SERVICE.md`: three commands — build the db, run tests, run the container
   — and the JSON a `GET /cutflow` returns.

That is the professional shape. Capstone services, the Week-40 copilot's
tools, and the Week-44 model wrapper are this file with a different function
behind `GET /predict`.

## Check yourself

1. Write a SQL query that returns steps with `n < 40000` from `cutflow`.
2. Why is `JOIN` without an index a Big-O problem? What is the fix?
3. What is the difference between HTTP `GET` and `POST`?
4. Why does FastAPI return 422 on a bad body instead of crashing?
5. Why must the container listen on `0.0.0.0`, not `127.0.0.1`?
6. Where does an API key live, and where must it never live?

## Answers

1. `SELECT step, n FROM cutflow WHERE n < 40000;`
2. Nested loops over both tables, O(|A|×|B|). An index on the join key.
3. `GET` reads and must not change server state; `POST` submits data and may.
4. The `BaseModel` schema validates types before your function runs.
5. `127.0.0.1` inside the container is only the container; the host cannot
   reach it. `0.0.0.0` accepts mapped-port traffic.
6. In an environment variable (or a secret store). Never in git, notebooks,
   logs, or Docker image layers you push publicly.

## New terms

- **SQL / SELECT / WHERE / JOIN / GROUP BY** — the query language; pick columns;
  filter rows; combine tables on a key; aggregate per key.
- **index** — a lookup structure that turns a scan into a log-time find.
- **SQL injection** — sneaking commands in through glued-together queries.
- **HTTP / method / status code / endpoint** — the request protocol; GET/POST;
  the numeric result; one URL that runs one function.
- **JSON** — dicts-and-lists as text, the default API body.
- **REST** — the convention of URLs-as-resources and methods-as-verbs.
- **API / FastAPI / Uvicorn / schema (Pydantic)** — programs talking; the
  library; the server; the typed contract.
- **environment variable / logging** — process-level config; timestamped
  records that must not contain secrets.
- **Docker / image / container / Dockerfile** — the tool; the recipe result;
  a running instance; the recipe file.

## Going deeper

- FastAPI tutorial, first steps through TestClient — the library in its own
  words; short.
- Docker getting-started, through "Run your image as a container."
- SQLite *EXPLAIN QUERY PLAN* — see whether your JOIN used the index.
- Optional: `optional-flows.md` in this folder if your capstone is generative
  fast-sim and you want exact likelihoods, not a bound.
