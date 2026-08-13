# Week 04 — Working like a professional

~10 hrs (plus the mini-project — see `project.md`). Before starting you should
be able to: drive the terminal and uv (Week 01); write functions and work with
files (Week 02); load, cut, and histogram data with NumPy/pandas/matplotlib
(Week 03).

Your Week 03 notebook found a peak. Now imagine a colleague asks: "Can I get
exactly that plot?" — and your honest answer today is "come to my laptop."
This week fixes that. You learn the four tools that turn code into *work
someone else can rerun*: **git** (a history of every change), **tests** (proof
the code does what you claim), **environments** (the exact same packages
everywhere), and **seeds** (the same "random" numbers every run). Then
`project.md` spends all four at once on real CERN data.

The standard for the rest of this course, stated once: *fresh clone +
`uv sync` + one command = same result.* Everything below serves that sentence.

## 1. git: a history for your code

You have already lived the problem: `analysis.py`, `analysis_v2.py`,
`analysis_v2_FINAL_fixed.py`. **Version control** replaces that mess with a
single folder whose entire history is recorded — every version of every file,
who changed what, when, and why. **git** is the universal version-control
tool. It is a program you drive from the terminal, and it stores the history
inside a hidden `.git` folder in your project. A project tracked by git is
called a **repository** ("repo").

One-time introduction (git signs every record with your name):

```
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

Make a repo out of a project folder:

```
cd ~/course/week04
uv init --package week04
cd week04
git init
```

(`uv init --package` is a variant of Week 01's `uv init` — it lays the
project out the professional way; Section 4 explains what it created.)
`git init` creates the `.git` folder. Now the core loop. git tracks changes
in two steps: you **stage** the files whose changes you want recorded, then
you **commit** — permanently record a snapshot of everything staged, with a
message:

```
git status                      # what changed? what is staged?
git add pyproject.toml src      # stage these
git commit -m "Start week04 package"
```

`git status` is the command you will run most in your life — it always tells
you where you stand and usually suggests the next command. The `-m` message
should say *why* in a short imperative sentence: "Add energy cut", "Fix
newline bug in reader". Your future self is the audience.

From here on, the rhythm is: edit → `git status` → `git diff` (shows the
exact changed lines) → `git add` → `git commit`. Small commits, one logical
change each. Inspect the history with:

```
git log --oneline
```

Each line is a commit: a short **hash** (the commit's unique ID, like
`3f2a91c`) and its message. Two consequences make this more than bookkeeping:
nothing committed is ever lost (you can view or restore any file from any
commit), and `git diff` answers "what did I change since the last working
state?" — the first question of all debugging.

One more file belongs in every repo: `.gitignore`, a plain text file listing
what git should *not* track — generated outputs, downloaded data, caches:

```
data/
__pycache__/
*.pdf
```

Rule: track the code that *produces* results, not the results themselves.

## 2. Branches, and publishing to GitHub

A **branch** is a movable label pointing at a line of commits. The default
branch is called `main`. Creating a new branch lets you commit experimental
work without touching `main` — and switch back to the working version at any
time:

```
git switch -c try-new-cut     # create branch and move to it
# ...edit, add, commit as usual...
git switch main               # main is exactly as you left it
git merge try-new-cut         # bring the branch's commits into main
```

`switch -c` creates-and-moves; `merge` folds the branch's commits into the
current branch. When the experiment is a dead end, switch back to `main` and
simply don't merge — that is the entire cost of a failed idea. Use branches
from the start for anything you're not sure about; `main` should always run.

git so far lives on your laptop. **GitHub** is a website that hosts git
repositories: your offsite backup, your portfolio, and (later in the course)
your collaboration hub. Create a free account, then a new *empty* repository
on the site (no README — the repo exists locally already). Connect and
publish:

```
git remote add origin git@github.com:YOURNAME/week04.git
git push -u origin main
```

A **remote** is a copy of the repo somewhere else; `origin` is the
conventional name for the main one. **push** uploads your commits to it.
(First time: GitHub needs an SSH key to know it's you — follow GitHub's
"Generate a new SSH key" doc, two commands and a paste into the website. Do
it once, forget it forever.) After the `-u` push, future publishing is just
`git push`. The reverse operations: `git pull` fetches new commits from the
remote, and `git clone <url>` copies an entire repository to a fresh
machine — the "fresh clone" in this course's reproducibility standard.

## 3. Environments with uv

Week 03's notebook needed `numpy`, `matplotlib`, `pandas`. Which versions? On
your colleague's machine, `pd.read_csv` might behave slightly differently in
a pandas from three years ago — or the import might just fail. An
**environment** is a private, per-project set of installed packages, so each
project carries exactly what it needs, at known versions, without projects
interfering with each other.

You have been using environments since Week 01 without ceremony — that is
what uv does. Now, the mechanics. A uv project contains:

- **`pyproject.toml`** — the project's description file: its name, and the
  list of **dependencies** (packages it needs) that `uv add` maintains.
- **`uv.lock`** — the exact resolved version of *every* package (including
  dependencies of dependencies), written automatically. You never edit it.
- **`.venv/`** — the environment itself, the actual installed files.
  Regenerable at any time, so it goes in `.gitignore`.

The division of labor: `pyproject.toml` says what you *want* ("numpy"),
`uv.lock` records what you *got* (numpy 2.3.1 and 14 exact transitive
versions), and `.venv` is where it lives. Commit the first two, ignore the
third. Then, on any machine:

```
git clone git@github.com:YOURNAME/week04.git
cd week04
uv sync
```

**`uv sync`** reads the lockfile and rebuilds the identical environment —
same packages, same versions, bit for bit. That is the whole magic of
"fresh clone + `uv sync`": the environment travels *as information* (two
small text files in git) rather than as gigabytes of installed code.

Daily commands, most of which you know: `uv add numpy` (add a dependency),
`uv add --dev pytest` (a **dev dependency** — needed to develop the project,
not to run it), `uv run <cmd>` (run anything inside the environment).

## 4. Laying out a package

`uv init --package week04` created this shape:

```
week04/
  pyproject.toml
  src/
    week04/
      __init__.py
```

A **module** is simply a `.py` file you can `import` — you have imported
other people's (`numpy`) since Week 03; now you import your own. A
**package** here means a folder of modules under a shared name (the
`__init__.py` file, usually near-empty, marks the folder as importable).
The `src/` ("source") layout is the professional convention: importable code
lives under `src/<name>/`, while tests, notebooks, and scripts live outside
it.

Add a module — put Week 01's conversion in `src/week04/convert.py`:

```python
EV_TO_JOULES = 1.602176634e-19

def gev_to_joules(energy_gev):
    return energy_gev * 1.0e9 * EV_TO_JOULES
```

Because uv installs your own project into its environment, any code run with
`uv run` — scripts, notebooks, tests, from any folder in the project — can
now say:

```python
from week04.convert import gev_to_joules
```

`from <package>.<module> import <name>` picks specific names out of a module;
plain `import week04.convert` brings the module in under its full name. Both
are fine; the `from` form is what you'll use most.

Why this matters beyond tidiness: notebooks are for *exploring*; functions
that compute results move into `src/` where they can be imported everywhere
— and, next section, *tested*. The workflow from now on: prototype in the
notebook, promote what works into the package, import it back.

## 5. Tests with pytest

A **test** is a small function that calls your code with known inputs and
checks the output automatically. `pytest` is the standard Python test runner:
it finds files named `test_*.py`, runs every function in them named
`test_*`, and reports which passed.

```
uv add --dev pytest
mkdir tests
```

In `tests/test_convert.py`:

```python
from week04.convert import gev_to_joules

def test_one_gev():
    assert gev_to_joules(1.0) == 1.602176634e-10

def test_zero():
    assert gev_to_joules(0.0) == 0.0

def test_proportional():
    assert gev_to_joules(200.0) == 200.0 * gev_to_joules(1.0)
```

**`assert`** is new syntax: `assert <condition>` does nothing if the
condition is true and raises an error if false. pytest turns each failed
assert into a readable report showing the values involved. Run the suite:

```
$ uv run pytest -q
...                                                    [100%]
3 passed in 0.02s
```

Green. Now the floating-point wrinkle from Week 01: `0.1 + 0.2 != 0.3`, so
`==` between computed floats is a trap. The fix is **`pytest.approx`**, which
compares within a tiny relative tolerance:

```python
import pytest

def test_scaling():
    assert gev_to_joules(13.7) == pytest.approx(2.194982e-18, rel=1e-4)
```

Rule: exact `==` for integers, strings, and counts; `approx` for anything
that went through float arithmetic.

What deserves a test? Every function in `src/`, with at least: one case with
a *known* answer (computed by hand or from a trusted reference — for physics,
a textbook value), one edge case (zero, empty, boundary), and — whenever you
fix a bug — a test that would have caught it (a **regression test**, which
guards against the bug's return). Tests are the software version of a
detector's closure test: not a proof of correctness, but a tripwire that
catches you the moment something silently breaks. From this week on,
`uv run pytest -q` green is part of "done" for every project in this course.

## 6. Seeds and reproducibility

Week 03 introduced `np.random.default_rng(42)`. Now the principle. Computers
generate "random" numbers deterministically: a **pseudorandom generator**
produces a sequence that passes every statistical test of randomness but is
completely determined by its starting state — the **seed**. Same seed, same
sequence, forever, on every machine:

```python
import numpy as np

rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
print(rng1.normal(size=3))
print(rng2.normal(size=3))    # identical
```

Unseeded (`default_rng()` with no argument), the generator seeds itself from
the operating system and every run differs. That is what you want for
production statistics, and precisely what you don't want while developing:
if the numbers change every run, you cannot tell your bug fix from luck.

The discipline, for every analysis in this course:

- Create one `rng = np.random.default_rng(SEED)` near the top, with `SEED` a
  named constant, and pass `rng` into any function that needs randomness —
  never create ad-hoc generators deep inside the code.
- A **deterministic** pipeline (same inputs → same outputs, always) is the
  goal; run it twice and `diff` the outputs to prove it.
- Randomness is only one enemy of determinism. The others you have already
  met: unpinned package versions (killed by `uv.lock`) and by-hand steps
  (killed by the next section).

## 7. One-command runs

The final rule: **the entire result must be reproducible by one command.**
Not "open the notebook and run cells 1, 3, 4, then 2" — one command. The
convention is a `run.py` at the project root that calls your package's
functions in order:

```python
from week04.convert import gev_to_joules

def main():
    print("energy conversion table")
    for e in [0.105, 1.0, 13.7, 200.0]:
        print(f"{e:8.3f} GeV = {gev_to_joules(e):.4e} J")

main()
```

```
uv run python run.py
```

`run.py` is deliberately thin — no logic, just orchestration: fetch, select,
fit, plot, print. All real work lives in tested functions under `src/`. The
payoff compounds: a one-command pipeline can be rerun after every change,
timed, tested in CI (a later-course topic), and — the point of this week —
handed to a colleague with a straight face.

The full standard, assembled: **git** preserves the code and its history;
**GitHub** publishes it; **`uv.lock`** freezes the environment; **tests**
prove the pieces; **seeds** pin the randomness; **`run.py`** replays the
whole thing. Fresh clone + `uv sync` + `uv run pytest -q` +
`uv run python run.py` = your result, on anyone's machine. That sentence is
the acceptance gate of `project.md`, and of every project after it.

## 8. Worked example — a tested, versioned micro-package

The whole week in ~15 minutes of typing. Goal: a package with one physics
function, one test file, a run script, and a clean two-branch git history.

```
cd ~/course
uv init --package radioactive
cd radioactive
git init
uv add numpy
uv add --dev pytest
```

Write `src/radioactive/decay.py` — the decay law $N(t) = N_0 \, 2^{-t/T}$,
where $N_0$ is the starting number of nuclei, $T$ the half-life, and $N(t)$
how many remain after time $t$ (Week 02's halving loop, now as a formula):

```python
def remaining(n0, half_life, t):
    return n0 * 2.0 ** (-t / half_life)
```

Write `tests/test_decay.py` — one known value (after exactly one half-life,
half remain), one edge, one float-safe check:

```python
import pytest
from radioactive.decay import remaining

def test_one_half_life():
    assert remaining(1000.0, 12.3, 12.3) == pytest.approx(500.0)

def test_time_zero():
    assert remaining(1000.0, 12.3, 0.0) == 1000.0

def test_two_half_lives():
    assert remaining(800.0, 5.0, 10.0) == pytest.approx(200.0)
```

Write `.gitignore` (`.venv/`, `__pycache__/`), then test, and commit the
working state:

```
uv run pytest -q          # 3 passed
git add .
git commit -m "Decay law with tests"
```

Now an experiment on a branch — add a `half_lives_until` function to
`decay.py` (Week 02's `while` loop, promoted into the package):

```
git switch -c add-threshold
```

```python
def half_lives_until(n0, threshold):
    n = 0
    amount = n0
    while amount >= threshold:
        amount = amount / 2.0
        n = n + 1
    return n
```

Add a test for it (`half_lives_until(1000.0, 10.0)` must be `7` — check it
by hand: 1000, 500, 250, 125, 62.5, 31.25, 15.6, 7.8), run `uv run pytest -q`,
commit, merge, and publish:

```
git add .
git commit -m "Add half_lives_until"
git switch main
git merge add-threshold
git remote add origin git@github.com:YOURNAME/radioactive.git
git push -u origin main
```

`git log --oneline` shows two commits; GitHub shows your code. Clone it into
a scratch folder, `uv sync`, `uv run pytest -q` — green on a "different
machine". You have just done, in miniature, exactly what `project.md` asks
for at full scale.

## 8. Pull requests and CI: a machine that runs your tests

Pushing to GitHub is a backup. A **pull request** (PR) is the professional
loop: you make a branch, push it, and ask GitHub to merge it into `main` only
after the tests pass and (later, in a team) after someone looks at the diff.
For this course you are both author and reviewer — still open the PR, still
read your own diff, still merge it on GitHub rather than with `git merge` on
the laptop. The habit is the point.

**CI** (continuous integration) is a machine that runs your tests on every
push so "it worked on my laptop" stops being the last word. GitHub Actions
does this with a YAML file in the repo. You do not need to learn YAML as a
language this week; you need to copy this file to
`.github/workflows/test.yml`, commit it, and watch the orange dot on GitHub
turn green:

```yaml
name: test
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen
      - run: uv run pytest -q
```

If `uv sync --frozen` fails because the runner has no lockfile yet, drop
`--frozen` until `uv.lock` is committed. A red X is a gift: it is a test
failure you would otherwise have discovered on a different day, on a
different machine, in front of someone else.

## 9. sqlite: a table that is a file

CSV is how data *arrives*. A **database** is how results *live* when you want
to ask questions later ("what was the cut-flow after the pt cut?") without
re-parsing a text file. **sqlite** is a database that is a single file — no
server to install, no password, the Python standard library talks to it
(`import sqlite3`). Week 23 will teach SQL properly; this week you use it as
a results notebook:

```python
import sqlite3

conn = sqlite3.connect("data/results.db")
conn.execute("CREATE TABLE IF NOT EXISTS cutflow (step TEXT, n INTEGER)")
conn.execute("DELETE FROM cutflow")
conn.execute("INSERT INTO cutflow VALUES (?, ?)", ("both global", 59485))
conn.commit()
rows = conn.execute("SELECT step, n FROM cutflow").fetchall()
conn.close()
```

`?` placeholders are how you pass values in — never glue strings together to
build a query (that is how SQL injection happens; Week 23/37 will say why).
The file `data/results.db` is generated, like the PDFs: do not commit it;
rebuild it from `run.py`.

## 10. Why some code is too slow (Big-O as vocabulary)

You do not need a computer-science course to say this out loud: **the number
of steps a program takes, as a function of the size of the input, is the
thing that kills you.** The shorthand is **Big-O**.

- A single pass over `n` rows (a pandas mask, a NumPy operation) is
  **O(n)** — double the events, double the time. Fine.
- A loop over `n` rows that, for each row, loops over `n` rows again is
  **O(n²)** — double the events, *four times* the time. At `n = 100,000`
  that is ten billion inner steps. That is why Week 03 taught masks instead
  of Python `for` over DataFrame rows.
- Looking up a key in a dictionary is **O(1)** — the time does not grow with
  how many keys you already stored. That is why dictionaries exist.

You are not being asked to prove anything. You are being asked, when a
pipeline is slow, to ask "am I doing n² work by accident?" before buying a
bigger machine. Week 23 will make this concrete on a JOIN.

## Check yourself

1. What is the difference between `git add` and `git commit`? Why two steps?
2. Your last commit worked; the code is now broken. Which two commands show
   you (a) what you changed and (b) the history to go back to?
3. What exactly does `uv.lock` record that `pyproject.toml` does not, and
   which of the three environment pieces is *not* committed?
4. Why does importable code live in `src/<name>/` instead of in the notebook
   that first developed it?
5. Write a pytest test asserting that `remaining(100.0, 5.0, 15.0)` is 12.5
   — float-safely.
6. Two runs of your pipeline give slightly different histograms. List the
   three usual suspects and the tool that eliminates each.
7. What may `run.py` contain, and what must it not?
8. State the course's reproducibility standard in one sentence.
9. What does CI do that `git push` alone does not?
10. A Python `for` over 10⁵ rows with another `for` inside is which Big-O, and
    what is the pandas alternative?

## Answers

1. `add` stages (selects which changes will be recorded); `commit` records a
   permanent snapshot of what's staged, with a message. Two steps let one
   commit capture exactly one logical change even when many files are dirty.
2. `git diff` (changes since the last commit) and `git log --oneline` (the
   history of commits you could restore).
3. `pyproject.toml` lists what you asked for ("numpy"); `uv.lock` pins the
   exact version of every package including transitive dependencies.
   `.venv/` is not committed — `uv sync` rebuilds it from the lockfile.
4. So it can be imported everywhere (other notebooks, `run.py`, tests) and
   covered by pytest; notebooks are for exploring, packages for keeping.
5. ```python
   import pytest
   from radioactive.decay import remaining

   def test_three_half_lives():
       assert remaining(100.0, 5.0, 15.0) == pytest.approx(12.5)
   ```
6. Unseeded randomness — fix the seed via `default_rng(SEED)`; different
   package versions — `uv.lock` + `uv sync`; by-hand steps run in varying
   order — a single `run.py` entry point.
7. Orchestration only: import tested functions from `src/` and call them in
   order. No formulas, no cuts, no logic that deserves a test of its own.
8. Fresh clone + `uv sync` + one command reproduces the result (with
   `pytest -q` green).
9. CI runs the test suite on a clean machine on every push, so a broken test
   is visible before anyone else clones.
10. O(n²); a vectorized boolean mask (one pass, O(n)).

## New terms

- **version control / git / repository** — recording every change to a project; the tool; a tracked project folder.
- **stage (`git add`) / commit / hash** — select changes to record / permanently record them with a message / a commit's unique ID.
- **`git status` / `git diff` / `git log`** — where you stand / exact changed lines / the history.
- **`.gitignore`** — the list of files git must not track (outputs, data, `.venv/`).
- **branch / `main` / merge** — a movable label for a line of commits; the default one; folding one branch's commits into another.
- **GitHub / remote / `origin` / push / pull / clone** — the hosting site; an elsewhere-copy of the repo; its usual name; upload commits; download commits; copy a whole repo fresh.
- **environment** — a project's private set of installed packages.
- **dependency / dev dependency** — a package the project needs to run / only to develop (tests, linters).
- **`pyproject.toml` / `uv.lock` / `uv sync`** — what you want / exactly what you got / rebuild the environment from the lockfile.
- **module / package / `src/` layout / `__init__.py`** — an importable `.py` file; a named folder of modules; the convention putting them under `src/`; the marker file.
- **test / pytest / `assert` / `pytest.approx` / regression test** — an automatic check with known inputs; the runner; the checking statement; float-tolerant comparison; a test pinning a fixed bug.
- **pseudorandom generator / seed** — deterministic "randomness"; the starting state that fixes the whole sequence.
- **deterministic** — same inputs always produce same outputs.
- **`run.py` / one-command run** — the thin entry point replaying the entire pipeline.
- **pull request (PR)** — a request to merge a branch into `main`, with a visible diff.
- **CI / GitHub Actions** — a machine that runs tests on every push; GitHub's CI product.
- **sqlite** — a database that is one file; queried with SQL; no server.
- **Big-O / O(n) / O(n²)** — how runtime grows with input size; one pass; nested passes.

## Going deeper

- *Pro Git* (Chacon & Straub, free online), chapters 1–3 — the standard git
  text; chapter 3 (branching) is the one that changes how you work. Do its
  examples by hand.
- uv official docs, the "Projects" guide — `pyproject.toml`, lockfiles,
  `uv sync`, in the tool's own words; short.
- pytest docs, "Get started" — one page; then the `approx` reference.
- Sandve et al., "Ten Simple Rules for Reproducible Computational Research"
  (PLOS Computational Biology, free) — this lesson's Sections 6–7 as ten
  memorable rules; read before starting `project.md`.
- GitHub Actions "Understanding GitHub Actions" — what the YAML file is doing.
- SQLite "SQLite as an Application File Format" — why a `.db` belongs next to
  the code that produced it. Week 23 teaches the query language.
