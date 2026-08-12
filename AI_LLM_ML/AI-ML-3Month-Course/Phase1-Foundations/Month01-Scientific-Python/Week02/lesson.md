# Week 02 — Control flow, functions, and data

~10 hrs. Before starting you should be able to: run Python in the interpreter,
a script, and a notebook (Week 01); use variables, ints, floats, strings, and
f-strings (Week 01); read a traceback bottom-up (Week 01).

Week 01's programs ran straight down the page, every line exactly once. Real
programs decide, repeat, and organize. This week adds the four missing pieces —
decisions (`if`), repetition (`for`/`while`), your own functions, and
collections (lists and dictionaries) — plus files, so programs can work on data
bigger than what you type in. At the end you build your first real program: a
word-frequency counter that reads any text file.

## 1. True, False, and comparisons

Programs decide by asking yes/no questions. A **comparison** produces a
**boolean** (`bool`) — a value that is either `True` or `False`:

```python
>>> 3 < 5
True
>>> 3 == 5
False
>>> 3 != 5
True
```

The operators: `<`, `>`, `<=`, `>=`, `==` (equal), `!=` (not equal). Note the
double `==`: single `=` is assignment (Week 01), double `==` is the question
"are these equal?". Mixing them up is the classic beginner error, and Python
refuses to let it happen silently (`if x = 3:` is a `SyntaxError`).

Booleans combine with `and`, `or`, `not`:

```python
>>> energy = 13.7
>>> energy > 2.0 and energy < 100.0
True
>>> not (energy > 100.0)
True
```

`and` is true only if both sides are; `or` if at least one is. One more
comparison you will use constantly: `in` asks whether something is inside a
collection — `"u" in "muon"` is `True`, and it works on lists and
dictionaries too (below).

## 2. Deciding: if, elif, else

An **`if` statement** runs a block of code only when a condition is true:

```python
energy = float(input("Energy in GeV? "))

if energy > 100.0:
    print("high energy")
elif energy > 1.0:
    print("medium energy")
else:
    print("low energy")

print("done")
```

Read it top to bottom: Python tests each condition in order and runs the block
under the *first* true one, skipping the rest; `else` catches everything that
fell through. `elif` ("else if") and `else` are optional.

The structure is marked by two things you must get exactly right:

- the **colon** `:` at the end of the `if`/`elif`/`else` line, and
- **indentation**: the block belongs to the `if` because it is shifted right.
  Python uses indentation the way English uses paragraphs — it is *syntax*,
  not decoration. Use 4 spaces per level, always (your editor's Tab key
  inserts them). Mismatched indentation is an `IndentationError`.

`print("done")` is back at the left margin, so it runs no matter which branch
was taken.

## 3. Lists

A **list** is an ordered collection of values in square brackets:

```python
>>> energies = [13.7, 2.1, 88.0, 0.4]
>>> len(energies)
4
>>> energies[0]
13.7
>>> energies[-1]
0.4
```

Positions are called **indices** and counting starts at zero: `energies[0]` is
the first item, `energies[3]` the last, and negative indices count from the
end (`[-1]` is the last item). Asking for `energies[4]` here raises an
**IndexError** — off the end of the list.

A **slice** takes a sub-list: `energies[1:3]` is `[2.1, 88.0]` — start
index 1, stop *before* index 3. Slices never include the stop position; this
"half-open" convention feels odd for a day and then becomes second nature
(`energies[:2]` + `energies[2:]` rebuilds the list with nothing doubled).

Lists are grown one item at a time with the method `.append`:

```python
>>> selected = []
>>> selected.append(13.7)
>>> selected.append(88.0)
>>> selected
[13.7, 88.0]
```

Starting from an empty list `[]` and appending inside a loop is the
fundamental accumulation pattern — you will write it hundreds of times.
Other list tools you need this week: `sum(energies)`, `min(...)`, `max(...)`,
`sorted(energies)` (returns a new sorted list), and `value in energies`.

Unlike strings, lists can be modified in place: `energies[0] = 14.0`
overwrites the first item. And a list can hold anything, including strings or
other lists.

## 4. Repeating: for loops

A **`for` loop** runs a block once per item of a collection:

```python
energies = [13.7, 2.1, 88.0, 0.4]

total = 0.0
for e in energies:
    total = total + e

print(f"mean = {total / len(energies)}")
```

Read `for e in energies:` as "for each item in `energies`, call it `e` and run
the block." Same colon, same indentation rules as `if`. The **loop variable**
`e` is a normal variable that takes each value in turn.

To loop a fixed number of times, use **`range`**: `range(5)` produces
0, 1, 2, 3, 4 (five numbers, starting at zero, stopping before 5 — the same
half-open convention as slices):

```python
for i in range(5):
    print(f"event {i}")
```

Loops and `if` nest inside each other, indenting one more level each time.
This is *selection* — keeping the items that pass a condition — the second
fundamental pattern:

```python
selected = []
for e in energies:
    if e > 1.0:
        selected.append(e)

print(f"kept {len(selected)} of {len(energies)}")
```

In particle physics this is called applying a **cut**: you keep the events
that pass a requirement and drop the rest. The word will come back in Week 03,
where NumPy does the same job in one line.

Python has a compact one-line spelling of exactly this build-a-list loop,
the **list comprehension**:

```python
selected = [e for e in energies if e > 1.0]
```

Read it inside-out: "each `e`, for `e` in `energies`, if `e > 1.0`." Use it
when it fits on one readable line; use the loop form otherwise. Both are
correct.

## 5. Repeating: while loops

A `for` loop runs once per item. A **`while` loop** runs as long as a
condition stays true — for when you don't know in advance how many times:

```python
activity = 1000.0
years = 0
while activity > 100.0:
    activity = activity / 2      # one half-life passes
    years = years + 12.3         # tritium half-life in years

print(f"below 100 after {years:.1f} years")
```

(Tritium is a radioactive form of hydrogen; every 12.3 years, half of any
sample decays away. The loop halves until the activity drops below the
threshold — we couldn't have written `range(...)` because the number of
halvings is the thing being computed.)

The danger: if the condition never becomes false, the loop runs forever — an
**infinite loop**. The program just hangs. Ctrl-C stops it (Week 01's escape
hatch). Before writing any `while`, ask: what inside the block moves the
condition toward false?

Rule of thumb: `for` when looping over things or a known count, `while` when
looping until a condition is met.

## 6. Functions: naming your own operations

You have called functions since Week 01 (`len`, `float`, `print`). Now you
write your own. A **function definition** packages a block of code under a
name, with **parameters** (the inputs it expects) and a **return value** (the
output it hands back):

```python
def kinetic_energy(mass, speed):
    return 0.5 * mass * speed ** 2

e1 = kinetic_energy(2.0, 3.0)
e2 = kinetic_energy(80.0, 1.5)
print(e1, e2)
```

`def` starts the definition; `kinetic_energy` is the name; `mass` and `speed`
are the parameters; the indented block is the **body**; **`return`** ends the
call and sends a value back to whoever called. Defining a function runs
nothing — the body executes only when the function is *called*, once per
call, with the parameters set to that call's arguments (`mass=2.0`,
`speed=3.0` for `e1`).

Why bother? Three reasons that grow with every week of this course:

- **Reuse** — write the formula once, call it everywhere; fix a bug once.
- **Naming** — `kinetic_energy(m, v)` states intent; `0.5 * m * v ** 2`
  inline makes the reader re-derive it.
- **Testing** — Week 04 will point `pytest` at exactly these units.

Two details worth learning now. First, variables created inside a function
(including parameters) are **local** — they exist only during the call and
are invisible outside. This is a feature: functions can't trample each
other's names. Second, a function with no `return` returns the special value
**`None`** ("nothing here") — if you `print(x)` and see `None`, you probably
forgot a `return`.

A function can return early and can be called by other functions:

```python
def classify(energy):
    if energy > 100.0:
        return "high"
    if energy > 1.0:
        return "medium"
    return "low"

def count_high(energies):
    n = 0
    for e in energies:
        if classify(e) == "high":
            n = n + 1
    return n
```

From here on, the default shape of every program you write is: a handful of
small functions, then a few lines at the bottom calling them.

## 7. Dictionaries

A list looks things up by position. A **dictionary** (`dict`) looks things up
by name — it stores **key: value** pairs in curly braces:

```python
>>> masses = {"electron": 0.000511, "muon": 0.1057, "proton": 0.9383}
>>> masses["muon"]
0.1057
>>> masses["pion"] = 0.1396      # add a new pair
>>> len(masses)
4
>>> "muon" in masses
True
```

(Those numbers are particle masses in GeV — the muon is a heavy cousin of the
electron; more on it in Week 04's project.) Asking for a key that isn't there
(`masses["higgs"]`) raises a **KeyError**, so check membership with `in`
first when unsure.

Looping over a dictionary visits its keys; `.items()` gives key and value
together:

```python
for name, mass in masses.items():
    print(f"{name}: {mass} GeV")
```

The dictionary's superpower is **counting and grouping**. To count occurrences
of things, use each thing as a key and keep a running total as its value:

```python
counts = {}
for name in ["muon", "pion", "muon", "muon"]:
    if name in counts:
        counts[name] = counts[name] + 1
    else:
        counts[name] = 1

print(counts)      # {'muon': 3, 'pion': 1}
```

Memorize that six-line pattern — it is the heart of this week's worked
example, and of half of all data analysis: *if the key exists, update it;
otherwise create it.*

## 8. Reading and writing text files

Programs become useful when they work on data that lives outside them. A
**text file** is a file containing plain characters — like the `.py` files
you already write. Python opens files with `open(...)`, best used in the
`with` form:

```python
with open("runlog.txt", "w") as f:
    f.write("run_001 42\n")
    f.write("run_002 87\n")
```

`open(path, "w")` opens the file for **writing** (`"w"` creates it, or wipes
and replaces it if it exists — careful). The **`with` block** guarantees the
file is properly closed when the block ends, even if an error occurs; `f` is
the open file, and `.write` puts a string into it. `"\n"` is the **newline
character** — the invisible character that ends a line; `write` does not add
it for you.

Reading back, mode `"r"`, and the file is loopable line by line:

```python
with open("runlog.txt", "r") as f:
    for line in f:
        parts = line.split()
        print(f"run {parts[0]} had {parts[1]} events")
```

Each `line` arrives *with* its trailing `"\n"`, and everything read from a
file is a string — the two facts responsible for most file-handling bugs.
The fixes are Week 01 tools: `.strip()` removes the newline (and stray
spaces), `int(...)`/`float(...)` convert. `line.split()` with no argument
splits on any whitespace, which handles both `"run_001 42"` and messier
spacing.

A wrong path raises `FileNotFoundError` — check `pwd`/`ls` in the terminal;
the script sees paths relative to the folder you *run it from*.

## 9. Worked example — a word-frequency counter

The week's parts, assembled into a real program: read any text file, count how
often each word appears, print the top 10. This is a genuinely useful tool —
the same shape as counting particle types in a detector log or tallying error
messages in a job's output.

Create `wordcount.py`:

```python
def clean_word(word):
    word = word.lower()
    word = word.strip(".,;:!?\"'()[]")
    return word

def count_words(path):
    counts = {}
    with open(path, "r") as f:
        for line in f:
            for word in line.split():
                word = clean_word(word)
                if word == "":
                    continue
                if word in counts:
                    counts[word] = counts[word] + 1
                else:
                    counts[word] = 1
    return counts

def top_n(counts, n):
    pairs = []
    for word, count in counts.items():
        pairs.append((count, word))
    pairs = sorted(pairs)
    pairs.reverse()
    return pairs[:n]

path = input("Text file to count: ")
counts = count_words(path)
print(f"{len(counts)} distinct words")
for count, word in top_n(counts, 10):
    print(f"{count:6d}  {word}")
```

Walk through the new bits:

- `clean_word` normalizes: `.lower()` so `The` and `the` count together, and
  `.strip(".,;:!?...")` with an argument strips those *specific* characters
  (punctuation) from the ends instead of whitespace.
- `continue` skips to the next loop iteration — used to discard words that
  were pure punctuation.
- The counting block is Section 7's pattern, verbatim.
- `top_n` converts the dictionary to a list of `(count, word)` pairs — a
  **tuple** is like a list but unchangeable, written with parentheses — so
  that `sorted` can order them by count; `.reverse()` flips to descending;
  `[:n]` slices the first `n`. The loop `for count, word in ...` unpacks each
  pair into two variables, exactly like `.items()` did.
- `{count:6d}` right-aligns the integer in 6 characters, so the numbers line
  up in a column.

Get a test file — any long public-domain text works. Project Gutenberg's
plain-text books are ideal (search "Project Gutenberg" and download, say,
*Frankenstein* as `.txt`), or use any lecture notes you have lying around:

```
$ uv run python wordcount.py
Text file to count: frankenstein.txt
7078 distinct words
  4194  the
  2976  and
  2850  i
  2745  of
  ...
```

Twenty-nine lines of Python, and it will happily chew through a million-word
file in under a second. Next week you will see what happens when the data is
numbers instead of words — and why loops like these get replaced by arrays.

## Check yourself

1. What is the difference between `=` and `==`?
2. `values = [10, 20, 30, 40]`. What are `values[1]`, `values[-1]`,
   `values[1:3]`, and `values[4]`?
3. Rewrite as a list comprehension: an empty list, then a `for` loop over
   `xs` appending `x * 2` whenever `x > 0`.
4. When do you reach for `while` instead of `for`? What must you check before
   running any `while` loop?
5. A function prints the right answer but the caller receives `None`. What is
   the bug?
6. Write the if/else pattern that counts occurrences of `key` in a dictionary
   `counts`, from memory.
7. You read `"42\n"` from a file and `int(line)` works, but
   `line == "42"` is `False`. Why?
8. In the worked example, why does `top_n` build `(count, word)` tuples
   instead of `(word, count)`?

## Answers

1. `=` assigns (stores a value under a name); `==` compares (asks whether two
   values are equal, producing `True`/`False`).
2. `20`; `40`; `[20, 30]` (stop index excluded); an `IndexError` — valid
   indices are 0–3.
3. `[x * 2 for x in xs if x > 0]`.
4. `while` when the number of repetitions isn't known in advance — you loop
   until a condition flips. Check that something in the body moves the
   condition toward `False`, or the loop never ends.
5. The function `print`s instead of `return`s (or the `return` is missing/
   mis-indented). Printing shows a value; only `return` hands it back.
6. ```python
   if key in counts:
       counts[key] = counts[key] + 1
   else:
       counts[key] = 1
   ```
7. Lines read from a file keep their trailing newline: the string is
   `"42\n"`, not `"42"`. (`int` tolerates the whitespace; `==` does not.)
   `.strip()` fixes it.
8. `sorted` orders tuples by their first element. With the count first, the
   sort is by frequency — which is the order we want the top-10 in. Word
   first would sort alphabetically.

## New terms

- **boolean (`bool`)** — a `True`/`False` value; produced by comparisons.
- **comparison operators** — `<`, `>`, `<=`, `>=`, `==`, `!=`.
- **`and` / `or` / `not`** — combine booleans; **`in`** — membership test.
- **`if` / `elif` / `else`** — run a block only when a condition holds.
- **indentation / block** — the 4-space shift that marks which lines belong to an `if`, loop, or function; `IndentationError` when inconsistent.
- **list** — ordered collection in `[...]`; **index** (from 0, negatives from the end); **IndexError** — off the end.
- **slice** — `xs[a:b]`, items from `a` up to but not including `b`.
- **`.append`** — grow a list by one item; the accumulation pattern starts from `[]`.
- **`for` loop / loop variable** — run a block once per item; **`range(n)`** — the counting sequence 0..n-1.
- **cut** — physics term for a selection condition applied to events.
- **list comprehension** — one-line build-a-list loop: `[f(x) for x in xs if c(x)]`.
- **`while` loop / infinite loop** — repeat while a condition holds / the bug where it always does.
- **`def` / parameter / body / `return` / return value** — the pieces of a function definition.
- **local variable** — exists only inside a function call.
- **`None`** — the "no value" value; what a function without `return` returns.
- **dictionary (`dict`) / key / value / KeyError** — lookup-by-name collection of `key: value` pairs; error for a missing key.
- **`.items()`** — loop over a dict's key/value pairs together.
- **text file / `open` / `with` block** — plain-character file; opening it; the block that guarantees it closes.
- **file mode** — `"r"` read, `"w"` write-and-replace.
- **newline (`"\n"`)** — the invisible end-of-line character; lines read from files keep it.
- **`continue`** — skip to the next loop iteration.
- **tuple** — an unchangeable list, written `(a, b)`; unpacks into variables in a `for`.
- **half-life** — time for half of a radioactive sample to decay.

## Going deeper

- Severance, *Python for Everybody*, chapters 3–9 (conditionals, functions,
  loops, strings, files, lists, dictionaries) — the same topics with many
  more worked examples; skim fast, slow down where the lesson felt thin.
- The official Python Tutorial, sections 4–5 (control flow, data structures)
  — the second pass; its notes on `range` and list methods fill corners the
  lesson skipped.
- Project Gutenberg (gutenberg.org) — free plain-text books; grab two or
  three as test inputs for your word counter.
