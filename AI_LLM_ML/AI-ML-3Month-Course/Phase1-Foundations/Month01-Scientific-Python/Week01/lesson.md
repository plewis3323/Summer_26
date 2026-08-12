# Week 01 — Your first programs

~10 hrs. Before starting you should be able to: Nothing. This is the beginning.

By the end of this week you will have installed Python, typed commands into a
terminal, written and run your first programs, and — most importantly — broken
things and read the error messages calmly. Everything else in this course is
built on those skills.

## 1. What a computer actually does

A computer does one thing: it follows instructions, very fast, without judgment.
Each instruction is tiny — add two numbers, copy a value from one place in
memory to another, compare two values and jump to a different instruction
depending on the result. **Memory** is the computer's scratch space: billions of
numbered slots, each holding a number. The **processor** (CPU) is the part that
reads instructions and executes them, billions per second.

A **program** is a list of instructions written down in advance. When you
"run" a program, the processor starts at the top and works through it. The
program has no idea what you *meant* — it does exactly what the instructions
say, which is why a program with a typo does the wrong thing perfectly.

Nobody writes those tiny instructions by hand anymore. We write in a
**programming language** — a notation that humans can read, with rules strict
enough that a computer can translate it into instructions. **Python** is the
programming language of this course. It is the standard language of scientific
computing and machine learning, and it is one of the friendliest languages to
learn first: you type a line, Python runs it, you see the result.

One more word you will meet constantly: a **file** is a named chunk of data
saved on disk — a document, a photo, or a program. Files live in **folders**
(also called **directories**), which can contain other folders. The whole thing
forms a tree, and every file has an address in that tree called a **path**,
like `/home/parker/course/week01/hello.py`.

## 2. The terminal

You are used to driving a computer by clicking. The **terminal** (also called
the **shell** or **command line**) is a window where you drive it by typing
commands instead. It looks bare — a line of text and a blinking cursor — but it
is the professional's steering wheel: every tool in this course (Python, git,
uv, Jupyter) is started from it.

Open one:

- **Windows:** install WSL2 first — see `02-Setup-Guide.md` at the course root
  ("Day 0"). It is one command in an administrator PowerShell (`wsl --install`)
  plus a reboot, and it gives you Ubuntu Linux inside Windows. Then open the
  "Ubuntu" app: that is your terminal.
- **macOS:** open the built-in app called Terminal.
- **Linux:** open the app called Terminal (or Console).

You will see a **prompt** — something like `parker@laptop:~$` — which means
"the shell is waiting for a command." You type a command, press Enter, the
shell runs it and prints the result, and the prompt comes back.

The shell always has a notion of *where you are* in the folder tree, called the
**working directory**. The five commands below are 90% of daily terminal use.
Type each one and press Enter:

```
pwd
```

`pwd` = *print working directory*. It prints the path of the folder you are in.
Fresh terminals start in your **home folder** (written `~` for short).

```
ls
```

`ls` = *list*. It shows the files and folders in the working directory.

```
mkdir course
```

`mkdir` = *make directory*. This creates a new folder named `course` inside the
working directory. Run `ls` again — it is there now.

```
cd course
```

`cd` = *change directory*. You have moved into the new folder; `pwd` confirms
it. Two special targets: `cd ..` moves *up* one level (to the parent folder),
and `cd` alone jumps back to your home folder.

Finally, the most important key on the keyboard:

- **Tab** autocompletes. Type `cd cou` and press Tab — the shell finishes the
  name for you. If nothing happens, press Tab twice to see the options.
  Professionals never type full names; they type three letters and hit Tab.
- **Up arrow** recalls previous commands so you can edit and rerun them.
- **Ctrl-C** cancels a running command and gives you the prompt back. If the
  terminal ever seems stuck, this is the escape hatch.

Two habits that prevent all beginner terminal disasters: stay inside your home
folder, and never run a command containing `sudo` (which means "do this with
administrator power") unless you understand exactly what it does. Inside your
home folder, the worst you can do is delete your own files.

## 3. Installing Python with uv

You could install Python a dozen ways. This course uses one: **uv**, a tool
that installs Python itself, creates per-project environments, and manages
packages (you will learn what those words mean in Week 04 — for now, uv is
"the thing that gives me a working Python"). Install it by pasting this into
your terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(`curl` downloads a file from the internet; the `| sh` part runs it as an
installation script. This is the one time in the course you run a downloaded
script — it comes from uv's official site.) Close and reopen the terminal so
it learns about the new tool, then check:

```
uv --version
```

If that prints a version number, you are done. Now make a home for this week's
work and start Python:

```
cd ~/course
mkdir week01
cd week01
uv init
uv run python
```

`uv init` marks this folder as a Python project (it creates a few small files;
ignore them until Week 04). `uv run python` starts **Python** itself. The first
run may pause to download Python — that is uv doing the install for you.

## 4. Python as a calculator

You are now looking at a new prompt: `>>>`. This is the **interpreter** (also
called the **REPL**): you type one line of Python, it runs it immediately and
prints the result. It is the ideal place to experiment. Try:

```python
>>> 2 + 3
5
>>> 7 * 6
42
>>> 2 ** 10
1024
>>> 7 / 2
3.5
>>> 7 // 2
3
>>> 7 % 2
1
```

Arithmetic works as in math class: `+`, `-`, `*` (times), `/` (divide),
`**` (power), and parentheses group as usual. Two you have not seen: `//` is
**integer division** (divide and drop the remainder) and `%` is the
**remainder** (also called *modulo*).

Python has two kinds of numbers. An **integer** (`int`) is a whole number:
`7`, `-3`, `1024`. A **float** (floating-point number) is a number with a
decimal point: `3.5`, `0.001`, `2.99792458e8` — that last one is scientific
notation, $2.99792458 \times 10^8$. Dividing with `/` always gives a float,
even when it comes out even. Floats are approximations stored in a fixed
number of bits, which occasionally shows:

```python
>>> 0.1 + 0.2
0.30000000000000004
```

That is not a bug; it is the price of storing infinitely many decimals in 64
bits. It will matter later (Week 04's tests use tolerances for exactly this
reason); for now, just don't be alarmed.

To leave the interpreter, type `exit()` and press Enter (or press Ctrl-D).

## 5. Variables

A **variable** is a name attached to a value. You create one with `=`, which in
Python means **assignment** ("store this value under this name"), not equality:

```python
>>> energy = 200.0
>>> events = 3
>>> energy
200.0
>>> energy * events
600.0
```

Once a name exists you can use it anywhere a value could go, and you can
reassign it:

```python
>>> total = 0
>>> total = total + 5
>>> total
5
```

Read `total = total + 5` right-to-left: compute `total + 5` first, then store
the result back under the name `total`. This pattern — updating a running
total — is everywhere in programming.

Names may contain letters, digits, and underscores, and may not start with a
digit. Use lowercase words joined by underscores (`beam_energy`, `n_events`)
and make them mean something: `x` is fine for scratch work, cruel in a program
someone (future you) must read.

## 6. Strings

A **string** (`str`) is a piece of text: characters in quotes. Single and
double quotes both work; pick one and be consistent.

```python
>>> name = "muon"
>>> greeting = "hello, " + name
>>> greeting
'hello, muon'
>>> len(name)
4
```

`+` on strings glues them together (**concatenation**), and `len(...)` — your
first **function** call, a named operation you invoke with parentheses —
returns the number of characters. Strings have their own operations, invoked
with a dot; these are called **methods**:

```python
>>> name.upper()
'MUON'
>>> "  spaced out  ".strip()
'spaced out'
>>> "a,b,c".split(",")
['a', 'b', 'c']
```

The single most useful string tool is the **f-string**: put `f` before the
opening quote, and anything inside `{...}` is filled in with a computed value.

```python
>>> mass = 0.1057
>>> f"the muon mass is {mass} GeV"
'the muon mass is 0.1057 GeV'
>>> f"rounded: {mass:.2f} GeV"
'rounded: 0.11 GeV'
```

The `:.2f` is a **format specification**: show the value as a float with 2
digits after the decimal point. You will use f-strings in nearly every program
you write in this course.

One trap: `"2" + "3"` is `'23'` (gluing text), not `5`. Strings that look like
numbers are still strings. Convert with `int("2")` or `float("3.5")` — and
`str(42)` goes the other way.

## 7. print, input, and your first script

The interpreter shows each result automatically, but a real program must say
things explicitly. `print(...)` writes its arguments to the screen; `input(...)`
shows a message, waits for the user to type a line, and returns it *as a
string*.

The interpreter forgets everything when you close it. A **script** is a program
saved in a file so you can run it again. Python scripts are plain text files
ending in `.py`. Create one with a **text editor** — an app for editing plain
text. Any works; if you have none, VS Code is free and standard (install it,
then `code hello.py` in the terminal opens the file). Put this in a file called
`hello.py` inside `~/course/week01`:

```python
name = input("What is your name? ")
print(f"Hello, {name}.")
print(f"Your name has {len(name)} letters.")
```

Run it from the terminal (not from inside `>>>` — scripts run from the shell
prompt):

```
uv run python hello.py
```

Python executes the file top to bottom: line 1 asks and waits, lines 2–3
print. That is all a script is — the interpreter session you would have typed,
saved and replayable.

Because `input` returns a string, math on user input needs a conversion:

```python
answer = input("Beam energy in GeV? ")
energy = float(answer)
print(f"Two beams: {2 * energy} GeV total.")
```

## 8. Notebooks

A **notebook** is the other place Python runs in this course. It is a document
in the browser made of **cells** — some containing code you can execute (the
output appears right under the cell), some containing formatted text. Notebooks
are ideal for exploration and for exercises; scripts are for programs you run
repeatedly. The notebook app is called **Jupyter**. Install and launch it from
your project folder:

```
uv add jupyter ipykernel
uv run jupyter lab
```

(`uv add` fetches a **package** — code written by others that you install and
use; Week 03's NumPy is a package too.) A browser tab opens. Create a notebook
(Python 3 kernel), type `2 + 3` in the first cell, and press **Shift-Enter** to
run it. Everything you did in the interpreter works in a cell.

One warning worth learning early: cells run in the order *you* run them, not
top-to-bottom order. If results seem impossible, use the menu item
"Run All" (or "Restart kernel and run all") to re-execute the notebook cleanly
from the top. Back in the terminal, Ctrl-C (twice) shuts Jupyter down.

Your weekly exercises arrive as notebooks generated from `exercises.md` — see
`NOTEBOOK_RULES.md` at the course root. The setup is written for you; you fill
in the lines marked `# TODO`.

## 9. Reading error messages

You will now break things on purpose, because reading error messages calmly is
the highest-value skill of Week 01. When Python hits something it cannot do, it
stops and prints a **traceback** — a report of where it was and what went
wrong. Try each of these in the interpreter and read the output:

```python
>>> print(mesage)
NameError: name 'mesage' is not defined
```

A **NameError** means you used a name that does not exist — almost always a
typo, or a variable you have not created yet.

```python
>>> 2 +* 3
SyntaxError: invalid syntax
```

A **SyntaxError** means the line is not legal Python — the interpreter could
not even start running it. Look at the little arrow `^` under the line: the
mistake is at or just *before* it. (A missing closing quote or parenthesis
often reports on the *next* line.)

```python
>>> "2" + 3
TypeError: can only concatenate str (not "int") to str
```

A **TypeError** means the operation does not make sense for these types —
here, gluing a string to an integer. The fix is a conversion: `int("2") + 3`.

```python
>>> 1 / 0
ZeroDivisionError: division by zero
```

Errors in scripts add location information. Read a traceback **bottom line
first**: it names the error and describes it in English. Then look one line up
for the file name and line number where it happened. That is the whole method:
bottom line for *what*, line number for *where*. Long tracebacks (you will meet
them in Week 03 when library code is involved) work the same way — your code
is usually the *top* entry, the error is always the bottom line.

Errors are not verdicts. Every working program you will ever write passed
through dozens of tracebacks on the way. The difference between a beginner and
a professional is not fewer errors — it is calmer reading.

## 10. Worked example — a unit-conversion script

Time to tie the week together: terminal, a script, variables, floats,
f-strings, `input`, and error handling by design.

Particle physicists measure energy in **electronvolts (eV)**: the energy one
electron gains crossing one volt. It is a tiny unit — everyday physics uses
joules (J) — and the conversion is $1\,\text{eV} = 1.602 \times 10^{-19}$ J.
Accelerator energies come in GeV (giga-electronvolts, $10^9$ eV), so a proton
at 100 GeV carries $100 \times 10^9 \times 1.602\times10^{-19}$ J. Our script
does this arithmetic for any input.

Create `ev_to_joules.py` in `~/course/week01`:

```python
# Convert an energy from GeV to joules.
EV_TO_JOULES = 1.602e-19    # one electronvolt, in joules
GEV = 1.0e9                 # one GeV, in eV

answer = input("Energy in GeV? ")
energy_gev = float(answer)

energy_ev = energy_gev * GEV
energy_j = energy_ev * EV_TO_JOULES

print(f"{energy_gev} GeV")
print(f"  = {energy_ev:.4e} eV")
print(f"  = {energy_j:.4e} J")
```

New things, deliberately few: lines starting with `#` are **comments** —
notes for humans that Python ignores; names in ALL_CAPS are a convention for
**constants**, values the program never changes; and `:.4e` formats a float in
scientific notation with 4 decimals. Run it:

```
$ uv run python ev_to_joules.py
Energy in GeV? 100
100.0 GeV
  = 1.0000e+11 eV
  = 1.6020e-08 J
```

Sixteen nano-joules — a mosquito's push, carried by one proton. Now run it
again and type `ten` instead of a number. Read the traceback bottom-up:
`ValueError: could not convert string to float: 'ten'`, at the line with
`float(answer)`. A **ValueError** means the type was right (a string is what
`float` expects) but the *value* was unusable. You diagnosed it in ten
seconds. That is the week's real deliverable.

## Check yourself

1. What is the difference between the shell prompt (`$`) and the Python
   prompt (`>>>`)? What runs in each?
2. You are somewhere deep in the folder tree and lost. Which two commands
   orient you, and which single command returns you home?
3. What does `7 // 2` give, and how is it different from `7 / 2`?
4. After `x = 4` then `x = x * 3`, what is `x`? Explain the order in which
   the second line executes.
5. `input("n? ")` returned `"12"`, and `"12" * 2` gave `"1212"`. Why? Write
   the corrected line that yields `24`.
6. Write the f-string that prints `pi is about 3.14` given `pi = 3.14159`.
7. A traceback ends with `NameError: name 'pirnt' is not defined`. What
   happened, and where would you look for the line to fix?
8. Why does `0.1 + 0.2 == 0.3` come out `False` in Python?

## Answers

1. `$` is the shell waiting for terminal commands (`ls`, `cd`,
   `uv run python`); `>>>` is the Python interpreter waiting for Python code.
   Shell commands typed at `>>>` (or Python typed at `$`) produce errors.
2. `pwd` prints where you are and `ls` shows what is there; `cd` with no
   argument returns to your home folder.
3. `7 // 2` is `3` — integer division drops the remainder. `7 / 2` is the
   float `3.5`.
4. `x` is `12`. The right side `x * 3` is computed first using the current
   value 4, then the result is stored back under the name `x`.
5. `input` always returns a string, and `*` on a string repeats it. Fix:
   `n = int(input("n? "))` then `n * 2` — or `int("12") * 2` directly.
6. `f"pi is about {pi:.2f}"`.
7. A misspelled name — `pirnt` instead of `print`. The traceback line just
   above the error names the file and line number to fix.
8. Floats store values in binary with finite precision; neither 0.1 nor 0.2
   is exactly representable, and the tiny rounding errors do not cancel.
   Compare floats with a tolerance, never with `==` (Week 04 makes this a
   habit via `pytest.approx`).

## New terms

- **memory** — the computer's scratch space of numbered slots holding values.
- **processor (CPU)** — the hardware that executes instructions.
- **program** — a written list of instructions the computer follows.
- **programming language** — human-readable notation translatable into instructions.
- **file / folder (directory) / path** — named data on disk; a container of files; the address of either.
- **terminal (shell, command line)** — the window where you drive the computer by typed commands.
- **prompt** — the marker showing a program is waiting for your input.
- **working directory** — the folder the shell is currently "in".
- **home folder (`~`)** — your personal folder; where terminals start.
- **`pwd` / `ls` / `cd` / `mkdir`** — print location / list contents / move / make folder.
- **uv** — the course's tool for installing Python and managing packages.
- **interpreter (REPL)** — interactive Python: type a line, see the result.
- **integer (`int`) / float** — whole number / decimal number (finite-precision).
- **integer division (`//`) / remainder (`%`)** — divide dropping the remainder / the remainder itself.
- **variable / assignment (`=`)** — a name attached to a value / the act of attaching it.
- **string (`str`)** — text in quotes; **concatenation** glues strings with `+`.
- **function / method** — a named operation called with `()`; a function attached to a value, called with a dot.
- **f-string / format specification** — a string with `{...}` slots filled by values; the `:.2f`-style display rule.
- **`print` / `input`** — write to the screen / read a typed line (always a string).
- **script** — a program saved as a `.py` file, run with `uv run python file.py`.
- **text editor** — an app for editing plain text (e.g. VS Code).
- **notebook / cell / Jupyter** — a browser document of runnable cells; the app serving it.
- **package** — installable code written by others (`uv add name`).
- **comment (`#`) / constant** — a note Python ignores / a value the program never changes (ALL_CAPS by convention).
- **traceback** — Python's error report; read the bottom line first.
- **NameError / SyntaxError / TypeError / ValueError / ZeroDivisionError** — undefined name / illegal code / wrong type / right type but unusable value / dividing by zero.
- **electronvolt (eV)** — the energy an electron gains crossing one volt; particle physics' energy unit.

## Going deeper

All free; the lesson stands alone, these reinforce it.

- Severance, *Python for Everybody*, chapters 1–2 ("Why program?", "Variables,
  expressions and statements") — the same ground at a gentler pace, with
  videos; written for people who have never programmed.
- The official Python Tutorial, sections 1–3 — a second pass over the
  interpreter, numbers, and strings in the language's own voice.
- `02-Setup-Guide.md` (course root) — the reference for your exact setup (WSL2
  notes, the uv commands, what gets installed when).
