# School Catalogue — Progress Tracker

**Project:** Codecademy MLE Path → Python / OOP → *School Catalogue*
**Work in:** `School_Catalogue_PROJECT.ipynb` ← your submitted code is now in here
**Started:** 8/19/26
**Notebook built:** 8/19/26

---

## Status: COMPLETE — all checks green, runs clean top to bottom

**Grade as submitted: A- (91/100)** — inheritance is correct everywhere: right parameters, right
`super()` argument order, right hardcoded level strings, `__repr__` properly extended rather than
retyped. Two deductions: a cosmetic double-period in both child `__repr__`s, and `MiddleSchool`
never written. Both fixed in the notebook's review section; your original code is preserved above it.

| Part | Score | Notes |
|---|---|---|
| 1 — `School` | 33/35 | Correct. Getter naming inconsistent (`getname` vs `set_number_of_students`). |
| 2 — `PrimarySchool` | 30/32 | Clean inheritance. `..` from parent's trailing period. |
| 3 — `HighSchool` | 30/32 | Same shape, same `..`. Snake_case here but not in Part 1. |
| Spec coverage | -2 | `MiddleSchool` from the narrative was never created. |

## Checklist

### Part 1 — The `School` class
- [x] Step 1 — `School` class
- [x] Steps 2–3 — constructor `self, name, level, numberOfStudents`, all stored on `self`
- [x] Step 4 — getters for all three properties
- [x] Step 5 — setter for `numberOfStudents` only
- [x] Step 6 — `__repr__` → `A {level} school named {name} with {numberOfStudents} students`
- [x] Step 7 — tested with getters, setter, and `print()`
- [x] `check_part1()` passes

### Part 2 — `PrimarySchool`
- [x] Step 8 — inherits from `School`
- [x] Step 9 — constructor takes `name`, `numberOfStudents`, `pickupPolicy` (no `level`)
- [x] Step 10 — `super().__init__(name, "primary", numberOfStudents)` — correct arg order
- [x] Step 11 — `self.pickupPolicy` set
- [x] Step 12 — `getpickupPolicy()`
- [x] Step 13 — `__repr__` overridden via `super().__repr__()`
- [x] Step 14 — tested
- [x] `check_part2()` passes

### Part 3 — `HighSchool`
- [x] Step 15 — class, `sportsTeams` property, getter, `__repr__` override
- [x] Step 16 — tested
- [x] `check_part3()` passes

### Beyond the numbered tasks
- [x] `MiddleSchool` — required by the narrative, not by any numbered step (added in review section)
- [x] Combined catalogue: one list holding all three school types, printed in a loop

## Fixes applied (8/19/26)
1. **Double period.** `School.__repr__` ended with `students.`, and each child appended
   `". The pickup policy is ..."` → `students.. The pickup policy`. Dropped the trailing period
   from the parent; children keep their leading `". "`. **The rule to remember: a method designed
   to be extended by subclasses shouldn't own the closing punctuation — the parent can't know
   whether it's the end of the sentence.**
2. **`MiddleSchool` added.** Two-line constructor forwarding to `super()` with `"middle"`.
   No new properties, getters, or `__repr__` — it inherits all six members from `School`.
3. **Getter naming unified to snake_case** — `get_name`, `get_level`, `get_number_of_students`,
   `get_pickup_policy`, `get_sports_teams`. Previously three conventions coexisted in one file.

## What you got right
- **`super().__init__()` argument order.** The single easiest thing to get wrong here — the parent
  is `(name, level, numberOfStudents)` and the child passes the level string into the *middle*
  slot. Getting this backwards sets `level = 300` and fails silently until something prints.
- **Dropping `level` from the child constructors.** You correctly saw that a `PrimarySchool` is
  always `'primary'`, so it's a hardcoded constant, not a caller-supplied argument.
- **`super().__repr__()` for extension.** Both children build on the parent's string rather than
  duplicating it, which is exactly what step 13's hint was steering toward.
- **`isinstance` relationships hold** — both children are genuine `School`s and inherit every
  getter without redeclaring any of them.

## How we work this project
1. You write the code; ask for a **hint**, a **walkthrough**, or **code** when stuck.
2. Checks are asserts — read the message, it names what's off.
3. On **"done"** → code review, TL;DR, tick this list, commit if asked.

## Open questions / to verify
- Nothing outstanding. Notebook executes clean; all four checks green.
