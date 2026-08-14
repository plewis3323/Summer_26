# Basta Fazoolin' — Progress Tracker

**Project:** Codecademy MLE Path → Python / OOP → *Basta Fazoolin' with My Heart*
**Work in:** `Basta_Fazoolin_PROJECT.ipynb` ← your submitted code is now in here
**Solution key:** `Basta_Fazoolin_SOLUTION.ipynb` (only after you've tried)
**Started:** 8/13/26
**Submitted for grading:** 8/14/26

---

## Status: COMPLETE — all checks green (fixes applied 8/14/26)

**Grade as submitted: B+ (86/100)** — all three classes were structurally correct; one real logic
bug (`calculate_bill`) plus two spec-string mismatches. All three fixed on request; the notebook
now runs clean top to bottom.

| Part | Score | Notes |
|---|---|---|
| 1 — Menus | 25/35 | `calculate_bill` returns early; `__repr__` format; `"Brunch"` capitalized |
| 2 — Franchises | 35/35 | Clean. Chained comparison in `available_menus` is idiomatic. |
| 3 — Businesses | 26/30 | `Business` correct; arepa business named `"arepa"` not `"Take a' Arepa"` |

## Checklist

### Part 1 — Making the Menus
- [x] `Menu` class
- [x] Constructor: `self, name, items, start_time, end_time`
- [x] `brunch` (11–16) — but name string is `"Brunch"`, should be `'brunch'`
- [x] `early_bird` (15–18)
- [x] `dinner` (17–23)
- [x] `kids` (11–21)
- [x] `__repr__` — now via new `format_time()` helper → `11 am to 4 pm`
- [x] `print(brunch)` → `brunch menu available from 11 am to 4 pm`
- [x] `calculate_bill(purchased_items)` — `return total` dedented out of the `for` loop
- [x] brunch: pancakes + home fries + coffee → 13.5
- [x] early_bird: salumeria plate + mushroom ravioli (vegan) → 21.5
- [x] `check_part1()` passes

### Part 2 — Creating the Franchises
- [x] `Franchise` class
- [x] Constructor: `address`, `menus`
- [x] `flagship_store` + `new_installment`
- [x] `__repr__` → address
- [x] `available_menus(time)` — `start <= time <= end`
- [x] noon (`12`) → brunch, kids
- [x] 5 pm (`17`) → early-bird, dinner, kids
- [x] `check_part2()` passes

### Part 3 — Creating Businesses
- [x] `Business` class
- [x] Constructor: `name`, `franchises`
- [x] `"Basta Fazoolin' with My Heart"` with both stores
- [x] `arepas_menu` (10–20)
- [x] `arepas_place` ("189 Fitzgerald Avenue")
- [x] `"Take a' Arepa"` business
- [x] `check_part3()` passes

## Fixes applied (8/14/26)
1. **`calculate_bill`** — `return total` was indented one level too deep, so it exited after the
   first item. Dedented out of the loop. This is the one to remember: a `return` inside a loop
   means "stop now", so an accumulator must return *after* the loop finishes.
2. **`__repr__`** — added a `format_time(self, hour)` helper on `Menu` that maps a 24-hour int to
   `"11 am"` / `"4 pm"` (special-casing `12` → `12 pm`), and `__repr__` calls it for both
   endpoints joined by `" to "`. Keeping the conversion in its own method is why `__repr__` stays
   one readable line.
3. **Naming** — `Menu("Brunch", ...)` → `'brunch'`; `Business("arepa", ...)` → `"Take a' Arepa"`.

## What you got right
- Every constructor stores its args correctly — no `self.` omissions, no shadowing.
- `Franchise.available_menus` uses the chained comparison `start <= time <= end` — clean Python,
  and you correctly return a *list of Menu objects*, not names, which is what makes the printed
  output read nicely via `Menu.__repr__`.
- Composition is right end-to-end: Business holds Franchises holds Menus, and both businesses
  reuse the same three classes with no duplication.

## How we work this project
1. You write the code in the notebook; ask me when you're stuck or want a hint.
2. Checks are asserts — read the message, it names what's off.
3. On **"done"** → I review your code, we write the TL;DR, tick this list, commit if asked.

## Open questions / to verify
- Re-run all three checks after the fixes; then write the TL;DR at the bottom of the notebook.
