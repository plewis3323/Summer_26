# Object-Oriented Python — Progress Tracker

**Lesson:** Codecademy MLE Path → SWE for ML → *Introduction to Classes*
**Note file:** `OOP_Python_Lesson.ipynb`
**Started:** 8/12/26

---

## Status: COMPLETE (8/12/26)
All Introduction to Classes sublessons folded in from the **real Codecademy source** with their real exercises.
Notebook runs top-to-bottom clean (`nbconvert --execute --inplace`, exit 0). Verified outputs:
`type(5)` / `{}` / `[]` → int, dict, list; `type(cool_instance)` → `<class '__main__.CoolClass'>`;
`drummer.title` → Rockstar; Dog method → `Dogs experience 7 years...`; `how_many_kms(5)` → **8.045**;
`Circle(36.0)` → `New circle with diameter: 36.0`; FakeDict join → `This works! This too!`;
`hasattr`/`getattr` → False / 800; `.count` check → dict no, str yes, int no, list yes;
circumferences **37.68 / 113.04 / 35984.4**; `__repr__` → `Circle with radius 6.0 / 18.0 / 5730.0`;
`print(argus)` after `__repr__` → `Argus Filch`. Review `Student`/`Grade` cell ran with no error
(`pieter.add_grade(Grade(100))`).

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded into the notebook **from the real source**.

- [x] **Introduction to Classes — Types** (`type()`, operations defined at the type level) + real checkpoint (`type(5)`, empty `my_dict`, empty `my_list`)
- [x] **Introduction to Classes — Class** (class as template; `class CoolClass: pass`; PEP 8 capitalize names) + real checkpoint (`class Facade: pass`)
- [x] **Introduction to Classes — Instantiation** (class is a schematic; `cool_instance = CoolClass()` creates an object) + real checkpoint (`facade_1 = Facade()`)
- [x] **Introduction to Classes — Object-Oriented Programming** (instance = object; OOP; `type()` returns the class; `__main__` = this file) + real checkpoint (`facade_1_type = type(facade_1)`)
- [x] **Introduction to Classes — Class Variables** (shared by every instance; `object.variable`; `.title` notation) + real checkpoint (`Grade.minimum_passing = 65`)
- [x] **Introduction to Classes — Methods** (functions on a class; first param is `self`; auto-passed on call) + real checkpoint (`Rules.washing_brushes`)
- [x] **Introduction to Classes — Methods With Arguments** (`self` still implicit; extra params passed on the call) + real checkpoint (`Circle.area`; pizza/table/round-room diameters halved)
- [x] **Introduction to Classes — Constructors** (dunder methods; `__init__` runs on instantiate; args to `ClassName()` go to `__init__`) + real checkpoint (`Circle(36.0)` prints diameter)
- [x] **Introduction to Classes — Instance Variables** (per-object data; not shared; `object.variable = ...`) + real checkpoint (`Store` instances `.store_name`)
- [x] **Introduction to Classes — Attribute Functions** (`AttributeError`; `hasattr` / `getattr` with optional default) + real checkpoint (`hasattr(..., "count")` on dict/str/int/list)
- [x] **Introduction to Classes — Self** (`self` = this object; `self.url = url` in `__init__`; methods use `self` for class + instance vars) + real checkpoint (`Circle` radius from diameter; `.circumference()`)
- [x] **Introduction to Classes — Everything Is an Object** (`dir()`; dunders auto-added; native types are objects too) + real checkpoint (`dir(5)`, `dir` on a function)
- [x] **Introduction to Classes — String Representation** (default `print(obj)` is useless; `__repr__` must return a string) + real checkpoint (`Circle.__repr__`; print three circles)
- [x] **Introduction to Classes — Review** (types, instantiate, class vs instance vars, methods, `__init__` / `__repr__`) + real checkpoint (`Student` / `Grade`; `pieter.add_grade(Grade(100))`)
- [x] Q/A captured — `__init__` (what it is + what actually happens on `Circle(12)`), dunder methods, `.format()`
- [x] Real Codecademy exercises swapped in

## How we work this lesson
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** for that sublesson → I add it to the **Q & A** section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant section. Your code as-is.
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify the notebook runs clean, commit if asked.

## Open questions / to verify
- None — lesson compiled and verified 8/12/26.
