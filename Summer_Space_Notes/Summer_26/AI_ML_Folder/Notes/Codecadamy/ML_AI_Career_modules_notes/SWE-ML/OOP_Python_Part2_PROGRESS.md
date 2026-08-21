# Object-Oriented Python, Part 2 — Progress Tracker

**Lesson:** Codecademy MLE Path → SWE for ML → *Object-Oriented Programming*
**Note file:** `OOP_Python_Part2_Lesson.ipynb`
**Started:** 8/18/26
**Prereq:** `OOP_Python_Lesson.ipynb` (Introduction to Classes) — COMPLETE 8/12/26

---

## Status: COMPLETE — 11 sublessons folded in, final pass done (8/18/26)
Notebook runs top-to-bottom clean (`nbconvert --execute --inplace`, exit 0). Verified outputs: `e1.say_id()` / `e2.say_id()` → **My id is 1** / **My id is 2**; inheritance demo → **Nom Nom Nom...eating food!** twice, then **Bark!** / **Meow!**; `e3.say_id()` on the `Admin` instance → **My id is 3.**; overriding demo → **Rex says, Grrrr** / **Maisy says, Meow!**; overridden `e3.say_id()` → **I am an Admin**; `super()` demo → **Rachel says, Meow!**; `Admin.say_id()` with `super()` → **My id is 3.** / **I am an admin.**; multi-level chain → **Mr. Cranky says, Hi!**; `e4.say_id()` cascade → **They are in charge** / **My id is 4.** / **I am an admin.**; `Hybrid` two-parent demo → **Fluffy wags tail. Awwww** / **Fluffy bites. OUCH!**; `Admin(Employee, User)` checkpoint → **My username is 3** / **My role is Admin**; polymorphism demo → **Bear says, Grrrr** / **Maisy says, Meow!** / **beep.boop...BEEEEP!!!**; `meeting` loop → **My id is 1.** then Admin (id 2 + admin line) then Manager (id 3 + admin + in charge); dunder demo → **Horse** / **Penguin** / **HorsePenguin**; `Meeting` checkpoint → **ID 1 added.** / **ID 2 added.** / **ID 3 added.** / **3**; abstraction demo → **Maisy says, Meow!** / **Amber says, Woof!**; `Employee(AbstractEmployee)` → **My id is 1**; `dir(e)` encapsulation checkpoint → **_Employee__id**, **_id**, **id**; getter/setter demo → **None** / **10** / **_age Deleted**; `get_name`/`set_name`/`del_name` → **Maisy** / **Fluffy**

**Final pass (8/18/26):** TL;DR written (9 bullets, one per pillar/mechanic), Q & A section closed out (no questions arose on this lesson — Part 1's Q & A covers the class/dunder basics this one builds on), header spacing fixed on both sections. Re-verified with `jupyter nbconvert --to notebook --execute --inplace` → **exit 0**, all 21 code cells executed in order with the outputs listed above.

## Sublesson checklist
Mark `[x]` when the summary + any Q/A + code exercise are folded into the notebook **from the real source**.

- [x] **Introduction to Object-Oriented Programming** (programming paradigms; OOP = classes/objects with properties + methods; `Dog` recap; the four pillars named) + real checkpoint (`Employee` with `new_id` class-variable counter; `e1`/`e2` → ids 1, 2)
- [x] **OOP Pillar: Inheritance** (shared behavior pulled into a parent; `class Child(Parent)` syntax; `Animal` → `Dog`/`Cat` both get `eat()`) + real checkpoint (`class Admin(Employee): pass`; `e3.say_id()` → id 3)
- [x] **Overriding Methods** (subclass redefines a parent method; child definition found first; `Animal.make_noise` → `Cat.make_noise`) + real checkpoint (`Admin.say_id()` overrides `Employee.say_id()`)
- [x] **`super()`** (proxy to the superclass; call the parent method inside an override; `Cat.__init__` → `super().__init__(name, "Meow!")`) + real checkpoint (`Admin.say_id()` calls `super().say_id()` then adds its own line)
- [x] **Multiple Inheritance: Part 1** (multi-level chains; `Angry_Cat` → `Cat` → `Animal`; lookup walks up until found) + real checkpoint (`Manager` → `Admin` → `Employee`; cascading `super().say_id()`; `e4` → id 4)
- [x] **Multiple Inheritance: Part 2** (`class Hybrid(Dog, Wolf)`; `super()` resolves to the first parent listed; `Wolf.action(self)` to reach the other, passing `self` explicitly; siblings can't see each other) + real checkpoint (`class Admin(Employee, User)`; `User.__init__(self, self.id, "Admin")`; `e3.say_user_info()` → **My username is 3** / **My role is Admin**)
- [x] **OOP Pillar: Polymorphism** (same method name, different behavior; no shared parent required — `Robot` isn't an `Animal`; iterate a mixed list and call `make_noise()`) + real checkpoint (`meeting = [Employee(), Admin(), Manager()]`; loop `.say_id()`)
- [x] **Dunder Methods** (operator overloading as polymorphism; `__repr__`, `__add__`; `a1 + a2` → `a1.__add__(a2)`) + real checkpoint (`Meeting.__len__`; `m1 + e1/e2/e3`; `len(m1)` → 3)
- [x] **OOP Pillar: Abstraction** (`ABC` + `@abstractmethod`; cannot instantiate the abstract class; subclasses must implement the marked methods) + real checkpoint (`Employee(AbstractEmployee)` implements `say_id()`; `e1.say_id()` → **My id is 1**)
- [x] **OOP Pillar: Encapsulation** (public / `_protected` convention / `__private` name mangling to `_Classname__x`; dunders are not mangled) + real checkpoint (`dir(e)` shows `_Employee__id`, `_id`, `id`)
- [x] **Getters, Setters and Deleters** (`get_` / `set_` / `delete_` wrap a `_protected` attribute; setter can type-check and raise; `del` removes the attribute) + real checkpoint (`get_name` / `set_name` / `del_name`; `Maisy` / `Fluffy`; post-delete `get_name()` → AttributeError)

*(pillar sublessons split into their real Codecademy exercise names as you paste them)*

## How we work this lesson
1. You paste/describe a **sublesson's text** → I fold a concise summary + key points into the notebook.
2. You give me **Q/A** for that sublesson → I add it to the **Q & A** section (no need to ask permission — just do it).
3. You give me the **code exercise** → I add a runnable code cell under the relevant section. Your code as-is.
4. On **"done"** → final pass, update TL;DR, tick this checklist, verify the notebook runs clean, commit if asked.

## Open questions / to verify
- None — lesson closed out.
- If Codecademy adds a `@property` / decorator-syntax sublesson after *Getters, Setters and Deleters*, it isn't in here yet (not in the source as read on 8/18/26).
