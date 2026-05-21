# 1st year physics review

Foundational physics review — reference materials, my notes, and the
teaching-skills folder used during 1st-year coursework.

## Contents tracked in this repo

- [`Grad_Review_Plan_Jun-Aug.pdf`](Grad_Review_Plan_Jun-Aug.pdf) — review plan for Jun–Aug
- [`Fall_2025_SP_26_Sum_26_Teaching_skills/`](Fall_2025_SP_26_Sum_26_Teaching_skills/) — reference PDFs by topic:
  - `Stat Mech/` — Pathria, Schroeder (the 207 MB Chemistry2e and >50 MB Thermal Physics PDFs are excluded — see below)
  - `QM/` — pdfcoffee Quantum Mechanics 3rd ed
  - `Colliders,TN,TDRs/` — ATLAS/ALICE/STAR/CMS/PHENIX TDRs, detector docs
  - top-level: `hodgesEMCalorimetry.pdf`, `Grad_Review_Plan_Jun-Aug.pdf`, `20260209_VW_Fun4All_KFP.pdf`
- [`Textbook files/My physics Notes Folder/`](Textbook%20files/My%20physics%20Notes%20Folder/) — **my own notes (Scrivener)**:
  - `Masters_review_topics.scrvn` — full review topics
  - `Skills Assignments/`:
    - `CM notes.scrvn` — Classical Mechanics
    - `Quantum notes.scrvn` — Quantum Mechanics
    - `Thermo and SM notes.scrvn` — Thermo / Stat Mech
    - `Statistics notes.scrvn` (+ DESKTOP backup variant)
    - `Codecamp  notes.scrvn`
    - `Skills Physics assignments.docx`
  - `Comp Tutorials class/`:
    - `Class Instructions/CompPhys_Fall23 - OneDrive.mhtml`
    - `Class Notes, Hw's, projects/Comp methods tutorial notes.scrvn`
    - `Coding Notes/FirstProgram.cpp`
  - `Chapter_2_CM_HM_Notes/` (if present)

## Excluded from git

- `Textbook files/` — all 25 subject-named subdirectories of reference textbooks (~35 GB) are excluded by `.gitignore`. Includes Classical Mechanics, EMF, Quantum, Stat Mech, Optics, dynamics, etc. They stay on disk in WSL at `~/Summer_Directory_26/1st year physics review/Textbook files/`. Only `My physics Notes Folder/` (your own work) is tracked.
- `Textbook files/My physics Notes Folder/Comp Tutorials class/Coding Files/ubuntu-compphys.ova` — 11 GB VM image, deleted from disk.
- Three specific PDFs >= 50 MB in `Fall_2025_SP_26_Sum_26_Teaching_skills/Stat Mech/` and `Colliders,TN,TDRs/` — see the `.gitignore` at the repo root for the exact paths.
