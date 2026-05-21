# Summer_26

Working repository for **Summer 2026 self-learning, coursework, research, and
projects** during my masters program. Built around the schedule in
[`Summer_planner.txt`](Summer_planner.txt).

> **Period:** May 15 – Aug 20, 2026
> **Lives in:** `/home/plewis/Summer_Directory_26` (WSL Ubuntu)
> **Remote:** [github.com/plewis3323/Summer_26](https://github.com/plewis3323/Summer_26)

---

## Quick navigation

| Folder | What's there |
| --- | --- |
| [Summer_planner.txt](Summer_planner.txt) | **Start here.** Full weekly plan: AI/ML hours, physics review schedule, coding goals, sPHENIX milestones. |
| [AI_LLM_ML/](AI_LLM_ML/) | AI / Machine Learning self-study (10 hrs/wk). Personal Claude course, Codecademy, Anthropic modules. |
| [1st year physics review/](1st%20year%20physics%20review/) | Foundational physics review. Teaching skills folder with Stat Mech, QM, Colliders/TDR references. My physics notes (Scrivener). |
| [Nuc1_Nuc2/](Nuc1_Nuc2/) | Nuclear Physics 1 & 2 — papers, notes, final readings (Pythia/strings/QGP). |
| [Summer_1/](Summer_1/) | Summer #1 block (May 15 – Jul 15): Nuc #2, Stat Mech, exp HEP Part 1. Research and AI/ML subfolders. |
| [Summer_2/](Summer_2/) | Summer #2 block (Jul 22 – Aug 18): Nuc #1, exp HEP Part 2. |
| [C++ review/](C++%20review/) | C++ self-study (HSF training site, Codecademy modules). |
| [ROOT review/](ROOT%20review/) | ROOT primer + personal Claude module notes. |
| [Shell scripts class/](Shell%20scripts%20class/) | Shell scripting class materials. |
| [Simulations_page/](Simulations_page/) | sPHENIX simulations — Pythia, Geant4, calorimeter sim examples. |
| [sPHENIX_Software_Page/](sPHENIX_Software_Page/) | sPHENIX software/framework course materials. |
| [Research_Creative_workPage/](Research_Creative_workPage/) | Active research — papers I'm reading (sPHENIX EMCal calib, jets in O+O, AI/HEP). |

---

## Summer plan at a glance

(See [`Summer_planner.txt`](Summer_planner.txt) for the full version.)

### AI / ML self-learning — 10 hrs/wk (Mon–Wed)
- **Personal Claude classes** (3 hrs/wk) — Months 1–3 modules
- **Codecademy** (4 hrs/wk) — AI Engineer, ML/AI Engineer, Building ML Modules, Data Science: ML
- **Anthropic modules** (3 hrs/wk) — AI Capabilities, AI Fluency, Claude 101, Claude Code 101/in Action, Claude API, Agent Skills, Subagents, MCP basics + advanced

### Physics master syllabus — 15 hrs/wk (Mon–Sat)
- **Summer #1** (May 15 – Jul 15): Nuc #2, Stat Mech, exp HEP Part 1
- **Summer #2** (Jul 22 – Aug 18): Nuc #1, exp HEP Part 2
- **Fall #1** (Sep–Oct): Quantum, RHI & Jets, E&M
- **Fall #2** (Nov–Dec): QFT #1, Classical Part 1
- **Spring #1** (Jan–Feb): QFT #2, Classical Part 2
- **Spring #2** (Mar–May): Astro & Nuclear, Optics
- Full list of review topics: Nuc 1/2, Stat Mech, Quantum, E&M, Classical, QFT 1/2, RHI & Jets, exp HEP, Astro & Nuclear Astro, Optics

### Java / C++ / ROOT — 5 hrs/wk (Mon–Wed)
- Codecademy intermediate & advanced C++
- HSF C++ training: https://hsf-training.github.io/hsf-training-cpp-webpage/
- ROOT Primer + personal Claude ROOT modules

### sPHENIX software & simulations — 5 hrs/wk (Thu–Sat)

---

## Editing workflow

The working copy lives **once** in WSL at `/home/plewis/Summer_Directory_26`.
You can edit it from either the WSL terminal or from Windows — they hit the
same files.

### From the WSL terminal

```bash
cd ~/Summer_Directory_26
git status
# edit files...
git add <files>
git commit -m "your message"
git push
```

### From Windows (PowerShell, File Explorer, VS Code)

- **File Explorer / PowerShell path:** `\\wsl$\Ubuntu\home\plewis\Summer_Directory_26`
- **VS Code:** install the "WSL" extension, then `code .` from inside the WSL terminal — opens this folder in a Remote-WSL window
- **PowerShell:**
  ```powershell
  Set-Location \\wsl$\Ubuntu\home\plewis\Summer_Directory_26
  git status
  ```

Both terminals hit the same `.git`, so there's nothing to sync between them.

### Pulling on a fresh machine

```bash
git clone git@github.com:plewis3323/Summer_26.git ~/Summer_Directory_26
```

### Deleting content

Just delete the files / folders the normal way, then:

```bash
git add -A      # stages the deletions too
git commit -m "remove <thing>: <reason>"
git push
```

To remove something from history (e.g. accidentally committed a huge file),
use `git filter-repo` — see notes in `plewis3323/Masters` README for an
example.

---

## What lives outside this repo

Things excluded from git, with where to find them:

| Excluded | Why | Where it actually lives |
| --- | --- | --- |
| `1st year physics review/Textbook files/Textbooks/` and 24 sibling textbook subdirs (~35 GB) | Reference material, not user-authored; some are copyrighted | Still on disk under WSL `~/Summer_Directory_26/1st year physics review/Textbook files/` |
| `AI_LLM_ML/Claude_Personal_AI_ML_Class_and_Notes/` | Already its own GitHub repo + lives in `plewis3323/Masters` | Run `git clone git@github.com:plewis3323/Claude_Personal_AI_ML_Class_and_Notes.git` |
| Files >= 50 MB (3 specific PDFs) | GitHub warns >50 MB; one is 207 MB which exceeds the 100 MB push limit | Listed at the bottom of `.gitignore`; still on disk |
| `.venv/`, `__pycache__/`, `*.pyc`, `*.so` | Build / runtime artifacts | Regenerate with the same setup steps in the relevant project |
| `*.ova` (the 11 GB `ubuntu-compphys.ova`) | VM image, too big for git | Was deleted from disk to free space |
| Zone.Identifier alt-data-stream files | Windows download metadata, not real content | Auto-created when files are downloaded on Windows; safe to delete |

See [`.gitignore`](.gitignore) for the full rule set.

---

## OneDrive copy

A parallel (smaller, mostly polished) copy of this content also lives at
`C:\Users\plewi\OneDrive - Ohio University\Fall_2025_Spring_Summer_26\Summer_Directory`.
That copy is the OneDrive-synced version — it has the cleaner subset of
files but is **not** a git working tree. The WSL location is the source of
truth going forward; treat the OneDrive copy as an automatic backup.
