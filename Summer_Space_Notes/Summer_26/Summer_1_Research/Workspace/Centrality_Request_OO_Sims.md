

                                                                                       #Main Task 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* You are a experimental high energy physics graduate student and your task today is as follows: 

1. Read and Learn from these directories and content  
   Path #1: /sphenix/user/plewis3323/Fall_2025/pi0_Eff/src
   Path #2: /sphenix/user/plewis3323/Spring_2026/calibration_tasks/coresoftware/calibrations/calorimeter/calo_emc_pi0_tbt
   
   
2. Now what I want after you learned the directory paths is to take info from Path #1 specifcially the main centrality mechanisms from that .cc/.h files 
   and edit Path #2 .cc/.h to where the centrality mechanisms can work with that source code framework (Main: I should say log every change and detail in readable format so i can understand) 
   

3. I also know that path #1 have a centrality_conversion method I want u to all make macro that allows my input root ntuples after I run fun4all to put it 
   through and see if it can work.  (don't actually run just code it up) 
   

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Task — Port Centrality Mechanisms into the TBT π⁰ Calibration Framework

## Role

You are an experimental high-energy physics graduate student working on sPHENIX. Today's job has three phases: (1) read and map two codebases, (2) port the centrality machinery from the older π⁰ efficiency code into the current tower-by-tower (TBT) π⁰ calibration code, and (3) write a standalone ROOT macro that exercises the ported `centrality_conversion` method on Fun4All output ntuples. Do not execute anything — write code only.

---

## Paths

**Path #1 — Source (read-only context).** Previous π⁰ efficiency analysis. Contains validated centrality logic that should be the reference implementation.

    /sphenix/user/plewis3323/Fall_2025/pi0_Eff/src

**Path #2 — Target (edit here).** Current TBT π⁰ calibration. Currently centrality-blind; needs the ported logic added.

    /sphenix/user/plewis3323/Spring_2026/calibration_tasks/coresoftware/calibrations/calorimeter/calo_emc_pi0_tbt

---

## Phase 1 — Read and Map

Before editing anything, walk both directories end-to-end.

1. List every `.cc` and `.h` file in each path with a one-sentence description of its purpose.
2. In **Path #1**, identify and extract:
   - Every block of code that touches centrality — node access (`CentralityInfo`, `MinimumBiasInfo`, etc.), bin selection, percentile cuts, MB-class definitions, helper structs/enums.
   - The `centrality_conversion` method specifically: signature, inputs, outputs, conversion table or formula, edge cases.
   - Any constants or thresholds tied to centrality.
3. In **Path #2**, identify:
   - The main analysis class (e.g. `pi0EtaByEta`) and its event loop hook points (`InitRun`, `process_event`, `End`).
   - Where centrality-aware logic would naturally insert itself.
   - Histograms/trees/outputs that should be split or binned by centrality once the port is done.

**Deliverable for Phase 1:** a file `PHASE1_MAPPING.md` in the target directory that maps each source feature to its target insertion site, in the form:

> Source `file.cc:lines` provides `<feature>` → belongs in target `file.cc` at `<hook>` because `<reason>`.

Do not edit any code yet.

---

## Phase 2 — Port the Centrality Mechanisms

Modify the target `.cc` and `.h` files so the centrality logic from Path #1 functions inside the calibration framework.

### Design constraints

- **Match the source's approach exactly.** Do not redesign, do not introduce alternative APIs, do not refactor. The source is the spec.
- **Additive only.** All existing TBT calibration behavior (mass peak fits, tower correction factors, the existing histogram set) must remain unchanged when centrality is disabled or absent.
- **Respect Fun4All node-tree semantics.** Do not register nodes that another upstream module already creates — pre-registered nodes silently block fresh writes and produce no error.
- **Use full absolute paths** in any input file lists.
- **Do not break `Fun4All_SIM_EMCal.C`** entry points; existing macro callers should still work.
- **Match the target's style** for indentation, braces, naming, and comment conventions. If the source and target styles disagree, follow the target.

### Change-log requirement

After every edit, append an entry to `CENTRALITY_PORT_CHANGELOG.md` in the target directory. Each entry must contain these fields, in order:

- A header line: `## <ISO timestamp> — <file>:<line range>`
- **What changed:** one-line summary of the edit.
- **Why:** the physics or framework reason. Tie it back to the source behavior being preserved. Real "why" examples:
  - "Central events have ~10× the multiplicity of peripheral events, so the photon-pair combinatoric background per event scales steeply — the source code applies a tighter ΔR cut in the 0–10% bin, which we replicate here."
  - "Source uses `getCentile()` which returns 0–100; target histograms expect 0–9 bin indices, so we convert via the same lookup table the source defines in `Centrality.h`."
- **Source reference:** `Path#1/file.cc:line` — what was ported from where.
- **Diff:** a unified diff snippet of the edit (use a fenced `diff` code block).

Group entries by target file. The log must be readable by someone who has not seen Path #1 — so explain the physics intent, not just describe what the code mechanically does.

---

## Phase 3 — Verification Macro (do not run)

Write a standalone ROOT macro that takes a Fun4All-output ntuple and pushes it through the ported `centrality_conversion` method, producing diagnostic plots and a TFile of results.

### Requirements

- **Filename:** `Macro_Verify_Centrality.C` in the target directory.
- **Invocation:** `root -l -b -q 'Macro_Verify_Centrality.C("path/to/ntuple.root")'`
- **Inputs:**
  - One ROOT ntuple path as argument (with a sensible default for convenience).
  - Optional second argument: output filename.
- **Behavior:**
  1. Open the ntuple and locate the centrality branch (use whatever name the target framework currently writes — confirm during Phase 1 mapping).
  2. Loop entries; pass each raw centrality value through `centrality_conversion`.
  3. Fill diagnostic histograms:
     - `h_raw_centrality` — raw centrality distribution.
     - `h_converted_centrality` — converted centrality distribution.
     - `h2_raw_vs_converted` — 2D correlation, raw on x, converted on y.
  4. Write all histograms to `centrality_verify_<input_basename>.root`.
- **Header comment block** at the top of the macro explaining purpose, expected ntuple structure, and the invocation line.
- **Do not execute the macro.** Write it only.

---

## Deliverables (final summary)

1. `PHASE1_MAPPING.md` — discovery output from Phase 1, in the target directory.
2. Modified target `.cc`/`.h` files with the centrality logic ported in.
3. `CENTRALITY_PORT_CHANGELOG.md` — every edit logged, in the target directory.
4. `Macro_Verify_Centrality.C` — standalone, uncompiled, unrun.

---

## Hard rules

- **Do not run anything.** No compilation, no Fun4All invocation, no macro execution, no `make`.
- **Do not modify Path #1.** It is read-only reference.
- **Do not delete files** in either path.
- **Do not introduce new external dependencies** without flagging them first and waiting for confirmation.
- **Stop and ask** if Path #1 contains multiple centrality conversion variants and it is not obvious which one to port — do not guess.








































































































