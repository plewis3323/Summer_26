

<!-- ===== LESSON: welcome ===== -->

<div class="hero">

# Welcome to sPHENIX Software Training

Your 8-week guided journey from zero to productive sPHENIX analyst.
Learn the Fun4All framework, master ROOT, write real physics analysis
code, and submit production jobs at BNL.

<div class="hero-stats">

<div class="hero-stat">

<div class="hero-stat-num">

4

</div>

<div class="hero-stat-label">

Modules

</div>

</div>

<div class="hero-stat">

<div class="hero-stat-num">

33

</div>

<div class="hero-stat-label">

Lessons

</div>

</div>

<div class="hero-stat">

<div class="hero-stat-num">

50+

</div>

<div class="hero-stat-label">

Code Examples

</div>

</div>

<div class="hero-stat">

<div class="hero-stat-num">

8wk

</div>

<div class="hero-stat-label">

Self-Paced

</div>

</div>

</div>

</div>

<div class="content-section">

## How This Course Works

This course is organized into 4 modules that map onto the 8-week
onboarding timeline used by sPHENIX mentors at Brookhaven National
Laboratory. Each module contains a sequence of short lessons. Every
lesson has:

  - **Explanation** — concepts in plain language, with diagrams where
    useful
  - **Code you can copy** — real, working sPHENIX code patterns
  - **Exercises** — hands-on tasks to cement what you learned
  - **Callouts** — tips, warnings, and gotchas from experienced analysts

Your progress is saved automatically in your browser. Use the sidebar to
jump to any lesson, and mark lessons complete with the button at the
bottom of each page.

</div>

<div class="content-section">

## Pick Your Path

<div class="module-cards">

<div class="module-card" onclick="showLesson(&#39;1.1&#39;)">

<div class="module-card-num">

Module 1 · Weeks 1-2

</div>

### Foundations

Environment, Linux, ROOT, C++ essentials, and Git workflow at SDCC.

<div class="module-card-meta">

<span>9 lessons</span><span>Start here →</span>

</div>

</div>

<div class="module-card" onclick="showLesson(&#39;2.1&#39;)">

<div class="module-card-num">

Module 2 · Weeks 3-4

</div>

### Fun4All Framework

Node tree, SubsysReco modules, DST files, and local builds.

<div class="module-card-meta">

<span>8 lessons</span><span>Core skills</span>

</div>

</div>

<div class="module-card" onclick="showLesson(&#39;3.1&#39;)">

<div class="module-card-num">

Module 3 · Weeks 5-6

</div>

### Simulation & Reco

Geant4 sims, calorimeter clusters, tracks, truth matching, centrality.

<div class="module-card-meta">

<span>8 lessons</span><span>Analysis skills</span>

</div>

</div>

<div class="module-card" onclick="showLesson(&#39;4.1&#39;)">

<div class="module-card-num">

Module 4 · Weeks 7-8

</div>

### Production Analysis

Condor jobs, datasets, calibrations, publication plots, capstone
project.

<div class="module-card-meta">

<span>9 lessons</span><span>Real work</span>

</div>

</div>

</div>

</div>

<div class="callout info">

<div class="callout-title">

Who is this course for?

</div>

New sPHENIX graduate students, postdocs, and any physicist joining the
collaboration who needs a structured path into the software ecosystem.
No prior experience with sPHENIX is assumed, but basic familiarity with
command-line Linux and C++ helps.

</div>

<div class="callout tip">

<div class="callout-title">

What you'll build

</div>

By the end of Week 8 you will have written a complete SubsysReco
analysis module, run it against real sPHENIX data, submitted a
multi-thousand-job Condor campaign, and produced publication-ready plots
— using the same workflow professional sPHENIX analysts use today.

</div>

<div class="lesson-nav">

<span></span>

Start Module 1 →

</div>


<!-- ===== LESSON: 1.1 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 1

</div>

# What is sPHENIX?

Meet the experiment you're going to spend the next several years of your
life with.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~15 min read

</div>

<div class="meta-item">

📚 Background

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

## The Elevator Pitch

**sPHENIX** is a high-energy nuclear physics experiment at the
**Relativistic Heavy Ion Collider (RHIC)** at Brookhaven National
Laboratory. It studies the **Quark-Gluon Plasma (QGP)** — the
extraordinary state of matter that filled the universe microseconds
after the Big Bang — by smashing gold nuclei (Au+Au) and protons (p+p)
together at nearly the speed of light.

"s" stands for "super" — sPHENIX inherits the interaction region from
the PHENIX detector (which ran 2000-2016) but is effectively a brand-new
machine designed for a very different physics program.

<div class="callout info">

<div class="callout-title">

Why does QGP matter?

</div>

For about 10 microseconds after the Big Bang, the entire observable
universe was a quark-gluon plasma — so hot that protons and neutrons
couldn't form. By recreating this state in miniature at RHIC, we learn
about the *strong force* (QCD) in the most extreme conditions nature
allows.

</div>

</div>

<div class="content-section">

## The Physics Program

sPHENIX pursues four flagship measurements. You will probably work on
one of these, directly or indirectly.

| Flagship Measurement           | What It Tells Us                                                                                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Jet substructure**           | How high-energy quarks/gluons lose energy passing through QGP — this is "jet quenching," and its detailed structure probes the plasma at different length scales. |
| **Upsilon spectroscopy**       | The three bottomonium states Y(1S), Y(2S), Y(3S) "melt" at different temperatures in QGP due to color screening — a quark-gluon plasma thermometer.               |
| **Open & closed heavy flavor** | Charm and bottom quarks are produced early and traverse the QGP — their transport properties probe plasma viscosity.                                              |
| **Photons, π⁰, η**             | Neutral mesons and direct photons give us access to the full kinematic range of QGP radiative processes.                                                          |

</div>

<div class="content-section">

## The Collision Systems

sPHENIX collides three main systems, and each gives us different handles
on the physics:

  - **Au+Au at √s<sub>NN</sub> = 200 GeV** — the main QGP production
    environment
  - **p+p at √s = 200 GeV** — the "vacuum" baseline (no QGP)
  - **p+Au at √s<sub>NN</sub> = 200 GeV** — controls for cold nuclear
    matter effects

Comparing observables across systems (via the **nuclear modification
factor** R<sub>AA</sub>) is how we isolate QGP effects.

</div>

<div class="content-section">

## Where You Fit In

As a new student/postdoc, you'll pick a *physics working group* (PWG) —
typically one of: Jets, Heavy Flavor, Upsilon, or Photon/Pi0. You'll
also likely have a *technical responsibility* for a detector subsystem
(tracking, calorimetry, calibrations, software infrastructure, etc.).

Your first several months are about becoming **technically competent** —
that's what this course teaches. Physics ideas come next.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.1</span>
<span class="exercise-title">Orient yourself</span>

</div>

Before moving on:

1.  Visit the sPHENIX public webpage at `wiki.bnl.gov/sPHENIX/`
2.  Find the list of physics working groups (PWGs) — which ones meet
    weekly?
3.  Identify at least one analysis currently underway in your area of
    interest
4.  Find your collaboration mentor's name on the authorship list

</div>

<div class="complete-section">

### Ready to move on?

Mark this lesson complete and head to the detector tour.

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Detector Subsystems →

</div>


<!-- ===== LESSON: 1.2 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 2

</div>

# Detector Subsystems

A tour of the hardware, and the software modules that process each
subsystem's data.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min read

</div>

<div class="meta-item">

📚 Background

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

sPHENIX is built as concentric layers around the collision point (the
"interaction region," or IR). Particles produced in the collision fly
outward and get detected by successive subsystems. Each subsystem has a
corresponding directory in the `coresoftware` repository that handles
its data.

<div class="diagram">

oHCal ┌─────────────┐ │ iHCal │ │ ┌───────┐ │ │ │ EMCal │ │ │ │ ┌───┐ │
│ │ │ │TPC│ │ │ │ │ │ I │ │ │ ← nominal IR (beam line) │ │ │NTT│ │ │ │
│ │MVTX │ │ │ │ └───┘ │ │ │ └───────┘ │ └─────────────┘ MBD (forward)
sEPD / ZDC (far forward)

</div>

</div>

<div class="content-section">

## The Full Subsystem Reference

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th>Subsystem</th>
<th>Role</th>
<th>Software Dir</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>MVTX</strong><br />
<span class="small">Monolithic Active Pixel Sensor</span></td>
<td>Precision vertex tracking — 3 layers of silicon pixels at small radii. Resolves displaced vertices from heavy flavor.</td>
<td><code>mvtx/</code></td>
</tr>
<tr class="even">
<td><strong>INTT</strong><br />
<span class="small">Intermediate Silicon Tracker</span></td>
<td>4 layers of silicon strips bridging MVTX and TPC. Provides fast timing for pile-up rejection.</td>
<td><code>intt/</code></td>
</tr>
<tr class="odd">
<td><strong>TPC</strong><br />
<span class="small">Time Projection Chamber</span></td>
<td>The main tracker. Gas-filled volume measures charged particle trajectories and momenta. Enormous data volume.</td>
<td><code>tpc/</code>, <code>tpccalib/</code></td>
</tr>
<tr class="even">
<td><strong>EMCal</strong><br />
<span class="small">Electromagnetic Calorimeter</span></td>
<td>SPACAL (W/SciFi) calorimeter measuring photon, electron, and π⁰ energies. ~25k towers.</td>
<td><code>CaloBase/</code>, <code>CaloReco/</code></td>
</tr>
<tr class="odd">
<td><strong>iHCal</strong><br />
<span class="small">Inner Hadronic Calorimeter</span></td>
<td>Steel-scintillator sampling calorimeter inside the solenoid. Captures start of hadronic showers.</td>
<td><code>CaloBase/</code></td>
</tr>
<tr class="even">
<td><strong>oHCal</strong><br />
<span class="small">Outer Hadronic Calorimeter</span></td>
<td>Steel-scintillator sampling calorimeter outside solenoid. Absorbs hadronic shower remnants.</td>
<td><code>CaloBase/</code></td>
</tr>
<tr class="odd">
<td><strong>MBD</strong><br />
<span class="small">Minimum Bias Detector</span></td>
<td>Fast quartz Cherenkov counters at forward rapidity. Primary trigger, vertex timing, centrality.</td>
<td><code>mbd/</code></td>
</tr>
<tr class="even">
<td><strong>sEPD</strong><br />
<span class="small">South Event Plane Detector</span></td>
<td>Segmented scintillator tiles. Measures the event plane for flow analyses.</td>
<td><code>epd/</code></td>
</tr>
<tr class="odd">
<td><strong>ZDC</strong><br />
<span class="small">Zero Degree Calorimeter</span></td>
<td>Very forward neutron calorimeter. Used in centrality and luminosity.</td>
<td><code>zdcinfo/</code></td>
</tr>
</tbody>
</table>

</div>

<div class="content-section">

## The Software Stack

When you write analysis code, you sit at the top of a layered stack.
Each layer below provides services:

<div class="diagram">

Layer 4 │ YOUR CODE (SubsysReco modules, ROOT macros) │ Layer 3 │
sPHENIX Offline Software (coresoftware repo) │ calo/ tracking/ trigger/
jets/ calibrations/ │ Layer 2 │ Fun4All Framework (event processing
engine) │ node tree, module chains, I/O managers │ Layer 1 │ External
Libraries │ ROOT · Geant4 · HepMC · FastJet · Eigen · Pythia8 │ Layer 0
│ Operating System │ AlmaLinux 9 · CVMFS · GPFS · HTCondor

</div>

You'll be living in Layers 3 and 4, but you need to understand how to
touch every layer below for diagnostics and debugging.

</div>

<div class="callout tip">

<div class="callout-title">

The mental model to lock in now

</div>

A collision event is a snapshot of *everything that happened in that
single bunch crossing*, across all subsystems, read out as raw signals.
Reconstruction turns raw signals into "physics objects" — clusters,
tracks, vertices, jets — that you work with. Your analysis queries and
filters those physics objects.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.2</span>
<span class="exercise-title">Find each subsystem's code</span>

</div>

Open
[github.com/sPHENIX-Collaboration/coresoftware](https://github.com/sPHENIX-Collaboration/coresoftware)
in a browser and:

1.  Navigate to the `offline/packages/` directory
2.  Locate the directories for MVTX, INTT, TPC, EMCal, and MBD
3.  For each, open one `.h` header file and skim it — notice how every
    class inherits from a base or uses PHObject
4.  Bookmark these directories; you'll return here often

</div>

<div class="complete-section">

### Got the lay of the land?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Getting Connected →

</div>


<!-- ===== LESSON: 1.3 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 3

</div>

# Getting Connected to SDCC

Your accounts, your filesystem, and your first successful SSH.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min hands-on

</div>

<div class="meta-item">

🔧 Setup

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

## Where You Work

sPHENIX computing runs on the **Scientific Data and Computing Center
(SDCC)** at Brookhaven National Laboratory. You reach it via SSH from a
Linux, macOS, or Windows machine. There is no meaningful local-only
workflow — you do everything on the cluster.

<div class="callout info">

<div class="callout-title">

Before you start, you need

</div>

  - A BNL guest number (if not already a BNL employee)
  - An SDCC account — request via
    `sdcc.bnl.gov/information/policy/account/`
  - A RHIC/sPHENIX account linked to the collaboration
  - SSH keys registered with SDCC (strongly preferred over passwords)

Your advisor or sPHENIX sponsor should help with the paperwork. It can
take a few weeks, so start early.

</div>

</div>

<div class="content-section">

## The SSH Gateway Chain

SDCC uses a two-step SSH: first into a gateway, then into an interactive
work node.

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">ssh
\<username\>@ssh.sdcc.bnl.gov</span> <span class="terminal-output">\#
After gateway auth:</span> <span class="terminal-prompt">\[ssh
\~\]$</span> <span class="terminal-cmd">ssh
sphnxuser.sdcc.bnl.gov</span> <span class="terminal-output">\# Now
you're on a sPHENIX interactive node. This is home.</span>
<span class="terminal-prompt">\[sphnxuser \~\]$</span>
<span class="terminal-cmd">hostname</span>
<span class="terminal-output">sphnx02.sdcc.bnl.gov</span>

</div>

A typical `~/.ssh/config` on your local machine to make this one step:

<div class="code-block">

<div class="code-header">

<span class="code-lang">ssh config</span>

Copy

</div>

    Host bnl-gw
      HostName ssh.sdcc.bnl.gov
      User yourname
      ServerAliveInterval 60
    
    Host sphnx
      HostName sphnxuser.sdcc.bnl.gov
      User yourname
      ProxyJump bnl-gw
      ServerAliveInterval 60

</div>

Now you just type `ssh sphnx` and you're there.

</div>

<div class="content-section">

## The Filesystem Layout

sPHENIX at SDCC uses GPFS and Lustre. You'll use these paths constantly:

| Path                           | What lives here                                                | Backup?         |
| ------------------------------ | -------------------------------------------------------------- | --------------- |
| `/sphenix/user/$USER/`         | Your home. Code, macros, personal builds.                      | ✅ Yes           |
| `/sphenix/data/`               | Shared data storage (read-mostly).                             | ✅ Yes           |
| `/sphenix/sim/`                | Simulation DSTs, generator output.                             | Partial         |
| `/sphenix/tg/`                 | Group scratch space. Don't park important data here long-term. | ❌ No            |
| `/sphenix/lustre01/sphnxpro/`  | Production output — the real, calibrated DSTs.                 | Source-of-truth |
| `/opt/sphenix/core/`           | sPHENIX software installation root.                            | Managed         |
| `/cvmfs/sphenix.sdcc.bnl.gov/` | CVMFS read-only distribution of releases.                      | Managed         |

<div class="callout warning">

<div class="callout-title">

Quota discipline

</div>

Your home has a quota (typically 100 GB - 1 TB depending on role). ROOT
files are huge. Check with `quota -s` regularly, and move old analysis
output to `/sphenix/tg/` or delete it. You *will* hit quota within 6
months otherwise.

</div>

</div>

<div class="content-section">

## First Commands You'll Run

<div class="terminal">

<span class="terminal-prompt">$</span>
<span class="terminal-cmd">whoami</span>
<span class="terminal-output">yourname</span>
<span class="terminal-prompt">$</span>
<span class="terminal-cmd">pwd</span>
<span class="terminal-output">/sphenix/u/yourname</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
/sphenix/user/$USER/</span> <span class="terminal-output">(empty — it's
your first day)</span> <span class="terminal-prompt">$</span>
<span class="terminal-cmd">quota -s</span>
<span class="terminal-output">Disk quotas for user yourname: ...</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
/cvmfs/sphenix.sdcc.bnl.gov/</span>
<span class="terminal-output">x86\_64/ el7/ el9/ ...</span>

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.3</span>
<span class="exercise-title">First login checklist</span>

</div>

1.  Successfully SSH into a sPHENIX interactive node
2.  Run `hostname`, `whoami`, `pwd`, and `quota -s`
3.  Create your working directory: `mkdir -p
    /sphenix/user/$USER/analysis`
4.  Create an install directory: `mkdir -p /sphenix/user/$USER/install`
5.  Run `ls /sphenix/lustre01/sphnxpro/` to see production data paths

</div>

<div class="complete-section">

### Logged in and oriented?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Environment Setup →

</div>


<!-- ===== LESSON: 1.4 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 4

</div>

# Environment Setup

The single most-run command of your sPHENIX career.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min hands-on

</div>

<div class="meta-item">

🔧 Setup

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

## The One Command You Always Run

Every time you log in, the *first* thing you do is source the sPHENIX
setup script. This configures dozens of environment variables so ROOT,
Geant4, CMake, and all sPHENIX libraries can find each other.

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    source /opt/sphenix/core/bin/sphenix_setup.sh -n new

</div>

The `-n new` flag picks the **latest weekly build**. The three build
flavors:

| Flag         | What it selects                                   | When to use                                                 |
| ------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| `-n new`     | Latest nightly / weekly build                     | Active development. Follow the bleeding edge.               |
| `-n ana.NNN` | A specific "analysis" release, e.g. `ana.464`     | Reproducible analyses. Every paper uses a pinned ana build. |
| `-n pro.NNN` | A specific production build used to make the DSTs | Running official production or reprocessing.                |

<div class="callout warning">

<div class="callout-title">

Match your build to your data

</div>

If your DSTs were made with `ana.464`, analyze them with `ana.464`.
Mixing releases across the DST boundary will cause subtle ABI
mismatches, missing nodes, or silent wrong answers.

</div>

</div>

<div class="content-section">

## Adding Your Local Install

When you build your own analysis module (Module 2), you install it into
*your* directory, not the shared sPHENIX install. You need one more line
so Fun4All finds your shared libraries:

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    source /opt/sphenix/core/bin/sphenix_setup.sh -n new
    export MYINSTALL=/sphenix/user/$USER/install
    source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL

</div>

This extends `LD_LIBRARY_PATH`, `ROOT_INCLUDE_PATH`, and CMake prefixes
so both your compiler and ROOT macros see what you built.

</div>

<div class="content-section">

## Automate It: Your `.bashrc`

You don't want to type those three lines every login. Put this at the
bottom of `~/.bashrc` (or `~/.bash_profile`):

<div class="code-block">

<div class="code-header">

<span class="code-lang">\~/.bashrc</span>

Copy

</div>

    # sPHENIX environment
    if [ -f /opt/sphenix/core/bin/sphenix_setup.sh ]; then
        source /opt/sphenix/core/bin/sphenix_setup.sh -n new
        export MYINSTALL=/sphenix/user/$USER/install
        source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL
    fi
    
    # Helpful shortcuts
    alias sphx="cd /sphenix/user/\\$USER"
    alias macros="cd /sphenix/user/\\$USER/macros"

</div>

<div class="callout tip">

<div class="callout-title">

The "is this sourced?" test

</div>

Any time something breaks, first check: `echo $OFFLINE_MAIN`. If that's
empty, you forgot to source. *This is the \#1 mistake of new students.*

</div>

</div>

<div class="content-section">

## Environment Variables Worth Knowing

| Variable             | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `$OPT_SPHENIX`       | Root of the sPHENIX install tree (`/opt/sphenix/core`) |
| `$OFFLINE_MAIN`      | The release directory for your selected build          |
| `$ROOTSYS`           | Where ROOT lives — used by many build systems          |
| `$G4_MAIN`           | Geant4 installation path                               |
| `$CALIBRATIONROOT`   | Calibration database path (used at runtime)            |
| `$MYINSTALL`         | Your personal install dir (you set this)               |
| `$LD_LIBRARY_PATH`   | Where the dynamic linker looks for `.so` files         |
| `$ROOT_INCLUDE_PATH` | Where ROOT looks for C++ headers during macro JIT      |

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.4</span>
<span class="exercise-title">Validate your environment</span>

</div>

Log out, log back in, and run each of these — copy the output to a notes
file:

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    echo "OFFLINE_MAIN=$OFFLINE_MAIN"
    echo "ROOTSYS=$ROOTSYS"
    echo "MYINSTALL=$MYINSTALL"
    which root
    root --version
    which cmake
    cmake --version

</div>

All should print something meaningful. If any echo a blank line, your
setup is not active.

</div>

<div class="complete-section">

### Environment working?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Linux Essentials →

</div>


<!-- ===== LESSON: 1.5 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 5

</div>

# Linux Essentials

The command-line fluency you need to survive on the cluster.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~30 min hands-on

</div>

<div class="meta-item">

💻 Skills

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

You live in a Linux terminal now. This lesson is a focused crash-course
in the commands you'll use every day. If you already know Linux well,
skim and skip to the scripting section.

</div>

<div class="content-section">

## Navigation & File Operations

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Where am I, and what's here?
    pwd
    ls -lah
    
    # Move around
    cd /sphenix/user/$USER
    cd ..                # up one
    cd -                 # back to previous dir
    cd ~                 # home
    
    # Create, copy, move, delete
    mkdir -p Week1/{scripts,data,plots}
    cp file.cc backup.cc
    mv old.cc renamed.cc
    rm -i badfile.root   # -i = prompt before delete (safer)
    rm -rf tmp/          # recursive force — BE CAREFUL
    
    # Find
    find . -name "*.cc"              # all .cc under current dir
    find /sphenix/user/$USER -size +1G   # files > 1 GB
    find . -name "*.root" -mtime -1   # modified in last day

</div>

</div>

<div class="content-section">

## Viewing & Searching File Contents

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Quick view
    cat small.txt
    head -50 log.txt
    tail -100 log.txt
    tail -f log.txt              # follow in real time
    
    # Better pager
    less big.log                 # / to search, q to quit
    
    # Grep — you'll use this daily
    grep -r "SvtxTrackMap" .               # recursive
    grep -n "TODO" *.cc                    # show line numbers
    grep -l "pi0" *.C                      # just list matching files
    grep --include="*.h" -r "RawCluster" .   # filter by type
    
    # Count lines
    wc -l filelist.txt

</div>

<div class="callout tip">

<div class="callout-title">

Power move

</div>

Combine `grep` with `git grep` inside a repo — it's much faster because
it only searches tracked files.

</div>

</div>

<div class="content-section">

## Processes & Sessions

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # What's running?
    ps aux | grep root
    top                    # interactive; q to quit
    htop                   # nicer if available
    
    # Kill a runaway job
    kill <PID>
    kill -9 <PID>         # force kill
    
    # Run in background, persist after logout
    nohup root -l -b -q 'myAnalysis.C' > run.log 2>&1 &
    
    # Detachable session — mandatory for long runs
    tmux new -s analysis         # start named session
    # Ctrl-B then D to detach
    tmux attach -t analysis      # reconnect later
    tmux ls                      # list sessions

</div>

</div>

<div class="content-section">

## File Transfer

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Copy single file from laptop to cluster
    scp plot.pdf sphnx:/sphenix/user/$USER/
    
    # Copy whole directory
    scp -r mydir/ sphnx:/sphenix/user/$USER/
    
    # rsync — resumable, skips unchanged files, the pro's choice
    rsync -avz mydir/ sphnx:/sphenix/user/$USER/mydir/
    rsync -avz --delete src/ backup/    # mirror exactly

</div>

</div>

<div class="content-section">

## Shell Scripting: The Minimum

You'll write lots of small shell scripts. A typical script:

<div class="code-block">

<div class="code-header">

<span class="code-lang">run\_analysis.sh</span>

Copy

</div>

    #!/bin/bash
    # Exit on any error, error on undefined vars, pipe failures propagate
    set -euo pipefail
    
    # Source environment
    source /opt/sphenix/core/bin/sphenix_setup.sh -n new
    export MYINSTALL=/sphenix/user/$USER/install
    source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL
    
    # Arguments
    RUN=\${1:-48080}
    NEVENTS=\${2:-1000}
    OUTDIR=/sphenix/user/$USER/output/run\${RUN}
    mkdir -p "$OUTDIR"
    
    # Loop over inputs
    for f in /sphenix/lustre01/sphnxpro/run\${RUN}/DST_CALO_*.root; do
        echo "Processing $f"
        root -l -b -q "Fun4All_MyAnalysis.C($NEVENTS, \\"$f\\")"
    done
    
    echo "Done. Output in $OUTDIR"

</div>

Make executable and run:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">chmod
+x run\_analysis.sh</span> <span class="terminal-prompt">$</span>
<span class="terminal-cmd">./run\_analysis.sh 48080 1000</span>

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.5</span>
<span class="exercise-title">Linux skills drill</span>

</div>

1.  Create the directory tree: `mkdir -p
    ~/Week1/{scripts,data,plots,logs}`
2.  Use `grep -r` to find all files in coresoftware containing
    `SvtxTrackMap`
3.  Use `find` to list every `.root` file in your home bigger than 10 MB
4.  Write a script that loops over all `.C` files in a directory and
    prints their line count
5.  Start a tmux session, run `top` inside it, detach, reattach

</div>

<div class="complete-section">

### Comfortable on the command line?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Git & GitHub →

</div>


<!-- ===== LESSON: 1.6 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 6

</div>

# Git & GitHub Workflow

Version control is how sPHENIX ships. Learn the branch-PR flow.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min hands-on

</div>

<div class="meta-item">

💻 Skills

</div>

<div class="meta-item">

<span class="tag easy">Beginner</span>

</div>

</div>

</div>

<div class="content-section">

## The Repositories You'll Touch

All sPHENIX code is on GitHub under the `sPHENIX-Collaboration` org:

| Repo           | What's in it                                         | When you'll use it                                               |
| -------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| `coresoftware` | All reconstruction + framework code                  | Read often; contribute PRs when adding/fixing framework features |
| `macros`       | Official Fun4All macros (simulation, reconstruction) | Start every project from here                                    |
| `tutorials`    | Hello-world style examples                           | Reference while learning                                         |
| `analysis`     | Physics analysis modules (user code)                 | This is where your module will probably live                     |

<div class="callout info">

<div class="callout-title">

Fork or direct?

</div>

For the `analysis` repo, most students push branches directly and open
PRs. For `coresoftware`, you fork to your own GitHub and open PRs from
the fork. Your advisor will tell you which pattern your group uses.

</div>

</div>

<div class="content-section">

## First-Time Setup

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # One-time identity
    git config --global user.name "Your Name"
    git config --global user.email "you@example.edu"
    git config --global core.editor vim
    
    # Set up SSH key with GitHub (do this once)
    ssh-keygen -t ed25519 -C "you@example.edu"
    cat ~/.ssh/id_ed25519.pub
    # Paste that key into GitHub: Settings → SSH and GPG keys

</div>

</div>

<div class="content-section">

## Clone, Branch, Commit, Push

The canonical sPHENIX daily-work loop:

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # 1. Clone once
    cd /sphenix/user/$USER
    git clone https://github.com/sPHENIX-Collaboration/analysis.git
    cd analysis
    
    # 2. Always branch before doing work
    git checkout master
    git pull                              # get latest
    git checkout -b feature/pi0-efficiency
    
    # 3. Work. Edit files. Then:
    git status                            # what changed?
    git diff                              # see the changes
    git add MyAnalysis.cc MyAnalysis.h
    git commit -m "Add photon pair invariant mass histogram"
    
    # 4. Push the branch
    git push -u origin feature/pi0-efficiency
    
    # 5. Open a Pull Request on github.com

</div>

<div class="callout danger">

<div class="callout-title">

Never do this

</div>

Never `git commit` directly on `master` / `main`, never `git push
--force` to a shared branch, and never commit large binary files
(`.root`, `.tar.gz`). Your PR will be closed, and you'll get a gentle
email from someone senior.

</div>

</div>

<div class="content-section">

## Everyday Git Commands

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Inspect history
    git log --oneline -20
    git log --all --graph --oneline --decorate
    git show <commit-hash>
    
    # Undo local changes to a file (before commit)
    git checkout -- badfile.cc
    
    # Unstage a file (after git add, before commit)
    git reset HEAD badfile.cc
    
    # Modify the last commit (only for unshared commits!)
    git commit --amend
    
    # Sync your branch with latest master
    git checkout master
    git pull
    git checkout feature/pi0-efficiency
    git merge master          # or: git rebase master
    
    # Stash uncommitted work
    git stash
    git stash pop

</div>

</div>

<div class="content-section">

## The Pull Request Flow

1.  Push your feature branch to GitHub
2.  Go to the repo's GitHub page → "Compare & pull request"
3.  Title your PR clearly (`[CaloReco] Fix cluster chi2 cut`)
4.  Describe what changed and *why*
5.  Request reviewers from your subsystem (your advisor will suggest
    names)
6.  Respond to review comments by pushing more commits to the same
    branch
7.  Once approved and CI passes, a maintainer merges

<div class="callout tip">

<div class="callout-title">

CI matters

</div>

sPHENIX runs automated build + basic test CI on every PR. If your PR
breaks the build, fix it *before* asking for review. Nothing burns
reviewer goodwill faster than "hey can you check my PR" when CI is red.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.6</span>
<span class="exercise-title">Your first clone + branch</span>

</div>

1.  Clone `tutorials`, `macros`, and `analysis` into your home dir
2.  In `analysis`, create a branch named `sandbox/yourname`
3.  Add a README file `sandbox/yourname/NOTES.md` with a single line of
    text
4.  Commit it with a descriptive message
5.  Push the branch (but don't open a PR yet — just confirm push
    succeeded)
6.  Run `git log --oneline -5` and verify your commit is at the top

</div>

<div class="complete-section">

### Git flow locked in?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: ROOT Fundamentals →

</div>


<!-- ===== LESSON: 1.7 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 7

</div>

# ROOT Fundamentals

ROOT is the backbone of every HEP analysis. You must be fluent.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~40 min hands-on

</div>

<div class="meta-item">

💻 Skills

</div>

<div class="meta-item">

<span class="tag medium">Intermediate</span>

</div>

</div>

</div>

<div class="content-section">

**ROOT** is CERN's data analysis framework: a C++ interpreter,
histogramming library, fitting engine, plotting toolkit, and persistent
columnar data format all in one. Every sPHENIX DST is a ROOT file. Every
plot in every sPHENIX paper is a ROOT plot. You will use ROOT every
single day.

</div>

<div class="content-section">

## Three Ways to Run ROOT

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Interactive
    root -l                          # no splash
    
    # Run a macro interactively
    root -l myAnalysis.C
    
    # Batch mode — no graphics, auto-quit. Use in scripts & Condor.
    root -l -b -q myAnalysis.C
    
    # With arguments (note the single quotes around the function call)
    root -l -b -q 'myAnalysis.C(1000, "file.root")'

</div>

</div>

<div class="content-section">

## The Classes You'll Use 90% of the Time

| Class                     | What it does                                              |
| ------------------------- | --------------------------------------------------------- |
| `TFile`                   | Read / write ROOT files                                   |
| `TTree`                   | Columnar "ntuple" — rows of events, columns of variables  |
| `TH1F` / `TH1D`           | 1D histogram (float / double). Your most-used class.      |
| `TH2F` / `TH2D`           | 2D histograms                                             |
| `TProfile`                | Mean of Y vs X in bins of X (useful for resolution plots) |
| `TCanvas`                 | A drawing surface                                         |
| `TLorentzVector`          | 4-momentum, invariant mass arithmetic                     |
| `TF1`                     | 1D function for fitting                                   |
| `TLegend`                 | Plot legends                                              |
| `TGraph` / `TGraphErrors` | x-y data points with optional error bars                  |

</div>

<div class="content-section">

## Exercise 1: Make a Histogram

Open ROOT interactively and type these in one at a time:

<div class="code-block">

<div class="code-header">

<span class="code-lang">ROOT C++ (.C macro)</span>

Copy

</div>

    TH1F *h = new TH1F("h", "Gaussian;x;counts", 100, -5, 5);
    for (int i = 0; i < 100000; ++i)
        h->Fill(gRandom->Gaus(0, 1));
    h->Draw();

</div>

You should see a Gaussian histogram pop up. Now fit it:

<div class="code-block">

<div class="code-header">

<span class="code-lang">ROOT C++</span>

Copy

</div>

    h->Fit("gaus");
    gPad->SaveAs("gauss.pdf");

</div>

</div>

<div class="content-section">

## Exercise 2: Read a TTree

TTrees are how sPHENIX stores per-event ntuples. Reading one looks like
this:

<div class="code-block">

<div class="code-header">

<span class="code-lang">readTree.C</span>

Copy

</div>

    void readTree() {
        TFile *f = new TFile("data.root");
        TTree *t = (TTree*)f->Get("ntp_cluster");
    
        // Quick inspection:
        t->Print();              // list all branches
        t->GetEntries();        // how many rows?
    
        // Quick plots via TTree::Draw()
        t->Draw("e");                              // energy distribution
        t->Draw("e", "pt>1.0 && abs(eta)<1.1"); // with a cut
        t->Draw("eta:phi", "", "colz");          // 2D heatmap
    }

</div>

<div class="callout tip">

<div class="callout-title">

The `TTree::Draw` syntax is weird but powerful

</div>

`tree->Draw("y:x", "cut", "option")` — that colon is NOT division. "y:x"
means "plot y on vertical, x on horizontal." `"colz"` is the option for
a 2D color map.

</div>

</div>

<div class="content-section">

## Exercise 3: A Real Analysis Macro

This is the pattern you'll copy-paste a hundred times. Memorize it.

<div class="code-block">

<div class="code-header">

<span class="code-lang">ptAnalysis.C</span>

Copy

</div>

    void ptAnalysis(const char* infile = "data.root") {
    
        // --- 1. Open file, grab tree ---
        TFile *f = new TFile(infile);
        TTree *t = (TTree*)f->Get("ntp_cluster");
    
        // --- 2. Hook variables to branches ---
        float e, pt, eta, phi;
        t->SetBranchAddress("e",   &e);
        t->SetBranchAddress("pt",  &pt);
        t->SetBranchAddress("eta", &eta);
        t->SetBranchAddress("phi", &phi);
    
        // --- 3. Book histograms ---
        TH1F *hpt = new TH1F("hpt",
            ";p_{T} [GeV/c];dN/dp_{T}", 100, 0, 10);
    
        // --- 4. Loop ---
        Long64_t n = t->GetEntries();
        for (Long64_t i = 0; i < n; ++i) {
            t->GetEntry(i);
            if (std::abs(eta) > 1.1) continue;
            if (e < 0.3) continue;
            hpt->Fill(pt);
        }
    
        // --- 5. Draw & save ---
        TCanvas *c = new TCanvas("c", "", 800, 600);
        c->SetLogy();
        hpt->Draw();
        c->SaveAs("pt_distribution.pdf");
    }

</div>

</div>

<div class="content-section">

## TLorentzVector — The Secret Weapon

4-momentum arithmetic is ubiquitous in HEP. `TLorentzVector` handles it:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    TLorentzVector p1, p2;
    p1.SetPtEtaPhiM(pt1, eta1, phi1, 0.0);   // photon: m=0
    p2.SetPtEtaPhiE(pt2, eta2, phi2, e2);     // or by energy
    
    TLorentzVector pair = p1 + p2;
    double mass = pair.M();                      // invariant mass
    double pair_pt = pair.Pt();
    double angle = p1.Angle(p2.Vect());        // opening angle

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.7</span>
<span class="exercise-title">Your first ROOT macro</span>

</div>

1.  Write a macro `myFirst.C` that fills three histograms: a Gaussian
    with mean=5 σ=1, a Gaussian with mean=5 σ=2, and a uniform
    distribution on \[0,10\]
2.  Draw all three on the same canvas with different colors
3.  Add a `TLegend` labeling each
4.  Save the plot to PDF
5.  Bonus: write a second macro that opens the PDF-producing macro's
    saved file and re-fits the Gaussians

</div>

<div class="complete-section">

### ROOT feeling comfortable?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: C++ for sPHENIX →

</div>


<!-- ===== LESSON: 1.8 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Lesson 8

</div>

# C++ for sPHENIX

The C++ subset you must be fluent in to read and write analysis code.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~30 min read

</div>

<div class="meta-item">

💻 Skills

</div>

<div class="meta-item">

<span class="tag medium">Intermediate</span>

</div>

</div>

</div>

<div class="content-section">

sPHENIX is \~1M lines of C++. You don't need to master templates or
metaprogramming — but you do need working knowledge of the language
features used everywhere in our codebase.

</div>

<div class="content-section">

## Classes and Inheritance

Every sPHENIX analysis module *is a class* that inherits from
`SubsysReco`. You need to be comfortable with:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    // Base class
    class SubsysReco {
     public:
        virtual int Init(PHCompositeNode* topNode) { return 0; }
        virtual int process_event(PHCompositeNode* topNode) = 0;  // pure
        virtual int End(PHCompositeNode* topNode) { return 0; }
    };
    
    // Derived (your code)
    class MyAnalysis : public SubsysReco {
     public:
        MyAnalysis(const std::string& name = "MyAnalysis");
        int Init(PHCompositeNode* topNode) override;
        int process_event(PHCompositeNode* topNode) override;
        int End(PHCompositeNode* topNode) override;
     private:
        std::string m_outname;
        TFile* m_file = nullptr;
    };

</div>

**Key ideas:** `virtual` and `override` enable polymorphism. Fun4All
holds a pointer to `SubsysReco*`, but when it calls `process_event`,
*your* override actually runs.

</div>

<div class="content-section">

## Pointers & References

ROOT and sPHENIX are *pointer-heavy*. You MUST be comfortable with:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    SvtxTrack* track = iter->second;         // raw pointer
    if (!track) return;                        // always null-check!
    float px = track->get_px();                 // -> dereferences
    
    // Reference version (non-null, doesn't rebind)
    auto& tmap = *trackmap;
    for (auto& [key, ptr] : tmap) { /* ... */ }

</div>

<div class="callout danger">

<div class="callout-title">

Null pointers are the \#1 cause of segfaults

</div>

Any time you call `findNode::getClass<T>`, *always* check the return. If
the node isn't there, you get `nullptr` and dereferencing it is an
instant segfault.

</div>

</div>

<div class="content-section">

## STL Containers You'll See Daily

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    // vector — ordered, indexable, the workhorse
    std::vector<TLorentzVector> photons;
    photons.push_back(p);
    photons.size();
    
    // map — key-value lookup (your track map is one!)
    std::map<int, SvtxTrack*> tracks;
    for (auto& [id, track] : tracks) {
        std::cout << "Track " << id << " pt=" << track->get_pt() << std::endl;
    }
    
    // pair — for returning two things, or as map value_type
    std::pair<float, float> eta_phi(0.5, 1.2);
    
    // set — unique, sorted
    std::set<int> run_numbers;
    
    // deque — double-ended queue (for event-mixing buffers)
    std::deque<std::vector<TLorentzVector>> event_buffer;

</div>

</div>

<div class="content-section">

## Header vs Source Split

Every class you write lives in TWO files:

  - `MyAnalysis.h` — declarations only (what the class *is*)
  - `MyAnalysis.cc` — definitions (what the methods *do*)

<div class="code-block">

<div class="code-header">

<span class="code-lang">MyAnalysis.h</span>

Copy

</div>

    #ifndef MYANALYSIS_H
    #define MYANALYSIS_H
    
    #include <fun4all/SubsysReco.h>
    #include <string>
    
    // Forward declarations — avoid pulling in huge headers
    class PHCompositeNode;
    class TFile;
    class TH1F;
    
    class MyAnalysis : public SubsysReco {
     public:
        MyAnalysis(const std::string& name = "MyAnalysis");
        int Init(PHCompositeNode*) override;
        int process_event(PHCompositeNode*) override;
        int End(PHCompositeNode*) override;
     private:
        TFile* m_file = nullptr;
        TH1F* m_hpt = nullptr;
    };
    #endif

</div>

The `#ifndef / #define / #endif` "include guard" prevents
double-inclusion. Forward-declaring `TFile`, `TH1F`, `PHCompositeNode`
instead of including their headers speeds up compilation enormously — a
best practice you'll see throughout coresoftware.

</div>

<div class="content-section">

## Modern C++ Bits You'll See

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++17</span>

Copy

</div>

    // auto — type deduction
    auto trackmap = findNode::getClass<SvtxTrackMap>(topNode, "SvtxTrackMap");
    
    // Range-based for — cleaner iteration
    for (auto& [id, track] : *trackmap) {
        // ...
    }
    
    // nullptr — use this, not NULL or 0
    TFile* f = nullptr;
    
    // Uniform initialization
    std::vector<int> v{1, 2, 3};
    
    // const correctness — mark anything that shouldn't change
    void print(const std::string& s);

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 1.8</span>
<span class="exercise-title">C++ self-assessment</span>

</div>

Write a small standalone C++ program (not a ROOT macro) that:

1.  Defines a class `Particle` with private members `pt, eta, phi` and
    getters
2.  Creates a `std::vector<Particle>` of 10 random particles
3.  Loops over them, prints pt for those with |eta|\<1
4.  Compiles with `g++ -std=c++17 -o test test.cc` and runs

If you can do this smoothly in under 20 minutes, your C++ is ready for
sPHENIX. If not, spend another day on
[learncpp.com](https://learncpp.com).

</div>

<div class="complete-section">

### C++ fluency check passed?

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Module 1 Quiz →

</div>


<!-- ===== LESSON: 1.q ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 1 · Foundations · Checkpoint

</div>

# Module 1 Checkpoint Quiz

Seven questions. Miss more than two? Revisit the weak lessons before
moving on.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~10 min

</div>

<div class="meta-item">

✅ Assessment

</div>

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

1\. Which collider does sPHENIX live on, and what are its flagship
collision species?

</div>

  - <span class="quiz-letter">A</span>LHC · p+p and Pb+Pb
  - <span class="quiz-letter">B</span>RHIC · Au+Au and p+p at 200 GeV
  - <span class="quiz-letter">C</span>Tevatron · p+pbar
  - <span class="quiz-letter">D</span>NSLS-II · electrons

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

2\. What command sets up the sPHENIX environment to use the latest
weekly build?

</div>

  - <span class="quiz-letter">A</span>`sphenix --load latest`
  - <span class="quiz-letter">B</span>`source
    /opt/sphenix/core/bin/sphenix_setup.sh -n new`
  - <span class="quiz-letter">C</span>`source setup.sh`
  - <span class="quiz-letter">D</span>`module load sphenix/new`

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

3\. You just ran ROOT and it complains that it can't find
libMyAnalysis.so. First thing you check?

</div>

  - <span class="quiz-letter">A</span>Rebuild the library
  - <span class="quiz-letter">B</span>`echo $MYINSTALL` — confirm the
    environment is sourced
  - <span class="quiz-letter">C</span>Try a different ROOT version
  - <span class="quiz-letter">D</span>Reboot the cluster

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

4\. Which directory holds official production DSTs?

</div>

  - <span class="quiz-letter">A</span>`/sphenix/user/$USER/`
  - <span class="quiz-letter">B</span>`$G4_MAIN/`
  - <span class="quiz-letter">C</span>`/sphenix/lustre01/sphnxpro/`
  - <span class="quiz-letter">D</span>`/sphenix/tg/`

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

5\. What is the correct sPHENIX git workflow?

</div>

  - <span class="quiz-letter">A</span>Commit everything on master, push
    when done
  - <span class="quiz-letter">B</span>Create a feature branch, commit,
    push, open a PR, respond to review
  - <span class="quiz-letter">C</span>Email patches to maintainers
  - <span class="quiz-letter">D</span>Force-push often to keep history
    clean

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

6\. Which ROOT class represents a 4-momentum vector with invariant-mass
arithmetic?

</div>

  - <span class="quiz-letter">A</span>`TVector3`
  - <span class="quiz-letter">B</span>`TLorentzVector`
  - <span class="quiz-letter">C</span>`TMatrix`
  - <span class="quiz-letter">D</span>`T4Vector`

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

7\. Why use forward declarations (`class TFile;`) in header files
instead of `#include <TFile.h>`?

</div>

  - <span class="quiz-letter">A</span>It makes the code run faster
  - <span class="quiz-letter">B</span>It reduces compilation time and
    header dependencies
  - <span class="quiz-letter">C</span>It avoids needing ROOT installed
  - <span class="quiz-letter">D</span>It is required by C++ syntax

<div class="quiz-feedback">

</div>

</div>

<div class="complete-section">

### Module 1 complete?

Mark the checkpoint done and move into the heart of sPHENIX software.

Mark Complete → Module 2

</div>

<div class="lesson-nav">

← Back

Module 2: Fun4All →

</div>


<!-- ===== LESSON: 2.1 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 1

</div>

# What is Fun4All?

The event processing engine that every sPHENIX job runs on.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min read

</div>

<div class="meta-item">

🏗 Framework

</div>

<div class="meta-item">

<span class="tag medium">Core concept</span>

</div>

</div>

</div>

<div class="content-section">

## The Big Idea

**Fun4All** is the event-processing framework for sPHENIX. Everything —
simulation, digitization, reconstruction, calibration, your analysis —
is a *chain of modules* running inside Fun4All. Once you internalize
this pattern, the whole codebase starts to make sense.

Fun4All gives you three things:

1.  An **event loop** that reads input events, pumps them through a
    chain, and writes output
2.  A shared in-memory data container called the **node tree** that
    modules write to and read from
3.  A plug-in architecture: **SubsysReco** modules that you register
    with a Fun4AllServer

</div>

<div class="content-section">

## The Event Loop, Illustrated

<div class="diagram">

┌─────────────────────────────────────────────────────────────┐ │
Fun4AllServer::run(N) │ │ │ │ for i in 1..N: │ │ │ │ ┌────────┐
┌──────────┐ ┌──────────┐ ┌────────┐ │ │ │Input
│──▶│Module A │──▶│Module B │─▶│Output │ │ │ │Manager │
│process\_ │ │process\_ │ │Manager │ │ │ └────────┘ │event() │
│event() │ └────────┘ │ │ └──────────┘ └──────────┘ │ │ │ │ │ │ ▼ ▼ │
│ ┌──────────────────────────┐ │ │ │ N O D E T R E E │ (shared mem) │
│ └──────────────────────────┘ │ │ │
└─────────────────────────────────────────────────────────────┘

</div>

Each module reads from / writes to the node tree. Modules are completely
decoupled — they don't know about each other. They only agree on *what's
on the tree*.

</div>

<div class="content-section">

## The Key Classes

| Class                  | Role                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `Fun4AllServer`        | The singleton conductor. Owns the event loop, module chain, input/output managers.                                 |
| `SubsysReco`           | Base class for every analysis/reconstruction module. You write these.                                              |
| `Fun4AllInputManager`  | Reads input events. Common subclasses: `Fun4AllDstInputManager` (ROOT DSTs), `Fun4AllPrdfInputManager` (raw data). |
| `Fun4AllOutputManager` | Writes output DSTs.                                                                                                |
| `PHCompositeNode`      | Directory-like node that holds other nodes. The "topNode" is the root of the node tree.                            |
| `PHIODataNode<T>`      | Leaf node that holds an object of type T.                                                                          |

</div>

<div class="content-section">

## The Four Lifecycle Methods

Every SubsysReco you write overrides some of these. Fun4All calls them
at well-defined times:

| Method                   | Called when          | Typical use                                     |
| ------------------------ | -------------------- | ----------------------------------------------- |
| `Init(topNode)`          | Once, at start       | Open output file, book histograms, create TTree |
| `InitRun(topNode)`       | At start of each run | Load calibrations, geometry for the run         |
| `process_event(topNode)` | Once per event       | Your actual analysis code                       |
| `End(topNode)`           | Once, at end         | Write histograms, close output file             |

<div class="callout tip">

<div class="callout-title">

Return codes matter

</div>

Every lifecycle method returns an int. Use these constants:

  - `Fun4AllReturnCodes::EVENT_OK` — continue normally
  - `Fun4AllReturnCodes::ABORTEVENT` — skip this event, go to next
  - `Fun4AllReturnCodes::ABORTRUN` — stop the whole job (serious errors)
  - `Fun4AllReturnCodes::DISCARDEVENT` — exclude from output DST

</div>

</div>

<div class="content-section">

## A Minimal Running Job

You'll see this skeleton repeated everywhere:

<div class="code-block">

<div class="code-header">

<span class="code-lang">Fun4All macro (C++)</span>

Copy

</div>

    // 1. Get the server
    Fun4AllServer* se = Fun4AllServer::instance();
    
    // 2. Register modules (order matters — they run in this order)
    se->registerSubsystem(new MyAnalysis());
    
    // 3. Register input
    Fun4AllInputManager* in = new Fun4AllDstInputManager("in");
    in->fileopen("DST_CALO_run2pp_ana464_...");
    se->registerInputManager(in);
    
    // 4. Go!
    se->run(1000);    // process 1000 events (0 = all)
    se->End();        // calls End() on every module

</div>

</div>

<div class="callout info">

<div class="callout-title">

Why "Fun4All"?

</div>

The framework inherits from a PHENIX-era codebase. The name is whimsical
historical baggage — there's no deep meaning. But the design is serious:
a node-tree-based pipeline is both easy to reason about and trivially
parallel across events.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.1</span>
<span class="exercise-title">Read the source</span>

</div>

1.  In coresoftware on GitHub, find
    `offline/framework/fun4all/Fun4AllServer.h`
2.  Identify where `run()` and `registerSubsystem()` are declared
3.  Find the implementation of `run()` in `Fun4AllServer.cc` and skim
    how it iterates events
4.  Locate `SubsysReco.h` and identify the four virtual lifecycle
    methods

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: The Node Tree →

</div>


<!-- ===== LESSON: 2.2 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 2

</div>

# The Node Tree Architecture

The in-memory bulletin board every module reads and writes.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min read

</div>

<div class="meta-item">

🏗 Framework

</div>

<div class="meta-item">

<span class="tag medium">Core concept</span>

</div>

</div>

</div>

<div class="content-section">

## What is the Node Tree?

The **node tree** is a hierarchical in-memory data structure. Think of
it as a shared filesystem where modules drop off "files" (physics
objects) for other modules to pick up. Every module gets a pointer to
the root (`topNode`) at each lifecycle call.

<div class="diagram">

TopNode │ ├── DST ← the event-by-event data │ ├── CLUSTER\_CEMC
RawClusterContainer (EMCal clusters) │ ├── CLUSTER\_HCALIN
RawClusterContainer │ ├── CLUSTER\_HCALOUT RawClusterContainer │ ├──
TOWER\_CALIB\_CEMC TowerInfoContainer (calibrated towers) │ ├──
TRKR\_CLUSTER TrkrClusterContainer (tracking hits) │ ├── SvtxTrackMap
Reconstructed tracks │ ├── SvtxVertexMap Reconstructed vertices │ ├──
GlobalVertexMap Combined vertex │ ├── CentralityInfo Event centrality │
├── AntiKt\_Tower\_r04 Jets │ └── G4TruthInfo MC truth (sim only) │
├── RUN ← run-level info (geometry, calibrations) │ ├──
TowerInfoGeomv1 Geometry of towers │ └── ActsGeometry Tracking geometry
│ └── PAR ← parameters └── ParameterMap

</div>

</div>

<div class="content-section">

## Reading From the Tree

The canonical pattern — *memorize this*:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <phool/getClass.h>
    #include <trackbase_historic/SvtxTrackMap.h>
    
    int MyAnalysis::process_event(PHCompositeNode* topNode) {
        SvtxTrackMap* trackmap =
            findNode::getClass<SvtxTrackMap>(topNode, "SvtxTrackMap");
    
        if (!trackmap) {
            std::cout << PHWHERE << "SvtxTrackMap not found!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    
        for (auto& [key, track] : *trackmap) {
            float pt = std::hypot(track->get_px(), track->get_py());
            // ...
        }
        return Fun4AllReturnCodes::EVENT_OK;
    }

</div>

**Decoded:**

  - `findNode::getClass<T>(topNode, "NodeName")` — search the tree for a
    node named `"NodeName"` that holds an object of type `T`. Returns
    `nullptr` if not found.
  - `PHWHERE` is a macro that expands to `"file:line:"` for clean error
    messages.
  - Always null-check. Always.

</div>

<div class="content-section">

## Writing to the Tree

Rarely needed in analysis code, but essential when your module
*produces* something for downstream modules (e.g., a jet finder creating
a `JetContainer`):

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <phool/PHCompositeNode.h>
    #include <phool/PHNodeIterator.h>
    #include <phool/PHIODataNode.h>
    
    int MyProducer::Init(PHCompositeNode* topNode) {
        PHNodeIterator iter(topNode);
        PHCompositeNode* dstNode =
            dynamic_cast<PHCompositeNode*>(iter.findFirst("PHCompositeNode", "DST"));
    
        MyContainer* container = new MyContainer();
        PHIODataNode<PHObject>* node =
            new PHIODataNode<PHObject>(container, "MyContainer", "PHObject");
        dstNode->addNode(node);
    
        return Fun4AllReturnCodes::EVENT_OK;
    }

</div>

</div>

<div class="content-section">

## The Most Common Node Names (Reference)

| Node Name             | Class                    | Contains                |
| --------------------- | ------------------------ | ----------------------- |
| `CLUSTER_CEMC`        | `RawClusterContainer`    | EMCal clusters          |
| `CLUSTER_HCALIN`      | `RawClusterContainer`    | Inner HCal clusters     |
| `CLUSTER_HCALOUT`     | `RawClusterContainer`    | Outer HCal clusters     |
| `TOWER_CALIB_CEMC`    | `TowerInfoContainer`     | Calibrated EMCal towers |
| `TOWER_CALIB_HCALIN`  | `TowerInfoContainer`     | Calibrated iHCal towers |
| `TOWER_CALIB_HCALOUT` | `TowerInfoContainer`     | Calibrated oHCal towers |
| `SvtxTrackMap`        | `SvtxTrackMap`           | Reconstructed tracks    |
| `SvtxVertexMap`       | `SvtxVertexMap`          | Reconstructed vertices  |
| `GlobalVertexMap`     | `GlobalVertexMap`        | Combined event vertex   |
| `G4TruthInfo`         | `PHG4TruthInfoContainer` | MC truth (sim only)     |
| `AntiKt_Tower_r04`    | `JetContainer`           | Anti-kT R=0.4 jets      |
| `CentralityInfo`      | `CentralityInfo`         | Centrality              |
| `TRKR_CLUSTER`        | `TrkrClusterContainer`   | Raw tracking clusters   |
| `MbdOut`              | `MbdOut`                 | MBD detector output     |

</div>

<div class="callout tip">

<div class="callout-title">

Printing the tree

</div>

When a node isn't where you expect, add `topNode->print()` to your
`Init()`. It dumps the entire tree structure. Indispensable for
debugging.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.2</span>
<span class="exercise-title">Tree exploration</span>

</div>

1.  Write a tiny SubsysReco (or even a ROOT macro loading a DST) that in
    `Init()` calls `topNode->print()`
2.  Run it against a DST\_CALO file and capture the output
3.  Run it against a DST\_TRACKS file and compare
4.  Notice which nodes appear only in one vs. both

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: SubsysReco Lifecycle →

</div>


<!-- ===== LESSON: 2.3 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 3

</div>

# SubsysReco Lifecycle

Understand Init → InitRun → process\_event → End before you write any
code.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min read

</div>

<div class="meta-item">

🏗 Framework

</div>

<div class="meta-item">

<span class="tag medium">Core concept</span>

</div>

</div>

</div>

<div class="content-section">

## The State Machine

<div class="diagram">

┌──────────┐ │ Init() │ Once per job. Open output. Book histos.
└────┬─────┘ │ ▼ ┌──────────┐ │InitRun() │ Once per
run. Load calibs. Fetch geometry. └────┬─────┘ │ ▼
┌────────────────────┐ │
process\_event() │ ◀──┐ │ ... your code ... │ │ N times (one per event)
└─────────┬──────────┘ │ │ │ └───────────────┘ │ (new run? → InitRun)
▼ ┌──────────┐ │ End() │ Once at job end. Write ROOT file. Close.
└──────────┘

</div>

</div>

<div class="content-section">

## Where Each Thing Lives

| You should do this...                                                 | ...in this method |
| --------------------------------------------------------------------- | ----------------- |
| Open output TFile, new histograms, new TTree, SetBranchAddress        | `Init`            |
| Fetch geometry pointers, load calibrations, set up detector constants | `InitRun`         |
| findNode on DST nodes, loop over physics objects, fill histos         | `process_event`   |
| Reset per-event counters if needed                                    | `ResetEvent`      |
| Write TTree, TFile::Close                                             | `End`             |

<div class="callout warning">

<div class="callout-title">

Common mistake

</div>

Don't book histograms in `process_event`. You'll create new ones every
event and leak memory. *Always* book in `Init`, fill in `process_event`,
write in `End`.

</div>

</div>

<div class="content-section">

## The Header Skeleton (Copy-Paste Starting Point)

<div class="code-block">

<div class="code-header">

<span class="code-lang">MyAnalysis.h</span>

Copy

</div>

    #ifndef MYANALYSIS_H
    #define MYANALYSIS_H
    
    #include <fun4all/SubsysReco.h>
    #include <string>
    
    class PHCompositeNode;
    class TFile;
    class TH1F;
    class TTree;
    
    class MyAnalysis : public SubsysReco {
     public:
        MyAnalysis(const std::string& name = "MyAnalysis");
        ~MyAnalysis() override;
    
        int Init(PHCompositeNode*) override;
        int InitRun(PHCompositeNode*) override;
        int process_event(PHCompositeNode*) override;
        int ResetEvent(PHCompositeNode*) override;
        int End(PHCompositeNode*) override;
    
        void set_output_file(const std::string& f) { m_outfile_name = f; }
    
     private:
        std::string m_outfile_name = "output.root";
        TFile* m_outfile = nullptr;
        TH1F*  m_hpt = nullptr;
        TTree* m_tree = nullptr;
    
        // Tree branch variables
        float m_tree_pt = 0, m_tree_eta = 0, m_tree_phi = 0, m_tree_e = 0;
    
        int m_nevt = 0;
    };
    
    #endif

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.3</span>
<span class="exercise-title">Sketch your first module</span>

</div>

On paper (or in a text editor), write — from memory — a SubsysReco
header for a module called `PhotonAnalysis` that will:

  - Book one histogram of photon pT
  - Book a TTree with pt, eta, phi branches
  - Write everything out to a configurable ROOT file

Compare yours to the skeleton above. Diff differences and understand
each.

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Building Your Module →

</div>


<!-- ===== LESSON: 2.4 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 4

</div>

# Building Your First Module

A complete, compilable SubsysReco from scratch.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~45 min hands-on

</div>

<div class="meta-item">

🏗 Framework

</div>

<div class="meta-item">

<span class="tag hard">Capstone of Module 2</span>

</div>

</div>

</div>

<div class="content-section">

Now we put together what you learned. This lesson walks through a
*complete, working* MyAnalysis module — header, source, and
implementation notes. You'll build and run this at the end of Module 2.

</div>

<div class="content-section">

## The Full Source File

<div class="code-block">

<div class="code-header">

<span class="code-lang">MyAnalysis.cc</span>

Copy

</div>

    #include "MyAnalysis.h"
    
    #include <fun4all/Fun4AllReturnCodes.h>
    #include <phool/PHCompositeNode.h>
    #include <phool/getClass.h>
    
    #include <trackbase_historic/SvtxTrack.h>
    #include <trackbase_historic/SvtxTrackMap.h>
    #include <globalvertex/GlobalVertex.h>
    #include <globalvertex/GlobalVertexMap.h>
    
    #include <TFile.h>
    #include <TH1F.h>
    #include <TTree.h>
    
    #include <cmath>
    #include <iostream>
    
    MyAnalysis::MyAnalysis(const std::string& name)
      : SubsysReco(name) {}
    
    MyAnalysis::~MyAnalysis() {}
    
    int MyAnalysis::Init(PHCompositeNode*) {
        m_outfile = new TFile(m_outfile_name.c_str(), "RECREATE");
    
        m_hpt = new TH1F("h_pt",
            "Track p_{T};p_{T} [GeV/c];Counts", 200, 0, 20);
    
        m_tree = new TTree("T", "Analysis Tree");
        m_tree->Branch("pt",  &m_tree_pt,  "pt/F");
        m_tree->Branch("eta", &m_tree_eta, "eta/F");
        m_tree->Branch("phi", &m_tree_phi, "phi/F");
        m_tree->Branch("e",   &m_tree_e,   "e/F");
    
        return Fun4AllReturnCodes::EVENT_OK;
    }
    
    int MyAnalysis::InitRun(PHCompositeNode*) {
        return Fun4AllReturnCodes::EVENT_OK;
    }
    
    int MyAnalysis::process_event(PHCompositeNode* topNode) {
        ++m_nevt;
    
        SvtxTrackMap* trackmap =
            findNode::getClass<SvtxTrackMap>(topNode, "SvtxTrackMap");
    
        if (!trackmap) {
            std::cout << "MyAnalysis: SvtxTrackMap not found, skipping event "
                      << m_nevt << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    
        for (auto& [key, track] : *trackmap) {
            if (!track) continue;
    
            float px = track->get_px();
            float py = track->get_py();
            float pz = track->get_pz();
            float pt = std::hypot(px, py);
            float p  = std::sqrt(px*px + py*py + pz*pz);
            float eta = 0.5f * std::log((p + pz) / (p - pz));
            float phi = std::atan2(py, px);
    
            // Quality cuts
            if (track->get_quality() > 10) continue;
            if (pt < 0.2f) continue;
    
            m_hpt->Fill(pt);
    
            m_tree_pt  = pt;
            m_tree_eta = eta;
            m_tree_phi = phi;
            m_tree_e   = p;
            m_tree->Fill();
        }
    
        return Fun4AllReturnCodes::EVENT_OK;
    }
    
    int MyAnalysis::ResetEvent(PHCompositeNode*) {
        return Fun4AllReturnCodes::EVENT_OK;
    }
    
    int MyAnalysis::End(PHCompositeNode*) {
        std::cout << "MyAnalysis: processed " << m_nevt << " events" << std::endl;
        m_outfile->cd();
        m_hpt->Write();
        m_tree->Write();
        m_outfile->Close();
        return Fun4AllReturnCodes::EVENT_OK;
    }

</div>

</div>

<div class="content-section">

## Line-by-Line, the Important Parts

#### Init()

Open the file with `"RECREATE"` so it's overwritten each run. Book
histograms and the tree *here*, once. `SetBranchAddress` equivalents
live in `Branch()` — the address of your local member variable.

#### process\_event()

`findNode::getClass<T>` is the one call you'll write a thousand times.
Always check for `nullptr`. Inside the track loop we compute derived
quantities (eta, phi from Cartesian momenta) and fill.

#### End()

*Very important:* do `m_outfile->cd()` before `Write()` so ROOT
associates the histogram with the correct file. Then `Close()`.

</div>

<div class="callout warning">

<div class="callout-title">

cd() before Write()

</div>

If you don't do `m_outfile->cd()` in End(), your histograms may be
written to whatever file ROOT last made "current" — which could be
nothing, and your output file will be empty. This bug bites 100% of new
students.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.4</span>
<span class="exercise-title">Build the module</span>

</div>

1.  Create a directory: `mkdir -p ~/analysis/MyAnalysis`
2.  Create `MyAnalysis.h` (from Lesson 2.3) and `MyAnalysis.cc` (above)
3.  Don't build yet — we'll do CMake in Lesson 2.6
4.  Read the code one more time, this time with a print-out of the
    SubsysReco header open next to it

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Fun4All Macro →

</div>


<!-- ===== LESSON: 2.5 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 5

</div>

# Fun4All Macro Walkthrough

The ROOT macro that wires your module into a running job.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min hands-on

</div>

<div class="meta-item">

🏗 Framework

</div>

</div>

</div>

<div class="content-section">

You wrote a SubsysReco class. Now you need a *Fun4All macro* — a ROOT .C
file that instantiates the Fun4AllServer, registers modules and I/O
managers, and calls `run()`.

</div>

<div class="content-section">

## The Canonical Macro

<div class="code-block">

<div class="code-header">

<span class="code-lang">Fun4All\_MyAnalysis.C</span>

Copy

</div>

    #include <fun4all/Fun4AllServer.h>
    #include <fun4all/Fun4AllInputManager.h>
    #include <fun4all/Fun4AllDstInputManager.h>
    #include <myanalysis/MyAnalysis.h>
    
    R__LOAD_LIBRARY(libfun4all.so)
    R__LOAD_LIBRARY(libMyAnalysis.so)
    
    void Fun4All_MyAnalysis(
        const int nEvents = 0,
        const std::string& inputFile =
            "DST_TRACKS_run2pp_ana464_2024p012-00048080-0000.root") {
    
        // 1. Get the server (singleton)
        Fun4AllServer* se = Fun4AllServer::instance();
        se->Verbosity(0);
    
        // 2. Register your module
        MyAnalysis* m = new MyAnalysis("MyAnalysis");
        m->set_output_file("my_output.root");
        se->registerSubsystem(m);
    
        // 3. Input manager
        Fun4AllInputManager* in = new Fun4AllDstInputManager("DST_in");
        in->fileopen(inputFile);
        se->registerInputManager(in);
    
        // 4. Run
        se->run(nEvents);     // 0 = all events
    
        // 5. Clean up
        se->End();
        delete se;
    
        std::cout << "All done!" << std::endl;
        gSystem->Exit(0);
    }

</div>

</div>

<div class="content-section">

## Anatomy of the Macro

### R\_\_LOAD\_LIBRARY

This ROOT directive tells the interpreter to load your shared library
*before* JIT-compiling the macro. Without
`R__LOAD_LIBRARY(libMyAnalysis.so)`, ROOT will report `undefined
reference to 'MyAnalysis'`.

### Fun4AllServer::instance()

It's a singleton. Call `instance()` wherever you need it; you always get
the same server back. Don't `new` it yourself.

### Verbosity

0 = quiet, 1+ = more chatter. Useful while debugging; drop to 0 for
batch jobs.

### Order of registerSubsystem calls matters

Modules run in the order you register them. If module B needs something
module A produces on the node tree, register A first.

### Input manager types

  - `Fun4AllDstInputManager` — reads standard DST ROOT files
  - `Fun4AllPrdfInputManager` — reads raw PRDF files from the DAQ
  - `Fun4AllHepMCInputManager` — reads HepMC event-generator output

### Using a file list

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    Fun4AllInputManager* in = new Fun4AllDstInputManager("DST_in");
    in->AddListFile("filelist.txt");   // one DST path per line
    se->registerInputManager(in);

</div>

</div>

<div class="content-section">

## Running It

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l -b -q Fun4All\_MyAnalysis.C</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l -b -q 'Fun4All\_MyAnalysis.C(1000)'</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l -b -q 'Fun4All\_MyAnalysis.C(0,
"DST\_TRACKS\_00048080-0000.root")'</span>

</div>

<div class="callout tip">

<div class="callout-title">

Quoting arguments

</div>

When passing string arguments to a macro, the single-quotes around
`'file(args)'` tell the shell not to expand anything, and the inner
double-quotes are what ROOT parses. Get this wrong and you'll see
cryptic "unexpected token" errors.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.5</span>
<span class="exercise-title">Write your Fun4All macro</span>

</div>

1.  Save the canonical macro above as `Fun4All_MyAnalysis.C` in your
    analysis dir
2.  For now the `R__LOAD_LIBRARY(libMyAnalysis.so)` line will fail — we
    haven't built yet
3.  Instead, replace it temporarily with
    `gSystem->Load("libfun4all.so")` and try to run on a real DST with
    an empty module list to confirm Fun4All itself starts

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: CMake & Local Install →

</div>


<!-- ===== LESSON: 2.6 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 6

</div>

# CMake & Local Install

Build your module into a .so that Fun4All picks up.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~30 min hands-on

</div>

<div class="meta-item">

🏗 Framework

</div>

<div class="meta-item">

<span class="tag medium">Build system</span>

</div>

</div>

</div>

<div class="content-section">

## Directory Layout

<div class="diagram">

\~/analysis/MyAnalysis/ ├── CMakeLists.txt ├── MyAnalysis.h ├──
MyAnalysis.cc └── build/ ← created by cmake, gitignored

</div>

</div>

<div class="content-section">

## CMakeLists.txt

<div class="code-block">

<div class="code-header">

<span class="code-lang">CMakeLists.txt</span>

Copy

</div>

    cmake_minimum_required(VERSION 3.0)
    project(MyAnalysis CXX)
    
    # Standard sPHENIX CMake setup
    find_package(sPHENIX REQUIRED)
    
    include_directories(\${PROJECT_SOURCE_DIR}/)
    
    # Build a shared library
    add_library(MyAnalysis SHARED
        MyAnalysis.cc
    )
    
    # Link against sPHENIX framework and data classes
    target_link_libraries(MyAnalysis
        fun4all
        phool
        SubsysReco
        trackbase_historic
        calo_io
        globalvertex_io
    )
    
    # Install to $MYINSTALL
    install(TARGETS MyAnalysis
            DESTINATION \${CMAKE_INSTALL_PREFIX}/lib)
    install(FILES MyAnalysis.h
            DESTINATION \${CMAKE_INSTALL_PREFIX}/include/myanalysis)

</div>

</div>

<div class="content-section">

## Building

<div class="code-block">

<div class="code-header">

<span class="code-lang">bash</span>

Copy

</div>

    # Make sure env is set up
    source /opt/sphenix/core/bin/sphenix_setup.sh -n new
    export MYINSTALL=/sphenix/user/$USER/install
    source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL
    
    cd ~/analysis/MyAnalysis
    mkdir -p build && cd build
    
    cmake .. -DCMAKE_INSTALL_PREFIX=$MYINSTALL
    make -j4
    make install

</div>

After a successful build+install:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
$MYINSTALL/lib/libMyAnalysis\*</span>
<span class="terminal-output">libMyAnalysis.so</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
$MYINSTALL/include/myanalysis/</span>
<span class="terminal-output">MyAnalysis.h</span>

</div>

</div>

<div class="content-section">

## A Convenience Script

Saving a rebuild script saves your fingers:

<div class="code-block">

<div class="code-header">

<span class="code-lang">rebuild.sh</span>

Copy

</div>

    #!/bin/bash
    set -euo pipefail
    cd "$(dirname "$0")"
    rm -rf build
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$MYINSTALL
    make -j4 install
    echo "Done. Library in $MYINSTALL/lib/"

</div>

</div>

<div class="content-section">

## Running the Complete Job

Now, back in your macro directory:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l -b -q 'Fun4All\_MyAnalysis.C(100,
"/sphenix/lustre01/sphnxpro/.../DST\_TRACKS\_...root")'</span>
<span class="terminal-output">MyAnalysis: processed 100 events</span>
<span class="terminal-output">All done\!</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
-lah my\_output.root</span> <span class="terminal-output">-rw-r--r-- 1
you group 1.2M Apr 19 14:23 my\_output.root</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l my\_output.root</span> <span class="terminal-output">root \[0\]
TBrowser b;</span>

</div>

<div class="callout tip">

<div class="callout-title">

You just did it

</div>

You wrote a SubsysReco from scratch, built it with CMake, and ran it
against real sPHENIX data. That's the core loop. Everything else in this
course is variations and extensions.

</div>

</div>

<div class="content-section">

## Common Build Errors

| Error                                              | Likely cause                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `Could not find sPHENIX`                           | Forgot to source `sphenix_setup.sh` before running cmake                 |
| `undefined reference to 'findNode'`                | Missing `phool` in `target_link_libraries`                               |
| `'SvtxTrack' has incomplete type`                  | Forgot `#include <trackbase_historic/SvtxTrack.h>` in .cc                |
| `libMyAnalysis.so: cannot open shared object file` | `$MYINSTALL` not on `LD_LIBRARY_PATH` — did you source `setup_local.sh`? |

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.6</span>
<span class="exercise-title">Full build & run</span>

</div>

1.  Build `libMyAnalysis.so` and install to `$MYINSTALL`
2.  Run the Fun4All macro on any DST\_TRACKS file with 100 events
3.  Open `my_output.root` and plot `h_pt`
4.  Now change a line in `MyAnalysis.cc` (e.g. loosen the quality cut),
    rebuild, rerun, and re-open — confirm you see a different
    distribution

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: DST File Format →

</div>


<!-- ===== LESSON: 2.7 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Fun4All Framework · Lesson 7

</div>

# DST File Format

Decode DST filenames and know which to use when.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~15 min read

</div>

<div class="meta-item">

🏗 Framework

</div>

</div>

</div>

<div class="content-section">

## What's a DST?

A **Data Summary Tape** — the name is historical, tape drives are gone —
is a ROOT file that stores a serialized snapshot of the node tree for
each event. When you read a DST, Fun4All reconstructs the node tree in
memory for each event in the file.

</div>

<div class="content-section">

## Decoding a Filename

<div class="diagram">

DST\_CALO\_run2pp\_ana464\_2024p012-00048080-0000.root │ │ │ │ │ │ │ │ │
│ │ │ │ └─ Segment number (0000, 0001, ...) │ │ │ │ │ └─ Run number
(zero-padded) │ │ │ │ └─ Production tag (year + production round) │ │ │
└─ Analysis build version (ana464 = ana.464) │ │ └─ Collision system
(run2pp = Run 2 p+p) │ └─ Content type └─ "DST" literal

</div>

</div>

<div class="content-section">

## DST Types You'll See Most

| Prefix              | Contents                                                | When you want it                        |
| ------------------- | ------------------------------------------------------- | --------------------------------------- |
| `DST_CALO_*`        | Calorimeter towers + clusters (EMCal, iHCal, oHCal)     | Photon/pi0/jet/neutral-meson analyses   |
| `DST_TRACKS_*`      | Tracker clusters + reconstructed tracks + vertices      | Charged-particle analyses, heavy flavor |
| `DST_GLOBAL_*`      | Global event info: centrality, MBD, vertex, event plane | Flow, centrality-dependent anything     |
| `DST_JET_*`         | Reconstructed jets (after jet finder runs)              | Jet analyses                            |
| `DST_TRUTH_*`       | MC truth record (from Geant4)                           | Simulation / efficiency studies only    |
| `DST_TRKR_HITSET_*` | Raw tracking hits (large)                               | Tracking experts only                   |

<div class="callout tip">

<div class="callout-title">

Chaining DSTs

</div>

You can register multiple input managers to combine DST streams — e.g.,
CALO + TRACKS + GLOBAL in a single job. Fun4All will synchronize them
event-by-event.

</div>

</div>

<div class="content-section">

## Chaining Multiple DSTs

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++ (in Fun4All macro)</span>

Copy

</div>

    Fun4AllDstInputManager* in1 = new Fun4AllDstInputManager("CALO");
    in1->fileopen(calo_file);
    se->registerInputManager(in1);
    
    Fun4AllDstInputManager* in2 = new Fun4AllDstInputManager("TRACKS");
    in2->fileopen(tracks_file);
    se->registerInputManager(in2);
    
    Fun4AllDstInputManager* in3 = new Fun4AllDstInputManager("GLOBAL");
    in3->fileopen(global_file);
    se->registerInputManager(in3);
    
    // Fun4All matches by run/segment/event number automatically

</div>

</div>

<div class="content-section">

## Inspecting a DST

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l my\_dst.root</span> <span class="terminal-output">root \[0\]</span>
<span class="terminal-cmd">T-\>Print()</span>
<span class="terminal-output">\# list branches (T is the event
tree)</span> <span class="terminal-output">root \[1\]</span>
<span class="terminal-cmd">T-\>GetEntries()</span>
<span class="terminal-output">\# number of events</span>
<span class="terminal-output">root \[2\]</span>
<span class="terminal-cmd">TBrowser b;</span>
<span class="terminal-output">\# interactive browser</span>

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 2.7</span>
<span class="exercise-title">DST archaeology</span>

</div>

1.  List DST files in a recent production run: `ls
    /sphenix/lustre01/sphnxpro/physics/slurp/calophysics/ | head`
2.  Pick one CALO and one TRACKS DST from the same run+segment
3.  Open each in ROOT and run `T->Print()`
4.  Identify which node classes each DST provides

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Module 2 Quiz →

</div>


<!-- ===== LESSON: 2.q ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 2 · Checkpoint

</div>

# Module 2 Checkpoint Quiz

Can you build the Fun4All mental model without peeking?

</div>

<div class="quiz-card">

<div class="quiz-question">

1\. A SubsysReco module communicates with other modules primarily
through:

</div>

  - <span class="quiz-letter">A</span>Direct C++ pointers between
    modules
  - <span class="quiz-letter">B</span>The node tree (PHCompositeNode)
  - <span class="quiz-letter">C</span>A global dictionary
  - <span class="quiz-letter">D</span>RPC calls

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

2\. Where do you book histograms in a SubsysReco module?

</div>

  - <span class="quiz-letter">A</span>`process_event()`
  - <span class="quiz-letter">B</span>`Init()`
  - <span class="quiz-letter">C</span>`End()`
  - <span class="quiz-letter">D</span>The constructor

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

3\. You call `findNode::getClass<SvtxTrackMap>(topNode, "SvtxTrackMap")`
and get back nullptr. You should:

</div>

  - <span class="quiz-letter">A</span>Dereference it anyway and see what
    happens
  - <span class="quiz-letter">B</span>Return `ABORTEVENT` and log which
    node was missing
  - <span class="quiz-letter">C</span>Return `ABORTRUN`
  - <span class="quiz-letter">D</span>Continue silently

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

4\. Which file contains EMCal clusters?

</div>

  - <span class="quiz-letter">A</span>`DST_TRACKS_*`
  - <span class="quiz-letter">B</span>`DST_CALO_*`
  - <span class="quiz-letter">C</span>`DST_TRUTH_*`
  - <span class="quiz-letter">D</span>`DST_GLOBAL_*`

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

5\. What does `R__LOAD_LIBRARY(libMyAnalysis.so)` do?

</div>

  - <span class="quiz-letter">A</span>Compiles your library
  - <span class="quiz-letter">B</span>Tells ROOT to load the shared
    library at macro load time
  - <span class="quiz-letter">C</span>Pulls the library from GitHub
  - <span class="quiz-letter">D</span>Runs the library after the macro

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

6\. Before calling `hist->Write()` in `End()`, you should:

</div>

  - <span class="quiz-letter">A</span>Normalize it
  - <span class="quiz-letter">B</span>Call `m_outfile->cd()`
  - <span class="quiz-letter">C</span>Call `hist->Freeze()`
  - <span class="quiz-letter">D</span>Call `hist->Reset()`

<div class="quiz-feedback">

</div>

</div>

<div class="complete-section">

Mark Complete → Module 3

</div>

<div class="lesson-nav">

← Back

Module 3: Simulation & Reco →

</div>


<!-- ===== LESSON: 3.1 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 1

</div>

# Running Geant4 Simulations

From an empty event to a fully reconstructed DST in one macro.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~30 min hands-on

</div>

<div class="meta-item">

🧪 Simulation

</div>

<div class="meta-item">

<span class="tag medium">Intermediate</span>

</div>

</div>

</div>

<div class="content-section">

## The Simulation Chain

sPHENIX uses **Geant4** for detector simulation. The full chain that
turns a physics event into reconstructed objects:

<div class="diagram">

┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
Generator │→ │ Geant4 │→ │ Digitization │→ │Reconstruction│ │ │ │ │ │ │
│ │ │ Pythia8 / │ │ Geometry, │ │ Energy → │ │ Cluster │ │ HIJING / │
│ materials, │ │ ADC counts, │ │ finding, │ │ Single part. │ │ B
field, │ │ noise, │ │ tracking, │ │ │ │ showers │ │ thresholds │ │ jets
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
│ ▼ ┌──────────────┐ │ DST\_\*.root │ └──────────────┘

</div>

</div>

<div class="content-section">

## The Standard Macro

The official simulation macro is `Fun4All_G4_sPHENIX.C` in the `macros`
repository:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">cd
/sphenix/user/$USER</span> <span class="terminal-prompt">$</span>
<span class="terminal-cmd">git clone
https://github.com/sPHENIX-Collaboration/macros.git</span>
<span class="terminal-prompt">$</span> <span class="terminal-cmd">cd
macros/detectors/sPHENIX/</span> <span class="terminal-prompt">$</span>
<span class="terminal-cmd">ls Fun4All\_G4\*.C</span>
<span class="terminal-output">Fun4All\_G4\_sPHENIX.C Fun4All\_G4\_Calo.C
...</span>

</div>

Run a tiny simulation:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">root
-l -b -q 'Fun4All\_G4\_sPHENIX.C(100)'</span>

</div>

This generates 100 events through the full sPHENIX detector geometry.
Output DSTs land in the current directory.

<div class="callout warning">

<div class="callout-title">

Geant4 is slow

</div>

Full sim with magnetic field is \~minutes per Au+Au event on a single
core. For development, prefer 10-100 events. For real physics studies,
run on Condor (Module 4).

</div>

</div>

<div class="content-section">

## What's Inside `Fun4All_G4_sPHENIX.C`?

It's a long macro that pulls in many helper macros. The skeleton:

<div class="code-block">

<div class="code-header">

<span class="code-lang">simplified excerpt</span>

Copy

</div>

    // Detector + reco settings (G4Setup_sPHENIX.C controls these)
    Enable::MVTX = true;
    Enable::INTT = true;
    Enable::TPC = true;
    Enable::EMCAL = true;
    Enable::HCALIN = true;
    Enable::HCALOUT = true;
    
    // Reco settings
    Enable::TRACKING = true;
    Enable::CALORECO = true;
    Enable::JETS = true;
    
    // Generator
    Input::PYTHIA8 = true;     // or HIJING, or single particle
    
    // Output
    Enable::DSTOUT = true;
    DstOut::OutputDir = ".";

</div>

</div>

<div class="content-section">

## Configuring Generators

The macro supports several event generators:

| Generator           | What it produces                          |
| ------------------- | ----------------------------------------- |
| Pythia8             | p+p hard QCD events                       |
| HIJING              | Au+Au heavy-ion events                    |
| Single particle gun | One known particle for efficiency studies |
| HepMC input         | Pre-generated events from another tool    |

</div>

<div class="content-section">

## Output DSTs

A successful run produces a family of DSTs:

<div class="terminal">

<span class="terminal-prompt">$</span> <span class="terminal-cmd">ls
\*.root</span> <span class="terminal-output">G4sPHENIX.root</span>
<span class="terminal-output">DST\_TRUTH\_g4hits.root</span>
<span class="terminal-output">DST\_CALO\_G4Hits.root</span>
<span class="terminal-output">DST\_TRACKS.root</span>

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.1</span>
<span class="exercise-title">Your first simulation</span>

</div>

1.  Clone `macros` and cd to `detectors/sPHENIX/`
2.  Run `Fun4All_G4_sPHENIX.C(10)` — just 10 events
3.  List the output DSTs
4.  Open the truth DST in ROOT and find the `G4TruthInfo` branch

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Single-Particle Embedding →

</div>


<!-- ===== LESSON: 3.2 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 2

</div>

# Single-Particle Embedding

Inject known particles into the detector to measure efficiency and
resolution.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min hands-on

</div>

<div class="meta-item">

🧪 Simulation

</div>

</div>

</div>

<div class="content-section">

## Why Single-Particle?

To measure your detector's **efficiency** (fraction of true particles
you reconstruct) and **resolution** (smearing in pT, eta, energy), you
need controlled inputs: shoot exactly one known particle into the
detector and see what you get back.

</div>

<div class="content-section">

## The Particle Gun

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++ in a Fun4All macro</span>

Copy

</div>

    #include <g4main/PHG4SimpleEventGenerator.h>
    
    PHG4SimpleEventGenerator* gen = new PHG4SimpleEventGenerator();
    gen->add_particles("pi0", 1);                // 1 pi0 per event
    
    // Production vertex (gaussian smear typical)
    gen->set_vertex_distribution_function(
        PHG4SimpleEventGenerator::Gaus,
        PHG4SimpleEventGenerator::Gaus,
        PHG4SimpleEventGenerator::Gaus);
    gen->set_vertex_distribution_mean(0, 0, 0);
    gen->set_vertex_distribution_width(0, 0, 10);   // 10 cm in z
    
    // Kinematic ranges
    gen->set_eta_range(-1.1, 1.1);
    gen->set_phi_range(-M_PI, M_PI);
    gen->set_pt_range(1.0, 20.0);                   // flat in pT
    
    se->registerSubsystem(gen);

</div>

</div>

<div class="content-section">

## Common Particle Codes

| String                    | What it is           |
| ------------------------- | -------------------- |
| `"e-"`                    | electron             |
| `"pi+"`, `"pi-"`, `"pi0"` | pions                |
| `"gamma"`                 | photon               |
| `"proton"`                | proton               |
| `"D0"`                    | D-meson              |
| `"Upsilon(1S)"`           | Upsilon ground state |

</div>

<div class="content-section">

## Embedding Into Background

For more realistic studies, you embed a single particle into a HIJING
Au+Au event so the reconstruction sees realistic occupancy:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    // HIJING for background
    PHPythia8* pythia = new PHPythia8();
    // (configuration omitted)
    
    // Single-particle "signal" at fixed eta/pt
    PHG4SimpleEventGenerator* gen = new PHG4SimpleEventGenerator();
    gen->add_particles("D0", 1);
    gen->set_pt_range(5.0, 5.0);
    gen->set_eta_range(-0.5, 0.5);
    
    se->registerSubsystem(pythia);
    se->registerSubsystem(gen);

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.2</span>
<span class="exercise-title">Photon gun</span>

</div>

1.  Modify `Fun4All_G4_sPHENIX.C` (or use a single-particle macro from
    `macros/`) to shoot 100 single photons with pT 1-20 GeV, |η|\<1.1
2.  Run the simulation
3.  Open the output DST and confirm the truth DST contains exactly one
    photon per event

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Calorimeter Clusters →

</div>


<!-- ===== LESSON: 3.3 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 3

</div>

# Calorimeter Cluster Analysis

Photons, electrons, and showers — read them off the node tree.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~30 min hands-on

</div>

<div class="meta-item">

📊 Analysis

</div>

<div class="meta-item">

<span class="tag medium">Intermediate</span>

</div>

</div>

</div>

<div class="content-section">

## From Towers to Clusters

The EMCal is divided into \~25,000 calorimeter **towers** (η-φ cells).
When a photon hits, its electromagnetic shower deposits energy in
several adjacent towers. The **cluster** is the reconstructed object
that sums up nearby towers and gives you a single (E, η, φ) for the
photon.

</div>

<div class="content-section">

## Reading EMCal Clusters

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <calobase/RawCluster.h>
    #include <calobase/RawClusterContainer.h>
    
    int MyAnalysis::process_event(PHCompositeNode* topNode) {
        RawClusterContainer* clusters =
            findNode::getClass<RawClusterContainer>(topNode, "CLUSTER_CEMC");
        if (!clusters) return Fun4AllReturnCodes::ABORTEVENT;
    
        for (auto& [key, cluster] : clusters->getClustersMap()) {
            float e    = cluster->get_energy();
            float eta  = cluster->get_eta();
            float phi  = cluster->get_phi();
            float chi2 = cluster->get_chi2();      // shower-shape goodness
            float pt   = e / std::cosh(eta);            // for massless approx
    
            // Standard quality cuts
            if (e < 0.3f) continue;     // minimum energy
            if (chi2 > 4.0f) continue;   // shower-shape compatible with EM shower
    
            m_h_cluster_e->Fill(e);
            m_h_cluster_eta_phi->Fill(eta, phi);
        }
        return Fun4AllReturnCodes::EVENT_OK;
    }

</div>

</div>

<div class="content-section">

## Cluster Quality: chi2 and Shower Shape

The `chi2` on a cluster is a measure of how well the energy distribution
across the towers matches the expected shape of an electromagnetic
shower. A real photon has chi2 ≈ 1; a hadronic shower or noise leaks to
higher chi2.

Typical cuts:

  - **chi2 \< 4** — clean photon-like clusters (photon/π⁰ analysis)
  - **energy \> 0.3 GeV** — above threshold
  - **tower count ≥ 2** — reject single-tower spikes

</div>

<div class="content-section">

## Tower-Level Access (When You Need It)

Sometimes you want raw tower energies — for noise studies, calibrations,
or custom clustering:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <calobase/TowerInfo.h>
    #include <calobase/TowerInfoContainer.h>
    
    TowerInfoContainer* towers =
        findNode::getClass<TowerInfoContainer>(topNode, "TOWER_CALIB_CEMC");
    int ntowers = towers->size();
    for (int i = 0; i < ntowers; ++i) {
        TowerInfo* tw = towers->get_tower_at_channel(i);
        float energy = tw->get_energy();
        if (!tw->get_isGood()) continue;     // skip masked/bad towers
    }

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.3</span>
<span class="exercise-title">Cluster spectrum</span>

</div>

Extend your `MyAnalysis`:

1.  Add a histogram `m_h_cluster_e` for cluster energy
2.  Add a 2D histogram `m_h_cluster_etaphi`
3.  Loop over `CLUSTER_CEMC` with the standard cuts
4.  Build, run on a CALO DST, and plot the energy spectrum

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Pi0 / Eta Reconstruction →

</div>


<!-- ===== LESSON: 3.4 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 4

</div>

# Pi0 & Eta Reconstruction

A flagship analysis pattern: pair photons, look for invariant-mass
peaks.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~40 min hands-on

</div>

<div class="meta-item">

📊 Analysis

</div>

<div class="meta-item">

<span class="tag medium">Classic technique</span>

</div>

</div>

</div>

<div class="content-section">

## The Physics

The neutral pion and eta meson decay almost exclusively to two photons:

  - π⁰ → γγ (BR ≈ 99%, m = 135 MeV)
  - η → γγ (BR ≈ 39%, m = 548 MeV)

To reconstruct them, take every pair of photon clusters in an event and
compute the invariant mass. Real π⁰s pile up at 135 MeV; everything else
is combinatorial background.

</div>

<div class="content-section">

## The Analysis Code

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++ inside process\_event</span>

Copy

</div>

    #include <TLorentzVector.h>
    #include <vector>
    
    // 1. Get clusters
    RawClusterContainer* clusters =
        findNode::getClass<RawClusterContainer>(topNode, "CLUSTER_CEMC");
    if (!clusters) return Fun4AllReturnCodes::ABORTEVENT;
    
    // 2. Build photon candidate list
    std::vector<TLorentzVector> photons;
    for (auto& [key, c] : clusters->getClustersMap()) {
        if (c->get_energy() < 0.3f) continue;
        if (c->get_chi2()  > 4.0f) continue;
    
        float eta = c->get_eta();
        float phi = c->get_phi();
        float e   = c->get_energy();
        float pt  = e / std::cosh(eta);
    
        TLorentzVector p;
        p.SetPtEtaPhiE(pt, eta, phi, e);   // massless photon
        photons.push_back(p);
    }
    
    // 3. Pair every two photons
    for (size_t i = 0; i < photons.size(); ++i) {
        for (size_t j = i + 1; j < photons.size(); ++j) {
            TLorentzVector pair = photons[i] + photons[j];
            float mass    = pair.M();
            float pair_pt = pair.Pt();
            float asym    = std::abs(photons[i].E() - photons[j].E())
                           / (photons[i].E() + photons[j].E());
    
            // Asymmetry cut suppresses background from 2-cluster fakes
            if (asym > 0.7f) continue;
    
            m_h_invmass->Fill(mass);
    
            if (mass > 0.10f && mass < 0.17f) m_h_pi0_pt->Fill(pair_pt);
            if (mass > 0.50f && mass < 0.60f) m_h_eta_pt->Fill(pair_pt);
        }
    }

</div>

</div>

<div class="content-section">

## What You Should See

Plot `m_h_invmass` on a log-y scale. You'll see a smooth combinatorial
background with two clear peaks:

  - A tall, narrow peak at **\~0.135 GeV** (π⁰ → γγ)
  - A smaller peak at **\~0.548 GeV** (η → γγ)

Fit each with a Gaussian + polynomial background to extract yields.

</div>

<div class="callout tip">

<div class="callout-title">

The asymmetry cut

</div>

Combinatorial pairs tend to be asymmetric (one high-E + one low-E
cluster). Real π⁰ decays have a flat asymmetry distribution, so cutting
`|E1-E2|/(E1+E2) < 0.7` kills background while keeping most signal.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.4</span>
<span class="exercise-title">See your first π⁰</span>

</div>

1.  Add the pi0 logic to `MyAnalysis` with histograms `h_invmass`,
    `h_pi0_pt`, `h_eta_pt`
2.  Build and run on at least 1000 events of a CALO DST
3.  Plot `h_invmass` on a log-y axis. Confirm you see the π⁰ peak.
4.  Bonus: fit the peak with `TF1` "gaus(0)+pol2(3)" and extract σ and
    signal yield

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Track Quality Cuts →

</div>


<!-- ===== LESSON: 3.5 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 5

</div>

# Track Quality Cuts

Standard selections every charged-particle analysis needs.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min hands-on

</div>

<div class="meta-item">

📊 Analysis

</div>

</div>

</div>

<div class="content-section">

## Why Track Cuts?

Not every reconstructed track is a real charged particle from the
primary vertex. You'll see:

  - **Ghosts** — pattern-recognition artifacts
  - **Secondaries** — particles from in-flight decays or interactions in
    detector material
  - **Pile-up tracks** — particles from a different bunch crossing
  - **Loopers** — low-pT tracks that curl in the magnetic field

Track quality cuts remove these.

</div>

<div class="content-section">

## The Standard Cut Set

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <trackbase_historic/SvtxTrack.h>
    #include <trackbase/TrkrDefs.h>
    
    for (auto& [key, track] : *trackmap) {
    
        // --- 1. Track-level quality (chi2/ndf-like) ---
        if (track->get_quality() > 10) continue;
    
        // --- 2. Hit counts per detector ---
        int nmvtx = 0, nintt = 0, ntpc = 0;
        for (auto it = track->begin_cluster_keys();
                  it != track->end_cluster_keys(); ++it) {
            auto det = TrkrDefs::getTrkrId(*it);
            if      (det == TrkrDefs::mvtxId) ++nmvtx;
            else if (det == TrkrDefs::inttId) ++nintt;
            else if (det == TrkrDefs::tpcId)  ++ntpc;
        }
        if (nmvtx < 2) continue;          // vertex pointing
        if (ntpc  < 20) continue;         // good momentum measurement
    
        // --- 3. Kinematic ---
        float px = track->get_px();
        float py = track->get_py();
        float pt = std::hypot(px, py);
        if (pt < 0.2f || pt > 50.0f) continue;
    
        // --- 4. Distance of Closest Approach (DCA) to primary vertex ---
        // (skipped here — needs vertex; see truth/vertex lesson)
    
        // Track passes — fill plots
        m_h_track_pt->Fill(pt);
    }

</div>

</div>

<div class="content-section">

## Why Each Cut?

| Cut                | Removes                                                                     |
| ------------------ | --------------------------------------------------------------------------- |
| `quality < 10`     | Tracks with poor fit chi2 — ghosts and badly seeded tracks                  |
| `nmvtx ≥ 2`        | Tracks not pointing back to the IR — secondaries from material interactions |
| `ntpc ≥ 20`        | Short tracks with bad momentum resolution                                   |
| `pt > 0.2 GeV`     | Loopers; below this momentum is harder to separate from background          |
| `DCA < some value` | Tracks not from the primary vertex (pile-up, secondaries)                   |

</div>

<div class="callout warning">

<div class="callout-title">

Cuts evolve

</div>

The exact thresholds drift as the tracker calibration improves. *Always*
check the latest recommendations from the Tracking PWG before publishing
— don't assume the cuts you copied from a friend's macro are still
current.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.5</span>
<span class="exercise-title">Cut flow</span>

</div>

1.  Make a histogram with bins labeled "all tracks", "after quality",
    "after MVTX", "after TPC", "after pT"
2.  Increment each bin in sequence after each cut
3.  Run on 100 events
4.  The "cut flow" tells you which cuts kill what fraction — useful for
    tuning

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Truth Matching →

</div>


<!-- ===== LESSON: 3.6 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 6

</div>

# Truth Matching (Monte Carlo)

Compare what you reconstructed to what was generated.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~25 min hands-on

</div>

<div class="meta-item">

🧪 Simulation

</div>

</div>

</div>

<div class="content-section">

## Why Truth Matching?

In simulation, you know the "truth": which photons were generated,
where, and with what 4-momentum. Comparing reconstructed objects to
truth lets you measure:

  - **Efficiency** — fraction of true particles you successfully
    reconstruct
  - **Fake rate** — fraction of reco objects with no truth match
  - **Resolution** — width of (reco - truth) distributions
  - **Bias** — mean offset of (reco - truth)

</div>

<div class="content-section">

## Reading Truth Particles

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <g4main/PHG4TruthInfoContainer.h>
    #include <g4main/PHG4Particle.h>
    #include <g4main/PHG4VtxPoint.h>
    
    PHG4TruthInfoContainer* truth =
        findNode::getClass<PHG4TruthInfoContainer>(topNode, "G4TruthInfo");
    if (!truth) return Fun4AllReturnCodes::ABORTEVENT;
    
    auto range = truth->GetPrimaryParticleRange();
    for (auto it = range.first; it != range.second; ++it) {
        PHG4Particle* p = it->second;
    
        int   pid = p->get_pid();           // PDG code
        float px = p->get_px();
        float py = p->get_py();
        float pz = p->get_pz();
        float e  = p->get_e();
        float pt = std::hypot(px, py);
        float eta = 0.5f * std::log((std::hypot(pt,pz)+pz)/(std::hypot(pt,pz)-pz));
    
        if (pid == 22)  m_h_truth_photon_pt->Fill(pt);   // photon
        if (pid == 111) m_h_truth_pi0_pt->Fill(pt);      // pi0
        if (pid == 221) m_h_truth_eta_pt->Fill(pt);      // eta
    }

</div>

</div>

<div class="content-section">

## Common PDG Codes

| Particle   | PDG Code     | Particle       | PDG Code     |
| ---------- | ------------ | -------------- | ------------ |
| γ (photon) | 22           | e⁻ / e⁺        | 11 / -11     |
| π⁻         | \-211        | π⁺             | 211          |
| π⁰         | 111          | η              | 221          |
| K⁺ / K⁻    | 321 / -321   | K⁰<sub>S</sub> | 310          |
| p / p̄     | 2212 / -2212 | n / n̄         | 2112 / -2112 |
| D⁰         | 421          | B⁰             | 511          |
| J/ψ        | 443          | Υ(1S)          | 553          |

</div>

<div class="content-section">

## Matching Reco to Truth (ΔR)

The simplest matching: for every reco object, find the closest truth
particle in (η, φ) space:

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    auto dR = [](float eta1, float phi1, float eta2, float phi2) {
        float dphi = std::remainder(phi1 - phi2, 2*M_PI);
        return std::hypot(eta1-eta2, dphi);
    };
    
    // For each reco cluster, find best truth photon
    for (auto& reco : reco_photons) {
        float bestDR = 1e9;
        PHG4Particle* match = nullptr;
        for (auto& tp : truth_photons) {
            float d = dR(reco.Eta(), reco.Phi(), tp.eta, tp.phi);
            if (d < bestDR) { bestDR = d; match = tp.particle; }
        }
        if (bestDR < 0.05f) {
            // matched! fill resolution histos
            m_h_dE->Fill((reco.E() - match->get_e()) / match->get_e());
        }
    }

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.6</span>
<span class="exercise-title">Photon energy resolution</span>

</div>

1.  Run a single-photon simulation (Lesson 3.2) of 1000 events at fixed
    pT=5 GeV, η=0
2.  In your analysis, match the leading reco cluster to the truth photon
3.  Plot (E\_reco - E\_truth)/E\_truth and fit a Gaussian
4.  The σ is the EMCal energy resolution at 5 GeV — sanity check it's
    roughly 10%/√E

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Centrality →

</div>


<!-- ===== LESSON: 3.7 ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Simulation & Reco · Lesson 7

</div>

# Centrality in Au+Au

How "head-on" was the collision? Why this matters for everything.

<div class="lesson-meta">

<div class="meta-item">

⏱ \~20 min read

</div>

<div class="meta-item">

🧪 Heavy-ion

</div>

</div>

</div>

<div class="content-section">

## What is Centrality?

In a heavy-ion collision, the two gold nuclei approach each other with
some **impact parameter** b. If b ≈ 0, they collide head-on ("central").
If b is large, they barely glance ("peripheral").

We can't measure b directly — but we can measure things *correlated*
with it: the total particle multiplicity, the energy in forward
calorimeters, etc. From these we assign each event to a percentile bin:

  - **0-10%** = the 10% most central events (highest multiplicity,
    smallest b)
  - **60-90%** = peripheral events (low multiplicity, large b)

QGP forms most strongly in central events. Your physics observables
almost always need to be reported as a function of centrality.

</div>

<div class="content-section">

## Reading Centrality

<div class="code-block">

<div class="code-header">

<span class="code-lang">C++</span>

Copy

</div>

    #include <centrality/CentralityInfo.h>
    
    CentralityInfo* cent =
        findNode::getClass<CentralityInfo>(topNode, "CentralityInfo");
    if (!cent) return Fun4AllReturnCodes::ABORTEVENT;
    
    // Centrality from MBD north+south combined
    float centile = cent->get_centile(CentralityInfo::PROP::mbd_NS);
    
    // Bin events
    if      (centile < 10)  m_h_pt_central->Fill(pt);
    else if (centile < 30)  m_h_pt_midcent->Fill(pt);
    else if (centile < 60)  m_h_pt_midperi->Fill(pt);
    else                   m_h_pt_peripheral->Fill(pt);

</div>

</div>

<div class="content-section">

## Standard Centrality Bins

sPHENIX papers typically use these bins:

  - 0-10%, 10-20%, 20-40%, 40-60%, 60-90% (5-bin standard)
  - 0-5%, 5-10%, 10-20%, ... (finer for high-statistics analyses)

</div>

<div class="content-section">

## Number-of-Collisions / Participants

For "scaled" measurements (e.g., R<sub>AA</sub>) you need
⟨N<sub>coll</sub>⟩ and ⟨N<sub>part</sub>⟩ — average number of binary
nucleon-nucleon collisions and number of participating nucleons in your
centrality bin. These come from a *Glauber Monte Carlo* calculation,
which the centrality group provides as look-up tables.

</div>

<div class="exercise-card">

<div class="exercise-header">

<span class="exercise-badge">Exercise 3.7</span>
<span class="exercise-title">Centrality-dependent pT</span>

</div>

1.  If you have access to Au+Au DSTs, add centrality reading to your
    analysis
2.  Make 5 pT histograms binned by centrality
3.  Plot all 5 normalized to events — observe how the spectrum hardens
    (or softens) with centrality
4.  (If only p+p data: skip — centrality doesn't apply.)

</div>

<div class="complete-section">

Mark Complete & Continue →

</div>

<div class="lesson-nav">

← Back

Next: Module 3 Quiz →

</div>


<!-- ===== LESSON: 3.q ===== -->

<div class="lesson-header">

<div class="lesson-breadcrumb">

Module 3 · Checkpoint

</div>

# Module 3 Checkpoint Quiz

You can simulate, reconstruct, and analyze. Prove it.

</div>

<div class="quiz-card">

<div class="quiz-question">

1\. The π⁰ → γγ decay is reconstructed by:

</div>

  - <span class="quiz-letter">A</span>Reading π⁰s directly from the
    tracker
  - <span class="quiz-letter">B</span>Pairing two EMCal photon clusters
    and computing invariant mass
  - <span class="quiz-letter">C</span>Looking for clusters in the HCal
  - <span class="quiz-letter">D</span>Reading MBD multiplicity

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

2\. The PDG code for a photon is:

</div>

  - <span class="quiz-letter">A</span>11
  - <span class="quiz-letter">B</span>22
  - <span class="quiz-letter">C</span>111
  - <span class="quiz-letter">D</span>443

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

3\. To suppress combinatorial background in a π⁰ → γγ analysis, the most
common single cut is:

</div>

  - <span class="quiz-letter">A</span>Photon pair energy asymmetry \<
    0.7
  - <span class="quiz-letter">B</span>Require ≥ 5 photons in the event
  - <span class="quiz-letter">C</span>|η|\<3
  - <span class="quiz-letter">D</span>Require MBD multiplicity \< 100

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

4\. The standard "central" centrality bin in sPHENIX Au+Au is:

</div>

  - <span class="quiz-letter">A</span>90-100%
  - <span class="quiz-letter">B</span>0-10%
  - <span class="quiz-letter">C</span>50-60%
  - <span class="quiz-letter">D</span>100%

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-question">

5\. To measure track reconstruction efficiency, you need:

</div>

  - <span class="quiz-letter">A</span>Only real data
  - <span class="quiz-letter">B</span>Simulation with truth matching
  - <span class="quiz-letter">C</span>Calibration data alone
  - <span class="quiz-letter">D</span>Just centrality

<div class="quiz-feedback">

</div>

</div>

<div class="complete-section">

Mark Complete → Module 4

</div>

<div class="lesson-nav">

← Back

Module 4: Production →

</div>


<!-- ===== LESSON: 4.1 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 1

</div>

# HTCondor Job Submission

Scale your analysis from one event to millions using the SDCC batch
farm.

</div>

<div class="content-section">

## Why batch submission?

Running 10 events interactively takes seconds. Running 10 million events
takes *days* on a single core. The SDCC operates a large compute farm
managed by **HTCondor** — you submit "jobs" that run in parallel on
hundreds of worker nodes simultaneously.

<div class="callout callout-info">

**Mental model:** Think of Condor as a queue of independent worker
scripts. You hand it 1,000 input files, it spreads them across 1,000
cores, and notifies you when each finishes.

</div>

</div>

<div class="content-section">

## The three files you need

A Condor submission requires three things: *(1)* a wrapper shell script
that executes one job, *(2)* a Condor submit description file, and *(3)*
a list of inputs to dispatch.

### 1\. The job wrapper (`run_job.sh`)

<div class="code-block">

Copy

    #!/bin/bash
    # Arg 1 = run number, Arg 2 = segment, Arg 3 = output dir
    
    source /opt/sphenix/core/bin/sphenix_setup.sh -n new
    export MYINSTALL=/sphenix/u/\$USER/install
    source /opt/sphenix/core/bin/setup_local.sh \$MYINSTALL
    
    RUN=\$1
    SEG=\$2
    OUT=\$3
    
    cd \$_CONDOR_SCRATCH_DIR
    root -b -q "Fun4All_Pi0Ana.C(0,\\"DST_CALO_run2pp-000\${RUN}-000\${SEG}.root\\",\\"out_\${RUN}_\${SEG}.root\\")"
    
    cp out_\${RUN}_\${SEG}.root \$OUT/

</div>

### 2\. The Condor description (`submit.sub`)

<div class="code-block">

Copy

    Universe        = vanilla
    Executable      = run_job.sh
    Arguments       = \$(RUN) \$(SEG) /sphenix/user/myname/output
    
    Output          = log/job_\$(RUN)_\$(SEG).out
    Error           = log/job_\$(RUN)_\$(SEG).err
    Log             = log/job_\$(RUN)_\$(SEG).log
    
    request_memory  = 2GB
    request_disk    = 5GB
    
    Queue RUN,SEG from filelist.txt

</div>

### 3\. The input list (`filelist.txt`)

<div class="code-block">

    23745, 0
    23745, 1
    23745, 2
    23746, 0
    23746, 1
    ...

</div>

</div>

<div class="content-section">

## Submitting and monitoring

<div class="terminal">

<div class="terminal-line">

<span class="prompt">\[user@sdcc\]$</span> mkdir -p log output

</div>

<div class="terminal-line">

<span class="prompt">\[user@sdcc\]$</span> condor\_submit submit.sub

</div>

<div class="terminal-out">

Submitting job(s)............................

</div>

<div class="terminal-out">

42 job(s) submitted to cluster 1234567.

</div>

<div class="terminal-line">

<span class="prompt">\[user@sdcc\]$</span> condor\_q $USER

</div>

<div class="terminal-out">

\-- Schedd: sched03.sdcc.bnl.gov

</div>

<div class="terminal-out">

OWNER BATCH\_NAME SUBMITTED DONE RUN IDLE TOTAL

</div>

<div class="terminal-out">

myname run\_job.sh 12:30 4 35 3 42

</div>

</div>

| Command                    | What it does                         |
| -------------------------- | ------------------------------------ |
| `condor_q`                 | List *your* queued/running jobs      |
| `condor_q -hold`           | See jobs stuck in HOLD state         |
| `condor_q -analyze JOB_ID` | Diagnose why a job won't start       |
| `condor_rm JOB_ID`         | Kill a single job                    |
| `condor_rm $USER`          | Kill ALL your jobs (use with care\!) |

</div>

<div class="content-section">

## Best practices that save you grief

<div class="callout callout-tip">

  - **Always test interactively first.** Run one segment by hand and
    verify the output before submitting 1,000 jobs.
  - **Write to `$_CONDOR_SCRATCH_DIR`**, then copy results out at the
    end. Worker nodes have local fast disk; writing to `/sphenix/`
    directly hammers the file system.
  - **Request realistic memory.** Asking for 16 GB when you need 2 GB
    delays your jobs in the queue.
  - **Hold a small "test pool"** of 5 jobs first; if they all crash,
    you've saved your reputation with the operations team.

</div>

<div class="callout callout-warning">

**Common holds:** "JobMemoryExceeded" → bump `request_memory`.
"PolicyViolation" → check that the executable is chmod +x. "Idle \> 24h"
→ the queue is just busy.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.1

</div>

Take the `Fun4All_Pi0Ana.C` macro you wrote in Module 3 and wrap it in a
Condor submission for 5 segments of one run. Use only your own scratch
space. Verify all 5 jobs complete and produce output ROOT files.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: File Lists →

</div>


<!-- ===== LESSON: 4.2 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 2

</div>

# File Lists & Datasets

How sPHENIX data is organized — and how to find what you need.

</div>

<div class="content-section">

## The data hierarchy

Real sPHENIX data lives in a structured filesystem on Lustre, and
metadata about it lives in the **FileCatalog** database. You'll usually
access files through one of three mechanisms:

| Source                        | When to use                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| `/sphenix/lustre01/sphnxpro/` | Production DSTs — official reconstructed data                 |
| `FileCatalog` (PSQL)          | Programmatic queries: "give me all DST\_CALO for run 23745"   |
| `CreateFileList.pl`           | Generate text file lists from the catalog for a Fun4All macro |

</div>

<div class="content-section">

## Anatomy of a DST filename

<div class="code-block">

    DST_CALO_run2pp-00023745-00000.root
    │   │    │       │         │
    │   │    │       │         └─ Segment number (each ~5,000 events)
    │   │    │       └─ Run number (zero-padded to 8 digits)
    │   │    └─ Dataset tag (run2pp = Run 2024 p+p collisions)
    │   └─ Stream type (CALO = calorimeters, TRACKING, GLOBAL, etc.)
    └─ DST = Data Summary Tape (reconstructed, ready for analysis)

</div>

<div class="callout callout-info">

**Other dataset tags:** `run2auau` (Au+Au 2024), `ana450_2024p009`
(analysis tag — production version), `nopileupcorr`, etc. Tags get long
because each one encodes a specific calibration + reconstruction
software version.

</div>

</div>

<div class="content-section">

## Generating a file list

<div class="code-block">

Copy

    # Built-in tool: queries FileCatalog and writes a list
    CreateFileList.pl -run 23745 -type DST_CALO_run2pp -build new -cdb 2024p009 -n 100
    
    # Output: dst_calo_run2pp-00023745.list — 100 file paths, one per line

</div>

Inside your Fun4All macro you read the list with
`Fun4AllDstInputManager`:

<div class="code-block">

Copy

    Fun4AllDstInputManager *in = new Fun4AllDstInputManager("DSTcalo");
    in->AddListFile("dst_calo_run2pp-00023745.list");
    se->registerInputManager(in);

</div>

</div>

<div class="content-section">

## Quality and "good run" lists

Not every run is usable. Detector glitches, beam aborts, and calibration
issues mean some runs must be excluded. The collaboration maintains
"Good Run Lists" (GRLs) — text files of approved run numbers per
analysis topic.

<div class="callout callout-tip">

Always filter your input file list against the official GRL for your
analysis. They're published in the
`sPHENIX-Collaboration/dataAndCalibQA` repo.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.2

</div>

Pick a recent run number from the FileCatalog. Generate a 50-segment
file list of `DST_CALO` for that run. Then write a Python or shell
one-liner that intersects your list with a Good Run List and outputs
only the surviving segments.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Calibration DB →

</div>


<!-- ===== LESSON: 4.3 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 3

</div>

# Calibration Database (CDB)

Connect your analysis to the right calibration constants for the right
run.

</div>

<div class="content-section">

## What lives in the CDB?

Detectors drift. Pedestals change between runs, gain calibrations evolve
over weeks, alignment shifts when the magnet is cycled. The **Conditions
Database** (CDB) stores time-versioned calibration objects, and your
analysis fetches the right ones automatically based on the run number
you process.

| Calibration type     | Examples                         |
| -------------------- | -------------------------------- |
| Detector pedestals   | EMCal tower offsets, INTT bias   |
| Tower-by-tower gains | EMCal, IHCal, OHCal              |
| Tracker alignment    | MVTX, INTT, TPC sector positions |
| Bad-channel masks    | Dead/hot tower lists per run     |
| Beam spot            | Vertex distribution per run      |

</div>

<div class="content-section">

## Wiring CDB into a Fun4All macro

<div class="code-block">

Copy

    #include <ffamodules/CDBInterface.h>
    #include <phool/recoConsts.h>
    
    // At the top of your macro:
    recoConsts *rc = recoConsts::instance();
    rc->set_StringFlag("CDB_GLOBALTAG", "ProdA_2024");
    rc->set_uint64Flag("TIMESTAMP", runnumber);
    
    CDBInterface::instance()->Verbosity(0);

</div>

<div class="callout callout-info">

**What's a "global tag"?** A global tag is a named bundle of calibration
sets — e.g. `ProdA_2024` means "use the calibrations approved for
production analysis A in the 2024 dataset." Switching tags lets you
reprocess data with newer calibrations without changing your code.

</div>

</div>

<div class="content-section">

## Looking up a calibration manually

Inside a SubsysReco module you can pull a specific payload:

<div class="code-block">

Copy

    #include <cdbobjects/CDBTTree.h>
    
    std::string calibfile = CDBInterface::instance()
        ->getUrl("CEMC_GAINS");
    
    CDBTTree *cdb = new CDBTTree(calibfile);
    cdb->LoadCalibrations();
    
    float gain = cdb->GetFloatValue(towerKey, "gain");

</div>

</div>

<div class="content-section">

## Common CDB pitfalls

<div class="callout callout-warning">

  - **Forgetting to set TIMESTAMP** → CDB returns a default that may be
    wrong for your run.
  - **Mismatched global tag** → reconstructed quantities silently drift;
    pi0 masses suddenly land at 120 MeV instead of 135 MeV.
  - **CDB connection timeout** → SDCC network blip; just retry. Add
    `CDBInterface::Verbosity(1)` to debug.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.3

</div>

Modify your Module 3 pi0 macro to set the global tag `ProdA_2024` and
the appropriate run-number timestamp. Re-run on 1,000 events and confirm
the pi0 peak position has changed (or stayed identical, if your prior
run was already correctly calibrated). Note the difference.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Publication Plots →

</div>


<!-- ===== LESSON: 4.4 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 4

</div>

# Publication-Quality Plots

From "looks fine on my screen" to "ready for a collaboration meeting."

</div>

<div class="content-section">

## The sPHENIX plot style guide

Every collaboration has plotting conventions. Following them isn't fussy
— it's how reviewers, conveners, and journal editors quickly read your
figures. The **sPHENIX style** includes specific font sizes, axis
labels, color schemes, and required annotations like *"sPHENIX
Preliminary"* or *"sPHENIX Internal"*.

<div class="callout callout-info">

**Where to find the official style:** the `sPHENIX-Collaboration/macros`
repo includes `sPhenixStyle.C`. Source it once at the top of any
plotting macro.

</div>

</div>

<div class="content-section">

## A complete plotting macro

<div class="code-block">

Copy

    #include "sPhenixStyle.C"
    
    void plot_pi0_mass() {
      SetsPhenixStyle();
    
      TFile *f = TFile::Open("output.root");
      TH1F *h = (TH1F*) f->Get("h_pi0_mass");
    
      TCanvas *c = new TCanvas("c", "", 800, 600);
      c->SetMargin(0.15, 0.05, 0.15, 0.05);
    
      h->GetXaxis()->SetTitle("M_{#gamma#gamma} [GeV/c^{2}]");
      h->GetYaxis()->SetTitle("Counts / 5 MeV");
      h->SetLineColor(kBlack);
      h->SetMarkerStyle(20);
      h->Draw("E1");
    
      // Required annotation
      TLatex tx;
      tx.SetNDC();
      tx.SetTextSize(0.04);
      tx.DrawLatex(0.20, 0.88, "#bf{sPHENIX} Internal");
      tx.DrawLatex(0.20, 0.83, "p+p #sqrt{s} = 200 GeV");
    
      c->SaveAs("pi0_mass.pdf");
      c->SaveAs("pi0_mass.png");
    }

</div>

</div>

<div class="content-section">

## Fitting the pi0 peak

<div class="code-block">

Copy

    // Gaussian signal + linear background
    TF1 *fit = new TF1("fit",
        "gaus(0) + pol1(3)", 0.08, 0.20);
    fit->SetParameters(1000, 0.135, 0.012, 0, 0);
    h->Fit(fit, "R");
    
    double mass  = fit->GetParameter(1);
    double sigma = fit->GetParameter(2);
    printf("Pi0 mass: %.4f +/- %.4f GeV/c^2\\n", mass, fit->GetParError(1));

</div>

<div class="callout callout-tip">

**Always quote uncertainties.** Statistical from the fit, plus an
estimate of systematic from varying the background model and fit range.

</div>

</div>

<div class="content-section">

## Polish checklist before showing a plot

| Check                     | Why                                               |
| ------------------------- | ------------------------------------------------- |
| Axis labels with units    | "Counts" alone is meaningless                     |
| Legend or text annotation | Reader shouldn't have to guess what each curve is |
| Collaboration label       | "sPHENIX Internal" or "Preliminary"               |
| Beam-energy / system      | e.g. "p+p √s = 200 GeV"                           |
| Error bars / bands        | Even if just statistical                          |
| Reasonable y-range        | Don't autosize — features get crushed             |

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.4

</div>

Take your pi0 mass histogram from Module 3, apply `SetsPhenixStyle()`,
fit it with Gaussian + polynomial background, and produce a
publication-quality PDF labeled with collaboration text and beam
conditions. Save both the macro and the resulting PDF.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Event Mixing →

</div>


<!-- ===== LESSON: 4.5 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 5

</div>

# Event Mixing for Combinatorial Background

Subtract the "false pairs" you create by combining all photons in an
event.

</div>

<div class="content-section">

## The combinatorial problem

When you combine every pair of photons in an event to look for pi0s, you
don't just get real pi0 decays — you get every accidental pair, too. In
a heavy-ion collision with hundreds of photons per event, the false-pair
"combinatorial background" can dwarf the signal.

<div class="callout callout-info">

**The trick:** photon pairs from a real pi0 are correlated; pairs from
accidentals are not. If you combine photons from *different* events with
similar properties, you reproduce the combinatorial shape *without* any
real pi0 signal — then you subtract.

</div>

</div>

<div class="content-section">

## Algorithm

1.  Define event "classes" by similar centrality, vertex z-position,
    event-plane angle, etc.
2.  Maintain a rolling buffer of the last N events per class.
3.  For each new event: pair every photon with photons from N *previous*
    events in the same class.
4.  Fill an "M\_mixed" histogram. Normalize and subtract from same-event
    "M\_same".

</div>

<div class="content-section">

## Sketch implementation

<div class="code-block">

Copy

    // Pseudo-code inside process_event()
    int centBin = getCentBin(centrality);
    int vzBin   = getVzBin(vertex_z);
    EventPool &pool = pools[centBin][vzBin];
    
    // Same-event pairs
    for (auto &p1 : photons) {
      for (auto &p2 : photons) {
        if (&p1 >= &p2) continue;
        h_same->Fill((p1+p2).M());
      }
    }
    
    // Mixed-event pairs (against pooled previous events)
    for (auto &p1 : photons) {
      for (auto &prevEvent : pool) {
        for (auto &p2 : prevEvent) {
          h_mixed->Fill((p1+p2).M());
        }
      }
    }
    
    pool.push(photons);
    if (pool.size() > 10) pool.pop_front();

</div>

</div>

<div class="content-section">

## Subtraction and normalization

The absolute scale of `h_mixed` isn't physical — only its *shape*.
Normalize to `h_same` in a sideband region (e.g. 0.18–0.30 GeV/c² where
there's no pi0 signal), then:

<div class="code-block">

Copy

    double norm = h_same->Integral(low,high) / h_mixed->Integral(low,high);
    h_mixed->Scale(norm);
    
    TH1F *h_signal = (TH1F*) h_same->Clone("h_signal");
    h_signal->Add(h_mixed, -1);

</div>

<div class="callout callout-warning">

Mixed-event subtraction is powerful but easy to get wrong. Always plot
`h_same`, `h_mixed`, and the subtracted result on top of each other to
verify the subtraction looks reasonable in sidebands.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.5

</div>

Add an event-mixing background to your pi0 macro. Bin the pool by
primary-vertex z (use 5 cm bins from -10 to +10). Plot the same-event
mass distribution, mixed background, and subtracted signal on the same
canvas with a legend.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Troubleshooting →

</div>


<!-- ===== LESSON: 4.6 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 6

</div>

# Troubleshooting Playbook

The most common errors you'll meet — and how to fix each one.

</div>

<div class="content-section">

## Build errors

| Error message                                 | Likely cause                                  | Fix                                                       |
| --------------------------------------------- | --------------------------------------------- | --------------------------------------------------------- |
| `command not found: cmake`                    | You didn't source the sphenix setup           | `source /opt/sphenix/core/bin/sphenix_setup.sh -n new`    |
| `fatal error: phool/PHObject.h: No such file` | `$MYINSTALL` not exported, or build dir wrong | Set `$MYINSTALL`, re-run `autogen.sh --prefix=$MYINSTALL` |
| `undefined reference to ...`                  | Missing library in `Makefile.am`              | Add the library to `libfoo_la_LIBADD`                     |
| `cannot find -lcalo_io`                       | Link line stale; offline\_main updated        | `make distclean && ./autogen.sh ...`                      |

</div>

<div class="content-section">

## Runtime crashes

| Symptom                      | What it usually means                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Segfault on first event      | Forgot `InitRun`, or accessed null node — check `findNode::getClass` return                    |
| Crash deep in ROOT TTree     | You wrote into a branch without resizing a vector — initialize containers in InitRun           |
| `ABORTRUN` from a SubsysReco | That module returned `Fun4AllReturnCodes::ABORTRUN` — read its log lines just before the abort |
| Hang at end-of-job           | Output file never written — make sure your output manager is registered                        |

</div>

<div class="content-section">

## Workflow / Condor issues

<div class="callout callout-warning">

  - **Job stays IDLE for hours:** queue is busy. Check `condor_q
    -analyze`. Don't resubmit — that pushes you down the priority list.
  - **Job HOLD with "memory exceeded":** double `request_memory` and
    resubmit just the held jobs (`condor_release`).
  - **Output files appear empty (0 bytes):** the wrapper script didn't
    `cd` back to scratch before `cp`. Inspect the .err log.
  - **Some jobs missing:** diff your `filelist.txt` against output
    filenames; resubmit the missing entries.

</div>

</div>

<div class="content-section">

## Physics / data issues

| Observation                          | What to check                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| Pi0 mass shifted from 135 MeV        | Wrong calibration tag, or stale local `$MYINSTALL`                             |
| Signal yield 10× lower than expected | Missing run in good-run list; centrality cut wrong; trigger filter not applied |
| Tracks all clustered at low pT       | Likely no quality cuts — require MVTX+INTT+TPC hits                            |
| "Truth" matching empty               | Confirm input is simulation DST, not real data                                 |

</div>

<div class="content-section">

## How to ask for help (and get answers)

<div class="callout callout-tip">

1.  State what you tried, what you expected, what happened.
2.  Paste the *full* command, the *full* error (with file/line numbers),
    and your sphenix build version.
3.  Identify the relevant Mattermost channel: `#calo-software`,
    `#tracking-software`, `#fun4all`, `#sdcc-help`.
4.  If posting code, link a gist or push to a branch — don't paste 200
    lines into chat.

</div>

</div>

<div class="exercise-card">

<div class="exercise-header">

Exercise 4.6

</div>

Intentionally break your pi0 macro three ways: (1) skip `InitRun`, (2)
request a non-existent input file, (3) use the wrong calibration tag.
For each, capture the error, identify which row of the troubleshooting
tables matches, and document what would be the next debugging step.

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Cheat Sheet →

</div>


<!-- ===== LESSON: 4.7 ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Lesson 7

</div>

# Commands Cheat Sheet

The commands you'll actually use, every single day.

</div>

<div class="content-section">

## Environment setup

<div class="code-block">

    # Source sPHENIX environment (every login)
    source /opt/sphenix/core/bin/sphenix_setup.sh -n new
    
    # Use your local installs
    export MYINSTALL=/sphenix/u/$USER/install
    source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL

</div>

</div>

<div class="content-section">

## Build cycle

<div class="code-block">

    cd source/MyModule
    mkdir -p build && cd build
    ../autogen.sh --prefix=$MYINSTALL
    make -j4 install
    make distclean   # nuclear reset when things go sideways

</div>

</div>

<div class="content-section">

## Running Fun4All

<div class="code-block">

    # Interactive (test small)
    root -b -q 'Fun4All_Pi0Ana.C(100)'
    
    # Quiet for batch
    root -b -q -l 'Fun4All_Pi0Ana.C(100, "input.root", "output.root")'

</div>

</div>

<div class="content-section">

## File catalog & data discovery

<div class="code-block">

    CreateFileList.pl -run 23745 -type DST_CALO_run2pp -build new -cdb 2024p009 -n 100
    psql FileCatalog -c "SELECT * FROM datasets WHERE runnumber=23745;"
    ls /sphenix/lustre01/sphnxpro/physics/slurp/calolist/run_00023700_00023800/

</div>

</div>

<div class="content-section">

## Condor — the big six

<div class="code-block">

    condor_submit submit.sub
    condor_q $USER
    condor_q -hold
    condor_q -analyze JOB_ID
    condor_release JOB_ID
    condor_rm JOB_ID

</div>

</div>

<div class="content-section">

## ROOT one-liners

<div class="code-block">

    root -l output.root            # open file
    .ls                            # list contents
    T->Print()                     # inspect a TTree
    T->Draw("mass")                # quick histogram
    T->Scan("pt:eta:phi")          # dump entries to terminal
    .q                             # quit

</div>

</div>

<div class="content-section">

## Git basics

<div class="code-block">

    git checkout -b my-feature
    git add -p
    git commit -m "Add pi0 systematic study"
    git push origin my-feature
    gh pr create --base main --title "..." --body "..."

</div>

</div>

<div class="content-section">

## Disk hygiene

<div class="code-block">

    du -sh ~/install ~/work
    quota -s                       # your /sphenix/u quota
    ls -lhSr | tail                # biggest files in CWD

</div>

<div class="callout callout-tip">

Print this lesson — it'll spend more time on your monitor than any
other.

</div>

</div>

<div class="complete-section">

Mark Complete →

</div>

<div class="lesson-nav">

← Back

Next: Capstone →

</div>


<!-- ===== LESSON: 4.f ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Capstone Project

</div>

# Capstone: Pi0 Yield vs. Centrality

Synthesize everything: simulation, reconstruction, batch processing,
calibration, plotting.

</div>

<div class="content-section">

## The brief

You'll measure the pi0 yield as a function of collision centrality in
Au+Au data, using the entire sPHENIX workflow. This is the kind of
mini-result a first-year student is genuinely expected to produce — and
it touches every skill from this course.

<div class="callout callout-info">

**Estimated time:** 1 to 2 weeks of focused work. Plan accordingly.

</div>

</div>

<div class="content-section">

## Required deliverables

1.  A SubsysReco module `Pi0CentralityAna` with InitRun, process\_event,
    End — written, built into `$MYINSTALL`, version-controlled in your
    fork.
2.  A Fun4All steering macro that reads `DST_CALO` + `DST_GLOBAL` for an
    Au+Au run, applies the correct calibration global tag, and produces
    an output ROOT file.
3.  A Condor submission that runs over at least 50 segments — log files
    retained for inspection.
4.  A plotting macro that, in 4 centrality bins (0–10%, 10–30%, 30–50%,
    50–80%):
      - Plots the same-event diphoton mass spectrum.
      - Constructs and subtracts a mixed-event combinatorial background.
      - Fits the pi0 peak (Gaussian + linear residual).
      - Extracts a yield with statistical uncertainty.
5.  A summary plot of **raw pi0 yield vs. centrality** in sPHENIX style,
    with collaboration label.
6.  A short (1–2 page) write-up explaining your method, results, and at
    least two sources of systematic uncertainty you would investigate
    next.

</div>

<div class="content-section">

## Suggested milestones (week-by-week)

| Day   | Milestone                                                             |
| ----- | --------------------------------------------------------------------- |
| 1–2   | Module skeleton compiles; runs over 100 events with debug printouts   |
| 3–4   | Same-event mass histogram in one centrality bin looks like a pi0 peak |
| 5–7   | Add centrality binning + Condor submission; reproduce on 50 segments  |
| 8–10  | Implement mixed-event background; compare same/mixed/subtracted       |
| 11–13 | Fitting code + yield extraction + sPHENIX-style final plot            |
| 14    | Write-up; commit everything; share with your mentor                   |

</div>

<div class="content-section">

## How you'll be assessed (informally, by your mentor)

<div class="callout callout-tip">

  - **Correctness:** Does the pi0 mass land near 135 MeV? Does the yield
    trend make physical sense (more pi0s in central events)?
  - **Code hygiene:** Reasonable variable names, comments, no committed
    binary blobs, clean Git history.
  - **Reproducibility:** Could someone else clone your repo and
    reproduce the final plot?
  - **Self-awareness:** Do you understand the limitations of your own
    measurement?

</div>

</div>

<div class="callout callout-info">

**Stuck?** That's expected — getting unstuck efficiently is the actual
skill being tested. Bring focused questions to your mentor and the
relevant Mattermost channel.

</div>

<div class="complete-section">

Mark Capstone Complete →

</div>

<div class="lesson-nav">

← Back

Next: Final Exam →

</div>


<!-- ===== LESSON: 4.q ===== -->

<div class="lesson-header">

<div class="lesson-eyebrow">

Module 4 · Final Exam

</div>

# Final Exam

Twelve questions covering the entire course. Aim for ≥ 10/12.

</div>

<div class="quiz-card">

<div class="quiz-q">

1\. Which command sources the new sPHENIX environment?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

module load sphenix

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

source /opt/sphenix/core/bin/sphenix\_setup.sh -n new

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

conda activate sphenix

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

export SPHENIX=new

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

2\. Inside Fun4All, where is reconstructed data shared between modules?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

In a global C++ map

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

On the PHCompositeNode tree

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

In a SQL database

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

In ROOT TFile branches

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

3\. Which SubsysReco method is called once per event?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Init

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

InitRun

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

process\_event

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

End

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

4\. Which DST stream contains calorimeter towers and clusters?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, true)">

DST\_CALO

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

DST\_TRACKS

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

DST\_TRUTH

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

DST\_GLOBAL

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

5\. What is the rest mass of the pi0 you should expect?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

\~ 91 GeV/c²

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

\~ 938 MeV/c²

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

\~ 135 MeV/c²

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

\~ 547 MeV/c²

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

6\. Which command kills ALL of your jobs at once?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

condor\_kill

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

condor\_rm $USER

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

condor\_drop --all

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

qdel \*

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

7\. What does setting `CDB_GLOBALTAG` do?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Selects which DST files to read

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

Selects a named bundle of calibration constants

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Tags your job for the Condor queue

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Names the output ROOT file

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

8\. Why subtract a mixed-event background in pi0 reconstruction?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

To remove dead-tower contributions

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

To correct for trigger inefficiency

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

To remove combinatorial false pairs from photons that aren't actually
decay products of the same pi0

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

To recalibrate cluster energies

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

9\. Which environment variable points to your local install of compiled
libraries?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

$SPHENIX\_HOME

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

$MYINSTALL

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

$LD\_LIBRARY\_PATH

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

$F4ALL

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

10\. To make plots conform to collaboration style, you should...

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Use the ROOT default style

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

Source `sPhenixStyle.C` and call `SetsPhenixStyle()`

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Export everything to matplotlib

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Manually set every gPad attribute

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

11\. A Condor job stays in HOLD with "JobMemoryExceeded". What's the
right fix?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Resubmit immediately as is

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

Increase `request_memory` in your submit file and use condor\_release

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Switch to a different login node

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

File a ticket with SDCC

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="quiz-card">

<div class="quiz-q">

12\. What's the right place for a worker job to write its temporary
output?

</div>

<div class="quiz-options">

<div class="quiz-option" onclick="selectQuizOption(this, false)">

/tmp on the worker node

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Directly to /sphenix/lustre01/

</div>

<div class="quiz-option" onclick="selectQuizOption(this, true)">

$\_CONDOR\_SCRATCH\_DIR, then copy to /sphenix at job end

</div>

<div class="quiz-option" onclick="selectQuizOption(this, false)">

Your $HOME directory

</div>

</div>

<div class="quiz-feedback">

</div>

</div>

<div class="callout callout-info">

**10–12 correct:** You're ready for real production work. Send your
capstone to your mentor.  
**7–9 correct:** Skim the cheat sheet and revisit Modules 3 & 4 — you're
close.  
**\< 7:** Step back through the modules in order; the foundation is what
makes the rest stick.

</div>

<div class="complete-section">

Mark Complete · Course Finished 🎓

</div>

<div class="lesson-nav">

← Back

Return to Welcome

</div>
