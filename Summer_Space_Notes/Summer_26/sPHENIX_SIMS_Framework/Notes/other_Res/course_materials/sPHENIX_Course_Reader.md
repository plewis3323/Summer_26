**sPHENIX**

Course Reader

*From Zero to Production in Eight Weeks*

Brookhaven National Laboratory

sPHENIX Collaboration • Relativistic Heavy Ion Collider

*Companion volume to the 8-Week Onboarding Syllabus*

How to Read This Book
=====================

This Reader is the core teaching text for the sPHENIX 8-Week Onboarding
course. One chapter per week, eight chapters total. Each chapter is
paired with a Lab Worksheet that you will work through on the cluster.

Read the chapter first --- out loud if that helps you stay attentive ---
then open the Lab Worksheet and work the exercises. Don't skip the
Check-for-Understanding boxes; they will tell you, fast, whether you
actually absorbed the material or just skimmed it.

The pace is deliberately demanding. The material is deliberately
concrete. Every code example in this book compiles --- if something in
it doesn't, that's a bug; please send a patch.

A typographical note. Code appears in a monospaced font with a light
background. Definitions appear in two-column tables. Asides and worked
examples appear in amber-tinted call-out boxes. Check-for-Understanding
questions appear in green-tinted boxes --- attempt each before moving
on.

Conventions
-----------

-   \$VAR denotes a shell environment variable.

-   \`command\` denotes a shell command; prompts (\$ or root \[n\]) are
    dropped unless relevant.

-   "the cluster" means an SDCC interactive node at BNL; "the farm"
    means the Condor worker pool.

-   "production DSTs" means files under /sphenix/lustre01/sphnxpro
    produced by the central reconstruction.

Contents
========

**Week 1** The Cluster, the Environment, and Your First ROOT Plot
*Module 1 --- Tooling up*

**Week 2** Git, C++ for HEP, and a Full TTree Analysis *Module 1 ---
Tooling up*

**Week 3** Fun4All and the Node Tree *Module 2 --- Fun4All deep dive*

**Week 4** Building with CMake and Running Fun4All *Module 2 --- Fun4All
deep dive*

**Week 5** Running the Full Simulation and Matching to Truth *Module 3
--- Simulation and physics*

**Week 6** Calorimeter Analysis and the Pi0 Peak *Module 3 ---
Simulation and physics*

**Week 7** Condor: Running at Scale *Module 4 --- Production*

**Week 8** Event Mixing, Publication Plots, and Contributing Back
*Module 4 --- Production*

**Chapter 1**

**Week 1: The Cluster, the Environment, and Your First ROOT Plot**

*Module 1 --- Tooling up*

Goals for this week
-------------------

-   Log into SDCC and know where you live on the filesystem.

-   Source the sPHENIX environment and explain what every variable
    points to.

-   Survive at the Linux command line long enough to do real work.

-   Write a ROOT macro that fills, fits, and plots a histogram.

1.1 What "the cluster" actually is
----------------------------------

Everything you will do in this course happens on the sPHENIX computing
cluster at the Scientific Data and Computing Center (SDCC) at Brookhaven
National Laboratory. Before we ever touch physics code, I want you to
have an accurate mental picture of what you're connecting to.

When you SSH into an SDCC interactive node, you are logging into one of
many identical Linux machines that all see the same shared filesystem.
The machine you land on is interchangeable with the one your neighbor
lands on, because the interesting state --- your files, the sPHENIX
software, the production data --- lives on network filesystems that
every node can see.

This matters for a practical reason: whatever you do on node A can be
picked up on node B, because the files you just wrote are already
visible to the whole farm. When we get to Condor in Week 7, a job Condor
assigns to a worker node on the other side of the data center can still
read your macro and your input DST, because they are on shared storage.
If you try to reason about the cluster as one monolithic computer,
you'll trip over edge cases; the correct mental model is "many machines,
one filesystem."

+----------------------------------------------------------------------+
| **Where you live**                                                   |
|                                                                      |
| Your home directory is /sphenix/user/\<username\>/. This is on GPFS  |
| (a high-performance parallel filesystem) and is backed up. It's      |
| where your code and your personal install area belong. Bulk data     |
| outputs go elsewhere (we'll get to that in Week 7).                  |
+----------------------------------------------------------------------+

1.2 The environment, and why you source it
------------------------------------------

sPHENIX has a lot of software: ROOT, Geant4, HepMC, FastJet, and the
whole sPHENIX offline stack. All of it is pre-installed under
/opt/sphenix and served via CVMFS (a read-only global filesystem). You
don't install it. You don't compile it. You just tell your shell to find
it, by sourcing the setup script:

> source /opt/sphenix/core/bin/sphenix\_setup.sh -n new

What that one line does is set about a dozen environment variables so
every command you type afterwards knows where ROOT, Geant4, and the
offline libraries live. The most important ones:

  ------------------- ---------------------------------------------------------------------------------------------
  \$OFFLINE\_MAIN     The root of the chosen sPHENIX build. Every sPHENIX include path and library is under here.
  \$ROOTSYS           Where ROOT lives.
  \$G4\_MAIN          Where Geant4 lives.
  \$CALIBRATIONROOT   Where calibration constants live.
  ------------------- ---------------------------------------------------------------------------------------------

The -n new flag picks the "new" build, which is the latest weekly
snapshot. For day-to-day development, new is what you want. For a paper
or internal note you'll eventually pin to a specific build tag (ana.XXX)
so your results are reproducible six months later.

Add two more lines to your shell startup to set up your personal install
area --- the place where the libraries you compile in Week 4 will live:

> export MYINSTALL=/sphenix/user/\$USER/install
>
> source /opt/sphenix/core/bin/setup\_local.sh \$MYINSTALL

The second line prepends \$MYINSTALL/lib to LD\_LIBRARY\_PATH and adds
\$MYINSTALL/include to the compiler search path. That's how Fun4All will
find the .so file you'll build in Module 2. Without this line, your
compiled module is invisible to the framework.

1.3 Enough Linux to not panic
-----------------------------

You don't need to be a Linux wizard, but there are about a dozen
commands you'll use every single day. Get fluent with these before you
move on.

  -------------- ----------------------------------------------------------------
  Navigation     cd, ls, pwd, find, locate
  File work      cp, mv, rm, mkdir, chmod
  Editing        vim, emacs, or nano. Pick one, commit to it.
  Viewing        cat, less, head, tail, grep
  Processes      ps, top, kill, nohup
  Long-running   screen or tmux --- detach a terminal, come back to it tomorrow
  Networking     ssh, scp, rsync
  -------------- ----------------------------------------------------------------

If you're new to a terminal, the one skill that pays the most dividends
per hour invested is shell scripting --- variables, loops, conditionals.
Most "tedious" sPHENIX tasks are one shell script away from automation,
and we'll be writing them all course long.

1.4 ROOT --- the framework that runs all of HEP
-----------------------------------------------

ROOT is the data-analysis framework that physics has been building on
since the late 1990s. You will use it for file I/O, columnar data
(TTree), histograms, fitting, and plotting. There is no sPHENIX analysis
that does not go through ROOT, so investing time now makes every later
week easier.

You launch ROOT with:

> root -l \# interactive, no splash
>
> root -l myMacro.C \# run a macro
>
> root -l -b -q macro.C \# batch, no graphics, quit when done

The -b -q combo is what you'll use for automated runs (inside a bash
wrapper, inside a Condor job) because it suppresses the graphical window
and exits as soon as the macro finishes.

A ROOT "macro" is literally just a C++ file that ROOT interprets. You
can put almost anything in it --- loops, functions, class definitions
--- and ROOT's Cling interpreter will run it without a compilation step.
For small analysis scripts that's exactly what you want.

+----------------------------------------------------------------------+
| **Worked example --- histogram a Gaussian**                          |
|                                                                      |
| This is the "hello world" of ROOT. It should compile cleanly on any  |
| sPHENIX node once the environment is sourced.                        |
+----------------------------------------------------------------------+

> void hello\_gauss() {
>
> TH1F \*h = new TH1F(\"h\", \"My Gaussian;x;counts\", 100, -5, 5);
>
> for (int i = 0; i \< 100000; ++i) h-\>Fill(gRandom-\>Gaus(0, 1));
>
> TCanvas \*c = new TCanvas(\"c\", \"\", 800, 600);
>
> h-\>Draw();
>
> h-\>Fit(\"gaus\");
>
> c-\>SaveAs(\"hello\_gauss.pdf\");
>
> }

Save as hello\_gauss.C and run with root -l -b -q hello\_gauss.C. You
should get a PDF in the current directory showing a bell curve with a
fit overlay. If that works, your environment is sound.

1.5 Reading a TTree
-------------------

Most sPHENIX data is stored in TTrees. A TTree is, roughly, a database
table: rows are events (or particles, or tracks), columns are variables.
The two methods you'll use constantly are Draw (quick interactive
exploration) and SetBranchAddress + GetEntry (for real analysis code).

> TFile \*f = new TFile(\"myfile.root\");
>
> TTree \*t = (TTree\*)f-\>Get(\"ntp\_cluster\");
>
> // Quick interactive draws
>
> t-\>Draw(\"e\"); // 1D: cluster energy
>
> t-\>Draw(\"e\", \"pt \> 1\"); // with a cut
>
> t-\>Draw(\"eta:phi\", \"\", \"colz\"); // 2D: eta vs phi

For anything non-trivial, connect each branch to a C++ variable and loop
yourself --- that's the pattern you'll use inside SubsysReco modules in
Weeks 3--4, so getting comfortable now is a good investment.

1.6 Check for understanding
---------------------------

  -------- -------------------------------------------------------------------------------------
  **Q1**   What does echo \$OFFLINE\_MAIN print before and after you source sphenix\_setup.sh?
  -------- -------------------------------------------------------------------------------------

  -------- ----------------------------------------------------------------------------------------------------------------------------------------------------------
  **Q2**   If you source the setup script but forget to source setup\_local.sh, will your own compiled library in \$MYINSTALL/lib be found by ROOT? Why or why not?
  -------- ----------------------------------------------------------------------------------------------------------------------------------------------------------

  -------- --------------------------------------------------------------------------------------------------------------------------------
  **Q3**   You have a TTree called T in a file called data.root. Write a one-liner to plot the pT distribution with a cut \|eta\| \< 1.1.
  -------- --------------------------------------------------------------------------------------------------------------------------------

*--- End of Week 1 reading ---*

*Now open Lab Worksheet 1.*

**Chapter 2**

**Week 2: Git, C++ for HEP, and a Full TTree Analysis**

*Module 1 --- Tooling up*

Goals for this week
-------------------

-   Clone a repo, make a branch, push a commit, open a pull request.

-   Read enough C++ to understand a SubsysReco header file.

-   Write a complete analysis macro that reads a TTree, applies cuts,
    fills histograms, and saves a plot.

2.1 Git isn't scary; it's version control for your sanity
---------------------------------------------------------

sPHENIX, like every modern physics collaboration, runs on git and
GitHub. You can get surprisingly far just knowing clone, branch, add,
commit, push, pull. The single most valuable habit: always work on a
branch, never directly on master. If you mess something up, you can
throw the branch away and start over.

The canonical sPHENIX repositories you will touch this course:

  -------------- -------------------------------------------------------------------------------------------------------------------------
  coresoftware   The framework, detector code, reconstruction --- you read it constantly; you rarely modify it until you're a committer.
  macros         The Fun4All macros that drive simulation and reconstruction. You will fork this and use your fork.
  tutorials      Minimal worked examples for specific tasks.
  analysis       Where student and analyst code lives. Your code belongs here (eventually).
  -------------- -------------------------------------------------------------------------------------------------------------------------

2.2 The workflow you will use for the rest of your career
---------------------------------------------------------

> \# One time
>
> git clone https://github.com/sPHENIX-Collaboration/macros.git
>
> cd macros
>
> \# Every time you start work
>
> git checkout master
>
> git pull
>
> git checkout -b fix/my-analysis-week2
>
> \# While working
>
> git status \# what changed
>
> git diff \# show unstaged diffs
>
> git add \<files\> \# stage them
>
> git commit -m \"msg\" \# commit locally
>
> git push origin fix/my-analysis-week2 \# publish

Commit messages are a gift to future you. A good message explains why,
not what --- the diff already shows what. "Rename variable" is useless;
"Rename pt to leadingPt for clarity after adding subleading cut" is
helpful.

2.3 Just enough C++
-------------------

You do not need to be a C++ expert. You need to be able to read sPHENIX
code, write a class that inherits from SubsysReco, and avoid the five or
six traps that bite HEP newcomers.

The traps, in order of how often they will bite you:

-   Null pointers. findNode::getClass returns nullptr if the node
    doesn't exist. Always check. Always.

-   Shadowing. Writing float pt = \...; inside a loop when pt is already
    a member variable --- your histogram looks empty and you can't
    figure out why.

-   Integer division. int a = 5, b = 2; float c = a/b; gives c = 2.0,
    not 2.5. Cast early.

-   Missing includes. The error messages in C++ are terrible; a missing
    include often shows up as a completely unrelated complaint 300 lines
    later.

-   Mixing new and stack allocation. Be consistent. In Fun4All, objects
    you create in Init and write in End should be heap-allocated with
    new; everything else can be on the stack.

Containers you'll use constantly: std::vector\<T\>, std::map\<K,V\>,
std::pair\<A,B\>. Get comfortable with range-based for loops:

> std::vector\<float\> pts;
>
> for (float pt : pts) { /\* \... \*/ }
>
> std::map\<int, SvtxTrack\*\> trackmap;
>
> for (auto &it : trackmap) {
>
> int key = it.first;
>
> SvtxTrack \*track = it.second;
>
> }

2.4 A complete analysis macro
-----------------------------

Here's the macro you should be able to write by the end of this week. It
reads a TTree, applies a cut, fills a histogram, and saves a PDF.
Everything you do later in the course is a variation on this pattern.

> void pt\_analysis(const char \*infile = \"clusters.root\") {
>
> // 1. Open file and get tree
>
> TFile \*f = TFile::Open(infile);
>
> if (!f \|\| f-\>IsZombie()) {
>
> std::cerr \<\< \"Cannot open \" \<\< infile \<\< std::endl;
>
> return;
>
> }
>
> TTree \*t = (TTree\*)f-\>Get(\"ntp\_cluster\");
>
> if (!t) {
>
> std::cerr \<\< \"No ntp\_cluster tree in \" \<\< infile \<\<
> std::endl;
>
> return;
>
> }
>
> // 2. Hook up branches
>
> float e, pt, eta;
>
> t-\>SetBranchAddress(\"e\", &e);
>
> t-\>SetBranchAddress(\"pt\", &pt);
>
> t-\>SetBranchAddress(\"eta\", &eta);
>
> // 3. Prepare histograms
>
> TH1F \*h\_pt\_all = new TH1F(\"h\_pt\_all\", \";p\_{T}
> \[GeV/c\];counts\", 100, 0, 20);
>
> TH1F \*h\_pt\_barrel = new TH1F(\"h\_pt\_barrel\", \";p\_{T}
> \[GeV/c\];counts\", 100, 0, 20);
>
> // 4. Loop
>
> Long64\_t n = t-\>GetEntries();
>
> for (Long64\_t i = 0; i \< n; ++i) {
>
> t-\>GetEntry(i);
>
> h\_pt\_all-\>Fill(pt);
>
> if (std::fabs(eta) \< 1.1) h\_pt\_barrel-\>Fill(pt);
>
> }
>
> // 5. Plot and save
>
> TCanvas \*c = new TCanvas(\"c\", \"\", 800, 600);
>
> h\_pt\_all-\>SetLineColor(kGray + 2);
>
> h\_pt\_barrel-\>SetLineColor(kBlue);
>
> h\_pt\_all-\>Draw();
>
> h\_pt\_barrel-\>Draw(\"same\");
>
> c-\>SetLogy();
>
> TLegend \*leg = new TLegend(0.6, 0.7, 0.88, 0.88);
>
> leg-\>AddEntry(h\_pt\_all, \"all\", \"l\");
>
> leg-\>AddEntry(h\_pt\_barrel, \"\|eta\| \< 1.1\", \"l\");
>
> leg-\>Draw();
>
> c-\>SaveAs(\"pt\_analysis.pdf\");
>
> }

Walk through this macro line by line with a colleague (or with Claude)
until you can explain every statement. When you can, you're ready for
Module 2.

2.5 Check for understanding
---------------------------

  -------- --------------------------------------------------------------------------------------------------------------------
  **Q1**   What's the difference between git pull and git pull \--rebase? Which does the course's workflow implicitly assume?
  -------- --------------------------------------------------------------------------------------------------------------------

  -------- ----------------------------------------------------------------------------------------------------------
  **Q2**   You write int n = t-\>GetEntries(); on a tree with 5 billion entries. Why is that a bug? What's the fix?
  -------- ----------------------------------------------------------------------------------------------------------

  -------- ----------------------------------------------------------------------------------------------------------------
  **Q3**   In the macro above, if ntp\_cluster is stored inside a subdirectory like outputs/, what changes about f-\>Get?
  -------- ----------------------------------------------------------------------------------------------------------------

*--- End of Week 2 reading ---*

*Now open Lab Worksheet 2.*

**Chapter 3**

**Week 3: Fun4All and the Node Tree**

*Module 2 --- Fun4All deep dive*

Goals for this week
-------------------

-   Explain what Fun4All does and how the node tree is organized.

-   Write a minimal SubsysReco that reads tracks off the node tree.

-   Know the difference between EVENT\_OK, ABORTEVENT, and ABORTRUN.

3.1 What Fun4All actually is
----------------------------

Every analysis in sPHENIX --- simulation, reconstruction, calibration,
your code --- runs inside Fun4All. Understanding its mental model is
non-negotiable, and fortunately the model is simple. Fun4All is an event
loop. You register modules. Fun4All processes events one at a time by
calling each module's process\_event method in the order you registered
them. Between modules, everything is communicated through a shared
in-memory data structure called the node tree.

If you've written Python decorators or middleware in a web framework,
the pattern should feel familiar: a pipeline of stages, with a shared
request context. Fun4All is that, for events.

3.2 The node tree
-----------------

Think of the node tree as a bulletin board. Modules post data to named
slots ("nodes") and other modules read those slots. Track reconstruction
posts SvtxTrackMap. Calorimeter clustering posts CLUSTER\_CEMC. Your
module reads whatever it needs and writes its own output (if any) to new
slots.

Canonical node names you'll meet this module:

  ------------------ ------------------------------------------------------------------------
  SvtxTrackMap       The collection of reconstructed tracks. Iterable, one track per entry.
  SvtxVertexMap      Reconstructed primary vertices.
  GlobalVertexMap    Globally-combined event vertices (MBD + silicon + ...).
  CLUSTER\_CEMC      EMCal clusters.
  CLUSTER\_HCALIN    Inner HCal clusters.
  CLUSTER\_HCALOUT   Outer HCal clusters.
  G4TruthInfo        Monte Carlo truth (only on simulation DSTs).
  CentralityInfo     Centrality for heavy-ion events.
  ------------------ ------------------------------------------------------------------------

The exact list depends on which DST type you're reading. A DST\_TRACKS
has tracking nodes; a DST\_CALO has calorimeter nodes; a DST\_TRUTH has
Monte Carlo truth. Often an analysis wants several, which is why
production also produces combined DSTs.

+----------------------------------------------------------------------+
| **Debug trick**                                                      |
|                                                                      |
| If you can't remember what's on the node tree in a given DST, add    |
| topNode-\>print() to your Init and redirect the output to a file.    |
| You get the whole hierarchy, typed. Keep the output somewhere handy  |
| --- you'll refer to it constantly.                                   |
+----------------------------------------------------------------------+

3.3 Anatomy of a SubsysReco
---------------------------

A SubsysReco is a C++ class with specific virtual methods. The three you
almost always override are Init, process\_event, and End. Here is the
minimal header, annotated.

> \#ifndef MYANALYSIS\_H
>
> \#define MYANALYSIS\_H
>
> \#include \<fun4all/SubsysReco.h\>
>
> \#include \<string\>
>
> class PHCompositeNode;
>
> class TFile;
>
> class TH1F;
>
> class MyAnalysis : public SubsysReco {
>
> public:
>
> MyAnalysis(const std::string &name = \"MyAnalysis\");
>
> \~MyAnalysis() override;
>
> int Init(PHCompositeNode \*topNode) override;
>
> int process\_event(PHCompositeNode \*topNode) override;
>
> int End(PHCompositeNode \*topNode) override;
>
> void set\_output\_file(const std::string &fn) { \_outfile\_name = fn;
> }
>
> private:
>
> std::string \_outfile\_name {\"myanalysis.root\"};
>
> TFile \*\_out {nullptr};
>
> TH1F \*\_h\_ntrk {nullptr};
>
> };
>
> \#endif

A few things to notice. (1) Forward declarations (PHCompositeNode,
TFile, TH1F) keep the header light --- the cc file does the actual
includes. (2) Default-initialized members with brace syntax are much
harder to get wrong than C-style initializer-list-only. (3) The setter
is public so the Fun4All macro can configure the module before
registering it.

3.4 A minimal process\_event
----------------------------

Here's a complete process\_event that reads tracks off the node tree and
fills one histogram.

> int MyAnalysis::process\_event(PHCompositeNode \*topNode) {
>
> SvtxTrackMap \*trackmap =
>
> findNode::getClass\<SvtxTrackMap\>(topNode, \"SvtxTrackMap\");
>
> if (!trackmap) {
>
> std::cerr \<\< \"MyAnalysis: SvtxTrackMap not found\" \<\< std::endl;
>
> return Fun4AllReturnCodes::ABORTEVENT;
>
> }
>
> \_h\_ntrk-\>Fill(trackmap-\>size());
>
> return Fun4AllReturnCodes::EVENT\_OK;
>
> }

Three things: we fetched a node by name, we null-checked, and we
returned EVENT\_OK. If the node were missing, we could have aborted the
event (ABORTEVENT, Fun4All skips to the next event) or the whole run
(ABORTRUN, Fun4All stops entirely). For a missing node, ABORTEVENT is
almost always correct --- a single bad event shouldn't kill your
analysis.

3.5 Return codes matter
-----------------------

  -------------- ----------------------------------------------------------------------------
  EVENT\_OK      All good; move on.
  ABORTEVENT     Skip this event; continue.
  ABORTRUN       Something catastrophic. Stop processing.
  DISCARDEVENT   Drop this event from the output DST (only meaningful in producer modules).
  -------------- ----------------------------------------------------------------------------

You will overwhelmingly use EVENT\_OK and occasionally ABORTEVENT.
Reserve ABORTRUN for conditions where no event in this run can possibly
succeed (e.g., a required calibration is missing).

3.6 Check for understanding
---------------------------

  -------- ------------------------------------------------------------------------------------------------------------------------------------------------------
  **Q1**   You see two SubsysReco modules registered in a macro: A before B. Does A's process\_event return code affect B at all? What if A returns ABORTEVENT?
  -------- ------------------------------------------------------------------------------------------------------------------------------------------------------

  -------- --------------------------------------
  **Q2**   Why is findNode::getClass templated?
  -------- --------------------------------------

  -------- -----------------------------------------------------------------------------------------
  **Q3**   What happens to your histograms if you allocate them in process\_event instead of Init?
  -------- -----------------------------------------------------------------------------------------

*--- End of Week 3 reading ---*

*Now open Lab Worksheet 3.*

**Chapter 4**

**Week 4: Building with CMake and Running Fun4All**

*Module 2 --- Fun4All deep dive*

Goals for this week
-------------------

-   Write a CMakeLists.txt that produces libMyAnalysis.so.

-   Install the library to \$MYINSTALL/lib so Fun4All finds it.

-   Drive your module from a Fun4All macro and produce real output.

-   Read the DST naming convention fluently.

4.1 CMake without tears
-----------------------

CMake is a meta-build system: you describe your project, it generates
Makefiles. sPHENIX provides a package config that hides most of the
pain. Your CMakeLists.txt stays short.

> cmake\_minimum\_required(VERSION 3.0)
>
> project(MyAnalysis CXX)
>
> find\_package(sPHENIX REQUIRED)
>
> include\_directories(\${PROJECT\_SOURCE\_DIR}/)
>
> add\_library(MyAnalysis SHARED MyAnalysis.cc)
>
> target\_link\_libraries(MyAnalysis
>
> fun4all
>
> phool
>
> SubsysReco
>
> trackbase\_historic
>
> calo\_io
>
> globalvertex\_io
>
> )
>
> install(TARGETS MyAnalysis DESTINATION \${CMAKE\_INSTALL\_PREFIX}/lib)
>
> install(FILES MyAnalysis.h DESTINATION
> \${CMAKE\_INSTALL\_PREFIX}/include/myanalysis)

What each piece does: project() declares the language;
find\_package(sPHENIX) imports sPHENIX-specific find modules and compile
flags; add\_library SHARED requests a .so (shared object) library;
target\_link\_libraries lists the framework pieces your code needs;
install tells CMake where to put the output when you run make install.

4.2 Building
------------

> cd \~/software/MyAnalysis
>
> mkdir -p build && cd build
>
> cmake .. -DCMAKE\_INSTALL\_PREFIX=\$MYINSTALL
>
> make -j4 && make install
>
> \# Confirm:
>
> ls \$MYINSTALL/lib/libMyAnalysis.so

The out-of-source build (mkdir build && cd build) keeps generated files
separate from your source tree. If a build gets weird, rm -rf build and
start fresh; you'll never corrupt your source.

4.3 The Fun4All macro
---------------------

A Fun4All macro is a small ROOT macro that registers modules, tells
Fun4All where to get events from, and calls run(). Here's the minimum
viable version for your analysis.

> \#include \<fun4all/Fun4AllServer.h\>
>
> \#include \<fun4all/Fun4AllDstInputManager.h\>
>
> \#include \<myanalysis/MyAnalysis.h\>
>
> R\_\_LOAD\_LIBRARY(libfun4all.so)
>
> R\_\_LOAD\_LIBRARY(libMyAnalysis.so)
>
> void Fun4All\_MyAnalysis(int n = 1000,
>
> const std::string &in = \"DST\_TRACKS\_....root\") {
>
> Fun4AllServer \*se = Fun4AllServer::instance();
>
> se-\>Verbosity(0);
>
> MyAnalysis \*m = new MyAnalysis();
>
> m-\>set\_output\_file(\"MyAnalysis.root\");
>
> se-\>registerSubsystem(m);
>
> Fun4AllInputManager \*imgr = new Fun4AllDstInputManager(\"DSTin\");
>
> imgr-\>fileopen(in);
>
> se-\>registerInputManager(imgr);
>
> se-\>run(n);
>
> se-\>End();
>
> delete se;
>
> gSystem-\>Exit(0);
>
> }

R\_\_LOAD\_LIBRARY is ROOT's way of pre-loading a shared library so the
symbols are available. You list libfun4all.so (for the framework) and
your own libMyAnalysis.so (for your module). If you get unresolved
symbols at runtime, it's almost always a missing R\_\_LOAD\_LIBRARY line
or a library missing from \$MYINSTALL/lib.

4.4 DST naming, decoded
-----------------------

Production DST filenames look intimidating but follow a strict
convention. Learn to read them at a glance.

> DST\_CALO\_run2pp\_ana464\_2024p012-00048080-0000.root

  ---------- --------------------------------------------------------------------------------
  DST        Data Summary Tape --- a serialized node tree.
  CALO       Content: calorimeter (CALO, TRACKS, TRUTH, GLOBAL, JET, ...).
  run2pp     Collision system (run2pp = Run 2 proton--proton; run2AuAu = Run 2 gold--gold).
  ana464     sPHENIX software build tag (for reproducibility).
  2024p012   Production tag (p012 = 12th production campaign of 2024).
  00048080   Run number.
  0000       Segment number (files are split for I/O reasons).
  ---------- --------------------------------------------------------------------------------

When a senior analyst says "we're running on p012 tracks," you now know
exactly what they mean.

4.5 Check for understanding
---------------------------

  -------- ----------------------------------------------------------------------------------------------------------------------------------------------
  **Q1**   You run make install and it succeeds, but your Fun4All macro fails with "cannot open shared object file." List the three most likely causes.
  -------- ----------------------------------------------------------------------------------------------------------------------------------------------

  -------- -----------------------------------------------------------------------------------------------------------------------
  **Q2**   Your CMakeLists links against fun4all and phool but not SubsysReco. Will the build succeed? Will the runtime succeed?
  -------- -----------------------------------------------------------------------------------------------------------------------

  -------- ----------------------------------------------------------------------------------------------------------
  **Q3**   Write the file-glob for all DST\_CALO files from run 48080 of production p012 with sPHENIX build ana464.
  -------- ----------------------------------------------------------------------------------------------------------

*--- End of Week 4 reading ---*

*Now open Lab Worksheet 4.*

**Chapter 5**

**Week 5: Running the Full Simulation and Matching to Truth**

*Module 3 --- Simulation and physics*

Goals for this week
-------------------

-   Run Fun4All\_G4\_sPHENIX.C and understand each stage.

-   Generate a known physics signal (single pi0s) for efficiency
    studies.

-   Match a reconstructed object to a truth particle by ΔR.

5.1 What simulation actually does
---------------------------------

Simulation in sPHENIX is a pipeline. You choose a physics process
(Pythia, HIJING, a single particle), Geant4 propagates particles through
the detector material, a digitization step converts energy deposits into
realistic detector signals, reconstruction then runs the same code as on
real data, and the final output is a DST that looks, to your analysis
code, exactly like a data DST --- except it also carries a G4TruthInfo
node with the particles you generated.

That last sentence is the point: for efficiency and resolution studies,
you compare reconstructed to truth. Everything Week 5 teaches is
scaffolding for that one comparison.

5.2 The stock simulation macro
------------------------------

> cd \~/software && git clone
> https://github.com/sPHENIX-Collaboration/macros.git
>
> cd macros/detectors/sPHENIX/
>
> root -l -b -q \'Fun4All\_G4\_sPHENIX.C(100)\'

That produces 100 fully simulated events in a few minutes (on an
interactive node) or longer (on a slow day). The macro is a long walk
through every subsystem --- open it in an editor and skim. Don't try to
understand every line. The key is to see that each detector subsystem
registers its own Geant4 geometry and reconstruction stages.

5.3 Single-particle embedding
-----------------------------

For efficiency studies, you don't want a minimum-bias physics generator;
you want to throw exactly one known particle per event into a known
kinematic window. That's what PHG4SimpleEventGenerator does.

> \#include \<g4main/PHG4SimpleEventGenerator.h\>
>
> // \...
>
> PHG4SimpleEventGenerator \*gen = new PHG4SimpleEventGenerator();
>
> gen-\>add\_particles(\"pi0\", 1);
>
> gen-\>set\_vertex\_distribution\_mean(0, 0, 0);
>
> gen-\>set\_eta\_range(-1.1, 1.1);
>
> gen-\>set\_phi\_range(-M\_PI, M\_PI);
>
> gen-\>set\_pt\_range(1.0, 20.0);
>
> se-\>registerSubsystem(gen);

You've now injected a single pi0 per event with flat pT in \[1, 20\]
GeV/c. Run this through the full simulation and you have a clean,
labeled sample for measuring pi0 reconstruction efficiency.

5.4 Truth matching
------------------

On a simulation DST, the G4TruthInfo node holds every Monte Carlo
particle that was generated. Primary particles are the ones you threw;
secondaries are everything produced in showers. You usually want
primaries only.

> \#include \<g4main/PHG4TruthInfoContainer.h\>
>
> \#include \<g4main/PHG4Particle.h\>
>
> // In process\_event:
>
> PHG4TruthInfoContainer \*truth =
>
> findNode::getClass\<PHG4TruthInfoContainer\>(topNode,
> \"G4TruthInfo\");
>
> if (!truth) return Fun4AllReturnCodes::ABORTEVENT;
>
> auto range = truth-\>GetPrimaryParticleRange();
>
> for (auto it = range.first; it != range.second; ++it) {
>
> PHG4Particle \*p = it-\>second;
>
> int pid = p-\>get\_pid(); // PDG code (111 = pi0)
>
> float px = p-\>get\_px();
>
> float py = p-\>get\_py();
>
> float pz = p-\>get\_pz();
>
> float e = p-\>get\_e();
>
> // \...
>
> }

ΔR matching: for each reconstructed object, compute Δη and Δφ relative
to every truth particle, and pick the truth with smallest ΔR = √(Δη² +
Δφ²). If the minimum ΔR is below a threshold (typically 0.02--0.05 for
calorimeter clusters, 0.01 for tracks), you call it a match. Anything
above is an unmatched (likely combinatoric) reconstructed object.

5.5 Check for understanding
---------------------------

  -------- --------------------------------------------------------------------------------------------------
  **Q1**   Why do we use primary particles, not all G4 particles, when measuring reconstruction efficiency?
  -------- --------------------------------------------------------------------------------------------------

  -------- ------------------------------------------------------
  **Q2**   What are the PDG codes for photon, pi0, eta, proton?
  -------- ------------------------------------------------------

  -------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Q3**   If you throw 1000 pi0s with a flat pT spectrum from 1 to 20 GeV/c, and you see 900 reconstructed photon pairs with mass in the pi0 window, what's your raw efficiency (ignoring acceptance)? What's wrong with quoting that number as the efficiency?
  -------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*--- End of Week 5 reading ---*

*Now open Lab Worksheet 5.*

**Chapter 6**

**Week 6: Calorimeter Analysis and the Pi0 Peak**

*Module 3 --- Simulation and physics*

Goals for this week
-------------------

-   Access EMCal clusters and apply cluster-quality cuts.

-   Reconstruct pi0 → γγ from photon pairs.

-   Understand centrality in heavy-ion events.

6.1 Calorimeter clusters
------------------------

EMCal clusters live on the CLUSTER\_CEMC node. Each cluster has energy,
position (usually in eta/phi), a shower-shape variable (chi2), and
metadata about constituent towers. The two cuts you will apply almost
always:

-   Minimum energy: typical threshold is 0.3 GeV. Below that, noise
    dominates.

-   Shower-shape chi2: typical cut at chi2 \< 4. Hadrons deposit energy
    with a wider shower shape than photons, so a chi2 cut strongly
    suppresses them.

> RawClusterContainer \*clusters =
>
> findNode::getClass\<RawClusterContainer\>(topNode, \"CLUSTER\_CEMC\");
>
> if (!clusters) return Fun4AllReturnCodes::ABORTEVENT;
>
> for (auto &it : clusters-\>getClustersMap()) {
>
> RawCluster \*c = it.second;
>
> float e = c-\>get\_energy();
>
> float eta = c-\>get\_eta();
>
> float phi = c-\>get\_phi();
>
> float chi2 = c-\>get\_chi2();
>
> if (e \< 0.3) continue;
>
> if (chi2 \> 4.0) continue;
>
> // \... use this cluster
>
> }

6.2 The pi0 invariant mass
--------------------------

Neutral pions decay 99% of the time into two photons. In the detector
rest frame, those two photons carry back-to-back momentum (in the pi0
rest frame) boosted into the lab. Their invariant mass is preserved, so
if you pair up photons in the EMCal and compute M = √((E1+E2)² −
\|p'1+p'2\|²), real pi0 decays pile up at 135 MeV.

+----------------------------------------------------------------------+
| **A useful shortcut**                                                |
|                                                                      |
| For massless photons, M² = 2E₁E₂(1 − cosθ). Equivalently, ROOT's     |
| TLorentzVector will do the bookkeeping. Use it --- your analysis     |
| will be cleaner.                                                     |
+----------------------------------------------------------------------+

> std::vector\<TLorentzVector\> photons;
>
> for (auto &it : clusters-\>getClustersMap()) {
>
> RawCluster \*c = it.second;
>
> if (c-\>get\_energy() \< 0.3) continue;
>
> if (c-\>get\_chi2() \> 4.0) continue;
>
> float e = c-\>get\_energy();
>
> float eta = c-\>get\_eta();
>
> float phi = c-\>get\_phi();
>
> float pt = e / std::cosh(eta);
>
> TLorentzVector v; v.SetPtEtaPhiE(pt, eta, phi, e);
>
> photons.push\_back(v);
>
> }
>
> for (size\_t i = 0; i \< photons.size(); ++i) {
>
> for (size\_t j = i + 1; j \< photons.size(); ++j) {
>
> TLorentzVector pair = photons\[i\] + photons\[j\];
>
> float mass = pair.M();
>
> float pt = pair.Pt();
>
> \_h\_mgg-\>Fill(mass);
>
> if (mass \> 0.10 && mass \< 0.17) \_h\_pi0\_pt-\>Fill(pt);
>
> }
>
> }

Run this on a simulation sample with a modest pi0 yield and you should
see a clear peak near 0.135 GeV. On real heavy-ion data the peak is
there too but under a much larger combinatorial background.

6.3 Backgrounds and event mixing (preview for Week 8)
-----------------------------------------------------

Every photon-photon pair is entered in the mass histogram --- the real
pi0 contributions sit on top of a huge combinatorial bump from photon
pairs that did not originate from the same pi0. Week 8 covers event
mixing, the standard technique for estimating that background.

6.4 Centrality
--------------

In Au+Au collisions, how "head-on" the nuclei collide is called
centrality. It's quantified by the forward multiplicity (MBD or ZDC).
Conventionally 0--10% is the most central and 90--100% the most
peripheral. The CentralityInfo node encodes this.

> \#include \<centrality/CentralityInfo.h\>
>
> CentralityInfo \*cent = findNode::getClass\<CentralityInfo\>(topNode,
> \"CentralityInfo\");
>
> if (cent) {
>
> float c = cent-\>get\_centile(CentralityInfo::PROP::mbd\_NS);
>
> // c is in \[0, 100\]; 0 = most central, 100 = peripheral
>
> }

Almost every QGP observable depends on centrality, so splitting your pi0
histogram into central and peripheral bins is usually the first thing
you do on a heavy-ion sample.

6.5 Check for understanding
---------------------------

  -------- -----------------------------------------------------------
  **Q1**   Why does the chi2 cut suppress hadrons more than photons?
  -------- -----------------------------------------------------------

  -------- ---------------------------------------------------------------------------------
  **Q2**   What is the physical origin of the combinatorial background under the pi0 peak?
  -------- ---------------------------------------------------------------------------------

  -------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Q3**   You quote your pi0 yield in "most central" events. A reviewer asks whether you mean 0--10% or 0--20%. Why does the answer affect your signal-to-background ratio?
  -------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------

*--- End of Week 6 reading ---*

*Now open Lab Worksheet 6.*

**Chapter 7**

**Week 7: Condor: Running at Scale**

*Module 4 --- Production*

Goals for this week
-------------------

-   Write a Condor submit file and a wrapper script.

-   Use file lists to parallelize over many DSTs.

-   Diagnose held jobs from the command line.

-   Merge outputs from a job array.

7.1 Why Condor
--------------

One DST file holds a few thousand events. A full production run has tens
of thousands of files. No interactive macro will chew through that on a
single node in the time you have. HTCondor is a batch scheduler: you
submit a job description, Condor assigns each job to a worker node, and
you harvest the outputs.

7.2 The three files you need
----------------------------

Every Condor workflow has the same three files. Learn them once, use
them forever.

-   filelist.txt --- one input DST per line. This drives the
    parallelism.

-   run\_analysis.sh --- a bash wrapper that sets up the environment and
    runs the macro.

-   myJob.sub --- the Condor submit description that ties them together.

The wrapper:

> \#!/bin/bash
>
> source /opt/sphenix/core/bin/sphenix\_setup.sh -n new
>
> export MYINSTALL=/sphenix/user/\$USER/install
>
> source /opt/sphenix/core/bin/setup\_local.sh \$MYINSTALL
>
> PROCESS=\$1
>
> INPUTFILE=\$(sed -n \"\$((PROCESS+1))p\" filelist.txt)
>
> OUTDIR=/sphenix/user/\$USER/analysis\_output
>
> mkdir -p \$OUTDIR
>
> cd \$OUTDIR
>
> root -l -b -q \"\$HOME/Week7/Fun4All\_MyAnalysis.C(0,
> \\\"\${INPUTFILE}\\\", \${PROCESS})\"

The submit file:

> Universe = vanilla
>
> Executable = /usr/bin/bash
>
> Arguments = /sphenix/user/\$ENV(USER)/Week7/run\_analysis.sh
> \$(Process)
>
> Output =
> /sphenix/user/\$ENV(USER)/Week7/condorOut/out\_\$(Process).out
>
> Error = /sphenix/user/\$ENV(USER)/Week7/condorOut/err\_\$(Process).err
>
> Log = /sphenix/user/\$ENV(USER)/Week7/condorOut/log\_\$(Process).log
>
> request\_memory = 4096MB
>
> PeriodicHold = (NumJobStarts \>= 1 && JobStatus == 1)
>
> Queue 50

Queue 50 launches 50 jobs, each getting \$(Process) from 0 to 49. The
wrapper uses that process number to pick the matching line from
filelist.txt. Simple, robust, stateless.

7.3 Monitoring and triage
-------------------------

  --------------------------------- -------------------------------------------------
  condor\_submit myJob.sub          Submit the jobs.
  condor\_q                         See your jobs (queued / running / held).
  condor\_q -hold                   Just the held ones.
  condor\_q -better-analyze JOBID   Explain why a job is held or idle.
  condor\_tail JOBID                Stream the stdout of a running job.
  condor\_rm JOBID                  Remove a job (or pass \$USER to wipe them all).
  --------------------------------- -------------------------------------------------

The most common reasons for held jobs: memory exceeded, input file not
found, environment not sourced in the wrapper. All three are fixable by
reading the .err file and the condor\_q -better-analyze output.

7.4 Merging
-----------

When all jobs finish, each one has produced MyAnalysis\_\<Process\>.root
in \$OUTDIR. ROOT's hadd merges them:

> hadd MyAnalysis\_merged.root
> /sphenix/user/\$USER/analysis\_output/MyAnalysis\_\*.root

hadd understands TH1, TH2, TTree and concatenates intelligently. Check
that the event counter in the merged file equals the sum of counters
from the individual files.

7.5 Check for understanding
---------------------------

  -------- -------------------------------------------------------------------------------
  **Q1**   Your filelist.txt has 100 lines. You Queue 50. What happens to lines 51--100?
  -------- -------------------------------------------------------------------------------

  -------- ------------------------------------------------------------------------------------------------------
  **Q2**   A job is held with "memory usage 4600 MB". Which line of the submit file do you change, and to what?
  -------- ------------------------------------------------------------------------------------------------------

  -------- ----------------------------------------------------------------------------
  **Q3**   Why does hadd work for TH1 and TTree but not, in general, for a node tree?
  -------- ----------------------------------------------------------------------------

*--- End of Week 7 reading ---*

*Now open Lab Worksheet 7.*

**Chapter 8**

**Week 8: Event Mixing, Publication Plots, and Contributing Back**

*Module 4 --- Production*

Goals for this week
-------------------

-   Implement event-mixing for combinatorial background.

-   Produce a sPHENIX-styled plot suitable for a working-group talk.

-   Open a pull request on the sPHENIX analysis repository.

8.1 Why event mixing
--------------------

Every photon pair in an event contributes to your invariant mass
histogram. Most of those pairs are not from the same pi0; they are
combinatorial background. To subtract that background, you need a shape
that describes it. Fitting a polynomial under the peak is a rough first
step, but has systematics. The standard technique is event mixing: pair
photons from different events. Those pairs are guaranteed not to be from
the same pi0, so their mass distribution is purely combinatorial.
Normalize and subtract, and you're left with the signal.

8.2 Implementing mixing
-----------------------

Hold a small buffer of photon lists from recent events. For each new
event, compute the same-event pair mass (as usual) and also the
mixed-event mass by pairing current photons with photons from buffer
events.

> \#include \<deque\>
>
> std::deque\<std::vector\<TLorentzVector\>\> \_mix;
>
> const size\_t MIX\_DEPTH = 10;
>
> // inside process\_event, after collecting current-event photons
> \"cur\":
>
> for (auto &prev : \_mix) {
>
> for (auto &g1 : cur) for (auto &g2 : prev) {
>
> TLorentzVector pair = g1 + g2;
>
> \_h\_mgg\_mixed-\>Fill(pair.M());
>
> }
>
> }
>
> \_mix.push\_back(cur);
>
> if (\_mix.size() \> MIX\_DEPTH) \_mix.pop\_front();

A subtlety: for heavy-ion analyses, mix events in centrality and
vertex-z bins, not globally. The shape of the combinatorial background
depends on multiplicity, so mixing a central event with a peripheral one
distorts the background estimate.

8.3 sPHENIX plot style
----------------------

Any plot that leaves your laptop should look like sPHENIX. That means:
clean axis labels with units, margins large enough to fit them, no title
(the label text takes its place), and a clear "sPHENIX Internal"
watermark unless the plot has been formally approved.

> void makePlot() {
>
> gStyle-\>SetOptStat(0);
>
> gStyle-\>SetOptTitle(0);
>
> TCanvas \*c = new TCanvas(\"c\", \"\", 800, 600);
>
> c-\>SetLeftMargin(0.15); c-\>SetBottomMargin(0.12);
>
> c-\>SetRightMargin(0.05); c-\>SetTopMargin(0.05);
>
> h-\>GetXaxis()-\>SetTitle(\"p\_{T} \[GeV/c\]\");
>
> h-\>GetYaxis()-\>SetTitle(\"dN/dp\_{T} \[(GeV/c)\^{-1}\]\");
>
> h-\>SetLineColor(kBlue); h-\>SetLineWidth(2);
>
> h-\>Draw();
>
> TLatex \*tex = new TLatex();
>
> tex-\>SetNDC(); tex-\>SetTextFont(42); tex-\>SetTextSize(0.04);
>
> tex-\>DrawLatex(0.20, 0.88, \"\#it{sPHENIX} Internal\");
>
> tex-\>DrawLatex(0.20, 0.83, \"Au+Au \#sqrt{s\_{NN}} = 200 GeV\");
>
> c-\>SaveAs(\"pt\_spectrum.pdf\");
>
> }

8.4 Contributing back
---------------------

The final checkpoint is to contribute your pipeline to the sPHENIX
analysis repo. This teaches a skill you'll use for every future
contribution:

-   Fork the analysis repo.

-   Create a branch with a descriptive name
    (feature/pi0-production-week8-\<user\>).

-   Add your code in a subdirectory under
    analysis/\<subsystem\>/\<your-name\>/.

-   Write a README explaining inputs, how to build, how to run, how to
    reproduce the final plot.

-   Open a pull request against upstream/master. Tag your mentor as a
    reviewer.

Review is the education. Your reviewer will comment on everything from
"this header include is unnecessary" to "your centrality binning will
bias the result for peripheral events." Read every comment, fix what
applies, push, and repeat. That's how professional physics code gets
written.

8.5 Check for understanding
---------------------------

  -------- ---------------------------------------------------------------------------------------------------------
  **Q1**   You use event mixing without binning in centrality. What bias do you introduce, and in which direction?
  -------- ---------------------------------------------------------------------------------------------------------

  -------- ----------------------------------------
  **Q2**   Why \#it{sPHENIX} rather than sPHENIX?
  -------- ----------------------------------------

  -------- ---------------------------------------------------------------------------------------------
  **Q3**   When you open the PR, what three things should your README always answer on the first page?
  -------- ---------------------------------------------------------------------------------------------

*--- End of Week 8 reading ---*

*Now open Lab Worksheet 8.*

Colophon
========

This Reader was produced as a companion to the sPHENIX 8-Week Onboarding
Syllabus. Body text is set in Calibri; headings in Arial; code in
Consolas. The palette is deep navy (\#1E2761) and garnet (\#8A1C1C).

The authoritative upstream for the sPHENIX codebase is
github.com/sPHENIX-Collaboration. Corrections and improvements to this
Reader are welcome --- open a PR in your fork of the training repo and
tag your mentor.

*Good luck, and welcome to the collaboration.*
