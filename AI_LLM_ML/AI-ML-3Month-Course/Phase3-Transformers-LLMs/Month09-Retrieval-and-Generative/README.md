# Month 09 — Retrieval & Generative Models II

Two threads converge on Capstone 3. Thread one is retrieval: embeddings and how
contrastive training shapes them (Week 33), then a full RAG system over the sPHENIX TDR
and your Zotero library, with retrieval evals so you know it works (Week 34). Thread
two is generation: the complete DDPM derivation and conditional diffusion, connected to
the Langevin dynamics you already know (Week 35).

Week 36 is the capstone fork: (a) a physics-literature assistant — Week 34's RAG plus
the Week 32 extractor, evaluated on a held-out Q&A set — or (b) a conditional diffusion
calo-shower generator validated head-to-head against your Capstone 2 VAE. Both paths
close the Phase 3 gate: fine-tuned checkpoint + retrieval or generative system callable
from one script, benchmark table, and the attention derivation re-done cold.

**Month-end deliverable:** the Capstone 3 repo — tested, reproducible (`uv sync` + one
command), with an honest writeup including what failed first.

**Sign-off:** tag `month-09-complete`, write `retro.md` (250 words) in this folder, and
open one open-question issue. Close last month's. Then take the Phase 3 gate check in
`00-Syllabus.md` §5 seriously before starting Phase 4.
