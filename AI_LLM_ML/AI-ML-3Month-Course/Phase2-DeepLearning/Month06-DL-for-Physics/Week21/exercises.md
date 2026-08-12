# Week 21 — Exercises

Work top to bottom. Setup (imports, data loading, constants, plot axes) is given by
the notebook; you write only the lines each exercise asks for. All exercises live in
the notebook except E7, whose deliverables are two markdown files in this folder.

Data: E1–E3 use small arrays built in the notebook. E4–E6 use the Week-20 toy
generator's clusters converted to point clouds (the notebook imports your Week-20
`src/`); if you want real jets instead, the HLS4ML LHC jet-tagging dataset or the
Top Quark Tagging Reference Dataset (both on Zenodo — search those names) are
drop-in alternatives with per-particle kinematics, and the notebook setup notes the
column mapping.

## E1 — Permutation stress test

On a toy set-classification task (label = whether a 16-point set contains a dense
sub-cluster), train (a) an MLP on the flattened, padded point list and (b) a Deep
Sets model: shared per-point MLP, sum pool, output MLP. Evaluate both on the test
set as-is and again with every event's point order shuffled.
Hint: the Deep Sets forward is three lines: `phi(x)`, `.sum(dim=1)`, `rho(...)`.
Accept when: printed test accuracies show the Deep Sets model identical under
shuffling (exactly, to float tolerance) and the flattened MLP degraded by ≥ 5
points.

## E2 — Message passing by hand

For the lesson §7 three-node graph (given weights, given features), compute one
EdgeConv update on paper: all six messages, the max-aggregation per node, the
output embeddings. Then reproduce it in plain PyTorch (no PyG) with loops over the
edge list.
Hint: work in a table — one row per edge, columns for $h_i$, $h_j - h_i$, MLP out.
Accept when: your loop implementation matches your paper numbers to 1e-6.

## E3 — The same layer in PyG

Implement the layer as a `MessagePassing` subclass (lesson §7 code as the
template, but write `message()` yourself) and run it on a PyG mini-batch of several
graphs built with `Batch.from_data_list`.
Accept when: on the E2 graph it matches E2 to 1e-6, and on the batch the output has
one row per node with no cross-event edges (checked by the provided assert on
`batch`).

## E4 — Clusters to point clouds

Convert Week-20 clusters to graphs: nodes = towers above threshold with features
`[log E, eta, phi]`, edges from `knn_graph` on $(\eta, \phi)$. Draw two event
displays (one photon, one merged-π⁰) with nodes sized by energy and edges drawn,
for k = 4 and k = 8.
Hint: the notebook gives the plotting scaffold; you fill node positions, sizes, and
the edge segments from `edge_index`.
Accept when: both displays exist and the k = 4 graph is connected within each
shower core (visually: no isolated high-energy tower).

## E5 — GNN classifier vs the Week-20 CNN

Train a 2–3 block EdgeConv classifier (global mean pool + MLP head) on the Week-20
photon/π⁰ splits — same splits, same seeds. Report test ROC AUC next to your
Week-20 CNN's AUC. Then rerun once with sum instead of max/mean aggregation and
report both.
Hint: reuse the Week-20 training loop; only the model and the data objects change.
Accept when: GNN test AUC is within 0.03 of the CNN (or above it), and the
aggregation ablation table prints both AUCs.

## E6 — Oversmoothing probe

For the E5 model rebuilt with 1, 2, 4, 8 message-passing rounds (weights shared or
fresh, your choice — say which), compute the mean pairwise variance of node
embeddings per event at the last layer, averaged over 100 test events. Plot
variance vs rounds.
Accept when: the plot exists with a one-line printed reading (does the spread
collapse, and by how much from 1 to 8 rounds).

## E7 — Paper notes

Write ~half a page each on ParticleNet and Particle Transformer: the claim, the
representation choice, the baseline it beat, and one criticism or open question.
Save as `notes_particlenet.md` and `notes_part.md` in this folder.
Accept when: both files exist and each names the paper's baseline and one
criticism.

## Review

1. (Week 17) A convolution's parameter sharing encodes which symmetry? State the
   analogous statement for a shared message function plus sum aggregation.
2. (Week 13) In one line: what does the backward pass compute, and for what
   purpose?
3. (Week 08) Write the definition of KL divergence KL(p‖q) and state its sign. You
   will use it heavily next week.
4. (Week 10) Why is ROC AUC preferred over accuracy for the photon/π⁰ problem at
   high energy, where the classes are imbalanced?
5. (Week 04) Name the two commands a fresh clone of your Week-20 repo needs to
   reproduce its headline number.
