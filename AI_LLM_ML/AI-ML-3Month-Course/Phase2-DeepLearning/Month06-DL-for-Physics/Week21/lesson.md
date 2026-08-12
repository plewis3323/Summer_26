# Week 21 — Graph Neural Networks

~3 hrs reading. Before starting you should be able to: build and train a CNN in
PyTorch (Weeks 15, 17); explain why convolution's parameter sharing encodes
translation symmetry (Week 17); generate and classify the toy calorimeter images
from the Week-20 project (Week 20); and describe attention as content-based lookup,
at intuition level (Week 19).

## 1. An event is not an image

The Week-20 project worked because a calorimeter cluster naturally *is* an image: a
fixed grid of towers, each with an energy. Most collider data is not like that.

Two examples you will meet all through HEP-ML:

**Jets.** Protons are made of quarks and gluons. Quarks and gluons cannot exist on
their own — the strong force confines them — so when a collision knocks one out, it
converts its energy into a narrow spray of ordinary particles (pions, kaons,
protons, ...) all flying in roughly the same direction. That spray is called a
**jet**. "Jet tagging" asks: which kind of particle started this spray? A jet from a
top quark looks subtly different from a jet from a gluon, and telling them apart is
one of the classic ML problems in particle physics. A jet is a *list of particles*:
each particle has a momentum and maybe a charge and a type, and different jets
contain different numbers of particles — 10 in one event, 60 in the next.

**Tracking.** Charged particles curve through a detector's magnetic field, and thin
sensor layers record a **hit** — a position measurement — wherever a particle
crosses them. **Tracking** is the job of connecting thousands of hits per event into
the handful of curved trajectories (tracks) that made them. The input is a *cloud of
3D points*, again with no fixed count and no natural order.

Both inputs share three properties that break the CNN template:

1. **Variable length.** Events have different numbers of particles/hits.
2. **No grid.** There is no pixel lattice; points sit at arbitrary positions.
3. **No order.** The particle list is stored in *some* order, but the physics does
   not depend on it. Swap rows 3 and 7 of the particle list and it is the same jet.

Property 3 is the one we will engineer into the architecture, so give it a name.

**Permutation.** A permutation is a reordering. For a list of $n$ items, a
permutation $\pi$ is a rule sending position $i$ to position $\pi(i)$. Applying a
permutation to the rows of a matrix $X$ can be written $PX$, where $P$ is a
**permutation matrix**: the identity matrix with its rows shuffled the same way.

A function $f$ on a set of particle feature vectors $\{x_1, \dots, x_n\}$ is
**permutation invariant** if reordering the inputs never changes the output:

$$f(x_{\pi(1)}, \dots, x_{\pi(n)}) = f(x_1, \dots, x_n) \quad \text{for every permutation } \pi.$$

A per-particle function $g$ (one output vector per input particle) is **permutation
equivariant** if reordering the inputs just reorders the outputs the same way:

$$g(PX) = P\,g(X).$$

Invariant: the answer ignores order. Equivariant: the answer moves *with* the order.
A jet classifier should be invariant. The layers inside it, which still produce one
vector per particle, should be equivariant — then a final symmetric pooling step
makes the whole thing invariant. That two-part recipe is this entire week.

## 2. Why not just flatten and pad?

The tempting hack: pad every event to 60 particles, flatten to one long vector, feed
an MLP. Two things go wrong.

First, the MLP has separate weights for "particle slot 1" and "particle slot 7", so
it must *learn* that the slots are interchangeable — burning capacity and data on a
fact we know for free. Compare Week 17: a fully connected layer on an image must
learn translation symmetry, while a convolution gets it for free by weight sharing.
Same story, different symmetry.

Second, it isn't robust. Shuffle the test-time particle order and the flattened
MLP's prediction changes; the physics didn't. Exercise E1 makes you watch this
happen with numbers.

The fix mirrors the CNN fix: build the symmetry into the architecture. For
translations, share weights across positions (convolution). For permutations, share
weights across particles and combine them with an operation that has no notion of
order.

## 3. The simplest invariant model: Deep Sets

Which operations ignore order? Sum, mean, max — applied elementwise over the set.
$x_1 + x_2 + x_3$ equals $x_3 + x_1 + x_2$. These are called **symmetric
aggregations**.

The Deep Sets recipe:

$$f(\{x_1,\dots,x_n\}) = \rho\left(\sum_{i=1}^{n} \phi(x_i)\right)$$

where $\phi$ is an MLP applied to each particle *with the same weights* (that is the
weight sharing), the sum is over particles, and $\rho$ is a second MLP applied to
the pooled vector. Symbols: $x_i$ is particle $i$'s feature vector, $\phi$ maps each
particle to an embedding, $\rho$ maps the pooled embedding to the output.

Invariance is immediate: the sum ignores order, and everything after the sum only
sees the sum. There is also a theorem (Zaheer et al., "Deep Sets") that *any*
well-behaved permutation-invariant function can be written this way for large enough
embeddings — so nothing is lost in principle.

What *is* lost in practice: $\phi$ looks at one particle at a time. It cannot use
relations *between* particles — "these two hits are close", "these two particles
share most of the jet's momentum". Relations are exactly what jets and tracks are
made of. To use them, we need a structure that says which pairs to relate: a graph.

## 4. Graphs, and how to build one from an event

A **graph** is a set of **nodes** (things) and **edges** (pairwise connections
between things). For us: nodes = particles or hits, each carrying a feature vector;
an edge $(j \to i)$ means "node $j$ is allowed to send information to node $i$". The
**neighborhood** $\mathcal{N}(i)$ is the set of nodes with an edge into $i$.

Code stores the edges as an **edge index**: a $2 \times E$ integer array whose
columns are (source, target) pairs, where $E$ counts the edges.

Detector data does not come with edges — we choose them. The standard choice is a
**k-nearest-neighbors (kNN) graph**: connect each node to the $k$ nodes closest to
it in some coordinate space. For jets the space is usually the angular position of
each particle, written $(\eta, \phi)$:

- $\phi$ (azimuth) is the angle around the beam pipe, like longitude.
- $\eta$ (**pseudorapidity**) measures the angle from the beam axis, remapped so
  that equal steps in $\eta$ are equally "expensive" for particle production;
  $\eta = 0$ is perpendicular to the beam, large $|\eta|$ is near the beam.
  Distances in the $(\eta, \phi)$ plane, $\Delta R = \sqrt{\Delta\eta^2 + \Delta\phi^2}$,
  are the standard "how far apart are these two particles" measure at a collider.

Two design decisions to remember:

- **Static vs dynamic graphs.** Static: build the kNN graph once, from the input
  coordinates, and keep it for all layers. Dynamic: rebuild kNN *in the current
  embedding space* before each layer, so "neighbors" means "similar according to
  what the network has learned so far". Dynamic graphs (the EdgeConv/ParticleNet
  choice) let the network group particles by learned similarity — the same
  content-based flavor as the attention lookup you met in Week 19.
- **k** trades locality against compute: larger $k$ mixes information faster but
  costs more edges and can blur local structure.

## 5. Message passing, derived

Now the general layer. We want an update rule that (a) produces one new vector per
node — so it can be stacked, (b) uses neighbor relations, (c) is permutation
equivariant. Build it in three steps, guided by the constraints.

**Step 1 — message.** For each edge $(j \to i)$, compute what $j$ tells $i$:

$$m_{ij} = M\big(h_i, h_j, e_{ij}\big)$$

Symbols: $h_i$ is node $i$'s current embedding ("h" for hidden), $e_{ij}$ is an
optional edge feature (for example $\Delta R$ between the two particles), and $M$ is
a small learned network — the **message function**. Crucially, the *same* $M$ is
used on every edge: that is the weight sharing, exactly as one convolution kernel is
reused at every image position.

**Step 2 — aggregate.** Node $i$ receives one message per neighbor, and its
neighbors have no order. So the messages must be combined by a symmetric
aggregation:

$$a_i = \bigoplus_{j \in \mathcal{N}(i)} m_{ij}, \qquad \bigoplus \in \{\text{sum, mean, max, ...}\}$$

This is the load-bearing step. If we combined messages in a way that cared about
order (say, concatenation), the layer would depend on how the particle list was
stored, and the symmetry would be gone.

**Step 3 — update.** Combine what $i$ knew with what it heard:

$$h_i' = U\big(h_i, a_i\big)$$

with $U$ another small learned network, again shared across nodes. One
message-passing layer = message, aggregate, update. Stack $L$ layers and information
propagates $L$ hops across the graph.

### 5.1 The permutation proof

Claim 1: *one message-passing layer is permutation equivariant.*

Relabel the nodes by a permutation $\pi$: node $i$ in the new labeling is node
$\pi(i)$ in the old one, and the edges are relabeled consistently (that is what
"same graph, new labels" means — $PAP^\top$ if you store edges as an adjacency
matrix $A$, the $n \times n$ 0/1 matrix with a 1 where an edge exists).

Look at what the layer computes for the node that used to be called $i$. Its
embedding $h_i$ is the same vector, just stored in a new row. Its neighborhood
contains the same physical nodes, relabeled. So the *set* of messages
$\{M(h_i, h_j, e_{ij}) : j \in \mathcal{N}(i)\}$ is the same set — $M$ is shared, so
no message changed; only the labels on them changed. A symmetric $\bigoplus$ of the
same multiset gives the same $a_i$, and then $U(h_i, a_i)$ gives the same output
vector. So every node gets the same new embedding it would have gotten before
relabeling; the embeddings are simply stored in permuted rows. That is precisely
$g(PX) = P\,g(X)$. ∎

Claim 2: *equivariant layers followed by a symmetric readout are invariant.*

Let the readout be $\hat{y} = \rho\big(\bigoplus_i h_i^{(L)}\big)$ — pool the final
node embeddings with a symmetric aggregation, then apply an MLP $\rho$. Feed in a
permuted event: by Claim 1 applied $L$ times, the final embeddings are the same
vectors in permuted rows, $P H^{(L)}$. The pooling sums (or averages, or maxes) over
rows, and a symmetric operation over permuted rows gives the identical result. So
$\hat{y}$ is unchanged: the classifier is permutation invariant. ∎

This proof is short but it is *the* design principle of the field: check every
operation in your model against "does it commute with relabeling?" — anything that
doesn't (concatenating node lists, indexing "the first particle", flattening) breaks
the guarantee.

### 5.2 EdgeConv: a concrete message function

The layer used by ParticleNet is **EdgeConv**:

$$h_i' = \max_{j \in \mathcal{N}(i)} \, \text{MLP}\big(h_i \,\|\, (h_j - h_i)\big)$$

where $\|$ means concatenating the two vectors into one longer vector, and the max
is taken elementwise over neighbors. The message function sees the *center* $h_i$
(global context: where am I?) and the *difference* $h_j - h_i$ (local context: how
does my neighbor differ from me?). Using differences makes the message depend only
on relative structure — the same instinct as building features from $\Delta\eta$,
$\Delta\phi$ rather than absolute positions.

### 5.3 CNNs are message passing on a grid

Sanity check that this framework generalizes what you know. Take a 1D "image" and
build a line graph: pixel $i$ has neighbors $\{i-1, i, i+1\}$. A convolution with a
size-3 kernel $(w_{-1}, w_0, w_{+1})$ computes $h_i' = \sum_{d=-1}^{1} w_d\, x_{i+d}$.
That is message passing where the message from neighbor $j$ is $m_{ij} = w_{j-i}\,x_j$
— a *position-dependent* message, allowed because the grid gives every neighbor a
well-defined relative position — and the aggregation is a sum. Convolution is
message passing that spends the grid's extra structure on sharper messages. Remove
the grid and you must make the messages position-independent (or learn edge
features); that is exactly the step from CNNs to GNNs.

## 6. Design choices and one failure mode

**Aggregation.** Sum preserves multiplicity (10 soft particles ≠ 1 soft particle);
mean is multiplicity-blind but scale-stable; max picks the single most activated
neighbor, good for "does any neighbor look like X" features. ParticleNet uses mean
in its EdgeConv blocks; Deep Sets papers argue for sum's expressiveness. Try more
than one — it is a one-line change (Exercise E5's ablation).

**Depth and oversmoothing.** Each layer mixes each node with its neighbors. Stack
many layers and every node's embedding becomes a blend of the whole graph — node
embeddings converge toward each other and the per-node information washes out. This
is **oversmoothing**, and it is why point-cloud GNNs are usually shallow (3–4
message-passing blocks) compared to CNNs. You will measure it directly in E6 by
tracking the spread between node embeddings as rounds of message passing increase.

**Where does the graph come from?** kNN is a choice, not a truth. Too-small $k$
fragments the event; too-large $k$ approaches a fully connected graph, at which
point you have effectively rebuilt attention (Week 25 will make that connection
precise).

## 7. PyTorch Geometric

**PyTorch Geometric (PyG)** is the standard PyTorch library for GNNs. Install with
`uv add torch-geometric`. Three things cover most usage:

**1. The `Data` object.** One graph = node features `x` (shape `[num_nodes,
num_features]`), the `edge_index` (shape `[2, num_edges]`), and optionally a label
`y`.

**2. Batching.** CNNs batch by stacking equal-size images. Graphs have different
sizes, so PyG batches by *disjoint union*: pack all nodes from all graphs into one
big graph with no edges between events, and keep a `batch` vector saying which graph
each node belongs to. Message passing never crosses events (no edges cross), and
`global_mean_pool(h, batch)` pools each event separately. One clever trick replaces
all the padding bookkeeping.

**3. The `MessagePassing` base class.** You write `message()` (step 1); the base
class does aggregate (step 2, chosen by `aggr=`) and hands you the result to
combine (step 3). Classes and inheritance appeared in Week 15 with `nn.Module`;
`MessagePassing` is an `nn.Module` with the aggregation plumbing built in.

Here is a complete, runnable EdgeConv on a toy 3-node graph — the same computation
Exercise E2 asks you to do by hand first:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: receiving node's features, one row per edge
        # x_j: sending node's features, one row per edge
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=1))


torch.manual_seed(0)
x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 nodes, 2 features
# directed edges: 1->0, 2->0, 0->1, 2->1, 0->2, 1->2  (fully connected)
edge_index = torch.tensor([[1, 2, 0, 2, 0, 1],
                           [0, 0, 1, 1, 2, 2]])

layer = EdgeConv(2, 4)
h = layer(x, edge_index)
print(h.shape)   # torch.Size([3, 4])
print(h)

# permutation check: relabel nodes (0,1,2) -> (2,0,1) and re-run
perm = torch.tensor([2, 0, 1])
inv = torch.argsort(perm)
x_p = x[perm]
edge_index_p = inv[edge_index]
h_p = layer(x_p, edge_index_p)
print(torch.allclose(h_p, h[perm], atol=1e-6))  # True: equivariant
```

Read the permutation check until it is obvious: we shuffle the node rows, relabel
the edges consistently, and the outputs come back as the same rows shuffled the same
way — Claim 1, verified numerically.

## 8. The literature you are joining

**ParticleNet** (Qu & Gouskos, arXiv:1902.08570). Represents a jet as a "particle
cloud" — exactly our point-cloud-with-kNN picture — and stacks dynamic EdgeConv
blocks. Its gains over image-based CNNs on jet tagging came less from architecture
cleverness than from the *representation*: no pixelation losses, natural handling of
variable multiplicity, symmetry built in. When you read it, grade it as a physics
paper: what baselines, what dataset, what actually moved the number.

**Particle Transformer / ParT** (Qu, Li & Qian, arXiv:2202.03772). Replaces kNN
message passing with attention over all particle pairs, and injects physics into the
attention scores as pairwise features (functions of $\Delta R$, momentum fractions,
pair mass). Two lessons: attention is message passing on the fully connected graph,
and hand-crafted pairwise physics features still pay. It also introduced JetClass,
a 100M-jet dataset — a scaling-laws story you will recognize in Phase 3.

**Exa.TrkX** (GNN tracking). Tracking recast as **edge classification**: nodes are
hits, candidate edges connect hits that could plausibly be consecutive points on a
track, and the GNN scores each edge as "same track" or not; trajectories are then
read off as connected chains of accepted edges. Notice the shift — Week 20 was
graph-level classification (one label per event), this is edge-level (one label per
edge). Same message-passing machinery, different readout head.

## 9. Worked example: photon vs merged-π⁰, as a point cloud

Tie it together on Week-20's problem. There, a cluster was an 8×8 tower image. The
point-cloud view: keep only towers above a small noise threshold and make each a
node with features $(E, \eta, \phi)$ — energy and position. A low-multiplicity
cluster becomes a 6-node graph instead of a mostly-zero 64-pixel image.

The pipeline (this is Exercises E4–E5, sketched):

1. **Nodes.** For each cluster, select towers with $E$ above threshold; features
   `[E, eta, phi]`, with $E$ log-scaled (energies span decades; Week 16's input-
   scaling lesson applies).
2. **Edges.** `knn_graph(pos, k=4)` on the $(\eta, \phi)$ coordinates.
3. **Model.** Two or three EdgeConv blocks, `global_mean_pool`, then a small MLP
   head to 2 classes — the invariance recipe of §5.1, literally.
4. **Training.** Nothing new: cross-entropy, Adam, the Week-16 recipe, the Week-20
   splits so the comparison is fair.
5. **Evaluation.** Same protocol as Week 20 — ROC AUC and background rejection at
   fixed efficiency, per energy bin — so the GNN-vs-CNN comparison in Week 24
   means something.

Expect the GNN to roughly match the CNN here: an 8×8 grid is small and the CNN's
grid assumption is not really wrong for it. The point-cloud representation wins when
grids fail — irregular geometry, sparse hits, variable multiplicity, or real jets.
That question — "does the graph structure actually help, or is it just parameters?"
— is exactly the ablation Capstone track (a) demands.

## Check yourself

1. Define permutation invariance and permutation equivariance. Which should a jet
   *classifier* have, and which should its internal layers have?
2. In the message/aggregate/update triple, which step would break the symmetry if
   done wrong, and why does a sum fix it?
3. Your friend pads every jet to 60 particles and trains an MLP on the flattened
   vector. Name two distinct problems.
4. Why does EdgeConv feed the MLP $(h_i, h_j - h_i)$ rather than $(h_i, h_j)$?
   What property do the differences give the messages?
5. What is oversmoothing, and what measurement would reveal it?
6. In one sentence each: what is a jet, and what is tracking-as-edge-classification?
7. How does PyG batch graphs of different sizes, and why does message passing not
   leak between events in a batch?
8. A 2D convolution is message passing on what graph, with what special property in
   its message function?

## Answers

1. Invariant: output unchanged under input reordering. Equivariant: output reorders
   the same way as the input. Classifier: invariant. Internal per-particle layers:
   equivariant (then a symmetric pooling makes the stack invariant).
2. Aggregation. Neighbors form an unordered set; a symmetric operation (sum, mean,
   max) gives the same result for any ordering of the same multiset of messages.
3. (a) Separate weights per slot force the network to learn interchangeability it
   could have for free; (b) predictions change under test-time reordering of the
   same physical event (and padding wastes capacity on empty slots).
4. Differences depend only on relative structure, so messages are unchanged if all
   embeddings shift together; $h_i$ keeps global context. Relative features
   generalize better across, e.g., jet positions in the detector.
5. Deep stacks blend every node with everything reachable, so node embeddings
   collapse toward a common vector. Measure the pairwise variance (spread) of node
   embeddings as a function of message-passing rounds and watch it fall.
6. A jet is the collimated spray of hadrons produced when a quark or gluon
   fragments. Tracking-as-edge-classification scores candidate hit-to-hit
   connections with a GNN and reads tracks off as chains of accepted edges.
7. As one disjoint big graph plus a `batch` vector mapping nodes to events; no
   edges connect different events, so messages cannot cross, and pooling groups by
   the `batch` vector.
8. The grid graph (each pixel connected to its window). Its messages are
   position-dependent — weight $w_d$ depends on the neighbor's relative offset $d$
   — which a general graph cannot support without edge features.

## New terms

- **jet** — collimated spray of hadrons from a fragmenting quark or gluon.
- **hit / tracking** — a position measurement in a sensor layer; reconstructing
  particle trajectories from hits.
- **permutation / permutation matrix** — a reordering; the shuffled identity matrix
  that implements it.
- **permutation invariance / equivariance** — output unchanged under reordering /
  output reorders with the input.
- **symmetric aggregation** — order-blind combination of a set (sum, mean, max).
- **Deep Sets** — shared per-item MLP, symmetric pool, output MLP; the minimal
  invariant architecture.
- **graph / node / edge / neighborhood / adjacency matrix / edge index** — the
  relational data structure and its storage.
- **kNN graph** — each node connected to its k nearest neighbors in a chosen space.
- **pseudorapidity (η), azimuth (φ), ΔR** — collider angular coordinates and the
  standard angular distance.
- **message passing (message / aggregate / update)** — the generic GNN layer.
- **EdgeConv** — message passing with MLP(center, neighbor − center) messages.
- **static vs dynamic graph** — edges fixed from input coordinates vs rebuilt in
  learned feature space each layer.
- **oversmoothing** — collapse of node embeddings toward each other with depth.
- **PyTorch Geometric (PyG) / `Data` / `batch` vector** — the GNN library and its
  graph container and batching scheme.
- **edge classification** — predicting a label per edge (Exa.TrkX tracking).
- **particle cloud** — a jet represented as an unordered set of particle feature
  vectors.

## Going deeper

- distill.pub, "A Gentle Introduction to Graph Neural Networks" — interactive
  pictures for every §5 idea; read it right after this lesson.
- distill.pub, "Understanding Convolutions on Graphs" — the §5.3 CNN connection
  done slowly and visually.
- Prince, *Understanding Deep Learning*, Ch. 13 (free PDF) — the same material with
  more formal notation; good second pass.
- Qu & Gouskos, "ParticleNet: Jet Tagging via Particle Clouds" (arXiv:1902.08570) —
  read fully; the representation argument is the point.
- Qu, Li & Qian, "Particle Transformer for Jet Tagging" (arXiv:2202.03772) —
  architecture and pairwise-feature sections; skim results.
- Exa.TrkX pipeline paper ("Performance of a geometric deep learning pipeline for
  HL-LHC particle tracking") — skim for the problem formulation only.
- PyTorch Geometric docs, "Introduction by Example" — work through it before E3.
