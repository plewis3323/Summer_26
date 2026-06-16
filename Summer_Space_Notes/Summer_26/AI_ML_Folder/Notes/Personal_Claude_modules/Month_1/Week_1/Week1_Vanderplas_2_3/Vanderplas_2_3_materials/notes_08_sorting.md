---
# Notes 08 — Sorting Arrays (Vanderplas Ch. 2.8)

> **Source:** https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html
> *Grad-student reading notes (Claude-generated run-through). Faithful to the source; anything beyond the source is flagged* (added context).

## Concept summary
NumPy ships fast, compiled sorting routines that vastly outperform hand-written Python sorts. The two workhorses are `np.sort` (returns sorted values) and `np.argsort` (returns the *indices* that would sort the array). Both can sort along a chosen `axis` of a multidimensional array. When you don't need a full sort — only the *k* smallest elements — `np.partition` / `np.argpartition` do the job faster. A capstone example uses these tools to compute k-nearest neighbors via broadcasting.

## Key ideas / idioms
- **Naive sorts are slow.** Selection sort is $$O(N^2)$$ — doubling $N$ roughly quadruples the time. Bogosort (shuffle until sorted) is $$O(N \times N!)$$ — a joke, never use it.
- **`np.sort` is fast.** Uses an introsort/quicksort by default at $$O(N \log N)$$ on average. Returns a *new* array; `x.sort()` sorts *in place*.
- **`np.argsort`** returns indices; `x[np.argsort(x)]` reconstructs the sorted array. Useful when you need the ordering to reindex *related* data.
- **`axis` keyword** sorts each row (`axis=1`) or each column (`axis=0`) independently — relationships across the other axis are lost.
- **Partitioning** (`np.partition(x, k)`) places the $k$ smallest values left of index `k` (in arbitrary order), the rest on the right. Faster than full sort when $k \ll N$. `np.argpartition` gives the indices.
- **Big-O intuition:** scaling matters most at huge $N$; on small arrays a worse-scaling algorithm can still win in wall-clock time.

## Worked code examples (runnable)

```python
import numpy as np

# --- Naive selection sort (O(N^2)) for intuition ---
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

print(selection_sort(np.array([2, 1, 4, 3, 5])))  # [1 2 3 4 5]
```

```python
import numpy as np

# --- np.sort vs in-place sort ---
x = np.array([2, 1, 4, 3, 5])
print(np.sort(x))   # [1 2 3 4 5]  (new array)
print(x)            # [2 1 4 3 5]  (unchanged)

x.sort()            # in-place
print(x)            # [1 2 3 4 5]

# --- np.argsort: indices that would sort the array ---
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)            # [1 0 3 2 4]
print(x[i])         # [1 2 3 4 5]  (fancy-index with the order)
```

```python
import numpy as np

# --- Sorting along an axis ---
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)
print(np.sort(X, axis=0))  # sort each column independently
print(np.sort(X, axis=1))  # sort each row independently
```

```python
import numpy as np

# --- Partial sort: k smallest with np.partition ---
x = np.array([7, 2, 3, 1, 6, 5, 4])
print(np.partition(x, 3))  # 3 smallest left of index 3, e.g. [2 1 3 4 6 5 7]

# argpartition along rows of a 2D array
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(np.argpartition(X, 2, axis=1))
```

```python
import numpy as np

# --- k-Nearest Neighbors via broadcasting + sorting ---
rand = np.random.RandomState(42)
X = rand.rand(10, 2)

# pairwise squared distances (10x10) via broadcasting
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)

# full sort: column 0 of each row is the point itself (distance 0)
nearest = np.argsort(dist_sq, axis=1)
print(nearest)

# efficient: only partition out the K+1 nearest (self + K neighbors)
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
print(nearest_partition[:, :K + 1])
```

## Why this matters / intuition
Sorting and "find the k smallest" are everywhere in data science: ranking, top-k retrieval, nearest-neighbor search, and feature selection. Knowing that `argsort`/`argpartition` return *indices* lets you reorder one array by another's values — the foundation of vectorized KNN. Choosing `partition` over a full `sort` when you only need the closest few can be a large speedup at scale. *(added context: this partition-for-top-k pattern is exactly what production KNN/ANN libraries optimize.)*

## Gotchas
- `np.sort(x)` returns a copy; `x.sort()` mutates in place — don't confuse the two.
- Sorting with `axis` treats each row/column as *independent*; you destroy cross-axis row alignment. To keep rows intact while ordering by a key column, use `argsort` on that column and fancy-index the whole array. *(added context)*
- `np.partition` does **not** sort: the $k$ smallest are merely on the left in arbitrary order. Don't assume `result[:k]` is itself sorted.
- In the KNN example, each point's nearest "neighbor" is itself (distance 0), so use `K + 1` and drop the first column.
- Better big-O does not guarantee faster runtime on small inputs — measure when in doubt.

## Suggested figure (optional)
A 2D scatter of the 10 KNN points with thin line segments drawn from each point to its 2 nearest neighbors, visually confirming that the partition-selected indices correspond to the geometrically closest points.

---

## 💬 Q&A (captured during session)

### Q: What is the `argsort` function?

**`np.argsort`** ("argument sort") returns the **indices that would sort the array** — the positions to pick, in order, to arrange the array smallest → largest — instead of the sorted values themselves.

**Sort vs argsort:**

```python
import numpy as np
x = np.array([30, 10, 50, 20])

np.sort(x)      # [10 20 30 50]   <- the sorted VALUES
np.argsort(x)   # [1  3  0  2]    <- the INDICES that sort it
```

Read `[1 3 0 2]` as: *"smallest value is at index 1, next at index 3, then 0, then 2."*

**Feed it back as fancy indexing** to reconstruct the sorted array:

```python
order = np.argsort(x)   # [1 3 0 2]
x[order]                # [10 20 30 50]  -> same as np.sort(x)
```

**Why use argsort instead of sort** — the index order lets you reorder *other* arrays the same way (sort one column, carry the rest along):

```python
names  = np.array(["Cara", "Al", "Eve", "Bob"])
scores = np.array([30, 10, 50, 20])

order = np.argsort(scores)     # [1 3 0 2]  (lowest score first)
names[order]                   # ['Al' 'Bob' 'Cara' 'Eve']
scores[order]                  # [10 20 30 50]
```

**Companions:**

```python
np.argsort(x)[::-1]          # descending (reverse the index list)
np.argmin(x), np.argmax(x)   # index of just the min / max
np.argsort(M, axis=1)        # row-wise index sort for a 2-D array
```

**Mental model:** `sort` answers *"what are the values in order?"*; `argsort` answers *"where do I find them, in order?"* — more useful whenever positions matter (reordering companions, top-k, ranking).
