# parallel-hierarchical-clustering
CPU-parallel implementation of agglomerative hierarchical clustering compared against a sequential baseline on benchmark datasets.

# CSE305 Project — Parallel Hierarchical Clustering

**Group of 2** — target: CPU with C++ `std::thread`

Hierarchical agglomerative clustering (HAC), implemented sequentially and parallelized on multicore CPU. Two parallel algorithms compared: pTRAD (cluster-per-thread, following Olson's PRAM scheme, from the naive HAC algorithm) and pPOP (cell-per-thread, partition-based). 
From the literature [paper pPOP] we have results on old hardware so we now want to know does pPOP's partitioning idea actually pay off on modern multicore hardware, and at what dataset size?

---

## Table of contents

1. [Problem definition](#1-problem-definition)
2. [Literature](#2-literature)
3. [Algorithms](#3-algorithms)
   - [Algorithm 1 — Naive sequential HAC](#algorithm-1--naive-sequential-hac)
   - [Algorithm 2 — pTRAD (parallel naive HAC, cluster-per-thread)](#algorithm-2--ptrad-parallel-naive-hac-cluster-per-thread)
   - [Algorithm 3 — Sequential POP](#algorithm-3--sequential-pop)
   - [Algorithm 4 — pPOP (parallel partition-based HAC, cell-per-thread)](#algorithm-4--ppop-parallel-partition-based-hac-cell-per-thread)
4. [Dataset](#4-dataset)
5. [Correctness verification](#5-correctness-verification)
6. [Benchmarking plan](#6-benchmarking-plan)
7. [Implementation order](#7-implementation-order)
8. [Division of work](#8-division-of-work)
9. [What the project sheet is grading](#9-what-the-project-sheet-is-grading)

---

## 1. Problem definition

Fixed for all four algorithms:

- **Input:** `N` points in `R^d` (real-valued, apparently dimension should be low like d=2 but we need to see w our dataset)
- **Point metric:** Euclidean distance (L2).
- **Cluster linkage:** average-link (UPGMA) as primary. Single-link and complete-link supported as parameters via the **Lance-Williams** update formula.
- **Output:** Dendogram, represented as an ordered merge list — for each of the `N - 1` merges, record `(cluster_a, cluster_b, merge_distance, new_cluster_id)`.

The Lance-Williams update formula expresses the distance from a newly merged cluster `(i ∪ j)` to any other cluster `k` as a linear combination of the prior distances `d(i,k)`, `d(j,k)`, `d(i,j)`. The coefficients depend on the linkage:

- single-link: `alpha_i = alpha_j = 1/2`, `beta = 0`, `gamma = -1/2`
- complete-link: `alpha_i = alpha_j = 1/2`, `beta = 0`, `gamma = +1/2`
- average-link: `alpha_i = |i| / (|i|+|j|)`, `alpha_j = |j| / (|i|+|j|)`, `beta = 0`, `gamma = 0`

`|i|` denotes the size of cluster `i`. Storing the formula once and switching linkage with a parameter is part of the design.
**!!!!!!!!!!!!ADD WHERE WE FOUND THAT**
---

## 2. Literature
**Given:**

- **Olson 1995** — *Parallel algorithms for hierarchical clustering.* The parallelization roadmap; PRAM model. Source of the cluster-per-processor scheme that becomes pTRAD.
- **Dash, Petrutiu, Scheuermann 2007 (pPOP)** — *Fast yet accurate parallel hierarchical clustering using partitioning.* Practical shared-memory paper. Source of the partition-based pPOP algorithm.
- **Rajasekaran 2005** — *Efficient parallel hierarchical clustering algorithms.* Theoretical (PRAM + AROB). Source of the single-link = Euclidean MST equivalence; useful for related-work framing.

**Beyond the given papers:** not sure if we use these 
- **Müllner 2011** — *Modern hierarchical, agglomerative clustering algorithms* (arXiv:1109.2378). Clean pseudocode for naive O(N³) and nearest-neighbor chain. Fills the "where are the actual sequential algorithms" gap.
- **Yu, Wang, Gu, Dhulipala, Shun 2021 (ParChain)** — *A Framework for Parallel Hierarchical Agglomerative Clustering using Nearest-Neighbor Chain* (arXiv:2106.04727). Modern parallel HAC, linear-memory variant. Used for related-work context.
- **Tan, Steinbach, Kumar** — *Introduction to Data Mining*, clustering chapters. Broad context for the report's introduction.

**Framing of the report.** Olson's PRAM model assumes `n/log n` idealized processors with free, instant memory access. Translating that to 8 or 16 real threads with a real memory hierarchy is the gap we fill in this project. pPOP showed one way to bridge that gap on 1990s hardware (8-CPU SGI Origin2000). This project re-runs the comparison on modern multicore CPUs, with the implementation and load-splitting decisions we made in `std::thread`.

---

## 3. Algorithms

### Algorithm 1 — Naive sequential HAC

Role: Baseline
Complexity: `O(N^3)` time, `O(N^2)` memory.

Input: Data (N,M)
Output: Dendrogram
1. for i = 1 to N  1
2. find the closest pair of clusters
3. merge the closest pair of clusters
4. return dendrogram

```
Algorithm: Naive Sequential HAC
Input:  N points, linkage parameter 
Output: Dendrogram (list of N-1 merges)

  --- setup ---
1.  active_clusters := { {p_1}, {p_2}, ..., {p_N} }
    (each point starts as its own singleton cluster)
2.  Build the N x N distance matrix D:
       for all i < j:  D[i][j] := euclidean(p_i, p_j)

  --- main loop ---
3.  Repeat N - 1 times:
4.     Scan D over all pairs (i, j) with i, j in active_clusters, i < j.
       Find (i*, j*) with the minimum D[i*][j*]; call this distance d*.
5.     Record the merge (i*, j*, d*, new_id) in the dendrogram.
6.     Remove i* and j* from active_clusters; add new_id.
7.     For every remaining cluster k in active_clusters, k != new_id:
          D[new_id][k] := Lance-Williams(D[i*][k], D[j*][k], D[i*][j*], |i*|, |j*|, |k|, linkage)
       Mark rows/columns i* and j* as inactive.

8.  Return dendrogram.
```

**Notes.**

- The "active cluster" bookkeeping can be a boolean array of size `N` — simpler than physically resizing the matrix.
- Cluster IDs: original points are `0..N-1`; merged clusters get IDs `N..2N-2`. The dendrogram is fully described by the merge list.
- Tie-break (when two pairs have equal distance): use lexicographic order of `(min(i,j), max(i,j))`. Document this and use it consistently in all four algorithms, so serial and parallel outputs are bit-identical.

---

### Algorithm 2 — pTRAD (parallel naive HAC, cluster-per-thread)

Role: first parallel target. Follows Olson's PRAM scheme (cluster-per-processor) translated to `std::thread`. This is the pPOP paper's "pTRAD" baseline — what pPOP is compared against.

Same algorithmic structure as Algorithm 1, but the three expensive regions run in parallel:
- distance matrix construction
- global closest-pair search
- post-merge distance update

```
Algorithm: pTRAD (parallel naive HAC)
Input:  N points, linkage parameter, number of threads P
Output: dendrogram

  --- setup ---
1.  active_clusters := { {p_1}, ..., {p_N} }
2.  Build distance matrix D in PARALLEL:
       partition the O(N^2) pairs (i, j) into P chunks (static scheduling)
       each thread computes its chunk of D

  --- main loop ---
3.  Repeat N - 1 times:

4.  // PARALLEL: each thread finds its local minimum
    On each thread t in 0..P-1:
       scan thread t's chunk of D restricted to active clusters
       record local_min_dist[t], local_min_pair[t]

5.  // REDUCTION: combine the P local minima into one global minimum
    Either: (a) a designated thread linearly scans the P candidates, or
            (b) a tree-style parallel reduction
    Result: (i*, j*, d*)

6.  // SERIAL: merge is a single step
    Record the merge in the dendrogram.
    Update active_clusters: remove i*, j*, add new_id.

7.  // PARALLEL: distance update
    Partition active_clusters \ {new_id} into P chunks (dynamic scheduling
    preferred — chunks shrink as the algorithm progresses).
    On each thread t:
       for each cluster k in thread t's chunk:
          D[new_id][k] := Lance-Williams(D[i*][k], D[j*][k], D[i*][j*], ...)

8.  Return dendrogram.
```

**Design decisions to make and justify (this is the graded "load split" thinking).**

- **Scheduling — step 2 (setup):** static. Work per pair is roughly equal, no benefit to dynamic.
- **Scheduling — step 4 (closest-pair search):** static if active set is large; the rows-per-thread split degrades as the active set shrinks because some chunks become empty. Consider switching to dynamic chunking in late iterations.
- **Scheduling — step 7 (distance update):** dynamic. Number of clusters being updated shrinks every iteration; static would leave threads idle.
- **Reduction (step 5):** start with the simple "designated thread scans P candidates" version; benchmark and consider tree reduction if P is large.
- **Synchronization:** the only required critical region is the reduction in step 5 — protect with a mutex or atomic compare-and-swap, or use a per-thread-local-minimum array and reduce serially without locks. The latter is simpler and probably faster for small `P`.
- **Thread infrastructure:** a hand-built thread pool with a work-queue, reused across iterations. Spawning fresh `std::thread`s per iteration is unacceptable overhead — explain why in the report.

**Why pTRAD's scaling will be sublinear (a finding to demonstrate, not hide).** As the active set shrinks, the per-iteration work shrinks too, but the synchronization cost (one reduction per merge, `N - 1` merges total) does not. This bounds achievable speedup and is exactly the kind of bottleneck the analysis section will diagnose.

---

### Algorithm 3 — Sequential POP

Role: sequential version of the partition-based approach. Faster than Algorithm 1 in principle (the "90-10 rule" payoff), but our main reason for implementing it is correctness and clarity: it's the cleanest reference for the parallel pPOP.

**The 90-10 observation:** in HAC, the vast majority of early merges happen between small, nearby clusters. Only the last few merges combine large, distant clusters. POP exploits this by restricting most merges to local cells.

Two phases:

- **Phase 1:** repeatedly merge the closest pair *within cells*; stop when the closest within-cell pair has distance ≥ δ.
- **Phase 2:** finish remaining clusters with traditional HAC (Algorithm 1).

**Correctness guarantee:** the overlapping cells (δ-region) ensure that *any* pair of clusters with distance < δ is together in at least one cell. So Phase 1 cannot miss a merge that traditional HAC would have made at that distance threshold. When Phase 1 stops, only merges at distance ≥ δ remain, and Phase 2 handles those correctly.

```
Algorithm: Sequential POP
Input:  N points, partitioning parameters (c, δ), linkage parameter
        c = number of cells; δ = overlap distance
Output: dendrogram

  --- setup ---
1.  Partition the 2-D data space into c axis-parallel cells (e.g. sqrt(c) x sqrt(c) grid).
2.  For each cell C, define its δ-extended region (the cell expanded outward by δ on all sides).
3.  For each point p_i:
       Compute its cell coordinates.
       Assign p_i to its container cell.
       If p_i also lies in any neighboring cell's δ-region, assign p_i to that cell too.
       (Points strictly inside their container cell — far from any boundary — belong to one cell.
        Points in a δ-region belong to multiple cells.)
4.  For each cell C, build a local distance structure (priority queue or local matrix) over
    the clusters currently assigned to C. Distances are only between clusters in the same cell.

  --- Phase 1: local merging ---
5.  Repeat:
6.     For each cell C, find the closest pair within C: (pair_C, dist_C).
7.     Among all (pair_C, dist_C), find the global minimum: (i*, j*, d*, container_cell).
8.     If d* >= δ: break out of Phase 1.
9.     Record merge (i*, j*, d*, new_id) in dendrogram.
10.    Update container_cell's local structure:
          - remove i* and j* from the cell's cluster list
          - add new_id
          - compute distances from new_id to every other cluster in container_cell via Lance-Williams
       If new_id (or either i* or j*) lies in a δ-region, also update every affected neighboring cell
       analogously. (Usually only the container cell is affected.)

  --- Phase 2: finish with traditional HAC ---
11. Collect all remaining clusters from all cells into a single global set.
    (Deduplicate: a cluster in a δ-region appeared in multiple cells.)
12. Run Algorithm 1 (Naive Sequential HAC) on this set until only one cluster remains,
    appending each merge to the dendrogram.

13. Return dendrogram.
```

**Parameter choice.**

- **δ:** ideally set to the "turning point" of the closest-pair-distance curve — the distance at which the 90-10 transition happens. In practice we won't know this in advance. Start with `δ = 0` and use the nested variant: re-run with progressively larger `δ` (e.g. `δ ← 1.1 × last_observed_min_dist` whenever the previous `δ` is exceeded).
- **c:** large enough that the average clusters-per-cell is roughly 5-20. For `N = 10,000` that suggests `c ≈ 500-2000`.

**Implementation notes.**

- A point in a δ-region must be tracked as "this same cluster lives in cells A and B." When it gets merged, both cells' local structures need to be updated, but the *merge record* in the dendrogram is recorded once.
- A cluster's representative point (for cell membership) is its centroid for average-link. For single-link this is less well-defined; for the initial pass, every point's "representative" is itself, so this only matters after merges.

---

### Algorithm 4 — pPOP (parallel partition-based HAC, cell-per-thread)

Role: second parallel target and the main contribution of the comparison study. Each cell is the unit of parallel work — fundamentally different from pTRAD's cluster-per-thread approach. This is the experiment the pPOP paper performed in 2007 on 1990s hardware; we rerun it on modern multicore CPUs.

```
Algorithm: pPOP (parallel partition-based HAC)
Input:  N points, partitioning parameters (c, δ), linkage, number of threads P
Output: dendrogram

  --- setup (parallel) ---
1.  Partition the data space into c overlapping cells (same as Sequential POP, step 1).
2.  Assign points to cells in PARALLEL:
       partition the N points across P threads (static scheduling)
       each thread assigns its points to container cell + any δ-region cells
       protect cell membership lists with per-cell mutexes (contention is low)
3.  Build per-cell distance structures in PARALLEL:
       partition the c cells across P threads (static scheduling — c/P cells per thread)
       each thread builds local structures for its cells independently (no contention)

  --- Phase 1: parallel local merging ---
4.  Repeat:

5.  // PARALLEL: find closest pair per cell, with DYNAMIC scheduling
    Reason: cells have very different cluster counts; static would idle some threads.
    Each thread repeatedly grabs the next available cell from a shared work queue and computes:
       (pair_C, dist_C) := closest pair within cell C
    Continue until all cells processed in this iteration.

6.  // REDUCTION: one thread combines the P local-minimum candidates
    Each thread submits its best (pair, dist, cell) into a shared slot.
    Designated thread: scan the P candidates, pick global minimum (i*, j*, d*, container_cell).
    Protect with mutex OR per-thread slot + serial reduction (no locks needed).

7.  If d* >= δ: break out of Phase 1.

8.  // SERIAL: record the merge
    Append (i*, j*, d*, new_id) to dendrogram.

9.  // PARALLEL: update container cell's local structure
    The clusters in container_cell are split across P threads.
    Each thread computes Lance-Williams distances from new_id to its assigned clusters
    and updates the local priority queue / matrix.

10. // SERIAL (cheap): identify δ-region neighbors
    Designated thread: check whether i*, j*, or new_id lies in any neighboring cell's δ-region.
    Most iterations: no neighbors affected.

11. // PARALLEL (when applicable): update affected neighboring cells
    Cells × clusters work, split across threads dynamically.

  --- Phase 2: finish ---
12. Collect remaining clusters globally (deduplicate δ-region duplicates).
13. Run pTRAD (Algorithm 2) on this small remainder. By construction the remainder is small,
    so Phase 2 cost is negligible — the choice of pTRAD vs serial here is mainly for symmetry
    with our infrastructure.

14. Return dendrogram.
```

**Design decisions to make and justify.**

- **Setup scheduling (steps 2, 3):** static. Roughly equal work per chunk; lowest overhead.
- **Phase 1 closest-pair search (step 5):** dynamic. Cells vary in cluster count by 1-2 orders of magnitude on real data; static splitting would leave the lightly-loaded threads idle.
- **Reduction (step 6):** P is small (≤ 16 typically), so a per-thread slot followed by serial scan is simpler and faster than a tree reduction.
- **Update of container cell (step 9):** dynamic if container cell has many clusters; static (or even sequential) if it has few. Worth measuring whether the threading overhead pays off when the cell has 5 clusters.
- **δ-region updates (step 11):** rare event. Implementing this in parallel adds complexity for limited gain — consider doing it serially in the first implementation and adding parallelism only if profiling shows it matters.

**Where pPOP can beat pTRAD.**

- Local distance structures are `O((N/c)^2)` per cell, not `O(N^2)` global. Memory locality is much better.
- Most iterations only touch one cell, so step-9 work per iteration is `O(N/c)` not `O(N)`.
- The parallel dimension scales with `c` (number of cells), not just `P` (number of threads) — so even when threads are not all busy at once, the partitioning still reduces total work.

**Where pPOP can lose to pTRAD.**

- For small `N`, the δ-region bookkeeping and per-cell setup overhead can dominate.
- If the data is not well-clusterable (uniform), the 90-10 assumption fails and Phase 2 becomes large.
- Load imbalance across cells can be severe for non-uniform distributions.

Finding the crossover point on modern hardware is the central empirical question of this project.

---

## 4. Datasets
Scaturchio, Lorenzo (2026). Spotify Tracks: Audio Features (50K Songs). Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k

---

## 5. Correctness verification

- **Serial-vs-parallel equivalence:** Algorithms 2 and 4 must produce exactly the same dendrogram as Algorithm 1 on every dataset. The shared tie-break rule makes this a bit-identical comparison.
- **Cross-thread-count consistency:** results must be identical at 1, 2, 4, 8, ... threads. Differences here mean races.
- **Quality evaluation:** cut each dendrogram at the known number of ground-truth clusters and compute Adjusted Rand Index against the labels. ARI confirms the clustering is meaningful, not just internally consistent.
- **Tie-break rule (applies to all four algorithms):** when two cluster pairs have equal distance, pick the pair with the lexicographically smaller `(min(i,j), max(i,j))`. Enforce this across thread orderings in the parallel versions.

---

## 6. Benchmarking plan

- **Speedup curves:** wall-clock time and speedup vs thread count (1, 2, 4, 8, ...) for pTRAD and pPOP, on each dataset size.
- **Core comparison:** pTRAD vs pPOP across the size ladder. Locate the crossover point (if any) where pPOP's partitioning starts to win.
- **Linkage axis:** does speedup differ across single / average / complete? Probably barely — and "barely" is itself a finding.
- **Methodology:** multiple runs per configuration, report mean and variance, fixed hardware, documented machine specs, warm-up runs discarded.
- **Bottleneck diagnosis:** where does speedup plateau and *why* — synchronization, load imbalance, memory bandwidth? Naming and diagnosing your own bottlenecks is a top-tier signal.
- **Honest limitations:** report cases where pPOP loses (e.g. small N where δ-region bookkeeping dominates). Don't hide them.

---

## 7. Implementation order

1. **Algorithm 1 first.** Without a correctness oracle, no parallel result can be trusted.
2. **Thread pool + work queue infrastructure.** Built once, reused by Algorithms 2 and 4.
3. **Algorithm 2 (pTRAD).** Validates infrastructure on a familiar algorithm. The "easy" parallel case.
4. **Algorithm 3 (sequential POP).** Built next because pPOP without sequential POP working is impossible to debug.
5. **Algorithm 4 (pPOP).** Final, hardest piece.
6. **Benchmarking harness, ARI evaluation, and the experimental campaign.**
7. **Report writing in parallel with steps 5-6.** Don't leave it for last — it's where the grade lives.

---

## 8. Division of work

- **Person 1:** Algorithm 1 (sequential baseline) + thread pool / work queue infrastructure + Algorithm 2 (pTRAD) + benchmarking harness.
- **Person 2:** Algorithm 3 (sequential POP) + Algorithm 4 (pPOP) partitioning machinery + dataset loading + ARI evaluation.
- **Both:** the parallelization-design decisions, the experiments, the analysis, and the report.

---

## 9. What the project sheet is grading

Decoded from the project sheet's own wording:

- **"Choosing the similarity for clusters and the metric employed"** — we explicitly chose Euclidean + average-link (primary), with single/complete supported via Lance-Williams. Documented in section 1.
- **"Implement a sequential algorithm for solving it"** — Algorithm 1 (and Algorithm 3 as a smarter sequential variant).
- **"Design a parallel version of the algorithm"** — note the verb *design*. Algorithms 2 and 4, with the load-splitting decisions and justifications listed in their respective sections.
- **"Translating these algorithms into a CPU setting would require thinking about the right split of the load"** — this sentence points directly at the design-decisions subsections of Algorithms 2 and 4. The static-vs-dynamic scheduling choices, the synchronization design, the load-balancing, and the bottleneck diagnosis are the deliverable.
- **"You are welcome to do some more literature search"** — Müllner 2011 and ParChain 2021 added to section 2.
- **"Compared against the serial version for different number of threads on benchmarks of varying size"** — the benchmarking plan in section 6.

The single A+ test, applied throughout: **for every implementation and parallelization decision, you can answer "why did you do it that way?" with a reason grounded in the parallelism, the data, or your measurements — not "because the paper did."**
