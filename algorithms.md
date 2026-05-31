### Algorithm 1 — Naive sequential HAC
Algorithm: Naive Sequential HAC
Input:  N points, linkage parameter
Output: dendrogram (list of N-1 merges)

1.  active[i] = true for all i in 0..N-1   // boolean array, O(N)
    D[i][j] = euclidean(p_i, p_j)          // N x N matrix, O(N^2)

2.  Repeat N-1 times:

3.     (i*, j*, d*) = (0, 0, INF)
       for each pair (i, j) where active[i] and active[j] and i < j:
           if D[i][j] < d*:
               (i*, j*, d*) = (i, j, D[i][j])
       // tie-break: lexicographic (i, j), smaller wins

4.     record merge (i*, j*, d*, new_id) in dendrogram

5.     active[i*] = false
       active[j*] = false
       active[new_id] = true

6.     for each k where active[k] and k != new_id:
           D[new_id][k] = lance_williams(D[i*][k], D[j*][k], D[i*][j*],
                                         size[i*], size[j*], linkage)
7.  return dendrogram

### Algorithm 2 — pTRAD
This is NearestNeighbour / SumParallel from td1, applied three times. The pattern is identical: split the work into blocks, each thread writes its local result into results[t], main thread reduces after join()

Algorithm: pTRAD
Input:  N points, linkage parameter, P threads
Output: dendrogram

  --- setup: build distance matrix in parallel ---
  // Same as SumParallel in td1: static split of O(N^2) pairs across P threads.
  // Each thread computes D[i][j] for its assigned pairs.
  // No synchronization needed — each pair is independent.

1.  block_size = (N*N) / P
    std::vector<std::thread> workers(P-1)
    for t in 0..P-2:
        workers[t] = std::thread(BuildDistBlock, t*block_size, (t+1)*block_size, ...)
    BuildDistBlock((P-1)*block_size, N*N, ...)   // calling thread handles last block
    for t in 0..P-2: workers[t].join()

  --- main loop: repeat N-1 times ---

  // Step A: find closest pair — same pattern as NearestNeighbour in td1.
  // Each thread scans its chunk of active clusters, writes local best into results[t].
  // After join(), main thread does the serial reduction (scan results[]).

2.  std::vector<LocalMin> results(P)   // struct { int i, j; double dist; }
    for t in 0..P-2:
        workers[t] = std::thread(FindLocalMin, t, active, D, results[t])
    FindLocalMin(P-1, active, D, results[P-1])
    for t in 0..P-2: workers[t].join()

    (i*, j*, d*) = reduce(results)   // serial scan of P elements, trivial

  // Step B: merge — serial, one step.
3.  record merge in dendrogram
    active[i*] = false, active[j*] = false, active[new_id] = true

  // Step C: update distances — same td1 pattern again.
  // Each thread updates D[new_id][k] for its chunk of active clusters k.

4.  for t in 0..P-2:
        workers[t] = std::thread(UpdateDist, t, active, D, i*, j*, new_id, linkage)
    UpdateDist(P-1, active, D, i*, j*, new_id, linkage)
    for t in 0..P-2: workers[t].join()

5.  return dendrogram

Important: do NOT spawn new std::thread objects every iteration — that's O(N) thread creations, each expensive. Reuse threads via a thread pool. Your SafeUnboundedQueueCV from td5 is the thread pool queue. Workers loop on queue.pop(), main thread pushes tasks. When tasks are done, main thread waits (join or a semaphore). This is the one piece that goes slightly beyond td1's simple spawn-join, but you already have the queue from td5.

### Algorithm 3 — Sequential POP

Algorithm: Sequential POP
Input:  N points, c cells, δ overlap distance, linkage
Output: dendrogram

  --- setup ---
1.  divide the 2D space into a sqrt(c) x sqrt(c) grid of cells
    each cell has bounds [x_min, x_max] x [y_min, y_max]
    δ-extended bounds: [x_min - δ, x_max + δ] x [y_min - δ, y_max + δ]

2.  for each point p:
        assign p to every cell whose δ-extended bounds contain p
        (usually 1 cell; up to 4 if p is near a corner)

3.  for each cell C:
        build local distance matrix D_C over clusters in C
        // same as Algorithm 1 step 1, but only for clusters in C

  --- Phase 1: local merging ---
4.  repeat:
5.      for each cell C:
            find local closest pair (i_C, j_C, d_C) in D_C  // scan D_C

6.      (i*, j*, d*, C*) = argmin over all cells of d_C
        // serial scan over c values — cheap

7.      if d* >= δ: break

8.      record merge (i*, j*, d*, new_id) in dendrogram
        update C*'s local distance matrix: remove i*, j*, add new_id via lance_williams
        if new_id lies in any neighboring cell's δ-region:
            update that cell's local distance matrix too
            // check at most 3 neighbors — cheap

  --- Phase 2: finish ---
9.  collect all remaining active clusters across all cells (deduplicate)
    run Algorithm 1 on this small remainder
    return dendrogram

### Algorithm 4 — pPOP
 combines the td1 static-split pattern (setup) with the SafeUnboundedQueueCV from td5 (dynamic scheduling of cells). Those are the only two pieces

Algorithm: pPOP
Input:  N points, c cells, δ, linkage, P threads
Output: dendrogram

  --- setup: parallel, static split ---
  // Same td1 pattern: split the c cells across P threads statically.
  // Each thread builds local distance matrices for its cells.
  // No shared state — cells are independent. No synchronization needed.

1.  block_size = c / P
    for t in 0..P-2:
        workers[t] = std::thread(BuildCells, t*block_size, (t+1)*block_size, cells)
    BuildCells((P-1)*block_size, c, cells)
    for t in 0..P-2: workers[t].join()

  --- Phase 1: parallel local merging ---
2.  repeat:

    // Step A: find closest pair per cell — DYNAMIC scheduling using td5's queue.
    // Main thread pushes all c cell indices into the work queue.
    // Each worker pops a cell index, finds local min, writes into results[t].

3.  SafeUnboundedQueueCV<int> work_queue
    for cell_id in 0..c-1: work_queue.push(cell_id)
    work_queue.push(SENTINEL)   // one sentinel per thread signals "done"
    // ... (P sentinels total)

    std::vector<LocalMin> results(P)
    for t in 0..P-2:
        workers[t] = std::thread(WorkerFindMin, work_queue, cells, results[t])
    WorkerFindMin(work_queue, cells, results[P-1])
    for t in 0..P-2: workers[t].join()

    // Each worker's loop:
    // while true:
    //     cell_id = work_queue.pop()   // blocks until work available — td5
    //     if cell_id == SENTINEL: break
    //     results[t] = min(results[t], find_local_min(cells[cell_id]))

4.  (i*, j*, d*, C*) = reduce(results)   // serial scan of P results
    if d* >= δ: break

    // Step B: merge — serial.
5.  record merge in dendrogram
    update C*'s local distance matrix (serial — one cell, fast)
    check δ-region neighbors, update if needed (serial — at most 3 cells)

  --- Phase 2: finish ---
6.  collect remaining clusters, run pTRAD on them, return dendrogram


# MOST IMPORTANT THINGS 
# get a speedup
# understand where it comes from
# understand why we cant get more 



# Normalize dataset -> explain -> scale them to gaussian as long as they are actually gaussian if they are not gaussian not the best idea  better to scale to [0,1] 