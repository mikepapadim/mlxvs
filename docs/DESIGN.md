# mlxvs design notes

How the operations map onto MLX and Metal. For the API see the [README](../README.md).

- Kernels are Metal source compiled at run time by `mlx::core::fast::metal_kernel`: no offline Metal toolchain
  needed, and they run on MLX arrays and MLX's command queue.
- GEMM-shaped work (distances, k-means assignment) uses MLX's matmul. Selection uses mlxvs kernels: MLX's
  `argpartition` runs a full merge sort along the axis, slow for k-NN shapes (k = 10..256 of 10^4..10^6 columns).
- `topk_merge`: one threadgroup per query, radix select on order-preserving float keys (4 passes of 8 bits,
  per-SIMD-group histograms against atomic contention), one gather pass, bitonic sort of the winners, merged
  with the running list, so a scan over base tiles keeps only [nq, k] between tiles.
- `nn_descent`: MLX custom kernels cannot update an input in place (every output is a fresh array, optionally
  pre-filled), so each round is a pure function of the graph: `sample(ids, flags) -> members [n, 64], flags'`
  (reverse samples written to hashed slots of a -1-filled array), `join(rows, norms, scores, members) -> proposals
  [n, P]` (64-bit id | score words in hashed slots, -1-filled; the random start passes permutation blocks as
  `members`), `merge(graph, flags', proposals) -> graph', changed [n]`. One `mx::eval` per round reads the sum of
  `changed` for the termination test. Returned distances are recomputed in FP32 and the lists re-sorted, so FP16
  joins only affect which neighbors are found. Up to 4096 rows `nn_descent` computes the graph exactly instead:
  below about twice the list size the hashed proposal slots drop colliding candidates and the rounds can stop
  before every row is proposed to every other, leaving lists short, and at that size exact k-NN is as fast.
- MLX has no stable C++ ABI: mlxvs is built against one MLX release. The Python extension links the `mlx`
  wheel's own `libmlx` (located with `python -m mlx --cmake-dir`), so one process holds one copy.
- `all_neighbors`: k-means (on unit rows for cosine) splits the rows; each row joins its `overlap_factor` nearest
  clusters. Each cluster's rows are gathered and its graph built on its own (`nn_descent`, or `brute_force_knn` for
  clusters of at most `brute_force_below` rows), its local ids mapped back to row ids, and every member's list merged
  with the new one on the GPU: concatenated, sorted by id to drop repeats (a pair found in two clusters) and the row
  itself, then sorted by distance and cut to k. The graph [n, k] stays resident; the NN-Descent state (lists of
  max(64, k), proposals) exists for one cluster at a time.
- Out-of-core rows and cluster sizes: `all_neighbors` reads rows through a gather function, so a host array or a
  memory-mapped file is never copied whole. A k-means sample, one 65536-row tile during assignment and one cluster
  at a time go to the GPU; a cluster's rows are copied by every core into one page-aligned buffer that MLX wraps
  without a further copy. A cluster holds at most `max_cluster_rows` rows (by default what fits 40% of the GPU's
  recommended working set at about `6 d + 2 KB` bytes per row). With `n_clusters = 0` there are enough clusters for
  the average to be two thirds of that cap, since k-means clusters are uneven. A cluster still over the cap is
  clustered again (its rows join the 2 nearest sub-clusters) rather than filled to a capacity by sending rows to
  farther clusters, which separates those rows from their neighbors.
- `graph_search`: CAGRA's single-CTA search, one threadgroup of 256 threads per query. Its state lives in threadgroup
  memory: a sorted list of `itopk_size` entries (two halves, merged from one into the other), one round's candidates,
  the query when `d <= 1024`, and a visited set (open addressing over atomic ints). The random start, as cuVS, fills
  `itopk_size + search_width * degree` slots, each with the nearest of `num_random_samplings` random nodes (one
  SIMD group per slot, in rounds of one round's candidate slots). Then each round
  marks the best unexpanded list entries as expanded (an id stored as `-(id + 2)`), gathers their graph neighbors,
  keeps the ones it inserts into the visited set (so no node is scored twice in a round, nor one in the list), scores
  them one SIMD group per candidate, sorts them (bitonic) and merges them into the list: each entry's place is its
  index plus the number of entries of the other list before it, found by binary search. When the visited set is half
  full it is cleared and refilled with the list's nodes, as CAGRA's small-hash mode. The search ends when every list
  entry is expanded, or after cuVS's `max_iterations`. Everything stays within 16 KB of threadgroup memory where it can: Metal allows 32 KB, but its
  shader validation (run in CI) keeps a shadow copy and counts every byte twice.
- `cagra_optimize`: five kernels. Prune: one threadgroup per node X loads N(X) into a threadgroup id -> rank table,
  then for every pair (i, r) reads Y = N(N(X)[i])[r] and, if Y is in N(X) at rank j > max(i, r), counts a detour
  of the edge X -> Y; each edge's place in the output is the number of edges with fewer detours (or as many and a
  lower rank). Reverse: each node keeps the `graph_degree` incoming edges of lowest rank, as cuVS (which launches one
  kernel per rank). A custom kernel's outputs are fresh arrays, so instead of one launch per rank: one thread per
  kept edge X -> Y counts Y's incoming edges by rank, one thread per node turns the counts into offsets, and one
  thread per edge writes (rank, X) at its rank's offset plus its place among that rank's edges (an atomic counter),
  if below `graph_degree`. Merge:
  one thread per node keeps the first half of its pruned list, adds its reverse edges in (rank, source) order that
  are not there yet, then the rest of the pruned list.
