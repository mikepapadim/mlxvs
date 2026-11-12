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
  entry is expanded, or after cuVS's `max_iterations`. Distances are computed by teams of lanes, as CAGRA: a team of
  4 to 32 lanes (by default 8 near d = 128, 16 near 256, 32 from 384 up, as cuVS picks) scores one candidate, each
  lane loading four columns at a time, and the team sums with SIMD shuffles; a SIMD group scores 32 / team
  candidates at once, which keeps more row reads in flight than one SIMD group per candidate.
- `graph_search` multi-CTA (`algo="multi_cta"`, or `"auto"` with fewer than two queries per GPU core, as cuVS
  decides by its SM count; the core count comes from IOKit): `max(search_width, itopk_size / 32)` threadgroups of
  128 threads per query, as cuVS's `search_multi_cta`. Each keeps a buffer of its 32-entry list plus one parent's
  neighbors, sorted together by one SIMD group (bitonic in threadgroup memory). Per iteration a threadgroup claims
  one parent in a traversed set in device memory shared by the query's threadgroups (open addressing with
  removal: a removed id is stored as `-(id + 2)`), so no node is expanded twice; former parents pushed out of the
  list leave the set; its own visited set is cleared and refilled from its buffer every iteration; children not
  visited nor traversed are written over the end of the buffer and scored; entries other threadgroups have since
  expanded are dropped. At the end each threadgroup writes its 32 best (claiming the non-parents in the traversed
  set, so each node comes from one threadgroup) and a top-k over all of them gives the result. Everything stays within 16 KB of threadgroup memory where it can: Metal allows 32 KB, but its
  shader validation (run in CI) keeps a shadow copy and counts every byte twice.
- Filters and seeds (both `graph_search` kernels): the filter is a bitset in device memory. Single-CTA compacts its
  list after every round, dropping expanded nodes the filter rejects (a prefix sum per SIMD group), and writes the
  first k allowed entries; multi-CTA drops a rejected parent after expanding it and skips rejected entries at the
  end. Seeds replace the random start's first slots (a draw index below `num_seeds`), as cuVS.
- `cagra_extend`: cuVS's `add_node_core`. The candidates come from `graph_search` on the GPU; the detour counts,
  reverse edges and interleaving run on the CPU over the copied graph, one new row after another (each sees the
  edges the earlier ones changed).
- `ivf_flat_search`: the probed lists come from `brute_force_knn` over the centers. Each query gets a candidate
  buffer as long as its probed lists together (offsets by prefix sums), the widest query setting the width. The
  (query, list) pairs are sorted by list and cut into items of up to 16 pairs of one list (fewer for wide rows, so
  the queries fit in 8 KB of threadgroup memory): one threadgroup per item loads the queries, and each thread
  reads rows of the list once, scoring them against every query of the item. `topk_merge` then keeps the best k
  of each buffer.
- `ivf_pq_search`: one threadgroup per (probed list, query). The query's residual to the center (L2), or the query
  (inner product, plus its dot product with the center), sits in threadgroup memory; the float32 lookup table is
  built 8 subspaces at a time (256 codes: 8 KB) and every thread adds the entries of its rows' codes, keeping its
  first 8 rows' sums in registers and the rest in the candidate buffer. The best k then go through `topk_merge` as
  IVF-Flat's.
- `graph_search_vpq` (CAGRA-Q): the single- and multi-CTA kernels compiled again with `mlxvs_score` replaced by
  `mlxvs_score_vpq`, whose team of lanes takes the subspaces in turn, adding each row's coarse center (device
  memory, few enough to stay in cache) and its codebook entries. `vpq_build` follows cuVS's `vpq_dataset`: k-means
  coarse centers on 100 rows each, then one codebook fitted on the residuals cut into `pq_len` pieces (at most 256
  pieces per code).
- `vamana_prune`: JVector's `VamanaDiversityProvider.retainDiverse` on the GPU. Per chunk of nodes (gathered rows
  near 256 MB), the candidates' rows are gathered and two batched matrix multiplies give the node-candidate and
  candidate-candidate dot products, turned into JVector's similarities (1 / (1 + squared distance) from norms and
  dot products for L2, (1 + dot) / 2 otherwise). One thread per node then orders its candidates (insertion sort,
  repeats dropped) and runs the passes alpha = 1.0, 1.2, ... over them, reading the similarities from device
  memory. `vamana_build` prunes the k-NN graph, adds each node's incoming edges of lowest rank (the reverse-edge
  passes of `cagra_optimize`, shared) and prunes the union again.
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
