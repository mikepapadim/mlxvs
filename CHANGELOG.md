# Changelog

## 0.12.0 (2026-11-12)

- CAGRA-Q: `CagraIndex.compress()` (C++ `vpq_build`, `graph_search_vpq`) replaces the stored rows by cuVS's VPQ
  codes: each row's nearest coarse center (`sqrt(n)` of them) plus `ceil(d / 4)` codes of its residual in one
  shared 256-entry codebook. The graph search computes distances from the codes (both single- and multi-CTA, filters
  included); `search(..., refine_ratio=4, dataset=rows)` re-ranks exactly, from rows on the GPU or in a NumPy array
  or `numpy.memmap`. SIFT1M: 36 MB instead of 505 MB, compressed in 6.3 s; 234,000 queries/s from the codes alone
  (recall@10 0.716), recall@10 0.981 at 103,000 queries/s refined (`itopk_size` 128).
- `search_kernels.h`: the VPQ kernels are the same sources with the distance function swapped.

## 0.11.0 (2026-11-12)

- `IvfPqIndex` (C++ `ivf_pq_build`, `ivf_pq_search`, `ivf_pq_extend`), as cuVS's IVF-PQ: k-means lists, rows
  stored as `pq_dim` codes of `pq_bits` (4 to 8) of their residual to the center, per-subspace codebooks, cuVS's
  `pq_dim` heuristic (zero columns pad the dimension to a multiple of `pq_dim`). Searches build a float32 lookup
  table per (query, list) in chunks of subspaces and sum it over each row's codes. SIFT1M: 63 MB of codes for
  505 MB of rows, built in 21.7 s; with `refine_ratio=4` recall@10 0.954 at 18,300 queries/s (`n_probes` 20) and
  0.993 at 8,100 (50), as IVF-Flat's.
- `refine` (C++ and Python), as cuVS's: exact re-ranking of candidate ids. From Python the rows may be a NumPy
  array or `numpy.memmap`: only the candidates' rows are read, on the host.

## 0.10.0 (2026-11-12)

- `CagraIndex.extend` (C++ `cagra_extend`), as cuVS's `cagra::extend`: each new row's `2 * degree` nearest rows
  come from a graph search and are pruned to `degree` by rank-based detour counts; each of its first `degree / 2`
  neighbors gives it the second-half slot of its neighbor with the most incoming edges; its list interleaves its
  own neighbors with those displaced. 18,000 rows extended by 2,000 search as well as an index built on all
  20,000 (recall@10 0.9994 against 0.9998).
- `IvfFlatIndex` (C++ `ivf_flat_build`, `ivf_flat_search`, `ivf_flat_extend`), as cuVS's IVF-Flat: k-means lists,
  the `n_probes` lists of nearest center scanned exactly. The scan groups (query, list) pairs by list, so each
  threadgroup reads a list's rows once for up to 16 queries. SIFT1M, 1024 lists, built in 9.5 s: recall@10 0.954
  at 35,700 queries/s (`n_probes` 20), 0.993 at 17,300 (50). `save` and `load` as one `.safetensors` file.

## 0.9.1 (2026-11-12)

- `brute_force_knn` (and every exact k-NN built on it): when more candidates tie with the k-th score than are
  kept, the ones of lowest id are kept. Before, the threads that arrived first decided, so ties (frequent with
  float32 distances of data far from the origin) could give different neighbors from run to run. scikit-learn's
  `check_fit_idempotent` caught it through the sklearn-ann transformer.

## 0.9.0 (2026-11-12)

- A filter per query, as cuVS's bitmap filter: `filter=` also takes bool `[nq, n]` or `uint32 [nq, ceil(n / 32)]`;
  C API `mlxvs_index_search_bitmap`.
- C API byte datasets: `mlxvs_index_build_typed` with `MLXVS_INT8` or `MLXVS_UINT8` rows, stored as they are.
- The C API's search functions share one implementation.

## 0.8.0 (2026-11-12)

- `save_hnswlib` and `save_jvector` in C++ (`include/mlxvs/mlxvs.h`) and the C API (`mlxvs_index_save_hnswlib`,
  `mlxvs_index_save_jvector`); `CagraIndex.save_hnswlib` and `save_jvector` call them and write the same bytes as
  before. From Java: build on the GPU through the C API and write a file JVector 4 searches
  (`examples/java_ffm`, then `examples/jvector`).

## 0.7.0 (2026-11-12)

- Byte datasets, as cuVS's int8 and uint8 datasets: `graph_search` reads `int8` and `uint8` rows as they are
  (four columns per 32-bit load), and `CagraIndex.build(..., dtype="uint8")` (or `"int8"`) stores rows that are
  integers in range, a quarter of float32's memory. Results equal float32's on integer data. SIFT1M quantized to
  uint8 (its file holds normalized floats, so this loses precision): 120 MB, 478,000 queries/s at recall@10 0.925
  and 267,000 at 0.965 (float32: 481 MB, 259,000 at 0.935 and 148,000 at 0.980).
- `bench_search.py --dtype uint8` and `int8`.

## 0.6.0 (2026-11-12)

- Filtered search, as cuVS's bitset filter: `graph_search(..., filter=)` (and `CagraIndex.search`) takes a bool mask
  `[n]` or a bitset `uint32 [ceil(n / 32)]`. Rejected rows are still traversed but leave the list once expanded
  and are never returned; multi-CTA's list grows with the filtering rate as cuVS's (`filtering_rate`, computed from
  the filter by default). SIFT1M, 10% of rows allowed: recall@10 0.978 multi-CTA (single-CTA, as in cuVS, 0.74 at
  `itopk_size` 64 and 0.86 at 256).
- Seed nodes, as cuVS's `seed_ptr`: `seeds=` `[nq, num_seeds]` start nodes per query.
- C API: `mlxvs_index_search_filtered` and `mlxvs_index_search_seeded`.
- The README's roadmap records an experiment left out: Vamana graphs from search-path candidates (DiskANN's) stay
  below CAGRA's graph on SIFT1M for four to five times the build time.

## 0.5.0 (2026-11-12)

- A C API, `include/mlxvs/mlxvs_c.h` in `libmlxvs_c.dylib`: an index handle (`mlxvs_index_build`, `_search`,
  `_save`, `_load`, `_from_graph`, `_graph`), `mlxvs_brute_force_knn`, parameter structs with cuVS's defaults,
  status codes and `mlxvs_last_error`. Only the `mlxvs_*` symbols are exported. Indexes are `CagraIndex`'s
  `.safetensors` files: built in Python, they load in C, and the other way round.
- `examples/java_ffm`: mlxvs from Java 22+ through the Foreign Function and Memory API, no JNI.
- `cmake --install` installs the C library and header.

## 0.4.0 (2026-11-12)

- `vamana_prune`: Vamana's robust pruning with JVector's diversity rule (`VamanaDiversityProvider`: passes with
  alpha = 1.0, 1.2, ... over the candidates by similarity, JVector's similarities) on the GPU; it equals a CPU
  reference of JVector's rule on every row of the tests.
- `vamana_build`: k-NN graph, `vamana_prune`, then each node's list with its incoming edges of lowest rank pruned
  again; `CagraIndex.build(..., graph="vamana", alpha=1.2)`. SIFT1M, degree 32 from 64 candidates: built in 15 s,
  recall@10 0.974 at `itopk_size` 64 (CAGRA's graph 0.980), and the same in JVector after `save_jvector`.
- The reverse-edge passes of `cagra_optimize` are shared with `vamana_build`.

## 0.3.0 (2026-11-12)

- `graph_search` multi-CTA, as cuVS's `search_multi_cta`: `max(search_width, itopk_size / 32)` threadgroups per
  query, each a list of 32 expanding one node per iteration, sharing the set of expanded nodes in device memory; a
  top-k over their lists gives the result. `algo="auto"` (default) picks it as cuVS does: for fewer than two queries
  per GPU core (read from IOKit) or `itopk_size` over 256 (up to 1024). One SIFT1M query: 0.45 ms at recall@10
  0.996 to 0.999, where single-CTA took 2.2 ms for 0.999 and hnswlib on one core 0.62 ms for 0.998.
- Distances by teams of lanes, as CAGRA: 4 to 32 lanes per candidate, four columns per load (`team_size`, by
  default chosen from the dimension as cuVS). Batches about twice as fast: SIFT1M 143,300 queries/s at recall@10
  0.980 (was 76,400 at 0.976), ada-002 1M 14,400 at 0.979 (was 10,000).
- `graph_search` options `algo`, `team_size` and `min_iterations`, as cuVS's `search_params`; `SearchAlgo` enum.
- `CagraIndex.save_jvector`: the index in JVector's on-disk format (format 6, JVector 4.0), searchable from Java;
  `examples/jvector` loads and searches it with JVector 4.0.2 (SIFT1M recall@10 0.994 with `rerankK` 128).
- `bench_search.py --algo` and `--latency N` (queries one at a time, against hnswlib on one thread);
  `bench/plot_bench.py` draws the README chart from its output.

## 0.2.1 (2026-11-12)

Closer to CAGRA (Ootomo et al., arXiv:2308.15136) as cuVS implements it:

- `cagra_optimize`: a node with more reverse edges than `graph_degree` keeps those of lowest rank (the rank of
  the edge in its source's pruned list), as cuVS, which adds them one rank at a time; it kept the first to arrive.
  The GPU graph now equals the CPU reference wherever the cut does not fall among edges of one rank.
- `graph_search`: the random start fills `itopk_size + search_width * degree` slots, each the nearest of
  `num_random_samplings` random nodes (cuVS's distillation), instead of `num_random_samplings` rounds of one
  round's candidates. SIFT1M, degree 32: recall@10 0.935, 0.981, 0.995, 0.999 at itopk 32, 64, 128, 256 (was
  0.931, 0.976, 0.993, 0.998), about 5% fewer queries per second at the same itopk.
- `graph_search`: `max_iterations = 0` is cuVS's automatic value (`itopk_size / search_width` plus the hops to
  reach n nodes at `degree / 2` new nodes per hop), the random start counting as the first iteration.

## 0.2.0 (2026-11-11)

- `CagraIndex`: a dataset and its CAGRA graph stored once for search (`build`, `search`, `save` and `load` as one
  `.safetensors` file); the dataset in float32 or float16, normalized for cosine.
- `CagraIndex.save_hnswlib`: the index in hnswlib's binary format (the CAGRA graph as its bottom layer, entered at
  the medoid), as cuVS's `serialize_to_hnswlib`: build on the GPU, search on any CPU with hnswlib.
- `graph_search` reads a float16 dataset as it is: half the memory, and faster searches with the same recall.
- NumPy views with strides (such as the rows of an `.fvecs` file) are made contiguous once on conversion; MLX kept
  their strides, and every kernel call copied the whole array again (about 9 ms per call on SIFT1M).
- `bench_search.py` scores recall from the true distances of the returned ids, and builds a `CagraIndex` with
  `--dtype`.

## 0.1.0 (2026-11-11)

- `brute_force_knn`: exact k-NN, tiled matrix multiply and a fused top-k kernel; queries taken in blocks, so a
  step's score matrix stays near 1 GB even for an exact k-NN graph (queries = base).
- `kmeans_fit`, `kmeans_predict`: k-means++ and Lloyd; the m nearest centroids per row.
- `nn_descent`: approximate k-NN graph (FP16 SIMD-group joins); up to 4096 rows built exactly.
- `all_neighbors`: k-NN graph in overlapping k-means clusters, NN-Descent or exact per cluster.
- Out-of-core `all_neighbors`: host arrays and `numpy.memmap` files larger than memory are read a cluster at a
  time; `n_clusters = 0` (default) sizes the clusters from the GPU's memory, oversized clusters are clustered again.
- `cagra_optimize`, `cagra_build`: CAGRA's graph optimization (rank-based pruning by two-hop detours, reverse
  edges) on the GPU; `cagra_build` prunes an `all_neighbors` graph, rows on the host included.
- `graph_search`: approximate k-NN of queries over any fixed-degree graph, CAGRA's single-CTA search (one
  threadgroup per query, hashed visited set in threadgroup memory, bitonic sort and merge-path merge).
- `NearestNeighbors`: scikit-learn style `fit` / `kneighbors` returning NumPy arrays.
- Inputs: MLX or NumPy arrays and array-likes in any real dtype; metrics by name.
- Examples: UMAP from a GPU k-NN graph, exact ground truth as `.ivecs`.
