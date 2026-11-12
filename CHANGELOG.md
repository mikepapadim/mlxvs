# Changelog

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
