# Changelog

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
