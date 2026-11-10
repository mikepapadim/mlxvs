# Changelog

## 0.1.0 (unreleased)

- `brute_force_knn`: exact k-NN, tiled matrix multiply and a fused top-k kernel.
- `kmeans_fit`, `kmeans_predict`: k-means++ and Lloyd; the m nearest centroids per row.
- `nn_descent`: approximate k-NN graph (FP16 SIMD-group joins); up to 4096 rows built exactly.
- `all_neighbors`: k-NN graph in overlapping k-means clusters, NN-Descent or exact per cluster.
- Out-of-core `all_neighbors`: host arrays and `numpy.memmap` files larger than memory are read a cluster at a
  time; `n_clusters = 0` (default) sizes the clusters from the GPU's memory, oversized clusters are clustered again.
- `NearestNeighbors`: scikit-learn style `fit` / `kneighbors` returning NumPy arrays.
- Inputs: MLX or NumPy arrays and array-likes in any real dtype; metrics by name.
- Examples: UMAP from a GPU k-NN graph, exact ground truth as `.ivecs`.
