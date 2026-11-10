# mlxvs

[![CI](https://github.com/mikepapadim/mlxvs/actions/workflows/ci.yml/badge.svg)](https://github.com/mikepapadim/mlxvs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

GPU k-nearest-neighbor search and k-NN graph construction for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx). The same operations and parameter names as NVIDIA
[cuVS](https://github.com/rapidsai/cuvs), on the Mac's GPU: exact k-NN, k-means, NN-Descent and batched
all-neighbors graphs. It takes NumPy arrays or MLX arrays and runs on any M-series Mac.

Use it for the k-NN graph behind UMAP, t-SNE, clustering and deduplication, for exact k-NN ground truth, or as the
build step of a graph index (HNSW, Vamana, CAGRA).

## What's in it

| Function | What it does | cuVS counterpart |
|---|---|---|
| `brute_force_knn(base, queries, k)` | Exact k-NN: matrix multiply and a fused top-k kernel, tiled over the base | `brute_force::search` |
| `kmeans_fit(x, clusters)`, `kmeans_predict(x, centroids, m)` | k-means++ and Lloyd; the `m` nearest centroids per row | `cluster::kmeans` |
| `nn_descent(rows, k)` | Approximate k-NN graph of the rows, FP16 SIMD-group joins | `nn_descent::build` |
| `all_neighbors(rows, k)` | k-NN graph built in overlapping k-means clusters, one on the GPU at a time; reads a NumPy array or memory-mapped file larger than memory where it is | `all_neighbors::build` |
| `NearestNeighbors(n_neighbors).fit(X).kneighbors(Q)` | scikit-learn's interface over the above, NumPy in and out | |

Metrics: `"l2"` (squared), `"cosine"`, `"inner_product"`. Next: graph pruning (CAGRA, Vamana) and graph search;
see [Roadmap](#roadmap).

## 1. Install

Needs an Apple-silicon Mac on macOS 14 or later (as MLX) and Python 3.10+.

```bash
pip install git+https://github.com/mikepapadim/mlxvs
```

pip builds the extension against your `mlx` wheel; that needs the Xcode Command Line Tools
(`xcode-select --install`) and takes about a minute.

## 2. Use

```python
import numpy as np
import mlxvs

base = np.random.rand(1_000_000, 128).astype(np.float32)
queries = base[:10_000]

# exact k-NN: ids [nq, k], distances [nq, k], nearest first
ids, dist = mlxvs.brute_force_knn(base, queries, k=10, metric="l2")

# approximate k-NN graph: every row's 32 nearest other rows
graph, graph_dist = mlxvs.nn_descent(base, k=32, metric="cosine")

# the same graph in overlapping k-means clusters, as many as the GPU's memory asks for
graph, graph_dist = mlxvs.all_neighbors(base, k=32, metric="cosine")

graph = np.asarray(graph)  # results are mlx arrays; np.asarray converts without a copy
```

- Inputs: MLX arrays, NumPy arrays, or anything `np.asarray` accepts, in any real dtype (computed in float32).
- Outputs: `int32` ids, `-1` where fewer neighbors exist; `float32` distances, squared for `"l2"`, similarities
  (larger is closer) for `"cosine"` and `"inner_product"`.
- `help(mlxvs.nn_descent)` and the others document every parameter.

Data larger than memory: pass a `numpy.memmap`. The rows stay on disk; `all_neighbors` copies one cluster at a time
to the GPU.

```python
rows = np.memmap("embeddings.f32", dtype=np.float32, mode="r").reshape(-1, 1024)
graph, graph_dist = mlxvs.all_neighbors(rows, k=32, metric="cosine", verbose=True)
```

Coming from scikit-learn, change the import:

```python
nn = mlxvs.NearestNeighbors(n_neighbors=10, metric="euclidean").fit(base)
distances, indices = nn.kneighbors(queries)   # NumPy arrays, scikit-learn's units
distances, indices = nn.kneighbors()          # the base rows' own neighbors
```

Examples: [UMAP from a GPU k-NN graph](examples/umap_knn.py) and
[exact ground truth for an ANN benchmark](examples/ground_truth.py) (`.fvecs` in, `.ivecs` out).

C++: the same functions in [`include/mlxvs/mlxvs.h`](include/mlxvs/mlxvs.h), taking `mx::array` and a trailing
`mx::StreamOrDevice s = {}` as MLX ops do. In CMake (MLX found through `CMAKE_PREFIX_PATH`, e.g. Homebrew's):

```cmake
add_subdirectory(mlxvs)          # or FetchContent
target_link_libraries(app PRIVATE mlxvs)
```

## 3. Run the benchmarks

```bash
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
pip install faiss-cpu pynndescent   # optional CPU baselines
python bench/bench_graph.py sift/sift_base.fvecs --k 32 --clusters 1,8 --baselines faiss,pynndescent
```

It builds the k-NN graph with each method and scores 2,000 random rows against the exact neighbors.

## Coming from cuVS

| mlxvs | cuVS | Note |
|---|---|---|
| `nn_descent(rows, k, ...)` | `nn_descent::index_params::graph_degree` | cuVS default 64 |
| lists of `max(64, k)` entries | `intermediate_graph_degree` | cuVS default 128 |
| `max_iterations` = 24 | `max_iterations` = 20 | mlxvs samples 16 new neighbors per round and stops early when the lists settle |
| `termination_threshold` = 1e-4 | `termination_threshold` = 1e-3 | tuned for graph quality in JVector; pass 1e-3 for cuVS's behaviour |
| `all_neighbors(..., n_clusters, overlap_factor)` | `all_neighbors::index_params::{n_clusters, overlap_factor}` | overlap 2 as cuVS; `n_clusters` defaults to 0 (chosen from the GPU's memory), cuVS's to 1 |
| `algo="brute_force"` / `"nn_descent"` | `all_neighbors::index_params::graph_build_params` | cuVS also offers IVF-PQ |
| `"cosine"` returns the cosine similarity | `CosineExpanded` returns 1 - cosine | similarities are "larger is closer"; `"l2"` is squared, as `L2Expanded` |
| `int32` ids | `int64_t` ids | n < 2^31 rows |

## Roadmap

1. Graph pruning: CAGRA rank-based and Vamana alpha pruning, export to HNSW and JVector layouts (`cagra::build`).
2. Graph search (`cagra::search`).
3. Wheels on PyPI; a C API for other languages (Java via Panama, as cuVS-java does).

How the kernels work: [docs/DESIGN.md](docs/DESIGN.md). Building, testing and code style:
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. The NN-Descent kernels are a port of JVector's `jvector-gpu` module (Apache-2.0, DataStax, Inc.); see
[NOTICE](NOTICE).
