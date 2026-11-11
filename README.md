# mlxVS

**cuVS for your Mac: GPU vector search on Apple silicon.**

[![CI](https://github.com/mikepapadim/mlxvs/actions/workflows/ci.yml/badge.svg)](https://github.com/mikepapadim/mlxvs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A k-NN graph of a million vectors in about 9 seconds, then 40,000 queries per second at 94% recall, on a MacBook
Pro's GPU. mlxVS brings the operations and parameter names of NVIDIA [cuVS](https://github.com/rapidsai/cuvs) to
the Mac, built on [MLX](https://github.com/ml-explore/mlx): exact k-NN, k-means, NN-Descent, batched all-neighbors
graphs and CAGRA-style graph search. NumPy or MLX arrays in, any M-series Mac.

```python
import mlxvs
graph, _ = mlxvs.nn_descent(vectors, k=32)                       # build: 1M x 128 in ~9 s
ids, dist = mlxvs.graph_search(vectors, graph, queries, k=10)    # search: ~40k queries/s
```

Use it for the k-NN graph behind UMAP, t-SNE, clustering and deduplication, for exact k-NN ground truth, for
approximate search over a graph, or as the build step of a graph index (HNSW, Vamana, CAGRA).

## What's in it

| Function | What it does | cuVS counterpart |
|---|---|---|
| `brute_force_knn(base, queries, k)` | Exact k-NN: matrix multiply and a fused top-k kernel, tiled over the base | `brute_force::search` |
| `kmeans_fit(x, clusters)`, `kmeans_predict(x, centroids, m)` | k-means++ and Lloyd; the `m` nearest centroids per row | `cluster::kmeans` |
| `nn_descent(rows, k)` | Approximate k-NN graph of the rows, FP16 SIMD-group joins | `nn_descent::build` |
| `all_neighbors(rows, k)` | k-NN graph built in overlapping k-means clusters, one on the GPU at a time; reads a NumPy array or memory-mapped file larger than memory where it is | `all_neighbors::build` |
| `graph_search(base, graph, queries, k)` | Approximate k-NN of queries by beam search of a graph, one threadgroup per query (CAGRA's single-CTA search) | `cagra::search` |
| `NearestNeighbors(n_neighbors).fit(X).kneighbors(Q)` | scikit-learn's interface over the above, NumPy in and out | |

Metrics: `"l2"` (squared), `"cosine"`, `"inner_product"`. Next: graph pruning (CAGRA, Vamana); see
[Roadmap](#roadmap).

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

# approximate k-NN of the queries by searching the graph; itopk_size trades speed for recall
ids, dist = mlxvs.graph_search(base, graph, queries, k=10, metric="cosine", itopk_size=128)

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

Search: build a graph, then recall@10 against queries per second over a sweep of `itopk_size`, optionally against
hnswlib on the CPU.

```bash
pip install hnswlib   # optional CPU baseline
python bench/bench_search.py sift/sift_base.fvecs sift/sift_query.fvecs --degree 32 --baselines hnswlib
```

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
| `graph_search(base, graph, queries, k, ...)` | `cagra::search` | takes the dataset and graph directly, no index object yet |
| `itopk_size` = 64, `search_width` = 1 | `search_params::{itopk_size, search_width}` | same defaults; `itopk_size` at most 256 |
| `max_iterations` = 0 (automatic) | `search_params::max_iterations` | stops earlier once every list entry is expanded |
| `num_random_samplings` = 1 | `search_params::num_random_samplings` | random start nodes; no seed nodes yet |
| `int32` ids | `int64_t` ids | n < 2^31 rows |

## Roadmap

1. Graph pruning: CAGRA rank-based and Vamana alpha pruning, export to HNSW and JVector layouts (`cagra::build`).
2. An index object holding the dataset and graph; float16 and int8 datasets for search.
3. A C API for other languages (Java via Panama, as cuVS-java does).

How the kernels work: [docs/DESIGN.md](docs/DESIGN.md). Building, testing and code style:
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. The NN-Descent kernels are a port of JVector's `jvector-gpu` module (Apache-2.0, DataStax, Inc.); see
[NOTICE](NOTICE).
