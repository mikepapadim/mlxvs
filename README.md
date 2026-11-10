# mlxvs

GPU k-nearest-neighbor search and k-NN graph construction for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx). The API follows NVIDIA [cuVS](https://github.com/rapidsai/cuvs) where
the two correspond.

Status: early. C++ and Python APIs for exact k-NN, k-means and NN-Descent, tested; all-neighbors batching and graph
pruning next. Private while the benchmarks against FAISS and HNSW baselines are run.

## Why

Apple-silicon GPUs have brute-force and IVF search through FAISS's Metal backend. They have no GPU *graph
construction*: building the approximate k-NN graph (NN-Descent, cuVS all-neighbors with batching) and pruning it
into a search graph (CAGRA, Vamana/HNSW-style diversity). That is the expensive part of building a graph index.
mlxvs targets it, starting from kernels proven in JVector's GPU build on Metal (TornadoVM), where they match the
CUDA build's graph quality.

## Usage

```python
import mlx.core as mx
import mlxvs

base = mx.random.normal((1_000_000, 128))
queries = mx.random.normal((10_000, 128))

# exact k-NN: ids int32 [nq, k], distances float32 [nq, k], nearest first
ids, dist = mlxvs.brute_force_knn(base, queries, k=10, metric=mlxvs.Metric.l2)

# approximate k-NN graph of the base (every row's 64 nearest other rows)
graph, graph_dist = mlxvs.nn_descent(base, k=64, metric=mlxvs.Metric.cosine)

# k-means for batching: the 2 nearest of 8 centroids for every row
centroids = mlxvs.kmeans_fit(base, 8)
assignment = mlxvs.kmeans_predict(base, centroids, m=2)
```

The C++ API is the same set of functions in `include/mlxvs/mlxvs.h`; every operation takes a trailing
`mx::StreamOrDevice s = {}` as MLX ops do. Metal kernels only: a CPU stream is rejected.

## Plan

| | Operation | cuVS counterpart | State |
|---|---|---|---|
| 1 | Exact k-NN: MLX matmul + fused radix-select top-k merge | `brute_force::search` | done, tested |
| 2 | k-means: k-means++ start, Lloyd with GEMM assignment and one-hot GEMM update; m-nearest prediction for overlapping batches | `cluster::kmeans` | done, tested |
| 3 | NN-Descent: port of JVector's Metal kernels (random start, S = 16 samples, 64-member FP16 SIMD-group join, hashed proposals, bitonic merge) | `nn_descent::build` | done, tested: recall@k 0.98-1.00 on clustered data and siftsmall (k = 32/96, all metrics); about 0.78-0.90 on iid Gaussian d = 64/128, the algorithm's known worst case |
| 4 | All-neighbors with batching (k-means clusters, overlap) for data larger than memory | `all_neighbors::build` | |
| 5 | Graph pruning (CAGRA rank-based / Vamana alpha), export to HNSW/JVector layouts | `cagra::build` (optimize) | |
| 6 | Graph search | `cagra::search` | later |

Bindings: C++ and Python (nanobind, `NB_DOMAIN mlx`, so arrays pass to and from `mlx.core` without copies).
Planned: a C API for other languages (Java via Panama, as cuVS-java does).

### cuVS parameter mapping

| mlxvs | cuVS | note |
|---|---|---|
| `nn_descent(rows, k, ...)` | `nn_descent::index_params::graph_degree` | cuVS default 64 |
| lists of `max(64, k)` entries | `intermediate_graph_degree` | cuVS default 128 |
| `max_iterations` = 24 | `max_iterations` = 20 | mlxvs samples 16 new neighbors per round and stops early when the lists settle, so it is given more rounds |
| `termination_threshold` = 1e-4 | `termination_threshold` = 1e-3 | mlxvs's default is the value tuned for graph quality in JVector; set 1e-3 for cuVS's behaviour |
| `Metric.cosine` returns the cosine similarity | `CosineExpanded` returns 1 - cosine | similarities in mlxvs are "larger is closer"; L2 is the squared distance, as `L2Expanded` |
| `kmeans_predict(x, c, m)` | `all_neighbors::index_params::{n_clusters, overlap_factor}` | cuVS defaults: 1 cluster, overlap 2 |
| `int32` ids | `int64_t` ids | int64 ids planned for the all-neighbors API |

## Design notes

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
  joins only affect which neighbors are found.
- MLX has no stable C++ ABI: mlxvs is built against one MLX release. The Python extension links the `mlx`
  wheel's own `libmlx` (located with `python -m mlx --cmake-dir`), so one process holds one copy.

## Code style

MLX's: its `.clang-format` (2-space indent, 80 columns, `// Copyright © ...` header lines), `[mlxvs::op] Message.`
error strings, every public operation taking a trailing `StreamOrDevice s = {}`, ruff for Python, and its
pre-commit hooks (`pre-commit install`).

## Build

C++:

```bash
brew install mlx cmake
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build
ctest --test-dir build                 # exact k-NN vs a CPU reference; NN-Descent recall vs exact k-NN
./build/bench_knn base.f32 1024 4096 65536 96
```

Python (builds against the interpreter's `mlx` wheel):

```bash
pip install .
pytest python/tests
```

CI (`.github/workflows/ci.yml`) runs both on `macos-15` Apple-silicon runners.

## Licence

Apache-2.0. The NN-Descent kernels are a port of JVector's `jvector-gpu` module (Apache-2.0, DataStax, Inc.); see
NOTICE.
