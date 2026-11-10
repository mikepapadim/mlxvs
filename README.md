# mlxvs

GPU k-nearest-neighbor search and k-NN graph construction for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx). The API follows NVIDIA cuVS where it can.

Status: early, local. Not published.

## Why

Apple-silicon GPUs now have brute-force and IVF search through FAISS's Metal backend. They have no GPU
*graph construction*: building the approximate k-NN graph (NN-Descent, cuVS all-neighbors with batching) and
pruning it into a search graph (CAGRA, Vamana/HNSW-style diversity). That is the expensive part of building a
graph index. mlxvs targets it, starting from kernels already proven in JVector's GPU build on Metal (TornadoVM),
where they match the CUDA build's graph quality.

## Plan

| | Operation | cuVS counterpart | State |
|---|---|---|---|
| 1 | Exact k-NN: MLX matmul + fused radix-select top-k merge | `brute_force::search` | done, tested |
| 2 | k-means: k-means++ start, Lloyd with GEMM assignment and one-hot GEMM update; m-nearest prediction for overlapping batches | `cluster::kmeans` | done, tested |
| 3 | NN-Descent: port of JVector's Metal kernels (random start, S = 16 samples, 64-member FP16 SIMD-group join, hashed proposals, bitonic merge) | `nn_descent::build` | done, tested: recall@k 0.98-1.00 on clustered data and siftsmall (k = 32/96, all metrics); about 0.78-0.90 on iid Gaussian d = 64/128, the algorithm's known worst case |
| 4 | All-neighbors with batching (k-means clusters, overlap) for data larger than memory | `all_neighbors::build` | |
| 5 | Graph pruning (CAGRA rank-based / Vamana alpha), export to HNSW/JVector layouts | `cagra::build` (optimize) | |
| 6 | Graph search | `cagra::search` | later |

Bindings: C++ (this), a C API for other languages (Java via Panama, as cuVS-java does), Python as an MLX
extension.

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
  `changed` for the termination test (cuVS's: fewer than 1e-4 n K entries replaced). Returned distances are
  recomputed in FP32 and the lists re-sorted, so FP16 joins only affect which neighbors are found.
- MLX has no stable C++ ABI: mlxvs is built against one MLX version (`find_package(MLX)`).

## Code style

MLX's: its `.clang-format` (2-space indent, 80 columns, `// Copyright © ...` header lines), `[mlxvs::op] Message.`
error strings, every public operation taking a trailing `StreamOrDevice s = {}`, and its pre-commit hooks
(`pre-commit install`). GPU only: the kernels are Metal.

## Build

```bash
brew install mlx cmake
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build
./build/test_knn                       # exact k-NN against a CPU reference
./build/test_nnd                       # NN-Descent recall vs exact k-NN (MLXVS_FVECS=base.fvecs adds real data)
./build/bench_knn base.f32 1024 4096 65536 96
```

## Licence

Apache-2.0.
