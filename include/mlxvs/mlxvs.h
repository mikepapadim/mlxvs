// Copyright © 2026 mikepapadim

// mlxvs: GPU k-nearest-neighbor search and k-NN graph construction on Apple
// silicon, built on MLX.

#pragma once

#include <utility>
#include <vector>

#include "mlx/mlx.h"

#define MLXVS_API __attribute__((visibility("default")))

namespace mlxvs {

namespace mx = mlx::core;

/** How two rows compare. */
enum class Metric {
  /** Dot product, larger is closer. */
  InnerProduct,
  /** Squared Euclidean distance, smaller is closer. */
  L2,
  /** Cosine similarity, larger is closer. Rows are normalized on the fly. */
  Cosine,
};

/** The k nearest rows of each query, nearest first. */
struct Neighbors {
  /** Row indices into the base, int32 [nq, k]; -1 where fewer exist. */
  mx::array ids;

  /**
   * The dot product, cosine similarity or squared distance of each neighbor,
   * float32 [nq, k]. A missing neighbor has -inf, or +inf for L2, so every row
   * stays sorted nearest first.
   */
  mx::array distances;
};

/** Options of brute_force_knn. */
struct KnnOptions {
  /**
   * Base rows scored per step. Queries are taken in blocks too, so one step's
   * [block, tile_rows] score matrix stays at about 1 GB.
   */
  int tile_rows = 65536;
};

/**
 * Exact k-nearest neighbors of every row of queries [nq, d] among the rows of
 * base [n, d] (any real dtype, computed in float32). Each tile of base rows is
 * scored with a matrix multiply and a fused radix-select kernel keeps the best
 * k per query across tiles. Requires 1 <= k <= 256. Runs on the GPU.
 */
MLXVS_API Neighbors brute_force_knn(
    const mx::array& base,
    const mx::array& queries,
    int k,
    Metric metric,
    const KnnOptions& options = {},
    mx::StreamOrDevice s = {});

/** Options of kmeans_fit. */
struct KMeansOptions {
  /** Lloyd iterations. */
  int max_iter = 10;

  /** Rows of the random sample fitted on, at least 64 per cluster. */
  int sample_rows = 100000;

  unsigned long long seed = 42;
};

/**
 * Fits k-means (squared Euclidean, k-means++ start, Lloyd iterations) to a
 * random sample of x [n, d] (any real dtype). Returns the centroids [clusters,
 * d].
 */
MLXVS_API mx::array kmeans_fit(
    const mx::array& x,
    int clusters,
    const KMeansOptions& options = {},
    mx::StreamOrDevice s = {});

/**
 * Returns the m nearest centroids of every row of x, nearest first, int32
 * [n, m]. m = 1 is the usual prediction; m > 1 assigns rows to overlapping
 * clusters, as batched all-neighbors does.
 */
MLXVS_API mx::array kmeans_predict(
    const mx::array& x,
    const mx::array& centroids,
    int m = 1,
    int tile_rows = 65536,
    mx::StreamOrDevice s = {});

/** Options of nn_descent, named after cuVS's nn_descent::index_params. */
struct NnDescentOptions {
  /** The most rounds after the random start; fewer once the lists settle. */
  int max_iterations = 24;

  /**
   * Stop after a round that replaced fewer than termination_threshold * n *
   * list size entries.
   */
  double termination_threshold = 0.0001;

  /**
   * Score the joins from float16 rows with SIMD-group matrices, accumulating in
   * float32; otherwise from float32 rows. Returned distances are recomputed in
   * float32 either way.
   */
  bool fp16 = true;

  unsigned long long seed = 42;

  /**
   * If set, receives the number of list entries replaced in each round, the
   * first four being the random start.
   */
  std::vector<long long>* updates = nullptr;
};

/**
 * Approximate k-nearest-neighbor graph of rows [n, d] (any real dtype) by
 * NN-Descent: the k nearest other rows of every row, nearest first, with the
 * ids and distances of brute_force_knn. Lists hold max(64, k) entries. Up to
 * 4096 rows the graph is computed exactly instead (as fast, and complete).
 * Requires 1 <= k <= 128 and n >= 2. Not deterministic: concurrent writes to
 * hashed proposal slots decide which colliding entry survives. Runs on the GPU.
 */
MLXVS_API Neighbors nn_descent(
    const mx::array& rows,
    int k,
    Metric metric,
    const NnDescentOptions& options = {},
    mx::StreamOrDevice s = {});

/** How all_neighbors builds the k-NN graph of each cluster. */
enum class GraphBuild {
  /** Exact: brute_force_knn of the cluster against itself. */
  BruteForce,
  /** Approximate: nn_descent. */
  NnDescent,
};

/** Options of all_neighbors, named after cuVS's all_neighbors::index_params. */
struct AllNeighborsOptions {
  GraphBuild algo = GraphBuild::NnDescent;

  /**
   * k-means clusters the rows are split into. 0: as many as the GPU's memory
   * asks for (one when the rows fit). 1: all rows at once.
   */
  int n_clusters = 0;

  /**
   * Clusters each row joins (its nearest centroids), so that neighbors across a
   * cluster boundary are found. Ignored when n_clusters is 1.
   */
  int overlap_factor = 2;

  /**
   * Clusters of at most this many rows are built exactly (brute force) even
   * with algo NnDescent: as fast, and exact.
   */
  int brute_force_below = 4096;

  /**
   * The most rows of one cluster; a larger k-means cluster is clustered again.
   * 0: from the GPU's recommended working set.
   */
  long long max_cluster_rows = 0;

  /** Print the clusters and the progress to stderr. */
  bool verbose = false;

  KMeansOptions kmeans;
  NnDescentOptions nn_descent;
  KnnOptions brute_force;
};

/**
 * k-nearest-neighbor graph of rows [n, d] (any real dtype): the k nearest other
 * rows of every row, nearest first, with the ids and distances of
 * brute_force_knn.
 *
 * With n_clusters > 1 the rows are clustered by k-means (on unit rows for
 * Cosine), each row joins its overlap_factor nearest clusters, the graph of
 * every cluster is built on its own (exact or NN-Descent), and each row's lists
 * from its clusters are merged (duplicates removed). Only one cluster's rows
 * and working set are on the GPU at a time, so this builds graphs of datasets
 * whose NN-Descent state does not fit at once. Requires 1 <= k <= 128 with
 * NnDescent (255 with BruteForce) and n >= 2. Runs on the GPU.
 */
MLXVS_API Neighbors all_neighbors(
    const mx::array& rows,
    int k,
    Metric metric,
    const AllNeighborsOptions& options = {},
    mx::StreamOrDevice s = {});

/**
 * Rows that stay on the host: n rows of d float32, row-major, e.g. a
 * memory-mapped file larger than memory. They must outlive the call.
 */
struct HostRows {
  const float* data;
  long long n;
  int d;
};

/**
 * all_neighbors of rows read where they are: only a k-means sample, one tile
 * during assignment and one cluster at a time are copied to the GPU, so the
 * rows need not fit in memory. Each cluster's rows are read from the host once
 * per cluster they belong to.
 */
MLXVS_API Neighbors all_neighbors(
    const HostRows& rows,
    int k,
    Metric metric,
    const AllNeighborsOptions& options = {},
    mx::StreamOrDevice s = {});

/**
 * One merge step of brute_force_knn, exposed for tests and benchmarks: the best
 * k of each row of scores [nq, t] (ids id_offset + column) together with the
 * running lists ids and best [nq, k], which are ignored when first is set.
 * Returns the merged ids and scores, best first.
 */
MLXVS_API std::pair<mx::array, mx::array> topk_merge(
    const mx::array& scores,
    const mx::array& ids,
    const mx::array& best,
    int id_offset,
    bool first,
    int k,
    mx::StreamOrDevice s = {});

} // namespace mlxvs
