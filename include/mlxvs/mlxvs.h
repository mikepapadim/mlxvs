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

/** Options of cagra_build, named after cuVS's cagra::index_params. */
struct CagraOptions {
  /** Degree of the k-NN graph built first (at most 128 with NN-Descent). */
  int intermediate_graph_degree = 64;

  /** Degree of the pruned graph. */
  int graph_degree = 32;

  /** How the k-NN graph is built: all_neighbors's options. */
  AllNeighborsOptions knn;
};

/**
 * CAGRA's graph optimization: prunes knn_graph [n, K] (ids, nearest first, as
 * nn_descent or all_neighbors return) to [n, graph_degree] for graph_search.
 * Rank-based pruning keeps the edges with the fewest two-hop detours (an edge
 * X -> Y has a detour through Z when Z comes before Y in X's list and Y before
 * it in Z's), ties by rank; then the second half of each list takes reverse
 * edges (Y -> X for kept X -> Y, by the edge's rank) not there yet, so that
 * nodes with few incoming edges stay reachable. Needs no distances. A node
 * receiving more than graph_degree reverse edges keeps those of lowest rank, as
 * cuVS; among edges of the rank where the cut falls, the first that arrive, so
 * the result can vary slightly between runs. Requires K <= 256. Runs on the
 * GPU.
 */
MLXVS_API mx::array cagra_optimize(
    const mx::array& knn_graph,
    int graph_degree,
    mx::StreamOrDevice s = {});

/**
 * A search graph of rows [n, d]: all_neighbors with intermediate_graph_degree
 * neighbors, then cagra_optimize to graph_degree. Returns int32 [n,
 * graph_degree] for graph_search.
 */
MLXVS_API mx::array cagra_build(
    const mx::array& rows,
    Metric metric,
    const CagraOptions& options = {},
    mx::StreamOrDevice s = {});

/** cagra_build of rows left on the host, as all_neighbors(HostRows). */
MLXVS_API mx::array cagra_build(
    const HostRows& rows,
    Metric metric,
    const CagraOptions& options = {},
    mx::StreamOrDevice s = {});

/**
 * Vamana's robust pruning with JVector's diversity rule: of each row's
 * candidates [n, C] (ids; -1, out-of-range ids, the row itself and repeats are
 * skipped), ordered by similarity to the row, passes with alpha = 1.0, 1.2,
 * ... up to alpha keep a candidate unless one already kept is more similar to
 * it than alpha times its similarity to the row, until graph_degree are kept
 * (JVector's VamanaDiversityProvider; similarities as JVector's: 1 / (1 +
 * squared distance) for L2, (1 + dot) / 2 otherwise, on unit rows for
 * Cosine). Returns int32 [n, graph_degree], most similar first, -1 filled.
 * Requires C <= 256 and alpha >= 1. Runs on the GPU.
 */
MLXVS_API mx::array vamana_prune(
    const mx::array& rows,
    const mx::array& candidates,
    int graph_degree,
    Metric metric,
    float alpha = 1.2f,
    mx::StreamOrDevice s = {});

/** Options of vamana_build. */
struct VamanaOptions {
  /** Degree of the k-NN graph pruned first (at most 128 with NN-Descent). */
  int intermediate_graph_degree = 64;

  /** Degree of the result. */
  int graph_degree = 32;

  /** Vamana's alpha: larger keeps longer edges. JVector's default. */
  float alpha = 1.2f;

  /** How the k-NN graph is built: all_neighbors's options. */
  AllNeighborsOptions knn;
};

/**
 * A Vamana search graph of rows [n, d]: all_neighbors with
 * intermediate_graph_degree neighbors, vamana_prune to graph_degree, then each
 * node's list together with its graph_degree incoming edges of lowest rank
 * pruned again (Vamana's backward edges). Returns int32 [n, graph_degree] for
 * graph_search, -1 filled.
 */
MLXVS_API mx::array vamana_build(
    const mx::array& rows,
    Metric metric,
    const VamanaOptions& options = {},
    mx::StreamOrDevice s = {});

/** How graph_search lays out a query's work, as cuVS's cagra::search_algo. */
enum class SearchAlgo {
  /**
   * As cuVS: SingleCta for at least two queries per GPU core (and itopk_size at
   * most 256), else MultiCta.
   */
  Auto,
  /** One threadgroup per query: throughput for large batches. */
  SingleCta,
  /**
   * max(search_width, itopk_size / 32) threadgroups per query, each keeping 32
   * entries and expanding one node per iteration: latency for small batches,
   * and itopk_size beyond 256.
   */
  MultiCta,
};

/** Options of graph_search, named after cuVS's cagra::search_params. */
struct SearchOptions {
  SearchAlgo algo = SearchAlgo::Auto;

  /**
   * Length of each query's internal sorted list (rounded up to a multiple of
   * 32, at least k; at most 256 with SingleCta, 1024 with MultiCta). Longer:
   * higher recall, slower.
   */
  int itopk_size = 64;

  /** Nodes expanded per round. search_width * graph degree <= 512. */
  int search_width = 1;

  /**
   * Lanes computing one distance (4, 8, 16 or 32), each loading four columns
   * at a time. 0: as cuVS, by the dimension (8 near 128, 16 near 256, 32 from
   * 384 up).
   */
  int team_size = 0;

  /**
   * The most iterations, the random start counting as the first, as cuVS. 0:
   * itopk_size / search_width plus log base max(2, degree / 2) of n, rounded
   * up.
   */
  int max_iterations = 0;

  /**
   * The fewest iterations: a search does not end earlier for lack of nodes to
   * expand.
   */
  int min_iterations = 0;

  /**
   * The random start fills itopk_size + search_width * degree slots (degree
   * per threadgroup with MultiCta), each with the nearest of this many random
   * nodes (cuVS's distillation).
   */
  int num_random_samplings = 1;

  unsigned long long seed = 0x128394;
};

/**
 * Approximate k-nearest neighbors of every row of queries [nq, d] among the
 * rows of base [n, d] by a greedy search of graph [n, degree] (the neighbor ids
 * of every base row; -1 entries are skipped), as CAGRA. Single-CTA: one
 * threadgroup per query keeps a sorted list of itopk_size nodes, starts from
 * random nodes and repeatedly expands the best unexpanded ones until all are
 * expanded or max_iterations is reached. Multi-CTA: several threadgroups per
 * query, each a list of 32 expanding one node per iteration, sharing the set
 * of expanded nodes; the result is the best k of their lists. Multi-CTA needs
 * degree <= 224. Works on any fixed-degree graph:
 * nn_descent's or all_neighbors's k-NN graph, or a pruned one. Returns ids and
 * distances as brute_force_knn. base may be float16 (read as is: half the
 * memory traffic) or any real dtype (converted to float32 on every call). For
 * Cosine, the base row norms are computed on every call; normalized rows with
 * InnerProduct avoid that. Runs on the GPU.
 */
MLXVS_API Neighbors graph_search(
    const mx::array& base,
    const mx::array& graph,
    const mx::array& queries,
    int k,
    Metric metric,
    const SearchOptions& options = {},
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
