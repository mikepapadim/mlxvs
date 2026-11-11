// Copyright © 2026 mikepapadim

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlxvs/mlxvs.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

NB_MODULE(_core, m) {
  m.doc() = "GPU k-NN search and k-NN graph construction on Apple silicon";

  nb::enum_<mlxvs::Metric>(m, "Metric", "How two rows compare.")
      .value("inner_product", mlxvs::Metric::InnerProduct)
      .value("l2", mlxvs::Metric::L2)
      .value("cosine", mlxvs::Metric::Cosine);

  nb::enum_<mlxvs::GraphBuild>(
      m, "GraphBuild", "How all_neighbors builds each cluster's graph.")
      .value("brute_force", mlxvs::GraphBuild::BruteForce)
      .value("nn_descent", mlxvs::GraphBuild::NnDescent);

  m.def(
      "brute_force_knn",
      [](const mx::array& base,
         const mx::array& queries,
         int k,
         mlxvs::Metric metric,
         int tile_rows,
         mx::StreamOrDevice s) {
        auto r =
            mlxvs::brute_force_knn(base, queries, k, metric, {tile_rows}, s);
        return std::make_pair(r.ids, r.distances);
      },
      "base"_a,
      "queries"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "tile_rows"_a = 65536,
      "stream"_a = nb::none(),
      R"(
        Exact k-nearest neighbors of every query among the base rows.

        Each tile of ``tile_rows`` base rows is scored with a matrix multiply
        and a fused radix-select kernel keeps the best ``k`` per query.

        Args:
            base (array): Base rows ``[n, d]``, any real dtype.
            queries (array): Query rows ``[nq, d]``, any real dtype.
            k (int): Neighbors per query, at most 256.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            tile_rows (int): Base rows scored per step.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` (-1 where fewer exist)
            and distances ``float32 [nq, k]``, nearest first. For ``l2`` the
            distances are squared and ascending; otherwise similarities,
            descending.
      )");

  m.def(
      "kmeans_fit",
      [](const mx::array& x,
         int clusters,
         int max_iter,
         int sample_rows,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        return mlxvs::kmeans_fit(x, clusters, {max_iter, sample_rows, seed}, s);
      },
      "x"_a,
      "clusters"_a,
      nb::kw_only(),
      "max_iter"_a = 10,
      "sample_rows"_a = 100000,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        Fit k-means (squared Euclidean) to a random sample of the rows.

        k-means++ start, then ``max_iter`` Lloyd iterations.

        Args:
            x (array): Rows ``[n, d]``, any real dtype.
            clusters (int): Number of centroids.
            max_iter (int): Lloyd iterations.
            sample_rows (int): Rows sampled (at least 64 per cluster).
            seed (int): Random seed.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: Centroids, float32 ``[clusters, d]``.
      )");

  m.def(
      "kmeans_predict",
      [](const mx::array& x,
         const mx::array& centroids,
         int m,
         int tile_rows,
         mx::StreamOrDevice s) {
        return mlxvs::kmeans_predict(x, centroids, m, tile_rows, s);
      },
      "x"_a,
      "centroids"_a,
      "m"_a = 1,
      nb::kw_only(),
      "tile_rows"_a = 65536,
      "stream"_a = nb::none(),
      R"(
        The ``m`` nearest centroids of every row, nearest first.

        Args:
            x (array): Rows ``[n, d]``, any real dtype.
            centroids (array): Centroids ``[c, d]``, any real dtype.
            m (int): Centroids per row; more than 1 assigns rows to
              overlapping clusters, as batched all-neighbors does.
            tile_rows (int): Rows scored per step.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: Centroid indices, int32 ``[n, m]``.
      )");

  m.def(
      "nn_descent",
      [](const mx::array& rows,
         int k,
         mlxvs::Metric metric,
         int max_iterations,
         double termination_threshold,
         bool fp16,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::NnDescentOptions options;
        options.max_iterations = max_iterations;
        options.termination_threshold = termination_threshold;
        options.fp16 = fp16;
        options.seed = seed;
        auto r = mlxvs::nn_descent(rows, k, metric, options, s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        Approximate k-nearest-neighbor graph of the rows by NN-Descent.

        The ``k`` nearest other rows of every row, nearest first. Parameters
        follow cuVS's ``nn_descent.index_params`` where they correspond
        (``k`` is its ``graph_degree``). Up to 4096 rows the graph is
        computed exactly instead (as fast, and complete).

        Args:
            rows (array): Rows ``[n, d]``, any real dtype, ``n >= 2``.
            k (int): Neighbors per row, at most 128.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            max_iterations (int): Rounds after the random start, at most.
            termination_threshold (float): Stop after a round that replaced
              fewer than this fraction of the list entries.
            fp16 (bool): Score the joins from float16 rows with SIMD-group
              matrices (float32 accumulation); distances are float32 either
              way.
            seed (int): Random seed. Results still vary slightly between
              runs: concurrent writes to hashed slots race.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [n, k]`` (-1 where fewer were
            found) and distances ``float32 [n, k]``, as ``brute_force_knn``.
      )");

  m.def(
      "graph_search",
      [](const mx::array& base,
         const mx::array& graph,
         const mx::array& queries,
         int k,
         mlxvs::Metric metric,
         int itopk_size,
         int search_width,
         int max_iterations,
         int num_random_samplings,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::SearchOptions options;
        options.itopk_size = itopk_size;
        options.search_width = search_width;
        options.max_iterations = max_iterations;
        options.num_random_samplings = num_random_samplings;
        options.seed = seed;
        auto r =
            mlxvs::graph_search(base, graph, queries, k, metric, options, s);
        return std::make_pair(r.ids, r.distances);
      },
      "base"_a,
      "graph"_a,
      "queries"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "itopk_size"_a = 64,
      "search_width"_a = 1,
      "max_iterations"_a = 0,
      "num_random_samplings"_a = 1,
      "seed"_a = 0x128394,
      "stream"_a = nb::none(),
      R"(
        Approximate k-nearest neighbors of every query by searching a graph.

        CAGRA's single-CTA search: one threadgroup per query keeps a sorted
        list of ``itopk_size`` nodes, starts from random nodes and expands the
        best unexpanded ones (``search_width`` per round) until every node of
        the list has been expanded. Parameters follow cuVS's
        ``cagra.SearchParams``. Any fixed-degree graph works: the k-NN graph of
        ``nn_descent`` or ``all_neighbors``, or a pruned one.

        Args:
            base (array): Base rows ``[n, d]``, any real dtype.
            graph (array): Neighbor ids of every base row ``[n, degree]``,
              integers; -1 entries are skipped.
            queries (array): Query rows ``[nq, d]``, any real dtype.
            k (int): Neighbors per query, at most ``itopk_size``.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            itopk_size (int): Length of the internal list (rounded up to a
              multiple of 32, at most 256). Longer: higher recall, slower.
            search_width (int): Nodes expanded per round;
              ``search_width * degree <= 512``.
            max_iterations (int): The most rounds; 0: automatic.
            num_random_samplings (int): Rounds of random start nodes.
            seed (int): Seed of the random start nodes.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` (-1 where fewer were
            found) and distances ``float32 [nq, k]``, as ``brute_force_knn``.
      )");

  auto all_neighbors_options = [](mlxvs::GraphBuild algo,
                                  int n_clusters,
                                  int overlap_factor,
                                  int brute_force_below,
                                  long long max_cluster_rows,
                                  bool verbose,
                                  int max_iterations,
                                  double termination_threshold,
                                  bool fp16,
                                  unsigned long long seed) {
    mlxvs::AllNeighborsOptions options;
    options.algo = algo;
    options.n_clusters = n_clusters;
    options.overlap_factor = overlap_factor;
    options.brute_force_below = brute_force_below;
    options.max_cluster_rows = max_cluster_rows;
    options.verbose = verbose;
    options.kmeans.seed = seed;
    options.nn_descent.max_iterations = max_iterations;
    options.nn_descent.termination_threshold = termination_threshold;
    options.nn_descent.fp16 = fp16;
    options.nn_descent.seed = seed;
    return options;
  };

  // rows left on the host (a NumPy array, a memory-mapped file): read through
  // their buffer, a cluster at a time
  m.def(
      "all_neighbors_host",
      [all_neighbors_options](
          nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
              rows,
          int k,
          mlxvs::Metric metric,
          mlxvs::GraphBuild algo,
          int n_clusters,
          int overlap_factor,
          int brute_force_below,
          long long max_cluster_rows,
          bool verbose,
          int max_iterations,
          double termination_threshold,
          bool fp16,
          unsigned long long seed,
          mx::StreamOrDevice s) {
        mlxvs::HostRows host{
            rows.data(), (long long)rows.shape(0), (int)rows.shape(1)};
        auto r = mlxvs::all_neighbors(
            host,
            k,
            metric,
            all_neighbors_options(
                algo,
                n_clusters,
                overlap_factor,
                brute_force_below,
                max_cluster_rows,
                verbose,
                max_iterations,
                termination_threshold,
                fp16,
                seed),
            s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "algo"_a = mlxvs::GraphBuild::NnDescent,
      "n_clusters"_a = 0,
      "overlap_factor"_a = 2,
      "brute_force_below"_a = 4096,
      "max_cluster_rows"_a = 0,
      "verbose"_a = false,
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      "all_neighbors of float32 rows read from host memory.");

  m.def(
      "all_neighbors",
      [all_neighbors_options](
          const mx::array& rows,
          int k,
          mlxvs::Metric metric,
          mlxvs::GraphBuild algo,
          int n_clusters,
          int overlap_factor,
          int brute_force_below,
          long long max_cluster_rows,
          bool verbose,
          int max_iterations,
          double termination_threshold,
          bool fp16,
          unsigned long long seed,
          mx::StreamOrDevice s) {
        auto r = mlxvs::all_neighbors(
            rows,
            k,
            metric,
            all_neighbors_options(
                algo,
                n_clusters,
                overlap_factor,
                brute_force_below,
                max_cluster_rows,
                verbose,
                max_iterations,
                termination_threshold,
                fp16,
                seed),
            s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "algo"_a = mlxvs::GraphBuild::NnDescent,
      "n_clusters"_a = 0,
      "overlap_factor"_a = 2,
      "brute_force_below"_a = 4096,
      "max_cluster_rows"_a = 0,
      "verbose"_a = false,
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        k-nearest-neighbor graph of the rows, in overlapping k-means clusters.

        As cuVS's ``all_neighbors.build``: with ``n_clusters > 1`` the rows
        are clustered by k-means (unit rows for ``cosine``), every row joins
        its ``overlap_factor`` nearest clusters, each cluster's graph is built
        on its own, and every row keeps the best ``k`` of the lists its
        clusters gave it. One cluster is on the GPU at a time.

        A float32 NumPy array, including a ``numpy.memmap`` of a file larger
        than memory, is read where it is: only one cluster's rows at a time
        are copied to the GPU.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype, ``n >= 2``.
            k (int): Neighbors per row, at most 128 with ``nn_descent`` (255
              with ``brute_force``).
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            algo (GraphBuild): ``nn_descent`` (approximate) or
              ``brute_force`` (exact) for each cluster.
            n_clusters (int): k-means clusters. 0 (default): as many as the
              GPU's memory asks for, one when the rows fit. 1: all rows at
              once.
            overlap_factor (int): Clusters each row joins.
            brute_force_below (int): Clusters of at most this many rows are
              built exactly whatever ``algo``.
            max_cluster_rows (int): The most rows of one cluster; a larger
              k-means cluster is clustered again. 0: from the GPU's memory.
            verbose (bool): Print the clusters and the progress to stderr.
            max_iterations (int): NN-Descent rounds after the random start.
            termination_threshold (float): NN-Descent stops after a round
              that replaced fewer than this fraction of the list entries.
            fp16 (bool): NN-Descent joins from float16 rows.
            seed (int): Random seed of k-means and NN-Descent.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [n, k]`` (-1 where fewer were
            found) and distances ``float32 [n, k]``, as ``brute_force_knn``.
      )");
}
