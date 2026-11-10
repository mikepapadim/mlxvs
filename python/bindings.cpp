// Copyright © 2026 mikepapadim

#include <nanobind/nanobind.h>
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
            base (array): Base rows, float32 ``[n, d]``.
            queries (array): Query rows, float32 ``[nq, d]``.
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
            x (array): Rows, float32 ``[n, d]``.
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
            x (array): Rows, float32 ``[n, d]``.
            centroids (array): Centroids, float32 ``[c, d]``.
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
        (``k`` is its ``graph_degree``).

        Args:
            rows (array): Rows, float32 ``[n, d]``, ``n >= 2``.
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
}
