// Copyright © 2026 mikepapadim

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <stdexcept>
#include <string>

#include <mlx/fast.h>

#include "cagra_kernels.h"
#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& prune_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_prune",
      {"knn", "params"},
      {"pruned"},
      kernels::PRUNE_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_count_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_count",
      {"pruned", "params"},
      {"hist"},
      kernels::REV_COUNT_SOURCE,
      "",
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/true);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_offset_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_offset",
      {"hist", "params"},
      {"offset"},
      kernels::REV_OFFSET_SOURCE);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_fill_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_fill",
      {"pruned", "offset", "params"},
      {"placed", "rev"},
      kernels::REV_FILL_SOURCE,
      "",
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/true);
  return kernel;
}

const mx::fast::CustomKernelFunction& merge_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_merge",
      {"pruned", "rev", "offset", "params"},
      {"graph"},
      kernels::MERGE_SOURCE);
  return kernel;
}

int pow2_at_least(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

} // namespace

mx::array cagra_optimize(
    const mx::array& knn_in,
    int graph_degree,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::cagra_optimize";
  mx::StreamContext context(detail::gpu_stream(s, op));
  if (knn_in.ndim() != 2 || !mx::issubdtype(knn_in.dtype(), mx::integer)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] knn_graph must be integer ids [n, degree].");
  }
  int n = knn_in.shape(0);
  int k = knn_in.shape(1);
  if (k < 1 || k > kernels::PRUNE_DEGREE_MAX) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] knn_graph degree must be in [1, " +
        std::to_string(kernels::PRUNE_DEGREE_MAX) + "].");
  }
  if (graph_degree < 1 || graph_degree > k) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] graph_degree must be in [1, knn_graph "
        "degree].");
  }
  if ((long long)n * graph_degree >= (1LL << 31)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] n * graph_degree must be below 2^31.");
  }
  if (n == 0) {
    return mx::zeros({0, graph_degree}, mx::int32);
  }
  auto knn =
      knn_in.dtype() == mx::int32 ? knn_in : mx::astype(knn_in, mx::int32);
  int d = graph_degree;
  int rcap = d;

  auto pruned = prune_kernel()(
      {knn, mx::array({n, d}, {2}, mx::int32)},
      {{n, d}},
      {mx::int32},
      {n * kernels::PRUNE_THREADS, 1, 1},
      {kernels::PRUNE_THREADS, 1, 1},
      {{"K", k}, {"HSIZE", pow2_at_least(2 * k)}},
      -1,
      false,
      {})[0];

  auto params = mx::array({n}, {1}, mx::int32);
  // reverse edges of lowest rank first, as cuVS adds them one rank at a time
  auto hist = rev_count_kernel()(
      {pruned, params},
      {{n, d}},
      {mx::int32},
      {n * d, 1, 1},
      {256, 1, 1},
      {{"D", d}},
      0,
      false,
      {})[0];
  auto offset = rev_offset_kernel()(
      {hist, params},
      {{n, d + 1}},
      {mx::int32},
      {n, 1, 1},
      {256, 1, 1},
      {{"D", d}},
      std::nullopt,
      false,
      {})[0];
  auto rev = rev_fill_kernel()(
      {pruned, offset, params},
      {{n, d}, {n, 2 * rcap}},
      {mx::int32, mx::int32},
      {n * d, 1, 1},
      {256, 1, 1},
      {{"D", d}, {"RCAP", rcap}},
      0,
      false,
      {})[1];

  return merge_kernel()(
      {pruned, rev, offset, params},
      {{n, d}},
      {mx::int32},
      {n, 1, 1},
      {256, 1, 1},
      {{"D", d}, {"RCAP", rcap}},
      -1,
      false,
      {})[0];
}

namespace {

void check_cagra_options(const CagraOptions& o) {
  if (o.graph_degree < 1 || o.graph_degree > o.intermediate_graph_degree) {
    throw std::invalid_argument(
        "[mlxvs::cagra_build] graph_degree must be in [1, "
        "intermediate_graph_degree].");
  }
}

} // namespace

mx::array cagra_build(
    const mx::array& rows,
    Metric metric,
    const CagraOptions& options,
    mx::StreamOrDevice s) {
  check_cagra_options(options);
  auto knn = all_neighbors(
      rows, options.intermediate_graph_degree, metric, options.knn, s);
  return cagra_optimize(knn.ids, options.graph_degree, s);
}

mx::array cagra_build(
    const HostRows& rows,
    Metric metric,
    const CagraOptions& options,
    mx::StreamOrDevice s) {
  check_cagra_options(options);
  auto knn = all_neighbors(
      rows, options.intermediate_graph_degree, metric, options.knn, s);
  return cagra_optimize(knn.ids, options.graph_degree, s);
}

} // namespace mlxvs
