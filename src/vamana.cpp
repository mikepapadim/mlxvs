// Copyright © 2026 mikepapadim

#include "graph.h"
#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "vamana_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& select_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_vamana_select",
      {"ids", "spc", "scc", "alpha", "params"},
      {"out"},
      kernels::VAMANA_SELECT_SOURCE);
  return kernel;
}

// Gathered rows and pairwise products of one chunk stay near this many bytes.
constexpr long long VAMANA_CHUNK_BYTES = 256LL << 20;

// JVector's similarity from dot products (and squared norms for L2): 1 / (1 +
// squared distance) for L2, (1 + dot) / 2 otherwise.
mx::array similarity(
    const mx::array& dots,
    const mx::array& norm_a,
    const mx::array& norm_b,
    bool l2) {
  if (!l2) {
    return (dots + mx::array(1.0f)) * mx::array(0.5f);
  }
  auto dist =
      mx::maximum(norm_a + norm_b - mx::array(2.0f) * dots, mx::array(0.0f));
  return mx::reciprocal(dist + mx::array(1.0f));
}

} // namespace

mx::array vamana_prune(
    const mx::array& rows_in,
    const mx::array& candidates_in,
    int graph_degree,
    Metric metric,
    float alpha,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::vamana_prune";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  if (candidates_in.ndim() != 2 ||
      !mx::issubdtype(candidates_in.dtype(), mx::integer) ||
      candidates_in.shape(0) != rows.shape(0)) {
    throw std::invalid_argument(
        "[mlxvs::vamana_prune] candidates must be integer ids [n, C], one row "
        "per row of rows.");
  }
  int n = rows.shape(0);
  int d = rows.shape(1);
  int c = candidates_in.shape(1);
  if (c < 1 || c > kernels::VAMANA_CANDIDATES_MAX) {
    throw std::invalid_argument(
        "[mlxvs::vamana_prune] candidates per row must be in [1, " +
        std::to_string(kernels::VAMANA_CANDIDATES_MAX) + "].");
  }
  if (graph_degree < 1 || graph_degree > c) {
    throw std::invalid_argument(
        "[mlxvs::vamana_prune] graph_degree must be in [1, candidates per "
        "row].");
  }
  if (!(alpha >= 1.0f)) {
    throw std::invalid_argument("[mlxvs::vamana_prune] alpha must be >= 1.");
  }
  if (n == 0) {
    return mx::zeros({0, graph_degree}, mx::int32);
  }
  auto cand = candidates_in.dtype() == mx::int32
      ? candidates_in
      : mx::astype(candidates_in, mx::int32);
  if (metric == Metric::Cosine) {
    auto norm = mx::sqrt(mx::sum(mx::square(rows), 1, /*keepdims=*/true));
    rows = rows / mx::maximum(norm, mx::array(1e-30f));
  }
  bool l2 = metric == Metric::L2;
  auto norms = mx::sum(mx::square(rows), 1);
  auto alpha_arr = mx::array({alpha});

  long long per_node = 4LL * c * std::max(d, c);
  int chunk = (int)std::clamp<long long>(VAMANA_CHUNK_BYTES / per_node, 1, n);
  std::vector<mx::array> parts;
  for (int c0 = 0; c0 < n; c0 += chunk) {
    int c1 = std::min(n, c0 + chunk);
    int b = c1 - c0;
    auto ids = mx::slice(cand, {c0, 0}, {c1, c});
    auto safe = mx::where(
        mx::logical_and(ids >= mx::array(0), ids < mx::array(n)),
        ids,
        mx::array(0));
    auto x = mx::reshape(mx::take(rows, mx::flatten(safe), 0), {b, c, d});
    auto p = mx::slice(rows, {c0, 0}, {c1, d});
    auto xn = mx::take(norms, safe);
    auto pn = mx::slice(norms, {c0}, {c1});
    auto dots_pc = mx::squeeze(mx::matmul(x, mx::expand_dims(p, 2)), 2);
    auto dots_cc = mx::matmul(x, mx::transpose(x, {0, 2, 1}));
    auto spc = similarity(dots_pc, mx::expand_dims(pn, 1), xn, l2);
    auto scc =
        similarity(dots_cc, mx::expand_dims(xn, 2), mx::expand_dims(xn, 1), l2);
    auto out = select_kernel()(
        {ids,
         spc,
         scc,
         alpha_arr,
         mx::array({b, n, c0, graph_degree}, {4}, mx::int32)},
        {{b, graph_degree}},
        {mx::int32},
        {b, 1, 1},
        {std::min(b, 64), 1, 1},
        {{"C", c}},
        std::nullopt,
        false,
        {})[0];
    if (n > chunk) {
      mx::eval(out);
    }
    parts.push_back(out);
  }
  return parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
}

mx::array vamana_build(
    const mx::array& rows,
    Metric metric,
    const VamanaOptions& options,
    mx::StreamOrDevice s) {
  if (options.graph_degree < 1 ||
      options.graph_degree > options.intermediate_graph_degree) {
    throw std::invalid_argument(
        "[mlxvs::vamana_build] graph_degree must be in [1, "
        "intermediate_graph_degree].");
  }
  auto knn = all_neighbors(
      rows, options.intermediate_graph_degree, metric, options.knn, s);
  // forward edges pruned, then each node's list with its incoming edges (of
  // lowest rank) pruned again, as Vamana's backward pass
  auto forward = vamana_prune(
      rows, knn.ids, options.graph_degree, metric, options.alpha, s);
  auto incoming = detail::reverse_lists(forward, options.graph_degree);
  auto both = mx::concatenate({forward, incoming}, 1);
  return vamana_prune(
      rows, both, options.graph_degree, metric, options.alpha, s);
}

} // namespace mlxvs
