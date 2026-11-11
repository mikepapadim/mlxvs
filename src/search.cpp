// Copyright © 2026 mikepapadim

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& search_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_graph_search",
      {"base", "graph", "queries", "inv_norm", "params"},
      {"out_ids", "out_scores"},
      kernels::SEARCH_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

int pow2_at_least(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

// queries per kernel launch, so that one launch stays well within the GPU's
// command time limits
constexpr int SEARCH_QUERY_BLOCK = 65536;

} // namespace

Neighbors graph_search(
    const mx::array& base_in,
    const mx::array& graph_in,
    const mx::array& queries_in,
    int k,
    Metric metric,
    const SearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::graph_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  // float16 rows are read as they are (half the memory traffic); other dtypes
  // are converted to float32
  auto base = base_in.ndim() == 2 && base_in.dtype() == mx::float16
      ? base_in
      : detail::float_rows(base_in, op, "base");
  auto queries = detail::float_rows(queries_in, op, "queries");
  if (graph_in.ndim() != 2 || graph_in.shape(0) != base.shape(0)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] graph must be [n, degree], one row per base "
        "row.");
  }
  if (!mx::issubdtype(graph_in.dtype(), mx::integer)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] graph must hold integer ids.");
  }
  if (base.shape(1) != queries.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] base and queries must have the same number of "
        "columns.");
  }
  int n = base.shape(0);
  int d = base.shape(1);
  int deg = graph_in.shape(1);
  int nq = queries.shape(0);
  int width = options.search_width;
  if (n < 1 || deg < 1) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] base and graph must not be empty.");
  }
  if (width < 1 || width > kernels::SEARCH_WIDTH_MAX ||
      (long long)width * deg > kernels::SEARCH_CANDIDATES_MAX) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] search_width must be in [1, 32] and "
        "search_width * graph degree at most " +
        std::to_string(kernels::SEARCH_CANDIDATES_MAX) + ".");
  }
  int m = std::max(options.itopk_size, k);
  m = (m + 31) / 32 * 32;
  if (k < 1 || m > kernels::SEARCH_ITOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] k and itopk_size must be in [1, " +
        std::to_string(kernels::SEARCH_ITOPK_MAX) + "].");
  }
  int cp = pow2_at_least(width * deg);
  int max_iter =
      options.max_iterations > 0 ? options.max_iterations : 2 * m / width + 16;
  int n_rand = std::max(1, options.num_random_samplings);
  // threadgroup memory: the list (two halves), the candidates, the query and
  // the visited set. Kept within 16 KB where it can be: Metal's limit is 32 KB,
  // but its shader validation (as run in CI) counts every byte twice. The
  // visited set gets what is left, at least twice the list and one round.
  bool qshared = d <= kernels::SEARCH_QUERY_SHARED_MAX;
  int qcap = qshared ? (d + 31) / 32 * 32 : 1;
  auto fixed = [&]() { return 16 * m + 8 * cp + 4 * qcap + 256; };
  int hmin = 9;
  while ((1 << hmin) < 2 * (m + cp)) {
    ++hmin;
  }
  if (qshared && fixed() + (4 << hmin) > 16384) {
    qshared = false;
    qcap = 1;
  }
  int hbits = hmin;
  while (hbits < 12 && fixed() + (4 << (hbits + 1)) <= 16384) {
    ++hbits;
  }

  auto graph = graph_in.dtype() == mx::int32 ? graph_in
                                             : mx::astype(graph_in, mx::int32);
  auto inv_norm = mx::array({1.0f});
  if (metric == Metric::Cosine) {
    auto qn = mx::sqrt(mx::sum(mx::square(queries), 1, /*keepdims=*/true));
    queries = queries / mx::maximum(qn, mx::array(1e-30f));
    inv_norm = mx::rsqrt(
        mx::maximum(
            mx::sum(mx::square(mx::astype(base, mx::float32)), 1),
            mx::array(1e-30f)));
  }
  int metric_code = metric == Metric::InnerProduct ? 0
      : metric == Metric::L2                       ? 1
                                                   : 2;
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }

  std::vector<mx::array> id_parts, dist_parts;
  for (int q0 = 0; q0 < nq; q0 += SEARCH_QUERY_BLOCK) {
    int q1 = std::min(nq, q0 + SEARCH_QUERY_BLOCK);
    int bq = q1 - q0;
    auto qs = mx::slice(queries, {q0, 0}, {q1, d});
    auto params = mx::array(
        {n,
         d,
         deg,
         k,
         width,
         max_iter,
         metric_code,
         (int)(options.seed & 0x7fffffffULL) ^ q0,
         n_rand},
        {9},
        mx::int32);
    auto out = search_kernel()(
        {base, graph, qs, inv_norm, params},
        {{bq, k}, {bq, k}},
        {mx::int32, mx::float32},
        {bq * kernels::SEARCH_THREADS, 1, 1},
        {kernels::SEARCH_THREADS, 1, 1},
        {{"M", m},
         {"CP", cp},
         {"QCAP", qcap},
         {"QSHARED", qshared},
         {"HBITS", hbits}},
        std::nullopt,
        false,
        {});
    if (nq > SEARCH_QUERY_BLOCK) {
      mx::eval(out[0], out[1]);
    }
    id_parts.push_back(out[0]);
    dist_parts.push_back(out[1]);
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], dist_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(dist_parts, 0)};
}

} // namespace mlxvs
