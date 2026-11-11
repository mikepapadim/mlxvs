// Copyright © 2026 mikepapadim

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& topk_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_topk_merge",
      {"scores", "prev_ids", "prev_scores", "params"},
      {"out_ids", "out_scores"},
      kernels::TOPK_SOURCE,
      kernels::TOPK_HEADER);
  return kernel;
}

mx::array unit_rows(const mx::array& x) {
  auto norm = mx::sqrt(mx::sum(mx::square(x), /*axis=*/1, /*keepdims=*/true));
  return x / mx::maximum(norm, mx::array(1e-30f));
}

} // namespace

std::pair<mx::array, mx::array> topk_merge(
    const mx::array& scores,
    const mx::array& ids,
    const mx::array& best,
    int id_offset,
    bool first,
    int k,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::topk_merge"));
  if (scores.ndim() != 2 || scores.dtype() != mx::float32) {
    throw std::invalid_argument(
        "[mlxvs::topk_merge] scores must be float32 [nq, t]");
  }
  if (k < 1 || k > kernels::TOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::topk_merge] k must be in [1, " +
        std::to_string(kernels::TOPK_MAX) + "]");
  }
  int nq = scores.shape(0);
  int t = scores.shape(1);
  auto params = mx::array({t, k, id_offset, first ? 1 : 0}, {4}, mx::int32);
  auto out = topk_kernel()(
      {scores, ids, best, params},
      {{nq, k}, {nq, k}},
      {mx::int32, mx::float32},
      {nq * kernels::TOPK_THREADS, 1, 1},
      {kernels::TOPK_THREADS, 1, 1},
      {},
      std::nullopt,
      false,
      {});
  return {out[0], out[1]};
}

Neighbors brute_force_knn(
    const mx::array& base_in,
    const mx::array& queries_in,
    int k,
    Metric metric,
    const KnnOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::brute_force_knn"));
  auto base = detail::float_rows(base_in, "mlxvs::brute_force_knn", "base");
  auto queries =
      detail::float_rows(queries_in, "mlxvs::brute_force_knn", "queries");
  if (base.shape(1) != queries.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::brute_force_knn] base and queries must have the same number "
        "of columns.");
  }
  int n = base.shape(0);
  int d = base.shape(1);
  int nq = queries.shape(0);
  int tile = std::max(1, options.tile_rows);
  // queries in blocks too, so that one step's [block, tile] score matrix stays
  // at about 1 GB whatever nq is (a k-NN graph has nq = n)
  int block = (int)std::max(
      1LL, std::min((long long)std::max(nq, 1), (1LL << 28) / tile));
  std::vector<mx::array> id_parts, best_parts;
  for (int q0 = 0; q0 < nq || (nq == 0 && q0 == 0); q0 += block) {
    int q1 = std::min(nq, q0 + block);
    mx::array qs = mx::slice(queries, {q0, 0}, {q1, d});
    mx::array q = metric == Metric::Cosine ? unit_rows(qs) : qs;
    mx::eval(q);
    mx::array ids = mx::full({q1 - q0, k}, -1, mx::int32);
    mx::array best = mx::full(
        {q1 - q0, k}, -std::numeric_limits<float>::infinity(), mx::float32);
    for (int t0 = 0; t0 < n; t0 += tile) {
      int t1 = std::min(n, t0 + tile);
      mx::array rows = mx::slice(base, {t0, 0}, {t1, d});
      if (metric == Metric::Cosine) {
        rows = unit_rows(rows);
      }
      mx::array scores = mx::matmul(q, mx::transpose(rows));
      if (metric == Metric::L2) {
        // -|q - b|^2 + |q|^2 = 2 q.b - |b|^2: the same order per query, |q|^2
        // added back at the end
        scores =
            2.0f * scores - mx::expand_dims(mx::sum(mx::square(rows), 1), 0);
      }
      auto merged = topk_merge(scores, ids, best, t0, t0 == 0, k);
      ids = merged.first;
      best = merged.second;
      mx::eval(ids, best); // one tile's score matrix at a time
    }
    if (metric == Metric::L2) {
      auto qq = mx::sum(mx::square(qs), 1, /*keepdims=*/true);
      // squared distances, ascending; a missing neighbor (id -1) is +inf so
      // that rows stay sorted
      best = mx::where(
          ids >= 0,
          mx::maximum(qq - best, mx::array(0.0f)),
          mx::array(std::numeric_limits<float>::infinity()));
      mx::eval(best);
    }
    id_parts.push_back(ids);
    best_parts.push_back(best);
    if (nq == 0) {
      break;
    }
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], best_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(best_parts, 0)};
}

} // namespace mlxvs
