// Copyright © 2026 mikepapadim

// IVF-Flat: rows grouped into k-means lists, a query scanning the lists of its
// nearest centers.

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "ivf_kernels.h"
#include "kernels.h"
#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& flat_scan_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_ivf_flat_scan",
      {"data", "queries", "items", "pairs", "offsets", "slots", "params"},
      {"scores", "rows"},
      kernels::IVF_FLAT_SCAN_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

// candidate buffers of one block of queries stay near this many bytes
constexpr long long IVF_BUFFER_BYTES = 256LL << 20;

mx::array unit_rows(const mx::array& x) {
  auto norm = mx::sqrt(mx::sum(mx::square(x), 1, /*keepdims=*/true));
  return x / mx::maximum(norm, mx::array(1e-30f));
}

// The list of every row of data, from the lists' offsets.
mx::array labels_from_offsets(const mx::array& offsets, int n) {
  int lists = offsets.shape(0) - 1;
  if (n == 0) {
    return mx::zeros({0}, mx::int32);
  }
  auto starts = mx::slice(offsets, {1}, {lists});
  auto marks = mx::scatter_add(
      mx::zeros({n + 1}, mx::int32),
      starts,
      mx::ones({lists - 1, 1}, mx::int32),
      0);
  return mx::slice(mx::cumsum(marks, 0), {0}, {n});
}

// Rows grouped by label: (data, ids, offsets).
void group(
    const mx::array& rows,
    const mx::array& ids,
    const mx::array& labels,
    int lists,
    IvfFlatIndex& out) {
  auto order = mx::argsort(labels);
  out.data = mx::contiguous(mx::take(rows, order, 0));
  out.indices = mx::take(ids, order);
  auto counts = mx::scatter_add(
      mx::zeros({lists}, mx::int32),
      labels,
      mx::ones({labels.shape(0), 1}, mx::int32),
      0);
  out.offsets =
      mx::concatenate({mx::zeros({1}, mx::int32), mx::cumsum(counts, 0)}, 0);
  mx::eval(out.data, out.indices, out.offsets);
}

} // namespace

IvfFlatIndex ivf_flat_build(
    const mx::array& rows_in,
    Metric metric,
    const IvfFlatOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_flat_build";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  int n = rows.shape(0);
  if (options.n_lists < 1 || options.n_lists > n) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_build] n_lists must be in [1, number of rows].");
  }
  if (!(options.kmeans_trainset_fraction > 0 &&
        options.kmeans_trainset_fraction <= 1)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_build] kmeans_trainset_fraction must be in (0, 1].");
  }
  if (metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  KMeansOptions km;
  km.max_iter = options.kmeans_n_iters;
  km.seed = options.seed;
  km.sample_rows = std::max(
      64 * options.n_lists,
      (int)std::min<double>(n, n * options.kmeans_trainset_fraction));
  IvfFlatIndex index{
      kmeans_fit(rows, options.n_lists, km),
      mx::array(0),
      mx::array(0),
      mx::array(0),
      metric};
  auto labels = mx::reshape(kmeans_predict(rows, index.centers, 1), {n});
  group(rows, mx::arange(n, mx::int32), labels, options.n_lists, index);
  mx::eval(index.centers);
  return index;
}

IvfFlatIndex ivf_flat_extend(
    const IvfFlatIndex& index,
    const mx::array& new_rows_in,
    const std::optional<mx::array>& new_ids,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_flat_extend";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(new_rows_in, op, "new_rows");
  int n = index.data.shape(0);
  int m = rows.shape(0);
  if (rows.shape(1) != index.data.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_extend] new_rows must have the index's columns.");
  }
  if (new_ids && (new_ids->ndim() != 1 || new_ids->shape(0) != m)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_extend] new_ids must hold one id per new row.");
  }
  if (index.metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  int lists = index.centers.shape(0);
  auto ids = new_ids
      ? mx::astype(*new_ids, mx::int32)
      : mx::add(
            mx::arange(m, mx::int32),
            mx::astype(
                mx::max(mx::concatenate({index.indices, mx::array({-1})}, 0)),
                mx::int32) +
                mx::array(1));
  auto labels = mx::reshape(kmeans_predict(rows, index.centers, 1), {m});
  IvfFlatIndex out{
      index.centers, mx::array(0), mx::array(0), mx::array(0), index.metric};
  group(
      mx::concatenate({index.data, rows}, 0),
      mx::concatenate({index.indices, ids}, 0),
      mx::concatenate({labels_from_offsets(index.offsets, n), labels}, 0),
      lists,
      out);
  return out;
}

Neighbors ivf_flat_search(
    const IvfFlatIndex& index,
    const mx::array& queries_in,
    int k,
    const IvfSearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_flat_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto queries = detail::float_rows(queries_in, op, "queries");
  int nq = queries.shape(0);
  int d = index.data.shape(1);
  int lists = index.centers.shape(0);
  if (queries.shape(1) != d) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_search] queries must have the index's columns.");
  }
  if (k < 1 || k > kernels::TOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::ivf_flat_search] k must be in [1, 256].");
  }
  int probes_n = std::clamp(options.n_probes, 1, lists);
  bool l2 = index.metric == Metric::L2;
  if (index.metric == Metric::Cosine) {
    queries = unit_rows(queries);
  }
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }

  // the lists to scan: the n_probes nearest centers (by L2 for L2 and cosine,
  // by inner product for inner product, as the lists were clustered)
  auto probes = brute_force_knn(
                    index.centers,
                    queries,
                    probes_n,
                    index.metric == Metric::InnerProduct ? Metric::InnerProduct
                                                         : Metric::L2)
                    .ids;
  auto sizes = mx::subtract(
      mx::slice(index.offsets, {1}, {lists + 1}),
      mx::slice(index.offsets, {0}, {lists}));
  auto psizes = mx::take(sizes, mx::maximum(probes, mx::array(0)));
  psizes = mx::where(probes >= mx::array(0), psizes, mx::array(0));
  auto slots = mx::cumsum(psizes, 1, /*reverse=*/false, /*inclusive=*/false);
  auto totals = mx::sum(psizes, 1);
  int width = std::max(1, mx::max(totals).item<int>());
  int kk = std::min(k, width);

  // queries per threadgroup: their tile stays within 8 KB of threadgroup
  // memory
  int dcap = (d + 3) / 4 * 4;
  int qt = std::clamp(2048 / dcap, 1, 32);
  int block = (int)std::clamp<long long>(
      IVF_BUFFER_BYTES / (8LL * width), 1, std::max(1, nq));
  std::vector<mx::array> id_parts, dist_parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int bq = q1 - q0;
    auto bprobes = mx::slice(probes, {q0, 0}, {q1, probes_n});
    auto bslots = mx::slice(slots, {q0, 0}, {q1, probes_n});
    // the (query, probe) pairs grouped by list, cut into items of at most qt
    auto flat = mx::flatten(bprobes);
    auto pairs = mx::astype(mx::argsort(flat), mx::int32);
    auto counts = mx::scatter_add(
        mx::zeros({lists}, mx::int32),
        flat,
        mx::ones({flat.shape(0), 1}, mx::int32),
        0);
    auto pair_start = mx::cumsum(counts, 0, false, /*inclusive=*/false);
    auto chunks = mx::floor_divide(counts + mx::array(qt - 1), mx::array(qt));
    auto item_start = mx::cumsum(chunks, 0, false, /*inclusive=*/false);
    int n_items = mx::sum(chunks).item<int>();
    auto item_list = labels_from_offsets(
        mx::concatenate({item_start, mx::array({n_items})}, 0), n_items);
    auto rank = mx::subtract(
        mx::arange(n_items, mx::int32), mx::take(item_start, item_list));
    auto first = mx::add(mx::take(pair_start, item_list), rank * mx::array(qt));
    auto count = mx::minimum(
        mx::array(qt),
        mx::subtract(mx::take(counts, item_list), rank * mx::array(qt)));
    auto items = mx::contiguous(
        mx::astype(mx::stack({item_list, first, count}, 1), mx::int32));
    auto params = mx::array({d, probes_n, width, l2 ? 1 : 0}, {4}, mx::int32);
    auto out = flat_scan_kernel()(
        {index.data,
         mx::slice(queries, {q0, 0}, {q1, d}),
         items,
         pairs,
         index.offsets,
         mx::flatten(bslots),
         params},
        {{bq, width}, {bq, width}},
        {mx::float32, mx::int32},
        {n_items * kernels::IVF_THREADS, 1, 1},
        {kernels::IVF_THREADS, 1, 1},
        {{"QT", qt}, {"DCAP", dcap}},
        -INFINITY,
        false,
        {});
    // rows: init -inf then cast; mark empty slots with -1 where score is -inf
    auto rows = mx::where(
        mx::isinf(out[0]), mx::array(-1), mx::astype(out[1], mx::int32));
    auto dummy = mx::zeros({bq, kk}, mx::int32);
    auto [cols, best] = topk_merge(
        out[0], dummy, mx::zeros({bq, kk}, mx::float32), 0, true, kk);
    auto pos = mx::take_along_axis(rows, mx::maximum(cols, mx::array(0)), 1);
    auto missing = mx::logical_or(mx::isinf(best), mx::less(pos, mx::array(0)));
    auto ids = mx::where(
        missing,
        mx::array(-1),
        mx::take(index.indices, mx::maximum(pos, mx::array(0))));
    auto dist = l2 ? mx::where(missing, mx::array(INFINITY), mx::negative(best))
                   : mx::where(missing, mx::array(-INFINITY), best);
    if (kk < k) {
      ids = mx::pad(ids, {{0, 0}, {0, k - kk}}, mx::array(-1));
      dist = mx::pad(
          dist, {{0, 0}, {0, k - kk}}, mx::array(l2 ? INFINITY : -INFINITY));
    }
    if (nq > block) {
      mx::eval(ids, dist);
    }
    id_parts.push_back(ids);
    dist_parts.push_back(dist);
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], dist_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(dist_parts, 0)};
}

} // namespace mlxvs
