// Copyright © 2026 mikepapadim

// IVF-Flat: rows grouped into k-means lists, a query scanning the lists of its
// nearest centers.

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "ivf_kernels.h"
#include "kernels.h"
#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& pq_scan_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_ivf_pq_scan",
      {"codes",
       "codebooks",
       "centers",
       "queries",
       "probes",
       "offsets",
       "slots",
       "params"},
      {"scores", "rows"},
      kernels::IVF_PQ_SCAN_SOURCE);
  return kernel;
}

const mx::fast::CustomKernelFunction& flat_scan_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_ivf_flat_scan",
      {"data", "queries", "items", "pairs", "offsets", "slots", "params"},
      {"scores", "rows"},
      kernels::IVF_FLAT_SCAN_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& sq_scan_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_ivf_sq_scan",
      {"codes",
       "queries",
       "centers",
       "vmin",
       "delta",
       "qbase",
       "items",
       "pairs",
       "offsets",
       "slots",
       "params"},
      {"scores", "rows"},
      kernels::IVF_SQ_SCAN_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& rabitq_scan_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_ivf_rabitq_scan",
      {"codes",
       "factors",
       "rq",
       "crot",
       "items",
       "pairs",
       "offsets",
       "slots",
       "params",
       "tau"},
      {"scores", "rows"},
      std::string(kernels::IVF_RABITQ_PROLOGUE) +
          kernels::IVF_RABITQ_SCAN_SOURCE);
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

// Rows (or their codes) grouped by label: payload, ids and list offsets.
void group(
    const mx::array& payload,
    const mx::array& ids,
    const mx::array& labels,
    int lists,
    mx::array& out_payload,
    mx::array& out_ids,
    mx::array& out_offsets) {
  auto order = mx::argsort(labels);
  out_payload = mx::contiguous(mx::take(payload, order, 0));
  out_ids = mx::take(ids, order);
  auto counts = mx::scatter_add(
      mx::zeros({lists}, mx::int32),
      labels,
      mx::ones({labels.shape(0), 1}, mx::int32),
      0);
  out_offsets =
      mx::concatenate({mx::zeros({1}, mx::int32), mx::cumsum(counts, 0)}, 0);
  mx::eval(out_payload, out_ids, out_offsets);
}

// New ids: given, or the largest id plus 1, 2, ...
mx::array next_ids(
    const mx::array& indices,
    int m,
    const std::optional<mx::array>& new_ids) {
  if (new_ids) {
    return mx::astype(*new_ids, mx::int32);
  }
  auto top = mx::max(mx::concatenate({indices, mx::array({-1})}, 0));
  return mx::add(mx::arange(m, mx::int32), mx::astype(top, mx::int32) + 1);
}

// The lists a query scans and where each starts in its candidate buffer.
struct Probed {
  mx::array probes; // [nq, P]
  mx::array slots; // [nq, P]
  int width; // the longest buffer
};

Probed probe(
    const mx::array& centers,
    const mx::array& offsets,
    const mx::array& queries,
    int n_probes,
    Metric metric) {
  int lists = centers.shape(0);
  // the n_probes nearest centers (by L2 for L2 and cosine, by inner product
  // for inner product, as the lists were clustered)
  auto probes =
      brute_force_knn(
          centers,
          queries,
          n_probes,
          metric == Metric::InnerProduct ? Metric::InnerProduct : Metric::L2)
          .ids;
  auto sizes = mx::subtract(
      mx::slice(offsets, {1}, {lists + 1}), mx::slice(offsets, {0}, {lists}));
  auto psizes = mx::take(sizes, mx::maximum(probes, mx::array(0)));
  psizes = mx::where(probes >= mx::array(0), psizes, mx::array(0));
  auto slots = mx::cumsum(psizes, 1, /*reverse=*/false, /*inclusive=*/false);
  int width = std::max(1, mx::max(mx::sum(psizes, 1)).item<int>());
  return {probes, slots, width};
}

// The best k of each query's candidate buffer, as ids and distances.
Neighbors finish(
    const mx::array& scores,
    const mx::array& rows_in,
    const mx::array& indices,
    int k,
    bool l2) {
  int bq = scores.shape(0);
  int kk = std::min(k, (int)scores.shape(1));
  auto rows = mx::where(
      mx::isinf(scores), mx::array(-1), mx::astype(rows_in, mx::int32));
  auto [cols, best] = topk_merge(
      scores,
      mx::zeros({bq, kk}, mx::int32),
      mx::zeros({bq, kk}, mx::float32),
      0,
      true,
      kk);
  auto pos = mx::take_along_axis(rows, mx::maximum(cols, mx::array(0)), 1);
  auto missing = mx::logical_or(mx::isinf(best), mx::less(pos, mx::array(0)));
  auto ids = mx::where(
      missing,
      mx::array(-1),
      mx::take(indices, mx::maximum(pos, mx::array(0))));
  auto dist = l2 ? mx::where(missing, mx::array(INFINITY), mx::negative(best))
                 : mx::where(missing, mx::array(-INFINITY), best);
  if (kk < k) {
    ids = mx::pad(ids, {{0, 0}, {0, k - kk}}, mx::array(-1));
    dist = mx::pad(
        dist, {{0, 0}, {0, k - kk}}, mx::array(l2 ? INFINITY : -INFINITY));
  }
  return {ids, dist};
}

Neighbors concat(std::vector<Neighbors>& parts) {
  if (parts.size() == 1) {
    return parts[0];
  }
  std::vector<mx::array> ids, dist;
  for (auto& p : parts) {
    ids.push_back(p.ids);
    dist.push_back(p.distances);
  }
  return {mx::concatenate(ids, 0), mx::concatenate(dist, 0)};
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
  group(
      rows,
      mx::arange(n, mx::int32),
      labels,
      options.n_lists,
      index.data,
      index.indices,
      index.offsets);
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
      out.data,
      out.indices,
      out.offsets);
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
  auto [probes, slots, width] =
      probe(index.centers, index.offsets, queries, probes_n, index.metric);

  // queries per threadgroup: their tile stays within 8 KB of threadgroup
  // memory
  int dcap = (d + 3) / 4 * 4;
  int qt = std::clamp(2048 / dcap, 1, 32);
  int block = (int)std::clamp<long long>(
      IVF_BUFFER_BYTES / (8LL * width), 1, std::max(1, nq));
  std::vector<Neighbors> parts;
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
    auto r = finish(out[0], out[1], index.indices, k, l2);
    if (nq > block) {
      mx::eval(r.ids, r.distances);
    }
    parts.push_back(r);
  }
  return concat(parts);
}

namespace {

// The (query, probe) pairs of a block of queries grouped by list and cut into
// items of at most qt pairs: items [n_items, 3] (list, first pair, pairs) and
// pairs sorted by list.
struct Items {
  mx::array items;
  mx::array pairs;
  int n_items;
};

Items group_pairs(const mx::array& bprobes, int lists, int qt) {
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
  return {items, pairs, n_items};
}

// cuVS's IVF-SQ codes of rows in the lists of labels:
// clamp(round((x - center - vmin) / delta), 0, 255), in blocks of rows.
mx::array sq_encode(
    const mx::array& rows,
    const mx::array& labels,
    const mx::array& centers,
    const mx::array& vmin,
    const mx::array& delta) {
  int n = rows.shape(0);
  int d = rows.shape(1);
  constexpr int BLOCK = 1 << 18;
  std::vector<mx::array> parts;
  for (int i = 0; i < n || (n == 0 && parts.empty()); i += BLOCK) {
    int j = std::min(n, i + BLOCK);
    auto x = mx::slice(rows, {i, 0}, {j, d});
    auto c = mx::take(centers, mx::slice(labels, {i}, {j}), 0);
    auto v = mx::divide(mx::subtract(mx::subtract(x, c), vmin), delta);
    auto code = mx::clip(
        mx::floor(mx::add(v, mx::array(0.5f))),
        mx::array(0.0f),
        mx::array(255.0f));
    parts.push_back(mx::astype(code, mx::uint8));
    if (n > BLOCK) {
      mx::eval(parts.back());
    }
    if (n == 0) {
      break;
    }
  }
  return parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
}

} // namespace

IvfSqIndex ivf_sq_build(
    const mx::array& rows_in,
    Metric metric,
    const IvfSqOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_sq_build";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  int n = rows.shape(0);
  int d = rows.shape(1);
  if (options.n_lists < 1 || options.n_lists > n) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_build] n_lists must be in [1, number of rows].");
  }
  if (options.max_train_points_per_cluster < 1) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_build] max_train_points_per_cluster must be > 0.");
  }
  if (metric == Metric::Cosine) {
    if (d < 2) {
      throw std::invalid_argument(
          "[mlxvs::ivf_sq_build] Cosine needs more than one dimension.");
    }
    rows = unit_rows(rows);
  }
  long long train_rows = std::min<long long>(
      n, (long long)options.n_lists * options.max_train_points_per_cluster);
  KMeansOptions km;
  km.max_iter = options.kmeans_n_iters;
  km.seed = options.seed;
  km.sample_rows = (int)train_rows;
  auto centers = kmeans_fit(rows, options.n_lists, km);
  // the quantization is trained on the residuals of a sample, as cuVS's
  auto train = rows;
  if (train_rows < n) {
    auto perm = mx::random::permutation(n, mx::random::key(options.seed));
    train = mx::take(rows, mx::slice(perm, {0}, {(int)train_rows}), 0);
  }
  auto tlabels = mx::reshape(kmeans_predict(train, centers, 1), {-1});
  auto resid = mx::subtract(train, mx::take(centers, tlabels, 0));
  auto lo = mx::min(resid, 0);
  auto hi = mx::max(resid, 0);
  auto range = mx::subtract(hi, lo);
  auto margin = mx::multiply(range, mx::array(0.05f));
  auto delta = mx::where(
      range > mx::array(0.0f),
      mx::divide(
          mx::add(range, mx::multiply(margin, mx::array(2.0f))),
          mx::array(255.0f)),
      mx::array(1.0f));
  auto vmin = mx::subtract(lo, margin);
  mx::eval(centers, vmin, delta);
  auto labels = mx::reshape(kmeans_predict(rows, centers, 1), {n});
  auto codes = sq_encode(rows, labels, centers, vmin, delta);
  IvfSqIndex index{
      centers, mx::array(0), mx::array(0), mx::array(0), vmin, delta, metric};
  group(
      codes,
      mx::arange(n, mx::int32),
      labels,
      options.n_lists,
      index.codes,
      index.indices,
      index.offsets);
  return index;
}

IvfSqIndex ivf_sq_extend(
    const IvfSqIndex& index,
    const mx::array& new_rows_in,
    const std::optional<mx::array>& new_ids,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_sq_extend";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(new_rows_in, op, "new_rows");
  int n = index.codes.shape(0);
  int m = rows.shape(0);
  if (rows.shape(1) != index.centers.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_extend] new_rows must have the index's columns.");
  }
  if (new_ids && (new_ids->ndim() != 1 || new_ids->shape(0) != m)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_extend] new_ids must hold one id per new row.");
  }
  if (index.metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  int lists = index.centers.shape(0);
  auto ids = next_ids(index.indices, m, new_ids);
  auto labels = mx::reshape(kmeans_predict(rows, index.centers, 1), {m});
  auto codes = sq_encode(rows, labels, index.centers, index.vmin, index.delta);
  IvfSqIndex out{
      index.centers,
      mx::array(0),
      mx::array(0),
      mx::array(0),
      index.vmin,
      index.delta,
      index.metric};
  group(
      mx::concatenate({index.codes, codes}, 0),
      mx::concatenate({index.indices, ids}, 0),
      mx::concatenate({labels_from_offsets(index.offsets, n), labels}, 0),
      lists,
      out.codes,
      out.indices,
      out.offsets);
  return out;
}

Neighbors ivf_sq_search(
    const IvfSqIndex& index,
    const mx::array& queries_in,
    int k,
    const IvfSearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_sq_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto queries = detail::float_rows(queries_in, op, "queries");
  int nq = queries.shape(0);
  int d = index.centers.shape(1);
  int lists = index.centers.shape(0);
  if (queries.shape(1) != d) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_search] queries must have the index's columns.");
  }
  if (k < 1 || k > kernels::TOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_search] k must be in [1, 256].");
  }
  int dcap = (d + 3) / 4 * 4;
  if (dcap > 2048) {
    throw std::invalid_argument(
        "[mlxvs::ivf_sq_search] at most 2048 dimensions.");
  }
  int probes_n = std::clamp(options.n_probes, 1, lists);
  bool l2 = index.metric == Metric::L2;
  int metric_code = l2 ? 1 : index.metric == Metric::Cosine ? 2 : 0;
  if (index.metric == Metric::Cosine) {
    queries = unit_rows(queries);
  }
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }
  auto [probes, slots, width] =
      probe(index.centers, index.offsets, queries, probes_n, index.metric);
  // the query's tile and the steps stay within 16 KB of threadgroup memory
  int qt = std::clamp(4096 / dcap - 1, 1, 32);
  int block = (int)std::clamp<long long>(
      IVF_BUFFER_BYTES / (8LL * width), 1, std::max(1, nq));
  auto shifted = mx::add(index.centers, index.vmin);
  std::vector<Neighbors> parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int bq = q1 - q0;
    auto bq_rows = mx::slice(queries, {q0, 0}, {q1, d});
    auto bprobes = mx::slice(probes, {q0, 0}, {q1, probes_n});
    auto bslots = mx::slice(slots, {q0, 0}, {q1, probes_n});
    // q . (center + vmin) of every (query, probe) pair
    auto qbase = l2
        ? mx::zeros({bq * probes_n}, mx::float32)
        : mx::flatten(
              mx::sum(
                  mx::multiply(
                      mx::expand_dims(bq_rows, 1),
                      mx::take(shifted, mx::maximum(bprobes, mx::array(0)), 0)),
                  2));
    auto g = group_pairs(bprobes, lists, qt);
    auto params = mx::array({d, probes_n, width, metric_code}, {4}, mx::int32);
    auto out = sq_scan_kernel()(
        {index.codes,
         bq_rows,
         index.centers,
         index.vmin,
         index.delta,
         qbase,
         g.items,
         g.pairs,
         index.offsets,
         mx::flatten(bslots),
         params},
        {{bq, width}, {bq, width}},
        {mx::float32, mx::int32},
        {g.n_items * kernels::IVF_THREADS, 1, 1},
        {kernels::IVF_THREADS, 1, 1},
        {{"QT", qt}, {"DCAP", dcap}},
        -INFINITY,
        false,
        {});
    auto r = finish(out[0], out[1], index.indices, k, l2);
    if (nq > block) {
      mx::eval(r.ids, r.distances);
    }
    parts.push_back(r);
  }
  return concat(parts);
}

namespace {

// Extended RaBitQ's scale for a unit vector o: the t maximizing the cosine of
// o and (m + 0.5) * sign(o), m = min(floor(t |o_j|), 2^ex - 1). The cosine
// changes only where some m_j steps, at t = k / |o_j|.
double best_scale(const std::vector<double>& o, int ex) {
  int top = (1 << ex) - 1;
  std::vector<std::pair<double, int>> events;
  events.reserve(o.size() * top);
  double num = 0, den = 0;
  for (size_t j = 0; j < o.size(); ++j) {
    double a = std::abs(o[j]);
    num += 0.5 * a;
    den += 0.25;
    if (a > 0) {
      for (int k = 1; k <= top; ++k) {
        events.emplace_back(k / a, (int)j);
      }
    }
  }
  std::sort(events.begin(), events.end());
  std::vector<int> m(o.size(), 0);
  double best = num / std::sqrt(den), best_t = 0;
  for (auto& [t, j] : events) {
    double a = std::abs(o[j]);
    double before = m[j] + 0.5;
    m[j] += 1;
    double after = m[j] + 0.5;
    num += a;
    den += after * after - before * before;
    double c = num / std::sqrt(den);
    if (c > best) {
      best = c;
      best_t = t;
    }
  }
  return best_t;
}

// cuVS's constant scale (fast quantize): the mean best scale of random unit
// vectors of the dimension.
double const_scale(int dp, int ex, unsigned long long seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  const int samples = 32;
  double total = 0;
  std::vector<double> o(dp);
  for (int i = 0; i < samples; ++i) {
    double n2 = 0;
    for (auto& v : o) {
      v = g(rng);
      n2 += v * v;
    }
    for (auto& v : o) {
      v /= std::sqrt(n2);
    }
    total += best_scale(o, ex);
  }
  return total / samples;
}

// Codes (bit planes) and factors of rotated residuals [m, dp].
void rabitq_encode(
    const mx::array& resid,
    int bits,
    double scale,
    bool fast,
    mx::array& codes,
    mx::array& factors) {
  int m = resid.shape(0);
  int dp = resid.shape(1);
  int ex = bits - 1;
  auto r2 = mx::sum(mx::square(resid), 1);
  auto positive = mx::greater(resid, mx::array(0.0f));
  // the sign bits' estimator: f1 and its error bound eps (RaBitQ's, with
  // epsilon_0 = 1.9)
  auto abs_sum = mx::sum(mx::abs(resid), 1);
  auto rnorm = mx::sqrt(r2);
  auto f1 = mx::where(
      mx::greater(abs_sum, mx::array(0.0f)),
      mx::divide(r2, mx::multiply(abs_sum, mx::array(0.5f))),
      mx::array(0.0f));
  auto cosv = mx::divide(
      abs_sum,
      mx::maximum(
          mx::multiply(rnorm, mx::array(std::sqrt((float)dp))),
          mx::array(1e-30f)));
  auto c2 = mx::maximum(mx::square(cosv), mx::array(1e-12f));
  auto eps = mx::multiply(
      mx::multiply(rnorm, mx::array(1.9f)),
      mx::sqrt(
          mx::divide(
              mx::maximum(mx::subtract(mx::array(1.0f), c2), mx::array(0.0f)),
              mx::multiply(c2, mx::array((float)(dp - 1))))));
  mx::array u = mx::astype(positive, mx::int32);
  if (ex > 0) {
    auto o = mx::divide(
        resid,
        mx::maximum(mx::sqrt(mx::expand_dims(r2, 1)), mx::array(1e-30f)));
    auto absd = mx::abs(o);
    float top = (float)((1 << ex) - 1);
    auto quant = [&](const mx::array& t) {
      return mx::minimum(mx::floor(mx::multiply(absd, t)), mx::array(top));
    };
    mx::array mag = quant(mx::array((float)scale));
    if (!fast) {
      // each row's best of 33 scales around the constant one
      auto best_cos = mx::full({m}, -1.0f);
      for (int i = 0; i <= 32; ++i) {
        float t = (float)(scale * (0.6 + 0.8 * i / 32.0));
        auto mi = quant(mx::array(t));
        auto c = mx::add(mi, mx::array(0.5f));
        auto cos = mx::divide(
            mx::sum(mx::multiply(c, absd), 1),
            mx::sqrt(mx::sum(mx::square(c), 1)));
        auto better = mx::greater(cos, best_cos);
        mag = mx::where(mx::expand_dims(better, 1), mi, mag);
        best_cos = mx::where(better, cos, best_cos);
      }
    }
    auto magi = mx::astype(mag, mx::int32);
    int base = 1 << ex;
    u = mx::where(
        positive,
        mx::add(magi, mx::array(base)),
        mx::subtract(mx::array(base - 1), magi));
  }
  float cb = 0.5f * (float)((1 << bits) - 1);
  auto centred = mx::subtract(mx::astype(u, mx::float32), mx::array(cb));
  auto denom = mx::sum(mx::multiply(centred, resid), 1);
  auto f = mx::where(
      mx::greater(denom, mx::array(0.0f)),
      mx::divide(r2, mx::maximum(denom, mx::array(1e-30f))),
      mx::array(0.0f));
  factors = mx::stack({r2, f, f1, eps}, 1);
  // bit planes: bit j % 32 of word j / 32 of plane b holds bit b of u_j
  auto uu = mx::astype(u, mx::uint32);
  auto weights = mx::left_shift(mx::array(1u), mx::arange(32, mx::uint32));
  std::vector<mx::array> planes;
  for (int b = 0; b < bits; ++b) {
    auto bit = mx::bitwise_and(
        mx::right_shift(uu, mx::array((uint32_t)b)), mx::array(1u));
    auto words =
        mx::sum(mx::multiply(mx::reshape(bit, {m, dp / 32, 32}), weights), 2);
    planes.push_back(mx::astype(words, mx::uint32));
  }
  codes = mx::contiguous(mx::stack(planes, 1));
}

mx::array pad_to(const mx::array& x, int dp) {
  int d = x.shape(1);
  return d == dp ? x : mx::pad(x, {{0, 0}, {0, dp - d}});
}

// Codes and factors of rows (float [m, d]) in the lists of labels, in blocks.
void rabitq_rows(
    const IvfRabitqIndex& index,
    const mx::array& rows,
    const mx::array& labels,
    double scale,
    bool fast,
    mx::array& codes,
    mx::array& factors) {
  int m = rows.shape(0);
  int dp = index.rotation.shape(0);
  constexpr int BLOCK = 1 << 16;
  std::vector<mx::array> cs, fs;
  for (int i = 0; i < std::max(m, 1); i += BLOCK) {
    int j = std::min(m, i + BLOCK);
    auto x = pad_to(mx::slice(rows, {i, 0}, {j, rows.shape(1)}), dp);
    auto resid = mx::subtract(
        mx::matmul(x, index.rotation),
        mx::take(index.centers_rot, mx::slice(labels, {i}, {j}), 0));
    mx::array c(0), f(0);
    rabitq_encode(resid, index.bits_per_dim, scale, fast, c, f);
    mx::eval(c, f);
    cs.push_back(c);
    fs.push_back(f);
    if (m == 0) {
      break;
    }
  }
  codes = cs.size() == 1 ? cs[0] : mx::concatenate(cs, 0);
  factors = fs.size() == 1 ? fs[0] : mx::concatenate(fs, 0);
}

} // namespace

IvfRabitqIndex ivf_rabitq_build(
    const mx::array& rows_in,
    Metric metric,
    const IvfRabitqOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_rabitq_build";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  int n = rows.shape(0);
  int d = rows.shape(1);
  if (options.n_lists < 1 || options.n_lists > n) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_build] n_lists must be in [1, number of rows].");
  }
  if (options.bits_per_dim < 1 || options.bits_per_dim > 9) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_build] bits_per_dim must be in [1, 9].");
  }
  int dp = (d + 63) / 64 * 64;
  if (dp > 3072) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_build] at most 3072 dimensions.");
  }
  if (metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  KMeansOptions km;
  km.max_iter = options.kmeans_n_iters;
  km.seed = options.seed;
  km.sample_rows = (int)std::min<long long>(
      n, (long long)options.n_lists * options.max_train_points_per_cluster);
  auto centers = kmeans_fit(rows, options.n_lists, km);
  // a random orthogonal rotation: Q of the QR of a Gaussian matrix (on the
  // CPU, as MLX's QR)
  auto gauss = mx::random::normal(
      {dp, dp}, mx::float32, 0.0f, 1.0f, mx::random::key(options.seed));
  auto qr = mx::linalg::qr(gauss, mx::Device::cpu);
  auto rotation = mx::contiguous(qr.first);
  auto centers_rot = mx::matmul(pad_to(centers, dp), rotation);
  mx::eval(centers, rotation, centers_rot);
  IvfRabitqIndex index{
      centers,
      centers_rot,
      rotation,
      mx::array(0),
      mx::array(0),
      mx::array(0),
      mx::array(0),
      metric,
      options.bits_per_dim};
  double scale = options.bits_per_dim > 1
      ? const_scale(dp, options.bits_per_dim - 1, options.seed)
      : 0.0;
  index.scale = (float)scale;
  auto labels = mx::reshape(kmeans_predict(rows, centers, 1), {n});
  mx::array codes(0), factors(0);
  rabitq_rows(
      index, rows, labels, scale, options.fast_quantize, codes, factors);
  mx::array grouped(0);
  group(
      mx::concatenate(
          {mx::reshape(codes, {n, -1}), mx::view(factors, mx::uint32)}, 1),
      mx::arange(n, mx::int32),
      labels,
      options.n_lists,
      grouped,
      index.indices,
      index.offsets);
  int cw = options.bits_per_dim * (dp / 32);
  index.codes = mx::contiguous(
      mx::reshape(
          mx::slice(grouped, {0, 0}, {n, cw}),
          {n, options.bits_per_dim, dp / 32}));
  index.factors = mx::contiguous(
      mx::view(mx::slice(grouped, {0, cw}, {n, cw + 4}), mx::float32));
  mx::eval(index.codes, index.factors);
  return index;
}

IvfRabitqIndex ivf_rabitq_extend(
    const IvfRabitqIndex& index,
    const mx::array& new_rows_in,
    const std::optional<mx::array>& new_ids,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_rabitq_extend";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(new_rows_in, op, "new_rows");
  int n = index.codes.shape(0);
  int m = rows.shape(0);
  int dp = index.rotation.shape(0);
  if (rows.shape(1) != index.centers.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_extend] new_rows must have the index's columns.");
  }
  if (new_ids && (new_ids->ndim() != 1 || new_ids->shape(0) != m)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_extend] new_ids must hold one id per new row.");
  }
  if (index.metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  int lists = index.centers.shape(0);
  auto ids = next_ids(index.indices, m, new_ids);
  auto labels = mx::reshape(kmeans_predict(rows, index.centers, 1), {m});
  mx::array codes(0), factors(0);
  rabitq_rows(index, rows, labels, index.scale, true, codes, factors);
  int cw = index.bits_per_dim * (dp / 32);
  auto packed_old = mx::concatenate(
      {mx::reshape(index.codes, {n, cw}), mx::view(index.factors, mx::uint32)},
      1);
  auto packed_new = mx::concatenate(
      {mx::reshape(codes, {m, cw}), mx::view(factors, mx::uint32)}, 1);
  IvfRabitqIndex out = index;
  mx::array grouped(0);
  group(
      mx::concatenate({packed_old, packed_new}, 0),
      mx::concatenate({index.indices, ids}, 0),
      mx::concatenate({labels_from_offsets(index.offsets, n), labels}, 0),
      lists,
      grouped,
      out.indices,
      out.offsets);
  int total = n + m;
  out.codes = mx::contiguous(
      mx::reshape(
          mx::slice(grouped, {0, 0}, {total, cw}),
          {total, index.bits_per_dim, dp / 32}));
  out.factors = mx::contiguous(
      mx::view(mx::slice(grouped, {0, cw}, {total, cw + 4}), mx::float32));
  mx::eval(out.codes, out.factors);
  return out;
}

Neighbors ivf_rabitq_search(
    const IvfRabitqIndex& index,
    const mx::array& queries_in,
    int k,
    const IvfRabitqSearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_rabitq_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto queries = detail::float_rows(queries_in, op, "queries");
  int nq = queries.shape(0);
  int d = index.centers.shape(1);
  int dp = index.rotation.shape(0);
  int lists = index.centers.shape(0);
  if (queries.shape(1) != d) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_search] queries must have the index's columns.");
  }
  if (k < 1 || k > kernels::TOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::ivf_rabitq_search] k must be in [1, 256].");
  }
  int probes_n = std::clamp(options.n_probes, 1, lists);
  bool l2 = index.metric == Metric::L2;
  if (index.metric == Metric::Cosine) {
    queries = unit_rows(queries);
  }
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }
  int bq = options.mode == RabitqMode::QUANT4 ? 4
      : options.mode == RabitqMode::QUANT8    ? 8
                                              : 0;
  auto rq = mx::contiguous(mx::matmul(pad_to(queries, dp), index.rotation));
  // the rotated query tile stays within 12 KB of threadgroup memory
  int qt = std::clamp(3072 / dp, 1, 16);
  auto scan = [&](const mx::array& brq,
                  const mx::array& probes,
                  const mx::array& slots,
                  int width,
                  const mx::array& tau) {
    int np = probes.shape(1);
    int nb = probes.shape(0);
    auto g = group_pairs(probes, lists, qt);
    auto params = mx::array({np, width, l2 ? 1 : 0}, {3}, mx::int32);
    return rabitq_scan_kernel()(
        {index.codes,
         index.factors,
         brq,
         index.centers_rot,
         g.items,
         g.pairs,
         index.offsets,
         mx::flatten(slots),
         params,
         tau},
        {{nb, width}, {nb, width}},
        {mx::float32, mx::int32},
        {g.n_items * kernels::IVF_THREADS, 1, 1},
        {kernels::IVF_THREADS, 1, 1},
        {{"QT", qt}, {"DP", dp}, {"B", index.bits_per_dim}, {"BQ", bq}},
        -INFINITY,
        false,
        {});
  };
  auto [probes, slots, width] =
      probe(index.centers, index.offsets, queries, probes_n, index.metric);
  auto sizes = mx::subtract(
      mx::slice(index.offsets, {1}, {lists + 1}),
      mx::slice(index.offsets, {0}, {lists}));
  int block = (int)std::clamp<long long>(
      IVF_BUFFER_BYTES / (8LL * width), 1, std::max(1, nq));
  std::vector<Neighbors> parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int nb = q1 - q0;
    auto brq = mx::slice(rq, {q0, 0}, {q1, dp});
    auto bprobes = mx::slice(probes, {q0, 0}, {q1, probes_n});
    auto bslots = mx::slice(slots, {q0, 0}, {q1, probes_n});
    // each query's threshold: the k-th best score of a search of its nearest
    // lists (enough of them to hold k rows), as cuVS's running threshold
    auto tau = mx::full({nb}, -INFINITY);
    if (probes_n > 1) {
      auto held =
          mx::cumsum(mx::take(sizes, mx::maximum(bprobes, mx::array(0))), 1);
      int pre =
          1 + mx::max(mx::sum(mx::less(held, mx::array(k)), 1)).item<int>();
      if (pre < probes_n) {
        auto p1 = mx::slice(bprobes, {0, 0}, {nb, pre});
        auto s1 = mx::slice(bslots, {0, 0}, {nb, pre});
        int w1 = std::max(
            1, mx::max(mx::slice(held, {0, pre - 1}, {nb, pre})).item<int>());
        auto first = scan(brq, p1, s1, w1, mx::full({nb}, -INFINITY));
        int kk = std::min(k, w1);
        tau = mx::flatten(
            mx::negative(
                mx::slice(
                    mx::partition(mx::negative(first[0]), kk - 1, 1),
                    {0, kk - 1},
                    {nb, kk})));
      }
    }
    auto out = scan(brq, bprobes, bslots, width, tau);
    auto r = finish(out[0], out[1], index.indices, k, l2);
    if (nq > block) {
      mx::eval(r.ids, r.distances);
    }
    parts.push_back(r);
  }
  return concat(parts);
}

namespace {

// cuVS's pq_dim heuristic: half the dimension from 128 up, rounded down to a
// multiple of 32, else the largest power of two not above it.
int default_pq_dim(int d) {
  int dim = d >= 128 ? d / 2 : d;
  int r = dim / 32 * 32;
  if (r > 0) {
    return r;
  }
  r = 1;
  while ((r << 1) <= dim) {
    r <<= 1;
  }
  return r;
}

// Rows padded with zero columns to rot_dim (cuVS rotates them when the
// dimension does not divide into pq_dim subspaces; zero padding is the
// identity rotation).
mx::array pad_cols(const mx::array& x, int rot_dim) {
  int d = x.shape(1);
  return d == rot_dim ? x : mx::pad(x, {{0, 0}, {0, rot_dim - d}});
}

// The codes of residuals [m, rot_dim] against codebooks [pq_dim, ncodes,
// pq_len]: uint8 [m, pq_dim].
mx::array encode(const mx::array& residuals, const mx::array& codebooks) {
  int pq_dim = codebooks.shape(0);
  int pq_len = codebooks.shape(2);
  int m = residuals.shape(0);
  std::vector<mx::array> cols;
  for (int j = 0; j < pq_dim; ++j) {
    auto sub = mx::slice(residuals, {0, j * pq_len}, {m, (j + 1) * pq_len});
    auto cb = mx::squeeze(
        mx::slice(codebooks, {j, 0, 0}, {j + 1, codebooks.shape(1), pq_len}),
        0);
    cols.push_back(mx::astype(kmeans_predict(sub, cb, 1), mx::uint8));
  }
  auto codes = mx::concatenate(cols, 1);
  mx::eval(codes);
  return codes;
}

} // namespace

IvfPqIndex ivf_pq_build(
    const mx::array& rows_in,
    Metric metric,
    const IvfPqOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_pq_build";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  int n = rows.shape(0);
  int d = rows.shape(1);
  if (options.n_lists < 1 || options.n_lists > n) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_build] n_lists must be in [1, number of rows].");
  }
  if (options.pq_bits < 4 || options.pq_bits > 8) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_build] pq_bits must be in [4, 8].");
  }
  int ncodes = 1 << options.pq_bits;
  int pq_dim = options.pq_dim > 0 ? options.pq_dim : default_pq_dim(d);
  int pq_len = (d + pq_dim - 1) / pq_dim;
  int rot_dim = pq_dim * pq_len;
  if (rot_dim > kernels::IVF_PQ_ROT_DIM_MAX) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_build] pq_dim * ceil(dim / pq_dim) must be at most " +
        std::to_string(kernels::IVF_PQ_ROT_DIM_MAX) + ".");
  }
  if (n < ncodes) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_build] needs at least 2^pq_bits rows.");
  }
  if (metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  rows = pad_cols(rows, rot_dim);

  KMeansOptions km;
  km.max_iter = options.kmeans_n_iters;
  km.seed = options.seed;
  int train = (int)std::min<double>(
      n, std::max<double>(n * options.kmeans_trainset_fraction, 64.0 * ncodes));
  km.sample_rows = std::max(64 * options.n_lists, train);
  auto centers = kmeans_fit(rows, options.n_lists, km);
  auto labels = mx::reshape(kmeans_predict(rows, centers, 1), {n});
  auto residuals = mx::subtract(rows, mx::take(centers, labels, 0));
  mx::eval(centers, residuals);

  // a codebook per subspace, fitted on the residuals of a sample (as cuVS's
  // PER_SUBSPACE codebooks)
  // 64 rows per code is plenty for the codebooks' low-dimensional k-means
  KMeansOptions pq;
  pq.max_iter = options.kmeans_n_iters;
  pq.seed = options.seed + 1;
  pq.sample_rows = std::min(train, 64 * ncodes);
  std::vector<mx::array> books;
  for (int j = 0; j < pq_dim; ++j) {
    auto sub = mx::slice(residuals, {0, j * pq_len}, {n, (j + 1) * pq_len});
    books.push_back(mx::expand_dims(kmeans_fit(sub, ncodes, pq), 0));
  }
  auto codebooks = mx::concatenate(books, 0);
  mx::eval(codebooks);

  IvfPqIndex index{
      centers, codebooks, mx::array(0), mx::array(0), mx::array(0), metric, d};
  group(
      encode(residuals, codebooks),
      mx::arange(n, mx::int32),
      labels,
      options.n_lists,
      index.codes,
      index.indices,
      index.offsets);
  return index;
}

IvfPqIndex ivf_pq_extend(
    const IvfPqIndex& index,
    const mx::array& new_rows_in,
    const std::optional<mx::array>& new_ids,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_pq_extend";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(new_rows_in, op, "new_rows");
  int n = index.codes.shape(0);
  int m = rows.shape(0);
  if (rows.shape(1) != index.dim) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_extend] new_rows must have the index's columns.");
  }
  if (new_ids && (new_ids->ndim() != 1 || new_ids->shape(0) != m)) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_extend] new_ids must hold one id per new row.");
  }
  if (index.metric == Metric::Cosine) {
    rows = unit_rows(rows);
  }
  rows = pad_cols(rows, index.centers.shape(1));
  auto labels = mx::reshape(kmeans_predict(rows, index.centers, 1), {m});
  auto residuals = mx::subtract(rows, mx::take(index.centers, labels, 0));
  IvfPqIndex out{
      index.centers,
      index.codebooks,
      mx::array(0),
      mx::array(0),
      mx::array(0),
      index.metric,
      index.dim};
  group(
      mx::concatenate({index.codes, encode(residuals, index.codebooks)}, 0),
      mx::concatenate({index.indices, next_ids(index.indices, m, new_ids)}, 0),
      mx::concatenate({labels_from_offsets(index.offsets, n), labels}, 0),
      index.centers.shape(0),
      out.codes,
      out.indices,
      out.offsets);
  return out;
}

Neighbors ivf_pq_search(
    const IvfPqIndex& index,
    const mx::array& queries_in,
    int k,
    const IvfSearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::ivf_pq_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto queries = detail::float_rows(queries_in, op, "queries");
  int nq = queries.shape(0);
  int lists = index.centers.shape(0);
  int rot_dim = index.centers.shape(1);
  int pq_dim = index.codebooks.shape(0);
  int ncodes = index.codebooks.shape(1);
  int pq_len = index.codebooks.shape(2);
  if (queries.shape(1) != index.dim) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_search] queries must have the index's columns.");
  }
  if (k < 1 || k > kernels::TOPK_MAX) {
    throw std::invalid_argument(
        "[mlxvs::ivf_pq_search] k must be in [1, 256].");
  }
  int probes_n = std::clamp(options.n_probes, 1, lists);
  bool l2 = index.metric == Metric::L2;
  if (index.metric == Metric::Cosine) {
    queries = unit_rows(queries);
  }
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }
  queries = mx::contiguous(pad_cols(queries, rot_dim));
  auto [probes, slots, width] =
      probe(index.centers, index.offsets, queries, probes_n, index.metric);
  // the lookup table in chunks of subspaces within 8 KB
  int ch = std::max(1, 2048 / ncodes);
  int block = (int)std::clamp<long long>(
      IVF_BUFFER_BYTES / (8LL * width), 1, std::max(1, nq));
  std::vector<Neighbors> parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int bq = q1 - q0;
    auto params = mx::array({probes_n, width, l2 ? 1 : 0}, {3}, mx::int32);
    auto out = pq_scan_kernel()(
        {index.codes,
         index.codebooks,
         index.centers,
         mx::slice(queries, {q0, 0}, {q1, rot_dim}),
         mx::slice(probes, {q0, 0}, {q1, probes_n}),
         index.offsets,
         mx::slice(slots, {q0, 0}, {q1, probes_n}),
         params},
        {{bq, width}, {bq, width}},
        {mx::float32, mx::int32},
        {probes_n * kernels::IVF_THREADS, bq, 1},
        {kernels::IVF_THREADS, 1, 1},
        {{"PQ_DIM", pq_dim},
         {"PQ_LEN", pq_len},
         {"NCODES", ncodes},
         {"CH", ch},
         {"RCAP", rot_dim}},
        -INFINITY,
        false,
        {});
    auto r = finish(out[0], out[1], index.indices, k, l2);
    if (nq > block) {
      mx::eval(r.ids, r.distances);
    }
    parts.push_back(r);
  }
  return concat(parts);
}

Neighbors refine(
    const mx::array& dataset,
    const mx::array& queries_in,
    const mx::array& candidates,
    int k,
    Metric metric,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::refine";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto queries = detail::float_rows(queries_in, op, "queries");
  int nq = queries.shape(0);
  int d = queries.shape(1);
  if (dataset.ndim() != 2 || dataset.shape(1) != d || candidates.ndim() != 2 ||
      candidates.shape(0) != nq) {
    throw std::invalid_argument(
        "[mlxvs::refine] dataset [n, d], queries [nq, d] and candidates [nq, "
        "c] must match.");
  }
  int c = candidates.shape(1);
  if (k < 1 || k > c) {
    throw std::invalid_argument(
        "[mlxvs::refine] k must be in [1, candidates per query].");
  }
  bool l2 = metric == Metric::L2;
  if (metric == Metric::Cosine) {
    queries = unit_rows(queries);
  }
  int n = dataset.shape(0);
  auto cand = mx::astype(candidates, mx::int32);
  auto valid = mx::logical_and(cand >= mx::array(0), cand < mx::array(n));
  int block = (int)std::clamp<long long>(
      (256LL << 20) / (4LL * c * d), 1, std::max(1, nq));
  std::vector<Neighbors> parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int bq = q1 - q0;
    auto bc = mx::slice(cand, {q0, 0}, {q1, c});
    auto bv = mx::slice(valid, {q0, 0}, {q1, c});
    auto rows = mx::reshape(
        mx::astype(
            mx::take(dataset, mx::flatten(mx::where(bv, bc, mx::array(0))), 0),
            mx::float32),
        {bq, c, d});
    if (metric == Metric::Cosine) {
      auto norm = mx::sqrt(mx::sum(mx::square(rows), 2, true));
      rows = rows / mx::maximum(norm, mx::array(1e-30f));
    }
    auto q = mx::expand_dims(mx::slice(queries, {q0, 0}, {q1, d}), 1);
    // larger is closer
    auto score = l2 ? mx::negative(mx::sum(mx::square(rows - q), 2))
                    : mx::sum(rows * q, 2);
    score = mx::where(bv, score, mx::array(-INFINITY));
    auto order =
        mx::slice(mx::argsort(mx::negative(score), 1), {0, 0}, {bq, k});
    auto best = mx::take_along_axis(score, order, 1);
    auto missing = mx::isinf(best);
    auto ids =
        mx::where(missing, mx::array(-1), mx::take_along_axis(bc, order, 1));
    auto dist = l2 ? mx::where(missing, mx::array(INFINITY), mx::negative(best))
                   : mx::where(missing, mx::array(-INFINITY), best);
    if (nq > block) {
      mx::eval(ids, dist);
    }
    parts.push_back({ids, dist});
  }
  return concat(parts);
}

} // namespace mlxvs
