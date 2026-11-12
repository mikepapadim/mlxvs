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
