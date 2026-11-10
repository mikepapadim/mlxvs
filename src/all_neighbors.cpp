// Copyright © 2026 mikepapadim

//
// All-neighbors k-NN graph, as cuVS's all_neighbors::build: k-means splits the
// rows into overlapping clusters (each row joins its overlap_factor nearest
// centroids), the graph of every cluster is built on its own, and every row
// keeps the best k of the lists its clusters gave it. A cluster's local ids
// are mapped back to row ids before the merge; a pair found in two clusters is
// kept once.
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlxvs/mlxvs.h"
#include "utils.h"

namespace mlxvs {

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();

/** The value of a missing neighbor: +inf for L2, -inf for similarities. */
float pad_distance(Metric metric) {
  return metric == Metric::L2 ? INF : -INF;
}

mx::array unit_rows(const mx::array& x) {
  auto norm = mx::sqrt(mx::sum(mx::square(x), 1, /*keepdims=*/true));
  return x / mx::maximum(norm, mx::array(1e-30f));
}

/**
 * The best k of each row's current list (ids, dists [c, k]) and a cluster's
 * list for it (more_ids, more_dists [c, k2]), nearest first: -1 entries, the
 * row itself (self [c]) and repeated ids dropped, -1 padded.
 */
std::pair<mx::array, mx::array> merge_lists(
    const mx::array& self,
    const mx::array& ids,
    const mx::array& dists,
    const mx::array& more_ids,
    const mx::array& more_dists,
    int k,
    Metric metric) {
  auto all_ids = mx::concatenate({ids, more_ids}, 1);
  auto all_dists = mx::concatenate({dists, more_dists}, 1);
  // smaller key is nearer; dropped entries get +inf
  auto key = metric == Metric::L2 ? all_dists : mx::negative(all_dists);
  auto dropped =
      mx::logical_or(all_ids < 0, mx::equal(all_ids, mx::expand_dims(self, 1)));
  key = mx::where(dropped, mx::array(INF), key);
  // repeats: sorted by id, an entry equal to its predecessor is dropped (both
  // hold the FP32 distance of the same pair)
  auto by_id = mx::argsort(all_ids, 1);
  auto sid = mx::take_along_axis(all_ids, by_id, 1);
  auto skey = mx::take_along_axis(key, by_id, 1);
  auto sdist = mx::take_along_axis(all_dists, by_id, 1);
  int c = sid.shape(0);
  int w = sid.shape(1);
  auto repeat = mx::concatenate(
      {mx::zeros({c, 1}, mx::bool_),
       mx::equal(
           mx::slice(sid, {0, 1}, {c, w}), mx::slice(sid, {0, 0}, {c, w - 1}))},
      1);
  skey = mx::where(repeat, mx::array(INF), skey);
  auto best = mx::slice(mx::argsort(skey, 1), {0, 0}, {c, k});
  auto out_key = mx::take_along_axis(skey, best, 1);
  auto kept = out_key < mx::array(INF);
  auto out_ids =
      mx::where(kept, mx::take_along_axis(sid, best, 1), mx::array(-1));
  auto out_dists = mx::where(
      kept,
      mx::take_along_axis(sdist, best, 1),
      mx::array(pad_distance(metric)));
  return {out_ids, out_dists};
}

/**
 * The k-NN lists of the rows x [c, d] among themselves, local ids: exact for
 * small clusters or BruteForce (k + 1 neighbors, the row itself is dropped by
 * the merge), NN-Descent otherwise.
 */
Neighbors cluster_graph(
    const mx::array& x,
    int k,
    Metric metric,
    const AllNeighborsOptions& options) {
  int c = x.shape(0);
  bool exact = options.algo == GraphBuild::BruteForce ||
      c <= std::max(options.brute_force_below, k + 1);
  if (exact) {
    return brute_force_knn(
        x, x, std::min(k + 1, c), metric, options.brute_force);
  }
  return nn_descent(x, k, metric, options.nn_descent);
}

} // namespace

Neighbors all_neighbors(
    const mx::array& rows_in,
    int k,
    Metric metric,
    const AllNeighborsOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::all_neighbors"));
  auto rows = detail::float_rows(rows_in, "mlxvs::all_neighbors");
  int n = rows.shape(0);
  int d = rows.shape(1);
  int max_k = options.algo == GraphBuild::NnDescent ? 128 : 255;
  if (k < 1 || k > max_k || n < 2 || d < 1) {
    throw std::invalid_argument(
        "[mlxvs::all_neighbors] 1 <= k <= " + std::to_string(max_k) +
        ", n >= 2 and d >= 1");
  }
  int clusters = std::max(1, options.n_clusters);
  int overlap = clusters == 1 ? 1 : options.overlap_factor;
  if (clusters > n || overlap < 1 || overlap > clusters) {
    throw std::invalid_argument(
        "[mlxvs::all_neighbors] 1 <= n_clusters <= n and 1 <= overlap_factor "
        "<= n_clusters");
  }

  // the members of every cluster, ascending row ids
  std::vector<std::vector<int>> members(clusters);
  if (clusters == 1) {
    members[0].resize(n);
    for (int i = 0; i < n; ++i)
      members[0][i] = i;
  } else {
    mx::array space = metric == Metric::Cosine ? unit_rows(rows) : rows;
    mx::eval(space);
    auto centroids = kmeans_fit(space, clusters, options.kmeans);
    auto assign = kmeans_predict(space, centroids, overlap);
    mx::eval(assign);
    const int* a = assign.data<int>();
    std::vector<int> sizes(clusters, 0);
    for (size_t e = 0; e < (size_t)n * overlap; ++e)
      ++sizes[a[e]];
    for (int c = 0; c < clusters; ++c)
      members[c].reserve(sizes[c]);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < overlap; ++j)
        members[a[(size_t)i * overlap + j]].push_back(i);
  }

  mx::array ids = mx::full({n, k}, -1, mx::int32);
  mx::array dists = mx::full({n, k}, pad_distance(metric), mx::float32);
  for (int c = 0; c < clusters; ++c) {
    int size = (int)members[c].size();
    if (size < 2) {
      continue; // a lone row has no neighbor in its cluster
    }
    mx::array idx(members[c].data(), {size}, mx::int32);
    mx::array x = clusters == 1 ? rows : mx::take(rows, idx, 0);
    mx::eval(x);
    auto local = cluster_graph(x, k, metric, options);
    // local ids to row ids, -1 kept
    auto found = mx::where(
        local.ids >= 0,
        mx::take(idx, mx::maximum(local.ids, mx::array(0)), 0),
        mx::array(-1));
    auto merged = merge_lists(
        idx,
        mx::take(ids, idx, 0),
        mx::take(dists, idx, 0),
        found,
        local.distances,
        k,
        metric);
    ids = mx::scatter(ids, idx, mx::expand_dims(merged.first, 1), 0);
    dists = mx::scatter(dists, idx, mx::expand_dims(merged.second, 1), 0);
    mx::eval(ids, dists); // one cluster's working set at a time
  }
  return {ids, dists};
}

} // namespace mlxvs
