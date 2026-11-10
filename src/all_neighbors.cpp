// Copyright © 2026 mikepapadim

//
// All-neighbors k-NN graph, as cuVS's all_neighbors::build: k-means splits the
// rows into overlapping clusters (each row joins its overlap_factor nearest
// centroids), the graph of every cluster is built on its own, and every row
// keeps the best k of the lists its clusters gave it. A cluster's local ids
// are mapped back to row ids before the merge; a pair found in two clusters is
// kept once.
//
// The rows are read through a gather function, so they can stay on the host (a
// NumPy array, a memory-mapped file larger than memory): only a k-means
// sample, one tile during assignment and one cluster at a time are on the GPU.
// With n_clusters = 0 the number of clusters follows from the GPU's memory;
// a cluster larger than the cap is clustered again (never filled up to a
// capacity by sending rows to farther clusters, which loses their neighbors).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
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

/** Clusters are sized to this share of the GPU's recommended working set. */
constexpr double GPU_SHARE = 0.4;
/** Automatic clusters average 1 / SLACK of the cap: k-means clusters are
 * uneven. */
constexpr double SLACK = 1.5;
/** Rows assigned to clusters per step. */
constexpr int ASSIGN_TILE = 65536;

/** How the build reads the rows. */
struct Rows {
  int n;
  int d;
  /** Rows ids[0..m) as float32 [m, d]. */
  std::function<mx::array(const int* ids, int m)> gather;
  /** All rows as one array, when they already are one. */
  const mx::array* whole = nullptr;
};

/**
 * GPU bytes of one row of a cluster in nn_descent: the FP32 and FP16 rows, two
 * generations of the list (ids, scores, flags), the proposal slots, the
 * members and the result.
 */
size_t row_bytes(int d, int k) {
  size_t list = std::max(64, k);
  return 6 * (size_t)d + 24 * list + 8 * (256 - list) + 256 + 8 * (size_t)k;
}

/** The most rows of one cluster. */
long long cluster_cap(int d, int k, const AllNeighborsOptions& options) {
  if (options.max_cluster_rows > 0) {
    return std::max(2LL, options.max_cluster_rows);
  }
  size_t memory = 8ull << 30;
  const auto& info = mx::device_info(mx::Device::gpu);
  for (const char* key : {"max_recommended_working_set_size", "memory_size"}) {
    auto it = info.find(key);
    if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
      memory = std::get<size_t>(it->second);
      break;
    }
  }
  long long by_memory = (long long)(GPU_SHARE * memory / row_bytes(d, k));
  // one threadgroup of 256 threads per row in a dispatch
  long long by_dispatch = std::numeric_limits<int>::max() / 256;
  return std::max(2LL, std::min(by_memory, by_dispatch));
}

/** The members of ids[0..m) nearest each of their `overlap` centroids. */
std::vector<std::vector<int>> assign(
    const Rows& rows,
    const std::vector<int>& ids,
    int clusters,
    int overlap,
    Metric metric,
    const AllNeighborsOptions& options,
    unsigned long long seed) {
  int m = (int)ids.size();
  // k-means on a random sample of the members
  KMeansOptions kmeans = options.kmeans;
  kmeans.seed = seed;
  int sample_size = std::min(m, std::max(kmeans.sample_rows, clusters * 64));
  std::vector<int> pick(ids);
  if (sample_size < m) {
    std::mt19937_64 rng(seed);
    std::shuffle(pick.begin(), pick.end(), rng);
    pick.resize(sample_size);
    std::sort(pick.begin(), pick.end());
  }
  auto space = [&](mx::array x) {
    return metric == Metric::Cosine ? unit_rows(x) : x;
  };
  auto sample = space(rows.gather(pick.data(), sample_size));
  mx::eval(sample);
  auto centroids = kmeans_fit(sample, clusters, kmeans);
  mx::eval(centroids);
  std::vector<std::vector<int>> members(clusters);
  for (int t0 = 0; t0 < m; t0 += ASSIGN_TILE) {
    int t = std::min(ASSIGN_TILE, m - t0);
    auto tile = space(rows.gather(ids.data() + t0, t));
    auto nearest = kmeans_predict(tile, centroids, overlap);
    mx::eval(nearest);
    const int* a = nearest.data<int>();
    for (int i = 0; i < t; ++i)
      for (int j = 0; j < overlap; ++j)
        members[a[(size_t)i * overlap + j]].push_back(ids[t0 + i]);
  }
  return members;
}

/**
 * Splits ids into clusters of at most cap rows, ascending row ids in each: the
 * k-means clusters, those larger than cap clustered again.
 */
void partition(
    const Rows& rows,
    std::vector<int> ids,
    int clusters,
    int overlap,
    long long cap,
    Metric metric,
    const AllNeighborsOptions& options,
    int depth,
    std::vector<std::vector<int>>& out) {
  if (clusters <= 1) {
    out.push_back(std::move(ids));
    return;
  }
  auto parts = assign(
      rows,
      ids,
      clusters,
      overlap,
      metric,
      options,
      options.kmeans.seed + depth);
  ids = {};
  for (auto& part : parts) {
    long long size = (long long)part.size();
    if (size <= cap) {
      out.push_back(std::move(part));
    } else if (depth < 4) {
      // its rows join the 2 nearest of enough smaller clusters
      int sub_overlap = 2;
      int sub = (int)std::max(
          2LL, (long long)std::ceil(SLACK * sub_overlap * size / cap));
      partition(
          rows,
          std::move(part),
          sub,
          sub_overlap,
          cap,
          metric,
          options,
          depth + 1,
          out);
    } else {
      // k-means cannot separate these rows (e.g. repeated rows): equal parts
      long long pieces = (size + cap - 1) / cap;
      for (long long piece = 0; piece < pieces; ++piece) {
        out.emplace_back(
            part.begin() + size * piece / pieces,
            part.begin() + size * (piece + 1) / pieces);
      }
    }
  }
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

Neighbors build(
    const Rows& rows,
    int k,
    Metric metric,
    const AllNeighborsOptions& options) {
  int n = rows.n;
  int d = rows.d;
  int max_k = options.algo == GraphBuild::NnDescent ? 128 : 255;
  if (k < 1 || k > max_k || n < 2 || d < 1) {
    throw std::invalid_argument(
        "[mlxvs::all_neighbors] 1 <= k <= " + std::to_string(max_k) +
        ", n >= 2 and d >= 1");
  }
  if (options.n_clusters < 0 || options.n_clusters > n ||
      options.overlap_factor < 1 ||
      (options.n_clusters > 1 && options.overlap_factor > options.n_clusters)) {
    throw std::invalid_argument(
        "[mlxvs::all_neighbors] 0 <= n_clusters <= n and 1 <= overlap_factor "
        "<= n_clusters");
  }
  long long cap = cluster_cap(d, k, options);
  int clusters = options.n_clusters;
  int overlap = options.overlap_factor;
  if (clusters == 0) {
    // automatic: one cluster when the rows fit, else clusters of cap / SLACK
    clusters = n <= cap
        ? 1
        : (int)std::ceil(SLACK * overlap * (double)n / (double)cap);
    overlap = std::min(overlap, clusters);
  }
  std::vector<std::vector<int>> members;
  {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    partition(
        rows,
        std::move(all),
        clusters,
        overlap,
        cap,
        metric,
        options,
        0,
        members);
  }
  if (options.verbose) {
    std::fprintf(
        stderr,
        "[mlxvs::all_neighbors] %d rows x %d in %zu clusters (at most %lld "
        "rows each)\n",
        n,
        d,
        members.size(),
        cap);
  }

  mx::array ids = mx::full({n, k}, -1, mx::int32);
  mx::array dists = mx::full({n, k}, pad_distance(metric), mx::float32);
  for (size_t c = 0; c < members.size(); ++c) {
    int size = (int)members[c].size();
    if (size < 2) {
      continue; // a lone row has no neighbor in its cluster
    }
    mx::array idx(members[c].data(), {size}, mx::int32);
    mx::array x = size == n && rows.whole != nullptr
        ? *rows.whole
        : rows.gather(members[c].data(), size);
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
    if (options.verbose) {
      std::fprintf(
          stderr,
          "[mlxvs::all_neighbors] cluster %zu / %zu: %d rows\n",
          c + 1,
          members.size(),
          size);
    }
  }
  return {ids, dists};
}

} // namespace

Neighbors all_neighbors(
    const mx::array& rows_in,
    int k,
    Metric metric,
    const AllNeighborsOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::all_neighbors"));
  auto x = detail::float_rows(rows_in, "mlxvs::all_neighbors");
  Rows rows{x.shape(0), x.shape(1), nullptr, &x};
  rows.gather = [&x](const int* ids, int m) {
    return mx::take(x, mx::array(ids, {m}, mx::int32), 0);
  };
  return build(rows, k, metric, options);
}

Neighbors all_neighbors(
    const HostRows& host,
    int k,
    Metric metric,
    const AllNeighborsOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::all_neighbors"));
  if (host.data == nullptr || host.n < 2 || host.d < 1 ||
      host.n > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "[mlxvs::all_neighbors] host rows: data, 2 <= n < 2^31 and d >= 1");
  }
  Rows rows{(int)host.n, host.d, nullptr, nullptr};
  // the rows of a cluster copied by every core into one page-aligned buffer,
  // which MLX takes without a further copy
  rows.gather = [host](const int* ids, int m) {
    size_t row = (size_t)host.d * sizeof(float);
    size_t page = 16384;
    size_t bytes = ((size_t)m * row + page - 1) / page * page;
    void* buffer = nullptr;
    if (posix_memalign(&buffer, page, bytes) != 0) {
      throw std::bad_alloc();
    }
    int threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(threads, std::max(1, m / 1024));
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([=] {
        int i0 = (int)((long long)m * t / threads);
        int i1 = (int)((long long)m * (t + 1) / threads);
        for (int i = i0; i < i1; ++i) {
          std::memcpy(
              (char*)buffer + (size_t)i * row,
              (const char*)host.data + (size_t)ids[i] * row,
              row);
        }
      });
    }
    for (auto& thread : pool)
      thread.join();
    return mx::array(buffer, mx::Shape{m, host.d}, mx::float32, [](void* p) {
      std::free(p);
    });
  };
  return build(rows, k, metric, options);
}

} // namespace mlxvs
