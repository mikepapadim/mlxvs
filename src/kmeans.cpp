// Copyright © 2026 mikepapadim

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "mlxvs/mlxvs.h"
#include "utils.h"

namespace mlxvs {

namespace {

/** Squared distances [n, c] of rows x [n, d] to centroids c [c, d]. */
mx::array squared_distances(const mx::array& x, const mx::array& c) {
  auto xx = mx::sum(mx::square(x), 1, /*keepdims=*/true);
  auto cc = mx::expand_dims(mx::sum(mx::square(c), 1), 0);
  return mx::maximum(
      xx + cc - 2.0f * mx::matmul(x, mx::transpose(c)), mx::array(0.0f));
}

} // namespace

mx::array kmeans_fit(
    const mx::array& x,
    int clusters,
    const KMeansOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::kmeans_fit"));
  if (x.ndim() != 2 || x.dtype() != mx::float32) {
    throw std::invalid_argument("[mlxvs::kmeans_fit] float32 [n, d] expected");
  }
  int n = x.shape(0);
  int d = x.shape(1);
  if (clusters < 1 || clusters > n) {
    throw std::invalid_argument("[mlxvs::kmeans_fit] 1 <= clusters <= n");
  }
  std::mt19937_64 rng(options.seed);
  // a random sample of the rows (Lloyd on all rows of a large dataset buys
  // little)
  int sample_size = std::min(n, std::max(options.sample_rows, clusters * 64));
  std::vector<int> pick(n);
  std::iota(pick.begin(), pick.end(), 0);
  if (sample_size < n) {
    std::shuffle(pick.begin(), pick.end(), rng);
    pick.resize(sample_size);
    std::sort(pick.begin(), pick.end());
  }
  mx::array sample = sample_size < n
      ? mx::take(x, mx::array(pick.data(), {sample_size}, mx::int32), 0)
      : x;
  mx::eval(sample);
  // k-means++ start: each next centroid is a sample row drawn with probability
  // proportional to its squared distance to the nearest centroid so far
  std::vector<int> start;
  start.reserve(clusters);
  start.push_back(std::uniform_int_distribution<int>(0, sample_size - 1)(rng));
  auto sample_norms = mx::sum(mx::square(sample), 1);
  mx::array nearest = mx::full(
      {sample_size}, std::numeric_limits<float>::infinity(), mx::float32);
  for (int c = 1; c <= clusters; ++c) {
    auto row = mx::slice(sample, {start.back(), 0}, {start.back() + 1, d});
    auto dist = mx::maximum(
        sample_norms + mx::sum(mx::square(row)) -
            2.0f *
                mx::reshape(
                    mx::matmul(sample, mx::transpose(row)), {sample_size}),
        mx::array(0.0f));
    nearest = mx::minimum(nearest, dist);
    if (c == clusters) {
      break;
    }
    mx::eval(nearest);
    const float* w = nearest.data<float>();
    double total = 0;
    for (int i = 0; i < sample_size; ++i)
      total += w[i];
    int next = std::uniform_int_distribution<int>(0, sample_size - 1)(rng);
    if (total > 0) {
      double target = std::uniform_real_distribution<double>(0, total)(rng);
      double acc = 0;
      for (int i = 0; i < sample_size; ++i) {
        acc += w[i];
        if (acc >= target) {
          next = i;
          break;
        }
      }
    }
    start.push_back(next);
  }
  mx::array centroids =
      mx::take(sample, mx::array(start.data(), {clusters}, mx::int32), 0);
  mx::eval(centroids);
  for (int it = 0; it < options.max_iter; ++it) {
    auto labels = mx::argmin(squared_distances(sample, centroids), 1);
    // sums and counts per cluster as one matrix product with the one-hot labels
    auto onehot = mx::astype(
        mx::equal(
            mx::expand_dims(labels, 1),
            mx::expand_dims(mx::arange(clusters, mx::uint32), 0)),
        mx::float32);
    auto counts = mx::sum(onehot, 0);
    auto sums = mx::matmul(mx::transpose(onehot), sample);
    mx::eval(counts, sums);
    // an empty cluster restarts at a random sample row
    std::vector<float> host_counts(
        counts.data<float>(), counts.data<float>() + clusters);
    std::vector<int> empty;
    for (int c = 0; c < clusters; ++c) {
      if (host_counts[c] == 0)
        empty.push_back(c);
    }
    auto updated =
        sums / mx::expand_dims(mx::maximum(counts, mx::array(1.0f)), 1);
    if (!empty.empty()) {
      std::uniform_int_distribution<int> any(0, sample_size - 1);
      std::vector<int> rows(empty.size());
      for (auto& r : rows)
        r = any(rng);
      auto replacement = mx::take(
          sample, mx::array(rows.data(), {(int)rows.size()}, mx::int32), 0);
      updated = mx::scatter(
          updated,
          mx::array(empty.data(), {(int)empty.size()}, mx::int32),
          mx::expand_dims(replacement, 1),
          0);
    }
    centroids = updated;
    mx::eval(centroids);
  }
  return centroids;
}

mx::array kmeans_predict(
    const mx::array& x,
    const mx::array& centroids,
    int m,
    int tile_rows,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::kmeans_predict"));
  int n = x.shape(0);
  int d = x.shape(1);
  int c = centroids.shape(0);
  if (m < 1 || m > c) {
    throw std::invalid_argument("[mlxvs::kmeans_predict] 1 <= m <= clusters");
  }
  // the m nearest centroids of each row, nearest first: a [rows, c] distance
  // tile sorted per row (c is small)
  std::vector<mx::array> parts;
  for (int r0 = 0; r0 < n; r0 += tile_rows) {
    int r1 = std::min(n, r0 + tile_rows);
    auto dist = squared_distances(mx::slice(x, {r0, 0}, {r1, d}), centroids);
    auto order = m == 1 ? mx::expand_dims(mx::argmin(dist, 1), 1)
                        : mx::slice(mx::argsort(dist, 1), {0, 0}, {r1 - r0, m});
    parts.push_back(mx::astype(order, mx::int32));
    mx::eval(parts.back());
  }
  return parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
}

} // namespace mlxvs
