// Copyright © 2026 mikepapadim

//
// Exact k-NN against a CPU reference (double precision) on random data.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "mlxvs/mlxvs.h"

namespace mx = mlx::core;
using mlxvs::Metric;

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}

static std::vector<float> random_rows(int n, int d, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g;
  std::vector<float> v((size_t)n * d);
  for (auto& x : v)
    x = g(rng);
  return v;
}

static double score(Metric m, const float* a, const float* b, int d) {
  double dot = 0, aa = 0, bb = 0, l2 = 0;
  for (int i = 0; i < d; ++i) {
    dot += (double)a[i] * b[i];
    aa += (double)a[i] * a[i];
    bb += (double)b[i] * b[i];
    l2 += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
  }
  switch (m) {
    case Metric::InnerProduct:
      return dot;
    case Metric::Cosine:
      return dot / std::sqrt(aa * bb);
    case Metric::L2:
      return -l2;
  }
  return 0;
}

// recall of the GPU ids against the exact top k (ties tolerated: an id counts
// when its exact score reaches the k-th)
static void
knn_case(Metric m, const char* name, int n, int nq, int d, int k, int tile) {
  auto base = random_rows(n, d, 1);
  auto queries = random_rows(nq, d, 2);
  mx::array b(base.data(), {n, d}, mx::float32);
  mx::array q(queries.data(), {nq, d}, mx::float32);
  auto result = mlxvs::brute_force_knn(b, q, k, m, {tile});
  mx::eval(result.ids, result.distances);
  const int* ids = result.ids.data<int>();
  const float* dist = result.distances.data<float>();
  int keff = std::min(k, n);
  long right = 0;
  double max_err = 0;
  bool ordered = true;
  for (int i = 0; i < nq; ++i) {
    std::vector<double> s(n);
    for (int j = 0; j < n; ++j)
      s[j] = score(m, &queries[(size_t)i * d], &base[(size_t)j * d], d);
    std::vector<double> sorted = s;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    double kth = sorted[keff - 1];
    for (int j = 0; j < k; ++j) {
      int id = ids[(size_t)i * k + j];
      if (j >= keff) {
        check(id == -1, "padding ids are -1");
        continue;
      }
      if (id < 0 || id >= n)
        continue;
      right += s[id] >= kth - 1e-4 * std::max(1.0, std::abs(kth));
      double expected = m == Metric::L2 ? -s[id] : s[id];
      max_err = std::max(
          max_err,
          std::abs(expected - dist[(size_t)i * k + j]) /
              std::max(1.0, std::abs(expected)));
      if (j > 0) {
        float prev = dist[(size_t)i * k + j - 1], cur = dist[(size_t)i * k + j];
        ordered &= m == Metric::L2 ? prev <= cur + 1e-5f : prev >= cur - 1e-5f;
      }
    }
  }
  double recall = right / (double)((long)nq * keff);
  std::printf(
      "%-6s %-12s n=%d nq=%d d=%d k=%d tile=%d: recall %.5f, max rel error %.2e, %s\n",
      recall > 0.999 && ordered ? "PASS" : "FAIL",
      name,
      n,
      nq,
      d,
      k,
      tile,
      recall,
      max_err,
      ordered ? "sorted" : "NOT sorted");
  check(recall > 0.999, "recall");
  check(ordered, "order");
  check(max_err < 1e-3, "distances");
}

// k-means on well-separated blobs: every blob gets its own centroid,
// predictions match the blobs
static void kmeans_case() {
  int blobs = 8, per = 2000, d = 16;
  std::mt19937 rng(9);
  std::normal_distribution<float> g;
  std::vector<float> centers((size_t)blobs * d), rows((size_t)blobs * per * d);
  for (auto& c : centers)
    c = 20.0f * g(rng);
  std::vector<int> truth(blobs * per);
  for (int i = 0; i < blobs * per; ++i) {
    truth[i] = i % blobs;
    for (int j = 0; j < d; ++j)
      rows[(size_t)i * d + j] = centers[(size_t)truth[i] * d + j] + g(rng);
  }
  mx::array x(rows.data(), {blobs * per, d}, mx::float32);
  // k-means++ starts: every seed must separate the blobs
  int best_agree = blobs * per;
  for (unsigned long long seed : {1ull, 2ull, 3ull, 4ull}) {
    auto c = mlxvs::kmeans_fit(x, blobs, {20, 100000, seed});
    auto labels = mlxvs::kmeans_predict(x, c, 2, 5000);
    mx::eval(labels);
    const int* l = labels.data<int>();
    // map each blob to its most common label (first choice), count agreement
    int agree = 0;
    for (int b = 0; b < blobs; ++b) {
      std::vector<int> votes(blobs, 0);
      for (int i = b; i < blobs * per; i += blobs)
        votes[l[(size_t)i * 2]]++;
      agree += *std::max_element(votes.begin(), votes.end());
    }
    best_agree = std::min(best_agree, agree);
  }
  double purity = best_agree / (double)(blobs * per);
  std::printf(
      "%-6s kmeans       8 blobs x 2000 x 16: purity %.4f (worst of 4 seeds), predict m=2\n",
      purity > 0.999 ? "PASS" : "FAIL",
      purity);
  check(purity > 0.999, "kmeans purity");
}

int main() {
  kmeans_case();
  knn_case(Metric::InnerProduct, "inner", 5000, 64, 64, 10, 65536);
  knn_case(Metric::InnerProduct, "inner tiles", 5000, 64, 64, 100, 777);
  knn_case(Metric::Cosine, "cosine tiles", 5000, 64, 96, 96, 1024);
  knn_case(Metric::L2, "l2 tiles", 5000, 64, 128, 32, 999);
  knn_case(Metric::L2, "l2 k=256", 3000, 16, 32, 256, 1000);
  knn_case(Metric::InnerProduct, "n < k", 50, 8, 16, 100, 20);
  knn_case(Metric::Cosine, "k=1", 4000, 32, 64, 1, 512);
  std::printf(
      failures == 0 ? "all tests passed\n" : "%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
