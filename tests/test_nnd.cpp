// Copyright © 2026 mikepapadim

//
// NN-Descent against the exact k-NN graph (brute_force_knn, self excluded) on
// Gaussian and clustered data.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <unordered_set>
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

static const char* metric_name(Metric m) {
  switch (m) {
    case Metric::InnerProduct:
      return "inner";
    case Metric::Cosine:
      return "cosine";
    case Metric::L2:
      return "l2";
  }
  return "?";
}

static std::vector<float> gaussian(int n, int d, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g;
  std::vector<float> v((size_t)n * d);
  for (auto& x : v)
    x = g(rng);
  return v;
}

// `clusters` Gaussian blobs (centers N(0, 4^2), points N(center, 1)) of random
// sizes
static std::vector<float> clustered(int n, int d, int clusters, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g;
  std::vector<float> centers((size_t)clusters * d);
  for (auto& c : centers)
    c = 4.0f * g(rng);
  std::uniform_int_distribution<int> pick(0, clusters - 1);
  std::vector<float> v((size_t)n * d);
  for (int i = 0; i < n; ++i) {
    int c = pick(rng);
    for (int j = 0; j < d; ++j)
      v[(size_t)i * d + j] = centers[(size_t)c * d + j] + g(rng);
  }
  return v;
}

static double exact_distance(Metric m, const float* a, const float* b, int d) {
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
      return l2;
  }
  return 0;
}

// recall@k and recall@10 of nn_descent against the exact graph, and the
// structure of the result: ids in range, distinct, never the row itself,
// distances sorted and equal to the exact ones
static void nnd_case(
    const char* data_name,
    const std::vector<float>& host,
    int n,
    int d,
    int k,
    Metric m,
    bool fp16,
    double min_recall) {
  mx::array x(host.data(), {n, d}, mx::float32);
  // exact: k + 1 neighbors, the row itself dropped (for inner product a row
  // need not be its own nearest)
  auto exact = mlxvs::brute_force_knn(x, x, k + 1, m);
  mx::eval(exact.ids, exact.distances);
  std::vector<long long> updates;
  mlxvs::NnDescentOptions options;
  options.fp16 = fp16;
  options.updates = &updates;
  auto t0 = std::chrono::steady_clock::now();
  auto approx = mlxvs::nn_descent(x, k, m, options);
  mx::eval(approx.ids, approx.distances);
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const int* eid = exact.ids.data<int>();
  const int* aid = approx.ids.data<int>();
  const float* adist = approx.distances.data<float>();
  int r10 = std::min(10, k);
  long hit_k = 0, hit_10 = 0, empty = 0;
  bool valid = true, distinct = true, no_self = true, sorted = true;
  double max_err = 0;
  // an entry counts when its distance is at least as close as the exact k-th
  // (10th) one, so that ties (frequent with integer data such as SIFT) are not
  // misses, as JVector's MetalSmoke counts
  const float* edist = exact.distances.data<float>();
  auto closer = [m](float a, float b) {
    float tol = 1e-5f * std::max(1.0f, std::abs(b));
    return m == Metric::L2 ? a <= b + tol : a >= b - tol;
  };
  std::vector<float> truth;
  for (int i = 0; i < n; ++i) {
    truth.clear();
    for (int j = 0; j <= k && (int)truth.size() < k; ++j) {
      if (eid[(size_t)i * (k + 1) + j] != i)
        truth.push_back(edist[(size_t)i * (k + 1) + j]);
    }
    float kth = truth[k - 1];
    float tenth = truth[r10 - 1];
    std::unordered_set<int> seen;
    const int* row = aid + (size_t)i * k;
    const float* dist = adist + (size_t)i * k;
    for (int j = 0; j < k; ++j) {
      int id = row[j];
      if (id < 0) {
        ++empty;
        continue;
      }
      valid &= id < n;
      no_self &= id != i;
      distinct &= seen.insert(id).second;
      hit_k += closer(dist[j], kth);
      if (j < r10)
        hit_10 += closer(dist[j], tenth);
      if (j > 0 && row[j - 1] >= 0) {
        sorted &=
            m == Metric::L2 ? dist[j - 1] <= dist[j] : dist[j - 1] >= dist[j];
      }
      if (i % 97 == 0 && id < n) {
        double e =
            exact_distance(m, &host[(size_t)i * d], &host[(size_t)id * d], d);
        max_err = std::max(
            max_err, std::abs(e - dist[j]) / std::max(1.0, std::abs(e)));
      }
    }
  }
  double recall_k = hit_k / (double)((long)n * k);
  double recall_10 = hit_10 / (double)((long)n * r10);
  int rounds = (int)updates.size();
  bool ok = recall_k >= min_recall && valid && distinct && no_self && sorted &&
      empty == 0 && max_err < 1e-4;
  std::printf(
      "%-6s nnd %-9s %-6s %s n=%d d=%d k=%d: recall@k %.4f, recall@10 %.4f, rounds %d (4 start), last updates %lld, "
      "%.2f s%s%s%s%s%s\n",
      ok ? "PASS" : "FAIL",
      data_name,
      metric_name(m),
      fp16 ? "fp16" : "fp32",
      n,
      d,
      k,
      recall_k,
      recall_10,
      rounds,
      updates.empty() ? 0LL : updates.back(),
      seconds,
      valid ? "" : ", INVALID ids",
      distinct ? "" : ", DUPLICATE ids",
      no_self ? "" : ", SELF in list",
      sorted ? "" : ", NOT sorted",
      empty == 0 ? "" : ", EMPTY entries");
  if (max_err >= 1e-4)
    std::printf("       max relative distance error %.2e\n", max_err);
  check(recall_k >= min_recall, "nnd recall@k");
  check(valid && distinct && no_self, "nnd ids");
  check(sorted, "nnd order");
  check(empty == 0, "nnd filled");
  check(max_err < 1e-4, "nnd distances");
}

// the rows of an .fvecs file (each row: int32 dim, then dim floats); empty if
// it cannot be read
static std::vector<float> read_fvecs(const char* path, int& n, int& d) {
  std::ifstream in(path, std::ios::binary);
  std::vector<float> v;
  n = 0;
  d = 0;
  int dim;
  while (in.read(reinterpret_cast<char*>(&dim), 4)) {
    if (d == 0)
      d = dim;
    if (dim != d)
      return {};
    size_t at = v.size();
    v.resize(at + d);
    if (!in.read(
            reinterpret_cast<char*>(v.data() + at), (std::streamsize)d * 4))
      return {};
    ++n;
  }
  return v;
}

// fewer rows than k + 1: every other row is listed, the rest is -1 padding
static void small_case() {
  int n = 40, d = 37, k = 64;
  auto host = gaussian(n, d, 5);
  mx::array x(host.data(), {n, d}, mx::float32);
  auto r = mlxvs::nn_descent(x, k, Metric::L2);
  mx::eval(r.ids, r.distances);
  const int* ids = r.ids.data<int>();
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> seen;
    for (int j = 0; j < k; ++j) {
      int id = ids[(size_t)i * k + j];
      if (j < n - 1) {
        ok &= id >= 0 && id < n && id != i && seen.insert(id).second;
      } else {
        ok &= id == -1;
      }
    }
  }
  std::printf(
      "%-6s nnd small     n=%d d=%d k=%d: all other rows listed, then -1\n",
      ok ? "PASS" : "FAIL",
      n,
      d,
      k);
  check(ok, "nnd small");
}

int main() {
  small_case();
  auto g64 = gaussian(20000, 64, 11);
  auto g128 = gaussian(10000, 128, 12);
  auto c64 = clustered(20000, 64, 64, 13);
  auto c128 = clustered(16000, 128, 40, 14);
  auto odd = clustered(10000, 37, 32, 15);
  double min_recall = 0.90;
  // iid Gaussian rows in 64+ dimensions are NN-Descent's worst case: no
  // structure for "a neighbor's neighbor is a neighbor", and L2 hubness; the
  // lists converge to a fixed point at about 0.78 (L2) / 0.88-0.90 (cosine,
  // inner product) here, as in JVector's Metal build (its notes: 50000 x 128
  // converges at 0.62). A regression floor only
  double gauss_floor = 0.70;
  // every metric, d and k on both kinds of data, FP16 SIMD-group join
  nnd_case("gauss", g64, 20000, 64, 32, Metric::L2, true, gauss_floor);
  nnd_case("gauss", g64, 20000, 64, 96, Metric::Cosine, true, gauss_floor);
  nnd_case(
      "gauss", g128, 10000, 128, 32, Metric::InnerProduct, true, gauss_floor);
  nnd_case("gauss", g128, 10000, 128, 96, Metric::L2, true, gauss_floor);
  nnd_case(
      "clustered", c64, 20000, 64, 96, Metric::InnerProduct, true, min_recall);
  nnd_case("clustered", c64, 20000, 64, 32, Metric::Cosine, true, min_recall);
  nnd_case("clustered", c128, 16000, 128, 96, Metric::L2, true, min_recall);
  nnd_case(
      "clustered",
      c128,
      16000,
      128,
      32,
      Metric::InnerProduct,
      true,
      min_recall);
  // the FP32 SIMT join, and an odd dimension (scalar FP16 staging, zero-padded
  // tile)
  nnd_case("gauss", g64, 20000, 64, 32, Metric::L2, false, gauss_floor);
  nnd_case(
      "clustered", c128, 16000, 128, 96, Metric::Cosine, false, min_recall);
  nnd_case("clust-odd", odd, 10000, 37, 32, Metric::L2, true, min_recall);
  // optional real data, e.g. MLXVS_FVECS=.../siftsmall_base.fvecs (JVector's
  // Metal build: recall@96 1.0000 L2, 0.9925 cosine at 8 rounds)
  if (const char* path = std::getenv("MLXVS_FVECS")) {
    int n, d;
    auto rows = read_fvecs(path, n, d);
    if (rows.empty() || n > 20000) {
      std::printf("FAIL   could not read %s (or more than 20000 rows)\n", path);
      ++failures;
    } else {
      nnd_case("fvecs", rows, n, d, 96, Metric::L2, true, min_recall);
      nnd_case("fvecs", rows, n, d, 96, Metric::Cosine, true, min_recall);
      nnd_case("fvecs", rows, n, d, 32, Metric::InnerProduct, true, min_recall);
    }
  }
  std::printf(
      failures == 0 ? "all tests passed\n" : "%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
