// Copyright © 2026 mikepapadim

//
// NN-Descent against the exact k-NN graph (brute_force_knn, self excluded) on
// Gaussian and clustered data.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "common.h"

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
  auto exact = exact_graph(x, k, m);
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
  auto q = score_graph(host, n, d, k, m, approx, exact);
  int rounds = (int)updates.size();
  bool ok = q.recall_k >= min_recall && q.structure();
  std::printf(
      "%-6s nnd %-9s %-6s %s n=%d d=%d k=%d: recall@k %.4f, recall@10 %.4f, rounds %d (4 start), last updates %lld, "
      "%.2f s",
      ok ? "PASS" : "FAIL",
      data_name,
      metric_name(m),
      fp16 ? "fp16" : "fp32",
      n,
      d,
      k,
      q.recall_k,
      q.recall_10,
      rounds,
      updates.empty() ? 0LL : updates.back(),
      seconds);
  q.print_flags();
  std::printf("\n");
  check(q.recall_k >= min_recall, "nnd recall@k");
  q.check_structure("nnd");
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
