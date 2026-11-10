// Copyright © 2026 mikepapadim

//
// all_neighbors against the exact k-NN graph (brute_force_knn, self excluded):
// one cluster and overlapping k-means clusters, exact and NN-Descent cluster
// graphs.
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "common.h"

using mlxvs::GraphBuild;

static const char* algo_name(GraphBuild a) {
  return a == GraphBuild::BruteForce ? "exact" : "nnd";
}

// recall@k of all_neighbors and the structure of its graph; returns recall@k
static double an_case(
    const char* data_name,
    const std::vector<float>& host,
    int n,
    int d,
    int k,
    Metric m,
    GraphBuild algo,
    int clusters,
    int overlap,
    double min_recall,
    bool on_host = false,
    long long max_cluster_rows = 0) {
  mx::array x(host.data(), {n, d}, mx::float32);
  auto exact = exact_graph(x, k, m);
  mlxvs::AllNeighborsOptions options;
  options.algo = algo;
  options.n_clusters = clusters;
  options.overlap_factor = overlap;
  options.max_cluster_rows = max_cluster_rows;
  auto t0 = std::chrono::steady_clock::now();
  // on_host: the rows are read from the host vector, a cluster at a time
  auto graph = on_host
      ? mlxvs::all_neighbors(mlxvs::HostRows{host.data(), n, d}, k, m, options)
      : mlxvs::all_neighbors(x, k, m, options);
  mx::eval(graph.ids, graph.distances);
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  auto q = score_graph(host, n, d, k, m, graph, exact);
  bool ok = q.recall_k >= min_recall && q.structure();
  std::printf(
      "%-6s all-neighbors %-9s %-6s %-5s %s n=%d d=%d k=%d clusters=%d overlap=%d cap=%lld: recall@k %.4f, recall@10 %.4f, %.2f s",
      ok ? "PASS" : "FAIL",
      data_name,
      metric_name(m),
      algo_name(algo),
      on_host ? "host " : "array",
      n,
      d,
      k,
      clusters,
      overlap,
      max_cluster_rows,
      q.recall_k,
      q.recall_10,
      seconds);
  q.print_flags();
  std::printf("\n");
  check(q.recall_k >= min_recall, "all-neighbors recall@k");
  q.check_structure("all-neighbors");
  return q.recall_k;
}

// fewer rows than k + 1: every other row is listed, the rest is -1 padding
static void small_case() {
  int n = 10, d = 7, k = 32;
  auto host = gaussian(n, d, 21);
  mx::array x(host.data(), {n, d}, mx::float32);
  auto r = mlxvs::all_neighbors(x, k, Metric::L2);
  mx::eval(r.ids, r.distances);
  const int* ids = r.ids.data<int>();
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> seen;
    for (int j = 0; j < k; ++j) {
      int id = ids[(size_t)i * k + j];
      ok &= j < n - 1 ? id >= 0 && id < n && id != i && seen.insert(id).second
                      : id == -1;
    }
  }
  std::printf(
      "%-6s all-neighbors small n=%d d=%d k=%d: all other rows listed, then -1\n",
      ok ? "PASS" : "FAIL",
      n,
      d,
      k);
  check(ok, "all-neighbors small");
}

// identical rows: k-means cannot split them, the clusters are cut to the cap
static void inseparable_case() {
  int n = 9000, d = 8, k = 16;
  std::vector<float> host((size_t)n * d, 1.0f);
  mlxvs::AllNeighborsOptions options;
  options.max_cluster_rows = 2500;
  auto r = mlxvs::all_neighbors(
      mlxvs::HostRows{host.data(), n, d}, k, Metric::L2, options);
  mx::eval(r.ids, r.distances);
  const int* ids = r.ids.data<int>();
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> seen;
    for (int j = 0; j < k; ++j) {
      int id = ids[(size_t)i * k + j];
      ok &= id >= 0 && id < n && id != i && seen.insert(id).second;
    }
  }
  std::printf(
      "%-6s all-neighbors identical rows n=%d cap=2500: lists full and valid\n",
      ok ? "PASS" : "FAIL",
      n);
  check(ok, "all-neighbors inseparable");
}

static void rejects_bad_arguments() {
  auto host = gaussian(100, 8, 22);
  mx::array x(host.data(), {100, 8}, mx::float32);
  auto throws = [&](int k, int clusters, int overlap, GraphBuild algo) {
    mlxvs::AllNeighborsOptions options;
    options.algo = algo;
    options.n_clusters = clusters;
    options.overlap_factor = overlap;
    try {
      mlxvs::all_neighbors(x, k, Metric::L2, options);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  bool ok = throws(0, 1, 1, GraphBuild::NnDescent) &&
      throws(129, 1, 1, GraphBuild::NnDescent) &&
      throws(256, 1, 1, GraphBuild::BruteForce) &&
      throws(8, 101, 2, GraphBuild::BruteForce) &&
      throws(8, 4, 5, GraphBuild::BruteForce) &&
      throws(8, 4, 0, GraphBuild::BruteForce);
  std::printf(
      "%-6s all-neighbors rejects bad k, n_clusters, overlap_factor\n",
      ok ? "PASS" : "FAIL");
  check(ok, "all-neighbors arguments");
}

int main() {
  small_case();
  rejects_bad_arguments();
  inseparable_case();
  auto c64 = clustered(40000, 64, 64, 31);
  auto c128 = clustered(30000, 128, 40, 32);
  auto g32 = gaussian(20000, 32, 33);
  // one cluster: exact is brute_force_knn without the row itself
  an_case(
      "clustered",
      c64,
      20000,
      64,
      32,
      Metric::L2,
      GraphBuild::BruteForce,
      1,
      1,
      0.999);
  an_case(
      "clustered",
      c64,
      20000,
      64,
      32,
      Metric::InnerProduct,
      GraphBuild::BruteForce,
      1,
      1,
      0.999);
  // one cluster with NN-Descent
  an_case(
      "clustered",
      c64,
      20000,
      64,
      32,
      Metric::Cosine,
      GraphBuild::NnDescent,
      1,
      1,
      0.90);
  // overlapping clusters of 15000-20000 rows: NN-Descent in each
  an_case(
      "clustered",
      c64,
      40000,
      64,
      32,
      Metric::L2,
      GraphBuild::NnDescent,
      4,
      2,
      0.90);
  an_case(
      "clustered",
      c128,
      30000,
      128,
      64,
      Metric::InnerProduct,
      GraphBuild::NnDescent,
      4,
      2,
      0.90);
  an_case(
      "clustered",
      c128,
      30000,
      128,
      32,
      Metric::Cosine,
      GraphBuild::BruteForce,
      8,
      2,
      0.95);
  // rows on the host, clusters chosen for a cap of 8000 rows (NN-Descent in
  // each), and the same from an array
  an_case(
      "clustered",
      c64,
      40000,
      64,
      32,
      Metric::L2,
      GraphBuild::NnDescent,
      0,
      2,
      0.90,
      true,
      8000);
  an_case(
      "clustered",
      c128,
      30000,
      128,
      32,
      Metric::Cosine,
      GraphBuild::NnDescent,
      0,
      2,
      0.90,
      false,
      8000);
  // 2 clusters asked for, each larger than the cap: clustered again
  an_case(
      "clustered",
      c64,
      40000,
      64,
      32,
      Metric::InnerProduct,
      GraphBuild::NnDescent,
      2,
      2,
      0.90,
      true,
      9000);
  // iid rows have no clusters to respect: overlap is what finds the neighbors
  // across a boundary, so recall rises with it
  double r1 = an_case(
      "gauss", g32, 20000, 32, 16, Metric::L2, GraphBuild::BruteForce, 8, 1, 0);
  double r2 = an_case(
      "gauss", g32, 20000, 32, 16, Metric::L2, GraphBuild::BruteForce, 8, 2, 0);
  double r3 = an_case(
      "gauss", g32, 20000, 32, 16, Metric::L2, GraphBuild::BruteForce, 8, 3, 0);
  bool rises = r1 < r2 && r2 < r3;
  std::printf(
      "%-6s all-neighbors recall rises with overlap: %.4f < %.4f < %.4f\n",
      rises ? "PASS" : "FAIL",
      r1,
      r2,
      r3);
  check(rises, "all-neighbors overlap");
  std::printf(
      failures == 0 ? "all tests passed\n" : "%d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
