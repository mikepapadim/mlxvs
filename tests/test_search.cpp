// Copyright © 2026 mikepapadim

//
// graph_search against exact k-NN (brute_force_knn): recall over a k-NN graph,
// and the structure of the results (ids distinct and in range, sorted,
// distances equal to the exact ones).
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"

struct SearchCase {
  const char* name;
  Metric metric = Metric::L2;
  int k = 10;
  int itopk = 64;
  int width = 1;
  double min_recall = 0.95;
};

// recall@k of graph_search (an id counts when its exact distance reaches the
// exact k-th one), checking the structure of every result row
static void search_case(
    const SearchCase& c,
    const std::vector<float>& host,
    int n,
    int d,
    const std::vector<float>& qhost,
    int nq,
    const mx::array& graph) {
  mx::array base(host.data(), {n, d}, mx::float32);
  mx::array queries(qhost.data(), {nq, d}, mx::float32);
  auto exact = mlxvs::brute_force_knn(base, queries, c.k, c.metric);
  mx::eval(exact.ids, exact.distances);
  mlxvs::SearchOptions options;
  options.itopk_size = c.itopk;
  options.search_width = c.width;
  auto t0 = std::chrono::steady_clock::now();
  auto got = mlxvs::graph_search(base, graph, queries, c.k, c.metric, options);
  mx::eval(got.ids, got.distances);
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const int* ids = got.ids.data<int>();
  const float* dist = got.distances.data<float>();
  const float* edist = exact.distances.data<float>();
  bool l2 = c.metric == Metric::L2;
  long hits = 0;
  bool valid = true, distinct = true, sorted = true;
  double max_err = 0;
  for (int i = 0; i < nq; ++i) {
    float kth = edist[(size_t)i * c.k + c.k - 1];
    float tol = 1e-5f * std::max(1.0f, std::abs(kth));
    std::unordered_set<int> seen;
    for (int j = 0; j < c.k; ++j) {
      int id = ids[(size_t)i * c.k + j];
      float dj = dist[(size_t)i * c.k + j];
      if (id < 0 || id >= n) {
        valid = false;
        continue;
      }
      distinct &= seen.insert(id).second;
      double e = exact_distance(
          c.metric, &qhost[(size_t)i * d], &host[(size_t)id * d], d);
      max_err =
          std::max(max_err, std::abs(e - dj) / std::max(1.0, std::abs(e)));
      hits += l2 ? dj <= kth + tol : dj >= kth - tol;
      if (j > 0) {
        float prev = dist[(size_t)i * c.k + j - 1];
        sorted &= l2 ? prev <= dj : prev >= dj;
      }
    }
  }
  double recall = hits / (double)((long)nq * c.k);
  bool ok =
      recall >= c.min_recall && valid && distinct && sorted && max_err < 1e-4;
  std::printf(
      "%-6s %-22s %-6s n=%d d=%d nq=%d k=%d itopk=%d width=%d: recall %.4f, %.0f queries/s%s%s%s\n",
      ok ? "PASS" : "FAIL",
      c.name,
      metric_name(c.metric),
      n,
      d,
      nq,
      c.k,
      c.itopk,
      c.width,
      recall,
      nq / seconds,
      valid ? "" : ", INVALID ids",
      distinct ? "" : ", DUPLICATE ids",
      sorted ? "" : ", NOT sorted");
  std::string t(c.name);
  check(recall >= c.min_recall, (t + " recall").c_str());
  check(valid && distinct, (t + " ids").c_str());
  check(sorted, (t + " order").c_str());
  check(max_err < 1e-4, (t + " distances").c_str());
}

// rows near an r-dimensional subspace: Gaussian coordinates in r dimensions,
// mapped by a random r x d matrix, plus a little noise (as embeddings, whose
// intrinsic dimension is much lower than d)
static std::vector<float> low_rank(int n, int d, int r, unsigned seed) {
  auto z = gaussian(n, r, seed);
  auto a = gaussian(r, d, seed + 1);
  auto noise = gaussian(n, d, seed + 2);
  std::vector<float> v((size_t)n * d);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < d; ++j) {
      float acc = 0.1f * noise[(size_t)i * d + j];
      for (int t = 0; t < r; ++t)
        acc += z[(size_t)i * r + t] * a[(size_t)t * d + j];
      v[(size_t)i * d + j] = acc;
    }
  return v;
}

// the exact k-NN graph of the rows, the row itself excluded
static mx::array knn_graph(const mx::array& x, int degree, Metric m) {
  mlxvs::AllNeighborsOptions options;
  options.algo = mlxvs::GraphBuild::BruteForce;
  options.n_clusters = 1;
  auto g = mlxvs::all_neighbors(x, degree, m, options);
  mx::eval(g.ids);
  return g.ids;
}

int main() {
  {
    int n = 20000, d = 48, nq = 1000;
    auto all = low_rank(n + nq, d, 12, 7);
    std::vector<float> host(all.begin(), all.begin() + (size_t)n * d);
    std::vector<float> qhost(all.begin() + (size_t)n * d, all.end());
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, 32, Metric::L2);
    search_case({"l2 itopk 64"}, host, n, d, qhost, nq, g);
    search_case(
        {"l2 width 4", Metric::L2, 10, 64, 4}, host, n, d, qhost, nq, g);
    search_case(
        {"l2 k=100 itopk 256", Metric::L2, 100, 256, 2},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"l2 k=1 itopk 32", Metric::L2, 1, 32, 1, 0.9},
        host,
        n,
        d,
        qhost,
        nq,
        g);

    // a graph with a quarter of its entries missing (-1)
    auto holes = mx::where(
        mx::random::uniform({n, 32}) < mx::array(0.25f), mx::array(-1), g);
    mx::eval(holes);
    search_case(
        {"l2 graph with holes", Metric::L2, 10, 64, 1, 0.85},
        host,
        n,
        d,
        qhost,
        nq,
        holes);

    auto gc = knn_graph(x, 32, Metric::Cosine);
    search_case({"cosine itopk 64", Metric::Cosine}, host, n, d, qhost, nq, gc);
    auto gi = knn_graph(x, 32, Metric::InnerProduct);
    search_case(
        {"inner itopk 128", Metric::InnerProduct, 10, 128, 1, 0.8},
        host,
        n,
        d,
        qhost,
        nq,
        gi);

    // a graph from nn_descent, as built in practice
    auto gn = mlxvs::nn_descent(x, 32, Metric::L2);
    mx::eval(gn.ids);
    search_case({"l2 nn_descent graph"}, host, n, d, qhost, nq, gn.ids);
  }
  {
    // queries too wide for threadgroup memory: read from device memory
    int n = 3000, d = 1500, nq = 200;
    auto all = low_rank(n + nq, d, 12, 11);
    std::vector<float> host(all.begin(), all.begin() + (size_t)n * d);
    std::vector<float> qhost(all.begin() + (size_t)n * d, all.end());
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, 24, Metric::L2);
    search_case({"l2 d=1500"}, host, n, d, qhost, nq, g);
  }
  {
    // fewer rows than one round's candidates
    int n = 300, d = 16, nq = 100;
    auto host = gaussian(n, d, 3);
    auto qhost = gaussian(nq, d, 4);
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, 16, Metric::L2);
    search_case(
        {"l2 n=300", Metric::L2, 10, 64, 2, 0.99}, host, n, d, qhost, nq, g);
  }
  {
    // fewer rows than k: every row once, then -1 (a complete graph, so that
    // every row is reachable)
    int n = 20, d = 8, k = 30;
    auto host = gaussian(n, d, 5);
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, n - 1, Metric::L2);
    auto got =
        mlxvs::graph_search(x, g, mx::slice(x, {0, 0}, {4, d}), k, Metric::L2);
    mx::eval(got.ids, got.distances);
    const int* ids = got.ids.data<int>();
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
      std::unordered_set<int> seen;
      for (int j = 0; j < k; ++j) {
        int id = ids[i * k + j];
        ok &= j < n ? (id >= 0 && id < n && seen.insert(id).second) : id == -1;
      }
    }
    std::printf("%-6s n=20 k=30: all rows, then -1\n", ok ? "PASS" : "FAIL");
    check(ok, "n < k");
  }
  {
    // argument checks
    int n = 100, d = 8;
    auto host = gaussian(n, d, 6);
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, 8, Metric::L2);
    bool threw = false;
    try {
      mlxvs::SearchOptions o;
      o.search_width = 100;
      mlxvs::graph_search(x, g, x, 10, Metric::L2, o);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "search_width rejected");
    threw = false;
    try {
      mlxvs::graph_search(x, mx::slice(g, {0, 0}, {50, 8}), x, 10, Metric::L2);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "graph rows rejected");
  }
  std::printf(failures == 0 ? "all tests passed\n" : "%d FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
