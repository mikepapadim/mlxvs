// Copyright © 2026 mikepapadim

//
// cagra_optimize against a CPU reference of the same steps (rank-based
// pruning, reverse edges, merge), and graph_search over the pruned graph
// against the k-NN graph of the same degree.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "common.h"

// The CPU reference. tied[x] is set where x has more than d reverse edges and
// the cut falls among edges of one rank: which of those the GPU keeps depends
// on the order they arrive in, as in cuVS.
static std::vector<int> reference_optimize(
    const std::vector<int>& knn,
    int n,
    int k,
    int d,
    std::vector<char>& tied) {
  std::vector<int> pruned((size_t)n * d, -1);
  for (int x = 0; x < n; ++x) {
    const int* nx = &knn[(size_t)x * k];
    std::vector<int> det(k, 0);
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < j; ++i)
        for (int r = 0; r < j; ++r)
          det[j] += knn[(size_t)nx[i] * k + r] == nx[j];
    std::vector<int> order(k);
    for (int j = 0; j < k; ++j)
      order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return det[a] < det[b];
    });
    for (int t = 0; t < d; ++t)
      pruned[(size_t)x * d + t] = nx[order[t]];
  }
  std::vector<std::vector<std::pair<int, int>>> rev(n);
  for (int x = 0; x < n; ++x)
    for (int r = 0; r < d; ++r)
      rev[pruned[(size_t)x * d + r]].push_back({r, x});
  tied.assign(n, 0);
  std::vector<int> out((size_t)n * d, -1);
  for (int x = 0; x < n; ++x) {
    std::sort(rev[x].begin(), rev[x].end());
    tied[x] = (int)rev[x].size() > d && rev[x][d - 1].first == rev[x][d].first;
    std::vector<int> list;
    auto add = [&](int v) {
      if ((int)list.size() < d &&
          std::find(list.begin(), list.end(), v) == list.end())
        list.push_back(v);
    };
    for (int j = 0; j < d / 2; ++j)
      list.push_back(pruned[(size_t)x * d + j]);
    for (int t = 0; t < std::min((int)rev[x].size(), d); ++t)
      add(rev[x][t].second);
    for (int j = d / 2; j < d; ++j)
      add(pruned[(size_t)x * d + j]);
    std::copy(list.begin(), list.end(), out.begin() + (size_t)x * d);
  }
  return out;
}

// rows near a 10-dimensional subspace (as in test_search)
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

static void reference_case(int n, int dim, int k, int d) {
  auto host = low_rank(n, dim, 10, 21);
  mx::array x(host.data(), {n, dim}, mx::float32);
  mlxvs::AllNeighborsOptions o;
  o.algo = mlxvs::GraphBuild::BruteForce;
  o.n_clusters = 1;
  auto knn = mlxvs::all_neighbors(x, k, Metric::L2, o).ids;
  auto got = mlxvs::cagra_optimize(knn, d);
  mx::eval(knn, got);
  std::vector<int> kv(knn.data<int>(), knn.data<int>() + (size_t)n * k);
  std::vector<char> tied;
  auto want = reference_optimize(kv, n, k, d, tied);
  const int* g = got.data<int>();
  long same = 0, compared = 0;
  bool valid = true;
  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> seen;
    for (int t = 0; t < d; ++t) {
      int v = g[(size_t)i * d + t];
      valid &= v >= 0 && v < n && v != i && seen.insert(v).second;
    }
    // reverse lists over capacity keep the edges of lowest rank; where the
    // cut falls among edges of one rank, compare the protected half only
    int upto = tied[i] ? d / 2 : d;
    for (int t = 0; t < upto; ++t) {
      same += g[(size_t)i * d + t] == want[(size_t)i * d + t];
      ++compared;
    }
  }
  bool ok = valid && same == compared;
  std::printf(
      "%-6s optimize vs CPU   n=%d k=%d degree=%d: %ld/%ld entries equal%s\n",
      ok ? "PASS" : "FAIL",
      n,
      k,
      d,
      same,
      compared,
      valid ? "" : ", INVALID or repeated ids");
  check(ok, "optimize matches the CPU reference");
}

// recall@10 of graph_search over graph, ties counted as hits
static double search_recall(
    const mx::array& x,
    const mx::array& graph,
    const mx::array& q,
    int itopk) {
  int k = 10;
  auto exact = mlxvs::brute_force_knn(x, q, k, Metric::L2);
  mlxvs::SearchOptions o;
  o.itopk_size = itopk;
  auto got = mlxvs::graph_search(x, graph, q, k, Metric::L2, o);
  mx::eval(exact.distances, got.distances);
  const float* e = exact.distances.data<float>();
  const float* g = got.distances.data<float>();
  long hits = 0;
  int nq = q.shape(0);
  for (int i = 0; i < nq; ++i) {
    float kth = e[(size_t)i * k + k - 1];
    for (int j = 0; j < k; ++j)
      hits += g[(size_t)i * k + j] <= kth + 1e-5f * std::max(1.0f, kth);
  }
  return hits / (double)((long)nq * k);
}

static void search_case() {
  int n = 30000, dim = 64, nq = 2000;
  auto all = low_rank(n + nq, dim, 16, 31);
  mx::array x(all.data(), {n, dim}, mx::float32);
  mx::array q(all.data() + (size_t)n * dim, {nq, dim}, mx::float32);
  mlxvs::CagraOptions o;
  o.intermediate_graph_degree = 64;
  o.graph_degree = 16;
  auto cagra = mlxvs::cagra_build(x, Metric::L2, o);
  auto knn = mlxvs::nn_descent(x, 16, Metric::L2).ids;
  mx::eval(cagra, knn);
  for (int itopk : {32, 64}) {
    double rc = search_recall(x, cagra, q, itopk);
    double rk = search_recall(x, knn, q, itopk);
    bool ok = rc > rk && rc > 0.9;
    std::printf(
        "%-6s search degree 16 itopk %d: recall@10 %.4f pruned, %.4f k-NN\n",
        ok ? "PASS" : "FAIL",
        itopk,
        rc,
        rk);
    check(ok, "pruned graph searches better than the k-NN graph");
  }
}

int main() {
  reference_case(3000, 32, 32, 16);
  reference_case(2000, 16, 64, 32);
  reference_case(1000, 8, 24, 7);
  search_case();
  {
    bool threw = false;
    try {
      mlxvs::cagra_optimize(mx::zeros({10, 8}, mx::int32), 9);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "graph_degree above the input degree rejected");
  }
  std::printf(failures == 0 ? "all tests passed\n" : "%d FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
