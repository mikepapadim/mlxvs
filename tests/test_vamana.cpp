// Copyright © 2026 mikepapadim

//
// vamana_prune against a CPU reference of JVector's diversity rule
// (VamanaDiversityProvider.retainDiverse), and graph_search over vamana_build's
// graph.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "common.h"

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

// JVector's similarity: 1 / (1 + squared distance) for L2, (1 + dot) / 2
static float sim(const float* a, const float* b, int d, bool l2) {
  double acc = 0;
  for (int j = 0; j < d; ++j)
    acc += l2 ? (a[j] - b[j]) * (double)(a[j] - b[j]) : a[j] * (double)b[j];
  return l2 ? (float)(1.0 / (1.0 + acc)) : (float)((1.0 + acc) / 2.0);
}

// JVector's retainDiverse over one row's candidates
static std::vector<int> reference_prune(
    const std::vector<float>& x,
    int d,
    int node,
    const int* cand,
    int c,
    int degree,
    float alpha,
    bool l2) {
  int n = (int)(x.size() / d);
  std::vector<int> ids;
  for (int i = 0; i < c; ++i) {
    int v = cand[i];
    if (v >= 0 && v < n && v != node &&
        std::find(ids.begin(), ids.end(), v) == ids.end())
      ids.push_back(v);
  }
  std::vector<float> s(ids.size());
  for (size_t i = 0; i < ids.size(); ++i)
    s[i] = sim(&x[(size_t)node * d], &x[(size_t)ids[i] * d], d, l2);
  std::vector<int> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return s[a] > s[b] || (s[a] == s[b] && ids[a] < ids[b]);
  });
  std::vector<char> kept(order.size(), 0);
  int nkept = 0;
  for (float a = 1.0f; a <= alpha + 1e-6f && nkept < degree; a += 0.2f) {
    for (size_t p = 0; p < order.size() && nkept < degree; ++p) {
      if (kept[p])
        continue;
      int i = order[p];
      bool diverse = true;
      for (size_t q = 0; q < order.size() && diverse; ++q)
        if (kept[q])
          diverse = sim(&x[(size_t)ids[i] * d],
                        &x[(size_t)ids[order[q]] * d],
                        d,
                        l2) <= s[i] * a;
      if (diverse) {
        kept[p] = 1;
        ++nkept;
      }
    }
  }
  std::vector<int> out;
  for (size_t p = 0; p < order.size(); ++p)
    if (kept[p])
      out.push_back(ids[order[p]]);
  out.resize(degree, -1);
  return out;
}

static void prune_case(Metric metric, int n, int dim, int k, int degree) {
  bool l2 = metric == Metric::L2;
  auto host = low_rank(n, dim, 8, 31);
  if (!l2) {
    // unit rows: inner product and cosine agree, and the reference needs no
    // normalization
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int j = 0; j < dim; ++j)
        s += host[(size_t)i * dim + j] * (double)host[(size_t)i * dim + j];
      for (int j = 0; j < dim; ++j)
        host[(size_t)i * dim + j] /= (float)std::sqrt(s);
    }
  }
  mx::array x(host.data(), {n, dim}, mx::float32);
  mlxvs::AllNeighborsOptions o;
  o.algo = mlxvs::GraphBuild::BruteForce;
  o.n_clusters = 1;
  auto knn = mlxvs::all_neighbors(x, k, metric, o).ids;
  // holes and repeats, which must be skipped
  auto cand = mx::where(
      mx::random::uniform({n, k}) < mx::array(0.05f), mx::array(-1), knn);
  cand = mx::concatenate({cand, mx::slice(knn, {0, 0}, {n, 4})}, 1);
  auto got = mlxvs::vamana_prune(x, cand, degree, metric, 1.2f);
  mx::eval(cand, got);
  int c = cand.shape(1);
  const int* cv = cand.data<int>();
  const int* g = got.data<int>();
  long same = 0;
  int rows_equal = 0;
  for (int i = 0; i < n; ++i) {
    auto want =
        reference_prune(host, dim, i, cv + (size_t)i * c, c, degree, 1.2f, l2);
    int eq = 0;
    for (int t = 0; t < degree; ++t)
      eq += g[(size_t)i * degree + t] == want[t];
    same += eq;
    rows_equal += eq == degree;
  }
  // the GPU's similarities come from norms and dot products (a matrix
  // multiply), the reference's from differences: near-ties may go either way
  double frac = rows_equal / (double)n;
  bool ok = frac >= 0.98;
  std::printf(
      "%-6s prune vs JVector rule %-6s n=%d k=%d degree=%d: %.4f of rows equal, %.4f of entries\n",
      ok ? "PASS" : "FAIL",
      metric_name(metric),
      n,
      k,
      degree,
      frac,
      same / (double)((long)n * degree));
  check(ok, "vamana_prune matches the reference");
}

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

static void build_case() {
  int n = 20000, dim = 32, nq = 1000;
  auto all = low_rank(n + nq, dim, 10, 41);
  mx::array x(all.data(), {n, dim}, mx::float32);
  mx::array q(all.data() + (size_t)n * dim, {nq, dim}, mx::float32);
  mlxvs::VamanaOptions o;
  o.graph_degree = 16;
  o.intermediate_graph_degree = 32;
  auto g = mlxvs::vamana_build(x, Metric::L2, o);
  mx::eval(g);
  const int* gv = g.data<int>();
  bool valid = true;
  long edges = 0;
  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> seen;
    bool tail = false;
    for (int t = 0; t < 16; ++t) {
      int v = gv[(size_t)i * 16 + t];
      if (v < 0) {
        tail = true;
        continue;
      }
      valid &= !tail && v < n && v != i && seen.insert(v).second;
      ++edges;
    }
  }
  for (int itopk : {32, 64}) {
    double r = search_recall(x, g, q, itopk);
    bool ok = valid && r >= (itopk == 32 ? 0.9 : 0.97);
    std::printf(
        "%-6s search vamana degree 16 itopk %d: recall@10 %.4f, %.1f edges per node%s\n",
        ok ? "PASS" : "FAIL",
        itopk,
        r,
        edges / (double)n,
        valid ? "" : ", INVALID graph");
    check(ok, "vamana graph search");
  }
}

int main() {
  prune_case(Metric::L2, 3000, 16, 32, 12);
  prune_case(Metric::InnerProduct, 3000, 16, 32, 12);
  prune_case(Metric::L2, 1500, 24, 96, 32);
  build_case();
  bool threw = false;
  try {
    mx::array x = mx::zeros({10, 4});
    mlxvs::vamana_prune(x, mx::zeros({10, 4}, mx::int32), 8, Metric::L2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "graph_degree above candidates rejected");
  std::printf(failures == 0 ? "all tests passed\n" : "%d FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
