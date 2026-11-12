// Copyright © 2026 mikepapadim

//
// IVF-Flat: recall against exact k-NN, exact when every list is scanned, and
// extend.
#include <algorithm>
#include <cstdio>
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

// recall@k by distance (ties count), and whether rows are sorted and ids
// distinct
static double recall(
    const mlxvs::Neighbors& got,
    const mlxvs::Neighbors& exact,
    int nq,
    int k,
    bool l2,
    bool& valid) {
  mx::eval(got.ids, got.distances, exact.distances);
  const int* ids = got.ids.data<int>();
  const float* g = got.distances.data<float>();
  const float* e = exact.distances.data<float>();
  long hits = 0;
  valid = true;
  for (int i = 0; i < nq; ++i) {
    float kth = e[(size_t)i * k + k - 1];
    std::unordered_set<int> seen;
    for (int j = 0; j < k; ++j) {
      int id = ids[(size_t)i * k + j];
      valid &= id >= 0 && seen.insert(id).second;
      float v = g[(size_t)i * k + j];
      float tol = 1e-4f * std::max(1.0f, std::abs(kth));
      hits += l2 ? v <= kth + tol : v >= kth - tol;
      if (j > 0) {
        float prev = g[(size_t)i * k + j - 1];
        valid &= l2 ? prev <= v + tol : prev >= v - tol;
      }
    }
  }
  return hits / (double)((long)nq * k);
}

static void ivf_case(Metric metric) {
  int n = 20000, d = 48, nq = 500, k = 10;
  auto all = low_rank(n + nq, d, 12, 51);
  mx::array x(all.data(), {n, d}, mx::float32);
  mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
  bool l2 = metric == Metric::L2;
  mlxvs::IvfFlatOptions o;
  o.n_lists = 128;
  auto index = mlxvs::ivf_flat_build(x, metric, o);
  auto exact = mlxvs::brute_force_knn(x, q, k, metric);
  for (int probes : {16, 128}) {
    mlxvs::IvfSearchOptions so;
    so.n_probes = probes;
    auto got = mlxvs::ivf_flat_search(index, q, k, so);
    bool valid;
    double r = recall(got, exact, nq, k, l2, valid);
    bool ok = valid && r >= (probes == 128 ? 0.999 : 0.9);
    std::printf(
        "%-6s ivf_flat %-6s lists=128 n_probes=%d: recall@10 %.4f%s\n",
        ok ? "PASS" : "FAIL",
        metric_name(metric),
        probes,
        r,
        valid ? "" : ", INVALID or unsorted");
    check(ok, "ivf_flat recall");
  }
}

static void extend_case() {
  int n = 12000, d = 32, k = 5;
  auto all = low_rank(n, d, 8, 61);
  mx::array x(all.data(), {n, d}, mx::float32);
  mlxvs::IvfFlatOptions o;
  o.n_lists = 64;
  auto index =
      mlxvs::ivf_flat_build(mx::slice(x, {0, 0}, {10000, d}), Metric::L2, o);
  index = mlxvs::ivf_flat_extend(index, mx::slice(x, {10000, 0}, {n, d}));
  // every new row finds itself first, under its new id
  mlxvs::IvfSearchOptions so;
  so.n_probes = 64;
  auto got =
      mlxvs::ivf_flat_search(index, mx::slice(x, {10000, 0}, {n, d}), k, so);
  mx::eval(got.ids);
  int self = 0;
  for (int i = 0; i < 2000; ++i) {
    self += got.ids.data<int>()[i * k] == 10000 + i;
  }
  bool ok = index.data.shape(0) == n && self == 2000;
  std::printf(
      "%-6s ivf_flat extend 10000 + 2000: %d/2000 new rows found first\n",
      ok ? "PASS" : "FAIL",
      self);
  check(ok, "ivf_flat extend");
}

// IVF-PQ: approximate on its own, near exact once refined from the rows
static void pq_case(Metric metric) {
  int n = 20000, d = 64, nq = 500, k = 10;
  auto all = low_rank(n + nq, d, 12, 71);
  mx::array x(all.data(), {n, d}, mx::float32);
  mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
  bool l2 = metric == Metric::L2;
  mlxvs::IvfPqOptions o;
  o.n_lists = 64;
  auto index = mlxvs::ivf_pq_build(x, metric, o);
  bool shape = index.codes.shape(0) == n && index.codes.shape(1) == 64 &&
      index.codes.dtype() == mx::uint8;
  auto exact = mlxvs::brute_force_knn(x, q, k, metric);
  mlxvs::IvfSearchOptions so;
  so.n_probes = 32;
  auto approx = mlxvs::ivf_pq_search(index, q, k, so);
  auto wide = mlxvs::ivf_pq_search(index, q, 4 * k, so);
  auto refined = mlxvs::refine(x, q, wide.ids, k, metric);
  bool valid_a, valid_r;
  double r_a = recall(approx, exact, nq, k, l2, valid_a);
  double r_r = recall(refined, exact, nq, k, l2, valid_r);
  bool ok = shape && valid_r && r_a >= 0.6 && r_r >= 0.97;
  std::printf(
      "%-6s ivf_pq %-6s pq_dim=64 (cuVS default for d=64) n_probes=32: recall@10 %.4f, refined from 40 %.4f%s\n",
      ok ? "PASS" : "FAIL",
      metric_name(metric),
      r_a,
      r_r,
      shape ? "" : ", WRONG code shape");
  check(ok, "ivf_pq recall");
}

static void pq_extend_case() {
  int n = 12000, d = 32, k = 5;
  auto all = low_rank(n, d, 8, 81);
  mx::array x(all.data(), {n, d}, mx::float32);
  mlxvs::IvfPqOptions o;
  o.n_lists = 32;
  auto index =
      mlxvs::ivf_pq_build(mx::slice(x, {0, 0}, {10000, d}), Metric::L2, o);
  index = mlxvs::ivf_pq_extend(index, mx::slice(x, {10000, 0}, {n, d}));
  mlxvs::IvfSearchOptions so;
  so.n_probes = 32;
  auto newq = mx::slice(x, {10000, 0}, {n, d});
  auto wide = mlxvs::ivf_pq_search(index, newq, 4 * k, so);
  auto got = mlxvs::refine(x, newq, wide.ids, k, Metric::L2);
  mx::eval(got.ids);
  int self = 0;
  for (int i = 0; i < 2000; ++i) {
    self += got.ids.data<int>()[i * k] == 10000 + i;
  }
  bool ok = index.codes.shape(0) == n && self >= 1990;
  std::printf(
      "%-6s ivf_pq extend 10000 + 2000: %d/2000 new rows found first after refine\n",
      ok ? "PASS" : "FAIL",
      self);
  check(ok, "ivf_pq extend");
}

int main() {
  pq_case(Metric::L2);
  pq_case(Metric::InnerProduct);
  pq_case(Metric::Cosine);
  pq_extend_case();
  ivf_case(Metric::L2);
  ivf_case(Metric::InnerProduct);
  ivf_case(Metric::Cosine);
  extend_case();
  std::printf(failures == 0 ? "all tests passed\n" : "%d FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
