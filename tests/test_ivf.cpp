// Copyright © 2026 mikepapadim

//
// IVF-Flat, IVF-PQ, IVF-SQ and IVF-RaBitQ: recall against exact k-NN, and
// extend.
#include <algorithm>
#include <cstdio>
#include <random>
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

static void sq_case(Metric metric) {
  int n = 20000, d = 64, nq = 500, k = 10;
  auto all = low_rank(n + nq, d, 12, 91);
  mx::array x(all.data(), {n, d}, mx::float32);
  mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
  bool l2 = metric == Metric::L2;
  mlxvs::IvfSqOptions o;
  o.n_lists = 64;
  auto index = mlxvs::ivf_sq_build(x, metric, o);
  bool shape = index.codes.shape(0) == n && index.codes.shape(1) == d &&
      index.codes.dtype() == mx::uint8;
  auto exact = mlxvs::brute_force_knn(x, q, k, metric);
  mlxvs::IvfSearchOptions so;
  so.n_probes = 64;
  auto approx = mlxvs::ivf_sq_search(index, q, k, so);
  bool valid;
  double r = recall(approx, exact, nq, k, l2, valid);
  // extend: the last 1000 rows added to an index of the others find
  // themselves
  auto part =
      mlxvs::ivf_sq_build(mx::slice(x, {0, 0}, {n - 1000, d}), metric, o);
  part = mlxvs::ivf_sq_extend(part, mx::slice(x, {n - 1000, 0}, {n, d}));
  auto self =
      mlxvs::ivf_sq_search(part, mx::slice(x, {n - 1000, 0}, {n, d}), 1, so);
  mx::eval(self.ids);
  int found = 0;
  for (int i = 0; i < 1000; ++i) {
    found += self.ids.data<int>()[i] == n - 1000 + i;
  }
  // a row is its own nearest by L2 and cosine, not by inner product
  bool ok = shape && valid && r >= 0.9 &&
      (metric == Metric::InnerProduct || found >= 950);
  std::printf(
      "%-6s ivf_sq %-6s n_probes=64: recall@10 %.4f; extend: %d/1000 find themselves\n",
      ok ? "PASS" : "FAIL",
      metric_name(metric),
      r,
      found);
  check(ok, "ivf_sq");
}

static void rabitq_case(Metric metric, int bits, mlxvs::RabitqMode mode) {
  int n = 20000, d = 64, nq = 500, k = 10;
  auto all = low_rank(n + nq, d, 12, 101);
  mx::array x(all.data(), {n, d}, mx::float32);
  mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
  bool l2 = metric == Metric::L2;
  mlxvs::IvfRabitqOptions o;
  o.n_lists = 64;
  o.bits_per_dim = bits;
  auto index = mlxvs::ivf_rabitq_build(x, metric, o);
  bool shape = index.codes.shape(0) == n && index.codes.shape(1) == bits &&
      index.codes.shape(2) == 2 && index.factors.shape(1) == 4;
  auto exact = mlxvs::brute_force_knn(x, q, k, metric);
  mlxvs::IvfRabitqSearchOptions so;
  so.n_probes = 64;
  so.mode = mode;
  auto approx = mlxvs::ivf_rabitq_search(index, q, k, so);
  auto wide = mlxvs::ivf_rabitq_search(index, q, 4 * k, so);
  auto refined = mlxvs::refine(x, q, wide.ids, k, metric);
  bool valid_a, valid_r;
  double r_a = recall(approx, exact, nq, k, l2, valid_a);
  double r_r = recall(refined, exact, nq, k, l2, valid_r);
  double floor = bits == 1 ? 0.3 : bits <= 4 ? 0.7 : 0.85;
  bool ok = shape && valid_r && r_a >= floor && r_r >= (bits == 1 ? 0.8 : 0.97);
  std::printf(
      "%-6s ivf_rabitq %-6s bits=%d mode=%d n_probes=64: recall@10 %.4f, refined from 40 %.4f\n",
      ok ? "PASS" : "FAIL",
      metric_name(metric),
      bits,
      (int)mode,
      r_a,
      r_r);
  check(ok, "ivf_rabitq");
}

static void binary_case() {
  int n = 20000, bytes = 32, nq = 300, k = 10;
  // clustered bits: rows of 40 centers with 10% of their bits flipped
  std::mt19937 rng(7);
  std::vector<uint8_t> centers(40 * bytes), rows((size_t)(n + nq) * bytes);
  for (auto& c : centers)
    c = (uint8_t)(rng() & 0xff);
  for (int i = 0; i < n + nq; ++i) {
    int c = rng() % 40;
    for (int j = 0; j < bytes; ++j) {
      uint8_t flip = 0;
      for (int b = 0; b < 8; ++b)
        flip |= (rng() % 10 == 0) << b;
      rows[(size_t)i * bytes + j] = centers[c * bytes + j] ^ flip;
    }
  }
  mx::array x(rows.data(), {n, bytes}, mx::uint8);
  mx::array q(rows.data() + (size_t)n * bytes, {nq, bytes}, mx::uint8);
  auto exact = mlxvs::hamming_knn(x, q, k);
  mx::eval(exact.ids, exact.distances);
  // the exact distances against a CPU count
  bool exact_ok = true;
  for (int i = 0; i < nq && exact_ok; ++i) {
    std::vector<int> d(n);
    for (int r = 0; r < n; ++r) {
      int h = 0;
      for (int j = 0; j < bytes; ++j)
        h += __builtin_popcount(
            rows[(size_t)r * bytes + j] ^ rows[(size_t)(n + i) * bytes + j]);
      d[r] = h;
    }
    std::nth_element(d.begin(), d.begin() + k - 1, d.end());
    exact_ok = exact.distances.data<float>()[i * k + k - 1] == (float)d[k - 1];
  }
  mlxvs::IvfBinaryOptions o;
  o.n_lists = 40;
  auto index = mlxvs::ivf_binary_build(x, o);
  mlxvs::IvfSearchOptions so;
  so.n_probes = 40;
  auto all = mlxvs::ivf_binary_search(index, q, k, so);
  so.n_probes = 4;
  auto few = mlxvs::ivf_binary_search(index, q, k, so);
  mx::eval(all.distances, few.distances);
  int same_all = 0, same_few = 0;
  for (int i = 0; i < nq; ++i) {
    float e = exact.distances.data<float>()[i * k + k - 1];
    same_all += all.distances.data<float>()[i * k + k - 1] == e;
    same_few += few.distances.data<float>()[i * k + k - 1] <= e;
  }
  bool ok = exact_ok && same_all == nq && same_few >= nq * 9 / 10;
  std::printf(
      "%-6s hamming_knn exact: %s; ivf_binary all lists %d/%d exact, 4 of 40 lists %d/%d\n",
      ok ? "PASS" : "FAIL",
      exact_ok ? "matches the CPU" : "WRONG",
      same_all,
      nq,
      same_few,
      nq);
  check(ok, "ivf_binary");
}

int main() {
  binary_case();
  sq_case(Metric::L2);
  sq_case(Metric::InnerProduct);
  sq_case(Metric::Cosine);
  rabitq_case(Metric::L2, 1, mlxvs::RabitqMode::QUANT4);
  rabitq_case(Metric::L2, 3, mlxvs::RabitqMode::QUANT4);
  rabitq_case(Metric::L2, 5, mlxvs::RabitqMode::QUANT8);
  rabitq_case(Metric::InnerProduct, 4, mlxvs::RabitqMode::LUT32);
  rabitq_case(Metric::Cosine, 7, mlxvs::RabitqMode::LUT16);
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
