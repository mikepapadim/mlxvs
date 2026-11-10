// Copyright © 2026 mikepapadim

//
// Helpers of the graph tests: data, checks, and a k-NN graph scored against the
// exact one.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <unordered_set>
#include <vector>

#include "mlxvs/mlxvs.h"

namespace mx = mlx::core;
using mlxvs::Metric;

inline int failures = 0;

inline void check(bool ok, const char* what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}

inline const char* metric_name(Metric m) {
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

inline std::vector<float> gaussian(int n, int d, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g;
  std::vector<float> v((size_t)n * d);
  for (auto& x : v)
    x = g(rng);
  return v;
}

// `clusters` Gaussian blobs (centers N(0, 4^2), points N(center, 1)) of random
// sizes
inline std::vector<float> clustered(int n, int d, int clusters, unsigned seed) {
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

// the rows of an .fvecs file (each row: int32 dim, then dim floats); empty if
// it cannot be read
inline std::vector<float> read_fvecs(const char* path, int& n, int& d) {
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

inline double exact_distance(Metric m, const float* a, const float* b, int d) {
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

/** A k-NN graph of the rows against the exact one. */
struct GraphQuality {
  double recall_k = 0;
  double recall_10 = 0;
  bool valid = true; // ids in range
  bool distinct = true;
  bool no_self = true;
  bool sorted = true;
  long empty = 0; // -1 entries
  double max_err = 0; // relative error of the distances, sampled rows

  bool structure() const {
    return valid && distinct && no_self && sorted && empty == 0 &&
        max_err < 1e-4;
  }

  // the flags that failed, for a result line
  void print_flags() const {
    std::printf(
        "%s%s%s%s%s",
        valid ? "" : ", INVALID ids",
        distinct ? "" : ", DUPLICATE ids",
        no_self ? "" : ", SELF in list",
        sorted ? "" : ", NOT sorted",
        empty == 0 ? "" : ", EMPTY entries");
  }

  // the structural checks, named after the test
  void check_structure(const char* test) const {
    std::string t(test);
    if (max_err >= 1e-4)
      std::printf("       max relative distance error %.2e\n", max_err);
    check(valid && distinct && no_self, (t + " ids").c_str());
    check(sorted, (t + " order").c_str());
    check(empty == 0, (t + " filled").c_str());
    check(max_err < 1e-4, (t + " distances").c_str());
  }
};

/**
 * The exact k-NN graph of the rows, the row itself excluded: k + 1 neighbors
 * from brute_force_knn (for inner product a row need not be its own nearest).
 */
inline mlxvs::Neighbors exact_graph(const mx::array& x, int k, Metric m) {
  auto exact = mlxvs::brute_force_knn(x, x, k + 1, m);
  mx::eval(exact.ids, exact.distances);
  return exact;
}

/**
 * Recall@k and recall@10 of graph [n, k] against exact (exact_graph), and its
 * structure: ids in range, distinct, never the row itself, distances sorted
 * and equal to the exact ones. An entry counts when its distance is at least as
 * close as the exact k-th (10th) one, so that ties (frequent with integer data
 * such as SIFT) are not misses, as JVector's MetalSmoke counts.
 */
inline GraphQuality score_graph(
    const std::vector<float>& host,
    int n,
    int d,
    int k,
    Metric m,
    const mlxvs::Neighbors& graph,
    const mlxvs::Neighbors& exact) {
  GraphQuality q;
  const int* eid = exact.ids.data<int>();
  const float* edist = exact.distances.data<float>();
  const int* aid = graph.ids.data<int>();
  const float* adist = graph.distances.data<float>();
  int r10 = std::min(10, k);
  long hit_k = 0, hit_10 = 0;
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
        ++q.empty;
        continue;
      }
      q.valid &= id < n;
      q.no_self &= id != i;
      q.distinct &= seen.insert(id).second;
      hit_k += closer(dist[j], kth);
      if (j < r10)
        hit_10 += closer(dist[j], tenth);
      if (j > 0 && row[j - 1] >= 0) {
        q.sorted &=
            m == Metric::L2 ? dist[j - 1] <= dist[j] : dist[j - 1] >= dist[j];
      }
      if (i % 97 == 0 && id < n) {
        double e =
            exact_distance(m, &host[(size_t)i * d], &host[(size_t)id * d], d);
        q.max_err = std::max(
            q.max_err, std::abs(e - dist[j]) / std::max(1.0, std::abs(e)));
      }
    }
  }
  q.recall_k = hit_k / (double)((long)n * k);
  q.recall_10 = hit_10 / (double)((long)n * r10);
  return q;
}
