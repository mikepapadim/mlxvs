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
  mlxvs::SearchAlgo algo = mlxvs::SearchAlgo::SingleCta;
  int team = 0;
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
  options.algo = c.algo;
  options.team_size = c.team;
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

    // multi-CTA: several threadgroups per query sharing the traversed set
    const auto mc = mlxvs::SearchAlgo::MultiCta;
    search_case(
        {"mc l2 itopk 64", Metric::L2, 10, 64, 1, 0.95, mc},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"mc l2 itopk 512", Metric::L2, 10, 512, 1, 0.99, mc},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"mc l2 width 4", Metric::L2, 10, 32, 4, 0.95, mc},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"mc l2 k=100 itopk 256", Metric::L2, 100, 256, 1, 0.95, mc},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"mc l2 graph with holes", Metric::L2, 10, 64, 1, 0.85, mc},
        host,
        n,
        d,
        qhost,
        nq,
        holes);
    search_case(
        {"mc cosine itopk 64", Metric::Cosine, 10, 64, 1, 0.95, mc},
        host,
        n,
        d,
        qhost,
        nq,
        gc);
    // a few queries: Auto picks multi-CTA
    search_case(
        {"auto l2 nq=8", Metric::L2, 10, 64, 1, 0.9, mlxvs::SearchAlgo::Auto},
        host,
        n,
        d,
        qhost,
        8,
        g);
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
    search_case(
        {"mc l2 d=1500",
         Metric::L2,
         10,
         64,
         1,
         0.95,
         mlxvs::SearchAlgo::MultiCta},
        host,
        n,
        d,
        qhost,
        nq,
        g);
  }
  {
    // filtered search: only allowed rows returned, recall against the exact
    // neighbors among them
    int n = 20000, d = 32, nq = 500, k = 10;
    auto all = low_rank(n + nq, d, 10, 17);
    mx::array x(all.data(), {n, d}, mx::float32);
    mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
    auto g = knn_graph(x, 32, Metric::L2);
    for (int every : {4, 10}) {
      // every-th row rejected (75%), or only every-th allowed (10%)
      std::vector<uint8_t> allow(n);
      std::vector<int> kept;
      for (int i = 0; i < n; ++i) {
        allow[i] = every == 4 ? i % 4 != 0 : i % 10 == 0;
        if (allow[i]) {
          kept.push_back(i);
        }
      }
      mx::array mask(allow.data(), {n}, mx::uint8);
      mask = mx::astype(mask, mx::bool_);
      mx::array kept_ids(kept.data(), {(int)kept.size()}, mx::int32);
      auto exact =
          mlxvs::brute_force_knn(mx::take(x, kept_ids, 0), q, k, Metric::L2);
      for (auto algo :
           {mlxvs::SearchAlgo::SingleCta, mlxvs::SearchAlgo::MultiCta}) {
        mlxvs::SearchOptions o;
        o.algo = algo;
        o.filter = mask;
        auto got = mlxvs::graph_search(x, g, q, k, Metric::L2, o);
        mx::eval(exact.distances, got.ids, got.distances);
        const int* ids = got.ids.data<int>();
        const float* gdist = got.distances.data<float>();
        const float* edist = exact.distances.data<float>();
        long hits = 0, missing = 0;
        bool allowed_only = true;
        for (int i = 0; i < nq; ++i) {
          float kth = edist[(size_t)i * k + k - 1];
          for (int j = 0; j < k; ++j) {
            int id = ids[(size_t)i * k + j];
            missing += id == -1;
            allowed_only &= id == -1 || (id >= 0 && id < n && allow[id]);
            hits +=
                gdist[(size_t)i * k + j] <= kth + 1e-5f * std::max(1.0f, kth);
          }
        }
        double recall = hits / (double)((long)nq * k);
        bool ok = allowed_only && recall >= (every == 4 ? 0.95 : 0.8);
        std::printf(
            "%-6s filter %s %s: recall %.4f, %ld missing%s\n",
            ok ? "PASS" : "FAIL",
            every == 4 ? "75% allowed" : "10% allowed",
            algo == mlxvs::SearchAlgo::SingleCta ? "single-CTA" : "multi-CTA",
            recall,
            missing,
            allowed_only ? "" : ", REJECTED rows returned");
        check(ok, "filtered search");
      }
    }
    // the same filter as a bitset
    std::vector<uint32_t> words((n + 31) / 32, 0);
    for (int i = 0; i < n; ++i) {
      if (i % 4 != 0) {
        words[i / 32] |= 1u << (i % 32);
      }
    }
    mlxvs::SearchOptions o;
    o.filter = mx::array(words.data(), {(int)words.size()}, mx::uint32);
    auto got = mlxvs::graph_search(x, g, q, k, Metric::L2, o);
    mx::eval(got.ids);
    bool ok = true;
    for (size_t i = 0; i < (size_t)nq * k; ++i) {
      ok &= got.ids.data<int>()[i] % 4 != 0;
    }
    std::printf("%-6s filter as a bitset\n", ok ? "PASS" : "FAIL");
    check(ok, "bitset filter");

    // a filter per query (cuVS's bitmap): query i rejects rows r with
    // (r + i) % 3 == 0
    std::vector<uint8_t> bitmap((size_t)nq * n);
    for (int i = 0; i < nq; ++i)
      for (int r = 0; r < n; ++r)
        bitmap[(size_t)i * n + r] = (r + i) % 3 != 0;
    auto bm =
        mx::astype(mx::array(bitmap.data(), {nq, n}, mx::uint8), mx::bool_);
    for (auto algo :
         {mlxvs::SearchAlgo::SingleCta, mlxvs::SearchAlgo::MultiCta}) {
      mlxvs::SearchOptions po;
      po.algo = algo;
      po.filter = bm;
      auto pg = mlxvs::graph_search(x, g, q, k, Metric::L2, po);
      mx::eval(pg.ids);
      bool allowed = true;
      long valid = 0;
      for (int i = 0; i < nq; ++i)
        for (int j = 0; j < k; ++j) {
          int id = pg.ids.data<int>()[(size_t)i * k + j];
          allowed &= id >= 0 && (id + i) % 3 != 0;
          valid += id >= 0;
        }
      std::printf(
          "%-6s filter per query %s: %ld/%d returned, all allowed: %s\n",
          allowed ? "PASS" : "FAIL",
          algo == mlxvs::SearchAlgo::SingleCta ? "single-CTA" : "multi-CTA",
          valid,
          nq * k,
          allowed ? "yes" : "NO");
      check(allowed, "per-query filter");
    }
  }
  {
    // seed nodes: with one iteration, the exact nearest row given as a seed
    // comes back first
    int n = 20000, d = 32, nq = 300;
    auto all = low_rank(n + nq, d, 10, 23);
    mx::array x(all.data(), {n, d}, mx::float32);
    mx::array q(all.data() + (size_t)n * d, {nq, d}, mx::float32);
    auto g = knn_graph(x, 16, Metric::L2);
    auto exact = mlxvs::brute_force_knn(x, q, 1, Metric::L2);
    auto seeds = mx::concatenate({exact.ids, mx::full({nq, 3}, -1)}, 1);
    for (auto algo :
         {mlxvs::SearchAlgo::SingleCta, mlxvs::SearchAlgo::MultiCta}) {
      mlxvs::SearchOptions o;
      o.algo = algo;
      o.max_iterations = 1;
      o.seeds = seeds;
      auto got = mlxvs::graph_search(x, g, q, 5, Metric::L2, o);
      mx::eval(exact.ids, got.ids);
      int first = 0;
      for (int i = 0; i < nq; ++i) {
        first += got.ids.data<int>()[i * 5] == exact.ids.data<int>()[i];
      }
      bool ok = first == nq;
      std::printf(
          "%-6s seeds %s: %d/%d nearest first\n",
          ok ? "PASS" : "FAIL",
          algo == mlxvs::SearchAlgo::SingleCta ? "single-CTA" : "multi-CTA",
          first,
          nq);
      check(ok, "seed nodes");
    }
  }
  {
    // uint8 and int8 rows read as they are: the same results as float32
    int n = 10000, d = 40, nq = 300, k = 10;
    auto all = low_rank(n + nq, d, 8, 29);
    std::vector<float> ints(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
      ints[i] = std::round(std::clamp(20.0f * all[i], -127.0f, 127.0f));
    }
    mx::array x(ints.data(), {n, d}, mx::float32);
    mx::array q(ints.data() + (size_t)n * d, {nq, d}, mx::float32);
    auto g = knn_graph(x, 24, Metric::L2);
    auto ref = mlxvs::graph_search(x, g, q, k, Metric::L2);
    mx::eval(ref.ids);
    struct {
      const char* name;
      mx::array rows;
    } cases[] = {
        {"int8", mx::astype(x, mx::int8)},
        {"uint8", mx::astype(x + mx::array(128.0f), mx::uint8)},
    };
    for (auto& c : cases) {
      // uint8 rows are shifted by 128: so are the queries, distances unchanged
      auto qc = std::string(c.name) == "uint8" ? q + mx::array(128.0f) : q;
      auto got = mlxvs::graph_search(c.rows, g, qc, k, Metric::L2);
      mx::eval(got.ids);
      long same = 0;
      for (int i = 0; i < nq * k; ++i) {
        same += got.ids.data<int>()[i] == ref.ids.data<int>()[i];
      }
      bool ok = same >= (long)nq * k * 99 / 100;
      std::printf(
          "%-6s %s rows: %ld/%d ids as float32\n",
          ok ? "PASS" : "FAIL",
          c.name,
          same,
          nq * k);
      check(ok, "byte rows");
    }
  }
  {
    // a dimension not a multiple of 4 (the scalar tail of each distance), and
    // every team size
    int n = 5000, d = 37, nq = 200;
    auto all = low_rank(n + nq, d, 8, 13);
    std::vector<float> host(all.begin(), all.begin() + (size_t)n * d);
    std::vector<float> qhost(all.begin() + (size_t)n * d, all.end());
    mx::array x(host.data(), {n, d}, mx::float32);
    auto g = knn_graph(x, 24, Metric::L2);
    const auto sc = mlxvs::SearchAlgo::SingleCta;
    const auto mc = mlxvs::SearchAlgo::MultiCta;
    search_case(
        {"l2 d=37 team 4", Metric::L2, 10, 64, 1, 0.95, sc, 4},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"l2 d=37 team 32", Metric::L2, 10, 64, 1, 0.95, sc, 32},
        host,
        n,
        d,
        qhost,
        nq,
        g);
    search_case(
        {"mc l2 d=37 team 16", Metric::L2, 10, 64, 1, 0.95, mc, 16},
        host,
        n,
        d,
        qhost,
        nq,
        g);
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
