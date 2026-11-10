// Copyright © 2026 mikepapadim
// Ported from JVector's jvector-gpu NN-Descent (Apache-2.0, DataStax, Inc.);
// see NOTICE.

//
// NN-Descent on MLX: a port of JVector's TornadoVM kernels (NndKernels.java,
// host loop JitCandidates.lists). Every node keeps a list of its best K =
// max(64, k) neighbors found so far (ids, JVector scores, "new" flags), best
// first.
//
// MLX's custom kernels cannot update an input in place: every output is a fresh
// array (optionally pre-filled). So a round is a pure function of the graph,
// run as three kernels:
//
//   sample(graph ids, new flags)           -> members [n, 4S] (samples +
//   reverse samples, -1 filled), new flags' join(rows, norms, graph scores,
//   members) -> proposals [n, P] (64-bit id | score words, -1 filled)
//   merge(graph, new flags', proposals)    -> graph' (ids, scores, flags),
//   changed [n]
//
// where JVector clears its reverse-sample and proposal buffers inside join and
// merge for the next iteration, here each round's buffers start filled with -1
// (metal_kernel's init_value). The random start (4 rounds) skips sample and
// joins blocks of 4S = 64 nodes of a random permutation, passed as the members
// array. One mx::eval per round reads the number of list entries replaced, for
// the termination test.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/fast.h>

#include "mlxvs/mlxvs.h"
#include "nnd_kernels.h"
#include "utils.h"

namespace mlxvs {

namespace {

namespace k = kernels;

const mx::fast::CustomKernelFunction& sample_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_nnd_sample",
      {"graph_ids", "graph_new", "params"},
      {"members", "new_flags"},
      k::NND_SAMPLE_SOURCE,
      k::NND_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& join_kernel(bool fp16) {
  static const mx::fast::CustomKernelFunction simd = mx::fast::metal_kernel(
      "mlxvs_nnd_join_simd",
      {"rows", "norms", "params", "graph_scores", "members"},
      {"proposals"},
      k::NND_JOIN_SIMD_SOURCE,
      k::NND_HEADER);
  static const mx::fast::CustomKernelFunction simt = mx::fast::metal_kernel(
      "mlxvs_nnd_join_simt",
      {"rows", "norms", "params", "graph_scores", "members"},
      {"proposals"},
      k::NND_JOIN_SIMT_SOURCE,
      k::NND_HEADER);
  return fp16 ? simd : simt;
}

const mx::fast::CustomKernelFunction& merge_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_nnd_merge",
      {"params", "graph_ids", "graph_scores", "graph_new", "proposals"},
      {"out_ids", "out_scores", "out_new", "changed"},
      k::NND_MERGE_SOURCE,
      k::NND_HEADER);
  return kernel;
}

int metric_code(Metric metric) {
  switch (metric) {
    case Metric::InnerProduct:
      return k::M_INNER_PRODUCT;
    case Metric::Cosine:
      return k::M_COSINE;
    case Metric::L2:
      return k::M_EUCLIDEAN;
  }
  return k::M_INNER_PRODUCT;
}

/**
 * The members of the random start's joins: NND_START_PASSES random permutations
 * of the nodes, each cut into blocks of NND_JOIN (int32 [blocks, NND_JOIN], the
 * last block -1 padded).
 */
std::vector<mx::array> start_blocks(int n, unsigned long long seed) {
  std::mt19937_64 rng(seed);
  int blocks = (n + k::NND_JOIN - 1) / k::NND_JOIN;
  std::vector<int> order(n);
  std::vector<mx::array> out;
  for (int pass = 0; pass < k::NND_START_PASSES; ++pass) {
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int> padded((size_t)blocks * k::NND_JOIN, -1);
    std::copy(order.begin(), order.end(), padded.begin());
    out.emplace_back(padded.data(), mx::Shape{blocks, k::NND_JOIN}, mx::int32);
  }
  return out;
}

/**
 * Exact FP32 distances (brute_force_knn's convention) of every row of x to its
 * neighbors ids [n, k], and both re-sorted nearest first (the lists were ranked
 * by the join's possibly FP16 scores), -1 entries last.
 */
Neighbors rescore(const mx::array& x, const mx::array& ids, Metric metric) {
  int n = x.shape(0);
  int d = x.shape(1);
  int kk = ids.shape(1);
  auto norms = mx::sum(mx::square(x), 1);
  // a chunk of rows at a time: the gathered neighbor rows [c, k, d] stay at
  // about 64 MB
  int chunk = std::max(1, (1 << 24) / std::max(1, kk * d));
  std::vector<mx::array> id_parts, dist_parts;
  for (int r0 = 0; r0 < n; r0 += chunk) {
    int r1 = std::min(n, r0 + chunk);
    int c = r1 - r0;
    auto cid = mx::slice(ids, {r0, 0}, {r1, kk});
    auto valid = cid >= 0;
    auto safe = mx::maximum(cid, mx::array(0));
    auto nb = mx::reshape(mx::take(x, mx::flatten(safe), 0), {c, kk, d});
    auto q = mx::slice(x, {r0, 0}, {r1, d});
    auto dot = mx::reshape(mx::matmul(nb, mx::expand_dims(q, 2)), {c, kk});
    auto qn = mx::expand_dims(mx::slice(norms, {r0}, {r1}), 1);
    auto bn = mx::take(norms, safe, 0);
    mx::array dist = dot;
    mx::array key = mx::negative(dot);
    if (metric == Metric::L2) {
      dist = mx::maximum(qn + bn - 2.0f * dot, mx::array(0.0f));
      key = dist;
    } else if (metric == Metric::Cosine) {
      dist = dot / mx::maximum(mx::sqrt(qn * bn), mx::array(1e-30f));
      key = mx::negative(dist);
    }
    float pad = metric == Metric::L2 ? std::numeric_limits<float>::infinity()
                                     : -std::numeric_limits<float>::infinity();
    dist = mx::where(valid, dist, mx::array(pad));
    key = mx::where(
        valid, key, mx::array(std::numeric_limits<float>::infinity()));
    auto order = mx::argsort(key, 1);
    id_parts.push_back(mx::take_along_axis(cid, order, 1));
    dist_parts.push_back(mx::take_along_axis(dist, order, 1));
    mx::eval(id_parts.back(), dist_parts.back());
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], dist_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(dist_parts, 0)};
}

} // namespace

Neighbors nn_descent(
    const mx::array& rows_in,
    int kq,
    Metric metric,
    const NnDescentOptions& options,
    mx::StreamOrDevice s) {
  mx::StreamContext context(detail::gpu_stream(s, "mlxvs::nn_descent"));
  auto rows = detail::float_rows(rows_in, "mlxvs::nn_descent");
  int n = rows.shape(0);
  int d = rows.shape(1);
  int list = std::max(64, kq);
  if (kq < 1 || list > k::NND_LIST / 2 || n < 2 || d < 1) {
    throw std::invalid_argument(
        "[mlxvs::nn_descent] 1 <= k <= " + std::to_string(k::NND_LIST / 2) +
        ", n >= 2 and d >= 1");
  }
  if ((long long)n * k::NND_GROUP > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "[mlxvs::nn_descent] n too large for one dispatch");
  }
  // proposal slots: list entries plus slots fill the merge's sort of NND_LIST
  int slots = k::NND_LIST - list;
  bool fp16 = options.fp16;
  // the join's rows (FP16 for the SIMD-group join) and their squared norms (of
  // the FP16 values, as JVector)
  mx::array data = fp16 ? mx::astype(rows, mx::float16) : rows;
  mx::array norms = mx::sum(mx::square(mx::astype(data, mx::float32)), 1);
  mx::eval(data, norms);
  auto blocks = start_blocks(n, options.seed);

  mx::array graph_ids = mx::full({n, list}, -1, mx::int32);
  mx::array graph_scores = mx::full({n, list}, k::NND_NONE, mx::float32);
  mx::array graph_new = mx::zeros({n, list}, mx::int32);
  const auto& join = join_kernel(fp16);
  int code = metric_code(metric);
  int seed_salt = (int)(uint32_t)(options.seed * 0x9E3779B97F4A7C15ull >> 32);
  int rounds = k::NND_START_PASSES + std::max(0, options.max_iterations);
  for (int round = 0; round < rounds; ++round) {
    bool start = round < k::NND_START_PASSES;
    int salt = (round * 7919 + 17) ^ seed_salt;
    auto params = mx::array(
        {n, d, list, slots, code, round, salt}, {k::P_COUNT}, mx::int32);
    // sample: the join's members and the flags with the sampled new neighbors
    // cleared (the start joins blocks instead)
    mx::array members = blocks[std::min(round, k::NND_START_PASSES - 1)];
    mx::array flags = graph_new;
    if (!start) {
      int grid = (n + k::NND_SMALL_GROUP - 1) / k::NND_SMALL_GROUP *
          k::NND_SMALL_GROUP;
      auto sampled = sample_kernel()(
          {graph_ids, graph_new, params},
          {{n, k::NND_JOIN}, {n, list}},
          {mx::int32, mx::int32},
          {grid, 1, 1},
          {k::NND_SMALL_GROUP, 1, 1},
          {},
          -1.0f,
          false,
          {});
      members = sampled[0];
      flags = sampled[1];
    }
    // join: one threadgroup per row of members
    int groups = members.shape(0);
    auto proposals = join(
        {data, norms, params, graph_scores, members},
        {{n, slots}},
        {mx::int64},
        {groups * k::NND_GROUP, 1, 1},
        {k::NND_GROUP, 1, 1},
        {},
        -1.0f,
        false,
        {})[0];
    // merge: one threadgroup per node
    auto m = merge_kernel()(
        {params, graph_ids, graph_scores, flags, proposals},
        {{n, list}, {n, list}, {n, list}, {n}},
        {mx::int32, mx::float32, mx::int32, mx::int32},
        {n * k::NND_GROUP, 1, 1},
        {k::NND_GROUP, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    graph_ids = m[0];
    graph_scores = m[1];
    graph_new = m[2];
    auto changed = mx::sum(mx::astype(m[3], mx::int64));
    mx::eval(graph_ids, graph_scores, graph_new, changed);
    long long replaced = changed.item<int64_t>();
    if (options.updates != nullptr) {
      options.updates->push_back(replaced);
    }
    if (!start && (double)replaced < options.termination_threshold * n * list) {
      break;
    }
  }
  return rescore(rows, mx::slice(graph_ids, {0, 0}, {n, kq}), metric);
}

} // namespace mlxvs
