// Copyright © 2026 mikepapadim

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mlx/fast.h>

#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& search_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_graph_search",
      {"base", "graph", "queries", "inv_norm", "filter", "seeds", "params"},
      {"out_ids", "out_scores"},
      kernels::SEARCH_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& search_mc_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_graph_search_mc",
      {"base", "graph", "queries", "inv_norm", "filter", "seeds", "params"},
      {"out_ids", "out_scores", "traversed"},
      kernels::SEARCH_MC_SOURCE,
      kernels::SEARCH_HEADER,
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/true);
  return kernel;
}

int pow2_at_least(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

// queries per kernel launch, so that one launch stays well within the GPU's
// command time limits
constexpr int SEARCH_QUERY_BLOCK = 65536;

// The GPU's core count (IOKit's gpu-core-count), the counterpart of cuVS's
// number of SMs; 8 if it cannot be read.
int gpu_cores() {
  static const int cores = [] {
    int found = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(
            kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) ==
        KERN_SUCCESS) {
      for (io_object_t service = IOIteratorNext(it); service != 0 && !found;
           service = IOIteratorNext(it)) {
        auto value = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value != nullptr) {
          if (CFGetTypeID(value) == CFNumberGetTypeID()) {
            CFNumberGetValue(value, kCFNumberIntType, &found);
          }
          CFRelease(value);
        }
        IOObjectRelease(service);
      }
      IOObjectRelease(it);
    }
    return found > 0 ? found : 8;
  }();
  return cores;
}

// One more hop for every power of max(2, degree / 2) below n: cuVS's estimate
// of the hops needed to reach every node.
int reach_hops(int n, int deg) {
  int hops = 0;
  long long reach = 1;
  while (reach < n) {
    reach *= std::max(2, deg / 2);
    ++hops;
  }
  return hops;
}

struct Prepared {
  mx::array base;
  mx::array graph;
  mx::array queries;
  mx::array inv_norm;
  int metric_code;
  int team;
  mx::array filter; // bitset words, or one word when there is no filter
  bool filtered;
  mx::array seeds; // [nq, num_seeds] int32, or one dummy entry
  int num_seeds;
};

// Lanes per distance, as cuVS picks its team size: the instance whose block
// dimension (128, 256, 512 for teams of 8, 16, 32) is nearest d.
int team_for(int d, int requested) {
  if (requested != 0) {
    return requested;
  }
  const int dims[] = {128, 256, 512};
  const int teams[] = {8, 16, 32};
  int best = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(d - dims[i]) < std::abs(d - dims[best])) {
      best = i;
    }
  }
  return teams[best];
}

// Scores (larger is closer) back to the metric's distances: squared distances
// for L2 (+inf where missing), similarities otherwise (-inf where missing).
mx::array to_distances(const mx::array& scores, int metric_code) {
  if (metric_code != 1) {
    return scores;
  }
  return mx::where(
      mx::isinf(scores),
      mx::array(INFINITY, mx::float32),
      mx::negative(scores));
}

Neighbors search_multi_cta(
    const Prepared& in,
    int k,
    int itopk,
    const SearchOptions& options) {
  int n = in.base.shape(0);
  int d = in.base.shape(1);
  int deg = in.graph.shape(1);
  int nq = in.queries.shape(0);
  int width = options.search_width;
  constexpr int list = kernels::SEARCH_MC_ITOPK;
  int buffer = (list + deg + 31) / 32 * 32;
  if (buffer > kernels::SEARCH_MC_BUFFER_MAX) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] multi-CTA search needs graph degree <= " +
        std::to_string(kernels::SEARCH_MC_BUFFER_MAX - list) + ".");
  }
  // as cuVS's search plan
  int num_cta = std::max(width, (itopk + list - 1) / list);
  int iterations = options.max_iterations;
  if (iterations <= 0) {
    const int minimum_depth = 16;
    iterations = minimum_depth +
        (list - minimum_depth + num_cta - 1) / num_cta + (k + list - 1) / list +
        reach_hops(n, deg);
  }
  iterations = std::max(iterations, options.min_iterations);
  // the visited set holds the buffer and one parent's neighbors at most half
  // full; the traversed set every parent of every threadgroup, as cuVS sizes it
  int vbits = 8;
  while ((1 << vbits) / 2 < list + 2 * deg) {
    ++vbits;
  }
  long long traversed_nodes = (long long)num_cta * std::max(list, iterations);
  int tbits = 11;
  while ((1LL << tbits) / 2 < traversed_nodes) {
    ++tbits;
  }
  if (tbits > 25) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] multi-CTA traversed set over 2^25 entries: "
        "lower itopk_size, search_width or max_iterations.");
  }
  bool qshared = d <= kernels::SEARCH_QUERY_SHARED_MAX;
  int qcap = qshared ? (d + 31) / 32 * 32 : 1;
  int bp = pow2_at_least(buffer);
  if (qshared && 4 * qcap + (4 << vbits) + 8 * bp + 64 > 16384) {
    qshared = false;
    qcap = 1;
  }
  // query blocks whose traversed sets take at most 256 MB
  int block = (int)std::max<long long>(
      1, std::min<long long>(SEARCH_QUERY_BLOCK, (64LL << 20) >> tbits));
  int cols = num_cta * list;

  std::vector<mx::array> id_parts, dist_parts;
  for (int q0 = 0; q0 < nq; q0 += block) {
    int q1 = std::min(nq, q0 + block);
    int bq = q1 - q0;
    auto qs = mx::slice(in.queries, {q0, 0}, {q1, d});
    auto qseeds = in.num_seeds > 0
        ? mx::slice(in.seeds, {q0, 0}, {q1, in.num_seeds})
        : in.seeds;
    auto params = mx::array(
        {n,
         d,
         deg,
         iterations,
         options.min_iterations,
         in.metric_code,
         (int)(options.seed & 0x7fffffffULL) ^ q0,
         std::max(1, options.num_random_samplings),
         num_cta,
         bq,
         tbits},
        {11},
        mx::int32);
    auto out = search_mc_kernel()(
        {in.base, in.graph, qs, in.inv_norm, in.filter, qseeds, params},
        {{bq, cols}, {bq, cols}, {bq, 1 << tbits}},
        {mx::int32, mx::float32, mx::int32},
        {num_cta * kernels::SEARCH_MC_THREADS, bq, 1},
        {kernels::SEARCH_MC_THREADS, 1, 1},
        {{"B", buffer},
         {"BP", bp},
         {"QCAP", qcap},
         {"QSHARED", qshared},
         {"VBITS", vbits},
         {"TEAM", in.team},
         {"FILTER", in.filtered},
         {"NSEEDS", in.num_seeds}},
        -1.0f,
        false,
        {});
    // the best k of the threadgroups' lists (each node comes from one)
    auto order =
        mx::slice(mx::argsort(mx::negative(out[1]), 1), {0, 0}, {bq, k});
    auto ids = mx::take_along_axis(out[0], order, 1);
    auto scores = mx::take_along_axis(out[1], order, 1);
    auto dist = to_distances(scores, in.metric_code);
    if (nq > block) {
      mx::eval(ids, dist);
    }
    id_parts.push_back(ids);
    dist_parts.push_back(dist);
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], dist_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(dist_parts, 0)};
}

Neighbors search_single_cta(
    const Prepared& in,
    int k,
    int itopk,
    const SearchOptions& options) {
  int n = in.base.shape(0);
  int d = in.base.shape(1);
  int deg = in.graph.shape(1);
  int nq = in.queries.shape(0);
  int width = options.search_width;
  int m = (itopk + 31) / 32 * 32;
  int cp = pow2_at_least(width * deg);
  // max_iterations as cuVS's search plan: itopk_size / search_width, plus one
  // per hop needed to reach n nodes at degree / 2 new nodes per hop. Its first
  // iteration is the random start, so the kernel's rounds of expansion are
  // one fewer.
  int iterations = options.max_iterations;
  if (iterations <= 0) {
    iterations = itopk / width + reach_hops(n, deg);
  }
  iterations = std::max(iterations, options.min_iterations);
  int max_iter = std::max(1, iterations - 1);
  int n_rand = std::max(1, options.num_random_samplings);
  // the random start fills the list and one round's candidates, as cuVS
  int rand_slots = m + width * deg;
  // threadgroup memory: the list (two halves), the candidates, the query and
  // the visited set. Kept within 16 KB where it can be: Metal's limit is 32 KB,
  // but its shader validation (as run in CI) counts every byte twice. The
  // visited set gets what is left, at least twice the list and one round.
  bool qshared = d <= kernels::SEARCH_QUERY_SHARED_MAX;
  int qcap = qshared ? (d + 31) / 32 * 32 : 1;
  auto fixed = [&]() { return 16 * m + 8 * cp + 4 * qcap + 256; };
  int hmin = 9;
  while ((1 << hmin) < 2 * (m + cp)) {
    ++hmin;
  }
  if (qshared && fixed() + (4 << hmin) > 16384) {
    qshared = false;
    qcap = 1;
  }
  int hbits = hmin;
  while (hbits < 12 && fixed() + (4 << (hbits + 1)) <= 16384) {
    ++hbits;
  }

  std::vector<mx::array> id_parts, dist_parts;
  for (int q0 = 0; q0 < nq; q0 += SEARCH_QUERY_BLOCK) {
    int q1 = std::min(nq, q0 + SEARCH_QUERY_BLOCK);
    int bq = q1 - q0;
    auto qs = mx::slice(in.queries, {q0, 0}, {q1, d});
    auto qseeds = in.num_seeds > 0
        ? mx::slice(in.seeds, {q0, 0}, {q1, in.num_seeds})
        : in.seeds;
    auto params = mx::array(
        {n,
         d,
         deg,
         k,
         width,
         max_iter,
         in.metric_code,
         (int)(options.seed & 0x7fffffffULL) ^ q0,
         n_rand,
         rand_slots},
        {10},
        mx::int32);
    auto out = search_kernel()(
        {in.base, in.graph, qs, in.inv_norm, in.filter, qseeds, params},
        {{bq, k}, {bq, k}},
        {mx::int32, mx::float32},
        {bq * kernels::SEARCH_THREADS, 1, 1},
        {kernels::SEARCH_THREADS, 1, 1},
        {{"M", m},
         {"CP", cp},
         {"QCAP", qcap},
         {"QSHARED", qshared},
         {"HBITS", hbits},
         {"TEAM", in.team},
         {"FILTER", in.filtered},
         {"NSEEDS", in.num_seeds}},
        std::nullopt,
        false,
        {});
    if (nq > SEARCH_QUERY_BLOCK) {
      mx::eval(out[0], out[1]);
    }
    id_parts.push_back(out[0]);
    dist_parts.push_back(out[1]);
  }
  if (id_parts.size() == 1) {
    return {id_parts[0], dist_parts[0]};
  }
  return {mx::concatenate(id_parts, 0), mx::concatenate(dist_parts, 0)};
}

} // namespace

Neighbors graph_search(
    const mx::array& base_in,
    const mx::array& graph_in,
    const mx::array& queries_in,
    int k,
    Metric metric,
    const SearchOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::graph_search";
  mx::StreamContext context(detail::gpu_stream(s, op));
  // float16 rows are read as they are (half the memory traffic); other dtypes
  // are converted to float32
  auto base = base_in.ndim() == 2 && base_in.dtype() == mx::float16
      ? base_in
      : detail::float_rows(base_in, op, "base");
  auto queries = detail::float_rows(queries_in, op, "queries");
  if (graph_in.ndim() != 2 || graph_in.shape(0) != base.shape(0)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] graph must be [n, degree], one row per base "
        "row.");
  }
  if (!mx::issubdtype(graph_in.dtype(), mx::integer)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] graph must hold integer ids.");
  }
  if (base.shape(1) != queries.shape(1)) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] base and queries must have the same number of "
        "columns.");
  }
  int n = base.shape(0);
  int deg = graph_in.shape(1);
  int nq = queries.shape(0);
  int width = options.search_width;
  if (n < 1 || deg < 1) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] base and graph must not be empty.");
  }
  if (width < 1 || width > kernels::SEARCH_WIDTH_MAX ||
      (long long)width * deg > kernels::SEARCH_CANDIDATES_MAX) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] search_width must be in [1, 32] and "
        "search_width * graph degree at most " +
        std::to_string(kernels::SEARCH_CANDIDATES_MAX) + ".");
  }
  int itopk = std::max(options.itopk_size, k);
  int m = (itopk + 31) / 32 * 32;
  // as cuVS's search plan: one threadgroup per query when there are at least
  // two queries per GPU core, multi-CTA otherwise
  bool multi = options.algo == SearchAlgo::MultiCta ||
      (options.algo == SearchAlgo::Auto &&
       (m > kernels::SEARCH_ITOPK_MAX || nq < 2 * gpu_cores()) &&
       (kernels::SEARCH_MC_ITOPK + deg + 31) / 32 * 32 <=
           kernels::SEARCH_MC_BUFFER_MAX);
  int itopk_max = multi ? 1024 : kernels::SEARCH_ITOPK_MAX;
  if (k < 1 || m > itopk_max) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] k and itopk_size must be in [1, " +
        std::to_string(itopk_max) + "].");
  }

  auto graph = graph_in.dtype() == mx::int32 ? graph_in
                                             : mx::astype(graph_in, mx::int32);
  auto inv_norm = mx::array({1.0f});
  if (metric == Metric::Cosine) {
    auto qn = mx::sqrt(mx::sum(mx::square(queries), 1, /*keepdims=*/true));
    queries = queries / mx::maximum(qn, mx::array(1e-30f));
    inv_norm = mx::rsqrt(
        mx::maximum(
            mx::sum(mx::square(mx::astype(base, mx::float32)), 1),
            mx::array(1e-30f)));
  }
  int metric_code = metric == Metric::InnerProduct ? 0
      : metric == Metric::L2                       ? 1
                                                   : 2;
  if (nq == 0) {
    return {mx::zeros({0, k}, mx::int32), mx::zeros({0, k}, mx::float32)};
  }
  if (options.team_size != 0 && options.team_size != 4 &&
      options.team_size != 8 && options.team_size != 16 &&
      options.team_size != 32) {
    throw std::invalid_argument(
        "[mlxvs::graph_search] team_size must be 0, 4, 8, 16 or 32.");
  }
  // the filter as bitset words
  auto filter = mx::array({0xffffffffu});
  bool filtered = options.filter.has_value();
  float rate = options.filtering_rate;
  if (filtered) {
    const auto& f = *options.filter;
    int words = (n + 31) / 32;
    if (f.ndim() == 1 && f.dtype() == mx::bool_ && f.shape(0) == n) {
      auto padded = mx::pad(mx::astype(f, mx::uint32), {{0, words * 32 - n}});
      auto bits = mx::left_shift(mx::array(1u), mx::arange(32, mx::uint32));
      filter = mx::sum(mx::multiply(mx::reshape(padded, {words, 32}), bits), 1);
      filter = mx::astype(filter, mx::uint32);
      if (rate < 0) {
        rate = 1.0f - mx::mean(mx::astype(f, mx::float32)).item<float>();
      }
    } else if (
        f.ndim() == 1 && f.shape(0) == words &&
        (f.dtype() == mx::uint32 || f.dtype() == mx::int32)) {
      filter = mx::view(f, mx::uint32);
      if (rate < 0) {
        // popcount of the words, the padding bits not counted
        auto bits = mx::bitwise_and(
            mx::right_shift(
                mx::expand_dims(filter, 1), mx::arange(32, mx::uint32)),
            mx::array(1u));
        auto allowed = mx::sum(
            mx::astype(mx::slice(mx::flatten(bits), {0}, {n}), mx::float32));
        rate = 1.0f - allowed.item<float>() / n;
      }
    } else {
      throw std::invalid_argument(
          "[mlxvs::graph_search] filter must be bool [n] or a bitset uint32 "
          "[ceil(n / 32)].");
    }
  }
  if (multi && filtered && rate > 0 && rate < 1) {
    // as cuVS's search plan: a longer list when the filter rejects rows
    int adjusted = (int)((float)k / (1.0f - rate) +
                         (float)(itopk - k) / std::sqrt(1.0f - rate));
    adjusted = (adjusted + 31) / 32 * 32;
    itopk = std::min(std::max(itopk, adjusted), 1024);
  }
  auto seeds = mx::array({-1});
  int num_seeds = 0;
  if (options.seeds.has_value()) {
    const auto& sd = *options.seeds;
    if (sd.ndim() != 2 || sd.shape(0) != nq ||
        !mx::issubdtype(sd.dtype(), mx::integer)) {
      throw std::invalid_argument(
          "[mlxvs::graph_search] seeds must be integer ids [nq, num_seeds].");
    }
    num_seeds = sd.shape(1);
    seeds = num_seeds > 0 ? mx::astype(sd, mx::int32) : seeds;
  }
  Prepared in{
      base,
      graph,
      queries,
      inv_norm,
      metric_code,
      team_for(base.shape(1), options.team_size),
      filter,
      filtered,
      seeds,
      num_seeds};
  return multi ? search_multi_cta(in, k, itopk, options)
               : search_single_cta(in, k, itopk, options);
}

} // namespace mlxvs
