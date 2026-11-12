// Copyright © 2026 mikepapadim

#include "graph.h"
#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/fast.h>

#include "cagra_kernels.h"
#include "search_kernels.h"

namespace mlxvs {

namespace {

const mx::fast::CustomKernelFunction& prune_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_prune",
      {"knn", "params"},
      {"pruned"},
      kernels::PRUNE_SOURCE,
      kernels::SEARCH_HEADER);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_count_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_count",
      {"pruned", "params"},
      {"hist"},
      kernels::REV_COUNT_SOURCE,
      "",
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/true);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_offset_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_offset",
      {"hist", "params"},
      {"offset"},
      kernels::REV_OFFSET_SOURCE);
  return kernel;
}

const mx::fast::CustomKernelFunction& rev_fill_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_fill",
      {"pruned", "offset", "params"},
      {"placed", "rev"},
      kernels::REV_FILL_SOURCE,
      "",
      /*ensure_row_contiguous=*/true,
      /*atomic_outputs=*/true);
  return kernel;
}

const mx::fast::CustomKernelFunction& merge_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_merge",
      {"pruned", "rev", "offset", "params"},
      {"graph"},
      kernels::MERGE_SOURCE);
  return kernel;
}

int pow2_at_least(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

} // namespace

namespace {

// Reverse edges of graph [n, D] (-1 entries skipped): each node's rcap
// incoming edges of lowest rank, as cuVS adds them one rank at a time.
// Returns (rev [n, 2 * rcap] (rank, source) pairs, offset [n, D + 1]: incoming
// edges of lower rank, the node's in-degree last).
std::pair<mx::array, mx::array> reverse_edges(
    const mx::array& graph,
    int rcap) {
  int n = graph.shape(0);
  int d = graph.shape(1);
  auto params = mx::array({n}, {1}, mx::int32);
  auto hist = rev_count_kernel()(
      {graph, params},
      {{n, d}},
      {mx::int32},
      {n * d, 1, 1},
      {256, 1, 1},
      {{"D", d}},
      0,
      false,
      {})[0];
  auto offset = rev_offset_kernel()(
      {hist, params},
      {{n, d + 1}},
      {mx::int32},
      {n, 1, 1},
      {256, 1, 1},
      {{"D", d}},
      std::nullopt,
      false,
      {})[0];
  auto rev = rev_fill_kernel()(
      {graph, offset, params},
      {{n, d}, {n, 2 * rcap}},
      {mx::int32, mx::int32},
      {n * d, 1, 1},
      {256, 1, 1},
      {{"D", d}, {"RCAP", rcap}},
      0,
      false,
      {})[1];
  return {rev, offset};
}

const mx::fast::CustomKernelFunction& rev_list_kernel() {
  static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
      "mlxvs_cagra_rev_list",
      {"rev", "offset", "params"},
      {"lists"},
      kernels::REV_LIST_SOURCE);
  return kernel;
}

} // namespace

namespace detail {

mx::array reverse_lists(const mx::array& graph, int rcap) {
  int n = graph.shape(0);
  int d = graph.shape(1);
  auto [rev, offset] = reverse_edges(graph, rcap);
  return rev_list_kernel()(
      {rev, offset, mx::array({n}, {1}, mx::int32)},
      {{n, rcap}},
      {mx::int32},
      {n, 1, 1},
      {256, 1, 1},
      {{"D", d}, {"RCAP", rcap}},
      std::nullopt,
      false,
      {})[0];
}

} // namespace detail

mx::array cagra_optimize(
    const mx::array& knn_in,
    int graph_degree,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::cagra_optimize";
  mx::StreamContext context(detail::gpu_stream(s, op));
  if (knn_in.ndim() != 2 || !mx::issubdtype(knn_in.dtype(), mx::integer)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] knn_graph must be integer ids [n, degree].");
  }
  int n = knn_in.shape(0);
  int k = knn_in.shape(1);
  if (k < 1 || k > kernels::PRUNE_DEGREE_MAX) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] knn_graph degree must be in [1, " +
        std::to_string(kernels::PRUNE_DEGREE_MAX) + "].");
  }
  if (graph_degree < 1 || graph_degree > k) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] graph_degree must be in [1, knn_graph "
        "degree].");
  }
  if ((long long)n * graph_degree >= (1LL << 31)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_optimize] n * graph_degree must be below 2^31.");
  }
  if (n == 0) {
    return mx::zeros({0, graph_degree}, mx::int32);
  }
  auto knn =
      knn_in.dtype() == mx::int32 ? knn_in : mx::astype(knn_in, mx::int32);
  int d = graph_degree;
  int rcap = d;

  auto pruned = prune_kernel()(
      {knn, mx::array({n, d}, {2}, mx::int32)},
      {{n, d}},
      {mx::int32},
      {n * kernels::PRUNE_THREADS, 1, 1},
      {kernels::PRUNE_THREADS, 1, 1},
      {{"K", k}, {"HSIZE", pow2_at_least(2 * k)}},
      -1,
      false,
      {})[0];

  auto params = mx::array({n}, {1}, mx::int32);
  auto [rev, offset] = reverse_edges(pruned, rcap);

  return merge_kernel()(
      {pruned, rev, offset, params},
      {{n, d}},
      {mx::int32},
      {n, 1, 1},
      {256, 1, 1},
      {{"D", d}, {"RCAP", rcap}},
      -1,
      false,
      {})[0];
}

namespace {

void check_cagra_options(const CagraOptions& o) {
  if (o.graph_degree < 1 || o.graph_degree > o.intermediate_graph_degree) {
    throw std::invalid_argument(
        "[mlxvs::cagra_build] graph_degree must be in [1, "
        "intermediate_graph_degree].");
  }
}

} // namespace

mx::array cagra_build(
    const mx::array& rows,
    Metric metric,
    const CagraOptions& options,
    mx::StreamOrDevice s) {
  check_cagra_options(options);
  auto knn = all_neighbors(
      rows, options.intermediate_graph_degree, metric, options.knn, s);
  return cagra_optimize(knn.ids, options.graph_degree, s);
}

mx::array cagra_build(
    const HostRows& rows,
    Metric metric,
    const CagraOptions& options,
    mx::StreamOrDevice s) {
  check_cagra_options(options);
  auto knn = all_neighbors(
      rows, options.intermediate_graph_degree, metric, options.knn, s);
  return cagra_optimize(knn.ids, options.graph_degree, s);
}

mx::array cagra_extend(
    const mx::array& dataset,
    const mx::array& graph_in,
    const mx::array& new_rows,
    Metric metric,
    const ExtendOptions& options,
    mx::StreamOrDevice s) {
  if (dataset.ndim() != 2 || new_rows.ndim() != 2 || graph_in.ndim() != 2 ||
      graph_in.shape(0) != dataset.shape(0) ||
      new_rows.shape(1) != dataset.shape(1) ||
      !mx::issubdtype(graph_in.dtype(), mx::integer)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_extend] dataset [n, d], graph [n, degree] and new_rows "
        "[m, d] must match.");
  }
  const long long old_size = dataset.shape(0);
  const long long num_add = new_rows.shape(0);
  const int degree = graph_in.shape(1);
  const long long new_size = old_size + num_add;
  if (new_size >= (1LL << 31)) {
    throw std::invalid_argument(
        "[mlxvs::cagra_extend] the extended graph needs fewer than 2^31 rows.");
  }
  if (num_add == 0) {
    return mx::astype(graph_in, mx::int32);
  }
  const int base_degree = 2 * degree;
  if (old_size < base_degree) {
    throw std::invalid_argument(
        "[mlxvs::cagra_extend] the index needs at least 2 * graph degree "
        "rows to extend.");
  }

  // Step 1, as cuVS's add_node_core: the 2 * degree nearest rows of the index
  // to each new row, by a graph search
  SearchOptions search = options.search;
  search.itopk_size = std::max(search.itopk_size, std::max(4 * degree, 256));
  auto found =
      graph_search(dataset, graph_in, new_rows, base_degree, metric, search, s)
          .ids;
  auto graph_old = mx::contiguous(mx::astype(graph_in, mx::int32));
  mx::eval(found, graph_old);

  std::vector<int32_t> g((size_t)new_size * degree);
  std::memcpy(
      g.data(),
      graph_old.data<int32_t>(),
      (size_t)old_size * degree * sizeof(int32_t));
  const int32_t* cand = found.data<int32_t>();

  // incoming edges of the index's nodes (not updated as edges are replaced,
  // as cuVS)
  std::vector<int> incoming(new_size, 0);
  for (size_t e = 0; e < (size_t)old_size * degree; ++e) {
    int v = g[e];
    if (v >= 0 && v < old_size) {
      ++incoming[v];
    }
  }

  // Step 2: rank-based reordering of each new row's candidates, by the number
  // of detours through an earlier candidate (invalid ids last)
  std::vector<std::pair<int32_t, int>> counts(base_degree);
  for (long long r = 0; r < num_add; ++r) {
    const int32_t* c = cand + (size_t)r * base_degree;
    for (int i = 0; i < base_degree; ++i) {
      int32_t a = c[i];
      if (a < 0 || a >= old_size) {
        counts[i] = {a, base_degree + 1};
        continue;
      }
      int detours = 0;
      for (int j = 0; j < i; ++j) {
        int32_t b = c[j];
        if (b < 0 || b >= old_size) {
          continue;
        }
        const int32_t* nb = &g[(size_t)b * degree];
        detours += std::find(nb, nb + degree, a) != nb + degree ? 1 : 0;
      }
      counts[i] = {a, detours};
    }
    std::stable_sort(
        counts.begin(), counts.end(), [](const auto& x, const auto& y) {
          return x.second < y.second;
        });
    for (int i = 0; i < degree; ++i) {
      g[(size_t)(old_size + r) * degree + i] = counts[i].first;
    }
  }

  // Step 3: reverse edges. Each of a new node's first degree / 2 neighbors
  // gives it the slot (in its second half) of the neighbor with the most
  // incoming edges; the new node's list interleaves its own list with the
  // displaced neighbors.
  const int num_rev = degree / 2;
  std::vector<int32_t> rev(num_rev), merged(degree);
  for (long long r = 0; r < num_add; ++r) {
    const int32_t node = (int32_t)(old_size + r);
    int32_t* own = &g[(size_t)node * degree];
    int nrev = 0;
    for (int i = 0; i < num_rev; ++i) {
      int32_t target = own[i];
      if (target < 0 || target >= new_size) {
        continue;
      }
      int32_t* tlist = &g[(size_t)target * degree];
      int32_t replace_id = -1;
      int replace_j = -1;
      int replace_in = 0;
      for (int j = degree - 1; j >= degree / 2; --j) {
        int32_t nbr = tlist[j];
        if (nbr < 0 || nbr >= new_size) {
          // an empty slot: take it
          replace_id = -1;
          replace_j = j;
          break;
        }
        if (incoming[nbr] > replace_in &&
            std::find(rev.begin(), rev.begin() + nrev, nbr) ==
                rev.begin() + nrev) {
          replace_in = incoming[nbr];
          replace_id = nbr;
          replace_j = j;
        }
      }
      if (replace_j < 0) {
        continue;
      }
      tlist[replace_j] = node;
      if (replace_id >= 0) {
        rev[nrev++] = replace_id;
      }
    }
    incoming[node] = num_rev;

    int num = 0, ri = 0, vi = 0, turn = 0;
    while (num < degree && (ri < degree || vi < nrev)) {
      const int32_t* list = turn == 0 ? own : rev.data();
      int& at = turn == 0 ? ri : vi;
      int end = turn == 0 ? degree : nrev;
      for (; at < end; ++at) {
        int32_t v = list[at];
        if (v < 0 || v >= new_size || v == node ||
            std::find(merged.begin(), merged.begin() + num, v) !=
                merged.begin() + num) {
          continue;
        }
        merged[num++] = v;
        ++at;
        break;
      }
      turn = 1 - turn;
    }
    for (int i = 0; i < degree; ++i) {
      own[i] = i < num ? merged[i] : -1;
    }
  }
  return mx::array(g.data(), {(int)new_size, degree}, mx::int32);
}

} // namespace mlxvs
