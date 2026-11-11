// Copyright © 2026 mikepapadim

//
// Metal sources of cagra_optimize, compiled at run time by
// mlx::core::fast::metal_kernel.
#pragma once

namespace mlxvs {
namespace kernels {

/** Threads per node of the pruning kernel. */
constexpr int PRUNE_THREADS = 256;
/** The longest input list (intermediate graph degree) cagra_optimize takes. */
constexpr int PRUNE_DEGREE_MAX = 256;

// Rank-based pruning, one threadgroup per node X with list N(X) (nearest
// first). An edge X -> Y = N(X)[j] has a detour through Z = N(X)[i] when Y =
// N(Z)[r] with max(i, r) < j: both hops rank before the direct edge. The
// edges with the fewest detours are kept, ties by rank, as CAGRA.
//
// template: K (input degree), HSIZE (slots of the id -> rank table, a power of
// two at least 2K)
// inputs: knn [n, K] int, params [n, out_degree] int
// outputs: pruned [n, out_degree] int (-1 filled)
constexpr const char* PRUNE_SOURCE = R"(
  const uint THREADS = 256;
  uint x = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  int n = params[0];
  int out_degree = params[1];

  threadgroup int nbr[K];
  threadgroup atomic_uint detours[K];
  threadgroup atomic_int slot_id[HSIZE];
  threadgroup atomic_int slot_rank[HSIZE];

  for (uint i = lid; i < HSIZE; i += THREADS) {
    atomic_store_explicit(&slot_id[i], -1, memory_order_relaxed);
    atomic_store_explicit(&slot_rank[i], 0x7fffffff, memory_order_relaxed);
  }
  for (uint j = lid; j < K; j += THREADS) {
    int v = knn[(size_t)x * K + j];
    nbr[j] = (v >= 0 && v < n && v != (int)x) ? v : -1;
    atomic_store_explicit(&detours[j], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // id -> its least rank in N(X)
  for (uint j = lid; j < K; j += THREADS) {
    int v = nbr[j];
    if (v < 0) {
      continue;
    }
    uint at = (mlxvs_mix((uint)v) >> 8) & (HSIZE - 1);
    for (;;) {
      int expected = -1;
      if (atomic_compare_exchange_weak_explicit(
              &slot_id[at], &expected, v, memory_order_relaxed, memory_order_relaxed) ||
          expected == v) {
        atomic_fetch_min_explicit(&slot_rank[at], (int)j, memory_order_relaxed);
        break;
      }
      if (expected != -1) {
        at = (at + 1) & (HSIZE - 1);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // a repeated id stays at its least rank only
  for (uint j = lid; j < K; j += THREADS) {
    int v = nbr[j];
    if (v < 0) {
      continue;
    }
    uint at = (mlxvs_mix((uint)v) >> 8) & (HSIZE - 1);
    while (atomic_load_explicit(&slot_id[at], memory_order_relaxed) != v) {
      at = (at + 1) & (HSIZE - 1);
    }
    if (atomic_load_explicit(&slot_rank[at], memory_order_relaxed) != (int)j) {
      nbr[j] = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // every (i, r): Y = N(N(X)[i])[r] is a detour of the edge to Y when its rank
  // j in N(X) exceeds both
  for (uint p = lid; p < (uint)(K * K); p += THREADS) {
    uint i = p / K;
    uint r = p % K;
    int z = nbr[i];
    if (z < 0 || r + 1 >= K) {
      continue;
    }
    int y = knn[(size_t)z * K + r];
    if (y < 0) {
      continue;
    }
    uint at = (mlxvs_mix((uint)y) >> 8) & (HSIZE - 1);
    for (;;) {
      int s = atomic_load_explicit(&slot_id[at], memory_order_relaxed);
      if (s == y) {
        uint j = (uint)atomic_load_explicit(&slot_rank[at], memory_order_relaxed);
        if (j > i && j > r) {
          atomic_fetch_add_explicit(&detours[j], 1u, memory_order_relaxed);
        }
        break;
      }
      if (s == -1) {
        break;
      }
      at = (at + 1) & (HSIZE - 1);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // each edge's place: the edges with fewer detours, or as many and a lower
  // rank, before it; missing entries last
  for (uint j = lid; j < K; j += THREADS) {
    if (nbr[j] < 0) {
      continue;
    }
    uint dj = atomic_load_explicit(&detours[j], memory_order_relaxed);
    uint place = 0;
    for (uint t = 0; t < K; ++t) {
      if (nbr[t] < 0) {
        continue;
      }
      uint dt = atomic_load_explicit(&detours[t], memory_order_relaxed);
      place += (dt < dj || (dt == dj && t < j)) ? 1u : 0u;
    }
    if (place < (uint)out_degree) {
      pruned[(size_t)x * out_degree + place] = nbr[j];
    }
  }
)";

// Reverse edges: one thread per edge X -> Y = pruned[X][r] appends (r, X) to
// Y's reverse list (the first RCAP that arrive; counted in column 0).
//
// template: D (pruned degree), RCAP
// inputs: pruned [n, D] int, params [n] int
// outputs: rev [n, 1 + 2 * RCAP] atomic int (0 filled)
constexpr const char* REVERSE_SOURCE = R"(
  uint e = thread_position_in_grid.x;
  int n = params[0];
  if (e >= (uint)n * D) {
    return;
  }
  int x = (int)(e / D);
  int r = (int)(e % D);
  int y = pruned[e];
  if (y < 0 || y >= n || y == x) {
    return;
  }
  size_t row = (size_t)y * (1 + 2 * RCAP);
  int slot = atomic_fetch_add_explicit(&rev[row], 1, memory_order_relaxed);
  if (slot < RCAP) {
    atomic_store_explicit(&rev[row + 1 + 2 * slot], r, memory_order_relaxed);
    atomic_store_explicit(&rev[row + 2 + 2 * slot], x, memory_order_relaxed);
  }
)";

// Merge, one thread per node: the first D/2 forward edges are kept, then the
// reverse edges (by rank of the edge in its source's list, then source id)
// that are not there yet, then the remaining forward edges, up to D.
//
// template: D, RCAP
// inputs: pruned [n, D] int, rev [n, 1 + 2 * RCAP] int, params [n] int
// outputs: graph [n, D] int (-1 filled)
constexpr const char* MERGE_SOURCE = R"(
  uint x = thread_position_in_grid.x;
  int n = params[0];
  if (x >= (uint)n) {
    return;
  }
  size_t frow = (size_t)x * D;
  size_t rrow = (size_t)x * (1 + 2 * RCAP);
  int out[D];
  int count = 0;
  const int protect = D / 2;
  for (int j = 0; j < protect; ++j) {
    int v = pruned[frow + j];
    if (v >= 0) {
      out[count++] = v;
    }
  }
  // reverse edges in (rank, source) order: selection of the next smallest
  int nrev = min(rev[rrow], RCAP);
  int last_r = -1;
  int last_x = -1;
  for (int taken = 0; taken < nrev && count < D; ++taken) {
    int best_r = 0x7fffffff;
    int best_x = 0x7fffffff;
    for (int s = 0; s < nrev; ++s) {
      int r = rev[rrow + 1 + 2 * s];
      int v = rev[rrow + 2 + 2 * s];
      bool after_last = r > last_r || (r == last_r && v > last_x);
      bool better = r < best_r || (r == best_r && v < best_x);
      if (after_last && better) {
        best_r = r;
        best_x = v;
      }
    }
    last_r = best_r;
    last_x = best_x;
    bool seen = false;
    for (int t = 0; t < count; ++t) {
      seen = seen || out[t] == best_x;
    }
    if (!seen) {
      out[count++] = best_x;
    }
  }
  for (int j = protect; j < D && count < D; ++j) {
    int v = pruned[frow + j];
    if (v < 0) {
      continue;
    }
    bool seen = false;
    for (int t = 0; t < count; ++t) {
      seen = seen || out[t] == v;
    }
    if (!seen) {
      out[count++] = v;
    }
  }
  for (int j = 0; j < D; ++j) {
    graph[frow + j] = j < count ? out[j] : -1;
  }
)";

} // namespace kernels
} // namespace mlxvs
