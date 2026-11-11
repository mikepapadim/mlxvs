// Copyright © 2026 mikepapadim

//
// Metal source of graph_search, compiled at run time by
// mlx::core::fast::metal_kernel.
#pragma once

namespace mlxvs::kernels {

/** Threads per query in graph_search. */
constexpr int SEARCH_THREADS = 256;
/** The longest internal list (itopk_size) graph_search keeps. */
constexpr int SEARCH_ITOPK_MAX = 256;
/** The most candidates of one round (search_width * graph degree). */
constexpr int SEARCH_CANDIDATES_MAX = 512;
/** The most nodes expanded per round. */
constexpr int SEARCH_WIDTH_MAX = 32;
/** Query dimensions kept in threadgroup memory; larger queries are read from
 * device memory. */
constexpr int SEARCH_QUERY_SHARED_MAX = 1024;

constexpr const char* SEARCH_HEADER = R"(
// A list entry's id: a node id (>= 0), -1 (empty), or -(id + 2) once the node
// has been expanded.
inline int mlxvs_node(int v) {
  return v >= -1 ? v : -v - 2;
}
// "a comes before b": higher score first, then lower id; empty entries
// (id -1, score -inf) last.
inline bool mlxvs_before(float sa, int ia, float sb, int ib) {
  return sa > sb || (sa == sb && (uint)ia < (uint)ib);
}
inline uint mlxvs_mix(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Inserts id into the visited set (open addressing, linear probing, empty
// slots -1). True if it was not there.
inline bool mlxvs_visit(threadgroup atomic_int* table, uint bits, int id) {
  uint mask = (1u << bits) - 1u;
  uint at = (mlxvs_mix((uint)id) >> (32u - bits)) & mask;
  for (uint probes = 0; probes <= mask;) {
    int expected = -1;
    if (atomic_compare_exchange_weak_explicit(
            &table[at], &expected, id, memory_order_relaxed, memory_order_relaxed)) {
      return true;
    }
    if (expected == id) {
      return false;
    }
    if (expected != -1) {
      at = (at + 1u) & mask;
      ++probes;
    }
    // expected == -1: a spurious failure of the weak exchange, try again
  }
  return false;
}
)";

// One threadgroup of 256 threads per query, as CAGRA's single-CTA search. The
// query keeps a sorted list of M entries (itopk). After a random start, each
// round expands the best unexpanded entries (up to W of them, marked in the
// list), gathers their graph neighbors, drops the ones already visited (a hash
// set in threadgroup memory, cleared and refilled with the list's nodes when
// half full), scores the rest (one SIMD group per candidate), sorts them
// (bitonic) and merges them into the list (each entry's rank by binary search
// in the other list). It stops when every entry of the list has been expanded,
// or after max_iter rounds.
//
// template: M (list length, multiple of 32), CP (candidate slots, power of
// two), QCAP (query floats in threadgroup memory, 1 when read from device),
// QSHARED, HBITS (log2 of the visited-set slots)
// inputs: base [n, d] float, graph [n, deg] int, queries [nq, d] float,
// inv_norm [n] float (cosine only), params [n, d, deg, k, W, max_iter, metric,
// seed, n_rand] int
// outputs: out_ids [nq, k] int, out_scores [nq, k] float
constexpr const char* SEARCH_SOURCE = R"(
  const uint THREADS = 256;
  const uint SIMDS = THREADS / 32;
  const uint HSIZE = 1u << HBITS;
  uint q = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid / 32;
  uint lane = lid % 32;
  int n = params[0];
  int d = params[1];
  int deg = params[2];
  int k = params[3];
  int width = params[4];
  int max_iter = params[5];
  int metric = params[6];  // 0 inner product, 1 l2, 2 cosine
  uint seed = (uint)params[7];
  int n_rand = params[8];
  size_t qrow = (size_t)q * (size_t)d;

  threadgroup float qv[QCAP];
  threadgroup atomic_int visited[HSIZE];
  threadgroup atomic_uint filled;
  threadgroup float top_s[2 * M];
  threadgroup int top_i[2 * M];
  threadgroup float cand_s[CP];
  threadgroup int cand_i[CP];
  threadgroup int parents[32];
  threadgroup int round_size;
  threadgroup int round_reset;

  if (QSHARED) {
    for (int j = (int)lid; j < d; j += (int)THREADS) {
      qv[j] = queries[qrow + j];
    }
  }
  for (uint i = lid; i < HSIZE; i += THREADS) {
    atomic_store_explicit(&visited[i], -1, memory_order_relaxed);
  }
  for (uint i = lid; i < 2 * M; i += THREADS) {
    top_s[i] = -INFINITY;
    top_i[i] = -1;
  }
  if (lid == 0) {
    atomic_store_explicit(&filled, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint cur = 0;
  for (int it = 0;; ++it) {
    bool random_round = it < n_rand;
    // the round's nodes: random ones, or the best unexpanded entries of the list
    if (!random_round && sg == 0) {
      uint found = 0;
      for (uint b = 0; b < M && found < (uint)width; b += 32) {
        uint i = cur * M + b + lane;
        int v = top_i[i];
        bool open = v >= 0;
        uint before = simd_prefix_exclusive_sum(open ? 1u : 0u);
        if (open && found + before < (uint)width) {
          parents[found + before] = v;
          top_i[i] = -v - 2;
        }
        found += simd_sum(open ? 1u : 0u);
      }
      if (lane == 0) {
        found = min(found, (uint)width);
        round_size = it - n_rand >= max_iter ? 0 : (int)found * deg;
      }
    }
    if (lid == 0) {
      if (random_round) {
        round_size = (int)CP;
      }
      uint fill = atomic_load_explicit(&filled, memory_order_relaxed);
      round_reset = fill + (uint)round_size > HSIZE / 2 ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int size = round_size;
    if (size == 0) {
      break;
    }

    if (round_reset != 0) {
      // the visited set is half full: start it again from the list's nodes
      for (uint i = lid; i < HSIZE; i += THREADS) {
        atomic_store_explicit(&visited[i], -1, memory_order_relaxed);
      }
      if (lid == 0) {
        atomic_store_explicit(&filled, 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = lid; i < M; i += THREADS) {
        int v = mlxvs_node(top_i[cur * M + i]);
        if (v >= 0 && mlxvs_visit(visited, HBITS, v)) {
          atomic_fetch_add_explicit(&filled, 1u, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // candidates not visited before
    for (uint c = lid; c < CP; c += THREADS) {
      int id = -1;
      if (c < (uint)size) {
        int node;
        if (random_round) {
          uint h = mlxvs_mix(seed ^ mlxvs_mix(q * 0x9e3779b9u + (uint)it * 0x85ebca6bu + c));
          node = (int)(h % (uint)n);
        } else {
          int p = parents[c / (uint)deg];
          node = graph[(size_t)p * (size_t)deg + c % (uint)deg];
        }
        if (node >= 0 && node < n && mlxvs_visit(visited, HBITS, node)) {
          atomic_fetch_add_explicit(&filled, 1u, memory_order_relaxed);
          id = node;
        }
      }
      cand_i[c] = id;
      cand_s[c] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // scores, one SIMD group per candidate; larger is closer
    for (uint c = sg; c < CP; c += SIMDS) {
      int id = cand_i[c];
      if (id < 0) {
        continue;
      }
      size_t row = (size_t)id * (size_t)d;
      float acc = 0.0f;
      for (int j = (int)lane; j < d; j += 32) {
        float a = QSHARED ? qv[j] : queries[qrow + j];
        float b = base[row + j];
        acc += metric == 1 ? -(a - b) * (a - b) : a * b;
      }
      acc = simd_sum(acc);
      if (lane == 0) {
        cand_s[c] = metric == 2 ? acc * inv_norm[id] : acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // bitonic sort of the candidates, best first
    for (uint span = 2; span <= CP; span <<= 1) {
      for (uint stride = span >> 1; stride > 0; stride >>= 1) {
        for (uint i = lid; i < CP; i += THREADS) {
          uint j = i ^ stride;
          if (j > i) {
            bool forward = (i & span) == 0;
            float si = cand_s[i];
            float sj = cand_s[j];
            int ii = cand_i[i];
            int ij = cand_i[j];
            if (forward != mlxvs_before(si, ii, sj, ij)) {
              cand_s[i] = sj;
              cand_s[j] = si;
              cand_i[i] = ij;
              cand_i[j] = ii;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // merge into the other half of the list: an entry's place is its index plus
    // the entries of the other list before it (ties: list entries first)
    uint src = cur * M;
    uint dst = (cur ^ 1u) * M;
    for (uint i = lid; i < M; i += THREADS) {
      float s = top_s[src + i];
      int v = top_i[src + i];
      int node = mlxvs_node(v);
      uint lo = 0;
      uint hi = CP;
      while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (mlxvs_before(cand_s[mid], cand_i[mid], s, node)) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (i + lo < M) {
        top_s[dst + i + lo] = s;
        top_i[dst + i + lo] = v;
      }
    }
    for (uint j = lid; j < CP; j += THREADS) {
      float s = cand_s[j];
      int v = cand_i[j];
      if (j >= M) {
        continue;
      }
      uint lo = 0;
      uint hi = M;
      while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (!mlxvs_before(s, v, top_s[src + mid], mlxvs_node(top_i[src + mid]))) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (j + lo < M) {
        top_s[dst + j + lo] = s;
        top_i[dst + j + lo] = v;
      }
    }
    cur ^= 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = lid; i < (uint)k; i += THREADS) {
    float s = top_s[cur * M + i];
    int v = mlxvs_node(top_i[cur * M + i]);
    bool ok = v >= 0;
    out_ids[(size_t)q * k + i] = v;
    out_scores[(size_t)q * k + i] = metric == 1 ? (ok ? -s : INFINITY) : (ok ? s : -INFINITY);
  }
)";

} // namespace mlxvs::kernels
