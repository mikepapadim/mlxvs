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
// Whether the filter (a bitset over the rows, bit set: allowed) allows id.
template <typename F>
inline bool mlxvs_allowed(F filter, int id) {
  return ((filter[id >> 5] >> (id & 31)) & 1u) != 0;
}
inline float4 mlxvs_load4(const device float* p) {
  return float4(*(const device packed_float4*)p);
}
inline float4 mlxvs_load4(const device half* p) {
  return float4(*(const device packed_half4*)p);
}
// Score of base row id against the query (larger is closer), computed by a
// team of TEAM consecutive lanes (CAGRA's team size), four columns per load;
// every lane of the team gets the total. Every lane of the SIMD group must
// call it (the sum is a shuffle); inactive teams get 0. metric: 0 inner
// product, 1 l2 (as minus the squared distance), 2 cosine (inner product times
// the row's inverse norm).
template <uint TEAM, typename T, typename Q, typename N>
inline float mlxvs_score(
    const device T* base,
    Q qv,
    N inv_norm,
    int id,
    int d,
    int metric,
    uint lane,
    bool active) {
  float acc = 0.0f;
  if (active) {
    int tl = (int)(lane % TEAM);
    const device T* row = base + (size_t)id * (size_t)d;
    int d4 = d & ~3;
    for (int j = tl * 4; j < d4; j += (int)TEAM * 4) {
      float4 b = mlxvs_load4(row + j);
      float4 a = float4(qv[j], qv[j + 1], qv[j + 2], qv[j + 3]);
      if (metric == 1) {
        float4 t = a - b;
        acc -= dot(t, t);
      } else {
        acc += dot(a, b);
      }
    }
    for (int j = d4 + tl; j < d; j += (int)TEAM) {
      float a = qv[j];
      float b = (float)row[j];
      acc += metric == 1 ? -(a - b) * (a - b) : a * b;
    }
  }
  for (uint off = TEAM / 2; off > 0; off >>= 1) {
    acc += simd_shuffle_xor(acc, (ushort)off);
  }
  return active && metric == 2 ? acc * inv_norm[id] : acc;
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
// The traversed set of the multi-CTA search, shared by a query's threadgroups
// in device memory (open addressing, linear probing): empty slots -1, a removed
// id stored as -(id + 2), as cuVS's hashmap with removal. True if id was not
// there (it is now).
inline bool mlxvs_dev_insert(device atomic_int* table, uint bits, int id) {
  uint mask = (1u << bits) - 1u;
  uint at = (mlxvs_mix((uint)id) >> (32u - bits)) & mask;
  int removed = -id - 2;
  for (uint probes = 0; probes <= mask;) {
    int expected = -1;
    if (atomic_compare_exchange_weak_explicit(
            &table[at], &expected, id, memory_order_relaxed, memory_order_relaxed)) {
      return true;
    }
    if (expected == id) {
      return false;
    }
    if (expected == removed) {
      int again = removed;
      if (atomic_compare_exchange_weak_explicit(
              &table[at], &again, id, memory_order_relaxed, memory_order_relaxed)) {
        return true;
      }
      if (again == id) {
        return false;
      }
      continue;
    }
    if (expected != -1) {
      at = (at + 1u) & mask;
      ++probes;
    }
  }
  return false;
}
inline bool mlxvs_dev_find(device atomic_int* table, uint bits, int id) {
  uint mask = (1u << bits) - 1u;
  uint at = (mlxvs_mix((uint)id) >> (32u - bits)) & mask;
  for (uint probes = 0; probes <= mask; ++probes) {
    int v = atomic_load_explicit(&table[at], memory_order_relaxed);
    if (v == id) {
      return true;
    }
    if (v == -1 || v == -id - 2) {
      return false;
    }
    at = (at + 1u) & mask;
  }
  return false;
}
inline void mlxvs_dev_remove(device atomic_int* table, uint bits, int id) {
  uint mask = (1u << bits) - 1u;
  uint at = (mlxvs_mix((uint)id) >> (32u - bits)) & mask;
  for (uint probes = 0; probes <= mask;) {
    int v = atomic_load_explicit(&table[at], memory_order_relaxed);
    if (v == -1) {
      return;
    }
    if (v == id) {
      int expected = id;
      if (atomic_compare_exchange_weak_explicit(
              &table[at], &expected, -id - 2, memory_order_relaxed, memory_order_relaxed) ||
          expected != id) {
        return;
      }
      continue;
    }
    at = (at + 1u) & mask;
    ++probes;
  }
}
)";

// One threadgroup of 256 threads per query, as CAGRA's single-CTA search. The
// query keeps a sorted list of M entries (itopk). The random start, as cuVS,
// fills M + W * deg slots (in rounds of at most CP), each with the best of
// n_rand random nodes. Then each round expands the best unexpanded entries (up
// to W of them, marked in the list), gathers their graph neighbors, drops the
// ones already visited (a hash set in threadgroup memory, cleared and refilled
// with the list's nodes when half full), scores the rest (one team of TEAM
// lanes per candidate), sorts them (bitonic) and merges them into the list
// (each entry's rank by binary search in the other list). It stops when every
// entry of the list has been expanded, or after max_iter rounds (cuVS's
// max_iterations - 1: its first iteration is the random start's).
//
// template: M (list length, multiple of 32), CP (candidate slots, power of
// two), QCAP (query floats in threadgroup memory, 1 when read from device),
// QSHARED, HBITS (log2 of the visited-set slots), TEAM (lanes per distance:
// 4, 8, 16 or 32), FILTER (filter [ceil(n / 32)] uint: bit set, row allowed;
// expanded nodes it rejects leave the list, and only allowed ones are output),
// NSEEDS (seeds [nq, NSEEDS] int: the first start slots' nodes, as cuVS)
// inputs: base [n, d] float, graph [n, deg] int, queries [nq, d] float,
// inv_norm [n] float (cosine only), params [n, d, deg, k, W, max_iter, metric,
// seed, n_rand, rand_slots] int
// outputs: out_ids [nq, k] int, out_scores [nq, k] float
constexpr const char* SEARCH_SOURCE = R"(
  const uint THREADS = 256;
  const uint TEAMS = THREADS / TEAM;
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
  int rand_slots = params[9];
  int rand_rounds = (rand_slots + (int)CP - 1) / (int)CP;
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
  threadgroup uint sg_count[THREADS / 32];

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
    bool random_round = it < rand_rounds;
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
        round_size = it - rand_rounds >= max_iter ? 0 : (int)found * deg;
      }
    }
    if (lid == 0) {
      if (random_round) {
        round_size = min((int)CP, rand_slots - it * (int)CP);
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

    if (random_round) {
      // each slot: the best of n_rand random nodes, one team per slot
      for (uint c0 = 0; c0 < CP; c0 += TEAMS) {
        uint c = c0 + lid / TEAM;
        bool active = c < (uint)size;
        float best = -INFINITY;
        int best_id = -1;
        uint slot = (uint)it * CP + c;
        for (int j = 0; j < n_rand; ++j) {
          uint g = slot + (uint)rand_slots * (uint)j;
          uint h = mlxvs_mix(seed ^ mlxvs_mix(q * 0x9e3779b9u + g * 0x85ebca6bu));
          int node = (int)(h % (uint)n);
          if (g < (uint)NSEEDS) {
            // the query's seed nodes first, as cuVS
            int sn = seeds[(size_t)q * NSEEDS + g];
            node = sn >= 0 && sn < n ? sn : node;
          }
          float sc = QSHARED
              ? mlxvs_score<TEAM>(base, (threadgroup const float*)qv, inv_norm, node, d, metric, lane, active)
              : mlxvs_score<TEAM>(base, queries + qrow, inv_norm, node, d, metric, lane, active);
          if (active && mlxvs_before(sc, node, best, best_id)) {
            best = sc;
            best_id = node;
          }
        }
        if (c < CP && lid % TEAM == 0) {
          bool fresh = active && best_id >= 0 && mlxvs_visit(visited, HBITS, best_id);
          if (fresh) {
            atomic_fetch_add_explicit(&filled, 1u, memory_order_relaxed);
          }
          cand_i[c] = fresh ? best_id : -1;
          cand_s[c] = fresh ? best : -INFINITY;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // candidates not visited before
      for (uint c = lid; c < CP; c += THREADS) {
        int id = -1;
        if (c < (uint)size) {
          int p = parents[c / (uint)deg];
          int node = graph[(size_t)p * (size_t)deg + c % (uint)deg];
          if (node >= 0 && node < n && mlxvs_visit(visited, HBITS, node)) {
            atomic_fetch_add_explicit(&filled, 1u, memory_order_relaxed);
            id = node;
          }
        }
        cand_i[c] = id;
        cand_s[c] = -INFINITY;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // scores, one team per candidate; larger is closer
      for (uint c0 = 0; c0 < (uint)size; c0 += TEAMS) {
        uint c = c0 + lid / TEAM;
        int id = c < (uint)size ? cand_i[c] : -1;
        float sc = QSHARED
            ? mlxvs_score<TEAM>(base, (threadgroup const float*)qv, inv_norm, id, d, metric, lane, id >= 0)
            : mlxvs_score<TEAM>(base, queries + qrow, inv_norm, id, d, metric, lane, id >= 0);
        if (id >= 0 && lid % TEAM == 0) {
          cand_s[c] = sc;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

    if (FILTER) {
      // expanded nodes the filter rejects leave the list, as cuVS: compact it
      // into the other half (M <= THREADS)
      bool keep = false;
      float ks = -INFINITY;
      int kv = -1;
      if (lid < M) {
        kv = top_i[cur * M + lid];
        ks = top_s[cur * M + lid];
        keep = kv >= 0 || (kv < -1 && mlxvs_allowed(filter, -kv - 2));
      }
      uint before = simd_prefix_exclusive_sum(keep ? 1u : 0u);
      if (lane == 31) {
        sg_count[sg] = before + (keep ? 1u : 0u);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint offset = 0;
      for (uint g = 0; g < sg; ++g) {
        offset += sg_count[g];
      }
      uint dst = (cur ^ 1u) * M;
      if (lid < M) {
        top_s[dst + lid] = -INFINITY;
        top_i[dst + lid] = -1;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (keep) {
        top_s[dst + offset + before] = ks;
        top_i[dst + offset + before] = kv;
      }
      cur ^= 1u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // the first k entries of the list (the filter allows), then -1
  bool ok = false;
  float s = -INFINITY;
  int v = -1;
  if (lid < M) {
    s = top_s[cur * M + lid];
    v = mlxvs_node(top_i[cur * M + lid]);
    ok = v >= 0 && (!FILTER || mlxvs_allowed(filter, v));
  }
  uint before = simd_prefix_exclusive_sum(ok ? 1u : 0u);
  if (lane == 31) {
    sg_count[sg] = before + (ok ? 1u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint offset = 0;
  uint total = 0;
  for (uint g = 0; g < THREADS / 32; ++g) {
    offset += g < sg ? sg_count[g] : 0u;
    total += sg_count[g];
  }
  uint at = offset + before;
  if (ok && at < (uint)k) {
    out_ids[(size_t)q * k + at] = v;
    out_scores[(size_t)q * k + at] = metric == 1 ? -s : s;
  }
  for (uint i = total + lid; i < (uint)k; i += THREADS) {
    out_ids[(size_t)q * k + i] = -1;
    out_scores[(size_t)q * k + i] = metric == 1 ? INFINITY : -INFINITY;
  }
)";

/** Threads per threadgroup (CTA) of the multi-CTA search. */
constexpr int SEARCH_MC_THREADS = 128;
/** List entries each threadgroup of the multi-CTA search keeps (cuVS's). */
constexpr int SEARCH_MC_ITOPK = 32;
/** The longest buffer (list and one parent's neighbors) of one threadgroup. */
constexpr int SEARCH_MC_BUFFER_MAX = 256;

// CAGRA's multi-CTA search, as cuVS's search_multi_cta: several threadgroups
// per query, each with its own list of 32 entries followed by one parent's
// neighbors (B entries, sorted together), expanding one parent per iteration.
// A traversed set in device memory, shared by the query's threadgroups, makes
// each node a parent of one threadgroup only; each threadgroup's visited set
// (threadgroup memory) is cleared every iteration and refilled with its
// buffer's nodes. Per iteration: sort the buffer; pick the best entry not
// expanded yet that the traversed set takes (an entry beyond the list moves
// into the list's last empty slot; ones another threadgroup took are
// dropped); drop former parents pushed beyond the list (and from the traversed
// set); write the parent's neighbors not visited nor traversed over the end
// of the buffer and score them; drop entries other threadgroups have since
// expanded. Each threadgroup writes its 32 best (parents, then nodes the
// traversed set takes, so each node comes from one threadgroup) for the
// final top-k.
//
// template: B (buffer, multiple of 32), BP (power of two >= B), QCAP, QSHARED,
// VBITS (log2 of the visited-set slots), TEAM, FILTER, NSEEDS (as above)
// inputs: base [n, d] float, graph [n, deg] int, queries [nq, d] float,
// inv_norm [n] float (cosine only), params [n, d, deg, max_iter, min_iter,
// metric, seed, n_rand, num_cta, nq, tbits] int
// outputs (atomic): out_ids [nq, num_cta * 32] int, out_scores [nq, num_cta *
// 32] float (larger is closer), traversed [nq, 1 << tbits] int (-1 filled)
constexpr const char* SEARCH_MC_SOURCE = R"(
  const uint THREADS = 128;
  const uint TEAMS = THREADS / TEAM;
  const uint ITOPK = 32;
  const uint VSIZE = 1u << VBITS;
  uint cta = threadgroup_position_in_grid.x;
  uint q = threadgroup_position_in_grid.y;
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid / 32;
  uint lane = lid % 32;
  int n = params[0];
  int d = params[1];
  int deg = params[2];
  int max_iter = params[3];
  int min_iter = params[4];
  int metric = params[5];
  uint seed = (uint)params[6];
  int n_rand = params[7];
  uint num_cta = (uint)params[8];
  uint nq = (uint)params[9];
  uint tbits = (uint)params[10];
  size_t qrow = (size_t)q * (size_t)d;
  device atomic_int* trav = traversed + (size_t)q * ((size_t)1 << tbits);

  threadgroup float qv[QCAP];
  threadgroup atomic_int visited[VSIZE];
  threadgroup float buf_s[BP];
  threadgroup int buf_i[BP];
  threadgroup int parent_slot;
  threadgroup atomic_int position;

  if (QSHARED) {
    for (int j = (int)lid; j < d; j += (int)THREADS) {
      qv[j] = queries[qrow + j];
    }
  }
  for (uint i = lid; i < VSIZE; i += THREADS) {
    atomic_store_explicit(&visited[i], -1, memory_order_relaxed);
  }
  for (uint i = lid; i < BP; i += THREADS) {
    buf_s[i] = -INFINITY;
    buf_i[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // random start: deg slots, each the best of n_rand random nodes; the
  // query's threadgroups draw different nodes
  uint block = cta + num_cta * q;
  uint blocks = num_cta * nq;
  for (uint c0 = 0; c0 < (uint)deg; c0 += TEAMS) {
    uint c = c0 + lid / TEAM;
    bool active = c < (uint)deg;
    float best = -INFINITY;
    int best_id = -1;
    for (int j = 0; j < n_rand; ++j) {
      uint g = block + blocks * (c + (uint)deg * (uint)j);
      int node = (int)(mlxvs_mix(seed ^ mlxvs_mix(g)) % (uint)n);
      // the query's seed nodes, spread over its threadgroups
      uint local = cta + num_cta * (c + (uint)deg * (uint)j);
      if (local < (uint)NSEEDS) {
        int sn = seeds[(size_t)q * NSEEDS + local];
        node = sn >= 0 && sn < n ? sn : node;
      }
      float sc = QSHARED
          ? mlxvs_score<TEAM>(base, (threadgroup const float*)qv, inv_norm, node, d, metric, lane, active)
          : mlxvs_score<TEAM>(base, queries + qrow, inv_norm, node, d, metric, lane, active);
      if (active && mlxvs_before(sc, node, best, best_id)) {
        best = sc;
        best_id = node;
      }
    }
    if (active && lid % TEAM == 0) {
      bool ok = best_id >= 0 && mlxvs_visit(visited, VBITS, best_id) &&
          !mlxvs_dev_find(trav, tbits, best_id);
      buf_i[c] = ok ? best_id : -1;
      buf_s[c] = ok ? best : -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 0;; ++it) {
    // sort the buffer, best first (one SIMD group)
    if (sg == 0) {
      for (uint span = 2; span <= BP; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
          for (uint i = lane; i < BP; i += 32) {
            uint j = i ^ stride;
            if (j > i) {
              bool forward = (i & span) == 0;
              float si = buf_s[i];
              float sj = buf_s[j];
              int ii = buf_i[i];
              int ij = buf_i[j];
              if (forward != mlxvs_before(si, mlxvs_node(ii), sj, mlxvs_node(ij))) {
                buf_s[i] = sj;
                buf_s[j] = si;
                buf_i[i] = ij;
                buf_i[j] = ii;
              }
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (it + 1 >= max_iter) {
      break;
    }

    if (sg == 0) {
      // the next parent: the best unexpanded entry of the first 64 that the
      // traversed set takes
      if (lane == 0) {
        parent_slot = -1;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      int free_slot = -1;
      bool done = false;
      for (uint w = 0; w < 2 && !done; ++w) {
        uint i = w * 32 + lane;
        int v = buf_i[i];
        bool invalid = v == -1;
        bool cand = v >= 0;
        uint cid = simd_prefix_exclusive_sum(cand ? 1u : 0u);
        uint total = simd_sum(cand ? 1u : 0u);
        for (uint t = 0; t < total; ++t) {
          bool taken = false;
          if (cand && cid == t) {
            cand = false;
            if (mlxvs_dev_insert(trav, tbits, v)) {
              if (i < ITOPK) {
                parent_slot = (int)i;
                buf_i[i] = -v - 2;
              } else {
                parent_slot = free_slot;
                buf_i[free_slot] = -v - 2;
                buf_s[free_slot] = buf_s[i];
                buf_i[i] = -1;
                buf_s[i] = -INFINITY;
              }
              taken = true;
            } else {
              // another threadgroup expanded it
              buf_i[i] = -1;
              buf_s[i] = -INFINITY;
              invalid = true;
            }
          }
          if (simd_any(taken)) {
            done = true;
            break;
          }
        }
        if (!done && w == 0) {
          free_slot = simd_max(invalid ? (int)lane : -1);
          done = free_slot < 0;
        }
      }
    } else {
      for (uint i = lid - 32; i < VSIZE; i += THREADS - 32) {
        atomic_store_explicit(&visited[i], -1, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int ps = parent_slot;
    if (ps < 0 && it >= min_iter) {
      break;
    }

    // former parents beyond the list leave the traversed set; the rest of
    // the buffer refills the visited set
    for (uint i = lid; i < B; i += THREADS) {
      int v = buf_i[i];
      if (v == -1) {
        continue;
      }
      if (i >= ITOPK && v < -1) {
        mlxvs_dev_remove(trav, tbits, -v - 2);
        buf_i[i] = -1;
        buf_s[i] = -INFINITY;
      } else {
        mlxvs_visit(visited, VBITS, mlxvs_node(v));
      }
    }
    if (lid == 0) {
      atomic_store_explicit(&position, (int)B, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the parent's neighbors, written from the end of the buffer
    if (ps >= 0) {
      int p = mlxvs_node(buf_i[ps]);
      for (uint c = lid; c < (uint)deg; c += THREADS) {
        int child = graph[(size_t)p * (size_t)deg + c];
        if (child >= 0 && child < n && mlxvs_visit(visited, VBITS, child) &&
            !mlxvs_dev_find(trav, tbits, child)) {
          int j = atomic_fetch_sub_explicit(&position, 1, memory_order_relaxed) - 1;
          buf_i[j] = child;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = (uint)atomic_load_explicit(&position, memory_order_relaxed);
    for (uint j0 = start; j0 < B; j0 += TEAMS) {
      uint j = j0 + lid / TEAM;
      bool active = j < B;
      int id = active ? buf_i[j] : 0;
      float sc = QSHARED
          ? mlxvs_score<TEAM>(base, (threadgroup const float*)qv, inv_norm, id, d, metric, lane, active)
          : mlxvs_score<TEAM>(base, queries + qrow, inv_norm, id, d, metric, lane, active);
      if (active && lid % TEAM == 0) {
        buf_s[j] = sc;
      }
    }
    // entries other threadgroups have expanded since
    for (uint i = lid; i < start; i += THREADS) {
      int v = buf_i[i];
      if (v >= 0 && mlxvs_dev_find(trav, tbits, v)) {
        buf_i[i] = -1;
        buf_s[i] = -INFINITY;
      }
    }
    // an expanded parent the filter rejects leaves the buffer, as cuVS
    if (FILTER && lid == 0 && ps >= 0 &&
        !mlxvs_allowed(filter, mlxvs_node(buf_i[ps]))) {
      buf_i[ps] = -1;
      buf_s[ps] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // this threadgroup's best 32: its parents, and nodes no other threadgroup
  // has written (claimed in the traversed set)
  if (sg == 0) {
    size_t out = ((size_t)q * num_cta + cta) * ITOPK;
    uint offset = 0;
    for (uint b = 0; b < B; b += 32) {
      uint i = b + lane;
      int v = buf_i[i];
      bool valid = false;
      int node = -1;
      if (v < -1) {
        valid = !FILTER || mlxvs_allowed(filter, -v - 2);
        node = -v - 2;
      } else if (v >= 0 && offset < ITOPK && (!FILTER || mlxvs_allowed(filter, v)) &&
                 mlxvs_dev_insert(trav, tbits, v)) {
        valid = true;
        node = v;
      }
      uint j = offset + simd_prefix_exclusive_sum(valid ? 1u : 0u);
      if (valid) {
        if (j < ITOPK) {
          atomic_store_explicit(&out_ids[out + j], node, memory_order_relaxed);
          atomic_store_explicit(&out_scores[out + j], buf_s[i], memory_order_relaxed);
        } else {
          mlxvs_dev_remove(trav, tbits, node);
        }
      }
      offset += simd_sum(valid ? 1u : 0u);
    }
    for (uint i = offset + lane; i < ITOPK; i += 32) {
      atomic_store_explicit(&out_ids[out + i], -1, memory_order_relaxed);
      atomic_store_explicit(&out_scores[out + i], -INFINITY, memory_order_relaxed);
    }
  }
)";

} // namespace mlxvs::kernels
