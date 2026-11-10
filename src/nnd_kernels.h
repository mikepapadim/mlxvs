// Copyright © 2026 mikepapadim
// Ported from JVector's jvector-gpu NN-Descent (Apache-2.0, DataStax, Inc.);
// see NOTICE.

//
// NN-Descent kernels: Metal sources compiled at run time by
// mlx::core::fast::metal_kernel. A port of JVector's TornadoVM kernels
// (NndKernels.java) to MLX's functional custom kernels: a kernel cannot update
// its inputs, so every round reads the graph and writes a fresh one (see
// nndescent.cpp).
#pragma once

namespace mlxvs::kernels {

/** New and old neighbors sampled per node per round; a join scores the 4 S
 * members of a node. */
constexpr int NND_S = 16;
/** Members of a join: sampled new, reverse new, sampled old, reverse old, NND_S
 * each. */
constexpr int NND_JOIN = 4 * NND_S;
/** Threads per threadgroup of the join and merge kernels. */
constexpr int NND_GROUP = 256;
/** Threads per threadgroup of the sample kernel (one node per thread). */
constexpr int NND_SMALL_GROUP = 64;
/** List entries plus proposal slots: the merge sorts NND_LIST entries, one per
 * thread. */
constexpr int NND_LIST = 256;
/** Rounds of the random start: block joins over a random permutation of the
 * nodes. */
constexpr int NND_START_PASSES = 4;
/** The score of an empty list entry. */
constexpr float NND_NONE = -3.0e38f;

/** Indices into the params input of every NN-Descent kernel. */
enum NndParam {
  P_N = 0,
  P_DIM = 1,
  P_K = 2,
  P_P = 3,
  P_METRIC = 4,
  P_PHASE = 5,
  P_SEED = 6,
  P_COUNT = 7
};
/** Metric codes in params[P_METRIC] (JVector's KnnKernels codes). */
enum NndMetric { M_INNER_PRODUCT = 0, M_COSINE = 1, M_EUCLIDEAN = 2 };

constexpr const char* NND_HEADER = R"(
#include <metal_simdgroup_matrix>

constant int NND_S = 16;
constant int NND_JOIN = 64;
constant float NND_NONE = -3.0e38f;

// JVector's score of a pair from its dot product and squared norms, larger is closer:
// L2 1 / (1 + d^2), cosine (1 + cos) / 2, inner product (1 + dot) / 2.
inline float nnd_similarity(int metric, float dot, float a_norm, float b_norm) {
  if (metric == 2) {
    float d2 = a_norm + b_norm - 2.0f * dot;
    return 1.0f / (1.0f + max(d2, 0.0f));
  } else if (metric == 1) {
    float c = dot / max(1e-30f, sqrt(a_norm * b_norm));
    return (1.0f + c) / 2.0f;
  }
  return (1.0f + dot) / 2.0f;
}

// A slot in [0, size) for id: Fibonacci hashing salted per round (Java's int arithmetic, as uint).
inline uint nnd_slot(int id, int salt, uint size) {
  uint h = (uint)(id ^ salt) * 2654435761u;
  h ^= h >> 15;
  return (h >> 1) % size;
}

// A proposal: the id in the high 32 bits, the score's bits in the low ones, stored as one 64-bit word so that the id
// and the score in a slot come from the same writer. -1: an empty slot.
inline int64_t nnd_pack(int id, float score) {
  return (int64_t)(((ulong)(uint)id << 32) | (ulong)as_type<uint>(score));
}

// Pair (r, c) of a join, r one of the first 2 S (new) members: every new x old pair and the new x new pairs above the
// diagonal, once each. Each side whose list's worst score the pair beats gets the other side proposed, in a hashed
// slot (a collision overwrites: the entry is lost, which NN-Descent tolerates; an id always lands in the same slot).
inline void nnd_propose(threadgroup const int* ids, threadgroup const float* norm, threadgroup const float* worst,
                        device int64_t* proposals, int metric, int p, int salt, uint r, uint c, float dot) {
  int x = ids[r];
  int y = ids[c];
  if ((r < c || c >= 2 * NND_S) && x >= 0 && y >= 0 && x != y) {
    float s = nnd_similarity(metric, dot, norm[r], norm[c]);
    if (s > worst[r]) {
      proposals[(size_t)x * p + nnd_slot(y, salt, (uint)p)] = nnd_pack(y, s);
    }
    if (s > worst[c]) {
      proposals[(size_t)y * p + nnd_slot(x, salt, (uint)p)] = nnd_pack(x, s);
    }
  }
}
)";

// Sample: one thread per node u. Takes up to S new neighbors of u's list (best
// first; their flags are cleared in the output flags) and up to S old ones into
// u's row of members, and adds u to each sampled neighbor's reverse samples (a
// hashed slot of that neighbor's row). Rows of members: [0, S) new, [S, 2S)
// reverse new, [2S, 3S) old, [3S, 4S) reverse old; the output is pre-filled
// with -1.
//
// inputs: graph_ids [n, K] int, graph_new [n, K] int, params int
// outputs: members [n, 4S] int (init -1), new_flags [n, K] int
constexpr const char* NND_SAMPLE_SOURCE = R"(
  uint u = thread_position_in_grid.x;
  int n = params[0];
  int k = params[2];
  int salt = params[6];
  if (u >= (uint)n) {
    return;
  }
  int n_new = 0;
  int n_old = 0;
  uint my_slot = nnd_slot((int)u, salt, (uint)NND_S);
  for (int j = 0; j < k; ++j) {
    size_t at = (size_t)u * k + j;
    int v = graph_ids[at];
    int flag = graph_new[at];
    if (v >= 0) {
      if (flag == 1) {
        if (n_new < NND_S) {
          members[(size_t)u * NND_JOIN + n_new] = v;
          n_new++;
          flag = 0;
          members[(size_t)v * NND_JOIN + NND_S + my_slot] = (int)u;
        }
      } else if (n_old < NND_S) {
        members[(size_t)u * NND_JOIN + 2 * NND_S + n_old] = v;
        n_old++;
        members[(size_t)v * NND_JOIN + 3 * NND_S + my_slot] = (int)u;
      }
    }
    new_flags[at] = flag;
  }
)";

// The local join with Apple's SIMD-group matrices: one threadgroup per row of
// members (a node's samples, or a block of a random permutation in the start).
// FP16 rows (simdgroup_half8x8) multiplied into FP32 accumulators
// (simdgroup_float8x8), as cuVS computes NN-Descent distances. The 32 new
// members (rows) against all 64 (columns): SIMD group g accumulates the 8 x 8
// blocks of row block g / 2 against column blocks 4 (g % 2) .. 4 (g % 2) + 3.
// The 32 x 64 dot products go through threadgroup memory to the proposals.
//
// inputs: rows [n, dim] half, norms [n] float (of the FP16 values), params int,
// graph_scores [n, K] float,
//         members [groups, 4S] int
// outputs: proposals [n, P] int64 (init -1)
constexpr const char* NND_JOIN_SIMD_SOURCE = R"(
  const uint GROUP = 256;
  const uint TILE = 64;
  const uint LDS = TILE + 8;  // padded row stride of the FP16 staging, against bank conflicts
  uint node = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint g = simdgroup_index_in_threadgroup;
  int dim = params[1];
  int k = params[2];
  int p = params[3];
  int metric = params[4];
  int salt = params[6];

  threadgroup int ids[NND_JOIN];
  threadgroup float norm[NND_JOIN];
  threadgroup float worst[NND_JOIN];
  threadgroup half a[NND_JOIN * LDS];
  threadgroup float out[2 * NND_S * NND_JOIN];

  // each member, its squared norm and its list's worst score, read once
  if (lid < (uint)NND_JOIN) {
    int id = members[(size_t)node * NND_JOIN + lid];
    ids[lid] = id;
    norm[lid] = id >= 0 ? norms[id] : 0.0f;
    worst[lid] = id >= 0 ? graph_scores[(size_t)id * k + k - 1] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint row_block = g / 2;
  uint col_block = (g % 2) * 4;
  simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k0 = 0; k0 < dim; k0 += (int)TILE) {
    // dimensions k0 .. k0 + TILE - 1 of the members (zero past dim or for an empty member), two per load when dim is
    // even
    if ((dim & 1) == 0) {
      // two scalar loads, not a half2 pointer: MLX may bind a small input in
      // the constant address space, where a device pointer cast fails
      for (uint e = lid; e < NND_JOIN * (TILE / 2); e += GROUP) {
        uint row = e / (TILE / 2);
        uint w = e - row * (TILE / 2);
        int id = ids[row];
        int d = k0 + 2 * (int)w;
        half2 v = half2(0.0h);
        if (id >= 0 && d < dim) {
          size_t at = (size_t)id * dim + d;
          v = half2(rows[at], rows[at + 1]);
        }
        a[row * LDS + 2 * w] = v.x;
        a[row * LDS + 2 * w + 1] = v.y;
      }
    } else {
      for (uint e = lid; e < NND_JOIN * TILE; e += GROUP) {
        uint row = e / TILE;
        uint kk = e - row * TILE;
        int id = ids[row];
        int d = k0 + (int)kk;
        a[row * LDS + kk] = id >= 0 && d < dim ? rows[(size_t)id * dim + d] : 0.0h;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < TILE; kk += 8) {
      simdgroup_half8x8 x;
      simdgroup_half8x8 bt;
      simdgroup_load(x, a + row_block * 8 * LDS + kk, LDS);
      simdgroup_load(bt, a + (col_block + 0) * 8 * LDS + kk, LDS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(m0, x, bt, m0);
      simdgroup_load(bt, a + (col_block + 1) * 8 * LDS + kk, LDS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(m1, x, bt, m1);
      simdgroup_load(bt, a + (col_block + 2) * 8 * LDS + kk, LDS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(m2, x, bt, m2);
      simdgroup_load(bt, a + (col_block + 3) * 8 * LDS + kk, LDS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(m3, x, bt, m3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the 32 x 64 dot products, row-major, then one proposal test per pair
  uint at = row_block * 8 * NND_JOIN + col_block * 8;
  simdgroup_store(m0, out + at + 0, NND_JOIN);
  simdgroup_store(m1, out + at + 8, NND_JOIN);
  simdgroup_store(m2, out + at + 16, NND_JOIN);
  simdgroup_store(m3, out + at + 24, NND_JOIN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = lid; e < 2 * NND_S * NND_JOIN; e += GROUP) {
    uint r = e / NND_JOIN;
    nnd_propose(ids, norm, worst, proposals, metric, p, salt, r, e - r * NND_JOIN, out[e]);
  }
)";

// The local join in FP32 SIMT, the same pairs and proposals as the SIMD-group
// join: 16 x 16 threads, each scores 2 new members (rows ty, ty + 16) against 4
// members (columns tx + 16 j) over staged tiles of 64 dimensions.
//
// inputs: rows [n, dim] float, norms [n] float, params int, graph_scores [n, K]
// float, members [groups, 4S] int outputs: proposals [n, P] int64 (init -1)
constexpr const char* NND_JOIN_SIMT_SOURCE = R"(
  const uint GROUP = 256;
  const uint TILE = 64;
  const uint LDS = TILE + 1;
  uint node = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint ty = lid / 16;
  uint tx = lid - ty * 16;
  int dim = params[1];
  int k = params[2];
  int p = params[3];
  int metric = params[4];
  int salt = params[6];

  threadgroup int ids[NND_JOIN];
  threadgroup float norm[NND_JOIN];
  threadgroup float worst[NND_JOIN];
  threadgroup float a[NND_JOIN * LDS];

  if (lid < (uint)NND_JOIN) {
    int id = members[(size_t)node * NND_JOIN + lid];
    ids[lid] = id;
    norm[lid] = id >= 0 ? norms[id] : 0.0f;
    worst[lid] = id >= 0 ? graph_scores[(size_t)id * k + k - 1] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // the old x old block, which proposes nothing, is not computed
  float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k0 = 0; k0 < dim; k0 += (int)TILE) {
    for (uint e = lid; e < NND_JOIN * TILE; e += GROUP) {
      uint row = e / TILE;
      uint kk = e - row * TILE;
      int id = ids[row];
      int d = k0 + (int)kk;
      a[row * LDS + kk] = id >= 0 && d < dim ? rows[(size_t)id * dim + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < TILE; ++kk) {
      float a0 = a[ty * LDS + kk];
      float a1 = a[(ty + 16) * LDS + kk];
      for (uint j = 0; j < 4; ++j) {
        float b = a[(tx + 16 * j) * LDS + kk];
        c0[j] += a0 * b;
        c1[j] += a1 * b;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < 4; ++j) {
    nnd_propose(ids, norm, worst, proposals, metric, p, salt, ty, tx + 16 * j, c0[j]);
    nnd_propose(ids, norm, worst, proposals, metric, p, salt, ty + 16, tx + 16 * j, c1[j]);
  }
)";

// Merge: one threadgroup of 256 threads per node. The list and the proposals
// not already listed are sorted together (score descending, id ascending;
// bitonic, one entry per thread), duplicates dropped and the best K kept,
// proposals flagged new. A node without new proposals copies its list.
// changed[u]: the proposals that entered u's list.
//
// inputs: params int, graph_ids [n, K] int, graph_scores [n, K] float,
// graph_new [n, K] int, proposals [n, P] int64 outputs: out_ids [n, K] int,
// out_scores [n, K] float, out_new [n, K] int, changed [n] int
constexpr const char* NND_MERGE_SOURCE = R"(
  const uint LIST = 256;
  uint u = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  int k = params[2];
  int p = params[3];
  size_t row = (size_t)u * k;

  threadgroup float keys[LIST];
  threadgroup int ids[LIST];
  threadgroup int fresh[LIST];
  threadgroup int listed[LIST];
  threadgroup int any_new;
  threadgroup atomic_int entered;

  // entries [0, K): the list; [K, K + P): the proposals
  if (lid < (uint)k) {
    listed[lid] = graph_ids[row + lid];
  }
  if (lid == 0) {
    any_new = 0;
    atomic_store_explicit(&entered, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the proposals not yet listed in listed[K + i] (-1 otherwise)
  float given = NND_NONE;
  if (lid >= (uint)k && lid < (uint)(k + p)) {
    int64_t w = proposals[(size_t)u * p + lid - k];
    int v = (int)(w >> 32);
    bool known = v < 0 || v == (int)u;
    if (v >= 0) {
      given = as_type<float>((uint)(w & 0xffffffff));
    }
    for (int j = 0; j < k && !known; ++j) {
      known = listed[j] == v;
    }
    listed[lid] = known ? -1 : v;
    if (!known) {
      any_new = 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // the test is the same in every thread: the barriers below are uniform
  if (any_new == 0) {
    // nothing new for u: its list stays as it is
    if (lid < (uint)k) {
      out_ids[row + lid] = graph_ids[row + lid];
      out_scores[row + lid] = graph_scores[row + lid];
      out_new[row + lid] = graph_new[row + lid];
    }
    if (lid == 0) {
      changed[u] = 0;
    }
  } else {
    float key = NND_NONE;
    int id = 2147483647;
    int is_new = 0;
    if (lid < (uint)k) {
      int v = listed[lid];
      if (v >= 0) {
        key = graph_scores[row + lid];
        id = v;
        is_new = graph_new[row + lid];
      }
    } else if (lid < (uint)(k + p)) {
      int v = listed[lid];
      if (v >= 0) {
        key = given;
        id = v;
        is_new = 1;
      }
    }
    keys[lid] = key;
    ids[lid] = id;
    fresh[lid] = is_new + (lid >= (uint)k ? 2 : 0);  // bit 0: the new flag, 2: came from a proposal
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // bitonic sort, best first: larger score, then smaller id
    for (uint size = 2; size <= LIST; size <<= 1) {
      for (uint step = size >> 1; step > 0; step >>= 1) {
        uint i = lid;
        uint j = i ^ step;
        if (j > i) {
          bool descending = (i & size) == 0;
          float ki = keys[i];
          float kj = keys[j];
          int pi = ids[i];
          int pj = ids[j];
          bool i_first = ki > kj || (ki == kj && pi < pj);
          if (descending != i_first) {
            keys[i] = kj;
            keys[j] = ki;
            ids[i] = pj;
            ids[j] = pi;
            int fi = fresh[i];
            fresh[i] = fresh[j];
            fresh[j] = fi;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    // compact the first K distinct entries in order: an entry is kept when it is filled and differs from the one
    // before it (a safeguard: the list is distinct and listed ids were dropped); its place is the number of kept
    // entries before it (an inclusive prefix sum, minus itself)
    int keep = keys[lid] > -1.0e38f && (lid == 0 || ids[lid] != ids[lid - 1]) ? 1 : 0;
    threadgroup int* scan = listed;
    scan[lid] = keep;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < LIST; step <<= 1) {
      int add = lid >= step ? scan[lid - step] : 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scan[lid] += add;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int at = scan[lid] - keep;
    int kept = scan[LIST - 1];
    if (keep == 1 && at < k) {
      out_ids[row + at] = ids[lid];
      out_scores[row + at] = keys[lid];
      out_new[row + at] = fresh[lid] & 1;
      if (fresh[lid] >= 2) {
        atomic_fetch_add_explicit(&entered, 1, memory_order_relaxed);
      }
    }
    if ((int)lid >= kept && lid < (uint)k) {
      out_ids[row + lid] = -1;
      out_scores[row + lid] = NND_NONE;
      out_new[row + lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      changed[u] = atomic_load_explicit(&entered, memory_order_relaxed);
    }
  }
)";

} // namespace mlxvs::kernels
