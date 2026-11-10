// Copyright © 2026 mikepapadim

//
// Metal sources, compiled at run time by mlx::core::fast::metal_kernel.
#pragma once

namespace mlxvs::kernels {

/** Threads per query row in topk_merge. */
constexpr int TOPK_THREADS = 256;
/** The largest k topk_merge keeps. */
constexpr int TOPK_MAX = 256;

constexpr const char* TOPK_HEADER = R"(
// A float as an unsigned int that sorts the same way: larger float, larger key.
inline uint mlxvs_key(float v) {
  uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// "a comes before b": higher score first, then lower id.
inline bool mlxvs_before(float sa, int ia, float sb, int ib) {
  return sa > sb || (sa == sb && (uint)ia < (uint)ib);
}
)";

// One threadgroup of 256 threads per query row. Candidates are the row's t
// scores (ids id_offset + column) and, unless first, the k entries of the
// running list. Radix select on the float keys finds the k-th largest key in
// four passes of 8 bits (per-SIMD-group histograms keep atomic contention low),
// one pass gathers the k winners, a bitonic sort orders them.
//
// inputs: scores [nq, t] float, prev_ids [nq, k] int, prev_scores [nq, k]
// float, params [t, k, id_offset, first] int outputs: out_ids [nq, k] int,
// out_scores [nq, k] float
constexpr const char* TOPK_SOURCE = R"(
  const uint THREADS = 256;
  const uint SIMDS = THREADS / 32;
  const uint KMAX = 256;
  uint q = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid / 32;
  int t = params[0];
  int k = params[1];
  int id_offset = params[2];
  int first = params[3];
  uint total = (uint)t + (first != 0 ? 0u : (uint)k);
  uint keff = min((uint)k, total);
  // inputs are indexed, not bound to pointers: MLX binds some inputs (seen
  // with small arrays) in the constant address space, others in device
  size_t row = (size_t)q * (size_t)t;
  size_t prev = (size_t)q * (size_t)k;

  threadgroup atomic_uint hist[SIMDS * 256];
  threadgroup uint shared_prefix;
  threadgroup uint shared_need;
  threadgroup atomic_uint taken_above;
  threadgroup atomic_uint taken_equal;
  threadgroup float sel_score[KMAX];
  threadgroup int sel_id[KMAX];

  // radix select: after the passes, prefix is the k-th largest key and need the number of entries equal to it to keep
  uint prefix = 0;
  uint mask = 0;
  uint need = keff;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = lid; i < SIMDS * 256; i += THREADS) {
      atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < total; i += THREADS) {
      float v = i < (uint)t ? scores[row + i] : prev_scores[prev + i - t];
      uint key = mlxvs_key(v);
      if ((key & mask) == prefix) {
        atomic_fetch_add_explicit(&hist[sg * 256 + ((key >> shift) & 255u)], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      uint acc = 0;
      uint digit = 0;
      for (int d = 255; d >= 0; --d) {
        uint c = 0;
        for (uint s = 0; s < SIMDS; ++s) {
          c += atomic_load_explicit(&hist[s * 256 + d], memory_order_relaxed);
        }
        if (acc + c >= need) {
          digit = (uint)d;
          break;
        }
        acc += c;
      }
      shared_prefix = prefix | (digit << shift);
      shared_need = need - acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = shared_prefix;
    need = shared_need;
    mask |= (255u << shift);
  }

  // gather the winners: every key above the k-th, and the first `need` equal to it
  if (lid == 0) {
    atomic_store_explicit(&taken_above, 0u, memory_order_relaxed);
    atomic_store_explicit(&taken_equal, 0u, memory_order_relaxed);
  }
  for (uint i = lid; i < KMAX; i += THREADS) {
    sel_score[i] = -INFINITY;
    sel_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint above = keff - need;
  if (keff > 0) {
    for (uint i = lid; i < total; i += THREADS) {
      float v;
      int id;
      if (i < (uint)t) {
        v = scores[row + i];
        id = id_offset + (int)i;
      } else {
        v = prev_scores[prev + i - t];
        id = prev_ids[prev + i - t];
      }
      uint key = mlxvs_key(v);
      if (key > prefix) {
        uint at = atomic_fetch_add_explicit(&taken_above, 1u, memory_order_relaxed);
        sel_score[at] = v;
        sel_id[at] = id;
      } else if (key == prefix) {
        uint at = atomic_fetch_add_explicit(&taken_equal, 1u, memory_order_relaxed);
        if (at < need) {
          sel_score[above + at] = v;
          sel_id[above + at] = id;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // bitonic sort of the winners, best first
  uint size_pow2 = 1;
  while (size_pow2 < keff) {
    size_pow2 <<= 1;
  }
  for (uint size = 2; size <= size_pow2; size <<= 1) {
    for (uint stride = size >> 1; stride > 0; stride >>= 1) {
      for (uint i = lid; i < size_pow2; i += THREADS) {
        uint j = i ^ stride;
        if (j > i) {
          bool forward = (i & size) == 0;
          float si = sel_score[i];
          float sj = sel_score[j];
          int ii = sel_id[i];
          int ij = sel_id[j];
          bool in_order = mlxvs_before(si, ii, sj, ij);
          if (forward != in_order) {
            sel_score[i] = sj;
            sel_score[j] = si;
            sel_id[i] = ij;
            sel_id[j] = ii;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (uint i = lid; i < (uint)k; i += THREADS) {
    bool ok = i < keff;
    out_scores[(size_t)q * k + i] = ok ? sel_score[i] : -INFINITY;
    out_ids[(size_t)q * k + i] = ok ? sel_id[i] : -1;
  }
)";

} // namespace mlxvs::kernels
