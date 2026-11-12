// Copyright © 2026 mikepapadim

//
// Metal sources of the IVF indexes, compiled at run time by
// mlx::core::fast::metal_kernel. They use SEARCH_HEADER's helpers.
#pragma once

namespace mlxvs::kernels {

/** Threads per threadgroup of the IVF scans. */
constexpr int IVF_THREADS = 256;
/** The most rotated dimensions (pq_dim * pq_len) the IVF-PQ scan takes: the
 * query's residual is kept in threadgroup memory. */
constexpr int IVF_PQ_ROT_DIM_MAX = 2048;

// IVF-Flat scan. The (query, probed list) pairs are grouped by list; one
// threadgroup takes up to QT pairs of one list, loads their queries into
// threadgroup memory, and every thread scores rows of the list against all QT
// queries, so each row is read once per QT queries instead of once per query.
// Scores go to each query's candidate buffer at the list's offset.
//
// template: QT (queries per threadgroup), DCAP (query floats per query in
// threadgroup memory, a multiple of 4)
// inputs: data [n, d] (rows grouped by list), queries [nq, d] float, items
// [n_items, 3] int (list, first pair, pairs), pairs [nq * P] int (query *
// P + probe, sorted by list), offsets [lists + 1] int, slots [nq, P] int,
// params [d, P, W, metric] int (metric: 0 inner product, 1 l2)
// outputs: scores [nq, W] float (-inf filled; larger is closer), rows [nq, W]
// int (positions in data)
constexpr const char* IVF_FLAT_SCAN_SOURCE = R"(
  const uint THREADS = 256;
  uint item = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  int d = params[0];
  int P = params[1];
  int W = params[2];
  int metric = params[3];

  int list = items[item * 3];
  int first = items[item * 3 + 1];
  int count = items[item * 3 + 2];

  threadgroup float4 qv4[QT * DCAP / 4];
  threadgroup float* qv = (threadgroup float*)qv4;
  threadgroup int qid[QT];
  threadgroup int qslot[QT];
  for (int i = (int)lid; i < QT * DCAP; i += (int)THREADS) {
    qv[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = (int)lid; t < count; t += (int)THREADS) {
    int pair = pairs[first + t];
    qid[t] = pair / P;
    qslot[t] = slots[pair];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = 0; t < count; ++t) {
    size_t qrow = (size_t)qid[t] * (size_t)d;
    for (int j = (int)lid; j < DCAP; j += (int)THREADS) {
      qv[t * DCAP + j] = j < d ? queries[qrow + j] : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int start = offsets[list];
  int len = offsets[list + 1] - start;
  int d4 = d & ~3;
  for (int r = (int)lid; r < len; r += (int)THREADS) {
    const device auto* row = data + (size_t)(start + r) * (size_t)d;
    float acc[QT];
    for (int t = 0; t < QT; ++t) {
      acc[t] = 0.0f;
    }
    for (int j = 0; j < d4; j += 4) {
      float4 b = mlxvs_load4(row + j);
      for (int t = 0; t < QT; ++t) {
        float4 a = qv4[(t * DCAP + j) / 4];
        if (metric == 1) {
          float4 u = a - b;
          acc[t] -= dot(u, u);
        } else {
          acc[t] += dot(a, b);
        }
      }
    }
    for (int j = d4; j < d; ++j) {
      float b = (float)row[j];
      for (int t = 0; t < QT; ++t) {
        float a = qv[t * DCAP + j];
        acc[t] += metric == 1 ? -(a - b) * (a - b) : a * b;
      }
    }
    for (int t = 0; t < count; ++t) {
      size_t out = (size_t)qid[t] * (size_t)W + (size_t)(qslot[t] + r);
      scores[out] = acc[t];
      rows[out] = start + r;
    }
  }
)";

// IVF-PQ scan, one threadgroup per (probed list, query), as cuVS's
// ivf_pq search with a float32 lookup table. Rows are stored as PQ codes of
// their residual to the list's center. The query's residual (L2) or the query
// itself (inner product) is kept in threadgroup memory; the lookup table is
// built CH subspaces at a time (CH * NCODES floats), and every thread adds
// its rows' table entries for those subspaces to their scores in the query's
// candidate buffer (the first chunk writes them).
//
// template: PQ_DIM, PQ_LEN, NCODES, CH (subspaces per chunk), RCAP (rotated
// dimensions, a multiple of PQ_LEN)
// inputs: codes [n, PQ_DIM] uint8 (rows grouped by list), codebooks [PQ_DIM,
// NCODES, PQ_LEN] float, centers [lists, RCAP] float, queries [nq, RCAP]
// float, probes [nq, P] int, offsets [lists + 1] int, slots [nq, P] int,
// params [P, W, metric] int (metric: 0 inner product, 1 l2)
// outputs: scores [nq, W] float (-inf filled; larger is closer), rows [nq, W]
// int
constexpr const char* IVF_PQ_SCAN_SOURCE = R"(
  const uint THREADS = 256;
  uint p = threadgroup_position_in_grid.x;
  uint q = threadgroup_position_in_grid.y;
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid / 32;
  uint lane = lid % 32;
  int P = params[0];
  int W = params[1];
  int metric = params[2];

  threadgroup float qr[RCAP];
  threadgroup float lut[CH * NCODES];
  threadgroup float partial[THREADS / 32];
  threadgroup float base_score;

  int list = probes[(size_t)q * P + p];
  int start = offsets[list];
  int len = offsets[list + 1] - start;
  size_t out = (size_t)q * (size_t)W + (size_t)slots[(size_t)q * P + p];

  // the query's residual to the center (L2), or the query and its dot product
  // with the center (inner product: q . x = q . c + q . residual)
  float dotc = 0.0f;
  for (int j = (int)lid; j < RCAP; j += (int)THREADS) {
    float qj = queries[(size_t)q * RCAP + j];
    float cj = centers[(size_t)list * RCAP + j];
    qr[j] = metric == 1 ? qj - cj : qj;
    dotc += qj * cj;
  }
  dotc = simd_sum(dotc);
  if (lane == 0) {
    partial[sg] = dotc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < THREADS / 32; ++i) {
      sum += partial[i];
    }
    base_score = metric == 1 ? 0.0f : sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float base = base_score;

  // the scores of a thread's first RPT rows stay in registers across chunks;
  // rows beyond (long lists) accumulate in the candidate buffer
  const int RPT = 8;
  float acc[RPT];
  for (int i = 0; i < RPT; ++i) {
    acc[i] = base;
  }
  for (int m0 = 0; m0 < PQ_DIM; m0 += CH) {
    // the table for subspaces m0 .. m0 + CH: larger is closer
    for (int e = (int)lid; e < CH * NCODES; e += (int)THREADS) {
      int mm = e / NCODES;
      int c = e % NCODES;
      int m = m0 + mm;
      float v = 0.0f;
      if (m < PQ_DIM) {
        const device float* cb = codebooks + ((size_t)m * NCODES + c) * PQ_LEN;
        for (int j = 0; j < PQ_LEN; ++j) {
          float a = qr[m * PQ_LEN + j];
          v += metric == 1 ? -(a - cb[j]) * (a - cb[j]) : a * cb[j];
        }
      }
      lut[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = (int)lid, i = 0; r < len; r += (int)THREADS, ++i) {
      const device uchar* code = codes + (size_t)(start + r) * PQ_DIM + m0;
      float sc = 0.0f;
      for (int mm = 0; mm < CH && m0 + mm < PQ_DIM; ++mm) {
        sc += lut[mm * NCODES + (int)code[mm]];
      }
      if (i < RPT) {
        acc[i] += sc;
      } else if (m0 == 0) {
        scores[out + r] = base + sc;
      } else {
        scores[out + r] += sc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = (int)lid, i = 0; r < len; r += (int)THREADS, ++i) {
    if (i < RPT) {
      scores[out + r] = acc[i];
    }
    rows[out + r] = start + r;
  }
)";

} // namespace mlxvs::kernels
