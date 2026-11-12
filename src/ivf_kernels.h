// Copyright © 2026 mikepapadim

//
// Metal sources of the IVF indexes, compiled at run time by
// mlx::core::fast::metal_kernel. They use SEARCH_HEADER's helpers.
#pragma once

namespace mlxvs::kernels {

/** Threads per threadgroup of the IVF scans. */
constexpr int IVF_THREADS = 256;

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

} // namespace mlxvs::kernels
