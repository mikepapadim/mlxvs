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

// IVF-SQ scan, as IVF-Flat's (pairs grouped by list, QT queries per
// threadgroup), over rows stored as cuVS's IVF-SQ codes: one byte per
// dimension of the residual to the list's center, decoded as
// center + vmin + delta * code. Per query the threadgroup holds
//   L2:            a = q - center - vmin  (distance: |a - delta * code|^2)
//   inner product: a = q * delta, base = q . (center + vmin)
//   cosine:        as inner product, divided by the decoded row's norm
// (the base of every (query, probe) pair comes in qbase). Threadgroup memory:
// (QT + 1) * DCAP floats.
//
// template: QT, DCAP (query floats per query, a multiple of 4)
// inputs: codes [n, d] uint8 (rows grouped by list), queries [nq, d] float,
// centers [lists, d] float, vmin [d], delta [d] float, qbase [nq * P] float,
// items [n_items, 3] int, pairs [nq * P] int, offsets [lists + 1] int, slots
// [nq, P] int, params [d, P, W, metric] int (metric: 0 inner product, 1 l2,
// 2 cosine)
// outputs: scores [nq, W] float (-inf filled; larger is closer), rows [nq, W]
// int
constexpr const char* IVF_SQ_SCAN_SOURCE = R"(
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
  threadgroup float4 dl4[DCAP / 4];
  threadgroup float* dl = (threadgroup float*)dl4;
  threadgroup int qid[QT];
  threadgroup int qslot[QT];
  threadgroup float qb[QT];
  size_t crow = (size_t)list * (size_t)d;
  for (int j = (int)lid; j < DCAP; j += (int)THREADS) {
    dl[j] = j < d ? delta[j] : 0.0f;
  }
  for (int i = (int)lid; i < QT * DCAP; i += (int)THREADS) {
    qv[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = (int)lid; t < count; t += (int)THREADS) {
    int pair = pairs[first + t];
    qid[t] = pair / P;
    qslot[t] = slots[pair];
    qb[t] = qbase[pair];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int t = 0; t < count; ++t) {
    size_t qrow = (size_t)qid[t] * (size_t)d;
    for (int j = (int)lid; j < d; j += (int)THREADS) {
      float q = queries[qrow + j];
      qv[t * DCAP + j] =
          metric == 1 ? q - centers[crow + j] - vmin[j] : q * dl[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int start = offsets[list];
  int len = offsets[list + 1] - start;
  int d4 = d & ~3;
  for (int r = (int)lid; r < len; r += (int)THREADS) {
    const device auto* row = codes + (size_t)(start + r) * (size_t)d;
    float acc[QT];
    for (int t = 0; t < QT; ++t) {
      acc[t] = 0.0f;
    }
    float nrm = 0.0f;
    for (int j = 0; j < d4; j += 4) {
      float4 b = mlxvs_load4(row + j);
      if (metric == 1) {
        float4 db = dl4[j / 4] * b;
        for (int t = 0; t < QT; ++t) {
          float4 u = qv4[(t * DCAP + j) / 4] - db;
          acc[t] -= dot(u, u);
        }
      } else {
        for (int t = 0; t < QT; ++t) {
          acc[t] += dot(qv4[(t * DCAP + j) / 4], b);
        }
        if (metric == 2) {
          // centers and vmin may be bound in the constant address space
          float4 x = float4(
                         centers[crow + j] + vmin[j],
                         centers[crow + j + 1] + vmin[j + 1],
                         centers[crow + j + 2] + vmin[j + 2],
                         centers[crow + j + 3] + vmin[j + 3]) +
              dl4[j / 4] * b;
          nrm += dot(x, x);
        }
      }
    }
    for (int j = d4; j < d; ++j) {
      float b = (float)row[j];
      if (metric == 1) {
        for (int t = 0; t < QT; ++t) {
          float u = qv[t * DCAP + j] - dl[j] * b;
          acc[t] -= u * u;
        }
      } else {
        for (int t = 0; t < QT; ++t) {
          acc[t] += qv[t * DCAP + j] * b;
        }
        if (metric == 2) {
          float x = centers[crow + j] + vmin[j] + dl[j] * b;
          nrm += x * x;
        }
      }
    }
    float inv = metric == 2 ? rsqrt(max(nrm, 1e-30f)) : 1.0f;
    for (int t = 0; t < count; ++t) {
      size_t out = (size_t)qid[t] * (size_t)W + (size_t)(qslot[t] + r);
      scores[out] = metric == 1 ? acc[t] : (acc[t] + qb[t]) * inv;
      rows[out] = start + r;
    }
  }
)";

// IVF-RaBitQ (RaBitQ and Extended RaBitQ, as cuVS's ivf_rabitq), searched in
// cuVS's two stages over pairs grouped by list as IVF-Flat's. Rows are stored
// as B-bit codes u of their rotated residual r' = P (x - c), in B bit planes
// of DP / 32 words (plane B - 1 holds the sign bits), and four factors:
//   |r|^2, f = |r|^2 / <u - cB, r'>, f1 = |r|^2 / <s - 1/2, r'> (s the sign
//   bits) and eps (|<r', y> - f1 <s - 1/2, y>| <= eps |y| w.h.p.)
// with cB = (2^B - 1) / 2. For y = P (q - c) (L2) or y = P q (inner product):
//   L2:            |q - x|^2 ~ |r|^2 + |y|^2 - 2 <r', y>
//   inner product: <q, x>    ~ <q, c> + <r', y>
// Per row, stage 1 estimates <r', y> from the sign bits alone, with y
// quantized to BQ bits (cuVS's QUANT4 / QUANT8: popcounts of bit planes) or,
// BQ = 0, as floats (the LUT modes), and its upper bound; rows whose bound
// reaches the query's threshold tau (as cuVS's running top-k threshold: the
// k-th best score of a search of the query's nearest list) are scored in
// stage 2 from all B planes and the float y; the others stay -inf.
//
// template: QT, DP (rotated dim, a multiple of 64), B, BQ
// inputs: codes [n, B, DP / 32] uint32, factors [n, 4] float, rq [nq, DP]
// float (P q), crot [lists, DP] float (P c), items, pairs, offsets, slots,
// params [P, W, metric] int (metric: 0 inner product, 1 l2), tau [nq]
// outputs: scores [nq, W] float (-inf filled; larger is closer), rows [nq, W]
constexpr const char* IVF_RABITQ_PROLOGUE = R"(
  const uint THREADS = 256;
  const int WORDS = DP / 32;
  uint item = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint sg = lid / 32;
  uint lane = lid % 32;
  int P = params[0];
  int W = params[1];
  int metric = params[2];

  int list = items[item * 3];
  int first = items[item * 3 + 1];
  int count = items[item * 3 + 2];

  threadgroup float y[QT * DP];
  threadgroup uint yq[QT * (BQ > 0 ? BQ : 1) * WORDS];
  threadgroup float red[THREADS / 32][5];
  threadgroup int qid[QT];
  threadgroup int qslot[QT];
  threadgroup float t_base[QT];
  threadgroup float t_sum[QT];
  threadgroup float t_norm[QT];
  threadgroup float t_vmin[QT];
  threadgroup float t_step[QT];

  for (int t = (int)lid; t < count; t += (int)THREADS) {
    int pair = pairs[first + t];
    qid[t] = pair / P;
    qslot[t] = slots[pair];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  size_t crow = (size_t)list * DP;
  for (int t = 0; t < count; ++t) {
    size_t qrow = (size_t)qid[t] * DP;
    float s2 = 0.0f, s1 = 0.0f, lo = INFINITY, hi = -INFINITY, cdot = 0.0f;
    for (int j = (int)lid; j < DP; j += (int)THREADS) {
      float q = rq[qrow + j];
      float c = crot[crow + j];
      float v = metric == 1 ? q - c : q;
      y[t * DP + j] = v;
      s2 += v * v;
      s1 += v;
      cdot += q * c;
      lo = min(lo, v);
      hi = max(hi, v);
    }
    s2 = simd_sum(s2);
    s1 = simd_sum(s1);
    cdot = simd_sum(cdot);
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (lane == 0) {
      red[sg][0] = s2;
      red[sg][1] = s1;
      red[sg][2] = lo;
      red[sg][3] = hi;
      red[sg][4] = cdot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float a = 0.0f, b = 0.0f, mn = INFINITY, mxv = -INFINITY, cd = 0.0f;
      for (uint i = 0; i < THREADS / 32; ++i) {
        a += red[i][0];
        b += red[i][1];
        mn = min(mn, red[i][2]);
        mxv = max(mxv, red[i][3]);
        cd += red[i][4];
      }
      t_base[t] = metric == 1 ? a : cd;
      t_sum[t] = b;
      t_norm[t] = sqrt(a);
      t_vmin[t] = mn;
      t_step[t] = BQ > 0 && mxv > mn ? (mxv - mn) / (float)((1 << BQ) - 1) : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  int start = offsets[list];
  int len = offsets[list + 1] - start;
)";

constexpr const char* IVF_RABITQ_SCAN_SOURCE = R"(
  if (BQ > 0) {
    // the quantized query's bit planes: word w of plane k of query t
    for (int e = (int)lid; e < count * BQ * WORDS; e += (int)THREADS) {
      int t = e / (BQ * WORDS);
      int k = (e / WORDS) % BQ;
      int w = e % WORDS;
      float vmin = t_vmin[t];
      float inv = 1.0f / t_step[t];
      uint word = 0;
      for (int i = 0; i < 32; ++i) {
        float v = y[t * DP + w * 32 + i];
        uint qv = (uint)min(max(rint((v - vmin) * inv), 0.0f), (float)((1 << BQ) - 1));
        word |= ((qv >> k) & 1u) << i;
      }
      yq[(t * BQ + k) * WORDS + w] = word;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float cB = 0.5f * (float)((1 << B) - 1);
  // the query rounding's error, three standard deviations
  const float qerr = BQ > 0 ? 0.433f * sqrt((float)DP) : 0.0f;
  for (int r = (int)lid; r < len; r += (int)THREADS) {
    size_t row = (size_t)(start + r);
    const device uint* code = codes + row * (size_t)(B * WORDS);
    const device uint* sign = code + (B - 1) * WORDS;
    // stage 1: the sign bits' estimate of <r', y>, for every query
    float acc[QT];
    uint iacc[QT];
    for (int t = 0; t < QT; ++t) {
      acc[t] = 0.0f;
      iacc[t] = 0u;
    }
    uint ones = 0u;
    for (int w = 0; w < WORDS; ++w) {
      uint c = sign[w];
      if (BQ > 0) {
        ones += popcount(c);
        for (int t = 0; t < QT; ++t) {
          uint s = 0u;
          for (int k = 0; k < BQ; ++k) {
            s += popcount(c & yq[(t * BQ + k) * WORDS + w]) << k;
          }
          iacc[t] += s;
        }
      } else {
        for (int t = 0; t < QT; ++t) {
          uint m = c;
          float part = 0.0f;
          while (m != 0u) {
            part += y[t * DP + w * 32 + ctz(m)];
            m &= m - 1u;
          }
          acc[t] += part;
        }
      }
    }
    float r2 = factors[row * 4];
    float f = factors[row * 4 + 1];
    float f1 = factors[row * 4 + 2];
    float eps = factors[row * 4 + 3];
    for (int t = 0; t < count; ++t) {
      size_t out = (size_t)qid[t] * (size_t)W + (size_t)(qslot[t] + r);
      rows[out] = start + r;
      float dot1 = BQ > 0 ? t_step[t] * (float)iacc[t] + t_vmin[t] * (float)ones : acc[t];
      float est1 = f1 * (dot1 - 0.5f * t_sum[t]);
      float err = eps * t_norm[t] + (BQ > 0 ? f1 * qerr * t_step[t] : 0.0f);
      float ub = metric == 1 ? -(r2 + t_base[t] - 2.0f * est1) + 2.0f * err
                             : t_base[t] + est1 + err;
      if (ub < tau[qid[t]]) {
        continue;
      }
      // stage 2: from all B bits and the float query
      float dot = 0.0f;
      if (B == 1) {
        dot = dot1;
        if (BQ > 0) {
          dot = 0.0f;
          for (int w = 0; w < WORDS; ++w) {
            uint m = sign[w];
            while (m != 0u) {
              dot += y[t * DP + w * 32 + ctz(m)];
              m &= m - 1u;
            }
          }
        }
      } else {
        for (int b = 0; b < B; ++b) {
          float part = 0.0f;
          for (int w = 0; w < WORDS; ++w) {
            uint m = code[b * WORDS + w];
            while (m != 0u) {
              part += y[t * DP + w * 32 + ctz(m)];
              m &= m - 1u;
            }
          }
          dot += (float)(1 << b) * part;
        }
      }
      float est = f * (dot - cB * t_sum[t]);
      scores[out] = metric == 1 ? -(r2 + t_base[t] - 2.0f * est) : t_base[t] + est;
    }
  }
)";

} // namespace mlxvs::kernels
