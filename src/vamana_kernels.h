// Copyright © 2026 mikepapadim

//
// Metal source of vamana_prune, compiled at run time by
// mlx::core::fast::metal_kernel.
#pragma once

namespace mlxvs::kernels {

/** The most candidates per node vamana_prune takes. */
constexpr int VAMANA_CANDIDATES_MAX = 256;

// Vamana's robust pruning with JVector's diversity rule
// (VamanaDiversityProvider.retainDiverse), one thread per node. The
// candidates (ids out of range, the node itself and repeats dropped) are
// ordered by their similarity to the node, best first (ties by id). Passes
// with alpha = 1.0, 1.2, ... up to the given alpha then keep, in that order, a
// candidate C not kept yet unless some kept S has sim(C, S) > alpha *
// sim(node, C), until R are kept. Similarities are JVector's: 1 / (1 + squared
// distance) for L2, (1 + dot) / 2 for inner product and cosine.
//
// template: C (candidates per node)
// inputs: ids [b, C] int, spc [b, C] float (similarity of the node to each
// candidate), scc [b, C, C] float (between candidates), alpha [1] float,
// params [b, n, first node of the chunk, R] int
// outputs: out [b, R] int: the kept candidates, most similar first, -1 filled
constexpr const char* VAMANA_SELECT_SOURCE = R"(
  uint t = thread_position_in_grid.x;
  int b = params[0];
  int n = params[1];
  int node = params[2] + (int)t;
  int R = params[3];
  if (t >= (uint)b) {
    return;
  }
  size_t row = (size_t)t * C;

  int order[C];
  int cnt = 0;
  for (int i = 0; i < C; ++i) {
    int v = ids[row + i];
    if (v < 0 || v >= n || v == node) {
      continue;
    }
    bool repeat = false;
    for (int j = 0; j < i && !repeat; ++j) {
      repeat = ids[row + j] == v;
    }
    if (repeat) {
      continue;
    }
    float s = spc[row + i];
    int at = cnt;
    while (at > 0) {
      int o = order[at - 1];
      float so = spc[row + o];
      if (so > s || (so == s && ids[row + o] < v)) {
        break;
      }
      order[at] = o;
      --at;
    }
    order[at] = i;
    ++cnt;
  }

  bool kept[C];
  for (int p = 0; p < C; ++p) {
    kept[p] = false;
  }
  int nkept = 0;
  float limit = alpha[0] + 1e-6f;
  for (float a = 1.0f; a <= limit && nkept < R; a += 0.2f) {
    for (int p = 0; p < cnt && nkept < R; ++p) {
      if (kept[p]) {
        continue;
      }
      int i = order[p];
      float bound = spc[row + i] * a;
      bool diverse = true;
      for (int q = 0; q < cnt && diverse; ++q) {
        if (kept[q]) {
          diverse = scc[(row + i) * C + order[q]] <= bound;
        }
      }
      if (diverse) {
        kept[p] = true;
        ++nkept;
      }
    }
  }

  int w = 0;
  for (int p = 0; p < cnt; ++p) {
    if (kept[p]) {
      out[(size_t)t * R + w++] = ids[row + order[p]];
    }
  }
  for (; w < R; ++w) {
    out[(size_t)t * R + w] = -1;
  }
)";

} // namespace mlxvs::kernels
