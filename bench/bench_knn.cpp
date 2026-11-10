// Copyright © 2026 mikepapadim

//
// bench_knn FILE.f32 DIM NQ N K: exact k-NN of the first NQ rows of a raw
// float32 file against the next N rows. Times (median of 5 warm runs): MLX's
// matmul alone; mlxvs::topk_merge on that score matrix; MLX's argpartition on
// it; and the whole mlxvs::brute_force_knn.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <vector>

#include "mlxvs/mlxvs.h"

namespace mx = mlx::core;

static double median_ms(const std::function<void()>& run, int reps = 5) {
  run(); // warm: compile, allocate
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    auto s = std::chrono::steady_clock::now();
    run();
    t.push_back(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - s)
            .count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: bench_knn FILE.f32 DIM NQ N K\n");
    return 2;
  }
  int d = std::atoi(argv[2]), nq = std::atoi(argv[3]), n = std::atoi(argv[4]),
      k = std::atoi(argv[5]);
  std::vector<float> rows((size_t)(nq + n) * d);
  std::ifstream in(argv[1], std::ios::binary);
  in.read(
      reinterpret_cast<char*>(rows.data()),
      (std::streamsize)(rows.size() * sizeof(float)));
  if (!in) {
    std::fprintf(
        stderr, "could not read %zu floats from %s\n", rows.size(), argv[1]);
    return 1;
  }
  mx::array q(rows.data(), {nq, d}, mx::float32);
  mx::array b(rows.data() + (size_t)nq * d, {n, d}, mx::float32);
  mx::eval(q, b);
  double gflop = 2.0 * nq * n * d / 1e9;
  std::printf(
      "%s: %d queries x %d rows x %d dims, k %d (device %s)\n",
      argv[1],
      nq,
      n,
      d,
      k,
      mx::default_device() == mx::Device::gpu ? "gpu" : "cpu");

  mx::array s = mx::matmul(q, mx::transpose(b));
  mx::eval(s);
  double mm = median_ms([&] { mx::eval(mx::matmul(q, mx::transpose(b))); });
  std::printf(
      "  MLX matmul (FP32)                      %8.1f ms  %5.2f TFLOP/s\n",
      mm,
      gflop / mm);

  auto ids0 = mx::full({nq, k}, -1, mx::int32);
  auto best0 = mx::full({nq, k}, -1e30f, mx::float32);
  mx::eval(ids0, best0);
  double sel = median_ms([&] {
    auto r = mlxvs::topk_merge(s, ids0, best0, 0, true, k);
    mx::eval(r.first, r.second);
  });
  std::printf("  mlxvs topk_merge (sorted top k)        %8.1f ms\n", sel);
  double part = median_ms([&] { mx::eval(mx::argpartition(-s, k - 1, 1)); });
  std::printf("  MLX argpartition (unsorted top k)      %8.1f ms\n", part);
  double full = median_ms([&] {
    auto r = mlxvs::brute_force_knn(b, q, k, mlxvs::Metric::InnerProduct);
    mx::eval(r.ids, r.distances);
  });
  std::printf("  mlxvs brute_force_knn (inner product)  %8.1f ms\n", full);
  return 0;
}
