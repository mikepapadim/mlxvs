// Copyright © 2026 mikepapadim

// VPQ datasets (cuVS's CAGRA-Q compression): a coarse center per row plus
// product-quantization codes of the residual, one codebook for all
// subspaces.

#include "mlxvs/mlxvs.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mlxvs {

VpqDataset vpq_build(
    const mx::array& rows_in,
    const VpqOptions& options,
    mx::StreamOrDevice s) {
  const char* op = "mlxvs::vpq_build";
  mx::StreamContext context(detail::gpu_stream(s, op));
  auto rows = detail::float_rows(rows_in, op);
  int n = rows.shape(0);
  int d = rows.shape(1);
  if (options.pq_bits < 4 || options.pq_bits > 8) {
    throw std::invalid_argument(
        "[mlxvs::vpq_build] pq_bits must be in [4, 8].");
  }
  int ncodes = 1 << options.pq_bits;
  // as cuVS's vpq defaults: pq_len 4, sqrt(n) coarse centers
  int pq_dim = options.pq_dim > 0 ? options.pq_dim : (d + 3) / 4;
  int pq_len = (d + pq_dim - 1) / pq_dim;
  int dpad = pq_dim * pq_len;
  int nvq = options.vq_n_centers > 0
      ? options.vq_n_centers
      : ((int)std::ceil(std::sqrt((double)n)) + 7) / 8 * 8;
  nvq = std::min(nvq, n);
  if (n < ncodes) {
    throw std::invalid_argument(
        "[mlxvs::vpq_build] needs at least 2^pq_bits rows.");
  }
  if (dpad != d) {
    rows = mx::pad(rows, {{0, 0}, {0, dpad - d}});
  }

  KMeansOptions vq;
  vq.max_iter = options.kmeans_n_iters;
  vq.seed = options.seed;
  double vq_fraction = options.vq_kmeans_trainset_fraction > 0
      ? options.vq_kmeans_trainset_fraction
      : std::min(1.0, 100.0 * nvq / n);
  vq.sample_rows =
      std::max(64 * nvq, (int)std::min<double>(n, vq_fraction * n));
  auto vq_centers = kmeans_fit(rows, nvq, vq);
  auto vq_ids = mx::reshape(kmeans_predict(rows, vq_centers, 1), {n});
  auto residuals = mx::subtract(rows, mx::take(vq_centers, vq_ids, 0));
  mx::eval(vq_centers, vq_ids, residuals);

  // one codebook for every subspace: the residuals as [n * pq_dim, pq_len]
  // samples, at most 256 per code (cuVS's max_train_points_per_pq_code)
  auto pieces = mx::reshape(residuals, {n * pq_dim, pq_len});
  KMeansOptions pq;
  pq.max_iter = options.kmeans_n_iters;
  pq.seed = options.seed + 1;
  double pq_fraction = options.pq_kmeans_trainset_fraction > 0
      ? options.pq_kmeans_trainset_fraction
      : std::min(1.0, 1000.0 * ncodes / n);
  long long train_rows = std::min<long long>(
      (long long)n * pq_dim,
      std::min<long long>(
          256LL * ncodes, (long long)(pq_fraction * n) * pq_dim));
  pq.sample_rows = (int)std::max<long long>(64LL * ncodes, train_rows);
  auto codebook = kmeans_fit(pieces, ncodes, pq);
  auto codes = mx::reshape(
      mx::astype(kmeans_predict(pieces, codebook, 1), mx::uint8), {n, pq_dim});
  mx::eval(codebook, codes);
  return {vq_centers, codebook, vq_ids, codes, d};
}

} // namespace mlxvs
