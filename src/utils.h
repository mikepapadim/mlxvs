// Copyright © 2026 mikepapadim

#pragma once

#include <stdexcept>
#include <string>

#include "mlx/mlx.h"

namespace mlxvs::detail {

namespace mx = mlx::core;

// The stream an operation runs on: s, or the default GPU stream. The kernels
// are Metal kernels, so other devices are rejected.
// Rows [n, d] as float32: any real dtype (float16, bfloat16, float64, integers)
// is converted; other shapes and dtypes are rejected.
inline mx::array
float_rows(const mx::array& x, const char* op, const char* name = "rows") {
  if (x.ndim() != 2) {
    throw std::invalid_argument(
        std::string("[") + op + "] " + name + " must be 2-D [n, d].");
  }
  if (x.dtype() == mx::bool_ || x.dtype() == mx::complex64) {
    throw std::invalid_argument(
        std::string("[") + op + "] " + name + " must be real numbers.");
  }
  return x.dtype() == mx::float32 ? x : mx::astype(x, mx::float32);
}

inline mx::Stream gpu_stream(mx::StreamOrDevice s, const char* op) {
  auto stream = mx::to_stream(s, mx::Device::gpu);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(
        std::string("[") + op + "] Only supported on the GPU.");
  }
  return stream;
}

} // namespace mlxvs::detail
