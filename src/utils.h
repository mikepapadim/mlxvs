// Copyright © 2026 mikepapadim

#pragma once

#include <stdexcept>
#include <string>

#include "mlx/mlx.h"

namespace mlxvs::detail {

namespace mx = mlx::core;

// The stream an operation runs on: s, or the default GPU stream. The kernels
// are Metal kernels, so other devices are rejected.
inline mx::Stream gpu_stream(mx::StreamOrDevice s, const char* op) {
  auto stream = mx::to_stream(s, mx::Device::gpu);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(
        std::string("[") + op + "] Only supported on the GPU.");
  }
  return stream;
}

} // namespace mlxvs::detail
