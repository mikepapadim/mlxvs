// Copyright © 2026 mikepapadim

// Graph helpers shared by the graph builders.
#pragma once

#include "mlx/mlx.h"

namespace mlxvs::detail {

namespace mx = mlx::core;

/**
 * Reverse edges of graph [n, D] (-1 entries skipped): for each node, the
 * sources of its rcap incoming edges of lowest rank (the edge's place in its
 * source's list), in (rank, source) order, -1 filled; int32 [n, rcap].
 */
mx::array reverse_lists(const mx::array& graph, int rcap);

} // namespace mlxvs::detail
