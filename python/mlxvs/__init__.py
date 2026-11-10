# Copyright © 2026 mikepapadim

"""GPU k-NN search and k-NN graph construction on Apple silicon, on MLX."""

from mlxvs._core import (
    Metric,
    brute_force_knn,
    kmeans_fit,
    kmeans_predict,
    nn_descent,
)

__all__ = [
    "Metric",
    "brute_force_knn",
    "kmeans_fit",
    "kmeans_predict",
    "nn_descent",
]
__version__ = "0.1.0"
