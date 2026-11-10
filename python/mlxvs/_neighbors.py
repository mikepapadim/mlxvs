# Copyright © 2026 mikepapadim

"""A scikit-learn style front end: NearestNeighbors on the GPU."""

import mlx.core as mx
import numpy as np

from mlxvs import _core

_METRICS = {
    "euclidean": _core.Metric.l2,
    "sqeuclidean": _core.Metric.l2,
    "l2": _core.Metric.l2,
    "cosine": _core.Metric.cosine,
}


class NearestNeighbors:
    """Exact or approximate nearest neighbors with scikit-learn's interface.

    A drop-in for ``sklearn.neighbors.NearestNeighbors`` where it overlaps:
    ``fit(X)``, then ``kneighbors(Q)`` returns NumPy ``(distances, indices)``
    in scikit-learn's units (Euclidean distance, not squared; cosine distance
    ``1 - cos``), nearest first.

    Args:
        n_neighbors (int): Neighbors per query.
        metric (str): ``"euclidean"``, ``"sqeuclidean"`` or ``"cosine"``.
        algorithm (str): ``"brute"`` (exact) or ``"nn_descent"``
          (approximate). Only ``kneighbors()`` without queries, the neighbors
          of the fitted rows themselves, uses NN-Descent; queries are always
          answered exactly.
        n_clusters (int): With ``"nn_descent"``, build the graph in this many
          overlapping k-means clusters (see ``mlxvs.all_neighbors``).
    """

    def __init__(
        self, n_neighbors=5, *, metric="euclidean", algorithm="brute", n_clusters=1
    ):
        if metric not in _METRICS:
            names = ", ".join(repr(m) for m in _METRICS)
            raise ValueError(f"unknown metric {metric!r}; one of {names}")
        if algorithm not in ("brute", "nn_descent"):
            raise ValueError("algorithm must be 'brute' or 'nn_descent'")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        """Keep the rows ``X [n, d]`` on the GPU."""
        self._fit_X = X if isinstance(X, mx.array) else mx.array(np.asarray(X))
        if self._fit_X.ndim != 2:
            raise ValueError("X must be 2-D [n, d]")
        mx.eval(self._fit_X)
        self.n_samples_fit_ = self._fit_X.shape[0]
        self.n_features_in_ = self._fit_X.shape[1]
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """The neighbors of each row of ``X``, or of each fitted row.

        Without ``X`` a row is not its own neighbor, as in scikit-learn.

        Returns:
            ``(distances, indices)`` NumPy arrays ``[nq, n_neighbors]``, or
            only the indices with ``return_distance=False``.
        """
        if not hasattr(self, "_fit_X"):
            raise RuntimeError("call fit(X) first")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        limit = self.n_samples_fit_ - (1 if X is None else 0)
        if not 1 <= k <= limit:
            raise ValueError(f"n_neighbors must be in [1, {limit}], got {k}")
        metric = _METRICS[self.metric]
        if X is None:
            algo = (
                _core.GraphBuild.nn_descent
                if self.algorithm == "nn_descent"
                else _core.GraphBuild.brute_force
            )
            ids, dist = _core.all_neighbors(
                self._fit_X, k, metric, algo=algo, n_clusters=self.n_clusters
            )
        else:
            queries = X if isinstance(X, mx.array) else mx.array(np.asarray(X))
            ids, dist = _core.brute_force_knn(self._fit_X, queries, k, metric)
        ids = np.asarray(ids).astype(np.int64)
        if not return_distance:
            return ids
        dist = np.array(dist)
        if self.metric == "euclidean":
            dist = np.sqrt(dist)
        elif self.metric == "cosine":
            dist = 1.0 - dist
        return dist, ids
