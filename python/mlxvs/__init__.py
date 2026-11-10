# Copyright © 2026 mikepapadim

"""GPU k-NN search and k-NN graph construction on Apple silicon, on MLX.

Every function takes rows as an ``mlx.core.array``, a NumPy array, or anything
``numpy.asarray`` accepts (lists, CPU torch tensors), in any real dtype, and
returns ``mlx.core.array`` results (``numpy.asarray`` converts them without a
copy). ``metric`` and ``algo`` take the enum or its name: ``"l2"``,
``"cosine"``, ``"inner_product"``; ``"nn_descent"``, ``"brute_force"``.
"""

import functools

import mlx.core as mx

from mlxvs import _core
from mlxvs._core import GraphBuild, Metric

__all__ = [
    "GraphBuild",
    "Metric",
    "NearestNeighbors",
    "all_neighbors",
    "brute_force_knn",
    "kmeans_fit",
    "kmeans_predict",
    "nn_descent",
]
__version__ = "0.1.0"


def __getattr__(name):
    # NearestNeighbors needs NumPy, which mlxvs otherwise does not
    if name == "NearestNeighbors":
        from mlxvs._neighbors import NearestNeighbors

        return NearestNeighbors
    raise AttributeError(f"module 'mlxvs' has no attribute {name!r}")


def _array(x):
    if isinstance(x, mx.array):
        return x
    try:
        return mx.array(x)
    except (TypeError, ValueError):
        import numpy as np

        return mx.array(np.asarray(x))


def _enum(kind, value):
    if isinstance(value, str):
        try:
            return getattr(kind, value)
        except AttributeError:
            names = ", ".join(repr(n) for n in kind.__members__)
            raise ValueError(f"unknown {kind.__name__} {value!r}; one of {names}")
    return value


def _metric(metric):
    return _enum(Metric, metric)


@functools.wraps(_core.brute_force_knn)
def brute_force_knn(base, queries, k, metric=Metric.l2, **kwargs):
    return _core.brute_force_knn(
        _array(base), _array(queries), k, _metric(metric), **kwargs
    )


@functools.wraps(_core.kmeans_fit)
def kmeans_fit(x, clusters, **kwargs):
    return _core.kmeans_fit(_array(x), clusters, **kwargs)


@functools.wraps(_core.kmeans_predict)
def kmeans_predict(x, centroids, m=1, **kwargs):
    return _core.kmeans_predict(_array(x), _array(centroids), m, **kwargs)


@functools.wraps(_core.nn_descent)
def nn_descent(rows, k, metric=Metric.l2, **kwargs):
    return _core.nn_descent(_array(rows), k, _metric(metric), **kwargs)


@functools.wraps(_core.all_neighbors)
def all_neighbors(rows, k, metric=Metric.l2, **kwargs):
    if "algo" in kwargs:
        kwargs["algo"] = _enum(GraphBuild, kwargs["algo"])
    return _core.all_neighbors(_array(rows), k, _metric(metric), **kwargs)


for _f in (brute_force_knn, kmeans_fit, kmeans_predict, nn_descent, all_neighbors):
    _f.__module__ = __name__
