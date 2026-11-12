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
    "CagraIndex",
    "GraphBuild",
    "Metric",
    "NearestNeighbors",
    "all_neighbors",
    "brute_force_knn",
    "cagra_build",
    "cagra_optimize",
    "graph_search",
    "kmeans_fit",
    "kmeans_predict",
    "nn_descent",
]
__version__ = "0.2.1"


def __getattr__(name):
    # NearestNeighbors needs NumPy, which mlxvs otherwise does not
    if name == "NearestNeighbors":
        from mlxvs._neighbors import NearestNeighbors

        return NearestNeighbors
    raise AttributeError(f"module 'mlxvs' has no attribute {name!r}")


def _array(x):
    if isinstance(x, mx.array):
        return x
    if type(x).__module__ == "numpy" and not x.flags.c_contiguous:
        # MLX keeps a NumPy array's strides; a kernel would then make the rows
        # contiguous again on every call
        import numpy as np

        x = np.ascontiguousarray(x)
    try:
        return mx.array(x)
    except (TypeError, ValueError):
        import numpy as np

        return mx.array(np.asarray(x))


def _host_rows(x):
    """A float32 C-contiguous NumPy array (or memmap): read where it is."""
    if type(x).__module__ != "numpy":
        return False
    return x.ndim == 2 and x.dtype == "float32" and x.flags.c_contiguous


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


@functools.wraps(_core.cagra_optimize)
def cagra_optimize(knn_graph, graph_degree, **kwargs):
    return _core.cagra_optimize(_array(knn_graph), graph_degree, **kwargs)


def cagra_build(
    rows, graph_degree=32, metric=Metric.l2, *, intermediate_graph_degree=64, **kwargs
):
    """A search graph of the rows, as cuVS's ``cagra.build``.

    The k-NN graph of the rows (``all_neighbors`` with
    ``intermediate_graph_degree`` neighbors; its keyword arguments, such as
    ``algo``, ``n_clusters`` or ``verbose``, pass through), pruned to
    ``graph_degree`` by ``cagra_optimize``. Rows may be a NumPy array or
    ``numpy.memmap`` larger than memory, as for ``all_neighbors``.

    Args:
        rows (array): Rows ``[n, d]``, any real dtype.
        graph_degree (int): Degree of the search graph.
        metric (Metric): ``inner_product``, ``l2`` or ``cosine``.
        intermediate_graph_degree (int): Degree of the k-NN graph pruned
          (at most 128 with NN-Descent).

    Returns:
        array: ``int32 [n, graph_degree]``, the ``graph`` of ``graph_search``.
    """
    if graph_degree > intermediate_graph_degree:
        raise ValueError("graph_degree must be at most intermediate_graph_degree")
    knn, _ = all_neighbors(rows, intermediate_graph_degree, metric, **kwargs)
    return cagra_optimize(knn, graph_degree)


@functools.wraps(_core.graph_search)
def graph_search(base, graph, queries, k, metric=Metric.l2, **kwargs):
    return _core.graph_search(
        _array(base), _array(graph), _array(queries), k, _metric(metric), **kwargs
    )


@functools.wraps(_core.all_neighbors)
def all_neighbors(rows, k, metric=Metric.l2, **kwargs):
    if "algo" in kwargs:
        kwargs["algo"] = _enum(GraphBuild, kwargs["algo"])
    if _host_rows(rows):
        return _core.all_neighbors_host(rows, k, _metric(metric), **kwargs)
    return _core.all_neighbors(_array(rows), k, _metric(metric), **kwargs)


for _f in (
    brute_force_knn,
    kmeans_fit,
    kmeans_predict,
    nn_descent,
    all_neighbors,
    graph_search,
    cagra_optimize,
):
    _f.__module__ = __name__

from mlxvs._index import CagraIndex

CagraIndex.__module__ = __name__
