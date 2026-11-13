# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.nn_descent``: the approximate k-NN graph of a
dataset, by NN-Descent on the GPU."""

import mlx.core as mx

import mlxvs
from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common

__all__ = ["Index", "IndexParams", "build"]


class IndexParams:
    """Parameters of :func:`build`, as cuVS's (None: cuVS's default)."""

    def __init__(
        self,
        *,
        metric=None,
        metric_arg=None,
        graph_degree=None,
        intermediate_graph_degree=None,
        max_iterations=None,
        termination_threshold=None,
        return_distances=None,
        dist_comp_dtype="auto",
    ):
        self.metric = "sqeuclidean" if metric is None else metric
        _common.index_metric(self.metric, "NN-Descent")
        self.metric_arg = 2.0 if metric_arg is None else metric_arg
        self.graph_degree = 64 if graph_degree is None else graph_degree
        self.intermediate_graph_degree = (
            128 if intermediate_graph_degree is None else intermediate_graph_degree
        )
        self.max_iterations = 20 if max_iterations is None else max_iterations
        self.termination_threshold = (
            0.0001 if termination_threshold is None else termination_threshold
        )
        self.return_distances = True if return_distances is None else return_distances
        self.dist_comp_dtype = dist_comp_dtype


class Index:
    """The k-NN graph: ``graph`` ``uint32 [n, graph_degree]`` and, with
    ``return_distances``, ``distances`` ``float32 [n, graph_degree]``. Each
    row lists the row itself first, as cuVS's."""

    def __init__(self, graph=None, distances=None):
        self.trained = graph is not None
        self._graph = graph
        self._distances = distances

    @property
    def graph(self):
        return self._graph

    @property
    def distances(self):
        return self._distances


def knn_with_self(x, k, m, **kwargs):
    """mlxvs's k-NN graph (which leaves each row out of its own list) with
    the row itself first at distance 0, as cuVS lists it."""
    n = x.shape[0]
    ids, scores = (
        mlxvs.all_neighbors(x, max(1, k - 1), m, **kwargs)
        if k > 1
        else (
            mx.zeros((n, 0), mx.int32),
            mx.zeros((n, 0), mx.float32),
        )
    )
    self_ids = mx.arange(n, dtype=mx.int32)[:, None]
    if m == "l2":
        self_score = mx.zeros((n, 1), mx.float32)
    elif m == "cosine":
        self_score = mx.ones((n, 1), mx.float32)
    else:
        xf = x.astype(mx.float32)
        self_score = mx.sum(xf * xf, axis=1, keepdims=True)
    ids = mx.concatenate([self_ids, ids.astype(mx.int32)], axis=1)
    scores = mx.concatenate([self_score, scores.astype(mx.float32)], axis=1)
    return ids, scores


@auto_sync_resources
def build(index_params, dataset, graph=None, resources=None):
    """The k-NN graph of the dataset, as cuVS's ``nn_descent.build``."""
    p = index_params
    m, t = _common.index_metric(p.metric, "NN-Descent")
    x = to_mlx(dataset)
    ids, scores = knn_with_self(
        x,
        p.graph_degree,
        m,
        algo="nn_descent",
        max_iterations=p.max_iterations,
        termination_threshold=p.termination_threshold,
    )
    g = _out(ids.astype(mx.uint32), graph)
    d = _out(_common.to_cuvs_distances(scores, t)) if p.return_distances else None
    return Index(g, d)
