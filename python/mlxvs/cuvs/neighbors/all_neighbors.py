# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.all_neighbors``: the k-NN graph of a dataset,
optionally in batches of overlapping k-means clusters."""

import mlx.core as mx

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common
from mlxvs.cuvs.neighbors.nn_descent import knn_with_self

__all__ = ["AllNeighborsParams", "build"]


class AllNeighborsParams:
    """Parameters of :func:`build`, as cuVS's. ``algo``: ``"nn_descent"``,
    ``"brute_force"`` or ``"ivf_pq"`` (built by NN-Descent)."""

    def __init__(
        self,
        *,
        algo="nn_descent",
        overlap_factor=2,
        n_clusters=1,
        metric="sqeuclidean",
        ivf_pq_params=None,
        nn_descent_params=None,
    ):
        if algo not in ("nn_descent", "brute_force", "ivf_pq"):
            raise ValueError(f"Unknown algo '{algo}'")
        _common.index_metric(metric, "all_neighbors")
        self.algo = algo
        self.overlap_factor = overlap_factor
        self.n_clusters = n_clusters
        self.metric = metric
        self.ivf_pq_params = ivf_pq_params
        self.nn_descent_params = nn_descent_params


@auto_sync_resources
def build(
    dataset,
    k,
    params,
    *,
    indices=None,
    distances=None,
    core_distances=None,
    alpha=1.0,
    resources=None,
):
    """The k nearest rows of every row (itself first), as cuVS's
    ``all_neighbors.build``.

    With ``core_distances`` (and ``distances``), the graph is in mutual
    reachability space, as HDBSCAN uses: the core distance of a row is the
    distance to its k-th neighbor, and each edge's distance becomes
    ``max(core[i], core[j], alpha * d(i, j))``, rows re-sorted.

    Returns:
        ``indices`` (``int64 [n, k]``), or a tuple with ``distances`` and
        ``core_distances`` when those were given.
    """
    if not isinstance(params, AllNeighborsParams):
        raise TypeError("params must be an instance of AllNeighborsParams")
    if core_distances is not None and distances is None:
        raise ValueError("core_distances needs distances")
    m, t = _common.index_metric(params.metric, "all_neighbors")
    x = to_mlx(dataset)
    kw = {"algo": "brute_force" if params.algo == "brute_force" else "nn_descent"}
    if params.n_clusters > 1:
        kw["n_clusters"] = params.n_clusters
        kw["overlap_factor"] = params.overlap_factor
    ids, scores = knn_with_self(x, k, m, **kw)
    dist = _common.to_cuvs_distances(scores, t)
    core = None
    if core_distances is not None:
        core = dist[:, -1]
        mr = mx.maximum(mx.maximum(core[:, None], core[ids]), alpha * dist)
        order = mx.argsort(mr, axis=1)
        ids = mx.take_along_axis(ids, order, axis=1)
        dist = mx.take_along_axis(mr, order, axis=1)
    result = [_out(ids.astype(mx.int64), indices)]
    if distances is not None:
        result.append(_out(dist.astype(mx.float32), distances))
    if core_distances is not None:
        result.append(_out(core.astype(mx.float32), core_distances))
    return result[0] if len(result) == 1 else tuple(result)
