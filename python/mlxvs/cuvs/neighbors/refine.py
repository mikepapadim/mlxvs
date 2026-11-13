# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.refine``: exact re-ranking of candidates."""

import mlx.core as mx
import numpy as np

import mlxvs
from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common

__all__ = ["refine"]


@auto_sync_resources
def refine(
    dataset,
    queries,
    candidates,
    k=None,
    metric="sqeuclidean",
    indices=None,
    distances=None,
    resources=None,
):
    """The k nearest of each query's candidates by exact distances, as cuVS's
    ``refine`` (typically after an IVF-PQ search of more than k).

    Args:
        dataset: The indexed rows ``[n, dim]`` (MLX, NumPy or a
          ``numpy.memmap``, of which only the candidates are read).
        queries: Rows ``[n_queries, dim]``.
        candidates: Candidate ids ``[n_queries, n_candidates]``.
        k (int): Neighbors to keep; the columns of ``indices`` when None.
        metric (str): ``"sqeuclidean"``, ``"euclidean"``, ``"inner_product"``
          or ``"cosine"``.

    Returns:
        tuple: ``(distances, indices)``, ``float32`` and ``int64``.
    """
    m, t = _common.index_metric(metric, "refine")
    if k is None:
        if indices is None:
            raise ValueError("k or indices must be given")
        k = np.asarray(indices).shape[1]
    cand = np.asarray(
        candidates.copy_to_host() if hasattr(candidates, "copy_to_host") else candidates
    )
    data = dataset
    if not (type(dataset).__module__ == "numpy"):
        data = to_mlx(dataset)
    ids, scores = mlxvs.refine(data, to_mlx(queries), cand, k, m)
    dist = _common.to_cuvs_distances(scores.astype(mx.float32), t)
    return _out(dist, distances), _out(ids.astype(mx.int64), indices)
