# Copyright © 2026 mikepapadim

"""Shared by the cuVS-compatible index modules: metrics and result layouts."""

import mlx.core as mx
import numpy as np

from mlxvs.cuvs import distance as _d

# cuVS metric name -> (mlxvs metric, cuVS DistanceType)
_INDEX_METRICS = {
    "sqeuclidean": ("l2", _d.L2Expanded),
    "l2": ("l2", _d.L2SqrtExpanded),
    "euclidean": ("l2", _d.L2SqrtExpanded),
    "inner_product": ("inner_product", _d.InnerProduct),
    "cosine": ("cosine", _d.CosineExpanded),
}


def index_metric(metric, what):
    """The mlxvs metric and cuVS DistanceType of a cuVS metric name."""
    if not isinstance(metric, str):
        metric = _d.DISTANCE_NAMES.get(int(metric), str(metric))
    if metric not in _INDEX_METRICS:
        raise ValueError(
            f"{what} supports the metrics {sorted(_INDEX_METRICS)}, not {metric!r}"
        )
    return _INDEX_METRICS[metric]


def metric_name(distance_type):
    if distance_type == _d.L2Expanded:
        return "sqeuclidean"
    if distance_type == _d.L2SqrtExpanded:
        return "euclidean"
    return _d.DISTANCE_NAMES[distance_type]


def to_cuvs_distances(scores, distance_type):
    """mlxvs scores (squared L2, inner product, cosine similarity) as the
    distances cuVS returns."""
    if distance_type in (_d.L2SqrtExpanded, _d.L2SqrtUnexpanded):
        return mx.sqrt(mx.maximum(scores, 0))
    if distance_type in (_d.L2Expanded, _d.L2Unexpanded):
        return mx.maximum(scores, 0)
    if distance_type == _d.CosineExpanded:
        return 1 - scores
    return scores


def is_min_close(distance_type):
    return distance_type != _d.InnerProduct


def filter_mask(filt, n_rows, n_queries):
    """A cuVS filter as an mlxvs filter: the bitset as it is (bit set: the row
    may be returned), a bitmap ``[n_queries * n_rows]`` bits as one
    word-padded bitset per query."""
    if filt is None or filt.type == "none":
        return None
    words = np.asarray(filt.array).astype(np.uint32).ravel()
    if filt.type == "bitset":
        return mx.array(words)
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")
    bits = bits[: n_queries * n_rows].reshape(n_queries, n_rows)
    return mx.array(bits.astype(np.bool_))


def select_k(dist, k, min_close=True):
    """The k smallest (largest) of every row: (values, positions)."""
    k = min(k, dist.shape[1])
    key = dist if min_close else -dist
    pos = (
        mx.argpartition(key, kth=k - 1, axis=1)[:, :k]
        if k < dist.shape[1]
        else mx.broadcast_to(mx.arange(dist.shape[1])[None], dist.shape)
    )
    vals = mx.take_along_axis(key, pos, axis=1)
    order = mx.argsort(vals, axis=1)
    pos = mx.take_along_axis(pos, order, axis=1)
    return mx.take_along_axis(dist, pos, axis=1), pos
