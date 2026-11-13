# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.brute_force``: exact k-NN search on the GPU."""

import mlx.core as mx
import numpy as np

import mlxvs
from mlxvs.cuvs import _serialize as io
from mlxvs.cuvs import distance as _d
from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common

__all__ = ["Index", "build", "load", "save", "search"]

_SERIALIZATION_VERSION = 0
# metrics brute_force_knn computes directly
_FAST = {
    _d.L2Expanded: "l2",
    _d.L2SqrtExpanded: "l2",
    _d.InnerProduct: "inner_product",
    _d.CosineExpanded: "cosine",
}
# queries per block of the pairwise path
_BLOCK = 1024


class Index:
    """A brute-force index: the dataset and its metric."""

    def __init__(self):
        self.trained = False
        self._dataset = None
        self._metric = _d.L2Expanded
        self._metric_arg = 2.0

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return 0 if self._dataset is None else self._dataset.shape[0]

    def __repr__(self):
        return f"Index(type=BruteForce, metric={_d.DISTANCE_NAMES.get(self._metric)})"


@auto_sync_resources
def build(dataset, metric="sqeuclidean", metric_arg=2.0, resources=None):
    """Builds the index of the dataset (it keeps the rows), as cuVS's
    ``brute_force.build``.

    Args:
        dataset: Rows ``[n, dim]``, float32 or float16.
        metric (str): Any of ``cuvs.distance.SUPPORTED_DISTANCES``.
        metric_arg (float): ``p`` of ``"minkowski"``.

    Returns:
        Index
    """
    x = to_mlx(dataset)
    if x.ndim != 2:
        raise ValueError("dataset must be a matrix")
    idx = Index()
    idx._dataset = mx.contiguous(x)
    idx._metric = _d._metric_type(metric)
    idx._metric_arg = float(metric_arg)
    idx.trained = True
    mx.eval(idx._dataset)
    return idx


@auto_sync_resources
def search(
    index, queries, k, neighbors=None, distances=None, resources=None, prefilter=None
):
    """The k nearest rows of every query, as cuVS's ``brute_force.search``.

    Args:
        index (Index): Built by :func:`build`.
        queries: Rows ``[n_queries, dim]``, the dataset's dtype.
        k (int): Neighbors per query.
        neighbors: Optional ``int64 [n_queries, k]`` output.
        distances: Optional ``float32 [n_queries, k]`` output.
        prefilter (Prefilter): ``filters.from_bitmap`` / ``from_bitset``.

    Returns:
        tuple: ``(distances, neighbors)``, nearest first.
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")
    q = to_mlx(queries)
    data = index._dataset
    t = index._metric
    n, nq = data.shape[0], q.shape[0]
    mask = _common.filter_mask(prefilter, n, nq)
    if t in _FAST and mask is None and k <= n:
        ids, scores = mlxvs.brute_force_knn(data, q, k, _FAST[t])
        dist = _common.to_cuvs_distances(scores, t)
    else:
        if mask is not None and mask.dtype == mx.uint32:
            bits = np.unpackbits(np.asarray(mask).view(np.uint8), bitorder="little")[:n]
            mask = mx.array(bits.astype(np.bool_))
        min_close = _common.is_min_close(t)
        bad = float("inf") if min_close else -float("inf")
        ids_parts, dist_parts = [], []
        for i in range(0, nq, _BLOCK):
            d = _d._pairwise(q[i : i + _BLOCK], data, t, index._metric_arg)
            if mask is not None:
                m = mask if mask.ndim == 1 else mask[i : i + _BLOCK]
                d = mx.where(m, d, bad)
            dd, pos = _common.select_k(d, k, min_close)
            mx.eval(dd, pos)
            ids_parts.append(pos)
            dist_parts.append(dd)
        ids = mx.concatenate(ids_parts)
        dist = mx.concatenate(dist_parts)
        ids = mx.where(mx.isinf(dist), -1, ids.astype(mx.int64))
    return (
        _out(dist.astype(mx.float32), distances),
        _out(ids.astype(mx.int64), neighbors),
    )


@auto_sync_resources
def save(filename, index, include_dataset=True, resources=None):
    """Saves the index in cuVS's brute-force format (version 0), which
    cuVS's ``brute_force.load`` reads on NVIDIA GPUs."""
    data = np.asarray(index._dataset)
    t = index._metric
    with open(filename, "wb") as f:
        f.write(io.dtype_prefix(data.dtype, 6))
        io.write_scalar(f, _SERIALIZATION_VERSION, np.int32)
        io.write_scalar(f, data.shape[0], np.uint64)
        io.write_scalar(f, data.shape[1], np.uint64)
        io.write_scalar(f, t, np.int32)
        io.write_scalar(f, index._metric_arg, np.float32)
        io.write_scalar(f, bool(include_dataset), np.bool_)
        if include_dataset:
            io.write_array(f, data)
        has_norms = t in (_d.L2Expanded, _d.L2SqrtExpanded, _d.CosineExpanded)
        io.write_scalar(f, has_norms, np.bool_)
        if has_norms:
            x = data.astype(np.float32)
            norms = np.sum(x * x, axis=1)
            if t == _d.CosineExpanded:
                norms = np.sqrt(norms)
            io.write_array(f, norms.astype(np.float32))


@auto_sync_resources
def load(filename, resources=None):
    """Loads a brute-force index saved by :func:`save` or by cuVS."""
    with open(filename, "rb") as f:
        dtype = io.read_prefix(f)
        ver = io.read_scalar(f)
        if ver != _SERIALIZATION_VERSION:
            raise ValueError(
                f"serialization version mismatch, expected {_SERIALIZATION_VERSION}, got {ver}"
            )
        io.read_scalar(f)
        io.read_scalar(f)
        metric = io.read_scalar(f)
        metric_arg = io.read_scalar(f)
        if not io.read_scalar(f):
            raise ValueError("the file holds no dataset; brute force needs it")
        data = io.read_array(f).astype(dtype, copy=False)
    idx = Index()
    idx._dataset = mx.array(data)
    idx._metric = int(metric)
    idx._metric_arg = float(metric_arg)
    idx.trained = True
    return idx
