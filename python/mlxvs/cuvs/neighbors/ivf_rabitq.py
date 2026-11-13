# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.ivf_rabitq``: the IVF-RaBitQ index on Apple silicon (rows
stored as RaBitQ codes of their rotated residual). cuVS has it in C++
(``cuvs::neighbors::ivf_rabitq``); this module follows its parameters."""

import mlx.core as mx

import mlxvs
from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common
from mlxvs.cuvs.neighbors._ivf_common import (
    filtered_search,
    load_any_name,
    save_any_name,
)

__all__ = [
    "Index",
    "IndexParams",
    "SearchParams",
    "build",
    "extend",
    "load",
    "save",
    "search",
]


class IndexParams:
    """Parameters of :func:`build`, as cuVS's ``ivf_rabitq::index_params``.
    The streaming options are kept for compatibility (unified memory)."""

    def __init__(
        self,
        *,
        n_lists=1024,
        metric="sqeuclidean",
        metric_arg=2.0,
        bits_per_dim=3,
        kmeans_n_iters=20,
        max_train_points_per_cluster=256,
        fast_quantize_flag=True,
        streaming_batch_size=100000,
        force_streaming=False,
        add_data_on_build=True,
        conservative_memory_allocation=False,
    ):
        _common.index_metric(metric, "IVF-RaBitQ")
        self.n_lists = n_lists
        self.metric = metric
        self.metric_arg = metric_arg
        if not 1 <= bits_per_dim <= 9:
            raise ValueError("bits_per_dim must be in [1, 9]")
        self.bits_per_dim = bits_per_dim
        self.fast_quantize_flag = fast_quantize_flag
        self.streaming_batch_size = streaming_batch_size
        self.force_streaming = force_streaming
        self.kmeans_n_iters = kmeans_n_iters
        self.max_train_points_per_cluster = max_train_points_per_cluster
        self.add_data_on_build = add_data_on_build
        self.conservative_memory_allocation = conservative_memory_allocation


class SearchParams:
    """Parameters of :func:`search`, as cuVS's: ``mode`` ``"quant4"``
    (default), ``"quant8"``, ``"lut16"`` or ``"lut32"``."""

    def __init__(self, *, n_probes=20, mode="quant4"):
        mode = str(mode).lower()
        if mode not in ("lut16", "lut32", "quant4", "quant8"):
            raise ValueError('mode must be "lut16", "lut32", "quant4" or "quant8"')
        self.n_probes = n_probes
        self.mode = mode


class Index:
    """An IVF-RaBitQ index."""

    def __init__(self):
        self.trained = False
        self._index = None
        self._metric = 0

    @property
    def n_lists(self):
        return self._index.n_lists

    @property
    def dim(self):
        return self._index.dim

    @property
    def centers(self):
        return self._index.centers

    @property
    def metric(self):
        return _common.metric_name(self._metric)

    @property
    def mlxvs_index(self):
        """The :class:`mlxvs.IvfRabitqIndex` it wraps."""
        return self._index

    def __len__(self):
        return 0 if self._index is None else len(self._index)

    def __repr__(self):
        return f"Index(type=IVF-RaBitQ, metric={self.metric}, size={len(self)})"


def _wrap(index, t):
    idx = Index()
    idx._index, idx._metric, idx.trained = index, t, True
    return idx


@auto_sync_resources
def build(index_params, dataset, resources=None):
    """Builds the index, as cuVS's ``ivf_rabitq.build``."""
    p = index_params
    m, t = _common.index_metric(p.metric, "IVF-RaBitQ")
    x = to_mlx(dataset)
    index = mlxvs.IvfRabitqIndex.build(
        x,
        n_lists=min(p.n_lists, x.shape[0]),
        metric=m,
        bits_per_dim=p.bits_per_dim,
        kmeans_n_iters=p.kmeans_n_iters,
        max_train_points_per_cluster=p.max_train_points_per_cluster,
        fast_quantize=p.fast_quantize_flag,
    )
    if not p.add_data_on_build:
        i = index
        index = mlxvs.IvfRabitqIndex(
            i._centers,
            i._centers_rot,
            i._rotation,
            i._codes[:0],
            i._factors[:0],
            i._indices[:0],
            mx.zeros_like(i._offsets),
            m,
            i._bits,
            i._scale,
        )
    return _wrap(index, t)


@auto_sync_resources
def search(
    search_params,
    index,
    queries,
    k,
    neighbors=None,
    distances=None,
    resources=None,
    filter=None,
):
    """The k nearest rows of every query in the ``n_probes`` nearest lists,
    by RaBitQ's distance estimates, as cuVS's ``ivf_rabitq::search``.

    Returns:
        tuple: ``(distances, neighbors)``, ``float32`` and ``int64``.
    """
    q = to_mlx(queries)
    n_probes = min(search_params.n_probes, index.n_lists)
    mask = _common.filter_mask(filter, len(index), q.shape[0])

    def run(kk):
        return index._index.search(q, kk, n_probes=n_probes, mode=search_params.mode)

    ids, scores = run(k) if mask is None else filtered_search(run, mask, k, len(index))
    dist = _common.to_cuvs_distances(scores.astype(mx.float32), index._metric)
    return _out(dist, distances), _out(ids.astype(mx.int64), neighbors)


@auto_sync_resources
def extend(index, new_vectors, new_indices, resources=None):
    """Encodes rows into the lists of their nearest centers, as cuVS's
    ``ivf_rabitq.extend``."""
    ids = None if new_indices is None else to_mlx(new_indices).astype(mx.int32)
    index._index.extend(to_mlx(new_vectors), ids)
    return index


@auto_sync_resources
def save(filename, index, resources=None):
    """Saves the index (mlxvs's safetensors format)."""
    save_any_name(index._index, filename)


@auto_sync_resources
def load(filename, resources=None):
    """Loads an index saved by :func:`save`."""
    ci = load_any_name(mlxvs.IvfRabitqIndex, filename)
    t = {"l2": 0, "inner_product": 6, "cosine": 2}[ci.metric]
    return _wrap(ci, t)
