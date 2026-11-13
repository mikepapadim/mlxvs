# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.ivf_flat``: the IVF-Flat index on Apple silicon."""

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
    """Parameters of :func:`build`, as cuVS's. ``adaptive_centers`` and
    ``conservative_memory_allocation`` are kept for compatibility."""

    def __init__(
        self,
        *,
        n_lists=1024,
        metric="sqeuclidean",
        metric_arg=2.0,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=0.5,
        adaptive_centers=False,
        add_data_on_build=True,
        conservative_memory_allocation=False,
    ):
        _common.index_metric(metric, "IVF-Flat")
        self.n_lists = n_lists
        self.metric = metric
        self.metric_arg = metric_arg
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.adaptive_centers = adaptive_centers
        self.add_data_on_build = add_data_on_build
        self.conservative_memory_allocation = conservative_memory_allocation


class SearchParams:
    """Parameters of :func:`search`, as cuVS's."""

    def __init__(self, *, n_probes=20):
        self.n_probes = n_probes


class Index:
    """An IVF-Flat index."""

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
        """The :class:`mlxvs.IvfFlatIndex` it wraps."""
        return self._index

    def __len__(self):
        return 0 if self._index is None else len(self._index)

    def __repr__(self):
        return f"Index(type=IVF-FLAT, metric={self.metric}, size={len(self)})"


def _wrap(index, t):
    idx = Index()
    idx._index, idx._metric, idx.trained = index, t, True
    return idx


@auto_sync_resources
def build(index_params, dataset, resources=None):
    """Builds the index, as cuVS's ``ivf_flat.build``: k-means lists of the
    rows, which are added unless ``add_data_on_build=False``."""
    p = index_params
    m, t = _common.index_metric(p.metric, "IVF-Flat")
    x = to_mlx(dataset)
    index = mlxvs.IvfFlatIndex.build(
        x,
        n_lists=min(p.n_lists, x.shape[0]),
        metric=m,
        kmeans_n_iters=p.kmeans_n_iters,
        kmeans_trainset_fraction=p.kmeans_trainset_fraction,
    )
    if not p.add_data_on_build:
        index = mlxvs.IvfFlatIndex(
            index._centers,
            index._data[:0],
            index._indices[:0],
            mx.zeros_like(index._offsets),
            m,
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
    as cuVS's ``ivf_flat.search``.

    Returns:
        tuple: ``(distances, neighbors)``, ``float32`` and ``int64 [n_queries,
        k]``, nearest first.
    """
    q = to_mlx(queries)
    n_probes = min(search_params.n_probes, index.n_lists)
    mask = _common.filter_mask(filter, len(index), q.shape[0])

    def run(kk):
        return index._index.search(q, kk, n_probes=n_probes)

    ids, scores = run(k) if mask is None else filtered_search(run, mask, k, len(index))
    dist = _common.to_cuvs_distances(scores.astype(mx.float32), index._metric)
    return _out(dist, distances), _out(ids.astype(mx.int64), neighbors)


@auto_sync_resources
def extend(index, new_vectors, new_indices, resources=None):
    """Adds rows to the lists of their nearest centers, with their ids, as
    cuVS's ``ivf_flat.extend``."""
    ids = None if new_indices is None else to_mlx(new_indices).astype(mx.int32)
    index._index.extend(to_mlx(new_vectors), ids)
    return index


@auto_sync_resources
def save(filename, index, include_dataset=True, resources=None):
    """Saves the index (mlxvs's safetensors format)."""
    save_any_name(index._index, filename)


@auto_sync_resources
def load(filename, resources=None):
    """Loads an index saved by :func:`save`."""
    ci = load_any_name(mlxvs.IvfFlatIndex, filename)
    t = {"l2": 0, "inner_product": 6, "cosine": 2}[ci.metric]
    return _wrap(ci, t)
