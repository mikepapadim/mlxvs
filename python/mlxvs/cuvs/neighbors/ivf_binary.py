# Copyright © 2026 mikepapadim

"""cuVS's IVF-binary index (``cuvs::neighbors::ivf_binary`` in C++) on Apple
silicon: bit-packed uint8 rows searched by Hamming distance."""

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
    """Parameters of :func:`build`, as cuVS's ``ivf_binary::index_params``.
    ``adaptive_centers`` and ``conservative_memory_allocation`` are kept for
    compatibility."""

    def __init__(
        self,
        *,
        n_lists=1024,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=0.5,
        adaptive_centers=False,
        conservative_memory_allocation=False,
        add_data_on_build=True,
    ):
        self.n_lists = n_lists
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.adaptive_centers = adaptive_centers
        self.conservative_memory_allocation = conservative_memory_allocation
        self.add_data_on_build = add_data_on_build


class SearchParams:
    """Parameters of :func:`search`, as cuVS's."""

    def __init__(self, *, n_probes=20):
        self.n_probes = n_probes


class Index:
    """An IVF-binary index."""

    def __init__(self):
        self.trained = False
        self._index = None

    @property
    def n_lists(self):
        return self._index.n_lists

    @property
    def dim(self):
        """Bytes per row, as cuVS's (the rows are bit-packed)."""
        return self._index.dim // 8

    @property
    def mlxvs_index(self):
        """The :class:`mlxvs.IvfBinaryIndex` it wraps."""
        return self._index

    def __len__(self):
        return 0 if self._index is None else len(self._index)

    def __repr__(self):
        return f"Index(type=IVF-BINARY, size={len(self)})"


def _wrap(index):
    idx = Index()
    idx._index, idx.trained = index, True
    return idx


@auto_sync_resources
def build(index_params, dataset, resources=None):
    """Builds the index of bit-packed uint8 rows, as cuVS's
    ``ivf_binary::build``."""
    p = index_params
    x = to_mlx(dataset)
    index = mlxvs.IvfBinaryIndex.build(
        x,
        n_lists=min(p.n_lists, x.shape[0]),
        kmeans_n_iters=p.kmeans_n_iters,
        kmeans_trainset_fraction=p.kmeans_trainset_fraction,
    )
    if not p.add_data_on_build:
        i = index
        index = mlxvs.IvfBinaryIndex(
            i._centers,
            i._codes[:0],
            i._indices[:0],
            mx.zeros_like(i._offsets),
            i._dim_bits,
        )
    return _wrap(index)


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
    """The k nearest rows of every bit-packed query by Hamming distance, as
    cuVS's ``ivf_binary::search``.

    Returns:
        tuple: ``(distances, neighbors)``: Hamming distances ``float32`` and
        ids ``int64``.
    """
    q = to_mlx(queries)
    n_probes = min(search_params.n_probes, index.n_lists)
    mask = _common.filter_mask(filter, len(index), q.shape[0])

    def run(kk):
        return index._index.search(q, kk, n_probes=n_probes)

    ids, dist = run(k) if mask is None else filtered_search(run, mask, k, len(index))
    return _out(dist.astype(mx.float32), distances), _out(
        ids.astype(mx.int64), neighbors
    )


@auto_sync_resources
def extend(index, new_vectors, new_indices, resources=None):
    """Adds bit-packed rows to the lists of their nearest centers."""
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
    return _wrap(load_any_name(mlxvs.IvfBinaryIndex, filename))
