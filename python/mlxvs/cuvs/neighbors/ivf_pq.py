# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.ivf_pq``: the IVF-PQ index on Apple silicon."""

import mlx.core as mx
import numpy as np

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
    """Parameters of :func:`build`, as cuVS's. Codebooks are per subspace
    (cuVS's default ``codebook_kind``); ``force_random_rotation``,
    ``conservative_memory_allocation`` and ``codes_layout`` are kept for
    compatibility."""

    def __init__(
        self,
        *,
        n_lists=1024,
        metric="sqeuclidean",
        metric_arg=2.0,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=0.5,
        pq_bits=8,
        pq_dim=0,
        codebook_kind="subspace",
        force_random_rotation=False,
        add_data_on_build=True,
        conservative_memory_allocation=False,
        max_train_points_per_pq_code=256,
        codes_layout="interleaved",
    ):
        _common.index_metric(metric, "IVF-PQ")
        if codebook_kind not in ("subspace", "cluster"):
            raise ValueError("codebook_kind must be 'subspace' or 'cluster'")
        self.n_lists = n_lists
        self.metric = metric
        self.metric_arg = metric_arg
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.pq_bits = pq_bits
        self.pq_dim = pq_dim
        self.codebook_kind = codebook_kind
        self.force_random_rotation = force_random_rotation
        self.add_data_on_build = add_data_on_build
        self.conservative_memory_allocation = conservative_memory_allocation
        self.max_train_points_per_pq_code = max_train_points_per_pq_code
        self.codes_layout = codes_layout


class SearchParams:
    """Parameters of :func:`search`, as cuVS's. The lookup tables and
    distances are float32 whatever the dtype options say."""

    def __init__(
        self,
        *,
        n_probes=20,
        lut_dtype=np.float32,
        internal_distance_dtype=np.float32,
        coarse_search_dtype=np.float32,
        max_internal_batch_size=4096,
    ):
        self.n_probes = n_probes
        self.lut_dtype = lut_dtype
        self.internal_distance_dtype = internal_distance_dtype
        self.coarse_search_dtype = coarse_search_dtype
        self.max_internal_batch_size = max_internal_batch_size


class Index:
    """An IVF-PQ index."""

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
    def pq_dim(self):
        return self._index.pq_dim

    @property
    def pq_bits(self):
        return self._index.pq_bits

    @property
    def pq_len(self):
        return self._index._codebooks.shape[2]

    @property
    def size(self):
        return len(self)

    @property
    def centers(self):
        return self._index._centers

    @property
    def pq_centers(self):
        return self._index._codebooks

    @property
    def list_sizes(self):
        o = self._index._offsets
        return (o[1:] - o[:-1]).astype(mx.uint32)

    @property
    def metric(self):
        return _common.metric_name(self._metric)

    @property
    def mlxvs_index(self):
        """The :class:`mlxvs.IvfPqIndex` it wraps."""
        return self._index

    def __len__(self):
        return 0 if self._index is None else len(self._index)

    def __repr__(self):
        return (
            f"Index(type=IVF-PQ, metric={self.metric}, size={len(self)}, "
            f"pq_dim={self.pq_dim}, pq_bits={self.pq_bits})"
        )


def _wrap(index, t):
    idx = Index()
    idx._index, idx._metric, idx.trained = index, t, True
    return idx


@auto_sync_resources
def build(index_params, dataset, resources=None):
    """Builds the index, as cuVS's ``ivf_pq.build``."""
    p = index_params
    m, t = _common.index_metric(p.metric, "IVF-PQ")
    x = to_mlx(dataset)
    index = mlxvs.IvfPqIndex.build(
        x,
        n_lists=min(p.n_lists, x.shape[0]),
        metric=m,
        pq_dim=p.pq_dim,
        pq_bits=p.pq_bits,
        kmeans_n_iters=p.kmeans_n_iters,
        kmeans_trainset_fraction=p.kmeans_trainset_fraction,
    )
    if not p.add_data_on_build:
        i = index
        index = mlxvs.IvfPqIndex(
            i._centers,
            i._codebooks,
            i._codes[:0],
            i._indices[:0],
            mx.zeros_like(i._offsets),
            m,
            i._dim,
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
    """The k nearest rows of every query by their PQ distances, as cuVS's
    ``ivf_pq.search`` (re-rank them exactly with ``refine``).

    Returns:
        tuple: ``(distances, neighbors)``, ``float32`` and ``int64``.
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
    """Encodes rows into the lists of their nearest centers, as cuVS's
    ``ivf_pq.extend``."""
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
    ci = load_any_name(mlxvs.IvfPqIndex, filename)
    t = {"l2": 0, "inner_product": 6, "cosine": 2}[ci.metric]
    return _wrap(ci, t)
