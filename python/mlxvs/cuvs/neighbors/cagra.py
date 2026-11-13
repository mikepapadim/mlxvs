# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.cagra``: the CAGRA graph index on Apple silicon.

The same classes, functions, parameters and results as cuVS 26.10, so code
written for cuVS runs by changing the import, and the same index files:
:func:`save` writes cuVS's CAGRA format (version 6 by default; 4 and 5 for
older cuVS) and :func:`load` reads the files cuVS writes, so an index built on
an NVIDIA GPU can be searched on a Mac and back.
"""

import mlx.core as mx
import numpy as np

import mlxvs
from mlxvs._index import _DTYPES, _unit_rows
from mlxvs.cuvs import _serialize as io
from mlxvs.cuvs import distance as _d
from mlxvs.cuvs.common import _mx_to_np, _out, auto_sync_resources, to_mlx
from mlxvs.cuvs.neighbors import _common

__all__ = [
    "AceParams",
    "Dataset",
    "ExtendParams",
    "Index",
    "IndexParams",
    "SearchParams",
    "build",
    "build_index",
    "extend",
    "from_graph",
    "load",
    "save",
    "search",
    "update_dataset",
]

SERIALIZATION_VERSION = 6
_KIND_DEVICE_PADDED = 1
_SERIALIZE_STRIDED_DATASET = 2


class AceParams:
    """ACE build parameters (cuVS builds graphs of datasets larger than GPU
    memory with them). The Mac's GPU shares the host memory, so the regular
    build is used; the parameters are kept for compatibility."""

    def __init__(
        self,
        *,
        npartitions=0,
        ef_construction=120,
        build_dir="/tmp/ace_build",
        use_disk=False,
        max_host_memory_gb=0,
        max_gpu_memory_gb=0,
    ):
        self.npartitions = npartitions
        self.ef_construction = ef_construction
        self.build_dir = build_dir
        self.use_disk = use_disk
        self.max_host_memory_gb = max_host_memory_gb
        self.max_gpu_memory_gb = max_gpu_memory_gb


class IndexParams:
    """Parameters of :func:`build`, as cuVS's.

    Args:
        metric (str): ``"sqeuclidean"``, ``"inner_product"`` or ``"cosine"``
          (``"euclidean"`` too: the graph is the same, distances are their
          square roots).
        intermediate_graph_degree (int): Degree of the k-NN graph pruned.
        graph_degree (int): Degree of the search graph.
        build_algo (str): How the k-NN graph is built: ``"nn_descent"``;
          ``"ivf_pq"`` (cuVS's default), ``"iterative_cagra_search"`` and
          ``"ace"`` build it with NN-Descent too.
        nn_descent_niter (int): NN-Descent iterations.
        refinement_rate (float): Kept for compatibility.
    """

    def __init__(
        self,
        *,
        metric="sqeuclidean",
        intermediate_graph_degree=128,
        graph_degree=64,
        build_algo="ivf_pq",
        nn_descent_niter=20,
        ivf_pq_build_params=None,
        ivf_pq_search_params=None,
        ace_params=None,
        refinement_rate=1.0,
    ):
        if build_algo not in ("ivf_pq", "nn_descent", "iterative_cagra_search", "ace"):
            raise ValueError(f"Unknown build_algo '{build_algo}'")
        _common.index_metric(metric, "CAGRA")
        self.metric = metric
        self.intermediate_graph_degree = intermediate_graph_degree
        self.graph_degree = graph_degree
        self.build_algo = build_algo
        self.nn_descent_niter = nn_descent_niter
        self.ivf_pq_build_params = ivf_pq_build_params
        self.ivf_pq_search_params = ivf_pq_search_params
        self.ace_params = ace_params
        self.refinement_rate = refinement_rate


class SearchParams:
    """Parameters of :func:`search`, as cuVS's. ``algo`` is ``"auto"``,
    ``"single_cta"``, ``"multi_cta"`` or ``"multi_kernel"`` (searched as
    multi-CTA). The hash map, thread block and persistent-kernel options are
    kept for compatibility: mlxvs sizes them itself."""

    def __init__(
        self,
        *,
        max_queries=0,
        itopk_size=64,
        max_iterations=0,
        algo="auto",
        team_size=0,
        search_width=1,
        min_iterations=0,
        thread_block_size=0,
        hashmap_mode="auto",
        hashmap_min_bitlen=0,
        hashmap_max_fill_rate=0.5,
        num_random_samplings=1,
        rand_xor_mask=0x128394,
        persistent=False,
        persistent_lifetime=None,
        persistent_device_usage=None,
    ):
        if algo not in ("auto", "single_cta", "multi_cta", "multi_kernel"):
            raise ValueError(f"Unknown algo '{algo}'")
        self.max_queries = max_queries
        self.itopk_size = itopk_size
        self.max_iterations = max_iterations
        self.algo = algo
        self.team_size = team_size
        self.search_width = search_width
        self.min_iterations = min_iterations
        self.thread_block_size = thread_block_size
        self.hashmap_mode = hashmap_mode
        self.hashmap_min_bitlen = hashmap_min_bitlen
        self.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.num_random_samplings = num_random_samplings
        self.rand_xor_mask = rand_xor_mask
        self.persistent = persistent
        self.persistent_lifetime = persistent_lifetime
        self.persistent_device_usage = persistent_device_usage

    def _kwargs(self):
        algo = {"multi_kernel": "multi_cta"}.get(self.algo, self.algo)
        kw = {
            "itopk_size": self.itopk_size,
            "search_width": self.search_width,
            "algo": algo,
            "team_size": self.team_size,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "num_random_samplings": self.num_random_samplings,
            "seed": self.rand_xor_mask & 0x7FFFFFFF,
        }
        return kw


class ExtendParams:
    """Parameters of :func:`extend`, as cuVS's."""

    def __init__(self, *, max_chunk_size=None):
        self.max_chunk_size = max_chunk_size


class Dataset:
    """The dataset of a loaded index (``load``'s ``out_dataset``)."""

    def __init__(self, array=None):
        self.array = array
        self.layout = "padded"


class Index:
    """A CAGRA index: the dataset and its search graph, on the GPU."""

    def __init__(self):
        self.trained = False
        self._index = None
        self._metric = _d.L2Expanded
        self._source_indices = None

    @property
    def dim(self):
        return self._index.dim

    @property
    def graph_degree(self):
        return self._index.graph_degree

    def __len__(self):
        return 0 if self._index is None else len(self._index)

    @property
    def dtype(self):
        return np.dtype(_mx_to_np(self._index.dtype))

    @property
    def dataset(self):
        return None if self._index is None else self._index.dataset

    @property
    def graph(self):
        return self._index.graph.astype(mx.uint32)

    @property
    def metric(self):
        return _common.metric_name(self._metric)

    @property
    def mlxvs_index(self):
        """The :class:`mlxvs.CagraIndex` it wraps."""
        return self._index

    def __repr__(self):
        return f"Index(type=CAGRA, metric={self.metric}, dim={self.dim if self.trained else 0})"


def _dtype_name(x):
    name = str(x.dtype).split(".")[-1]
    if name not in _DTYPES:
        return "float32"
    return name


def _prepare(dataset, mlxvs_metric):
    """The dataset as mlxvs.CagraIndex stores it."""
    x = to_mlx(dataset)
    name = _dtype_name(x)
    if mlxvs_metric == "cosine" and name in ("float32", "float16"):
        x = _unit_rows(x)
    return mx.contiguous(x.astype(_DTYPES[name])), name


def _wrap(cagra_index, metric_type, source_indices=None):
    idx = Index()
    idx._index = cagra_index
    idx._metric = metric_type
    idx._source_indices = source_indices
    idx.trained = True
    return idx


@auto_sync_resources
def build(index_params, dataset, resources=None):
    """Builds the CAGRA index of the dataset, as cuVS's ``cagra.build``: the
    k-NN graph of the rows, pruned to ``graph_degree`` by CAGRA's rank-based
    detours.

    Args:
        index_params (IndexParams): Build parameters.
        dataset: Rows ``[n, dim]``: float32, float16, int8 or uint8.

    Returns:
        Index
    """
    p = index_params
    mlxvs_metric, t = _common.index_metric(p.metric, "CAGRA")
    x = to_mlx(dataset)
    name = _dtype_name(x)
    ig = min(p.intermediate_graph_degree, max(1, x.shape[0] - 1))
    g = min(p.graph_degree, ig)
    index = mlxvs.CagraIndex.build(
        x,
        g,
        mlxvs_metric,
        intermediate_graph_degree=ig,
        dtype=name,
        max_iterations=max(1, int(p.nn_descent_niter)),
    )
    return _wrap(index, t)


def build_index(index_params, dataset, resources=None):
    """Deprecated alias of :func:`build`, as cuVS's."""
    return build(index_params, dataset, resources=resources)


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
    """The k nearest rows of every query, as cuVS's ``cagra.search``.

    Args:
        search_params (SearchParams): Search parameters.
        index (Index): A built or loaded index.
        queries: Rows ``[n_queries, dim]``, the dataset's dtype or float32.
        k (int): Neighbors per query.
        neighbors: Optional ``uint32 [n_queries, k]`` output.
        distances: Optional ``float32 [n_queries, k]`` output.
        filter (Prefilter): Rows that may be returned
          (``filters.from_bitset`` / ``from_bitmap``).

    Returns:
        tuple: ``(distances, neighbors)``: squared distances for
        ``"sqeuclidean"``, inner products for ``"inner_product"``, cosine
        distances for ``"cosine"``; nearest first.
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")
    q = to_mlx(queries)
    kw = search_params._kwargs()
    mask = _common.filter_mask(filter, len(index), q.shape[0])
    if mask is not None:
        kw["filter"] = mask
    ids, scores = index._index.search(q, k, **kw)
    dist = _common.to_cuvs_distances(scores, index._metric)
    if index._source_indices is not None:
        src = index._source_indices
        ids = mx.where(ids >= 0, src[mx.maximum(ids, 0)], ids)
    ids = ids.astype(mx.uint32)
    return _out(dist.astype(mx.float32), distances), _out(ids, neighbors)


@auto_sync_resources
def from_graph(graph, dataset, metric="sqeuclidean", resources=None):
    """An index of a graph built elsewhere and its dataset, as cuVS's
    ``cagra.from_graph``."""
    mlxvs_metric, t = _common.index_metric(metric, "CAGRA")
    data, _ = _prepare(dataset, mlxvs_metric)
    g = to_mlx(graph).astype(mx.int32)
    return _wrap(mlxvs.CagraIndex(data, g, mlxvs_metric), t)


@auto_sync_resources
def update_dataset(index, dataset, resources=None):
    """Replaces the dataset of the index (after loading a file saved without
    it), as cuVS's ``cagra.update_dataset``."""
    data, _ = _prepare(dataset, index._index.metric)
    if data.shape[0] != len(index):
        raise ValueError("the dataset must have a row per graph node")
    index._index._dataset = data
    return index


@auto_sync_resources
def extend(params, index, extended_dataset, new_start_row, resources=None):
    """Adds rows to the index without rebuilding it, as cuVS's
    ``cagra.extend``: ``extended_dataset`` holds the indexed rows followed by
    the new ones, which start at ``new_start_row`` (the index's size)."""
    if new_start_row != len(index):
        raise ValueError("new_start_row must equal the index size")
    x = to_mlx(extended_dataset)
    index._index.extend(x[new_start_row:])
    return index


def _padded_stride(dim, itemsize):
    align = max(1, 16 // itemsize)
    return -(-dim // align) * align


@auto_sync_resources
def save(
    filename, index, include_dataset=True, resources=None, version=SERIALIZATION_VERSION
):
    """Saves the index in cuVS's CAGRA format, which cuVS's ``cagra.load``
    reads.

    Args:
        filename (str): The file.
        index (Index): The index.
        include_dataset (bool): Write the rows too (else load needs
          ``update_dataset``).
        version (int): The format: 6 (cuVS 26.10 on), 5 (26.02 to 26.08) or
          4 (25.x).
    """
    if version not in (4, 5, 6):
        raise ValueError("version must be 4, 5 or 6")
    ci = index._index
    if ci.compressed and include_dataset:
        raise ValueError(
            "compressed (VPQ) indexes store only the graph; save with include_dataset=False"
        )
    data = None if ci.compressed else np.asarray(ci.dataset)
    graph = np.asarray(ci.graph).astype(np.uint32)
    n, degree = graph.shape
    dtype = np.dtype(_mx_to_np(ci.dtype))
    include_dataset = include_dataset and data is not None and n > 0
    src = index._source_indices
    with open(filename, "wb") as f:
        f.write(io.dtype_prefix(dtype, version))
        io.write_scalar(f, version, np.int32)
        if version >= 6:
            io.write_scalar(f, _KIND_DEVICE_PADDED if include_dataset else 0, np.uint32)
        io.write_scalar(f, n, np.uint32)
        io.write_scalar(f, index.dim, np.uint32)
        io.write_scalar(f, degree, np.uint32)
        io.write_scalar(f, index._metric, np.int32)
        io.write_array(f, graph)
        if version == 4:
            io.write_scalar(f, include_dataset, np.bool_)
        else:
            content = (1 if include_dataset else 0) | (2 if src is not None else 0)
            io.write_scalar(f, content, np.uint32)
        if include_dataset:
            io.write_scalar(f, _SERIALIZE_STRIDED_DATASET, np.uint32)
            io.write_scalar(f, io.CUDA_TYPES[dtype], np.uint32)
            io.write_scalar(f, n, np.int64)
            io.write_scalar(f, data.shape[1], np.uint32)
            io.write_scalar(f, _padded_stride(data.shape[1], dtype.itemsize), np.uint32)
            io.write_array(f, data)
        if version >= 5 and src is not None:
            io.write_array(f, np.asarray(src).astype(np.uint32))


def _read(filename):
    with open(filename, "rb") as f:
        dtype = io.read_prefix(f)
        version = io.read_scalar(f)
        if version not in (4, 5, 6):
            raise ValueError(f"unsupported CAGRA serialization version {version}")
        if version >= 6:
            io.read_scalar(f)  # the dataset kind: every dense kind reads alike
        n = io.read_scalar(f)
        dim = io.read_scalar(f)
        io.read_scalar(f)
        metric = io.read_scalar(f)
        graph = io.read_array(f)
        content = io.read_scalar(f)
        has_dataset = bool(content & 1)
        data = None
        if has_dataset:
            tag = io.read_scalar(f)
            if tag != _SERIALIZE_STRIDED_DATASET:
                raise ValueError(
                    "only dense datasets can be read; a VPQ dataset is saved apart"
                )
            io.read_scalar(f)
            io.read_scalar(f)
            io.read_scalar(f)
            io.read_scalar(f)
            data = io.read_array(f).astype(dtype, copy=False)
        src = io.read_array(f) if version >= 5 and content & 2 else None
    return dtype, int(n), int(dim), int(metric), graph, data, src


@auto_sync_resources
def load(index, filename=None, out_dataset=None, resources=None):
    """Loads a CAGRA index file written by cuVS (format 4, 5 or 6) or by
    :func:`save`, as cuVS's ``cagra.load(index, filename)``. Also callable
    as ``load(filename)``, returning the index.

    A file saved without its dataset needs :func:`update_dataset` before
    search.
    """
    if isinstance(index, (str, bytes)) or hasattr(index, "__fspath__"):
        index, filename = Index(), index
    dtype, n, dim, metric, graph, data, src = _read(filename)
    mlxvs_metric, _ = _common.index_metric(_common.metric_name(metric), "CAGRA")
    if data is None:
        # a placeholder until update_dataset: zero rows of the right shape
        data = np.zeros((n, dim), dtype=dtype)
    elif mlxvs_metric == "cosine" and dtype.kind == "f":
        data = np.asarray(_unit_rows(mx.array(data)).astype(_DTYPES[dtype.name]))
    g = mx.array(graph.astype(np.int64).astype(np.int32))
    ci = mlxvs.CagraIndex(mx.array(np.ascontiguousarray(data)), g, mlxvs_metric)
    index._index = ci
    index._metric = metric
    index._source_indices = (
        None if src is None else mx.array(src.astype(np.int64).astype(np.int32))
    )
    index.trained = True
    if out_dataset is not None:
        out_dataset.array = ci.dataset
    return index
