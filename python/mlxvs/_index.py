# Copyright © 2026 mikepapadim

"""CagraIndex: a dataset and its CAGRA graph, kept on the GPU for search."""

import mlx.core as mx

import mlxvs

_FORMAT = "mlxvs.cagra"
_DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "int8": mx.int8,
    "uint8": mx.uint8,
}
_INT_RANGE = {"int8": (-128, 127), "uint8": (0, 255)}


def _unit_rows(x):
    x = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
    return x / mx.maximum(norm, 1e-30)


class CagraIndex:
    """A CAGRA search index: the dataset and its pruned graph, on the GPU.

    The dataset is stored once, ready for search: contiguous, in ``float32`` or
    ``float16`` (half the memory, and faster searches since they mostly read
    rows), and for ``"cosine"`` normalized, so that no search call converts or
    normalizes the rows again. Build one with :meth:`build`, or :meth:`load` a
    saved one.

    Example::

        index = mlxvs.CagraIndex.build(vectors, graph_degree=32, metric="cosine")
        ids, sims = index.search(queries, k=10)
        index.save("docs.safetensors")
        index = mlxvs.CagraIndex.load("docs.safetensors")
    """

    def __init__(self, dataset, graph, metric):
        metric = getattr(metric, "name", metric)
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"unknown metric {metric!r}")
        if dataset.ndim != 2 or graph.ndim != 2 or graph.shape[0] != dataset.shape[0]:
            raise ValueError("dataset must be [n, d] and graph [n, degree]")
        self._dataset = dataset
        self._graph = graph
        self._metric = metric

    @classmethod
    def build(
        cls,
        rows,
        graph_degree=32,
        metric="l2",
        *,
        intermediate_graph_degree=64,
        dtype="float32",
        graph="cagra",
        alpha=1.2,
        **kwargs,
    ):
        """Builds the index of the rows, as cuVS's ``cagra.build``.

        The graph is ``cagra_build``'s, or ``vamana_build``'s with
        ``graph="vamana"`` (keyword arguments such as ``algo``,
        ``n_clusters`` or ``verbose`` pass through to ``all_neighbors``).

        Args:
            rows (array): Rows ``[n, d]``: an MLX or NumPy array (a
              ``numpy.memmap`` works for the graph build, but the index then
              holds the rows in memory), any real dtype.
            graph_degree (int): Degree of the search graph.
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            intermediate_graph_degree (int): Degree of the k-NN graph pruned.
            dtype (str): How the dataset is stored for search: ``"float32"``,
              ``"float16"``, or for rows that are integers in range
              ``"uint8"`` / ``"int8"`` (a quarter of the memory, as cuVS's
              byte datasets such as SIFT's).
            graph (str): How the k-NN graph is pruned: ``"cagra"`` (CAGRA's
              rank-based pruning, needs no distances) or ``"vamana"``
              (JVector's diversity rule, as JVector builds its graphs).
            alpha (float): Vamana's alpha, with ``graph="vamana"``.

        Returns:
            CagraIndex
        """
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        if graph not in ("cagra", "vamana"):
            raise ValueError('graph must be "cagra" or "vamana"')
        metric = getattr(metric, "name", metric)
        if graph == "vamana":
            graph = mlxvs.vamana_build(
                rows,
                graph_degree,
                metric,
                intermediate_graph_degree=intermediate_graph_degree,
                alpha=alpha,
                **kwargs,
            )
        else:
            graph = mlxvs.cagra_build(
                rows,
                graph_degree,
                metric,
                intermediate_graph_degree=intermediate_graph_degree,
                **kwargs,
            )
        dataset = mlxvs._array(rows)
        if dtype in _INT_RANGE:
            # integer rows as they are (cuVS's int8 and uint8 datasets): they
            # must already be integers in range; cosine keeps them unnormalized
            lo, hi = _INT_RANGE[dtype]
            x = dataset.astype(mx.float32)
            ok = mx.all(x == mx.round(x)) & (mx.min(x) >= lo) & (mx.max(x) <= hi)
            if not ok.item():
                raise ValueError(f"dtype={dtype!r} needs integer rows in [{lo}, {hi}]")
        elif metric == "cosine":
            dataset = _unit_rows(dataset)
        dataset = mx.contiguous(dataset.astype(_DTYPES[dtype]))
        mx.eval(dataset, graph)
        return cls(dataset, graph, metric)

    def search(self, queries, k=10, **kwargs):
        """The k nearest dataset rows of every query, as cuVS's ``cagra.search``.

        Args:
            queries (array): Query rows ``[nq, d]``, any real dtype.
            k (int): Neighbors per query.
            **kwargs: ``graph_search``'s options: ``itopk_size`` (default 64),
              ``search_width``, ``algo`` (``"auto"``, ``"single_cta"``,
              ``"multi_cta"``), ``max_iterations``, ``min_iterations``,
              ``num_random_samplings``, ``seed``, ``filter`` (rows that may
              be returned: a bool mask or a uint32 bitset), ``seeds`` (start
              nodes per query, ``[nq, num_seeds]``).

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` and distances ``float32
            [nq, k]``, nearest first: squared distances for ``"l2"``,
            similarities for ``"cosine"`` and ``"inner_product"``.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        metric = self._metric
        if metric == "cosine" and self._dataset.dtype in (mx.int8, mx.uint8):
            # integer rows are stored as they are: cosine from their norms
            return mlxvs.graph_search(
                self._dataset, self._graph, q, k, metric, **kwargs
            )
        if metric == "cosine":
            # the dataset is stored normalized: the dot product of unit rows is
            # the cosine similarity
            q = _unit_rows(q)
            metric = "inner_product"
        return mlxvs.graph_search(self._dataset, self._graph, q, k, metric, **kwargs)

    def save(self, path):
        """Saves the dataset, graph and metric to a ``.safetensors`` file."""
        path = str(path)
        if not path.endswith(".safetensors"):
            raise ValueError("the file name must end in .safetensors")
        mx.save_safetensors(
            path,
            {"dataset": self._dataset, "graph": self._graph},
            metadata={"format": _FORMAT, "version": "1", "metric": self._metric},
        )

    def save_hnswlib(self, path):
        """Saves the index in hnswlib's binary format, for search on the CPU.

        As cuVS's ``serialize_to_hnswlib``: the CAGRA graph becomes hnswlib's
        bottom layer (no upper layers), entered at the medoid; hnswlib's
        ``M`` is half the graph degree. Rows are written as float32 (hnswlib's
        only type), unit rows for ``"cosine"``. Load it with::

            import hnswlib
            p = hnswlib.Index(space="l2", dim=index.dim)  # "cosine", "ip"
            p.load_index(path)
            p.set_ef(64)
            labels, distances = p.knn_query(queries, k=10)

        Labels are row ids. hnswlib's distances are its own: squared L2, or
        1 - similarity for ``"cosine"`` and ``"ip"``.
        """
        mlxvs._core.save_hnswlib(
            str(path), self._dataset, self._graph, self._metric_enum
        )

    def save_jvector(self, path):
        """Saves the index in JVector's on-disk format, for search from Java.

        JVector's ``OnDiskGraphIndex`` file, format version 6 (JVector 4.0):
        one layer (the CAGRA graph, entered at the medoid) and the rows as
        ``INLINE_VECTORS`` (float32, unit rows for ``"cosine"``), with the
        header repeated in the footer as JVector 4 writes it. Load it with::

            var index = OnDiskGraphIndex.load(ReaderSupplierFactory.open(path));

        and search it with ``GraphSearcher`` and
        ``VectorSimilarityFunction.EUCLIDEAN`` (``"l2"``), ``COSINE`` or
        ``DOT_PRODUCT``. Node ordinals are row ids.
        """
        mlxvs._core.save_jvector(
            str(path), self._dataset, self._graph, self._metric_enum
        )

    @property
    def _metric_enum(self):
        return getattr(mlxvs.Metric, self._metric)

    @classmethod
    def load(cls, path):
        """Loads an index saved by :meth:`save`."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != _FORMAT or "dataset" not in arrays:
            raise ValueError(f"{path} is not a saved mlxvs CagraIndex")
        return cls(arrays["dataset"], arrays["graph"], meta["metric"])

    @property
    def dataset(self):
        """The stored rows ``[n, d]`` (unit rows for ``"cosine"``)."""
        return self._dataset

    @property
    def graph(self):
        """The search graph, ``int32 [n, graph_degree]``."""
        return self._graph

    @property
    def metric(self):
        return self._metric

    @property
    def dim(self):
        return self._dataset.shape[1]

    @property
    def graph_degree(self):
        return self._graph.shape[1]

    @property
    def dtype(self):
        return self._dataset.dtype

    def __len__(self):
        return self._dataset.shape[0]

    def __repr__(self):
        return (
            f"CagraIndex(n={len(self)}, dim={self.dim}, "
            f"graph_degree={self.graph_degree}, metric={self._metric!r}, "
            f"dtype={self.dtype})"
        )
