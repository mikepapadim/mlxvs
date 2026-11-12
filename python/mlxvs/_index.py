# Copyright © 2026 mikepapadim

"""CagraIndex: a dataset and its CAGRA graph, kept on the GPU for search."""

import mlx.core as mx

import mlxvs

_FORMAT = "mlxvs.cagra"
_DTYPES = {"float32": mx.float32, "float16": mx.float16}


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
        **kwargs,
    ):
        """Builds the index of the rows, as cuVS's ``cagra.build``.

        The graph is ``cagra_build``'s (keyword arguments such as ``algo``,
        ``n_clusters`` or ``verbose`` pass through to ``all_neighbors``).

        Args:
            rows (array): Rows ``[n, d]``: an MLX or NumPy array (a
              ``numpy.memmap`` works for the graph build, but the index then
              holds the rows in memory), any real dtype.
            graph_degree (int): Degree of the search graph.
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            intermediate_graph_degree (int): Degree of the k-NN graph pruned.
            dtype (str): How the dataset is stored for search: ``"float32"``
              or ``"float16"``.

        Returns:
            CagraIndex
        """
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        metric = getattr(metric, "name", metric)
        graph = mlxvs.cagra_build(
            rows,
            graph_degree,
            metric,
            intermediate_graph_degree=intermediate_graph_degree,
            **kwargs,
        )
        dataset = mlxvs._array(rows)
        if metric == "cosine":
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
              ``num_random_samplings``, ``seed``.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` and distances ``float32
            [nq, k]``, nearest first: squared distances for ``"l2"``,
            similarities for ``"cosine"`` and ``"inner_product"``.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        metric = self._metric
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
        import struct

        import numpy as np

        data = np.asarray(self._dataset.astype(mx.float32))
        graph = np.asarray(self._graph)
        n, d = data.shape
        degree = graph.shape[1]
        m = (degree + 1) // 2
        max_m0 = 2 * m
        entry = self._medoid()

        # each element: link count (uint32), max_m0 links (uint32), the row,
        # the label (uint64); hnswlib reads the count's low 16 bits, and a zero
        # third byte means "not deleted"
        record = np.dtype(
            [
                ("count", "<u4"),
                ("links", "<u4", (max_m0,)),
                ("row", "<f4", (d,)),
                ("label", "<u8"),
            ]
        )
        elements = np.zeros(n, dtype=record)
        valid = graph >= 0
        elements["count"] = valid.sum(axis=1)
        links = np.zeros((n, max_m0), dtype=np.uint32)
        # valid ids first in each row (cagra_optimize writes them first)
        order = np.argsort(~valid, axis=1, kind="stable")
        links[:, :degree] = np.where(
            np.take_along_axis(valid, order, 1),
            np.take_along_axis(graph, order, 1),
            0,
        )
        elements["links"] = links
        elements["row"] = data
        elements["label"] = np.arange(n, dtype=np.uint64)

        size_links0 = 4 + 4 * max_m0
        header = struct.pack(
            "<QQQQQQiIQQQdQ",
            0,  # offsetLevel0
            n,  # max_elements
            n,  # cur_element_count
            record.itemsize,  # size_data_per_element
            size_links0 + 4 * d,  # label_offset
            size_links0,  # offsetData
            0,  # maxlevel
            entry,  # enterpoint_node
            m,  # maxM
            max_m0,  # maxM0
            m,  # M
            1.0 / np.log(m) if m > 1 else 1.0,  # mult
            200,  # ef_construction (unused by search)
        )
        with open(path, "wb") as f:
            f.write(header)
            f.write(elements.tobytes())
            # no element has upper layers: every link-list size is 0
            f.write(np.zeros(n, dtype=np.uint32).tobytes())

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
        import numpy as np

        data = np.asarray(self._dataset.astype(mx.float32))
        graph = np.asarray(self._graph)
        n, d = data.shape
        degree = graph.shape[1]
        entry = self._medoid()
        max_layers = 32  # V4_MAX_LAYERS

        # big-endian, as Java's DataOutput: magic, version, layer 0 size,
        # dimension, entry node, layer 0 degree, id upper bound, layer count,
        # (size, degree) of 32 layers; then the features: their count and
        # ordinals (INLINE_VECTORS = 0, which has no header of its own)
        layers = [n, degree] + [0] * (2 * max_layers - 2)
        header = np.array(
            [0xFFFF0D61, 6, n, d, entry, degree, n, 1, *layers, 1, 0], dtype=">u4"
        ).tobytes()

        # one record per node: its ordinal, the row, the neighbor count and
        # the neighbors (-1 padded)
        record = np.dtype(
            [
                ("ordinal", ">i4"),
                ("row", ">f4", (d,)),
                ("count", ">i4"),
                ("neighbors", ">i4", (degree,)),
            ]
        )
        nodes = np.zeros(n, dtype=record)
        nodes["ordinal"] = np.arange(n)
        nodes["row"] = data
        valid = graph >= 0
        nodes["count"] = valid.sum(axis=1)
        order = np.argsort(~valid, axis=1, kind="stable")
        nodes["neighbors"] = np.where(
            np.take_along_axis(valid, order, 1),
            np.take_along_axis(graph, order, 1),
            -1,
        )
        with open(path, "wb") as f:
            f.write(header)
            f.write(nodes.tobytes())
            # the footer: the header again, its offset, the footer magic
            footer_at = len(header) + nodes.nbytes
            f.write(header)
            f.write(np.array([footer_at], dtype=">i8").tobytes())
            f.write(np.array([0x4A564244], dtype=">u4").tobytes())

    def _medoid(self):
        """The row nearest the mean (largest dot product with it for the
        similarity metrics): the entry point of exported graphs."""
        import numpy as np

        mean = mx.mean(self._dataset.astype(mx.float32), axis=0, keepdims=True)
        metric = "inner_product" if self._metric == "cosine" else self._metric
        ids = mlxvs.brute_force_knn(self._dataset, mean, 1, metric)[0]
        return int(np.asarray(ids)[0, 0])

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
