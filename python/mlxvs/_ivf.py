# Copyright © 2026 mikepapadim

"""IvfFlatIndex: rows grouped into k-means lists, scanned on the GPU."""

import mlx.core as mx

import mlxvs

_FORMAT = "mlxvs.ivf_flat"


class IvfFlatIndex:
    """An IVF-Flat index, as cuVS's ``ivf_flat``: rows grouped into the lists
    of their nearest k-means center; a query scans the ``n_probes`` lists of
    nearest center, exactly. Cheaper to build than :class:`CagraIndex` and
    easy to extend; for the same recall the graph index answers faster.

    Example::

        index = mlxvs.IvfFlatIndex.build(vectors, n_lists=1024)
        ids, dist = index.search(queries, k=10, n_probes=20)
        index.extend(more_vectors)
        index.save("ivf.safetensors")
    """

    def __init__(self, centers, data, indices, offsets, metric):
        metric = getattr(metric, "name", metric)
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"unknown metric {metric!r}")
        self._centers, self._data = centers, data
        self._indices, self._offsets = indices, offsets
        self._metric = metric

    @classmethod
    def build(cls, rows, n_lists=1024, metric="l2", **kwargs):
        """Builds the index of the rows, as cuVS's ``ivf_flat.build``.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype.
            n_lists (int): k-means lists (at most n).
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            **kwargs: ``kmeans_n_iters`` (20), ``kmeans_trainset_fraction``
              (0.5), ``seed``.

        Returns:
            IvfFlatIndex
        """
        metric = getattr(metric, "name", metric)
        parts = mlxvs._core.ivf_flat_build(
            mlxvs._array(rows), getattr(mlxvs.Metric, metric), n_lists=n_lists, **kwargs
        )
        return cls(*parts, metric)

    def search(self, queries, k=10, n_probes=20):
        """The k nearest rows of every query, as cuVS's ``ivf_flat.search``.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` (-1 where the scanned
            lists hold fewer rows) and distances ``float32 [nq, k]``, as
            ``brute_force_knn``.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        return mlxvs._core.ivf_flat_search(
            self._centers,
            self._data,
            self._indices,
            self._offsets,
            getattr(mlxvs.Metric, self._metric),
            q,
            k,
            n_probes=n_probes,
        )

    def extend(self, rows, ids=None):
        """Adds rows to the lists of their nearest centers (the centers do not
        move), with ``ids``, or the largest id plus 1, 2, ... by default.

        Returns:
            IvfFlatIndex: this index, extended.
        """
        x = mlxvs._array(rows)
        if x.ndim == 1:
            x = x[None]
        new_ids = None if ids is None else mlxvs._array(ids).astype(mx.int32)
        parts = mlxvs._core.ivf_flat_extend(
            self._centers,
            self._data,
            self._indices,
            self._offsets,
            getattr(mlxvs.Metric, self._metric),
            x,
            new_ids,
        )
        self._centers, self._data, self._indices, self._offsets = parts
        return self

    def save(self, path):
        """Saves the index to a ``.safetensors`` file."""
        path = str(path)
        if not path.endswith(".safetensors"):
            raise ValueError("the file name must end in .safetensors")
        mx.save_safetensors(
            path,
            {
                "centers": self._centers,
                "data": self._data,
                "indices": self._indices,
                "offsets": self._offsets,
            },
            metadata={"format": _FORMAT, "version": "1", "metric": self._metric},
        )

    @classmethod
    def load(cls, path):
        """Loads an index saved by :meth:`save`."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != _FORMAT:
            raise ValueError(f"{path} is not a saved mlxvs IvfFlatIndex")
        return cls(
            arrays["centers"],
            arrays["data"],
            arrays["indices"],
            arrays["offsets"],
            meta["metric"],
        )

    @property
    def centers(self):
        return self._centers

    @property
    def n_lists(self):
        return self._centers.shape[0]

    @property
    def metric(self):
        return self._metric

    @property
    def dim(self):
        return self._data.shape[1]

    def __len__(self):
        return self._data.shape[0]

    def __repr__(self):
        return (
            f"IvfFlatIndex(n={len(self)}, dim={self.dim}, n_lists={self.n_lists}, "
            f"metric={self._metric!r})"
        )


_PQ_FORMAT = "mlxvs.ivf_pq"


def refine(dataset, queries, candidates, k, metric="l2"):
    """Exact re-ranking, as cuVS's ``refine``: the k nearest of each query's
    candidate ids by exact distances, nearest first.

    ``dataset`` may be an MLX array, or a NumPy array or ``numpy.memmap``
    (only the candidate rows are read, on the host, so the rows need not fit
    on the GPU).

    Returns:
        tuple(array, array): ids and distances as ``brute_force_knn``.
    """
    import numpy as np

    metric = getattr(metric, "name", metric)
    q = mlxvs._array(queries)
    if q.ndim == 1:
        q = q[None]
    cand = np.asarray(candidates).astype(np.int64)
    if isinstance(dataset, mx.array):
        return mlxvs._core.refine(
            dataset,
            q,
            mx.array(cand.astype(np.int32)),
            k,
            getattr(mlxvs.Metric, metric),
        )
    # gather the distinct candidate rows once, on the host
    valid = cand >= 0
    rows, local = np.unique(np.where(valid, cand, 0), return_inverse=True)
    local = np.where(valid, local.reshape(cand.shape), -1).astype(np.int32)
    gathered = mx.array(np.ascontiguousarray(dataset[rows], dtype=np.float32))
    ids, dist = mlxvs._core.refine(
        gathered, q, mx.array(local), k, getattr(mlxvs.Metric, metric)
    )
    ids = np.asarray(ids)
    ids = np.where(ids >= 0, rows[np.maximum(ids, 0)], -1).astype(np.int32)
    return mx.array(ids), dist


class IvfPqIndex:
    """An IVF-PQ index, as cuVS's ``ivf_pq``: rows in the lists of their
    nearest k-means center, stored as product-quantization codes of their
    residual (``pq_dim`` bytes per row with 8-bit codes). Searches use
    approximate distances; pass the rows to :meth:`search` with
    ``refine_ratio > 1`` to re-rank the best candidates exactly.

    Example::

        index = mlxvs.IvfPqIndex.build(vectors, n_lists=1024)   # 8x smaller than float32 at d=128
        ids, dist = index.search(queries, k=10, n_probes=20, refine_ratio=4, dataset=vectors)
    """

    def __init__(self, centers, codebooks, codes, indices, offsets, metric, dim):
        metric = getattr(metric, "name", metric)
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"unknown metric {metric!r}")
        self._centers, self._codebooks, self._codes = centers, codebooks, codes
        self._indices, self._offsets = indices, offsets
        self._metric, self._dim = metric, int(dim)

    @classmethod
    def build(cls, rows, n_lists=1024, metric="l2", pq_dim=0, pq_bits=8, **kwargs):
        """Builds the index of the rows, as cuVS's ``ivf_pq.build``.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype.
            n_lists (int): k-means lists.
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            pq_dim (int): Subspaces per row; 0: cuVS's heuristic (half the
              dimension from 128 up, rounded down to a multiple of 32).
            pq_bits (int): Bits per code, 4 to 8.
            **kwargs: ``kmeans_n_iters``, ``kmeans_trainset_fraction``,
              ``seed``.
        """
        metric = getattr(metric, "name", metric)
        parts = mlxvs._core.ivf_pq_build(
            mlxvs._array(rows),
            getattr(mlxvs.Metric, metric),
            n_lists=n_lists,
            pq_dim=pq_dim,
            pq_bits=pq_bits,
            **kwargs,
        )
        return cls(*parts[:5], metric, parts[5])

    def _parts(self):
        return (
            self._centers,
            self._codebooks,
            self._codes,
            self._indices,
            self._offsets,
            getattr(mlxvs.Metric, self._metric),
            self._dim,
        )

    def search(self, queries, k=10, n_probes=20, refine_ratio=1.0, dataset=None):
        """The k nearest rows of every query, as cuVS's ``ivf_pq.search``.

        With ``refine_ratio > 1`` and the rows as ``dataset`` (MLX, NumPy or
        ``numpy.memmap``), ``ceil(k * refine_ratio)`` candidates are re-ranked
        by exact distances (:func:`refine`).

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` and distances ``float32
            [nq, k]``, approximate unless refined.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        if refine_ratio > 1 and dataset is None:
            raise ValueError("refine_ratio > 1 needs the rows as dataset")
        kc = min(256, max(k, int(-(-k * refine_ratio // 1)))) if refine_ratio > 1 else k
        ids, dist = mlxvs._core.ivf_pq_search(*self._parts(), q, kc, n_probes=n_probes)
        if kc > k:
            return refine(dataset, q, ids, k, self._metric)
        return ids, dist

    def extend(self, rows, ids=None):
        """Adds rows to their lists, coded with the index's codebooks, with
        ``ids`` or the largest id plus 1, 2, ... by default.

        Returns:
            IvfPqIndex: this index, extended.
        """
        x = mlxvs._array(rows)
        if x.ndim == 1:
            x = x[None]
        new_ids = None if ids is None else mlxvs._array(ids).astype(mx.int32)
        self._codes, self._indices, self._offsets = mlxvs._core.ivf_pq_extend(
            *self._parts(), x, new_ids
        )
        return self

    def save(self, path):
        """Saves the index to a ``.safetensors`` file (codes, not rows)."""
        path = str(path)
        if not path.endswith(".safetensors"):
            raise ValueError("the file name must end in .safetensors")
        mx.save_safetensors(
            path,
            {
                "centers": self._centers,
                "codebooks": self._codebooks,
                "codes": self._codes,
                "indices": self._indices,
                "offsets": self._offsets,
            },
            metadata={
                "format": _PQ_FORMAT,
                "version": "1",
                "metric": self._metric,
                "dim": str(self._dim),
            },
        )

    @classmethod
    def load(cls, path):
        """Loads an index saved by :meth:`save`."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != _PQ_FORMAT:
            raise ValueError(f"{path} is not a saved mlxvs IvfPqIndex")
        return cls(
            arrays["centers"],
            arrays["codebooks"],
            arrays["codes"],
            arrays["indices"],
            arrays["offsets"],
            meta["metric"],
            int(meta["dim"]),
        )

    @property
    def pq_dim(self):
        return self._codebooks.shape[0]

    @property
    def pq_bits(self):
        return int(self._codebooks.shape[1]).bit_length() - 1

    @property
    def n_lists(self):
        return self._centers.shape[0]

    @property
    def metric(self):
        return self._metric

    @property
    def dim(self):
        return self._dim

    def __len__(self):
        return self._codes.shape[0]

    def __repr__(self):
        return (
            f"IvfPqIndex(n={len(self)}, dim={self.dim}, n_lists={self.n_lists}, "
            f"pq_dim={self.pq_dim}, pq_bits={self.pq_bits}, metric={self._metric!r})"
        )


_SQ_FORMAT = "mlxvs.ivf_sq"


class IvfSqIndex:
    """An IVF-SQ index, as cuVS's ``ivf_sq``: rows in the lists of their
    nearest k-means center, stored as one byte per dimension of their
    residual to that center (``center + vmin + delta * code``, the per-
    dimension range of the residuals in 255 steps). A quarter of IVF-Flat's
    memory, and searched from the codes with nearly its recall.

    Example::

        index = mlxvs.IvfSqIndex.build(vectors, n_lists=1024)
        ids, dist = index.search(queries, k=10, n_probes=20)
    """

    def __init__(self, centers, codes, indices, offsets, vmin, delta, metric):
        metric = getattr(metric, "name", metric)
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"unknown metric {metric!r}")
        self._centers, self._codes = centers, codes
        self._indices, self._offsets = indices, offsets
        self._vmin, self._delta = vmin, delta
        self._metric = metric

    @classmethod
    def build(cls, rows, n_lists=1024, metric="l2", **kwargs):
        """Builds the index of the rows, as cuVS's ``ivf_sq.build``.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype.
            n_lists (int): k-means lists.
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            **kwargs: ``kmeans_n_iters`` (20), ``max_train_points_per_cluster``
              (256: the centers and the quantization are trained on
              ``n_lists`` times this many rows), ``seed``.
        """
        metric = getattr(metric, "name", metric)
        parts = mlxvs._core.ivf_sq_build(
            mlxvs._array(rows), getattr(mlxvs.Metric, metric), n_lists=n_lists, **kwargs
        )
        return cls(*parts, metric)

    def _parts(self):
        return (
            self._centers,
            self._codes,
            self._indices,
            self._offsets,
            self._vmin,
            self._delta,
            getattr(mlxvs.Metric, self._metric),
        )

    def search(self, queries, k=10, n_probes=20):
        """The k nearest rows of every query by their decoded rows, as cuVS's
        ``ivf_sq.search``.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` and distances, as
            ``brute_force_knn``.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        return mlxvs._core.ivf_sq_search(*self._parts(), q, k, n_probes=n_probes)

    def extend(self, rows, ids=None):
        """Encodes rows into the lists of their nearest centers (centers and
        quantization unchanged), with ``ids`` or the largest id plus 1, 2, ...

        Returns:
            IvfSqIndex: this index, extended.
        """
        x = mlxvs._array(rows)
        if x.ndim == 1:
            x = x[None]
        new_ids = None if ids is None else mlxvs._array(ids).astype(mx.int32)
        parts = mlxvs._core.ivf_sq_extend(*self._parts(), x, new_ids)
        (
            self._centers,
            self._codes,
            self._indices,
            self._offsets,
            self._vmin,
            self._delta,
        ) = parts
        return self

    def save(self, path):
        """Saves the index to a ``.safetensors`` file."""
        path = str(path)
        if not path.endswith(".safetensors"):
            raise ValueError("the file name must end in .safetensors")
        mx.save_safetensors(
            path,
            {
                "centers": self._centers,
                "codes": self._codes,
                "indices": self._indices,
                "offsets": self._offsets,
                "vmin": self._vmin,
                "delta": self._delta,
            },
            metadata={"format": _SQ_FORMAT, "version": "1", "metric": self._metric},
        )

    @classmethod
    def load(cls, path):
        """Loads an index saved by :meth:`save`."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != _SQ_FORMAT:
            raise ValueError(f"{path} is not a saved mlxvs IvfSqIndex")
        return cls(
            arrays["centers"],
            arrays["codes"],
            arrays["indices"],
            arrays["offsets"],
            arrays["vmin"],
            arrays["delta"],
            meta["metric"],
        )

    @property
    def centers(self):
        return self._centers

    @property
    def n_lists(self):
        return self._centers.shape[0]

    @property
    def metric(self):
        return self._metric

    @property
    def dim(self):
        return self._centers.shape[1]

    def __len__(self):
        return self._codes.shape[0]

    def __repr__(self):
        return (
            f"IvfSqIndex(n={len(self)}, dim={self.dim}, n_lists={self.n_lists}, "
            f"metric={self._metric!r})"
        )


_RABITQ_FORMAT = "mlxvs.ivf_rabitq"


class IvfRabitqIndex:
    """An IVF-RaBitQ index, as cuVS's ``ivf_rabitq``: rows in the lists of
    their nearest k-means center, their residuals rotated by a random
    orthogonal matrix and stored as ``bits_per_dim``-bit codes (RaBitQ at 1
    bit, Extended RaBitQ above) with two factors per row that make the
    distance estimates unbiased. At 3 bits a row of SIFT takes 56 bytes
    against 512 as float32.

    Example::

        index = mlxvs.IvfRabitqIndex.build(vectors, n_lists=1024, bits_per_dim=4)
        ids, dist = index.search(queries, k=10, n_probes=20, mode="quant8")
    """

    _FIELDS = (
        "centers",
        "centers_rot",
        "rotation",
        "codes",
        "factors",
        "indices",
        "offsets",
    )

    def __init__(
        self,
        centers,
        centers_rot,
        rotation,
        codes,
        factors,
        indices,
        offsets,
        metric,
        bits_per_dim,
        scale,
    ):
        metric = getattr(metric, "name", metric)
        if metric not in ("l2", "cosine", "inner_product"):
            raise ValueError(f"unknown metric {metric!r}")
        self._centers, self._centers_rot, self._rotation = (
            centers,
            centers_rot,
            rotation,
        )
        self._codes, self._factors = codes, factors
        self._indices, self._offsets = indices, offsets
        self._metric = metric
        self._bits = int(bits_per_dim)
        self._scale = float(scale)

    @classmethod
    def build(cls, rows, n_lists=1024, metric="l2", bits_per_dim=3, **kwargs):
        """Builds the index of the rows, as cuVS's ``ivf_rabitq::build``.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype.
            n_lists (int): k-means lists.
            metric (str): ``"l2"``, ``"cosine"`` or ``"inner_product"``.
            bits_per_dim (int): 1 to 9: more is larger and more accurate.
            **kwargs: ``kmeans_n_iters`` (20), ``max_train_points_per_cluster``
              (256), ``fast_quantize`` (True: one scale for every row, as
              cuVS's default), ``seed``.
        """
        metric = getattr(metric, "name", metric)
        parts = mlxvs._core.ivf_rabitq_build(
            mlxvs._array(rows),
            getattr(mlxvs.Metric, metric),
            n_lists=n_lists,
            bits_per_dim=bits_per_dim,
            **kwargs,
        )
        return cls(*parts[:7], metric, bits_per_dim, parts[7])

    def _parts(self):
        return (
            self._centers,
            self._centers_rot,
            self._rotation,
            self._codes,
            self._factors,
            self._indices,
            self._offsets,
            getattr(mlxvs.Metric, self._metric),
            self._bits,
            self._scale,
        )

    def search(self, queries, k=10, n_probes=20, mode="quant4"):
        """The k nearest rows of every query by RaBitQ's distance estimates,
        as cuVS's ``ivf_rabitq::search``.

        Args:
            mode (str): ``"quant4"`` (cuVS's default) or ``"quant8"``: the
              query quantized to 4 / 8 bits and scored with popcounts;
              ``"lut16"`` / ``"lut32"``: scored with the float query.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` and estimated
            distances, as ``brute_force_knn``.
        """
        q = mlxvs._array(queries)
        if q.ndim == 1:
            q = q[None]
        return mlxvs._core.ivf_rabitq_search(
            *self._parts(),
            q,
            k,
            n_probes=n_probes,
            mode=mlxvs._enum(mlxvs._core.RabitqMode, mode),
        )

    def extend(self, rows, ids=None):
        """Encodes rows into the lists of their nearest centers, with ``ids``
        or the largest id plus 1, 2, ...

        Returns:
            IvfRabitqIndex: this index, extended.
        """
        x = mlxvs._array(rows)
        if x.ndim == 1:
            x = x[None]
        new_ids = None if ids is None else mlxvs._array(ids).astype(mx.int32)
        self._codes, self._factors, self._indices, self._offsets = (
            mlxvs._core.ivf_rabitq_extend(*self._parts(), x, new_ids)
        )
        return self

    def save(self, path):
        """Saves the index to a ``.safetensors`` file."""
        path = str(path)
        if not path.endswith(".safetensors"):
            raise ValueError("the file name must end in .safetensors")
        arrays = {f: getattr(self, "_" + f) for f in self._FIELDS}
        mx.save_safetensors(
            path,
            arrays,
            metadata={
                "format": _RABITQ_FORMAT,
                "version": "1",
                "metric": self._metric,
                "bits_per_dim": str(self._bits),
                "scale": repr(self._scale),
            },
        )

    @classmethod
    def load(cls, path):
        """Loads an index saved by :meth:`save`."""
        arrays, meta = mx.load(str(path), return_metadata=True)
        if meta.get("format") != _RABITQ_FORMAT:
            raise ValueError(f"{path} is not a saved mlxvs IvfRabitqIndex")
        return cls(
            *[arrays[f] for f in cls._FIELDS],
            meta["metric"],
            int(meta["bits_per_dim"]),
            float(meta["scale"]),
        )

    @property
    def bits_per_dim(self):
        return self._bits

    @property
    def centers(self):
        return self._centers

    @property
    def n_lists(self):
        return self._centers.shape[0]

    @property
    def metric(self):
        return self._metric

    @property
    def dim(self):
        return self._centers.shape[1]

    def __len__(self):
        return self._codes.shape[0]

    def __repr__(self):
        return (
            f"IvfRabitqIndex(n={len(self)}, dim={self.dim}, n_lists={self.n_lists}, "
            f"bits_per_dim={self._bits}, metric={self._metric!r})"
        )
