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
