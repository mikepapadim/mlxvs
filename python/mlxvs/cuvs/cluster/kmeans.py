# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.cluster.kmeans``: k-means on the GPU of Apple silicon."""

import mlx.core as mx
import numpy as np

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx

__all__ = ["KMeansParams", "cluster_cost", "fit", "predict"]

_INIT_METHODS = ("KMeansPlusPlus", "Random", "Array")
# rows per block of the assignment step
_BLOCK = 1 << 16


class KMeansParams:
    """Parameters of :func:`fit` and :func:`predict`, as cuVS's (None:
    cuVS's default). ``init_method``: ``"KMeansPlusPlus"`` (default),
    ``"Random"`` or ``"Array"`` (the ``centroids`` passed to :func:`fit`).
    Batching, buffering and hierarchical options are kept for compatibility:
    unified memory needs no batching."""

    def __init__(
        self,
        *,
        metric=None,
        n_clusters=None,
        init_method=None,
        max_iter=None,
        tol=None,
        n_init=None,
        oversampling_factor=None,
        batch_samples=None,
        batch_centroids=None,
        init_size=None,
        device_buffer_samples=None,
        hierarchical=None,
        hierarchical_n_iters=None,
        seed=0,
    ):
        self.metric = "sqeuclidean" if metric is None else metric
        if self.metric not in ("sqeuclidean", "euclidean", "l2"):
            raise ValueError("k-means supports the sqeuclidean and euclidean metrics")
        self.n_clusters = 8 if n_clusters is None else n_clusters
        self.init_method = "KMeansPlusPlus" if init_method is None else init_method
        if self.init_method not in _INIT_METHODS:
            raise ValueError(f"init_method must be one of {_INIT_METHODS}")
        self.max_iter = 300 if max_iter is None else max_iter
        self.tol = 1e-4 if tol is None else tol
        self.n_init = 1 if n_init is None else n_init
        self.oversampling_factor = (
            2.0 if oversampling_factor is None else oversampling_factor
        )
        self.batch_samples = 1 << 15 if batch_samples is None else batch_samples
        self.batch_centroids = 0 if batch_centroids is None else batch_centroids
        self.init_size = 0 if init_size is None else init_size
        self.device_buffer_samples = device_buffer_samples or 0
        self.hierarchical = bool(hierarchical)
        self.hierarchical_n_iters = hierarchical_n_iters
        self.seed = seed


def _assign(x, c):
    """Nearest center and squared distance of every row."""
    cn = mx.sum(c * c, axis=1)[None]
    labels, dists = [], []
    for i in range(0, x.shape[0], _BLOCK):
        xb = x[i : i + _BLOCK]
        d = mx.maximum(mx.sum(xb * xb, axis=1, keepdims=True) + cn - 2 * (xb @ c.T), 0)
        lab = mx.argmin(d, axis=1)
        labels.append(lab)
        dists.append(mx.take_along_axis(d, lab[:, None], axis=1)[:, 0])
        mx.eval(labels[-1], dists[-1])
    return mx.concatenate(labels), mx.concatenate(dists)


def _kmeanspp(x, w, k, rng):
    """k-means++ seeding: each next center drawn with probability
    proportional to weight times squared distance to the nearest center."""
    n = x.shape[0]
    p = np.asarray(w, dtype=np.float64)
    first = rng.choice(n, p=p / p.sum())
    centers = [first]
    c = x[first][None]
    best = mx.maximum(mx.sum(mx.square(x - c), axis=1), 0)
    for _ in range(1, k):
        score = np.asarray(best * w, dtype=np.float64)
        total = score.sum()
        nxt = rng.choice(n, p=score / total) if total > 0 else rng.integers(n)
        centers.append(nxt)
        best = mx.minimum(best, mx.sum(mx.square(x - x[int(nxt)][None]), axis=1))
        mx.eval(best)
    return x[mx.array(np.asarray(centers))]


def _lloyd(x, w, c, max_iter, tol):
    k, d = c.shape
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        labels, _ = _assign(x, c)
        sums = mx.zeros((k, d), mx.float32).at[labels].add(x * w[:, None])
        counts = mx.zeros((k,), mx.float32).at[labels].add(w)
        new = mx.where(
            counts[:, None] > 0, sums / mx.maximum(counts[:, None], 1e-30), c
        )
        shift = mx.sum(mx.square(new - c)).item()
        c = new
        mx.eval(c)
        if shift < tol:
            break
    labels, dist = _assign(x, c)
    return c, mx.sum(dist * w).item(), n_iter


def _weights(sample_weights, n, normalize):
    if sample_weights is None:
        return mx.ones((n,), mx.float32)
    w = to_mlx(sample_weights).astype(mx.float32)
    if normalize:
        w = w * (n / mx.sum(w))
    return w


@auto_sync_resources
def fit(params, X, centroids=None, sample_weights=None, resources=None):
    """Fits k-means, as cuVS's ``kmeans.fit``: k-means++ seeding (or the
    given ``centroids``), then Lloyd iterations until the centers move less
    than ``tol`` (squared) or ``max_iter``; the best of ``n_init`` runs.

    Returns:
        tuple: ``(centroids, inertia, n_iter)``.
    """
    x = to_mlx(X).astype(mx.float32)
    n = x.shape[0]
    k = params.n_clusters
    if k > n:
        raise ValueError("n_clusters must be at most the number of rows")
    w = _weights(sample_weights, n, True)
    rng = np.random.default_rng(params.seed)
    best = None
    runs = 1 if params.init_method == "Array" else max(1, params.n_init)
    for _ in range(runs):
        if params.init_method == "Array" or (
            centroids is not None and params.init_method != "Random"
        ):
            if centroids is None:
                raise ValueError('init_method="Array" needs centroids')
            c0 = to_mlx(centroids).astype(mx.float32)
        elif params.init_method == "Random":
            c0 = x[mx.array(rng.choice(n, k, replace=False))]
        else:
            c0 = _kmeanspp(x, w, k, rng)
        c, inertia, n_iter = _lloyd(x, w, c0, params.max_iter, params.tol)
        if best is None or inertia < best[1]:
            best = (c, inertia, n_iter)
    c, inertia, n_iter = best
    return (
        _out(c, centroids if params.init_method == "Array" else None),
        inertia,
        n_iter,
    )


@auto_sync_resources
def predict(
    params,
    X,
    centroids,
    sample_weights=None,
    labels=None,
    normalize_weight=True,
    resources=None,
):
    """The nearest centroid of every row, as cuVS's ``kmeans.predict``.

    Returns:
        tuple: ``(labels, inertia)``, labels ``int32 [n]``.
    """
    x = to_mlx(X).astype(mx.float32)
    c = to_mlx(centroids).astype(mx.float32)
    lab, dist = _assign(x, c)
    w = _weights(sample_weights, x.shape[0], normalize_weight)
    return _out(lab.astype(mx.int32), labels), mx.sum(dist * w).item()


@auto_sync_resources
def cluster_cost(X, centroids, resources=None):
    """The sum of squared distances of the rows to their nearest centroid,
    as cuVS's ``kmeans.cluster_cost``."""
    _, dist = _assign(
        to_mlx(X).astype(mx.float32), to_mlx(centroids).astype(mx.float32)
    )
    return mx.sum(dist).item()
