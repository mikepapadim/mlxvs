# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.preprocessing.pca``: principal component analysis from the
covariance matrix (computed on the GPU, eigendecomposed on the CPU)."""

from collections import namedtuple

import mlx.core as mx
import numpy as np

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx

__all__ = ["Params", "fit", "fit_transform", "inverse_transform", "transform"]

FitOutput = namedtuple(
    "FitOutput",
    [
        "components",
        "explained_var",
        "explained_var_ratio",
        "singular_vals",
        "mu",
        "noise_vars",
    ],
)
FitTransformOutput = namedtuple(
    "FitTransformOutput",
    [
        "trans_input",
        "components",
        "explained_var",
        "explained_var_ratio",
        "singular_vals",
        "mu",
        "noise_vars",
    ],
)


class Params:
    """PCA parameters, as cuVS's: ``n_components`` (1), ``copy``,
    ``whiten`` (scale the projections to unit variance), ``algorithm``,
    ``tol``, ``n_iterations`` (the eigensolver is LAPACK's either way)."""

    def __init__(
        self,
        *,
        n_components=None,
        copy=None,
        whiten=None,
        algorithm=None,
        tol=None,
        n_iterations=None,
    ):
        self.n_components = 1 if n_components is None else n_components
        self.copy = True if copy is None else copy
        self.whiten = False if whiten is None else whiten
        self.algorithm = "cov_eig_dq" if algorithm is None else algorithm
        if self.algorithm not in ("cov_eig_dq", "cov_eig_jacobi"):
            raise ValueError('algorithm must be "cov_eig_dq" or "cov_eig_jacobi"')
        self.tol = 0.0 if tol is None else tol
        self.n_iterations = 15 if n_iterations is None else n_iterations


def _fit(params, x):
    n, d = x.shape
    k = params.n_components
    if not 0 < k <= d:
        raise ValueError("n_components must be in [1, n_features]")
    mu = mx.mean(x, axis=0)
    xc = x - mu[None]
    cov = (xc.T @ xc) / max(n - 1, 1)
    w, v = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(w)[::-1]
    w, v = np.maximum(w[order], 0), v[:, order]
    comps = v[:, :k].T
    # sign: the largest entry of each component positive (sklearn's svd_flip
    # on the components)
    signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
    signs[signs == 0] = 1
    comps = comps * signs[:, None]
    total = w.sum()
    var = w[:k]
    noise = w[k:].mean() if k < min(n, d) and len(w[k:]) else 0.0
    return (
        mx.array(comps.astype(np.float32)),
        mx.array(var.astype(np.float32)),
        mx.array((var / total if total > 0 else np.zeros_like(var)).astype(np.float32)),
        mx.array(np.sqrt(var * max(n - 1, 1)).astype(np.float32)),
        mu,
        mx.array(np.array([noise], dtype=np.float32)),
    )


def _project(params, x, comps, singular_vals, mu, n_rows):
    t = (x - mu[None]) @ comps.T
    if params.whiten:
        t = t * (np.sqrt(max(n_rows - 1, 1)) / singular_vals[None])
    return t


@auto_sync_resources
def fit(params, X, resources=None):
    """The principal components of X, as cuVS's ``pca.fit``: a
    ``FitOutput`` of ``components [n_components, n_features]``,
    ``explained_var``, ``explained_var_ratio``, ``singular_vals``, ``mu``
    and ``noise_vars``."""
    x = to_mlx(X).astype(mx.float32)
    return FitOutput(*[_out(a) for a in _fit(params, x)])


@auto_sync_resources
def fit_transform(params, X, resources=None):
    """:func:`fit`, and X projected on the components."""
    x = to_mlx(X).astype(mx.float32)
    parts = _fit(params, x)
    t = _project(params, x, parts[0], parts[3], parts[4], x.shape[0])
    return FitTransformOutput(_out(t), *[_out(a) for a in parts])


@auto_sync_resources
def transform(
    params, X, components, singular_vals, mu, trans_input=None, resources=None
):
    """X projected on the components, as cuVS's ``pca.transform``."""
    x = to_mlx(X).astype(mx.float32)
    sv = to_mlx(singular_vals).astype(mx.float32)
    t = _project(
        params, x, to_mlx(components).astype(mx.float32), sv, to_mlx(mu), x.shape[0]
    )
    return _out(t, trans_input)


@auto_sync_resources
def inverse_transform(
    params, trans_input, components, singular_vals, mu, output=None, resources=None
):
    """Projections back to the original space, as cuVS's
    ``pca.inverse_transform``."""
    t = to_mlx(trans_input).astype(mx.float32)
    comps = to_mlx(components).astype(mx.float32)
    if params.whiten:
        sv = to_mlx(singular_vals).astype(mx.float32)
        t = t * (sv[None] / np.sqrt(max(t.shape[0] - 1, 1)))
    return _out(t @ comps + to_mlx(mu)[None], output)
