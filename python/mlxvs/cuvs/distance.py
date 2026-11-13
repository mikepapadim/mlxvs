# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.distance``: pairwise distances on the GPU of Apple silicon."""

import mlx.core as mx

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx

__all__ = ["DISTANCE_NAMES", "DISTANCE_TYPES", "pairwise_distance"]

# cuvs::distance::DistanceType
L2Expanded = 0
L2SqrtExpanded = 1
CosineExpanded = 2
L1 = 3
L2Unexpanded = 4
L2SqrtUnexpanded = 5
InnerProduct = 6
Linf = 7
Canberra = 8
LpUnexpanded = 9
CorrelationExpanded = 10
JaccardExpanded = 11
HellingerExpanded = 12
Haversine = 13
BrayCurtis = 14
JensenShannon = 15
HammingUnexpanded = 16
KLDivergence = 17
RusselRaoExpanded = 18
DiceExpanded = 19
BitwiseHamming = 20

DISTANCE_TYPES = {
    "l2": L2SqrtExpanded,
    "sqeuclidean": L2Expanded,
    "euclidean": L2SqrtExpanded,
    "l1": L1,
    "cityblock": L1,
    "inner_product": InnerProduct,
    "chebyshev": Linf,
    "canberra": Canberra,
    "cosine": CosineExpanded,
    "lp": LpUnexpanded,
    "correlation": CorrelationExpanded,
    "jaccard": JaccardExpanded,
    "hellinger": HellingerExpanded,
    "braycurtis": BrayCurtis,
    "jensenshannon": JensenShannon,
    "hamming": HammingUnexpanded,
    "kl_divergence": KLDivergence,
    "minkowski": LpUnexpanded,
    "russellrao": RusselRaoExpanded,
    "dice": DiceExpanded,
    "bitwise_hamming": BitwiseHamming,
}

DISTANCE_NAMES = {v: k for k, v in DISTANCE_TYPES.items()}

SUPPORTED_DISTANCES = [
    "euclidean",
    "l1",
    "cityblock",
    "l2",
    "inner_product",
    "chebyshev",
    "minkowski",
    "canberra",
    "kl_divergence",
    "correlation",
    "russellrao",
    "hellinger",
    "lp",
    "hamming",
    "jensenshannon",
    "cosine",
    "sqeuclidean",
]

# elementwise metrics broadcast [rows of X, rows of Y, k]: this many floats at
# most per block
_BLOCK_FLOATS = 1 << 26


def _metric_type(metric):
    if isinstance(metric, str):
        if metric not in DISTANCE_TYPES:
            raise ValueError(f"metric {metric} is not supported")
        return DISTANCE_TYPES[metric]
    return int(metric)


def _dot_metric(x, y, t):
    """Metrics computed from the matrix product, as cuVS's expanded ones."""
    k = x.shape[1]
    if t == InnerProduct:
        return x @ y.T
    if t in (L2Expanded, L2SqrtExpanded, L2Unexpanded, L2SqrtUnexpanded):
        xn = mx.sum(x * x, axis=1, keepdims=True)
        yn = mx.sum(y * y, axis=1)[None]
        d = mx.maximum(xn + yn - 2 * (x @ y.T), 0)
        return mx.sqrt(d) if t in (L2SqrtExpanded, L2SqrtUnexpanded) else d
    if t == CosineExpanded:
        xn = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
        yn = mx.sqrt(mx.sum(y * y, axis=1))[None]
        return 1 - (x @ y.T) / (xn * yn)
    if t == CorrelationExpanded:
        sx = mx.sum(x, axis=1, keepdims=True)
        sy = mx.sum(y, axis=1)[None]
        sx2 = mx.sum(x * x, axis=1, keepdims=True)
        sy2 = mx.sum(y * y, axis=1)[None]
        numer = k * (x @ y.T) - sx * sy
        return 1 - numer / mx.sqrt((k * sx2 - sx * sx) * (k * sy2 - sy * sy))
    if t == HellingerExpanded:
        final = 1 - mx.sqrt(x) @ mx.sqrt(y).T
        return mx.sqrt(mx.maximum(final, 0))
    if t == RusselRaoExpanded:
        return (k - x @ y.T) / k
    return None


def _elementwise(x, y, t, p):
    """Metrics of the elementwise differences, over a block of X's rows."""
    a = x[:, None, :]
    b = y[None, :, :]
    if t == L1:
        return mx.sum(mx.abs(a - b), axis=2)
    if t == Linf:
        return mx.max(mx.abs(a - b), axis=2)
    if t == Canberra:
        diff = mx.abs(a - b)
        add = mx.abs(a) + mx.abs(b)
        return mx.sum(mx.where(add != 0, diff / mx.where(add == 0, 1, add), 0), axis=2)
    if t == LpUnexpanded:
        return mx.power(mx.sum(mx.power(mx.abs(a - b), p), axis=2), 1.0 / p)
    if t == HammingUnexpanded:
        return mx.mean((a != b).astype(mx.float32), axis=2)
    if t == JensenShannon:
        m = 0.5 * (a + b)
        log_m = mx.where(m == 0, 0, mx.log(mx.where(m == 0, 1, m)))
        log_a = mx.log(mx.where(a == 0, 1, a))
        log_b = mx.log(mx.where(b == 0, 1, b))
        acc = mx.sum(-a * (log_m - log_a) - b * (log_m - log_b), axis=2)
        return mx.sqrt(0.5 * acc)
    if t == KLDivergence:
        log_a = mx.log(mx.where(a == 0, 1, a))
        log_b = mx.where(b == 0, 0, mx.log(mx.where(b == 0, 1, b)))
        return 0.5 * mx.sum(a * (log_a - log_b), axis=2)
    if t == BrayCurtis:
        return mx.sum(mx.abs(a - b), axis=2) / mx.sum(mx.abs(a + b), axis=2)
    raise ValueError(f"metric {DISTANCE_NAMES.get(t, t)} is not supported")


def _pairwise(x, y, t, p=2.0):
    """``[m, n]`` float32 distances of the rows of x to the rows of y."""
    x = x.astype(mx.float32)
    y = y.astype(mx.float32)
    d = _dot_metric(x, y, t)
    if d is not None:
        return d
    m, n, k = x.shape[0], y.shape[0], x.shape[1]
    step = max(1, _BLOCK_FLOATS // max(1, n * k))
    parts = []
    for i in range(0, m, step):
        part = _elementwise(x[i : i + step], y, t, p)
        mx.eval(part)
        parts.append(part)
    return mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


@auto_sync_resources
def pairwise_distance(X, Y, out=None, metric="euclidean", p=2.0, resources=None):
    """Distances between every row of X and every row of Y, as cuVS's
    ``pairwise_distance``.

    Valid values for metric: ``"euclidean"``, ``"l2"``, ``"l1"``,
    ``"cityblock"``, ``"inner_product"``, ``"chebyshev"``, ``"canberra"``,
    ``"lp"``, ``"hellinger"``, ``"jensenshannon"``, ``"kl_divergence"``,
    ``"russellrao"``, ``"minkowski"``, ``"correlation"``, ``"cosine"``,
    ``"sqeuclidean"``, ``"hamming"``.

    Args:
        X: Rows ``[m, k]``, float32 or float16.
        Y: Rows ``[n, k]``, the same dtype.
        out: Optional ``[m, n]`` array to write into.
        metric (str): The distance.
        p (float): The exponent of ``"minkowski"`` / ``"lp"``.

    Returns:
        ``[m, n]`` distances (``out`` when given).
    """
    x = to_mlx(X)
    y = to_mlx(Y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Inputs must have same number of columns. a={x.shape[-1]}, b={y.shape[-1]}"
        )
    if metric not in SUPPORTED_DISTANCES:
        raise ValueError(f"metric {metric} is not supported")
    if x.dtype != y.dtype:
        raise ValueError("Inputs must have the same dtypes")
    d = _pairwise(x, y, DISTANCE_TYPES[metric], p)
    if x.dtype == mx.float16:
        d = d.astype(mx.float16)
    return _out(d, out)
