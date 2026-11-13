# Copyright © 2026 mikepapadim

"""cuVS's binary quantizer: one bit per dimension, packed eight to a byte."""

import mlx.core as mx
import numpy as np

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx

__all__ = ["Quantizer", "QuantizerParams", "train", "transform"]


class QuantizerParams:
    """``threshold``: the value each dimension is compared with: ``"zero"``
    (cuVS's default), ``"mean"`` (the column's mean) or
    ``"sampling_median"`` (the median of a ``sampling_ratio`` sample), as
    cuVS's C++ ``binary::params``."""

    def __init__(self, *, threshold="zero", sampling_ratio=0.1):
        if threshold not in ("zero", "mean", "sampling_median"):
            raise ValueError('threshold must be "zero", "mean" or "sampling_median"')
        self.threshold = threshold
        self.sampling_ratio = sampling_ratio


class Quantizer:
    """A trained binary quantizer: the per-dimension thresholds (None for
    zero)."""

    def __init__(self, threshold=None):
        self.threshold = threshold


@auto_sync_resources
def train(params, dataset, resources=None):
    """The thresholds of the dataset, as cuVS's ``binary::train``."""
    x = to_mlx(dataset).astype(mx.float32)
    if params.threshold == "zero":
        return Quantizer()
    if params.threshold == "mean":
        return Quantizer(mx.mean(x, axis=0))
    n = x.shape[0]
    m = max(1, int(n * params.sampling_ratio))
    rows = np.sort(np.random.default_rng(0).choice(n, m, replace=False))
    return Quantizer(
        mx.median(x[mx.array(rows)], axis=0)
        if hasattr(mx, "median")
        else mx.array(np.median(np.asarray(x[mx.array(rows)]), axis=0))
    )


@auto_sync_resources
def transform(dataset_or_quantizer, dataset=None, output=None, resources=None):
    """Bits ``x > threshold`` packed eight to a byte, dimension ``j`` in bit
    ``j % 8`` of byte ``j // 8``: ``uint8 [n, ceil(dim / 8)]``, as cuVS's
    ``binary.transform(dataset)`` (threshold zero), or
    ``transform(quantizer, dataset)`` with trained thresholds."""
    if isinstance(dataset_or_quantizer, Quantizer):
        quantizer, x = dataset_or_quantizer, to_mlx(dataset)
    else:
        quantizer, x = Quantizer(), to_mlx(dataset_or_quantizer)
    x = x.astype(mx.float32)
    if quantizer.threshold is not None:
        x = x - quantizer.threshold[None]
    n, dim = x.shape
    words = -(-dim // 8)
    bits = mx.pad((x > 0).astype(mx.uint8), [(0, 0), (0, words * 8 - dim)])
    weights = mx.left_shift(mx.array(1, mx.uint8), mx.arange(8, dtype=mx.uint8))
    packed = mx.sum(bits.reshape(n, words, 8) * weights, axis=2).astype(mx.uint8)
    return _out(packed, output)
