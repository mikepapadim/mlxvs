# Copyright © 2026 mikepapadim

"""cuVS's scalar quantizer: floats to int8 by a linear map of the interval
between two quantiles of the data onto [-128, 127]."""

import math

import mlx.core as mx
import numpy as np

from mlxvs.cuvs.common import _out, auto_sync_resources, to_mlx

__all__ = ["Quantizer", "QuantizerParams", "inverse_transform", "train", "transform"]

# cuVS samples at most this many values to find the quantiles
_MAX_SAMPLES = 1000000


class QuantizerParams:
    """``quantile``: the share of values inside the interval mapped (the
    rest, half at each end, clamp to -128 and 127), in (0, 1]; cuVS's
    default 0.99."""

    def __init__(self, *, quantile=None):
        self.quantile = 0.99 if quantile is None else float(quantile)


class Quantizer:
    """A trained scalar quantizer: the interval ``[min, max]``."""

    def __init__(self, min=0.0, max=1.0, dtype=np.float32):
        self.min = min
        self.max = max
        self._dtype = np.dtype(dtype)

    def _scale_offset(self):
        lo, hi = float(self.min), float(self.max)
        scalar = 255.0 / (hi - lo) if hi > lo else 1.0
        return scalar, -128.0 - lo * scalar

    def __repr__(self):
        return f"scalar.Quantizer(min={self.min}, max={self.max})"


@auto_sync_resources
def train(params, dataset, resources=None):
    """The interval of the values between the quantiles, as cuVS's
    ``scalar.train``: a sample of rows (all of them up to a million values),
    sorted, the values at ranks ``ceil((0.5 + q / 2) * n) - 1`` and its
    mirror."""
    q = params.quantile
    if not 0.0 < q <= 1.0:
        raise ValueError(
            f"quantile for scalar quantization needs to be within (0, 1] but is {q:f}"
        )
    x = to_mlx(dataset)
    n_rows, dim = x.shape
    n_sample = min(math.ceil(_MAX_SAMPLES / dim), n_rows)
    if n_sample < n_rows:
        rows = np.sort(np.random.default_rng(0).choice(n_rows, n_sample, replace=False))
        x = x[mx.array(rows)]
    values = mx.sort(x.astype(mx.float32).reshape(-1))
    size = values.shape[0]
    pos_max = math.ceil((0.5 + 0.5 * q) * size) - 1
    pos_min = size - pos_max - 1
    dtype = np.float16 if x.dtype == mx.float16 else np.float32
    return Quantizer(values[pos_min].item(), values[pos_max].item(), dtype)


@auto_sync_resources
def transform(quantizer, dataset, output=None, resources=None):
    """The rows as int8, as cuVS's ``scalar.transform``: values at or below
    ``min`` become -128, at or above ``max`` 127, the rest
    ``round(scalar * x + offset)``."""
    x = to_mlx(dataset).astype(mx.float32)
    scalar, offset = quantizer._scale_offset()
    lo, hi = float(quantizer.min), float(quantizer.max)
    v = scalar * x + offset
    # lroundf: halves away from zero
    r = mx.sign(v) * mx.floor(mx.abs(v) + 0.5)
    q = mx.where(x <= lo, -128, mx.where(x >= hi, 127, r))
    return _out(mx.clip(q, -128, 127).astype(mx.int8), output)


@auto_sync_resources
def inverse_transform(quantizer, dataset, output=None, resources=None):
    """int8 codes back to floats, as cuVS's ``scalar.inverse_transform``."""
    q = to_mlx(dataset).astype(mx.float32)
    scalar, offset = quantizer._scale_offset()
    x = (q - offset) / scalar
    dt = mx.float16 if quantizer._dtype == np.float16 else mx.float32
    return _out(x.astype(dt), output)
