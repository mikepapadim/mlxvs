# Copyright © 2026 mikepapadim

"""The arrays and resources of the cuVS Python API, on MLX.

cuVS takes and returns ``pylibraft.common.device_ndarray`` (CUDA memory).
Here a :class:`device_ndarray` holds an ``mlx.core.array`` in the Mac's
unified memory, with the same ``copy_to_host()``, ``shape`` and ``dtype``, so
code written against cuVS runs unchanged. Inputs may be ``device_ndarray``,
``mlx.core.array``, NumPy arrays, or anything ``numpy.asarray`` accepts.
"""

import functools

import mlx.core as mx
import numpy as np

import mlxvs

__all__ = ["Resources", "auto_sync_resources", "device_ndarray", "set_output_as"]

_NP_TO_MX = {
    np.dtype("float32"): mx.float32,
    np.dtype("float16"): mx.float16,
    np.dtype("int8"): mx.int8,
    np.dtype("uint8"): mx.uint8,
    np.dtype("int32"): mx.int32,
    np.dtype("uint32"): mx.uint32,
    np.dtype("int64"): mx.int64,
    np.dtype("uint64"): mx.uint64,
    np.dtype("bool"): mx.bool_,
}


class Resources:
    """cuVS's resources handle (a CUDA stream there). MLX orders its own GPU
    work, so this only offers ``sync()``, which waits for it."""

    def __init__(self, stream=None):
        self.stream = stream

    def sync(self):
        mx.synchronize()

    def get_c_obj(self):
        return 0


def auto_sync_resources(f):
    """Runs ``f`` and waits for the GPU when ``resources`` was not passed, as
    cuVS's decorator of the same name."""

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        out = f(*args, resources=resources, **kwargs)
        if resources is None:
            mx.synchronize()
        return out

    return wrapper


class device_ndarray:
    """An array on the GPU, as ``pylibraft.common.device_ndarray``.

    ``device_ndarray(np_array)`` copies a host array to the GPU,
    ``copy_to_host()`` returns it as NumPy, and ``numpy.asarray`` and
    ``mlx.core.array`` read it without a copy.
    """

    def __init__(self, np_ndarray):
        if isinstance(np_ndarray, device_ndarray):
            self._array = np_ndarray._array
        else:
            self._array = to_mlx(np_ndarray)

    @classmethod
    def empty(cls, shape, dtype=np.float32, order="C"):
        out = cls.__new__(cls)
        out._array = mx.zeros(shape, dtype=_NP_TO_MX[np.dtype(dtype)])
        return out

    @classmethod
    def _wrap(cls, array):
        out = cls.__new__(cls)
        out._array = array
        return out

    @property
    def shape(self):
        return tuple(self._array.shape)

    @property
    def dtype(self):
        return np.dtype(_mx_to_np(self._array.dtype))

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def c_contiguous(self):
        return True

    @property
    def f_contiguous(self):
        return self._array.ndim < 2

    def copy_to_host(self):
        return np.array(self._array)

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self._array)
        return a if dtype is None else a.astype(dtype)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        return self.copy_to_host()[key]

    def __repr__(self):
        return f"device_ndarray(shape={self.shape}, dtype={self.dtype})"

    @property
    def array(self):
        """The ``mlx.core.array`` it holds."""
        return self._array


def _mx_to_np(dtype):
    for k, v in _NP_TO_MX.items():
        if v == dtype:
            return k
    return np.dtype("float32")


def to_mlx(x, dtype=None):
    """An ``mlx.core.array`` of any array-like input (no copy when possible)."""
    if isinstance(x, device_ndarray):
        a = x._array
    elif isinstance(x, mx.array):
        a = x
    else:
        if hasattr(x, "copy_to_host"):
            x = x.copy_to_host()
        elif hasattr(x, "get") and type(x).__module__.startswith("cupy"):
            x = x.get()
        a = mlxvs._array(x if type(x).__module__ == "numpy" else np.asarray(x))
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype)
    return a


_output_as = "raft"


def set_output_as(output):
    """How results are returned, as ``pylibraft.config.set_output_as``:
    ``"raft"`` (:class:`device_ndarray`, the default), ``"numpy"``,
    ``"mlx"``, ``"torch"``, or a callable taking a ``device_ndarray``."""
    global _output_as
    if output not in ("raft", "numpy", "mlx", "torch") and not callable(output):
        raise ValueError('output must be "raft", "numpy", "mlx", "torch" or a callable')
    _output_as = output


def _out(array, given=None):
    """Writes ``array`` into the caller's preallocated ``given``, or returns
    it as :func:`set_output_as` says."""
    if given is not None:
        if isinstance(given, device_ndarray):
            given._array = array.astype(given._array.dtype)
        elif isinstance(given, np.ndarray):
            given[...] = np.asarray(array).astype(given.dtype)
        elif hasattr(given, "__setitem__"):
            given[...] = np.asarray(array)
        return given
    if _output_as == "raft":
        return device_ndarray._wrap(array)
    if _output_as == "numpy":
        return np.array(array)
    if _output_as == "mlx":
        return array
    if _output_as == "torch":
        import torch

        return torch.from_numpy(np.array(array))
    return _output_as(device_ndarray._wrap(array))
