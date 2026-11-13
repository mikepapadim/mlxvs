# Copyright © 2026 mikepapadim

"""cuVS's index files: every scalar and array framed as a NumPy ``.npy``
record (RAFT's ``numpy_serializer``), little endian."""

import ast
import struct

import numpy as np

_MAGIC = b"\x93NUMPY"

# RAFT's dtype strings: one-byte integers (and bool) have no byte order, half
# is "<e2"
_DESCR = {
    np.dtype("float32"): "<f4",
    np.dtype("float64"): "<f8",
    np.dtype("float16"): "<e2",
    np.dtype("int8"): "|i1",
    np.dtype("uint8"): "|u1",
    np.dtype("bool"): "|u1",
    np.dtype("int32"): "<i4",
    np.dtype("uint32"): "<u4",
    np.dtype("int64"): "<i8",
    np.dtype("uint64"): "<u8",
}
_FROM_DESCR = {
    "<f4": np.float32,
    "<f8": np.float64,
    "<e2": np.float16,
    "<f2": np.float16,
    "|i1": np.int8,
    "|u1": np.uint8,
    "<i4": np.int32,
    "<u4": np.uint32,
    "<i8": np.int64,
    "<u8": np.uint64,
    "|b1": np.bool_,
}


def write_header(f, dtype, shape):
    """RAFT's ``write_header``: the dict, then spaces up to a multiple of 64
    bytes (1 to 64 of them), then a newline."""
    if len(shape) == 0:
        tup = "()"
    elif len(shape) == 1:
        tup = f"({shape[0]},)"
    else:
        tup = "(" + ", ".join(str(int(s)) for s in shape) + ")"
    d = f"{{'descr': '{_DESCR[np.dtype(dtype)]}', 'fortran_order': False, 'shape': {tup}}}"
    preamble = len(_MAGIC) + 2 + 2 + len(d) + 1
    pad = 64 - preamble % 64
    f.write(_MAGIC + b"\x01\x00")
    f.write(struct.pack("<H", len(d) + pad + 1))
    f.write(d.encode() + b" " * pad + b"\n")


def write_scalar(f, value, dtype):
    write_header(f, dtype, ())
    f.write(np.array(value, dtype=np.dtype(dtype)).tobytes())


def write_array(f, a, dtype=None):
    a = np.ascontiguousarray(a, dtype=dtype)
    write_header(f, a.dtype, a.shape)
    f.write(a.tobytes())


def read_header(f):
    magic = f.read(8)
    if magic[:6] != _MAGIC:
        raise ValueError("not a cuVS index file: missing a NumPy record")
    (n,) = struct.unpack("<H", f.read(2))
    d = ast.literal_eval(f.read(n).decode("latin1").strip())
    if d.get("fortran_order"):
        raise ValueError("Fortran-ordered arrays are not supported")
    return np.dtype(_FROM_DESCR[d["descr"]]), tuple(d["shape"])


def read_scalar(f):
    dtype, shape = read_header(f)
    if shape:
        raise ValueError(f"expected a scalar, found shape {shape}")
    return np.frombuffer(f.read(dtype.itemsize), dtype=dtype)[0].item()


def read_array(f):
    dtype, shape = read_header(f)
    count = int(np.prod(shape)) if shape else 1
    a = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype)
    if a.size != count:
        raise ValueError("truncated cuVS index file")
    return a.reshape(shape)


def dtype_prefix(dtype, version):
    """The 4-byte dtype prefix of an index file: half is "<f2" from CAGRA
    format 6 on, "<e2" before."""
    s = _DESCR[np.dtype(dtype)]
    if s == "<e2" and version >= 6:
        s = "<f2"
    return s.encode().ljust(4, b"\0")


def read_prefix(f):
    s = f.read(4).rstrip(b"\0").decode()
    if s not in _FROM_DESCR:
        raise ValueError(f"unknown dtype prefix {s!r}")
    return np.dtype(_FROM_DESCR[s])


# cudaDataType_t of the dataset payload
CUDA_TYPES = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 2,
    np.dtype("int8"): 3,
    np.dtype("uint8"): 8,
}
