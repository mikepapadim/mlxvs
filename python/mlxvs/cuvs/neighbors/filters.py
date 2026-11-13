# Copyright © 2026 mikepapadim

"""cuVS's ``cuvs.neighbors.filters``: rows a search may return."""

import numpy as np

__all__ = ["Prefilter", "from_bitmap", "from_bitset", "no_filter"]


class Prefilter:
    """A filter of the rows: ``type`` is ``"none"``, ``"bitset"`` (one bit per
    dataset row, set when the row may be returned) or ``"bitmap"`` (one bit
    per query and row, row-major)."""

    def __init__(self, type="none", array=None, parent=None):
        self.type = type
        self.array = array
        self.parent = parent

    def __repr__(self):
        return f"Prefilter(type={self.type!r})"


def _words(a):
    if hasattr(a, "copy_to_host"):
        a = a.copy_to_host()
    a = np.asarray(a)
    if a.dtype == np.bool_:
        a = np.packbits(a.ravel(), bitorder="little")
        a = np.pad(a, (0, -len(a) % 4)).view(np.uint32)
    return a.astype(np.uint32, copy=False)


def no_filter():
    """A filter that keeps every row."""
    return Prefilter()


def from_bitset(bitset):
    """A filter of uint32 words with one bit per dataset row (bit ``i % 32``
    of word ``i // 32``), set when row ``i`` may be returned. A bool mask of
    the rows is packed into words."""
    return Prefilter("bitset", _words(bitset), bitset)


def from_bitmap(bitmap):
    """A filter of uint32 words with one bit per (query, row), row-major
    (``n_queries * n_rows`` bits), set when the row may be returned for the
    query. A bool ``[n_queries, n_rows]`` mask is packed into words."""
    return Prefilter("bitmap", _words(bitmap), bitmap)
