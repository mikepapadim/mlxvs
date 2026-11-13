# Copyright © 2026 mikepapadim

"""Filtered search of the IVF indexes: the scans have no filter, so the
nearest rows are fetched in growing batches until k pass the filter."""

import mlx.core as mx
import numpy as np


def filtered_search(search, mask, k, n_rows):
    """``search(kk)`` -> (ids, scores); keep the first k ids that pass
    ``mask`` (bool ``[n_rows]`` or ``[n_queries, n_rows]``, or bitset
    words)."""
    if mask.dtype == mx.uint32:
        bits = np.unpackbits(np.asarray(mask).view(np.uint8), bitorder="little")[
            :n_rows
        ]
        mask = mx.array(bits.astype(np.bool_))
    allowed = np.asarray(mask)
    kk = k
    while True:
        kk = min(n_rows, max(2 * kk, 32))
        ids, scores = search(kk)
        ids = np.asarray(ids).astype(np.int64)
        scores = np.asarray(scores)
        valid = ids >= 0
        safe = np.where(valid, ids, 0)
        ok = valid & (
            allowed[safe]
            if allowed.ndim == 1
            else np.take_along_axis(allowed, safe, axis=1)
        )
        if kk >= n_rows or np.all(ok.sum(axis=1) >= k):
            break
    out_ids = np.full((ids.shape[0], k), -1, np.int64)
    out_scores = np.full((ids.shape[0], k), np.nan, np.float32)
    for r in range(ids.shape[0]):
        sel = np.flatnonzero(ok[r])[:k]
        out_ids[r, : len(sel)] = ids[r, sel]
        out_scores[r, : len(sel)] = scores[r, sel]
    return mx.array(out_ids), mx.array(out_scores)


def save_any_name(index, filename):
    """mlxvs saves safetensors files, whose names end in .safetensors: save
    there and move the file to the name asked for."""
    import os

    if str(filename).endswith(".safetensors"):
        index.save(filename)
        return
    tmp = str(filename) + ".tmp.safetensors"
    index.save(tmp)
    os.replace(tmp, filename)


def load_any_name(cls, filename):
    import os
    import tempfile

    if str(filename).endswith(".safetensors"):
        return cls.load(filename)
    with tempfile.TemporaryDirectory() as d:
        link = os.path.join(d, "index.safetensors")
        os.symlink(os.path.abspath(filename), link)
        return cls.load(link)
