# Copyright © 2026 mikepapadim

"""Exact ground truth for an ANN benchmark, written as .ivecs.

The k exact nearest base rows of every query, the file every ANN benchmark
needs and the slow part of preparing a dataset on a CPU.

    python examples/ground_truth.py sift_base.fvecs sift_query.fvecs gt.ivecs \\
        --k 100 --metric l2
"""

import argparse
import time

import mlxvs
import numpy as np


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:]).view(np.float32)


def write_ivecs(path, ids):
    n, k = ids.shape
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = ids
    out.tofile(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", help="base rows, .fvecs")
    p.add_argument("queries", help="query rows, .fvecs")
    p.add_argument("out", help="ground truth to write, .ivecs")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--metric", default="l2", choices=["l2", "cosine", "inner_product"])
    a = p.parse_args()

    base, queries = read_fvecs(a.base), read_fvecs(a.queries)
    t = time.perf_counter()
    ids, _ = mlxvs.brute_force_knn(base, queries, a.k, a.metric)
    ids = np.asarray(ids)
    seconds = time.perf_counter() - t
    write_ivecs(a.out, ids)
    print(
        f"{len(queries)} queries x {len(base)} base rows, k={a.k}, {a.metric}: "
        f"{seconds:.2f} s -> {a.out}"
    )


if __name__ == "__main__":
    main()
