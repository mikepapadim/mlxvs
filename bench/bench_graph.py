# Copyright © 2026 mikepapadim

"""k-NN graph build time and recall: mlxvs against CPU baselines.

Builds the k-NN graph of the rows (every row's k nearest other rows) with
mlxvs's nn_descent and all_neighbors, and optionally with FAISS (exact, CPU)
and PyNNDescent (NN-Descent, CPU, as UMAP uses), and scores a random sample of
rows against the exact neighbors. Recall counts an entry when it is at least as
close as the exact k-th, so ties are not misses.

Rows come from an .fvecs file (as SIFT1M's sift_base.fvecs) or a raw
little-endian float32 file with --dim.

    python bench/bench_graph.py sift_base.fvecs --k 32 --clusters 1,4,8 \\
        --baselines faiss,pynndescent
"""

import argparse
import os
import time

import mlx.core as mx
import mlxvs
import numpy as np


def load(path, dim):
    if path.endswith(".fvecs"):
        raw = np.fromfile(path, dtype=np.int32)
        d = int(raw[0])
        return raw.reshape(-1, d + 1)[:, 1:].view(np.float32)
    if not dim:
        raise SystemExit("--dim is required for a raw float32 file")
    return np.fromfile(path, dtype=np.float32).reshape(-1, dim)


def drop_self(ids, dist, k, rows):
    """k + 1 lists without the row itself (or the last entry, if absent)."""
    keep = ids != rows[:, None]
    keep[keep.sum(1) > k, -1] = False
    return ids[keep].reshape(-1, k), dist[keep].reshape(-1, k)


def recall(metric, ids, dist, exact_dist, k):
    kth = exact_dist[:, k - 1 : k]
    tol = 1e-5 * np.maximum(1.0, np.abs(kth))
    hit = dist <= kth + tol if metric == "l2" else dist >= kth - tol
    return float((hit & (ids >= 0)).sum() / ids.size)


def mlxvs_builds(x, a):
    yield "mlxvs nn_descent", lambda: mlxvs.nn_descent(x, a.k, a.metric)
    for c in [int(c) for c in a.clusters.split(",") if c]:
        yield (
            f"mlxvs all_neighbors clusters={c} overlap={a.overlap}",
            lambda c=c: mlxvs.all_neighbors(
                x, a.k, a.metric, n_clusters=c, overlap_factor=a.overlap
            ),
        )


def faiss_build(base, a):
    import faiss

    rows = base.copy()
    if a.metric == "cosine":
        faiss.normalize_L2(rows)
    index = faiss.IndexFlatL2(rows.shape[1])
    if a.metric != "l2":
        index = faiss.IndexFlatIP(rows.shape[1])
    index.add(rows)
    dist, ids = index.search(rows, a.k + 1)
    return drop_self(ids, dist, a.k, np.arange(len(rows)))


def pynndescent_build(base, a):
    from pynndescent import NNDescent

    metric = {"l2": "sqeuclidean", "cosine": "cosine", "inner_product": "dot"}
    index = NNDescent(
        base, metric=metric[a.metric], n_neighbors=a.k + 1, n_jobs=-1, verbose=False
    )
    ids, dist = index.neighbor_graph
    if a.metric != "l2":
        dist = 1 - dist  # PyNNDescent's cosine/dot distances to similarities
    return drop_self(ids, dist, a.k, np.arange(len(base)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rows", help=".fvecs file, or raw float32 rows with --dim")
    p.add_argument("--dim", type=int, default=0)
    p.add_argument("--n", type=int, default=0, help="first n rows (0: all)")
    p.add_argument("--metric", default="l2", choices=["l2", "cosine", "inner_product"])
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--clusters", default="1,4,8", help="all_neighbors n_clusters")
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument(
        "--baselines", default="", help="comma-separated: faiss, pynndescent"
    )
    p.add_argument("--sample", type=int, default=2000, help="rows scored")
    p.add_argument("--runs", type=int, default=1, help="timed runs of mlxvs builds")
    a = p.parse_args()

    base = load(a.rows, a.dim)
    if a.n:
        base = base[: a.n]
    base = np.ascontiguousarray(base, dtype=np.float32)
    n, d = base.shape
    x = mx.array(base)
    mx.eval(x)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(n, size=min(a.sample, n), replace=False))
    eids, edist = mlxvs.brute_force_knn(x, x[mx.array(sample)], a.k + 1, a.metric)
    eids, edist = np.array(eids), np.array(edist)
    _, exact_dist = drop_self(eids, edist, a.k, sample)
    print(
        f"{os.path.basename(a.rows)}: n={n} d={d} {a.metric} k={a.k}, "
        f"recall over {len(sample)} rows"
    )

    def report(label, seconds, ids, dist):
        ids, dist = np.asarray(ids)[sample], np.asarray(dist)[sample]
        r = recall(a.metric, ids, dist, exact_dist, a.k)
        print(f"  {label:44s} {seconds:8.2f} s  recall@{a.k} {r:.4f}", flush=True)

    mlxvs.nn_descent(x[:4096], a.k, a.metric)  # compile the kernels
    for label, build in mlxvs_builds(x, a):
        times = []
        for _ in range(a.runs):
            t = time.perf_counter()
            ids, dist = build()
            mx.eval(ids, dist)
            times.append(time.perf_counter() - t)
        report(label, min(times), ids, dist)

    baselines = {"faiss": faiss_build, "pynndescent": pynndescent_build}
    for name in [b for b in a.baselines.split(",") if b]:
        baselines[name](base[:4096], a)  # warm-up (PyNNDescent's numba compile)
        t = time.perf_counter()
        ids, dist = baselines[name](base, a)
        seconds = time.perf_counter() - t
        report(f"{name} (CPU, {os.cpu_count()} threads)", seconds, ids, dist)


if __name__ == "__main__":
    main()
