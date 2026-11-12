# Copyright © 2026 mikepapadim

"""Graph search: recall@k against queries per second.

Builds a search graph of the base rows with mlxvs (cagra_build with
--intermediate, else the k-NN graph of nn_descent, or all_neighbors with
--clusters), computes the exact neighbors of the queries with
brute_force_knn, then searches the graph with graph_search at several
itopk_size values (and --algo layouts), all queries in one batch. Optionally
the same sweep with hnswlib (HNSW on the CPU, all cores) at several ef values.
With --latency N, the first N queries are instead searched one at a time
(median and p95 latency; hnswlib on one thread).

Recall counts a returned id when its true (float32) distance is at least as
close as the exact k-th, so ties are not misses.

    python bench/bench_search.py sift_base.fvecs sift_query.fvecs --metric l2 \\
        --degree 32 --baselines hnswlib
"""

import argparse
import os
import time

import mlx.core as mx
import mlxvs
import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    # contiguous: MLX keeps a NumPy view's strides, and a kernel would make
    # the rows contiguous again on every call
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:].view(np.float32))


def true_dist(metric, base, queries, ids):
    """The float32 distances of the returned ids, whatever the dataset dtype."""
    x = base[np.maximum(ids, 0)]
    q = queries[:, None, :]
    if metric == "l2":
        d = ((x - q) ** 2).sum(-1)
    else:
        d = (x * q).sum(-1)
        if metric == "cosine":
            d /= np.linalg.norm(x, axis=-1) * np.linalg.norm(q, axis=-1) + 1e-30
    return np.where(ids >= 0, d, np.inf if metric == "l2" else -np.inf)


def recall(metric, dist, exact_dist, k):
    kth = exact_dist[:, k - 1 : k]
    tol = 1e-5 * np.maximum(1.0, np.abs(kth))
    hit = dist[:, :k] <= kth + tol if metric == "l2" else dist[:, :k] >= kth - tol
    return float(hit.sum() / hit.size)


def timed(run, runs):
    run()  # compile and warm up
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        out = run()
        best = min(best, time.perf_counter() - t0)
    return out, best


def latencies(search, inputs):
    """Median and p95 milliseconds of searching each one-query input alone, and
    the ids."""
    search(inputs[0])  # compile and warm up
    times, ids = [], []
    for q in inputs:
        t0 = time.perf_counter()
        ids.append(search(q))
        times.append(time.perf_counter() - t0)
    ms = 1e3 * np.array(times)
    return np.median(ms), np.percentile(ms, 95), np.concatenate(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("base", help="base rows, .fvecs")
    p.add_argument("queries", help="query rows, .fvecs")
    p.add_argument("--metric", default="l2", choices=["l2", "cosine", "inner_product"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--degree", type=int, default=32, help="graph degree")
    p.add_argument("--clusters", type=int, default=-1, help="all_neighbors clusters")
    p.add_argument(
        "--intermediate",
        type=int,
        default=0,
        help="cagra_build with this intermediate_graph_degree (0: k-NN graph)",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="CagraIndex dataset dtype (with --intermediate)",
    )
    p.add_argument("--itopk", default="32,64,96,128,192,256")
    p.add_argument("--width", type=int, default=1, help="search_width")
    p.add_argument(
        "--algo",
        default="auto",
        help="comma-separated search layouts: auto, single_cta, multi_cta",
    )
    p.add_argument(
        "--latency", type=int, default=0, help="time this many queries one at a time"
    )
    p.add_argument("--nq", type=int, default=0, help="queries used, 0: all")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--baselines", default="", help="comma-separated: hnswlib")
    p.add_argument("--ef", default="16,32,64,128,256")
    a = p.parse_args()

    base = load(a.base)
    queries = load(a.queries)
    if a.nq:
        queries = queries[: a.nq]
    n, d = base.shape
    print(f"{n} x {d} base, {len(queries)} queries, {a.metric}, k={a.k}")
    xb = mx.array(base)
    xq = mx.array(queries)

    _, exact_dist = mlxvs.brute_force_knn(xb, xq, a.k, a.metric)
    exact_dist = np.asarray(exact_dist)

    t0 = time.perf_counter()
    if a.intermediate:
        kw = {"n_clusters": a.clusters} if a.clusters >= 0 else {}
        index = mlxvs.CagraIndex.build(
            xb,
            a.degree,
            a.metric,
            intermediate_graph_degree=a.intermediate,
            dtype=a.dtype,
            **kw,
        )
        graph = index.graph
        how = f"CagraIndex (intermediate degree {a.intermediate}, {a.dtype})"
    elif a.clusters >= 0:
        graph, _ = mlxvs.all_neighbors(xb, a.degree, a.metric, n_clusters=a.clusters)
        how = f"all_neighbors (clusters {a.clusters or 'auto'})"
    else:
        graph, _ = mlxvs.nn_descent(xb, a.degree, a.metric)
        how = "nn_descent"
    mx.eval(graph)
    print(f"graph: {how}, degree {a.degree}, {time.perf_counter() - t0:.1f} s\n")

    if a.latency:
        nl = min(a.latency, len(queries))
        lat_q = [mx.array(queries[i : i + 1]) for i in range(nl)]
        mx.eval(*lat_q)
        exact_lat = exact_dist[:nl]
        head = f"{'method':<34} {'param':>6} {'recall@' + str(a.k):>10}"
        print(f"{head} {'median ms':>10} {'p95 ms':>8}")
    else:
        print(f"{'method':<34} {'param':>6} {'recall@' + str(a.k):>10} {'QPS':>12}")
    for algo in [s for s in a.algo.split(",") if s]:
        for itopk in [int(s) for s in a.itopk.split(",") if s]:
            if itopk < a.k or (algo == "single_cta" and itopk > 256):
                continue  # single-CTA keeps at most 256 entries
            kw = {"itopk_size": itopk, "search_width": a.width, "algo": algo}

            def search(q, kw=kw):
                if a.intermediate:
                    ids, dist = index.search(q, a.k, **kw)
                else:
                    ids, dist = mlxvs.graph_search(xb, graph, q, a.k, a.metric, **kw)
                mx.eval(ids, dist)
                return np.asarray(ids)

            name = f"mlxvs {algo} w={a.width}"
            if a.latency:
                med, p95, ids = latencies(search, lat_q)
                dist = true_dist(a.metric, base, queries[:nl], ids)
                r = recall(a.metric, dist, exact_lat, a.k)
                print(f"{name:<34} {itopk:>6} {r:>10.4f} {med:>10.3f} {p95:>8.3f}")
                continue
            ids, sec = timed(lambda: search(xq), a.runs)
            # scored by the true distances of the returned ids, so that a
            # float16 dataset's rounded distances do not count near-ties as
            # misses
            dist = true_dist(a.metric, base, queries, ids)
            r = recall(a.metric, dist, exact_dist, a.k)
            print(f"{name:<34} {itopk:>6} {r:>10.4f} {len(queries) / sec:>12,.0f}")

    if "hnswlib" in a.baselines:
        import hnswlib

        space = {"l2": "l2", "cosine": "cosine", "inner_product": "ip"}[a.metric]
        hnsw = hnswlib.Index(space=space, dim=d)
        hnsw.init_index(max_elements=n, M=a.degree // 2, ef_construction=200)
        hnsw.set_num_threads(os.cpu_count())
        t0 = time.perf_counter()
        hnsw.add_items(base)
        print(
            f"\nhnswlib build (M={a.degree // 2}, efc 200): {time.perf_counter() - t0:.1f} s"
        )
        if a.latency:
            hnsw.set_num_threads(1)
        for ef in [int(s) for s in a.ef.split(",") if s]:
            if ef < a.k:
                continue
            hnsw.set_ef(ef)
            if a.latency:
                med, p95, labels = latencies(
                    lambda q: hnsw.knn_query(q, k=a.k)[0],
                    [queries[i : i + 1] for i in range(nl)],
                )
                dist = true_dist(a.metric, base, queries[:nl], labels.astype(np.int64))
                r = recall(a.metric, dist, exact_lat, a.k)
                print(
                    f"{'hnswlib (CPU, 1 thread)':<34} {ef:>6} {r:>10.4f} {med:>10.3f} {p95:>8.3f}"
                )
                continue
            (labels, _), sec = timed(lambda: hnsw.knn_query(queries, k=a.k), a.runs)
            dist = true_dist(a.metric, base, queries, labels.astype(np.int64))
            r = recall(a.metric, dist, exact_dist, a.k)
            print(
                f"{'hnswlib (CPU)':<34} {ef:>6} {r:>10.4f} {len(queries) / sec:>12,.0f}"
            )


if __name__ == "__main__":
    main()
