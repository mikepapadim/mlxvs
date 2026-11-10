# Copyright © 2026 mikepapadim

"""UMAP with its k-NN graph built on the GPU.

UMAP's first step is the k-NN graph of the data, by default PyNNDescent on the
CPU. This builds it with mlxvs and hands it to UMAP as ``precomputed_knn``.
UMAP warns that ``transform`` on new data is unavailable with a graph it did
not build itself; ``fit_transform`` is unaffected.

    pip install umap-learn
    python examples/umap_knn.py                  # 200,000 random 64-d rows
    python examples/umap_knn.py rows.npy         # your own [n, d] array
"""

import sys
import time

import mlxvs
import numpy as np
import umap


def knn_for_umap(x, n_neighbors, metric="euclidean"):
    """(indices, distances) as UMAP expects: each row first, itself at 0."""
    nn = mlxvs.NearestNeighbors(n_neighbors - 1, metric=metric, algorithm="nn_descent")
    dist, ids = nn.fit(x).kneighbors()
    n = len(ids)
    ids = np.hstack([np.arange(n)[:, None], ids])
    dist = np.hstack([np.zeros((n, 1), np.float32), dist])
    return ids, dist


def main():
    if len(sys.argv) > 1:
        x = np.load(sys.argv[1]).astype(np.float32)
    else:
        rng = np.random.default_rng(0)
        centers = 5 * rng.standard_normal((50, 64), dtype=np.float32)
        x = centers[rng.integers(0, 50, 200_000)] + rng.standard_normal(
            (200_000, 64), dtype=np.float32
        )
    n_neighbors = 15

    t = time.perf_counter()
    ids, dist = knn_for_umap(x, n_neighbors)
    print(f"k-NN graph of {x.shape} on the GPU: {time.perf_counter() - t:.2f} s")

    t = time.perf_counter()
    embedding = umap.UMAP(
        n_neighbors=n_neighbors, precomputed_knn=(ids, dist, None)
    ).fit_transform(x)
    print(f"UMAP from that graph: {time.perf_counter() - t:.2f} s -> {embedding.shape}")


if __name__ == "__main__":
    main()
