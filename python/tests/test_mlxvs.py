# Copyright © 2026 mikepapadim

import mlx.core as mx
import mlxvs
import numpy as np
import pytest


def rows(n, d, seed):
    return np.random.default_rng(seed).standard_normal((n, d), dtype=np.float32)


def exact(metric, q, b):
    """Scores of every query against every base row; larger is closer."""
    if metric == mlxvs.Metric.l2:
        return -(((q[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    if metric == mlxvs.Metric.cosine:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return q @ b.T


def recall(ids, scores, k):
    """Tie-tolerant recall@k against exact scores (float64)."""
    kth = -np.sort(-scores, axis=1)[:, k - 1 : k]
    hit = np.take_along_axis(scores, np.maximum(ids, 0), axis=1) >= kth - 1e-5
    return (hit & (ids >= 0)).sum() / (ids.shape[0] * k)


@pytest.mark.parametrize(
    "metric", [mlxvs.Metric.inner_product, mlxvs.Metric.l2, mlxvs.Metric.cosine]
)
@pytest.mark.parametrize("k,tile", [(10, 65536), (100, 777), (256, 1000)])
def test_brute_force_knn(metric, k, tile):
    b, q = rows(3000, 48, 1), rows(32, 48, 2)
    ids, dist = mlxvs.brute_force_knn(
        mx.array(b), mx.array(q), k, metric, tile_rows=tile
    )
    ids, dist = np.array(ids), np.array(dist)
    assert ids.shape == (32, k) and dist.shape == (32, k)
    scores = exact(metric, q.astype(np.float64), b.astype(np.float64))
    assert recall(ids, scores, k) == 1.0
    expected = -scores if metric == mlxvs.Metric.l2 else scores
    got = np.take_along_axis(expected, ids, axis=1)
    assert np.allclose(got, dist, rtol=1e-4, atol=1e-4)
    # sorted nearest first
    step = np.diff(dist, axis=1)
    assert (step >= -1e-5).all() if metric == mlxvs.Metric.l2 else (step <= 1e-5).all()


def test_brute_force_knn_tiny_shapes():
    # small inputs reach the kernel in Metal's constant address space
    b, q = rows(100, 8, 21), rows(2, 8, 22)
    ids, _ = mlxvs.brute_force_knn(mx.array(b), mx.array(q), 3, mlxvs.Metric.cosine)
    scores = exact(mlxvs.Metric.cosine, q.astype(np.float64), b.astype(np.float64))
    assert recall(np.array(ids), scores, 3) == 1.0
    ids, _ = mlxvs.brute_force_knn(
        mx.array(b), mx.array(q), 1, mlxvs.Metric.l2, tile_rows=7
    )
    assert np.array(ids).shape == (2, 1)


def test_brute_force_knn_fewer_rows_than_k():
    b, q = rows(5, 8, 3), rows(4, 8, 4)
    ids, dist = mlxvs.brute_force_knn(mx.array(b), mx.array(q), 8, mlxvs.Metric.l2)
    ids, dist = np.array(ids), np.array(dist)
    assert (ids[:, :5] >= 0).all() and (ids[:, 5:] == -1).all()
    assert np.isinf(dist[:, 5:]).all() and (dist[:, 5:] > 0).all()
    assert sorted(ids[0, :5].tolist()) == list(range(5))


def test_brute_force_knn_rejects_bad_input():
    with pytest.raises(ValueError):
        mlxvs.brute_force_knn(mx.zeros((4, 3)), mx.zeros((2, 5)), 1)
    with pytest.raises(ValueError):
        mlxvs.brute_force_knn(mx.zeros((4, 3)), mx.zeros((2, 3)), 300)
    with pytest.raises(ValueError):
        mlxvs.brute_force_knn(mx.zeros((4, 3), dtype=mx.float16), mx.zeros((2, 3)), 1)


def test_brute_force_knn_non_contiguous_input():
    b = rows(500, 64, 5)
    bt = mx.array(np.ascontiguousarray(b.T)).T  # a transposed view
    q = mx.array(rows(8, 64, 6))
    ids_view, _ = mlxvs.brute_force_knn(bt, q, 10, mlxvs.Metric.l2)
    ids, _ = mlxvs.brute_force_knn(mx.array(b), q, 10, mlxvs.Metric.l2)
    assert np.array_equal(np.array(ids_view), np.array(ids))


def test_kmeans_separates_blobs():
    rng = np.random.default_rng(7)
    centers = 20 * rng.standard_normal((6, 16), dtype=np.float32)
    truth = np.arange(6000) % 6
    x = centers[truth] + rng.standard_normal((6000, 16), dtype=np.float32)
    c = mlxvs.kmeans_fit(mx.array(x), 6, max_iter=20)
    assert c.shape == (6, 16)
    labels = np.array(mlxvs.kmeans_predict(mx.array(x), c, 2))
    assert labels.shape == (6000, 2)
    purity = sum(np.bincount(labels[truth == b, 0]).max() for b in range(6)) / 6000
    assert purity > 0.999
    assert (labels[:, 0] != labels[:, 1]).all()


@pytest.mark.parametrize(
    "metric", [mlxvs.Metric.inner_product, mlxvs.Metric.l2, mlxvs.Metric.cosine]
)
def test_nn_descent_clustered(metric):
    rng = np.random.default_rng(11)
    centers = 6 * rng.standard_normal((40, 64), dtype=np.float32)
    x = centers[np.arange(12000) % 40] + rng.standard_normal(
        (12000, 64), dtype=np.float32
    )
    k = 32
    ids, dist = mlxvs.nn_descent(mx.array(x), k, metric)
    ids, dist = np.array(ids), np.array(dist)
    assert ids.shape == (12000, k)
    assert (ids != np.arange(12000)[:, None]).all()  # never the row itself
    valid = ids >= 0
    assert valid.mean() > 0.999
    for r in range(0, 12000, 1000):  # distinct per row
        v = ids[r][ids[r] >= 0]
        assert len(set(v.tolist())) == len(v)
    exact_ids, _ = mlxvs.brute_force_knn(mx.array(x), mx.array(x), k + 1, metric)
    exact_ids = np.array(exact_ids)[:, 1:]  # drop self
    hits = np.array(
        [len(set(a) & set(b)) for a, b in zip(ids.tolist(), exact_ids.tolist())]
    )
    assert hits.sum() / (12000 * k) > 0.95


def test_nn_descent_fp32_path_and_small_n():
    x = rows(300, 24, 12)
    ids, _ = mlxvs.nn_descent(mx.array(x), 8, mlxvs.Metric.l2, fp16=False)
    assert np.array(ids).shape == (300, 8)
    ids, _ = mlxvs.nn_descent(mx.array(rows(5, 4, 1)), 8, mlxvs.Metric.l2)
    ids = np.array(ids)
    assert (ids[:, :4] >= 0).all() and (ids[:, 4:] == -1).all()


def test_nn_descent_rejects_bad_input():
    with pytest.raises(ValueError):
        mlxvs.nn_descent(mx.zeros((10, 4)), 200)
    with pytest.raises(ValueError):
        mlxvs.nn_descent(mx.zeros((1, 4)), 8)
    with pytest.raises(ValueError):
        mlxvs.nn_descent(mx.zeros((10,)), 8)


def blobs(n, d, clusters, seed):
    rng = np.random.default_rng(seed)
    centers = 6 * rng.standard_normal((clusters, d), dtype=np.float32)
    return centers[np.arange(n) % clusters] + rng.standard_normal(
        (n, d), dtype=np.float32
    )


@pytest.mark.parametrize(
    "metric", [mlxvs.Metric.inner_product, mlxvs.Metric.l2, mlxvs.Metric.cosine]
)
@pytest.mark.parametrize(
    "algo", [mlxvs.GraphBuild.brute_force, mlxvs.GraphBuild.nn_descent]
)
def test_all_neighbors_clusters(metric, algo):
    x = blobs(16000, 64, 40, 13)
    k = 32
    ids, dist = mlxvs.all_neighbors(
        mx.array(x), k, metric, algo=algo, n_clusters=4, overlap_factor=2
    )
    ids, dist = np.array(ids), np.array(dist)
    assert ids.shape == (16000, k) and dist.shape == (16000, k)
    assert (ids != np.arange(16000)[:, None]).all()  # never the row itself
    assert (ids >= 0).all()
    for r in range(0, 16000, 1000):  # distinct per row
        assert len(set(ids[r].tolist())) == k
    step = np.diff(dist, axis=1)  # nearest first
    assert (step >= -1e-5).all() if metric == mlxvs.Metric.l2 else (step <= 1e-5).all()
    sample = np.arange(0, 16000, 160)  # 100 rows: exact() broadcasts [100, n, d]
    scores = exact(metric, x[sample], x)
    scores[np.arange(len(sample)), sample] = -np.inf  # self excluded
    assert recall(ids[sample], scores, k) > 0.95


def test_all_neighbors_one_cluster_brute_force_is_exact():
    x = rows(2000, 16, 14)
    k = 10
    ids, _ = mlxvs.all_neighbors(
        mx.array(x), k, mlxvs.Metric.l2, algo=mlxvs.GraphBuild.brute_force
    )
    scores = exact(mlxvs.Metric.l2, x, x)
    np.fill_diagonal(scores, -np.inf)
    assert recall(np.array(ids), scores, k) > 0.999


def test_all_neighbors_small_n_and_bad_input():
    ids, _ = mlxvs.all_neighbors(mx.array(rows(5, 4, 1)), 8, mlxvs.Metric.l2)
    ids = np.array(ids)
    assert (ids[:, :4] >= 0).all() and (ids[:, 4:] == -1).all()
    with pytest.raises(ValueError):
        mlxvs.all_neighbors(mx.zeros((10, 4)), 200)
    with pytest.raises(ValueError):
        mlxvs.all_neighbors(mx.zeros((10, 4)), 4, n_clusters=11)
    with pytest.raises(ValueError):
        mlxvs.all_neighbors(mx.zeros((10, 4)), 4, n_clusters=2, overlap_factor=3)

