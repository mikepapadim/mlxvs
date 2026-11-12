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
        mlxvs.brute_force_knn(mx.zeros((4, 3), dtype=mx.bool_), mx.zeros((2, 3)), 1)


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
    x = rows(5000, 24, 12)  # above the 4096 rows built exactly
    ids, _ = mlxvs.nn_descent(mx.array(x), 8, mlxvs.Metric.l2, fp16=False)
    assert np.array(ids).shape == (5000, 8)
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


@pytest.mark.parametrize(
    "convert",
    [
        lambda x: x,  # numpy float32
        lambda x: x.astype(np.float64),
        lambda x: x.tolist(),
        lambda x: mx.array(x).astype(mx.float16),
        lambda x: mx.array(x).astype(mx.bfloat16),
    ],
    ids=["numpy", "float64", "list", "float16", "bfloat16"],
)
def test_array_likes_and_dtypes(convert):
    x = rows(500, 16, 15)
    want, _ = mlxvs.brute_force_knn(x, x, 5, "l2")
    got, _ = mlxvs.brute_force_knn(convert(x), convert(x), 5, "l2")
    # float16 inputs round the rows, so near ties may swap
    assert (np.array(got)[:, 0] == np.arange(500)).all()
    assert (np.array(got) == np.array(want)).mean() > 0.9
    graph, _ = mlxvs.all_neighbors(convert(x), 4, "cosine", algo="brute_force")
    assert np.array(graph).shape == (500, 4)


def test_metric_and_algo_names():
    x = rows(300, 8, 16)
    for name in ["l2", "cosine", "inner_product"]:
        a, _ = mlxvs.brute_force_knn(x, x, 3, name)
        b, _ = mlxvs.brute_force_knn(x, x, 3, getattr(mlxvs.Metric, name))
        assert (np.array(a) == np.array(b)).all()
    with pytest.raises(ValueError, match="unknown Metric 'cosin'"):
        mlxvs.nn_descent(x, 8, "cosin")
    with pytest.raises(ValueError, match="unknown GraphBuild"):
        mlxvs.all_neighbors(x, 8, algo="ivf_pq")


def test_shape_errors():
    with pytest.raises(ValueError, match="2-D"):
        mlxvs.nn_descent(np.zeros(10, np.float32), 4)
    with pytest.raises(ValueError, match="same number of columns"):
        mlxvs.brute_force_knn(rows(10, 4, 1), rows(10, 5, 1), 2)
    with pytest.raises(ValueError, match="same number of columns"):
        mlxvs.kmeans_predict(rows(10, 4, 1), rows(2, 5, 1))


@pytest.mark.parametrize("n", [3, 41, 66, 129, 300])
@pytest.mark.parametrize("k", [32, 64, 128])
def test_nn_descent_small_n_lists_complete(n, k):
    # small n is built exactly: every row lists all the others it has room for
    for seed in range(3):
        x = rows(n, 24, seed)
        ids = np.array(mlxvs.nn_descent(x, k, "l2", seed=seed)[0])
        want = min(k, n - 1)
        assert (ids[:, :want] >= 0).all() and (ids[:, want:] == -1).all()
        assert (ids != np.arange(n)[:, None]).all()


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cosine"])
def test_nearest_neighbors_matches_numpy(metric):
    x, q = rows(3000, 16, 17), rows(40, 16, 18)
    nn = mlxvs.NearestNeighbors(6, metric=metric).fit(x)
    assert (nn.n_samples_fit_, nn.n_features_in_) == (3000, 16)

    def reference(a, b):
        if metric == "cosine":
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            return 1 - a @ b.T
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2) if metric == "euclidean" else d2

    want = reference(q, x)
    dist, ids = nn.kneighbors(q)
    assert ids.dtype == np.int64 and ids.shape == (40, 6)
    assert (ids == np.argsort(want, axis=1)[:, :6]).mean() > 0.99
    assert np.allclose(dist, np.sort(want, axis=1)[:, :6], atol=1e-3)
    # without queries: the fitted rows' neighbors, never the row itself
    dist, ids = nn.kneighbors(n_neighbors=3)
    assert ids.shape == (3000, 3) and (ids != np.arange(3000)[:, None]).all()
    assert (np.diff(dist, axis=1) >= -1e-5).all()
    assert nn.kneighbors(q, return_distance=False).shape == (40, 6)


def test_nearest_neighbors_errors():
    with pytest.raises(ValueError, match="unknown metric"):
        mlxvs.NearestNeighbors(metric="manhattan")
    with pytest.raises(ValueError, match="algorithm"):
        mlxvs.NearestNeighbors(algorithm="kd_tree")
    with pytest.raises(RuntimeError, match="fit"):
        mlxvs.NearestNeighbors().kneighbors()
    nn = mlxvs.NearestNeighbors(5).fit(rows(5, 3, 1))
    with pytest.raises(ValueError, match="n_neighbors"):
        nn.kneighbors()  # only 4 other rows


def test_all_neighbors_memmap_out_of_core(tmp_path):
    # rows read from a memory-mapped file, clusters capped well below n
    x = blobs(30000, 32, 30, 19)
    path = tmp_path / "rows.f32"
    x.tofile(path)
    mapped = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 32)
    k = 16
    ids, _ = mlxvs.all_neighbors(mapped, k, "l2", max_cluster_rows=7000)
    ids = np.array(ids)
    assert ids.shape == (30000, k) and (ids >= 0).all()
    assert (ids != np.arange(30000)[:, None]).all()
    sample = np.arange(0, 30000, 300)
    scores = exact(mlxvs.Metric.l2, x[sample], x)
    scores[np.arange(len(sample)), sample] = -np.inf
    assert recall(ids[sample], scores, k) > 0.95
    # a non-contiguous view takes the copying path and agrees
    view = x[::2]
    a, _ = mlxvs.all_neighbors(view, 4, "l2", algo="brute_force")
    b, _ = mlxvs.all_neighbors(np.ascontiguousarray(view), 4, "l2", algo="brute_force")
    assert (np.array(a) == np.array(b)).mean() > 0.999


@pytest.mark.parametrize("algo", ["single_cta", "multi_cta"])
@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
def test_graph_search_recall(metric, algo):
    rng = np.random.default_rng(3)
    z = rng.standard_normal((8200, 8)).astype(np.float32)
    x = z @ rng.standard_normal((8, 32)).astype(np.float32)
    x += 0.05 * rng.standard_normal(x.shape).astype(np.float32)
    base, queries = x[:8000], x[8000:]
    graph, _ = mlxvs.all_neighbors(base, 24, metric, algo="brute_force", n_clusters=1)
    ids, dist = mlxvs.graph_search(base, graph, queries, 10, metric, algo=algo)
    exact, _ = mlxvs.brute_force_knn(base, queries, 10, metric)
    ids, exact = np.asarray(ids), np.asarray(exact)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])
    assert recall > 0.95
    dist = np.asarray(dist)
    assert np.all(
        np.diff(dist, axis=1) >= 0 if metric == "l2" else np.diff(dist, axis=1) <= 0
    )


def test_graph_search_numpy_graph_and_errors():
    x = np.random.default_rng(0).standard_normal((500, 16)).astype(np.float32)
    graph = np.asarray(mlxvs.nn_descent(x, 16)[0]).astype(np.int64)
    ids, _ = mlxvs.graph_search(x, graph, x[:20], 5, itopk_size=32, search_width=2)
    assert np.asarray(ids)[:, 0].tolist() == list(range(20))
    with pytest.raises(ValueError):
        mlxvs.graph_search(x, graph[:100], x[:5], 5)
    with pytest.raises(ValueError):
        mlxvs.graph_search(x, graph, x[:5], 5, search_width=64)
    with pytest.raises(ValueError):
        mlxvs.graph_search(x, graph, x[:5], 300, algo="single_cta")
    with pytest.raises(ValueError):
        mlxvs.graph_search(x, graph, x[:5], 2000)
    with pytest.raises(ValueError):
        mlxvs.graph_search(x, graph, x[:5], 5, algo="sideways")


def test_cagra_build_searches_better_than_knn_graph():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((10500, 12)).astype(np.float32)
    x = z @ rng.standard_normal((12, 48)).astype(np.float32)
    x += 0.05 * rng.standard_normal(x.shape).astype(np.float32)
    base, queries = x[:10000], x[10000:]
    graph = mlxvs.cagra_build(base, 16, intermediate_graph_degree=48)
    g = np.asarray(graph)
    assert g.shape == (10000, 16) and g.dtype == np.int32
    assert (g >= 0).all() and (g != np.arange(10000)[:, None]).all()
    assert all(len(set(row)) == 16 for row in g)
    knn, _ = mlxvs.nn_descent(base, 16)
    exact, _ = mlxvs.brute_force_knn(base, queries, 10)
    exact = np.asarray(exact)

    def recall(gr):
        ids, _ = mlxvs.graph_search(base, gr, queries, 10, itopk_size=32)
        ids = np.asarray(ids)
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact)])

    assert recall(graph) > recall(knn)
    with pytest.raises(ValueError):
        mlxvs.cagra_build(base, 64, intermediate_graph_degree=32)
    with pytest.raises(ValueError):
        mlxvs.cagra_optimize(knn, 17)


def _low_rank(n, d=48, r=12, seed=9):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, r)).astype(np.float32)
    x = x @ rng.standard_normal((r, d)).astype(np.float32)
    return x + 0.05 * rng.standard_normal(x.shape).astype(np.float32)


def _recall(ids, exact):
    ids, exact = np.asarray(ids), np.asarray(exact)
    return np.mean([len(set(a) & set(b)) / exact.shape[1] for a, b in zip(ids, exact)])


@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cagra_index_build_search(metric, dtype):
    x = _low_rank(8300)
    base, queries = x[:8000], x[8000:]
    index = mlxvs.CagraIndex.build(base, 24, metric, dtype=dtype)
    assert len(index) == 8000 and index.dim == 48 and index.graph_degree == 24
    assert index.metric == metric and index.dtype == getattr(mx, dtype)
    ids, dist = index.search(queries, k=10)
    exact, exact_dist = mlxvs.brute_force_knn(base, queries, 10, metric)
    assert _recall(ids, exact) > (0.95 if dtype == "float32" else 0.9)
    if dtype == "float32":
        # the same distances as exact search (cosine: similarities of unit rows)
        d0, e0 = np.asarray(dist)[:, 0], np.asarray(exact_dist)[:, 0]
        hit = np.asarray(ids)[:, 0] == np.asarray(exact)[:, 0]
        assert np.allclose(d0[hit], e0[hit], rtol=1e-4, atol=1e-4)


def test_cagra_index_save_load(tmp_path):
    x = _low_rank(3000)
    index = mlxvs.CagraIndex.build(x, 16, "cosine", dtype="float16")
    path = tmp_path / "index.safetensors"
    index.save(path)
    loaded = mlxvs.CagraIndex.load(path)
    assert repr(loaded) == repr(index)
    a = np.asarray(index.search(x[:50], k=5, seed=7)[0])
    b = np.asarray(loaded.search(x[:50], k=5, seed=7)[0])
    assert (a == b).all()
    with pytest.raises(ValueError):
        index.save(tmp_path / "index.bin")
    mx.save_safetensors(str(tmp_path / "other.safetensors"), {"a": mx.zeros((2,))})
    with pytest.raises(ValueError):
        mlxvs.CagraIndex.load(tmp_path / "other.safetensors")


def test_cagra_index_inputs():
    x = _low_rank(2000)
    strided = np.concatenate([x, x], axis=1)[:, :48]  # a non-contiguous view
    assert not strided.flags.c_contiguous
    index = mlxvs.CagraIndex.build(strided, 16)
    ids, _ = index.search(strided[0], k=3)  # one query as a 1-D row
    assert np.asarray(ids).shape == (1, 3) and np.asarray(ids)[0, 0] == 0
    with pytest.raises(ValueError):
        mlxvs.CagraIndex.build(x, 16, dtype="int8")


@pytest.mark.parametrize(
    "metric,space", [("l2", "l2"), ("cosine", "cosine"), ("inner_product", "ip")]
)
def test_cagra_index_save_hnswlib(tmp_path, metric, space):
    hnswlib = pytest.importorskip("hnswlib")
    x = _low_rank(6200)
    base, queries = x[:6000], x[6000:]
    index = mlxvs.CagraIndex.build(base, 24, metric, dtype="float16")
    path = str(tmp_path / "index.hnsw")
    index.save_hnswlib(path)
    p = hnswlib.Index(space=space, dim=index.dim)
    p.load_index(path)
    assert p.get_current_count() == 6000 and p.element_count == 6000
    p.set_ef(64)
    labels, _ = p.knn_query(queries, k=10)
    exact, _ = mlxvs.brute_force_knn(base, queries, 10, metric)
    assert _recall(labels, exact) > 0.9


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_cagra_index_save_jvector(tmp_path, metric):
    # the layout JVector 4's OnDiskGraphIndex (format 6) reads, big-endian;
    # searched from Java in examples/jvector
    base = _low_rank(3000)
    index = mlxvs.CagraIndex.build(base, 16, metric)
    path = tmp_path / "index.jvector"
    index.save_jvector(str(path))
    raw = path.read_bytes()
    n, d, degree = len(index), index.dim, index.graph_degree
    words = 8 + 64 + 2
    header = np.frombuffer(raw[: 4 * words], dtype=">u4")
    assert header[0] == 0xFFFF0D61 and header[1] == 6
    assert list(header[2:8]) == [n, d, header[4], degree, n, 1]
    assert 0 <= header[4] < n  # the entry node
    assert list(header[8:10]) == [n, degree] and not header[10:72].any()
    assert list(header[72:]) == [1, 0]  # one feature: INLINE_VECTORS
    record = np.dtype(
        [
            ("ordinal", ">i4"),
            ("row", ">f4", (d,)),
            ("count", ">i4"),
            ("nbr", ">i4", (degree,)),
        ]
    )
    nodes = np.frombuffer(raw, dtype=record, count=n, offset=4 * words)
    assert (nodes["ordinal"] == np.arange(n)).all()
    np.testing.assert_array_equal(nodes["row"], np.asarray(index.dataset))
    graph = np.asarray(index.graph)
    assert (nodes["count"] == (graph >= 0).sum(1)).all()
    np.testing.assert_array_equal(np.sort(nodes["nbr"], 1), np.sort(graph, 1))
    footer = 4 * words + nodes.nbytes
    assert raw[footer : footer + 4 * words] == raw[: 4 * words]
    tail = raw[footer + 4 * words :]
    assert len(tail) == 12
    assert np.frombuffer(tail[:8], dtype=">i8")[0] == footer
    assert np.frombuffer(tail[8:], dtype=">u4")[0] == 0x4A564244


@pytest.mark.parametrize("metric", ["l2", "cosine", "inner_product"])
def test_vamana_build_search(metric):
    x = _low_rank(6200)
    base, queries = x[:6000], x[6000:]
    graph = mlxvs.vamana_build(base, 16, metric, intermediate_graph_degree=32)
    g = np.asarray(graph)
    assert g.shape == (6000, 16) and g.dtype == np.int32
    valid = g >= 0
    assert (g[valid] < 6000).all() and (g != np.arange(6000)[:, None]).all()
    if metric != "inner_product":
        # (1 + dot) / 2 on rows that are not unit keeps fewer edges, as JVector
        assert valid.sum(1).mean() > 8
    ids, _ = mlxvs.graph_search(base, graph, queries, 10, metric)
    exact, _ = mlxvs.brute_force_knn(base, queries, 10, metric)
    assert _recall(ids, exact) > 0.95


def test_vamana_prune_alpha_and_index():
    x = _low_rank(3000)
    knn, _ = mlxvs.all_neighbors(x, 32, "l2", algo="brute_force", n_clusters=1)
    tight = (np.asarray(mlxvs.vamana_prune(x, knn, 32, alpha=1.0)) >= 0).sum()
    loose = (np.asarray(mlxvs.vamana_prune(x, knn, 32, alpha=2.0)) >= 0).sum()
    assert loose > tight  # a larger alpha keeps more (longer) edges
    with pytest.raises(ValueError):
        mlxvs.vamana_prune(x, knn, 64)
    index = mlxvs.CagraIndex.build(x, 16, graph="vamana")
    ids, _ = index.search(x[:20], k=1)
    assert np.asarray(ids)[:, 0].tolist() == list(range(20))
    with pytest.raises(ValueError):
        mlxvs.CagraIndex.build(x, 16, graph="hnsw")


@pytest.mark.parametrize("algo", ["single_cta", "multi_cta"])
def test_graph_search_filter(algo):
    x = _low_rank(8200)
    base, queries = x[:8000], x[8000:]
    allow = np.arange(8000) % 3 != 0
    index = mlxvs.CagraIndex.build(base, 24)
    ids, _ = index.search(queries, 10, algo=algo, filter=allow)
    ids = np.asarray(ids)
    assert allow[ids].all() and (ids >= 0).all()
    kept = np.flatnonzero(allow)
    exact, _ = mlxvs.brute_force_knn(base[kept], queries, 10)
    assert _recall(ids, kept[np.asarray(exact)]) > 0.95
    words = np.packbits(allow, bitorder="little").view(np.uint32)
    ids2, _ = index.search(queries, 10, algo=algo, filter=words)
    assert allow[np.asarray(ids2)].all()
    with pytest.raises(ValueError):
        index.search(queries, 10, filter=allow[:100])


def test_graph_search_seeds():
    x = _low_rank(5100)
    base, queries = x[:5000], x[5000:]
    index = mlxvs.CagraIndex.build(base, 16)
    nearest, _ = mlxvs.brute_force_knn(base, queries, 1)
    for algo in ("single_cta", "multi_cta"):
        ids, _ = index.search(
            queries, 5, algo=algo, max_iterations=1, seeds=np.asarray(nearest)
        )
        assert (np.asarray(ids)[:, 0] == np.asarray(nearest)[:, 0]).all()


@pytest.mark.parametrize("dtype,shift", [("uint8", 128), ("int8", 0)])
def test_cagra_index_byte_rows(tmp_path, dtype, shift):
    x = np.clip(np.round(10 * _low_rank(5100)), -127, 127) + shift
    base, queries = x[:5000], x[5000:]
    index = mlxvs.CagraIndex.build(base, 16, dtype=dtype)
    assert index.dtype == getattr(mx, dtype)
    ids, _ = index.search(queries, 10)
    exact, _ = mlxvs.brute_force_knn(base, queries, 10)
    assert _recall(ids, exact) > 0.95
    path = str(tmp_path / "bytes.safetensors")
    index.save(path)
    assert mlxvs.CagraIndex.load(path).dtype == index.dtype
    with pytest.raises(ValueError):
        mlxvs.CagraIndex.build(base + 0.5, 16, dtype=dtype)  # not integers


def test_graph_search_filter_per_query():
    x = _low_rank(4100)
    base, queries = x[:4000], x[4000:]
    index = mlxvs.CagraIndex.build(base, 16)
    rows = np.arange(4000)
    bitmap = (rows[None, :] + np.arange(100)[:, None]) % 4 != 0  # [nq, n]
    for algo in ("single_cta", "multi_cta"):
        ids, _ = index.search(queries, 10, algo=algo, filter=bitmap)
        ids = np.asarray(ids)
        assert (ids >= 0).all()
        assert bitmap[np.arange(100)[:, None], ids].all()
