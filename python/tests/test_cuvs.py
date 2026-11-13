# Copyright © 2026 mikepapadim

"""mlxvs.cuvs: cuVS's Python API, checked as cuVS's own tests check it."""

import numpy as np
import pytest

import mlx.core as mx

from mlxvs.cuvs import common
from mlxvs.cuvs.cluster import kmeans
from mlxvs.cuvs.common import device_ndarray
from mlxvs.cuvs.distance import pairwise_distance
from mlxvs.cuvs.neighbors import (
    all_neighbors,
    brute_force,
    cagra,
    filters,
    ivf_flat,
    ivf_pq,
    nn_descent,
    refine,
)

spatial = pytest.importorskip("scipy.spatial.distance")


def _recall(found, truth):
    found, truth = np.asarray(found), np.asarray(truth)
    hits = sum(len(set(a) & set(b)) for a, b in zip(found, truth))
    return hits / truth.size


def _exact(dataset, queries, k, metric):
    sk = {"sqeuclidean": "sqeuclidean", "euclidean": "euclidean", "cosine": "cosine"}
    if metric == "inner_product":
        d = -(queries @ dataset.T)
    else:
        d = spatial.cdist(queries, dataset, sk[metric])
    return np.argsort(d, axis=1, kind="stable")[:, :k]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((40, 32)).astype(np.float32)
    x = centers[rng.integers(0, 40, 6000)] + 0.4 * rng.standard_normal((6000, 32))
    return x.astype(np.float32), x[:200].astype(np.float32) + 0.05


# --- common ------------------------------------------------------------------


def test_device_ndarray():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    d = device_ndarray(a)
    assert d.shape == (3, 4) and d.dtype == np.float32
    np.testing.assert_array_equal(d.copy_to_host(), a)
    np.testing.assert_array_equal(np.asarray(d), a)
    e = device_ndarray.empty((2, 5), dtype="uint32")
    assert e.shape == (2, 5) and e.dtype == np.uint32
    assert isinstance(d.array, mx.array)


def test_set_output_as():
    x = np.random.default_rng(1).random((10, 4), dtype=np.float32)
    try:
        common.set_output_as("numpy")
        assert isinstance(pairwise_distance(x, x), np.ndarray)
        common.set_output_as("mlx")
        assert isinstance(pairwise_distance(x, x), mx.array)
        common.set_output_as(lambda d: d.copy_to_host().sum())
        assert (
            np.isscalar(pairwise_distance(x, x))
            or np.ndim(pairwise_distance(x, x)) == 0
        )
    finally:
        common.set_output_as("raft")
    assert isinstance(pairwise_distance(x, x), device_ndarray)


# --- distance ----------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "sqeuclidean",
        "cityblock",
        "l1",
        "chebyshev",
        "canberra",
        "minkowski",
        "correlation",
        "cosine",
        "hamming",
        "jensenshannon",
        "russellrao",
        "hellinger",
        "kl_divergence",
        "inner_product",
    ],
)
def test_pairwise_distance(metric):
    # as cuVS's test_distance.py
    rng = np.random.default_rng(2)
    x = rng.random((50, 40), dtype=np.float32)
    y = rng.random((30, 40), dtype=np.float32)
    if metric in ("hamming", "russellrao"):
        x, y = (x > 0.5).astype(np.float32), (y > 0.5).astype(np.float32)
    if metric in ("jensenshannon", "hellinger", "kl_divergence"):
        x /= x.sum(1, keepdims=True)
        y /= y.sum(1, keepdims=True)
    out = device_ndarray.empty((50, 30), dtype=np.float32)
    pairwise_distance(x, y, out, metric=metric, p=3.0)
    got = out.copy_to_host()
    if metric == "inner_product":
        want = x @ y.T
    elif metric == "hellinger":
        want = np.sqrt(np.maximum(1 - np.sqrt(x) @ np.sqrt(y).T, 0))
    elif metric == "kl_divergence":
        want = 0.5 * np.array(
            [[np.sum(a * np.log(a / b)) for b in y] for a in x], dtype=np.float32
        )
    else:
        sk = {"l1": "cityblock", "minkowski": "minkowski"}.get(metric, metric)
        kw = {"p": 3.0} if metric == "minkowski" else {}
        want = spatial.cdist(x, y, sk, **kw)
    np.testing.assert_allclose(got, want, atol=1e-3, rtol=1e-3)


def test_pairwise_distance_errors():
    x = np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError):
        pairwise_distance(x, np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError):
        pairwise_distance(x, x, metric="dice")


# --- brute_force ---------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cityblock",
        "chebyshev",
        "canberra",
        "correlation",
        "russellrao",
        "cosine",
        "sqeuclidean",
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("serialize", [True, False])
def test_brute_force_knn(metric, inplace, serialize, tmp_path):
    # as cuVS's test_brute_force.py
    rng = np.random.default_rng(3)
    index = rng.random((100, 40)).astype(np.float32)
    queries = rng.random((32, 40)).astype(np.float32)
    if metric == "russellrao":
        index, queries = (
            (index >= 0.5).astype(np.float32),
            (queries >= 0.5).astype(np.float32),
        )
    k = 5
    indices = device_ndarray(np.zeros((32, k), dtype="int64"))
    distances = device_ndarray(np.zeros((32, k), dtype="float32"))
    bf = brute_force.build(device_ndarray(index), metric)
    if serialize:
        path = str(tmp_path / "bf.bin")
        brute_force.save(path, bf)
        bf = brute_force.load(path)
    ret_d, ret_i = brute_force.search(
        bf,
        device_ndarray(queries),
        k,
        neighbors=indices if inplace else None,
        distances=distances if inplace else None,
        prefilter=filters.no_filter(),
    )
    got = (distances if inplace else ret_d).copy_to_host()
    got[got <= 1e-5] = 0.0
    pw = spatial.cdist(queries, index, metric=metric)
    want = np.sort(pw, axis=1)[:, :k]
    np.testing.assert_allclose(want, got, atol=1e-3, rtol=1e-3)
    assert (indices if inplace else ret_i).dtype == np.int64


@pytest.mark.parametrize("kind", ["bitset", "bitmap"])
def test_brute_force_prefilter(kind):
    rng = np.random.default_rng(4)
    index = rng.random((300, 16)).astype(np.float32)
    queries = rng.random((20, 16)).astype(np.float32)
    if kind == "bitset":
        keep = rng.random(300) < 0.3
        pre = filters.from_bitset(keep)
        allowed = np.broadcast_to(keep, (20, 300))
    else:
        allowed = rng.random((20, 300)) < 0.3
        pre = filters.from_bitmap(allowed)
    d, i = brute_force.search(brute_force.build(index), queries, 7, prefilter=pre)
    i = i.copy_to_host()
    pw = np.where(allowed, spatial.cdist(queries, index, "sqeuclidean"), np.inf)
    np.testing.assert_array_equal(np.sort(i, 1), np.sort(np.argsort(pw, 1)[:, :7], 1))


def test_brute_force_file_layout(tmp_path):
    # cuVS's brute-force format: a dtype prefix, then .npy records
    x = np.random.default_rng(5).random((10, 3)).astype(np.float32)
    path = tmp_path / "bf.bin"
    brute_force.save(str(path), brute_force.build(x, "cosine"))
    raw = path.read_bytes()
    assert raw[:4] == b"<f4\0"
    assert raw[4:10] == b"\x93NUMPY"


# --- cagra ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "cosine", "euclidean"]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
def test_cagra(data, metric, dtype):
    dataset, queries = data
    if dtype in (np.int8, np.uint8):
        if metric in ("inner_product", "cosine"):
            pytest.skip("skip normalization for int8/uint8 data")
        lo = -60 if dtype == np.int8 else 0
        dataset = np.clip(
            np.round(dataset * 15 + (0 if dtype == np.int8 else 64)), lo, 127
        )
        queries = np.clip(
            np.round(queries * 15 + (0 if dtype == np.int8 else 64)), lo, 127
        )
    elif metric in ("inner_product", "cosine"):
        dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    dataset, queries = dataset.astype(dtype), queries.astype(dtype)
    params = cagra.IndexParams(
        metric=metric, intermediate_graph_degree=64, graph_degree=32
    )
    index = cagra.build(params, device_ndarray(dataset))
    k = 10
    out_idx = device_ndarray(np.zeros((len(queries), k), dtype=np.uint32))
    out_dist = device_ndarray(np.zeros((len(queries), k), dtype=np.float32))
    cagra.search(
        cagra.SearchParams(),
        index,
        device_ndarray(queries),
        k,
        neighbors=out_idx,
        distances=out_dist,
    )
    idx = out_idx.copy_to_host()
    assert idx.dtype == np.uint32
    truth = _exact(dataset.astype(np.float32), queries.astype(np.float32), k, metric)
    assert _recall(idx, truth) > 0.9
    # distances as cuVS returns them
    q = queries.astype(np.float32)
    rows = dataset.astype(np.float32)[idx[:, 0]]
    want = {
        "sqeuclidean": np.sum((q - rows) ** 2, 1),
        "euclidean": np.sqrt(np.sum((q - rows) ** 2, 1)),
        "inner_product": np.sum(q * rows, 1),
        "cosine": 1
        - np.sum(q * rows, 1)
        / (np.linalg.norm(q, axis=1) * np.linalg.norm(rows, axis=1)),
    }[metric]
    np.testing.assert_allclose(
        out_dist.copy_to_host()[:, 0], want, rtol=2e-2, atol=2e-2
    )
    assert index.graph.shape == (len(dataset), 32)
    assert index.dim == dataset.shape[1] and len(index) == len(dataset)


@pytest.mark.parametrize("version", [4, 5, 6])
@pytest.mark.parametrize("include_dataset", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.uint8])
def test_cagra_serialize(data, tmp_path, version, include_dataset, dtype):
    dataset, queries = data
    if dtype == np.uint8:
        dataset = np.clip(np.round(dataset * 15 + 64), 0, 255)
        queries = np.clip(np.round(queries * 15 + 64), 0, 255)
    dataset, queries = dataset.astype(dtype), queries.astype(dtype)
    index = cagra.build(
        cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), dataset
    )
    sp = cagra.SearchParams()
    d0, i0 = cagra.search(sp, index, queries, 10)
    path = str(tmp_path / "cagra.bin")
    cagra.save(path, index, include_dataset=include_dataset, version=version)
    loaded = cagra.Index()
    cagra.load(loaded, path)
    if not include_dataset:
        cagra.update_dataset(loaded, dataset)
    np.testing.assert_array_equal(np.asarray(loaded.graph), np.asarray(index.graph))
    d1, i1 = cagra.search(sp, loaded, queries, 10)
    np.testing.assert_array_equal(i0.copy_to_host(), i1.copy_to_host())
    np.testing.assert_allclose(d0.copy_to_host(), d1.copy_to_host(), rtol=1e-5)
    # load(filename) returns the index too
    assert len(cagra.load(path)) == len(dataset)


def test_cagra_file_records(data, tmp_path):
    # every record is a NumPy v1 header that numpy itself parses, padded to
    # a multiple of 64 bytes as RAFT writes them
    import io as _io

    dataset, _ = data
    index = cagra.build(
        cagra.IndexParams(graph_degree=16, intermediate_graph_degree=32), dataset[:500]
    )
    path = tmp_path / "c.bin"
    cagra.save(str(path), index)
    f = _io.BytesIO(path.read_bytes())
    assert f.read(4) == b"<f4\0"
    shapes = []
    while True:
        start = f.tell()
        magic = f.read(6)
        if not magic:
            break
        assert magic == b"\x93NUMPY"
        f.seek(start)
        np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        assert not fortran and f.tell() % 64 == start % 64
        shapes.append(shape)
        f.seek(int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize, 1)
    # version, kind, n, dim, degree, metric, graph, content, tag, dtype,
    # n_rows, dim, stride, rows
    assert shapes == [
        (),
        (),
        (),
        (),
        (),
        (),
        (500, 16),
        (),
        (),
        (),
        (),
        (),
        (),
        (500, 32),
    ]


def test_cagra_filter(data):
    dataset, queries = data
    index = cagra.build(
        cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), dataset
    )
    keep = np.zeros(len(dataset), bool)
    keep[::3] = True
    _, idx = cagra.search(
        cagra.SearchParams(itopk_size=128),
        index,
        queries,
        10,
        filter=filters.from_bitset(keep),
    )
    idx = idx.copy_to_host()
    assert keep[idx].all()
    truth = _exact(dataset[keep], queries, 10, "sqeuclidean")
    assert _recall(np.flatnonzero(keep)[truth], idx) > 0.85


def test_cagra_extend_and_from_graph(data):
    dataset, queries = data
    half = len(dataset) // 2
    index = cagra.build(
        cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), dataset[:half]
    )
    cagra.extend(cagra.ExtendParams(), index, dataset, half)
    assert len(index) == len(dataset)
    _, idx = cagra.search(cagra.SearchParams(), index, queries, 10)
    assert (
        _recall(idx.copy_to_host(), _exact(dataset, queries, 10, "sqeuclidean")) > 0.9
    )
    again = cagra.from_graph(index.graph, dataset)
    _, idx2 = cagra.search(cagra.SearchParams(), again, queries, 10)
    np.testing.assert_array_equal(idx.copy_to_host(), idx2.copy_to_host())


@pytest.mark.parametrize("algo", ["single_cta", "multi_cta", "multi_kernel", "auto"])
def test_cagra_search_params(data, algo):
    dataset, queries = data
    index = cagra.build(
        cagra.IndexParams(graph_degree=32, intermediate_graph_degree=64), dataset
    )
    sp = cagra.SearchParams(algo=algo, itopk_size=64, search_width=2, max_iterations=0)
    _, idx = cagra.search(sp, index, queries, 10)
    # cuVS's test_cagra.py threshold
    assert (
        _recall(idx.copy_to_host(), _exact(dataset, queries, 10, "sqeuclidean")) > 0.7
    )


# --- ivf -----------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product", "euclidean"])
def test_ivf_flat(data, tmp_path, metric):
    dataset, queries = data
    p = ivf_flat.IndexParams(n_lists=32, metric=metric)
    index = ivf_flat.build(p, dataset)
    d, i = ivf_flat.search(ivf_flat.SearchParams(n_probes=32), index, queries, 10)
    assert i.dtype == np.int64
    truth = _exact(dataset, queries, 10, metric)
    assert _recall(i.copy_to_host(), truth) > 0.99
    path = str(tmp_path / "ivf.bin")
    ivf_flat.save(path, index)
    d2, i2 = ivf_flat.search(
        ivf_flat.SearchParams(n_probes=32), ivf_flat.load(path), queries, 10
    )
    np.testing.assert_array_equal(i.copy_to_host(), i2.copy_to_host())
    assert index.n_lists == 32 and index.dim == 32 and len(index) == len(dataset)


def test_ivf_flat_extend_filter(data):
    dataset, queries = data
    p = ivf_flat.IndexParams(n_lists=16, add_data_on_build=False)
    index = ivf_flat.build(p, dataset)
    assert len(index) == 0
    ivf_flat.extend(index, dataset, np.arange(len(dataset)))
    assert len(index) == len(dataset)
    keep = np.zeros(len(dataset), bool)
    keep[1::2] = True
    _, i = ivf_flat.search(
        ivf_flat.SearchParams(n_probes=16),
        index,
        queries,
        10,
        filter=filters.from_bitset(keep),
    )
    i = i.copy_to_host()
    assert keep[i].all()
    truth = np.flatnonzero(keep)[_exact(dataset[keep], queries, 10, "sqeuclidean")]
    assert _recall(i, truth) > 0.99


def test_ivf_pq_refine(data, tmp_path):
    dataset, queries = data
    index = ivf_pq.build(ivf_pq.IndexParams(n_lists=32, pq_dim=16), dataset)
    assert index.pq_dim == 16 and index.pq_bits == 8 and index.pq_len == 2
    _, cand = ivf_pq.search(ivf_pq.SearchParams(n_probes=32), index, queries, 40)
    d, i = refine(dataset, queries, cand, k=10)
    truth = _exact(dataset, queries, 10, "sqeuclidean")
    assert _recall(i.copy_to_host(), truth) > 0.95
    want = np.sort(spatial.cdist(queries, dataset, "sqeuclidean"), 1)[:, :10]
    np.testing.assert_allclose(
        np.sort(d.copy_to_host(), 1)[:, 0], want[:, 0], rtol=1e-3, atol=1e-3
    )
    path = str(tmp_path / "pq.bin")
    ivf_pq.save(path, index)
    assert len(ivf_pq.load(path)) == len(dataset)


# --- graphs --------------------------------------------------------------------


def test_nn_descent(data):
    dataset, _ = data
    idx = nn_descent.build(
        nn_descent.IndexParams(graph_degree=16, intermediate_graph_degree=32), dataset
    )
    g = idx.graph.copy_to_host()
    assert g.shape == (len(dataset), 16) and g.dtype == np.uint32
    assert (g[:, 0] == np.arange(len(dataset))).all()
    bf = brute_force.build(dataset)
    _, truth = brute_force.search(bf, dataset, 16)
    assert _recall(g, truth.copy_to_host()) > 0.9
    assert idx.distances.shape == g.shape


@pytest.mark.parametrize("algo", ["nn_descent", "brute_force"])
def test_all_neighbors(data, algo):
    dataset, _ = data
    k = 12
    params = all_neighbors.AllNeighborsParams(algo=algo)
    ind, dist = all_neighbors.build(
        dataset,
        k,
        params,
        distances=device_ndarray.empty((len(dataset), k), np.float32),
    )
    ind = ind.copy_to_host()
    _, truth = brute_force.search(brute_force.build(dataset), dataset, k)
    assert _recall(ind, truth.copy_to_host()) > (
        0.999 if algo == "brute_force" else 0.9
    )
    assert np.all(np.diff(dist.copy_to_host(), axis=1) >= -1e-4)


def test_all_neighbors_mutual_reachability(data):
    dataset, _ = data
    x = dataset[:1000]
    k = 8
    ind, dist, core = all_neighbors.build(
        x,
        k,
        all_neighbors.AllNeighborsParams(algo="brute_force"),
        distances=device_ndarray.empty((1000, k), np.float32),
        core_distances=device_ndarray.empty((1000,), np.float32),
        alpha=1.0,
    )
    pw = spatial.cdist(x, x, "sqeuclidean")
    want_core = np.sort(pw, 1)[:, k - 1]
    np.testing.assert_allclose(core.copy_to_host(), want_core, rtol=1e-3, atol=1e-3)
    ind, dist = ind.copy_to_host(), dist.copy_to_host()
    rows = np.arange(1000)[:, None]
    want = np.maximum(np.maximum(want_core[:, None], want_core[ind]), pw[rows, ind])
    np.testing.assert_allclose(dist, want, rtol=1e-3, atol=1e-3)


# --- kmeans --------------------------------------------------------------------


def test_kmeans(data):
    dataset, _ = data
    params = kmeans.KMeansParams(n_clusters=40, max_iter=50)
    centroids, inertia, n_iter = kmeans.fit(params, dataset)
    assert centroids.shape == (40, 32) and n_iter >= 1
    labels, inertia2 = kmeans.predict(params, dataset, centroids)
    assert labels.dtype == np.int32
    np.testing.assert_allclose(inertia, inertia2, rtol=1e-3)
    np.testing.assert_allclose(
        kmeans.cluster_cost(dataset, centroids), inertia, rtol=1e-3
    )
    # 40 well-separated blobs: k-means++ finds them all
    lab = labels.copy_to_host()
    assert len(np.unique(lab)) == 40
    c = centroids.copy_to_host()
    nearest = np.argmin(spatial.cdist(dataset, c, "sqeuclidean"), axis=1)
    np.testing.assert_array_equal(nearest, lab)


def test_kmeans_array_init_weights(data):
    dataset, _ = data
    init = dataset[:10].copy()
    params = kmeans.KMeansParams(n_clusters=10, init_method="Array", max_iter=5)
    w = np.ones(len(dataset), np.float32)
    c, inertia, _ = kmeans.fit(
        params, dataset, centroids=device_ndarray(init), sample_weights=w
    )
    c2, inertia2, _ = kmeans.fit(params, dataset, centroids=device_ndarray(init))
    np.testing.assert_allclose(
        c.copy_to_host(), c2.copy_to_host(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(inertia, inertia2, rtol=1e-4)
