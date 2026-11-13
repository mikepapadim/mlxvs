# Copyright © 2026 mikepapadim

"""mlxvs.cuvs: cuVS's Python API, checked as cuVS's own tests check it."""

import mlx.core as mx
import numpy as np
import pytest
from mlxvs.cuvs import common
from mlxvs.cuvs.cluster import kmeans
from mlxvs.cuvs.common import device_ndarray
from mlxvs.cuvs.distance import pairwise_distance
from mlxvs.cuvs.neighbors import (
    all_neighbors,
    brute_force,
    cagra,
    filters,
    ivf_binary,
    ivf_flat,
    ivf_pq,
    ivf_rabitq,
    ivf_sq,
    nn_descent,
    refine,
)
from mlxvs.cuvs.preprocessing import pca
from mlxvs.cuvs.preprocessing.quantize import binary, scalar

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
    _, i = brute_force.search(brute_force.build(index), queries, 7, prefilter=pre)
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
    _, i = ivf_flat.search(ivf_flat.SearchParams(n_probes=32), index, queries, 10)
    assert i.dtype == np.int64
    truth = _exact(dataset, queries, 10, metric)
    assert _recall(i.copy_to_host(), truth) > 0.99
    path = str(tmp_path / "ivf.bin")
    ivf_flat.save(path, index)
    _, i2 = ivf_flat.search(
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


# --- preprocessing ---------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("inplace", [True, False])
def test_scalar_quantizer(dtype, inplace):
    # as cuVS's test_scalar_quantizer.py
    x = np.random.default_rng(6).random((1000, 50)).astype(dtype)
    out = device_ndarray(np.zeros((1000, 50), np.int8)) if inplace else None
    q = scalar.train(scalar.QuantizerParams(quantile=0.99), device_ndarray(x))
    t = scalar.transform(q, device_ndarray(x), output=out)
    got = (out if inplace else t).copy_to_host()
    expected = np.int8(255 * (x - q.min) / (q.max - q.min) - 128)
    assert np.allclose(expected, got, atol=2, rtol=2)
    back = scalar.inverse_transform(q, t).copy_to_host().astype(np.float32)
    inside = (x > q.min) & (x < q.max)
    np.testing.assert_allclose(
        back[inside], x.astype(np.float32)[inside], atol=(q.max - q.min) / 255
    )
    # the interval holds the middle 99% of the values
    share = np.mean((x >= q.min) & (x <= q.max))
    assert 0.985 <= share <= 0.995


@pytest.mark.parametrize("n_cols", [7, 64, 100])
def test_binary_quantizer(n_cols):
    # as cuVS's test_binary_quantizer.py
    x = np.random.default_rng(7).standard_normal((500, n_cols)).astype(np.float32)
    got = binary.transform(device_ndarray(x)).copy_to_host()
    np.testing.assert_array_equal(got, np.packbits(x > 0, axis=-1, bitorder="little"))
    q = binary.train(binary.QuantizerParams(threshold="mean"), x)
    got = binary.transform(q, x).copy_to_host()
    np.testing.assert_array_equal(
        got, np.packbits(x > x.mean(0), axis=-1, bitorder="little")
    )


@pytest.mark.parametrize("whiten", [False, True])
def test_pca(whiten):
    rng = np.random.default_rng(8)
    x = (rng.standard_normal((800, 6)) @ rng.standard_normal((6, 20))).astype(
        np.float32
    )
    x += 0.01 * rng.standard_normal(x.shape).astype(np.float32)
    params = pca.Params(n_components=6, whiten=whiten)
    res = pca.fit_transform(params, x)
    assert res.trans_input.shape == (800, 6) and res.components.shape == (6, 20)
    # against the SVD of the centred data
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc.astype(np.float64), full_matrices=False)
    np.testing.assert_allclose(res.singular_vals.copy_to_host(), s[:6], rtol=1e-3)
    np.testing.assert_allclose(
        res.explained_var.copy_to_host(), s[:6] ** 2 / 799, rtol=1e-3
    )
    comps = res.components.copy_to_host()
    np.testing.assert_allclose(np.abs(comps @ vt[:6].T), np.eye(6), atol=1e-3)
    t = res.trans_input.copy_to_host()
    if whiten:
        np.testing.assert_allclose(t.var(0, ddof=1), np.ones(6), rtol=1e-2)
    t2 = pca.transform(
        params, x, res.components, res.singular_vals, res.mu
    ).copy_to_host()
    np.testing.assert_allclose(t, t2, rtol=1e-4, atol=1e-4)
    back = pca.inverse_transform(
        params, res.trans_input, res.components, res.singular_vals, res.mu
    )
    assert np.max(np.abs(back.copy_to_host() - x)) < 0.1


# --- ivf_sq --------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "cosine", "euclidean"]
)
def test_ivf_sq(data, tmp_path, metric):
    dataset, queries = data
    index = ivf_sq.build(ivf_sq.IndexParams(n_lists=32, metric=metric), dataset)
    d, i = ivf_sq.search(ivf_sq.SearchParams(n_probes=32), index, queries, 10)
    i, d = i.copy_to_host(), d.copy_to_host()
    assert i.dtype == np.int64
    truth = _exact(dataset, queries, 10, metric)
    assert _recall(i, truth) > 0.9
    # the distances are those of the decoded rows: close to the exact ones
    q, rows = queries, dataset[i[:, 0]]
    want = {
        "sqeuclidean": np.sum((q - rows) ** 2, 1),
        "euclidean": np.sqrt(np.sum((q - rows) ** 2, 1)),
        "inner_product": np.sum(q * rows, 1),
        "cosine": 1
        - np.sum(q * rows, 1)
        / (np.linalg.norm(q, axis=1) * np.linalg.norm(rows, axis=1)),
    }[metric]
    # one byte per dimension: errors of a fraction of the quantization step
    assert np.abs(d[:, 0] - want).mean() < 0.05 * np.abs(want).mean() + 0.01
    path = str(tmp_path / "sq.bin")
    ivf_sq.save(path, index)
    _, i2 = ivf_sq.search(
        ivf_sq.SearchParams(n_probes=32), ivf_sq.load(path), queries, 10
    )
    np.testing.assert_array_equal(i, i2.copy_to_host())


def test_ivf_sq_extend_filter(data):
    dataset, queries = data
    index = ivf_sq.build(
        ivf_sq.IndexParams(n_lists=16, add_data_on_build=False), dataset
    )
    assert len(index) == 0
    ivf_sq.extend(index, dataset, np.arange(len(dataset)))
    keep = np.zeros(len(dataset), bool)
    keep[::2] = True
    _, i = ivf_sq.search(
        ivf_sq.SearchParams(n_probes=16),
        index,
        queries,
        10,
        filter=filters.from_bitset(keep),
    )
    i = i.copy_to_host()
    assert keep[i].all()
    truth = np.flatnonzero(keep)[_exact(dataset[keep], queries, 10, "sqeuclidean")]
    assert _recall(i, truth) > 0.9


@pytest.mark.parametrize("d", [3, 130, 1100])
def test_ivf_sq_dims(d):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2000, d)).astype(np.float32)
    idx = __import__("mlxvs").IvfSqIndex.build(x, n_lists=8)
    ids, _ = idx.search(x[:50], 5, n_probes=8)
    assert (np.asarray(ids)[:, 0] == np.arange(50)).mean() > 0.95


# --- ivf_rabitq ----------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 3, 5, 9])
@pytest.mark.parametrize("mode", ["quant4", "quant8", "lut16", "lut32"])
def test_ivf_rabitq_modes(data, bits, mode):
    dataset, queries = data
    index = ivf_rabitq.build(
        ivf_rabitq.IndexParams(n_lists=32, bits_per_dim=bits), dataset
    )
    d, i = ivf_rabitq.search(
        ivf_rabitq.SearchParams(n_probes=32, mode=mode), index, queries, 10
    )
    i, d = i.copy_to_host(), d.copy_to_host()
    truth = _exact(dataset, queries, 10, "sqeuclidean")
    # more bits: closer to exact (RaBitQ's error falls as 2^-bits)
    floor = {1: 0.45, 3: 0.8, 5: 0.88, 9: 0.9}[bits]
    if mode == "quant4":
        floor = min(floor, 0.85)
    assert _recall(i, truth) > floor
    exact = np.sum((queries[:, None] - dataset[i]) ** 2, -1)
    rel = np.abs(d - exact).mean() / exact.mean()
    assert rel < {1: 0.35, 3: 0.1, 5: 0.1, 9: 0.1}[bits]


@pytest.mark.parametrize("metric", ["inner_product", "cosine", "euclidean"])
def test_ivf_rabitq_metrics(data, metric, tmp_path):
    dataset, queries = data
    if metric != "euclidean":
        dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    index = ivf_rabitq.build(
        ivf_rabitq.IndexParams(n_lists=32, metric=metric, bits_per_dim=7), dataset
    )
    sp = ivf_rabitq.SearchParams(n_probes=32, mode="quant8")
    _, i = ivf_rabitq.search(sp, index, queries, 10)
    assert _recall(i.copy_to_host(), _exact(dataset, queries, 10, metric)) > 0.85
    path = str(tmp_path / "rq.bin")
    ivf_rabitq.save(path, index)
    _, i2 = ivf_rabitq.search(sp, ivf_rabitq.load(path), queries, 10)
    np.testing.assert_array_equal(i.copy_to_host(), i2.copy_to_host())


def test_ivf_rabitq_extend_filter(data):
    dataset, queries = data
    half = len(dataset) // 2
    index = ivf_rabitq.build(
        ivf_rabitq.IndexParams(n_lists=16, bits_per_dim=4), dataset[:half]
    )
    ivf_rabitq.extend(index, dataset[half:], np.arange(half, len(dataset)))
    assert len(index) == len(dataset)
    keep = np.zeros(len(dataset), bool)
    keep[::2] = True
    sp = ivf_rabitq.SearchParams(n_probes=16, mode="lut32")
    _, i = ivf_rabitq.search(sp, index, queries, 10, filter=filters.from_bitset(keep))
    i = i.copy_to_host()
    assert keep[i].all()
    truth = np.flatnonzero(keep)[_exact(dataset[keep], queries, 10, "sqeuclidean")]
    assert _recall(i, truth) > 0.85


@pytest.mark.parametrize("d", [3, 200, 1100])
def test_ivf_rabitq_dims(d):
    import mlxvs

    rng = np.random.default_rng(10)
    x = rng.standard_normal((1500, d)).astype(np.float32)
    idx = mlxvs.IvfRabitqIndex.build(x, n_lists=8, bits_per_dim=6, fast_quantize=False)
    ids, _ = idx.search(x[:40] + 0.01, 5, n_probes=8, mode="lut32")
    assert (np.asarray(ids)[:, 0] == np.arange(40)).mean() > 0.9


# --- binary: Hamming brute force and IVF-binary ------------------------------


def _hamming(a, b):
    return np.unpackbits(a[:, None] ^ b[None], axis=2).sum(2)


@pytest.mark.parametrize("bytes_", [3, 16, 128])
def test_bitwise_hamming_brute_force(bytes_, tmp_path):
    rng = np.random.default_rng(11)
    data = rng.integers(0, 256, (700, bytes_), dtype=np.uint8)
    queries = rng.integers(0, 256, (40, bytes_), dtype=np.uint8)
    index = brute_force.build(data, metric="bitwise_hamming")
    path = str(tmp_path / "h.bin")
    brute_force.save(path, index)
    d, i = brute_force.search(brute_force.load(path), queries, 7)
    d, i = d.copy_to_host(), i.copy_to_host()
    want = np.sort(_hamming(queries, data), axis=1)[:, :7]
    np.testing.assert_array_equal(d, want)
    np.testing.assert_array_equal(
        np.take_along_axis(_hamming(queries, data), i, 1), want
    )
    keep = rng.random(700) < 0.4
    _, i2 = brute_force.search(index, queries, 5, prefilter=filters.from_bitset(keep))
    assert keep[i2.copy_to_host()].all()


def test_ivf_binary(tmp_path):
    rng = np.random.default_rng(12)
    centers = rng.standard_normal((30, 256)).astype(np.float32)
    x = centers[rng.integers(0, 30, 5000)] + 0.5 * rng.standard_normal(
        (5000, 256)
    ).astype(np.float32)
    codes = np.packbits(x > 0, axis=1)
    queries = codes[:100]
    index = ivf_binary.build(ivf_binary.IndexParams(n_lists=30), codes)
    assert index.dim == 32 and len(index) == 5000
    d, i = ivf_binary.search(ivf_binary.SearchParams(n_probes=30), index, queries, 10)
    d, i = d.copy_to_host(), i.copy_to_host()
    exact = np.sort(_hamming(queries, codes), axis=1)[:, :10]
    np.testing.assert_array_equal(d, exact)
    d5, _ = ivf_binary.search(ivf_binary.SearchParams(n_probes=5), index, queries, 10)
    assert np.mean(d5.copy_to_host() == exact) > 0.9
    path = str(tmp_path / "ib.bin")
    ivf_binary.save(path, index)
    _, i2 = ivf_binary.search(
        ivf_binary.SearchParams(n_probes=30), ivf_binary.load(path), queries, 10
    )
    np.testing.assert_array_equal(np.sort(i, 1), np.sort(i2.copy_to_host(), 1))
    empty = ivf_binary.build(
        ivf_binary.IndexParams(n_lists=30, add_data_on_build=False), codes
    )
    ivf_binary.extend(empty, codes, np.arange(5000))
    d3, _ = ivf_binary.search(ivf_binary.SearchParams(n_probes=30), empty, queries, 10)
    np.testing.assert_array_equal(d3.copy_to_host(), exact)


def test_ivf_binary_reconstruct():
    import mlxvs

    rng = np.random.default_rng(13)
    codes = rng.integers(0, 256, (3000, 13), dtype=np.uint8)
    idx = mlxvs.IvfBinaryIndex.build(codes, n_lists=16)
    got = np.asarray(idx.reconstruct(np.array([[0, 5, 2999], [-1, 7, 1]])))
    np.testing.assert_array_equal(got[0], codes[[0, 5, 2999]])
    np.testing.assert_array_equal(got[1, 1:], codes[[7, 1]])
    assert not got[1, 0].any()
    idx.extend(codes[:10], np.arange(5000, 5010))
    np.testing.assert_array_equal(np.asarray(idx.reconstruct([5003])), codes[[3]])


def test_tiny_inputs():
    # a couple of rows: MLX may bind them in the constant address space
    import mlxvs

    codes = np.array([[0b1111, 0], [0b1, 0]], dtype=np.uint8)
    ids, dist = mlxvs.hamming_knn(codes, codes[:1], 2)
    assert np.asarray(ids)[0].tolist() == [0, 1] and np.asarray(dist)[0].tolist() == [0.0, 3.0]
    idx = mlxvs.IvfBinaryIndex.build(codes, n_lists=1)
    assert np.asarray(idx.search(codes, 1, n_probes=1)[0])[:, 0].tolist() == [0, 1]
    x = np.random.default_rng(14).standard_normal((3, 8)).astype(np.float32)
    for cls in (mlxvs.IvfRabitqIndex, mlxvs.IvfSqIndex, mlxvs.IvfFlatIndex):
        ids, _ = cls.build(x, n_lists=1).search(x, 1, n_probes=1)
        assert np.asarray(ids)[:, 0].tolist() == [0, 1, 2]
