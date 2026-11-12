// Copyright © 2026 mikepapadim

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "../src/graph.h"
#include "mlxvs/mlxvs.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

NB_MODULE(_core, m) {
  m.doc() = "GPU k-NN search and k-NN graph construction on Apple silicon";

  nb::enum_<mlxvs::Metric>(m, "Metric", "How two rows compare.")
      .value("inner_product", mlxvs::Metric::InnerProduct)
      .value("l2", mlxvs::Metric::L2)
      .value("cosine", mlxvs::Metric::Cosine);

  nb::enum_<mlxvs::GraphBuild>(
      m, "GraphBuild", "How all_neighbors builds each cluster's graph.")
      .value("brute_force", mlxvs::GraphBuild::BruteForce)
      .value("nn_descent", mlxvs::GraphBuild::NnDescent);

  nb::enum_<mlxvs::SearchAlgo>(
      m, "SearchAlgo", "How graph_search lays out a query's work.")
      .value("auto", mlxvs::SearchAlgo::Auto)
      .value("single_cta", mlxvs::SearchAlgo::SingleCta)
      .value("multi_cta", mlxvs::SearchAlgo::MultiCta);

  m.def(
      "brute_force_knn",
      [](const mx::array& base,
         const mx::array& queries,
         int k,
         mlxvs::Metric metric,
         int tile_rows,
         mx::StreamOrDevice s) {
        auto r =
            mlxvs::brute_force_knn(base, queries, k, metric, {tile_rows}, s);
        return std::make_pair(r.ids, r.distances);
      },
      "base"_a,
      "queries"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "tile_rows"_a = 65536,
      "stream"_a = nb::none(),
      R"(
        Exact k-nearest neighbors of every query among the base rows.

        Each tile of ``tile_rows`` base rows is scored with a matrix multiply
        and a fused radix-select kernel keeps the best ``k`` per query.

        Args:
            base (array): Base rows ``[n, d]``, any real dtype.
            queries (array): Query rows ``[nq, d]``, any real dtype.
            k (int): Neighbors per query, at most 256.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            tile_rows (int): Base rows scored per step.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` (-1 where fewer exist)
            and distances ``float32 [nq, k]``, nearest first. For ``l2`` the
            distances are squared and ascending; otherwise similarities,
            descending.
      )");

  m.def(
      "kmeans_fit",
      [](const mx::array& x,
         int clusters,
         int max_iter,
         int sample_rows,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        return mlxvs::kmeans_fit(x, clusters, {max_iter, sample_rows, seed}, s);
      },
      "x"_a,
      "clusters"_a,
      nb::kw_only(),
      "max_iter"_a = 10,
      "sample_rows"_a = 100000,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        Fit k-means (squared Euclidean) to a random sample of the rows.

        k-means++ start, then ``max_iter`` Lloyd iterations.

        Args:
            x (array): Rows ``[n, d]``, any real dtype.
            clusters (int): Number of centroids.
            max_iter (int): Lloyd iterations.
            sample_rows (int): Rows sampled (at least 64 per cluster).
            seed (int): Random seed.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: Centroids, float32 ``[clusters, d]``.
      )");

  m.def(
      "kmeans_predict",
      [](const mx::array& x,
         const mx::array& centroids,
         int m,
         int tile_rows,
         mx::StreamOrDevice s) {
        return mlxvs::kmeans_predict(x, centroids, m, tile_rows, s);
      },
      "x"_a,
      "centroids"_a,
      "m"_a = 1,
      nb::kw_only(),
      "tile_rows"_a = 65536,
      "stream"_a = nb::none(),
      R"(
        The ``m`` nearest centroids of every row, nearest first.

        Args:
            x (array): Rows ``[n, d]``, any real dtype.
            centroids (array): Centroids ``[c, d]``, any real dtype.
            m (int): Centroids per row; more than 1 assigns rows to
              overlapping clusters, as batched all-neighbors does.
            tile_rows (int): Rows scored per step.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: Centroid indices, int32 ``[n, m]``.
      )");

  m.def(
      "nn_descent",
      [](const mx::array& rows,
         int k,
         mlxvs::Metric metric,
         int max_iterations,
         double termination_threshold,
         bool fp16,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::NnDescentOptions options;
        options.max_iterations = max_iterations;
        options.termination_threshold = termination_threshold;
        options.fp16 = fp16;
        options.seed = seed;
        auto r = mlxvs::nn_descent(rows, k, metric, options, s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        Approximate k-nearest-neighbor graph of the rows by NN-Descent.

        The ``k`` nearest other rows of every row, nearest first. Parameters
        follow cuVS's ``nn_descent.index_params`` where they correspond
        (``k`` is its ``graph_degree``). Up to 4096 rows the graph is
        computed exactly instead (as fast, and complete).

        Args:
            rows (array): Rows ``[n, d]``, any real dtype, ``n >= 2``.
            k (int): Neighbors per row, at most 128.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            max_iterations (int): Rounds after the random start, at most.
            termination_threshold (float): Stop after a round that replaced
              fewer than this fraction of the list entries.
            fp16 (bool): Score the joins from float16 rows with SIMD-group
              matrices (float32 accumulation); distances are float32 either
              way.
            seed (int): Random seed. Results still vary slightly between
              runs: concurrent writes to hashed slots race.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [n, k]`` (-1 where fewer were
            found) and distances ``float32 [n, k]``, as ``brute_force_knn``.
      )");

  m.def(
      "vamana_prune",
      [](const mx::array& rows,
         const mx::array& candidates,
         int graph_degree,
         mlxvs::Metric metric,
         float alpha,
         mx::StreamOrDevice s) {
        return mlxvs::vamana_prune(
            rows, candidates, graph_degree, metric, alpha, s);
      },
      "rows"_a,
      "candidates"_a,
      "graph_degree"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "alpha"_a = 1.2f,
      "stream"_a = nb::none(),
      R"(
        Vamana's robust pruning with JVector's diversity rule.

        Each row's candidates (ids; -1, out-of-range ids, the row itself and
        repeats are skipped) are ordered by similarity to the row; passes with
        alpha = 1.0, 1.2, ... up to ``alpha`` then keep a candidate unless one
        already kept is more similar to it than ``alpha`` times its
        similarity to the row, until ``graph_degree`` are kept, as JVector's
        ``VamanaDiversityProvider``. Similarities are JVector's: 1 / (1 +
        squared distance) for ``l2``, (1 + dot) / 2 otherwise.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype.
            candidates (array): Candidate ids ``[n, C]``, ``C <= 256``, such as
              a k-NN graph.
            graph_degree (int): The most neighbors kept, at most ``C``.
            metric (Metric): ``inner_product``, ``l2`` or ``cosine``.
            alpha (float): At least 1; larger keeps longer edges.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: ``int32 [n, graph_degree]``, most similar first, -1 filled.
      )");

  m.def(
      "cagra_extend",
      [](const mx::array& dataset,
         const mx::array& graph,
         const mx::array& new_rows,
         mlxvs::Metric metric,
         int itopk_size,
         mx::StreamOrDevice s) {
        mlxvs::ExtendOptions options;
        options.search.itopk_size = itopk_size;
        return mlxvs::cagra_extend(
            dataset, graph, new_rows, metric, options, s);
      },
      "dataset"_a,
      "graph"_a,
      "new_rows"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "itopk_size"_a = 256,
      "stream"_a = nb::none(),
      R"(
        Adds rows to a CAGRA search graph without rebuilding it, as cuVS's
        ``cagra.extend``.

        Each new row's ``2 * degree`` nearest rows are found by a graph
        search and pruned to ``degree`` by rank-based detour counts; each of
        its first ``degree / 2`` neighbors gives it the slot (in its second
        half) of the neighbor with the most incoming edges; the new row's list
        interleaves its own neighbors with those displaced.
        ``CagraIndex.extend`` calls it.

        Args:
            dataset (array): The index's rows ``[n, d]``, as stored for search.
            graph (array): Its search graph ``[n, degree]``.
            new_rows (array): Rows to add ``[m, d]``, transformed as the
              dataset (unit rows for cosine).
            metric (Metric): As the search.
            itopk_size (int): The search list length (at least
              ``max(4 * degree, 256)``).
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: ``int32 [n + m, degree]``, new rows at ids ``n`` and up.
      )");

  m.def(
      "ivf_flat_build",
      [](const mx::array& rows,
         mlxvs::Metric metric,
         int n_lists,
         int kmeans_n_iters,
         double kmeans_trainset_fraction,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::IvfFlatOptions o;
        o.n_lists = n_lists;
        o.kmeans_n_iters = kmeans_n_iters;
        o.kmeans_trainset_fraction = kmeans_trainset_fraction;
        o.seed = seed;
        auto i = mlxvs::ivf_flat_build(rows, metric, o, s);
        return nb::make_tuple(i.centers, i.data, i.indices, i.offsets);
      },
      "rows"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "n_lists"_a = 1024,
      "kmeans_n_iters"_a = 20,
      "kmeans_trainset_fraction"_a = 0.5,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      "Builds an IVF-Flat index; returns (centers, data, indices, offsets). "
      "IvfFlatIndex.build calls it.");

  m.def(
      "ivf_flat_search",
      [](const mx::array& centers,
         const mx::array& data,
         const mx::array& indices,
         const mx::array& offsets,
         mlxvs::Metric metric,
         const mx::array& queries,
         int k,
         int n_probes,
         mx::StreamOrDevice s) {
        mlxvs::IvfFlatIndex i{centers, data, indices, offsets, metric};
        mlxvs::IvfSearchOptions o;
        o.n_probes = n_probes;
        auto r = mlxvs::ivf_flat_search(i, queries, k, o, s);
        return std::make_pair(r.ids, r.distances);
      },
      "centers"_a,
      "data"_a,
      "indices"_a,
      "offsets"_a,
      "metric"_a,
      "queries"_a,
      "k"_a,
      nb::kw_only(),
      "n_probes"_a = 20,
      "stream"_a = nb::none(),
      "Searches an IVF-Flat index (IvfFlatIndex.search calls it).");

  m.def(
      "ivf_flat_extend",
      [](const mx::array& centers,
         const mx::array& data,
         const mx::array& indices,
         const mx::array& offsets,
         mlxvs::Metric metric,
         const mx::array& new_rows,
         std::optional<mx::array> new_ids,
         mx::StreamOrDevice s) {
        mlxvs::IvfFlatIndex i{centers, data, indices, offsets, metric};
        auto r = mlxvs::ivf_flat_extend(i, new_rows, new_ids, s);
        return nb::make_tuple(r.centers, r.data, r.indices, r.offsets);
      },
      "centers"_a,
      "data"_a,
      "indices"_a,
      "offsets"_a,
      "metric"_a,
      "new_rows"_a,
      "new_ids"_a = nb::none(),
      "stream"_a = nb::none(),
      "Adds rows to an IVF-Flat index (IvfFlatIndex.extend calls it).");

  m.def(
      "ivf_pq_build",
      [](const mx::array& rows,
         mlxvs::Metric metric,
         int n_lists,
         int pq_dim,
         int pq_bits,
         int kmeans_n_iters,
         double kmeans_trainset_fraction,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::IvfPqOptions o;
        o.n_lists = n_lists;
        o.pq_dim = pq_dim;
        o.pq_bits = pq_bits;
        o.kmeans_n_iters = kmeans_n_iters;
        o.kmeans_trainset_fraction = kmeans_trainset_fraction;
        o.seed = seed;
        auto i = mlxvs::ivf_pq_build(rows, metric, o, s);
        return nb::make_tuple(
            i.centers, i.codebooks, i.codes, i.indices, i.offsets, i.dim);
      },
      "rows"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "n_lists"_a = 1024,
      "pq_dim"_a = 0,
      "pq_bits"_a = 8,
      "kmeans_n_iters"_a = 20,
      "kmeans_trainset_fraction"_a = 0.5,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      "Builds an IVF-PQ index; returns (centers, codebooks, codes, indices, "
      "offsets, dim). IvfPqIndex.build calls it.");

  m.def(
      "ivf_pq_search",
      [](const mx::array& centers,
         const mx::array& codebooks,
         const mx::array& codes,
         const mx::array& indices,
         const mx::array& offsets,
         mlxvs::Metric metric,
         int dim,
         const mx::array& queries,
         int k,
         int n_probes,
         mx::StreamOrDevice s) {
        mlxvs::IvfPqIndex i{
            centers, codebooks, codes, indices, offsets, metric, dim};
        mlxvs::IvfSearchOptions o;
        o.n_probes = n_probes;
        auto r = mlxvs::ivf_pq_search(i, queries, k, o, s);
        return std::make_pair(r.ids, r.distances);
      },
      "centers"_a,
      "codebooks"_a,
      "codes"_a,
      "indices"_a,
      "offsets"_a,
      "metric"_a,
      "dim"_a,
      "queries"_a,
      "k"_a,
      nb::kw_only(),
      "n_probes"_a = 20,
      "stream"_a = nb::none(),
      "Searches an IVF-PQ index (IvfPqIndex.search calls it).");

  m.def(
      "ivf_pq_extend",
      [](const mx::array& centers,
         const mx::array& codebooks,
         const mx::array& codes,
         const mx::array& indices,
         const mx::array& offsets,
         mlxvs::Metric metric,
         int dim,
         const mx::array& new_rows,
         std::optional<mx::array> new_ids,
         mx::StreamOrDevice s) {
        mlxvs::IvfPqIndex i{
            centers, codebooks, codes, indices, offsets, metric, dim};
        auto r = mlxvs::ivf_pq_extend(i, new_rows, new_ids, s);
        return nb::make_tuple(r.codes, r.indices, r.offsets);
      },
      "centers"_a,
      "codebooks"_a,
      "codes"_a,
      "indices"_a,
      "offsets"_a,
      "metric"_a,
      "dim"_a,
      "new_rows"_a,
      "new_ids"_a = nb::none(),
      "stream"_a = nb::none(),
      "Adds rows to an IVF-PQ index (IvfPqIndex.extend calls it).");

  m.def(
      "refine",
      [](const mx::array& dataset,
         const mx::array& queries,
         const mx::array& candidates,
         int k,
         mlxvs::Metric metric,
         mx::StreamOrDevice s) {
        auto r = mlxvs::refine(dataset, queries, candidates, k, metric, s);
        return std::make_pair(r.ids, r.distances);
      },
      "dataset"_a,
      "queries"_a,
      "candidates"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      "stream"_a = nb::none(),
      R"(
        Exact re-ranking, as cuVS's ``refine``: the k nearest of each query's
        candidate ids by exact distances, nearest first.

        Args:
            dataset (array): The rows ``[n, d]`` the ids index.
            queries (array): ``[nq, d]``.
            candidates (array): Candidate ids ``int [nq, c]`` (-1 skipped),
              such as an approximate search for ``c = 2k`` to ``4k``.
            k (int): At most ``c``.
            metric (Metric): ``inner_product``, ``l2`` or ``cosine``.

        Returns:
            tuple(array, array): ids and distances as ``brute_force_knn``.
      )");

  m.def(
      "vpq_build",
      [](const mx::array& rows,
         int pq_dim,
         int pq_bits,
         int vq_n_centers,
         int kmeans_n_iters,
         unsigned long long seed,
         mx::StreamOrDevice s) {
        mlxvs::VpqOptions o;
        o.pq_dim = pq_dim;
        o.pq_bits = pq_bits;
        o.vq_n_centers = vq_n_centers;
        o.kmeans_n_iters = kmeans_n_iters;
        o.seed = seed;
        auto v = mlxvs::vpq_build(rows, o, s);
        return nb::make_tuple(
            v.vq_centers, v.pq_codebook, v.vq_ids, v.pq_codes, v.dim);
      },
      "rows"_a,
      nb::kw_only(),
      "pq_dim"_a = 0,
      "pq_bits"_a = 8,
      "vq_n_centers"_a = 0,
      "kmeans_n_iters"_a = 25,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      "Compresses rows as cuVS's VPQ dataset; returns (vq_centers, "
      "pq_codebook, vq_ids, pq_codes, dim). CagraIndex.compress calls it.");

  m.def(
      "graph_search_vpq",
      [](const mx::array& vq_centers,
         const mx::array& pq_codebook,
         const mx::array& vq_ids,
         const mx::array& pq_codes,
         int dim,
         const mx::array& graph,
         const mx::array& queries,
         int k,
         mlxvs::Metric metric,
         int itopk_size,
         int search_width,
         mlxvs::SearchAlgo algo,
         std::optional<mx::array> filter,
         mx::StreamOrDevice s) {
        mlxvs::VpqDataset v{vq_centers, pq_codebook, vq_ids, pq_codes, dim};
        mlxvs::SearchOptions o;
        o.itopk_size = itopk_size;
        o.search_width = search_width;
        o.algo = algo;
        o.filter = filter;
        auto r = mlxvs::graph_search_vpq(v, graph, queries, k, metric, o, s);
        return std::make_pair(r.ids, r.distances);
      },
      "vq_centers"_a,
      "pq_codebook"_a,
      "vq_ids"_a,
      "pq_codes"_a,
      "dim"_a,
      "graph"_a,
      "queries"_a,
      "k"_a,
      "metric"_a,
      nb::kw_only(),
      "itopk_size"_a = 64,
      "search_width"_a = 1,
      "algo"_a = mlxvs::SearchAlgo::Auto,
      "filter"_a = nb::none(),
      "stream"_a = nb::none(),
      "graph_search over a VPQ dataset (CagraIndex.search calls it when the "
      "index is compressed).");

  m.def(
      "save_hnswlib",
      [](const std::string& path,
         const mx::array& dataset,
         const mx::array& graph,
         mlxvs::Metric metric) {
        mlxvs::save_hnswlib(path, dataset, graph, metric);
      },
      "path"_a,
      "dataset"_a,
      "graph"_a,
      "metric"_a = mlxvs::Metric::L2,
      R"(
        Writes rows and their search graph in hnswlib's binary format, as
        cuVS's ``serialize_to_hnswlib`` (``CagraIndex.save_hnswlib`` calls
        it): the graph as hnswlib's bottom layer entered at the medoid,
        ``M`` half the degree, float32 rows, row ids as labels.
      )");

  m.def(
      "save_jvector",
      [](const std::string& path,
         const mx::array& dataset,
         const mx::array& graph,
         mlxvs::Metric metric) {
        mlxvs::save_jvector(path, dataset, graph, metric);
      },
      "path"_a,
      "dataset"_a,
      "graph"_a,
      "metric"_a = mlxvs::Metric::L2,
      R"(
        Writes rows and their search graph in JVector's on-disk format
        (``OnDiskGraphIndex``, format 6 as JVector 4 writes it; called by
        ``CagraIndex.save_jvector``): one layer entered at the medoid, the
        rows as inline float32 vectors.
      )");

  m.def(
      "_reverse_lists",
      [](const mx::array& graph, int rcap, mx::StreamOrDevice s) {
        mx::StreamContext context(mx::to_stream(s, mx::Device::gpu));
        auto g =
            graph.dtype() == mx::int32 ? graph : mx::astype(graph, mx::int32);
        return mlxvs::detail::reverse_lists(g, rcap);
      },
      "graph"_a,
      "rcap"_a,
      "stream"_a = nb::none());

  m.def(
      "cagra_optimize",
      [](const mx::array& knn_graph, int graph_degree, mx::StreamOrDevice s) {
        return mlxvs::cagra_optimize(knn_graph, graph_degree, s);
      },
      "knn_graph"_a,
      "graph_degree"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Prunes a k-NN graph into a search graph, as CAGRA's ``optimize``.

        Rank-based pruning keeps the edges with the fewest two-hop detours
        (an edge X -> Y has a detour through Z when Z comes before Y in X's
        list and Y before it in Z's), ties by rank; then the second half of
        each list takes reverse edges not there yet, so that nodes with few
        incoming edges stay reachable. Needs only the ids.

        Args:
            knn_graph (array): Neighbor ids ``[n, K]``, nearest first, as
              ``nn_descent`` or ``all_neighbors`` return; ``K <= 256``.
            graph_degree (int): Degree of the result, at most ``K``.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            array: ``int32 [n, graph_degree]`` for ``graph_search``.
      )");

  m.def(
      "graph_search",
      [](const mx::array& base,
         const mx::array& graph,
         const mx::array& queries,
         int k,
         mlxvs::Metric metric,
         int itopk_size,
         int search_width,
         mlxvs::SearchAlgo algo,
         int team_size,
         int max_iterations,
         int min_iterations,
         int num_random_samplings,
         unsigned long long seed,
         std::optional<mx::array> filter,
         float filtering_rate,
         std::optional<mx::array> seeds,
         mx::StreamOrDevice s) {
        mlxvs::SearchOptions options;
        options.filter = filter;
        options.seeds = seeds;
        options.filtering_rate = filtering_rate;
        options.algo = algo;
        options.team_size = team_size;
        options.itopk_size = itopk_size;
        options.search_width = search_width;
        options.max_iterations = max_iterations;
        options.min_iterations = min_iterations;
        options.num_random_samplings = num_random_samplings;
        options.seed = seed;
        auto r =
            mlxvs::graph_search(base, graph, queries, k, metric, options, s);
        return std::make_pair(r.ids, r.distances);
      },
      "base"_a,
      "graph"_a,
      "queries"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "itopk_size"_a = 64,
      "search_width"_a = 1,
      "algo"_a = mlxvs::SearchAlgo::Auto,
      "team_size"_a = 0,
      "max_iterations"_a = 0,
      "min_iterations"_a = 0,
      "num_random_samplings"_a = 1,
      "seed"_a = 0x128394,
      "filter"_a = nb::none(),
      "filtering_rate"_a = -1.0f,
      "seeds"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        Approximate k-nearest neighbors of every query by searching a graph.

        CAGRA's search. Single-CTA: one threadgroup per query keeps a sorted
        list of ``itopk_size`` nodes, starts from random nodes and expands the
        best unexpanded ones (``search_width`` per round) until every node of
        the list has been expanded. Multi-CTA: ``max(search_width, itopk_size
        / 32)`` threadgroups per query, each a list of 32 expanding one node
        per iteration, sharing the set of expanded nodes: lower latency for a
        few queries. Parameters follow cuVS's ``cagra.SearchParams``. Any fixed-degree graph works: the k-NN graph of
        ``nn_descent`` or ``all_neighbors``, or a pruned one.

        Args:
            base (array): Base rows ``[n, d]``, any real dtype.
            graph (array): Neighbor ids of every base row ``[n, degree]``,
              integers; -1 entries are skipped.
            queries (array): Query rows ``[nq, d]``, any real dtype.
            k (int): Neighbors per query, at most ``itopk_size``.
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            itopk_size (int): Length of the internal list (rounded up to a
              multiple of 32; at most 256 single-CTA, 1024 multi-CTA). Longer:
              higher recall, slower.
            search_width (int): Nodes expanded per round;
              ``search_width * degree <= 512``.
            algo (SearchAlgo): ``"auto"`` (as cuVS: single-CTA for at least
              two queries per GPU core and ``itopk_size <= 256``),
              ``"single_cta"`` or ``"multi_cta"`` (graph degree at most 224).
            team_size (int): Lanes per distance, 4, 8, 16 or 32; 0: by the
              dimension, as cuVS.
            max_iterations (int): The most iterations, the random start
              counting as the first; 0: automatic, as cuVS.
            min_iterations (int): The fewest iterations.
            num_random_samplings (int): Each random start slot takes the
              nearest of this many random nodes.
            seed (int): Seed of the random start nodes.
            filter (array, optional): Rows that may be returned, as cuVS's
              filters: bool ``[n]`` (True: allowed) or a bitset ``uint32
              [ceil(n / 32)]`` for every query, or bool ``[nq, n]`` /
              ``uint32 [nq, ceil(n / 32)]``, one per query (cuVS's bitmap). Rejected rows are still traversed
              but never returned. With few rows allowed, use multi-CTA (its
              list grows with the filtering rate, as cuVS) or a larger
              ``itopk_size``.
            filtering_rate (float): The fraction of rows the filter rejects,
              for multi-CTA's list length; negative: computed.
            seeds (array, optional): Start nodes per query, integers
              ``[nq, num_seeds]`` (out-of-range: random), as cuVS's seed
              nodes.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [nq, k]`` (-1 where fewer were
            found) and distances ``float32 [nq, k]``, as ``brute_force_knn``.
      )");

  auto all_neighbors_options = [](mlxvs::GraphBuild algo,
                                  int n_clusters,
                                  int overlap_factor,
                                  int brute_force_below,
                                  long long max_cluster_rows,
                                  bool verbose,
                                  int max_iterations,
                                  double termination_threshold,
                                  bool fp16,
                                  unsigned long long seed) {
    mlxvs::AllNeighborsOptions options;
    options.algo = algo;
    options.n_clusters = n_clusters;
    options.overlap_factor = overlap_factor;
    options.brute_force_below = brute_force_below;
    options.max_cluster_rows = max_cluster_rows;
    options.verbose = verbose;
    options.kmeans.seed = seed;
    options.nn_descent.max_iterations = max_iterations;
    options.nn_descent.termination_threshold = termination_threshold;
    options.nn_descent.fp16 = fp16;
    options.nn_descent.seed = seed;
    return options;
  };

  // rows left on the host (a NumPy array, a memory-mapped file): read through
  // their buffer, a cluster at a time
  m.def(
      "all_neighbors_host",
      [all_neighbors_options](
          nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
              rows,
          int k,
          mlxvs::Metric metric,
          mlxvs::GraphBuild algo,
          int n_clusters,
          int overlap_factor,
          int brute_force_below,
          long long max_cluster_rows,
          bool verbose,
          int max_iterations,
          double termination_threshold,
          bool fp16,
          unsigned long long seed,
          mx::StreamOrDevice s) {
        mlxvs::HostRows host{
            rows.data(), (long long)rows.shape(0), (int)rows.shape(1)};
        auto r = mlxvs::all_neighbors(
            host,
            k,
            metric,
            all_neighbors_options(
                algo,
                n_clusters,
                overlap_factor,
                brute_force_below,
                max_cluster_rows,
                verbose,
                max_iterations,
                termination_threshold,
                fp16,
                seed),
            s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "algo"_a = mlxvs::GraphBuild::NnDescent,
      "n_clusters"_a = 0,
      "overlap_factor"_a = 2,
      "brute_force_below"_a = 4096,
      "max_cluster_rows"_a = 0,
      "verbose"_a = false,
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      "all_neighbors of float32 rows read from host memory.");

  m.def(
      "all_neighbors",
      [all_neighbors_options](
          const mx::array& rows,
          int k,
          mlxvs::Metric metric,
          mlxvs::GraphBuild algo,
          int n_clusters,
          int overlap_factor,
          int brute_force_below,
          long long max_cluster_rows,
          bool verbose,
          int max_iterations,
          double termination_threshold,
          bool fp16,
          unsigned long long seed,
          mx::StreamOrDevice s) {
        auto r = mlxvs::all_neighbors(
            rows,
            k,
            metric,
            all_neighbors_options(
                algo,
                n_clusters,
                overlap_factor,
                brute_force_below,
                max_cluster_rows,
                verbose,
                max_iterations,
                termination_threshold,
                fp16,
                seed),
            s);
        return std::make_pair(r.ids, r.distances);
      },
      "rows"_a,
      "k"_a,
      "metric"_a = mlxvs::Metric::L2,
      nb::kw_only(),
      "algo"_a = mlxvs::GraphBuild::NnDescent,
      "n_clusters"_a = 0,
      "overlap_factor"_a = 2,
      "brute_force_below"_a = 4096,
      "max_cluster_rows"_a = 0,
      "verbose"_a = false,
      "max_iterations"_a = 24,
      "termination_threshold"_a = 0.0001,
      "fp16"_a = true,
      "seed"_a = 42,
      "stream"_a = nb::none(),
      R"(
        k-nearest-neighbor graph of the rows, in overlapping k-means clusters.

        As cuVS's ``all_neighbors.build``: with ``n_clusters > 1`` the rows
        are clustered by k-means (unit rows for ``cosine``), every row joins
        its ``overlap_factor`` nearest clusters, each cluster's graph is built
        on its own, and every row keeps the best ``k`` of the lists its
        clusters gave it. One cluster is on the GPU at a time.

        A float32 NumPy array, including a ``numpy.memmap`` of a file larger
        than memory, is read where it is: only one cluster's rows at a time
        are copied to the GPU.

        Args:
            rows (array): Rows ``[n, d]``, any real dtype, ``n >= 2``.
            k (int): Neighbors per row, at most 128 with ``nn_descent`` (255
              with ``brute_force``).
            metric (Metric): ``inner_product``, ``l2`` (squared) or ``cosine``.
            algo (GraphBuild): ``nn_descent`` (approximate) or
              ``brute_force`` (exact) for each cluster.
            n_clusters (int): k-means clusters. 0 (default): as many as the
              GPU's memory asks for, one when the rows fit. 1: all rows at
              once.
            overlap_factor (int): Clusters each row joins.
            brute_force_below (int): Clusters of at most this many rows are
              built exactly whatever ``algo``.
            max_cluster_rows (int): The most rows of one cluster; a larger
              k-means cluster is clustered again. 0: from the GPU's memory.
            verbose (bool): Print the clusters and the progress to stderr.
            max_iterations (int): NN-Descent rounds after the random start.
            termination_threshold (float): NN-Descent stops after a round
              that replaced fewer than this fraction of the list entries.
            fp16 (bool): NN-Descent joins from float16 rows.
            seed (int): Random seed of k-means and NN-Descent.
            stream (Stream, optional): Stream or device. GPU only.

        Returns:
            tuple(array, array): ids ``int32 [n, k]`` (-1 where fewer were
            found) and distances ``float32 [n, k]``, as ``brute_force_knn``.
      )");
}
