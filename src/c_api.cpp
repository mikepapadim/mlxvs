// Copyright © 2026 mikepapadim

// The C API (include/mlxvs/mlxvs_c.h) over the C++ one.

#include "mlxvs/mlxvs.h"
#include "mlxvs/mlxvs_c.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifndef MLXVS_VERSION
#define MLXVS_VERSION "unknown"
#endif

namespace mx = mlx::core;

struct mlxvs_index_s {
  mx::array dataset; // float32 or float16 [n, d], unit rows for cosine
  mx::array graph; // int32 [n, degree]
  mlxvs_metric metric;
};

namespace {

thread_local std::string last_error;

// An mlxvs_status from the exception being handled, its message kept.
template <typename F>
int guarded(F&& f, int io_status = MLXVS_ERROR_INTERNAL) {
  try {
    f();
    last_error.clear();
    return MLXVS_OK;
  } catch (const std::invalid_argument& e) {
    last_error = e.what();
    return MLXVS_ERROR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    last_error = e.what();
    return io_status;
  } catch (...) {
    last_error = "unknown error";
    return MLXVS_ERROR_INTERNAL;
  }
}

void require(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("[mlxvs] ") + what);
  }
}

const char* metric_name(mlxvs_metric m) {
  switch (m) {
    case MLXVS_INNER_PRODUCT:
      return "inner_product";
    case MLXVS_L2:
      return "l2";
    case MLXVS_COSINE:
      return "cosine";
  }
  return "";
}

mlxvs::Metric to_metric(mlxvs_metric m) {
  switch (m) {
    case MLXVS_INNER_PRODUCT:
      return mlxvs::Metric::InnerProduct;
    case MLXVS_L2:
      return mlxvs::Metric::L2;
    case MLXVS_COSINE:
      return mlxvs::Metric::Cosine;
  }
  throw std::invalid_argument("[mlxvs] unknown metric.");
}

mx::array unit_rows(const mx::array& x) {
  auto norm = mx::sqrt(mx::sum(mx::square(x), 1, /*keepdims=*/true));
  return x / mx::maximum(norm, mx::array(1e-30f));
}

mx::array host_rows(const float* rows, int64_t n, int32_t d, const char* name) {
  require(rows != nullptr || n == 0, "rows must not be NULL.");
  require(
      n >= 0 && n < (1LL << 31), "the number of rows must be in [0, 2^31).");
  require(d >= 1, "the dimension must be at least 1.");
  (void)name;
  return mx::array(rows, {(int)n, d}, mx::float32);
}

// The dataset as stored for search: unit rows for cosine, float32 or float16.
mx::array stored(const mx::array& rows, mlxvs_metric metric, bool float16) {
  auto x = metric == MLXVS_COSINE ? unit_rows(rows) : rows;
  return mx::contiguous(mx::astype(x, float16 ? mx::float16 : mx::float32));
}

void copy_out(const mx::array& a, void* out, size_t bytes) {
  auto c = mx::contiguous(a);
  mx::eval(c);
  std::memcpy(out, c.data<char>(), bytes);
}

// The search behind every mlxvs_index_search* function; throws on errors.
void search_impl(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const mx::array* filter,
    const int32_t* seeds,
    int32_t num_seeds,
    int32_t* ids,
    float* distances) {
  require(index != nullptr, "index must not be NULL.");
  require(ids != nullptr, "ids must not be NULL.");
  mlxvs_search_params p;
  mlxvs_search_params_init(&p);
  if (params != nullptr) {
    p = *params;
  }
  require(
      p.algo >= MLXVS_SEARCH_AUTO && p.algo <= MLXVS_SEARCH_MULTI_CTA,
      "algo must be a mlxvs_search_algo.");
  auto q = host_rows(queries, nq, index->dataset.shape(1), "queries");
  auto metric = to_metric(index->metric);
  bool bytes =
      index->dataset.dtype() == mx::int8 || index->dataset.dtype() == mx::uint8;
  if (index->metric == MLXVS_COSINE && !bytes) {
    // the dataset is stored normalized: the dot product of unit rows is the
    // cosine similarity
    q = unit_rows(q);
    metric = mlxvs::Metric::InnerProduct;
  }
  mlxvs::SearchOptions o;
  o.algo = static_cast<mlxvs::SearchAlgo>(p.algo);
  o.itopk_size = p.itopk_size;
  o.search_width = p.search_width;
  o.max_iterations = p.max_iterations;
  o.min_iterations = p.min_iterations;
  o.team_size = p.team_size;
  o.num_random_samplings = p.num_random_samplings;
  o.seed = p.seed;
  if (filter != nullptr) {
    o.filter = *filter;
  }
  if (seeds != nullptr && num_seeds > 0) {
    o.seeds = mx::array(seeds, {(int)nq, num_seeds}, mx::int32);
  }
  auto r = mlxvs::graph_search(index->dataset, index->graph, q, k, metric, o);
  size_t count = (size_t)nq * k;
  copy_out(r.ids, ids, count * sizeof(int32_t));
  if (distances != nullptr) {
    copy_out(r.distances, distances, count * sizeof(float));
  }
}

} // namespace

extern "C" {

void mlxvs_build_params_init(mlxvs_build_params* p) {
  if (p == nullptr) {
    return;
  }
  p->graph_degree = 32;
  p->intermediate_graph_degree = 64;
  p->graph = MLXVS_GRAPH_CAGRA;
  p->alpha = 1.2f;
  p->float16 = 0;
}

void mlxvs_search_params_init(mlxvs_search_params* p) {
  if (p == nullptr) {
    return;
  }
  mlxvs::SearchOptions o;
  p->itopk_size = o.itopk_size;
  p->search_width = o.search_width;
  p->algo = MLXVS_SEARCH_AUTO;
  p->max_iterations = o.max_iterations;
  p->min_iterations = o.min_iterations;
  p->team_size = o.team_size;
  p->num_random_samplings = o.num_random_samplings;
  p->seed = o.seed;
}

const char* mlxvs_last_error(void) {
  return last_error.c_str();
}

const char* mlxvs_version(void) {
  return MLXVS_VERSION;
}

int mlxvs_index_build(
    const float* rows,
    int64_t n,
    int32_t d,
    mlxvs_metric metric,
    const mlxvs_build_params* params,
    mlxvs_index* out) {
  return mlxvs_index_build_typed(
      rows, MLXVS_FLOAT32, n, d, metric, params, out);
}

int mlxvs_index_build_typed(
    const void* rows,
    mlxvs_dtype dtype,
    int64_t n,
    int32_t d,
    mlxvs_metric metric,
    const mlxvs_build_params* params,
    mlxvs_index* out) {
  return guarded([&] {
    require(out != nullptr, "out must not be NULL.");
    *out = nullptr;
    mlxvs_build_params p;
    mlxvs_build_params_init(&p);
    if (params != nullptr) {
      p = *params;
    }
    require(
        p.graph == MLXVS_GRAPH_CAGRA || p.graph == MLXVS_GRAPH_VAMANA,
        "graph must be MLXVS_GRAPH_CAGRA or MLXVS_GRAPH_VAMANA.");
    require(
        dtype == MLXVS_FLOAT32 || dtype == MLXVS_INT8 || dtype == MLXVS_UINT8,
        "dtype must be a mlxvs_dtype.");
    require(rows != nullptr || n == 0, "rows must not be NULL.");
    require(
        n >= 0 && n < (1LL << 31), "the number of rows must be in [0, 2^31).");
    require(d >= 1, "the dimension must be at least 1.");
    auto raw = dtype == MLXVS_INT8
        ? mx::array((const int8_t*)rows, {(int)n, d}, mx::int8)
        : dtype == MLXVS_UINT8
        ? mx::array((const uint8_t*)rows, {(int)n, d}, mx::uint8)
        : mx::array((const float*)rows, {(int)n, d}, mx::float32);
    auto x = mx::astype(raw, mx::float32);
    auto m = to_metric(metric);
    mx::array graph = mx::array(0);
    if (p.graph == MLXVS_GRAPH_VAMANA) {
      mlxvs::VamanaOptions o;
      o.graph_degree = p.graph_degree;
      o.intermediate_graph_degree = p.intermediate_graph_degree;
      o.alpha = p.alpha;
      graph = mlxvs::vamana_build(x, m, o);
    } else {
      mlxvs::CagraOptions o;
      o.graph_degree = p.graph_degree;
      o.intermediate_graph_degree = p.intermediate_graph_degree;
      graph = mlxvs::cagra_build(x, m, o);
    }
    auto dataset = dtype == MLXVS_FLOAT32 ? stored(x, metric, p.float16 != 0)
                                          : mx::contiguous(raw);
    mx::eval(dataset, graph);
    *out = new mlxvs_index_s{dataset, graph, metric};
  });
}

int mlxvs_index_from_graph(
    const float* rows,
    int64_t n,
    int32_t d,
    const int32_t* graph,
    int32_t graph_degree,
    mlxvs_metric metric,
    mlxvs_index* out) {
  return guarded([&] {
    require(out != nullptr, "out must not be NULL.");
    *out = nullptr;
    require(graph != nullptr, "graph must not be NULL.");
    require(graph_degree >= 1, "graph_degree must be at least 1.");
    to_metric(metric);
    auto x = host_rows(rows, n, d, "rows");
    mx::array g(graph, {(int)n, graph_degree}, mx::int32);
    auto dataset = stored(x, metric, false);
    mx::eval(dataset, g);
    *out = new mlxvs_index_s{dataset, g, metric};
  });
}

int mlxvs_index_search(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    int32_t* ids,
    float* distances) {
  return mlxvs_index_search_filtered(
      index, queries, nq, k, params, nullptr, ids, distances);
}

int mlxvs_index_search_filtered(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const uint32_t* filter,
    int32_t* ids,
    float* distances) {
  return mlxvs_index_search_seeded(
      index, queries, nq, k, params, filter, nullptr, 0, ids, distances);
}

int mlxvs_index_search_bitmap(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const uint32_t* bitmap,
    int32_t* ids,
    float* distances) {
  return guarded([&] {
    require(
        index != nullptr && bitmap != nullptr,
        "index and bitmap must not be NULL.");
    int words = (int)((index->dataset.shape(0) + 31) / 32);
    mx::array filter(bitmap, {(int)nq, words}, mx::uint32);
    search_impl(
        index, queries, nq, k, params, &filter, nullptr, 0, ids, distances);
  });
}

int mlxvs_index_search_seeded(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const uint32_t* filter,
    const int32_t* seeds,
    int32_t num_seeds,
    int32_t* ids,
    float* distances) {
  return guarded([&] {
    std::optional<mx::array> f;
    if (filter != nullptr) {
      require(index != nullptr, "index must not be NULL.");
      int words = (int)((index->dataset.shape(0) + 31) / 32);
      f = mx::array(filter, {words}, mx::uint32);
    }
    search_impl(
        index,
        queries,
        nq,
        k,
        params,
        f ? &*f : nullptr,
        seeds,
        num_seeds,
        ids,
        distances);
  });
}

int mlxvs_index_save(mlxvs_index index, const char* path) {
  return guarded(
      [&] {
        require(
            index != nullptr && path != nullptr,
            "index and path must not be NULL.");
        std::string p(path);
        require(
            p.size() > 12 && p.rfind(".safetensors") == p.size() - 12,
            "the file name must end in .safetensors.");
        mx::save_safetensors(
            p,
            {{"dataset", index->dataset}, {"graph", index->graph}},
            {{"format", "mlxvs.cagra"},
             {"version", "1"},
             {"metric", metric_name(index->metric)}});
      },
      MLXVS_ERROR_IO);
}

int mlxvs_index_save_hnswlib(mlxvs_index index, const char* path) {
  return guarded(
      [&] {
        require(
            index != nullptr && path != nullptr,
            "index and path must not be NULL.");
        mlxvs::save_hnswlib(
            path, index->dataset, index->graph, to_metric(index->metric));
      },
      MLXVS_ERROR_IO);
}

int mlxvs_index_save_jvector(mlxvs_index index, const char* path) {
  return guarded(
      [&] {
        require(
            index != nullptr && path != nullptr,
            "index and path must not be NULL.");
        mlxvs::save_jvector(
            path, index->dataset, index->graph, to_metric(index->metric));
      },
      MLXVS_ERROR_IO);
}

int mlxvs_index_load(const char* path, mlxvs_index* out) {
  return guarded(
      [&] {
        require(
            out != nullptr && path != nullptr,
            "path and out must not be NULL.");
        *out = nullptr;
        auto [arrays, meta] = mx::load_safetensors(path);
        auto format = meta.find("format");
        if (format == meta.end() || format->second != "mlxvs.cagra" ||
            !arrays.count("dataset") || !arrays.count("graph")) {
          throw std::runtime_error(
              std::string("[mlxvs] ") + path + " is not a saved mlxvs index.");
        }
        const std::string metric = meta.count("metric") ? meta["metric"] : "";
        mlxvs_metric m = metric == "l2" ? MLXVS_L2
            : metric == "cosine"        ? MLXVS_COSINE
            : metric == "inner_product" ? MLXVS_INNER_PRODUCT
                                        : (mlxvs_metric)-1;
        if ((int)m < 0) {
          throw std::runtime_error(
              "[mlxvs] unknown metric in " + std::string(path));
        }
        auto dataset = arrays.at("dataset");
        auto graph = arrays.at("graph");
        mx::eval(dataset, graph);
        *out = new mlxvs_index_s{dataset, graph, m};
      },
      MLXVS_ERROR_IO);
}

int64_t mlxvs_index_size(mlxvs_index index) {
  return index == nullptr ? 0 : index->dataset.shape(0);
}

int32_t mlxvs_index_dim(mlxvs_index index) {
  return index == nullptr ? 0 : index->dataset.shape(1);
}

int32_t mlxvs_index_graph_degree(mlxvs_index index) {
  return index == nullptr ? 0 : index->graph.shape(1);
}

mlxvs_metric mlxvs_index_metric(mlxvs_index index) {
  return index == nullptr ? MLXVS_L2 : index->metric;
}

int mlxvs_index_graph(mlxvs_index index, int32_t* graph) {
  return guarded([&] {
    require(
        index != nullptr && graph != nullptr,
        "index and graph must not be NULL.");
    copy_out(
        mx::astype(index->graph, mx::int32),
        graph,
        (size_t)index->graph.size() * sizeof(int32_t));
  });
}

void mlxvs_index_free(mlxvs_index index) {
  delete index;
}

int mlxvs_brute_force_knn(
    const float* base,
    int64_t n,
    const float* queries,
    int64_t nq,
    int32_t d,
    int32_t k,
    mlxvs_metric metric,
    int32_t* ids,
    float* distances) {
  return guarded([&] {
    require(ids != nullptr, "ids must not be NULL.");
    auto b = host_rows(base, n, d, "base");
    auto q = host_rows(queries, nq, d, "queries");
    auto r = mlxvs::brute_force_knn(b, q, k, to_metric(metric));
    size_t count = (size_t)nq * k;
    copy_out(r.ids, ids, count * sizeof(int32_t));
    if (distances != nullptr) {
      copy_out(r.distances, distances, count * sizeof(float));
    }
  });
}

} // extern "C"
