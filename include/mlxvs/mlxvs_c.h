/* Copyright © 2026 mikepapadim */

/*
 * mlxvs C API: GPU k-nearest-neighbor search on Apple silicon, for languages
 * that call C (Java through Panama, Rust, Go, Swift, ...).
 *
 * Every function returns MLXVS_OK (0) or an error code; mlxvs_last_error()
 * then describes the failure (per thread). Arrays are row-major host memory:
 * the library copies inputs and writes outputs into buffers the caller owns.
 * Indexes saved here and by Python's mlxvs.CagraIndex.save are the same
 * .safetensors files, so either side can load the other's.
 */
#ifndef MLXVS_C_H
#define MLXVS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MLXVS_C_API __attribute__((visibility("default")))

/** The C API's version: MLXVS_C_API_VERSION = major * 10000 + minor. */
#define MLXVS_C_API_VERSION 100

typedef enum {
  MLXVS_OK = 0,
  /** An argument was out of range (mlxvs_last_error says which). */
  MLXVS_ERROR_INVALID_ARGUMENT = 1,
  /** A file could not be read or written, or is not an mlxvs index. */
  MLXVS_ERROR_IO = 2,
  /** Anything else, e.g. the GPU ran out of memory. */
  MLXVS_ERROR_INTERNAL = 3,
} mlxvs_status;

/** How two rows compare. */
typedef enum {
  /** Dot product, larger is closer. */
  MLXVS_INNER_PRODUCT = 0,
  /** Squared Euclidean distance, smaller is closer. */
  MLXVS_L2 = 1,
  /** Cosine similarity, larger is closer. */
  MLXVS_COSINE = 2,
} mlxvs_metric;

/** How the k-NN graph is pruned into a search graph. */
typedef enum {
  /** CAGRA's rank-based pruning plus reverse edges. */
  MLXVS_GRAPH_CAGRA = 0,
  /** Vamana's robust pruning with JVector's diversity rule. */
  MLXVS_GRAPH_VAMANA = 1,
} mlxvs_graph;

/** How a search lays out a query's work, as cuVS's search_algo. */
typedef enum {
  MLXVS_SEARCH_AUTO = 0,
  MLXVS_SEARCH_SINGLE_CTA = 1,
  MLXVS_SEARCH_MULTI_CTA = 2,
} mlxvs_search_algo;

/** Build options; mlxvs_build_params_init sets the defaults. */
typedef struct {
  /** Degree of the search graph (default 32). */
  int32_t graph_degree;
  /** Degree of the k-NN graph pruned (default 64, at most 128). */
  int32_t intermediate_graph_degree;
  /** MLXVS_GRAPH_CAGRA (default) or MLXVS_GRAPH_VAMANA. */
  int32_t graph;
  /** Vamana's alpha (default 1.2). */
  float alpha;
  /** Store the rows as float16 for search: half the memory (default 0). */
  int32_t float16;
} mlxvs_build_params;

/** Search options; mlxvs_search_params_init sets cuVS's defaults. */
typedef struct {
  /** Internal list length (default 64). */
  int32_t itopk_size;
  /** Nodes expanded per iteration (default 1). */
  int32_t search_width;
  /** A mlxvs_search_algo (default MLXVS_SEARCH_AUTO). */
  int32_t algo;
  /** 0: automatic, as cuVS. */
  int32_t max_iterations;
  int32_t min_iterations;
  /** Lanes per distance, 0: by the dimension. */
  int32_t team_size;
  /** Random draws per start slot (default 1). */
  int32_t num_random_samplings;
  uint64_t seed;
} mlxvs_search_params;

/** A dataset and its search graph, kept on the GPU. */
typedef struct mlxvs_index_s* mlxvs_index;

MLXVS_C_API void mlxvs_build_params_init(mlxvs_build_params* params);
MLXVS_C_API void mlxvs_search_params_init(mlxvs_search_params* params);

/** The last error of this thread, or "" after a success. */
MLXVS_C_API const char* mlxvs_last_error(void);

/** The library version, such as "0.5.0". */
MLXVS_C_API const char* mlxvs_version(void);

/**
 * Builds an index of n rows of d float32 (row-major). params may be NULL for
 * the defaults. Release it with mlxvs_index_free.
 */
MLXVS_C_API int mlxvs_index_build(
    const float* rows,
    int64_t n,
    int32_t d,
    mlxvs_metric metric,
    const mlxvs_build_params* params,
    mlxvs_index* out);

/**
 * An index of rows and a search graph built elsewhere: graph holds n rows of
 * graph_degree int32 neighbor ids (-1 entries are skipped).
 */
MLXVS_C_API int mlxvs_index_from_graph(
    const float* rows,
    int64_t n,
    int32_t d,
    const int32_t* graph,
    int32_t graph_degree,
    mlxvs_metric metric,
    mlxvs_index* out);

/**
 * The k nearest rows of nq queries (row-major float32 [nq, d]): ids int32
 * [nq, k] (-1 where fewer were found) and distances float32 [nq, k] (squared
 * distances for L2, similarities otherwise), nearest first. distances may be
 * NULL. params may be NULL for the defaults.
 */
MLXVS_C_API int mlxvs_index_search(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    int32_t* ids,
    float* distances);

/**
 * mlxvs_index_search returning only rows the filter allows: filter is a
 * bitset of (n + 31) / 32 uint32 words, bit i % 32 of word i / 32 set when row
 * i may be returned (cuVS's bitset filter). Rejected rows are still traversed.
 */
MLXVS_C_API int mlxvs_index_search_filtered(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const uint32_t* filter,
    int32_t* ids,
    float* distances);

/**
 * mlxvs_index_search starting from given nodes, as cuVS's seed nodes: seeds
 * holds nq rows of num_seeds int32 ids (out-of-range ids: random nodes).
 * filter may be NULL (no filter) or a bitset as mlxvs_index_search_filtered's.
 */
MLXVS_C_API int mlxvs_index_search_seeded(
    mlxvs_index index,
    const float* queries,
    int64_t nq,
    int32_t k,
    const mlxvs_search_params* params,
    const uint32_t* filter,
    const int32_t* seeds,
    int32_t num_seeds,
    int32_t* ids,
    float* distances);

/** Saves the index as .safetensors (as Python's CagraIndex.save). */
MLXVS_C_API int mlxvs_index_save(mlxvs_index index, const char* path);

/** Loads an index saved by mlxvs_index_save or CagraIndex.save. */
MLXVS_C_API int mlxvs_index_load(const char* path, mlxvs_index* out);

MLXVS_C_API int64_t mlxvs_index_size(mlxvs_index index);
MLXVS_C_API int32_t mlxvs_index_dim(mlxvs_index index);
MLXVS_C_API int32_t mlxvs_index_graph_degree(mlxvs_index index);
MLXVS_C_API mlxvs_metric mlxvs_index_metric(mlxvs_index index);

/** Copies the search graph into graph (n * graph_degree int32). */
MLXVS_C_API int mlxvs_index_graph(mlxvs_index index, int32_t* graph);

MLXVS_C_API void mlxvs_index_free(mlxvs_index index);

/**
 * Exact k-nearest neighbors of nq queries among n base rows (float32,
 * row-major), as mlxvs_index_search returns them. 1 <= k <= 256.
 */
MLXVS_C_API int mlxvs_brute_force_knn(
    const float* base,
    int64_t n,
    const float* queries,
    int64_t nq,
    int32_t d,
    int32_t k,
    mlxvs_metric metric,
    int32_t* ids,
    float* distances);

#ifdef __cplusplus
}
#endif

#endif /* MLXVS_C_H */
