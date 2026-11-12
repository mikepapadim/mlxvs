/* Copyright © 2026 mikepapadim */

/*
 * The C API: build, search, save and load an index, from C, against exact
 * neighbors; errors reported through mlxvs_last_error.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlxvs/mlxvs_c.h"

static int failures = 0;

static void check(int ok, const char* what) {
  if (!ok) {
    ++failures;
    printf("FAIL %s (%s)\n", what, mlxvs_last_error());
  }
}

static float unif(void) {
  return (float)(rand() / (RAND_MAX + 1.0)) - 0.5f;
}

/* rows near a 6-dimensional subspace, as embeddings */
static float* make_rows(int n, int d, unsigned seed) {
  float* basis = malloc(sizeof(float) * 6 * d);
  float* x = malloc(sizeof(float) * (size_t)n * d);
  srand(seed);
  for (int i = 0; i < 6 * d; ++i)
    basis[i] = unif();
  for (int i = 0; i < n; ++i) {
    float z[6];
    for (int t = 0; t < 6; ++t)
      z[t] = unif();
    for (int j = 0; j < d; ++j) {
      float acc = 0.01f * unif();
      for (int t = 0; t < 6; ++t)
        acc += z[t] * basis[t * d + j];
      x[(size_t)i * d + j] = acc;
    }
  }
  free(basis);
  return x;
}

/* recall@k: returned ids whose distance reaches the exact k-th */
static double
recall(const float* got, const float* exact, int nq, int k, mlxvs_metric m) {
  long hits = 0;
  for (int i = 0; i < nq; ++i) {
    float kth = exact[i * k + k - 1];
    for (int j = 0; j < k; ++j) {
      float g = got[i * k + j];
      hits += m == MLXVS_L2 ? g <= kth + 1e-5f * fmaxf(1.0f, kth)
                            : g >= kth - 1e-5f * fmaxf(1.0f, fabsf(kth));
    }
  }
  return hits / (double)(nq * k);
}

static void index_case(mlxvs_metric metric, int graph, const char* name) {
  const int n = 8000, nq = 200, d = 32, k = 10;
  float* x = make_rows(n + nq, d, 7);
  const float* q = x + (size_t)n * d;
  int32_t* ids = malloc(sizeof(int32_t) * nq * k);
  int32_t* ids2 = malloc(sizeof(int32_t) * nq * k);
  float* dist = malloc(sizeof(float) * nq * k);
  float* exact = malloc(sizeof(float) * nq * k);
  int32_t* eids = malloc(sizeof(int32_t) * nq * k);

  mlxvs_build_params bp;
  mlxvs_build_params_init(&bp);
  bp.graph_degree = 16;
  bp.intermediate_graph_degree = 32;
  bp.graph = graph;
  mlxvs_index index = NULL;
  check(mlxvs_index_build(x, n, d, metric, &bp, &index) == MLXVS_OK, "build");
  check(
      mlxvs_index_size(index) == n && mlxvs_index_dim(index) == d &&
          mlxvs_index_graph_degree(index) == 16 &&
          mlxvs_index_metric(index) == metric,
      "properties");

  check(
      mlxvs_brute_force_knn(x, n, q, nq, d, k, metric, eids, exact) == MLXVS_OK,
      "brute force");
  mlxvs_search_params sp;
  mlxvs_search_params_init(&sp);
  check(sp.itopk_size == 64 && sp.search_width == 1, "search defaults");
  double r_single = 0, r_multi = 0;
  sp.algo = MLXVS_SEARCH_SINGLE_CTA;
  check(
      mlxvs_index_search(index, q, nq, k, &sp, ids, dist) == MLXVS_OK,
      "search");
  r_single = recall(dist, exact, nq, k, metric);
  sp.algo = MLXVS_SEARCH_MULTI_CTA;
  check(
      mlxvs_index_search(index, q, nq, k, &sp, ids, dist) == MLXVS_OK,
      "search multi-CTA");
  r_multi = recall(dist, exact, nq, k, metric);

  /* save, load, search again: the same ids */
  char path[] = "/tmp/mlxvs_test_c_api.safetensors";
  check(mlxvs_index_save(index, path) == MLXVS_OK, "save");
  mlxvs_index loaded = NULL;
  check(mlxvs_index_load(path, &loaded) == MLXVS_OK, "load");
  check(
      mlxvs_index_search(loaded, q, nq, k, &sp, ids2, NULL) == MLXVS_OK,
      "search loaded");
  check(memcmp(ids, ids2, sizeof(int32_t) * nq * k) == 0, "loaded ids equal");
  remove(path);
  check(
      mlxvs_index_save_hnswlib(index, "/tmp/mlxvs_test_c_api.hnsw") == MLXVS_OK,
      "save hnswlib");
  check(
      mlxvs_index_save_jvector(index, "/tmp/mlxvs_test_c_api.jvector") ==
          MLXVS_OK,
      "save jvector");
  FILE* jv = fopen("/tmp/mlxvs_test_c_api.jvector", "rb");
  unsigned char magic[4] = {0};
  check(
      jv != NULL && fread(magic, 1, 4, jv) == 4 && magic[0] == 0xFF &&
          magic[1] == 0xFF && magic[2] == 0x0D && magic[3] == 0x61,
      "jvector magic");
  if (jv)
    fclose(jv);
  remove("/tmp/mlxvs_test_c_api.hnsw");
  remove("/tmp/mlxvs_test_c_api.jvector");

  /* an index from the same graph, copied out */
  int32_t* graph_ids = malloc(sizeof(int32_t) * n * 16);
  check(mlxvs_index_graph(index, graph_ids) == MLXVS_OK, "graph");
  mlxvs_index again = NULL;
  check(
      mlxvs_index_from_graph(x, n, d, graph_ids, 16, metric, &again) ==
          MLXVS_OK,
      "from graph");
  check(
      mlxvs_index_search(again, q, nq, k, &sp, ids2, NULL) == MLXVS_OK &&
          memcmp(ids, ids2, sizeof(int32_t) * nq * k) == 0,
      "from-graph ids equal");

  /* a filter: even rows only */
  uint32_t* words = calloc((n + 31) / 32, sizeof(uint32_t));
  for (int i = 0; i < n; i += 2)
    words[i / 32] |= 1u << (i % 32);
  check(
      mlxvs_index_search_filtered(index, q, nq, k, &sp, words, ids2, NULL) ==
          MLXVS_OK,
      "filtered search");
  int even = 1;
  for (int i = 0; i < nq * k; ++i)
    even &= ids2[i] >= 0 && ids2[i] % 2 == 0;
  check(even, "filtered ids allowed");
  free(words);

  /* seeds: each query's exact nearest row, one iteration */
  int32_t* seeds = malloc(sizeof(int32_t) * nq);
  for (int i = 0; i < nq; ++i)
    seeds[i] = eids[i * k];
  mlxvs_search_params one = sp;
  one.max_iterations = 1;
  check(
      mlxvs_index_search_seeded(
          index, q, nq, k, &one, NULL, seeds, 1, ids2, NULL) == MLXVS_OK,
      "seeded search");
  int seeded = 1;
  for (int i = 0; i < nq; ++i)
    seeded &= ids2[i * k] == seeds[i];
  check(seeded, "seeds come back first");
  free(seeds);

  int ok = r_single >= 0.95 && r_multi >= 0.95;
  printf(
      "%-6s %-22s recall@10 %.4f single-CTA, %.4f multi-CTA\n",
      ok ? "PASS" : "FAIL",
      name,
      r_single,
      r_multi);
  check(ok, "recall");
  mlxvs_index_free(index);
  mlxvs_index_free(loaded);
  mlxvs_index_free(again);
  free(graph_ids);
  free(x);
  free(ids);
  free(ids2);
  free(dist);
  free(exact);
  free(eids);
}

/* uint8 rows stored as they are, and a filter per query */
static void bytes_and_bitmap_case(void) {
  const int n = 6000, nq = 100, d = 24, k = 10;
  float* x = make_rows(n + nq, d, 11);
  uint8_t* bytes = malloc((size_t)n * d);
  float* q = malloc(sizeof(float) * nq * d);
  for (int i = 0; i < n * d; ++i)
    bytes[i] = (uint8_t)lroundf(fminf(fmaxf(128.0f + 100.0f * x[i], 0), 255));
  for (int i = 0; i < nq * d; ++i)
    q[i] = roundf(fminf(fmaxf(128.0f + 100.0f * x[n * d + i], 0), 255));
  mlxvs_build_params bp;
  mlxvs_build_params_init(&bp);
  bp.graph_degree = 16;
  bp.intermediate_graph_degree = 32;
  mlxvs_index index = NULL;
  check(
      mlxvs_index_build_typed(
          bytes, MLXVS_UINT8, n, d, MLXVS_L2, &bp, &index) == MLXVS_OK,
      "build uint8");
  int32_t* ids = malloc(sizeof(int32_t) * nq * k);
  int words = (n + 31) / 32;
  uint32_t* bitmap = calloc((size_t)nq * words, sizeof(uint32_t));
  for (int i = 0; i < nq; ++i)
    for (int r = 0; r < n; ++r)
      if ((r + i) % 5 != 0)
        bitmap[(size_t)i * words + r / 32] |= 1u << (r % 32);
  check(
      mlxvs_index_search_bitmap(index, q, nq, k, NULL, bitmap, ids, NULL) ==
          MLXVS_OK,
      "bitmap search");
  int ok = 1;
  for (int i = 0; i < nq; ++i)
    for (int j = 0; j < k; ++j)
      ok &= ids[i * k + j] >= 0 && (ids[i * k + j] + i) % 5 != 0;
  printf("%-6s uint8 rows, filter per query\n", ok ? "PASS" : "FAIL");
  check(ok, "uint8 and bitmap");
  mlxvs_index_free(index);
  free(x);
  free(bytes);
  free(q);
  free(ids);
  free(bitmap);
}

int main(void) {
  printf("mlxvs %s, C API %d\n", mlxvs_version(), MLXVS_C_API_VERSION);
  index_case(MLXVS_L2, MLXVS_GRAPH_CAGRA, "l2 cagra");
  index_case(MLXVS_COSINE, MLXVS_GRAPH_CAGRA, "cosine cagra");
  index_case(MLXVS_L2, MLXVS_GRAPH_VAMANA, "l2 vamana");
  bytes_and_bitmap_case();

  /* errors */
  float rows[8] = {0};
  mlxvs_index index = NULL;
  int st = mlxvs_index_build(rows, 2, 4, (mlxvs_metric)7, NULL, &index);
  check(
      st == MLXVS_ERROR_INVALID_ARGUMENT && index == NULL &&
          strlen(mlxvs_last_error()) > 0,
      "bad metric rejected");
  st = mlxvs_index_load("/nonexistent/x.safetensors", &index);
  check(st == MLXVS_ERROR_IO && index == NULL, "missing file reported");
  st = mlxvs_index_search(NULL, rows, 1, 1, NULL, NULL, NULL);
  check(st == MLXVS_ERROR_INVALID_ARGUMENT, "NULL index rejected");
  printf(failures == 0 ? "all tests passed\n" : "%d FAILED\n", failures);
  return failures == 0 ? 0 : 1;
}
