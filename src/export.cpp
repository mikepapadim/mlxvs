// Copyright © 2026 mikepapadim

// Index files other libraries read: hnswlib's binary format and JVector's
// OnDiskGraphIndex.

#include "mlxvs/mlxvs.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlxvs {

namespace {

struct Rows {
  mx::array data; // float32 [n, d], contiguous, evaluated
  mx::array graph; // int32 [n, degree], contiguous, evaluated
  int n, d, degree;
};

Rows host_rows(
    const mx::array& dataset,
    const mx::array& graph,
    const char* op) {
  if (dataset.ndim() != 2 || graph.ndim() != 2 ||
      graph.shape(0) != dataset.shape(0) ||
      !mx::issubdtype(graph.dtype(), mx::integer)) {
    throw std::invalid_argument(
        std::string("[") + op +
        "] dataset must be [n, d] and graph integer ids [n, degree].");
  }
  auto data = mx::contiguous(mx::astype(dataset, mx::float32));
  auto g = mx::contiguous(mx::astype(graph, mx::int32));
  mx::eval(data, g);
  return {data, g, dataset.shape(0), dataset.shape(1), graph.shape(1)};
}

// The row nearest the mean (largest dot product with it for the similarity
// metrics): the entry point of exported graphs.
int medoid(const mx::array& dataset, Metric metric) {
  auto x = mx::astype(dataset, mx::float32);
  auto mean = mx::mean(x, 0, /*keepdims=*/true);
  auto m = metric == Metric::Cosine ? Metric::InnerProduct : metric;
  auto ids = brute_force_knn(x, mean, 1, m).ids;
  mx::eval(ids);
  return ids.data<int32_t>()[0];
}

std::ofstream open_out(const std::string& path, const char* op) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error(
        std::string("[") + op + "] cannot write " + path + ".");
  }
  return out;
}

template <typename T>
void put_le(std::vector<char>& buf, T v) {
  char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T)); // Apple silicon is little-endian
  buf.insert(buf.end(), b, b + sizeof(T));
}

void put_be32(std::vector<char>& buf, uint32_t v) {
  put_le(buf, __builtin_bswap32(v));
}

void put_be64(std::vector<char>& buf, uint64_t v) {
  put_le(buf, __builtin_bswap64(v));
}

} // namespace

void save_hnswlib(
    const std::string& path,
    const mx::array& dataset,
    const mx::array& graph,
    Metric metric) {
  const char* op = "mlxvs::save_hnswlib";
  auto r = host_rows(dataset, graph, op);
  uint64_t n = r.n, d = r.d;
  uint64_t m = (r.degree + 1) / 2;
  uint64_t max_m0 = 2 * m;
  uint64_t size_links0 = 4 + 4 * max_m0;
  uint64_t record = size_links0 + 4 * d + 8;
  int entry = medoid(dataset, metric);

  // hnswlib's header (packed, little-endian): offsetLevel0, max_elements,
  // cur_element_count, size_data_per_element, label_offset, offsetData,
  // maxlevel, enterpoint_node, maxM, maxM0, M, mult, ef_construction
  std::vector<char> buf;
  put_le<uint64_t>(buf, 0);
  put_le<uint64_t>(buf, n);
  put_le<uint64_t>(buf, n);
  put_le<uint64_t>(buf, record);
  put_le<uint64_t>(buf, size_links0 + 4 * d);
  put_le<uint64_t>(buf, size_links0);
  put_le<int32_t>(buf, 0);
  put_le<uint32_t>(buf, (uint32_t)entry);
  put_le<uint64_t>(buf, m);
  put_le<uint64_t>(buf, max_m0);
  put_le<uint64_t>(buf, m);
  put_le<double>(buf, m > 1 ? 1.0 / std::log((double)m) : 1.0);
  put_le<uint64_t>(buf, 200); // ef_construction, unused by search
  auto out = open_out(path, op);
  out.write(buf.data(), buf.size());

  // each element: link count (hnswlib reads its low 16 bits; a zero third
  // byte means "not deleted"), max_m0 links (valid ids first, 0 filled), the
  // row, the label (the row id)
  const float* rows = r.data.data<float>();
  const int32_t* g = r.graph.data<int32_t>();
  for (uint64_t i = 0; i < n; ++i) {
    buf.clear();
    std::vector<uint32_t> links(max_m0, 0);
    uint32_t count = 0;
    for (int t = 0; t < r.degree; ++t) {
      int v = g[i * r.degree + t];
      if (v >= 0) {
        links[count++] = (uint32_t)v;
      }
    }
    put_le<uint32_t>(buf, count);
    for (auto l : links) {
      put_le<uint32_t>(buf, l);
    }
    for (uint64_t j = 0; j < d; ++j) {
      put_le<float>(buf, rows[i * d + j]);
    }
    put_le<uint64_t>(buf, i);
    out.write(buf.data(), buf.size());
  }
  // no element has upper layers: every link-list size is 0
  std::vector<char> zeros(4 * n, 0);
  out.write(zeros.data(), zeros.size());
  if (!out) {
    throw std::runtime_error(
        std::string("[") + op + "] writing " + path + " failed.");
  }
}

void save_jvector(
    const std::string& path,
    const mx::array& dataset,
    const mx::array& graph,
    Metric metric) {
  const char* op = "mlxvs::save_jvector";
  auto r = host_rows(dataset, graph, op);
  int entry = medoid(dataset, metric);
  const int max_layers = 32; // JVector's V4_MAX_LAYERS

  // big-endian, as Java's DataOutput: magic, version (6, JVector 4), layer 0
  // size, dimension, entry node, layer 0 degree, id upper bound, layer count,
  // (size, degree) of 32 layers; then the features: their count and ordinals
  // (INLINE_VECTORS = 0, which has no header of its own)
  std::vector<char> header;
  for (uint32_t v :
       {0xFFFF0D61u,
        6u,
        (uint32_t)r.n,
        (uint32_t)r.d,
        (uint32_t)entry,
        (uint32_t)r.degree,
        (uint32_t)r.n,
        1u}) {
    put_be32(header, v);
  }
  put_be32(header, (uint32_t)r.n);
  put_be32(header, (uint32_t)r.degree);
  for (int i = 1; i < max_layers; ++i) {
    put_be32(header, 0);
    put_be32(header, 0);
  }
  put_be32(header, 1);
  put_be32(header, 0);

  auto out = open_out(path, op);
  out.write(header.data(), header.size());
  // one record per node: its ordinal, the row, the neighbor count, the
  // neighbors (valid ids first, -1 filled)
  const float* rows = r.data.data<float>();
  const int32_t* g = r.graph.data<int32_t>();
  std::vector<char> buf;
  for (int i = 0; i < r.n; ++i) {
    buf.clear();
    put_be32(buf, (uint32_t)i);
    for (int j = 0; j < r.d; ++j) {
      uint32_t bits;
      std::memcpy(&bits, &rows[(size_t)i * r.d + j], 4);
      put_be32(buf, bits);
    }
    std::vector<int32_t> nbrs;
    for (int t = 0; t < r.degree; ++t) {
      int v = g[(size_t)i * r.degree + t];
      if (v >= 0) {
        nbrs.push_back(v);
      }
    }
    put_be32(buf, (uint32_t)nbrs.size());
    for (int t = 0; t < r.degree; ++t) {
      put_be32(buf, t < (int)nbrs.size() ? (uint32_t)nbrs[t] : 0xFFFFFFFFu);
    }
    out.write(buf.data(), buf.size());
  }
  // the footer: the header again, its offset, the footer magic
  uint64_t footer_at = header.size() +
      (uint64_t)r.n * (4 + 4 * (uint64_t)r.d + 4 + 4 * (uint64_t)r.degree);
  buf.clear();
  put_be64(buf, footer_at);
  put_be32(buf, 0x4A564244u);
  out.write(header.data(), header.size());
  out.write(buf.data(), buf.size());
  if (!out) {
    throw std::runtime_error(
        std::string("[") + op + "] writing " + path + " failed.");
  }
}

} // namespace mlxvs
