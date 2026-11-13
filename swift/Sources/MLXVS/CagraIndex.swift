// Copyright © 2026 mikepapadim

import Foundation
import MLX

/// A CAGRA search index: the dataset and its graph, searched on the GPU.
///
/// Build and save one with the Python package (`pip install mlxvs`), then
/// search it from an app:
///
/// ```python
/// index = mlxvs.CagraIndex.build(vectors, graph_degree=32, metric="cosine")
/// index.save("docs.safetensors")
/// ```
///
/// ```swift
/// let index = try CagraIndex.load(url: Bundle.main.url(forResource: "docs", withExtension: "safetensors")!)
/// let result = try index.search(queries, k: 10)
/// ```
public struct CagraIndex {
    /// The stored rows `[n, d]` (unit rows for cosine; float32, float16, int8
    /// or uint8).
    public let dataset: MLXArray
    /// The neighbor lists `[n, degree]`.
    public let graph: MLXArray
    public let metric: Metric

    public init(dataset: MLXArray, graph: MLXArray, metric: Metric) {
        self.dataset = dataset
        self.graph = graph
        self.metric = metric
    }

    /// Loads an index saved by the Python package's `CagraIndex.save`.
    public static func load(url: URL) throws -> CagraIndex {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard metadata["format"] == "mlxvs.cagra", let graph = arrays["graph"] else {
            throw MLXVSError.invalidArgument("\(url.path) is not a saved mlxvs CagraIndex")
        }
        guard let dataset = arrays["dataset"] else {
            throw MLXVSError.invalidArgument(
                "\(url.path) holds a compressed (VPQ) index, which the Swift package does not read yet")
        }
        guard let name = metadata["metric"], let metric = Metric(rawValue: name) else {
            throw MLXVSError.invalidArgument("unknown metric in \(url.path)")
        }
        return CagraIndex(dataset: dataset, graph: graph, metric: metric)
    }

    /// Builds an index of rows with a graph computed elsewhere (for example
    /// by the Python package's `cagra_build`): unit rows are stored for cosine.
    public static func fromGraph(rows: MLXArray, graph: MLXArray, metric: Metric) -> CagraIndex {
        var data = rows.asType(.float32)
        if metric == .cosine {
            data = data / maximum(sqrt((data * data).sum(axis: 1, keepDims: true)), MLXArray(Float(1e-30)))
        }
        return CagraIndex(dataset: data, graph: graph.asType(.int32), metric: metric)
    }

    public var count: Int { dataset.dim(0) }
    public var dimension: Int { dataset.dim(1) }

    /// The k nearest rows of every query `[nq, d]`, as the Python package's
    /// `CagraIndex.search`: squared distances for L2, similarities for inner
    /// product and cosine.
    public func search(_ queries: MLXArray, k: Int, options: SearchOptions = SearchOptions())
        throws -> Neighbors
    {
        var q = (queries.ndim == 1 ? queries.reshaped([1, -1]) : queries).asType(.float32)
        var metric = self.metric
        let byteRows = dataset.dtype == .int8 || dataset.dtype == .uint8
        if metric == .cosine && !byteRows {
            // the rows are stored normalized: the dot product of unit rows is
            // the cosine similarity
            q = q / maximum(sqrt((q * q).sum(axis: 1, keepDims: true)), MLXArray(Float(1e-30)))
            metric = .innerProduct
        }
        return try graphSearch(base: dataset, graph: graph, queries: q, k: k, metric: metric, options: options)
    }
}
