// Copyright © 2026 mikepapadim

import Foundation
import MLX
import XCTest

@testable import MLXVS

final class MLXVSTests: XCTestCase {

    /// Clustered rows, as embeddings are.
    func rows(_ n: Int, _ d: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let parts = MLXRandom.split(key: key, into: 3)
        let centers = MLXRandom.normal([20, d], key: parts[0])
        let which = MLXRandom.randInt(0 ..< 20, [n], key: parts[1])
        return centers[which] + 0.3 * MLXRandom.normal([n, d], key: parts[2])
    }

    /// The exact k nearest by brute force on the CPU's terms: full sort of the
    /// distance matrix.
    func exact(_ base: MLXArray, _ queries: MLXArray, k: Int, metric: Metric) -> [[Int32]] {
        var scores: MLXArray
        switch metric {
        case .l2:
            let diff = queries.expandedDimensions(axis: 1) - base.expandedDimensions(axis: 0)
            scores = -(diff * diff).sum(axis: 2)
        case .innerProduct:
            scores = matmul(queries, base.T)
        case .cosine:
            let qn = queries / sqrt((queries * queries).sum(axis: 1, keepDims: true))
            let bn = base / sqrt((base * base).sum(axis: 1, keepDims: true))
            scores = matmul(qn, bn.T)
        }
        let order = argSort(-scores, axis: 1)[0..., 0 ..< k].asType(.int32)
        return order.asArray(Int32.self).chunked(k)
    }

    func recall(_ got: MLXArray, _ truth: [[Int32]]) -> Double {
        let k = truth[0].count
        let rows = got.asArray(Int32.self).chunked(k)
        var hits = 0
        for (a, b) in zip(rows, truth) { hits += Set(a).intersection(Set(b)).count }
        return Double(hits) / Double(truth.count * k)
    }

    func testBruteForceIsExact() throws {
        let base = rows(3000, 24, seed: 1)
        let queries = rows(50, 24, seed: 2)
        for metric in [Metric.l2, .innerProduct, .cosine] {
            let result = try bruteForceKNN(base: base, queries: queries, k: 10, metric: metric, tileRows: 1000)
            XCTAssertEqual(recall(result.ids, exact(base, queries, k: 10, metric: metric)), 1.0, "\(metric)")
        }
        // squared distances, ascending
        let l2 = try bruteForceKNN(base: base, queries: queries, k: 5)
        let d = l2.distances.asArray(Float.self).chunked(5)
        XCTAssertTrue(d.allSatisfy { zip($0, $0.dropFirst()).allSatisfy { $0 <= $1 + 1e-4 } })
    }

    func testGraphSearchSingleAndMultiCTA() throws {
        let base = rows(4000, 32, seed: 3)
        let queries = rows(64, 32, seed: 4)
        // an exact 32-NN graph is a good search graph
        let graph = try bruteForceKNN(base: base, queries: base, k: 32).ids
        let truth = exact(base, queries, k: 10, metric: .l2)
        for algo in [SearchAlgo.singleCTA, .multiCTA] {
            var options = SearchOptions()
            options.algo = algo
            options.itopkSize = 128
            let result = try graphSearch(base: base, graph: graph, queries: queries, k: 10, options: options)
            // the Python package gets 0.922 and 0.919 on these rows and graph
            XCTAssertGreaterThan(recall(result.ids, truth), 0.9, "\(algo)")
        }
    }

    func testFilter() throws {
        let base = rows(2000, 16, seed: 5)
        let queries = rows(20, 16, seed: 6)
        let graph = try bruteForceKNN(base: base, queries: base, k: 32).ids
        let allowed = MLXArray(0 ..< 2000) % 3 .== 0
        var options = SearchOptions()
        options.filter = allowed
        options.itopkSize = 128
        let result = try graphSearch(base: base, graph: graph, queries: queries, k: 10, options: options)
        XCTAssertTrue(result.ids.asArray(Int32.self).allSatisfy { $0 % 3 == 0 })
    }

    func testLoadsPythonIndex() throws {
        let url = Bundle.module.url(forResource: "cosine", withExtension: "safetensors", subdirectory: "Resources")!
        let index = try CagraIndex.load(url: url)
        XCTAssertEqual(index.metric, .cosine)
        XCTAssertEqual(index.count, 1500)
        let queries = index.dataset[0 ..< 40] + 0.01
        var options = SearchOptions()
        options.itopkSize = 128
        let result = try index.search(queries, k: 5, options: options)
        XCTAssertGreaterThan(recall(result.ids, exact(index.dataset, queries, k: 5, metric: .cosine)), 0.95)
        // cosine similarities, in [-1, 1], largest first
        let sims = result.distances.asArray(Float.self)
        XCTAssertTrue(sims.allSatisfy { $0 <= 1.0001 && $0 >= -1.0001 })
    }
}

extension Array {
    func chunked(_ size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { Array(self[$0 ..< Swift.min($0 + size, count)]) }
    }
}
