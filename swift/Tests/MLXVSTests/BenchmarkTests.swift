// Copyright © 2026 mikepapadim

import Foundation
import MLX
import XCTest

@testable import MLXVS

/// Search speed on a saved index: runs when MLXVS_BENCH_INDEX (a CagraIndex
/// .safetensors) and MLXVS_BENCH_QUERIES (an .fvecs file) are set.
final class BenchmarkTests: XCTestCase {
    func testSearchSpeed() throws {
        let env = ProcessInfo.processInfo.environment
        guard let indexPath = env["MLXVS_BENCH_INDEX"], let queryPath = env["MLXVS_BENCH_QUERIES"] else {
            throw XCTSkip("set MLXVS_BENCH_INDEX and MLXVS_BENCH_QUERIES")
        }
        let index = try CagraIndex.load(url: URL(fileURLWithPath: indexPath))
        let raw = try Data(contentsOf: URL(fileURLWithPath: queryPath))
        let d = Int(raw.withUnsafeBytes { $0.load(as: Int32.self) })
        let count = raw.count / (4 * (d + 1))
        var values = [Float](repeating: 0, count: count * d)
        raw.withUnsafeBytes { buffer in
            for i in 0 ..< count {
                for j in 0 ..< d {
                    values[i * d + j] = buffer.load(fromByteOffset: 4 * (i * (d + 1) + 1 + j), as: Float.self)
                }
            }
        }
        let queries = MLXArray(values, [count, d])
        for itopk in [32, 64, 128] {
            var options = SearchOptions()
            options.itopkSize = itopk
            options.algo = .singleCTA
            var r = try index.search(queries, k: 10, options: options)
            eval(r.ids, r.distances)
            let start = Date()
            r = try index.search(queries, k: 10, options: options)
            eval(r.ids, r.distances)
            let seconds = Date().timeIntervalSince(start)
            print(String(format: "swift itopk %3d: %8.0f queries/s", itopk, Double(count) / seconds))
        }
    }
}
