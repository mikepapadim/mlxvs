// Copyright © 2026 mikepapadim

import MLX
import MLXFast

/// The best k of each row of scores `[nq, t]` (ids idOffset + column) merged
/// with the running lists: the radix-select top-k kernel of the C++ library.
func topkMerge(
    _ scores: MLXArray, ids: MLXArray, best: MLXArray, idOffset: Int, first: Bool, k: Int
) -> (MLXArray, MLXArray) {
    let nq = scores.dim(0)
    let t = scores.dim(1)
    let params = MLXArray([t, k, idOffset, first ? 1 : 0].map { Int32($0) })
    let out = KernelCache.topk(
        [scores, ids, best, params],
        grid: (nq * Kernels.topkThreads, 1, 1),
        threadGroup: (Kernels.topkThreads, 1, 1),
        outputShapes: [[nq, k], [nq, k]],
        outputDTypes: [.int32, .float32])
    return (out[0], out[1])
}

/// Exact k-NN of the queries among the base rows on the GPU: a matrix multiply
/// per tile of rows and a fused radix-select top-k, as cuVS's brute force.
///
/// - Parameters:
///   - base: rows `[n, d]`, any real dtype.
///   - queries: rows `[nq, d]`.
///   - k: neighbors per query, at most 256.
///   - tileRows: base rows scored per step.
/// - Returns: ids `int32 [nq, k]` and distances, nearest first: squared
///   distances for L2, similarities for inner product and cosine.
public func bruteForceKNN(
    base: MLXArray, queries queriesIn: MLXArray, k: Int, metric: Metric = .l2,
    tileRows: Int = 65536
) throws -> Neighbors {
    let base = base.asType(.float32)
    var queries = (queriesIn.ndim == 1 ? queriesIn.reshaped([1, -1]) : queriesIn).asType(.float32)
    guard base.ndim == 2, queries.dim(1) == base.dim(1) else {
        throw MLXVSError.invalidArgument("base must be [n, d] and queries [nq, d]")
    }
    guard k >= 1, k <= Kernels.topkMax else {
        throw MLXVSError.invalidArgument("k must be in [1, \(Kernels.topkMax)]")
    }
    let n = base.dim(0)
    let nq = queries.dim(0)
    func unit(_ x: MLXArray) -> MLXArray {
        x / maximum(sqrt((x * x).sum(axis: 1, keepDims: true)), MLXArray(Float(1e-30)))
    }
    if metric == .cosine { queries = unit(queries) }
    var ids = MLXArray.full([nq, k], values: MLXArray(Int32(-1)))
    var best = MLXArray.full([nq, k], values: MLXArray(-Float.infinity))
    for t0 in stride(from: 0, to: n, by: max(1, tileRows)) {
        let t1 = min(n, t0 + tileRows)
        var rows = base[t0 ..< t1]
        if metric == .cosine { rows = unit(rows) }
        var scores = matmul(queries, rows.T)
        if metric == .l2 {
            // 2 q.b - |b|^2: the order of -|q - b|^2, |q|^2 added back at the end
            scores = 2 * scores - (rows * rows).sum(axis: 1).expandedDimensions(axis: 0)
        }
        (ids, best) = topkMerge(scores, ids: ids, best: best, idOffset: t0, first: t0 == 0, k: k)
        eval(ids, best)
    }
    if metric == .l2 {
        let qq = (queries * queries).sum(axis: 1, keepDims: true)
        best = which(ids .>= 0, maximum(qq - best, MLXArray(Float(0))), MLXArray(Float.infinity))
    }
    return Neighbors(ids: ids, distances: best)
}
