// Copyright © 2026 mikepapadim

import MLX
import MLXFast

enum BuildKernels {
    static let prune = MLXFast.metalKernel(
        name: "mlxvs_cagra_prune", inputNames: ["knn", "params"], outputNames: ["pruned"],
        source: Kernels.pruneSource, header: Kernels.searchHeader)
    static let revCount = MLXFast.metalKernel(
        name: "mlxvs_cagra_rev_count", inputNames: ["pruned", "params"], outputNames: ["hist"],
        source: Kernels.revCountSource, header: "", ensureRowContiguous: true, atomicOutputs: true)
    static let revOffset = MLXFast.metalKernel(
        name: "mlxvs_cagra_rev_offset", inputNames: ["hist", "params"], outputNames: ["offset"],
        source: Kernels.revOffsetSource)
    static let revFill = MLXFast.metalKernel(
        name: "mlxvs_cagra_rev_fill", inputNames: ["pruned", "offset", "params"],
        outputNames: ["placed", "rev"], source: Kernels.revFillSource, header: "",
        ensureRowContiguous: true, atomicOutputs: true)
    static let merge = MLXFast.metalKernel(
        name: "mlxvs_cagra_merge", inputNames: ["pruned", "rev", "offset", "params"],
        outputNames: ["graph"], source: Kernels.mergeSource)
}

/// CAGRA's graph optimization of a k-NN graph `[n, k]` (ids of each row's
/// nearest rows, nearest first) into a search graph `[n, graphDegree]`:
/// rank-based pruning (fewest two-hop detours), then each node's list
/// completed with its incoming edges of lowest rank. The same kernels as the
/// C++ library's `cagra_optimize`.
public func cagraOptimize(knnGraph: MLXArray, graphDegree: Int) throws -> MLXArray {
    guard knnGraph.ndim == 2 else {
        throw MLXVSError.invalidArgument("knnGraph must be integer ids [n, k]")
    }
    let n = knnGraph.dim(0)
    let k = knnGraph.dim(1)
    guard k >= 1, k <= Kernels.pruneDegreeMax else {
        throw MLXVSError.invalidArgument("the k-NN graph's degree must be in [1, \(Kernels.pruneDegreeMax)]")
    }
    guard graphDegree >= 1, graphDegree <= k else {
        throw MLXVSError.invalidArgument("graphDegree must be in [1, k-NN graph degree]")
    }
    if n == 0 { return MLXArray.zeros([0, graphDegree], type: Int32.self) }
    let knn = knnGraph.asType(.int32)
    let d = graphDegree
    let rcap = d
    let pruned = BuildKernels.prune(
        [knn, MLXArray([Int32(n), Int32(d)])],
        template: [("K", k), ("HSIZE", pow2AtLeast(2 * k))],
        grid: (n * Kernels.pruneThreads, 1, 1), threadGroup: (Kernels.pruneThreads, 1, 1),
        outputShapes: [[n, d]], outputDTypes: [.int32], initValue: -1)[0]
    // reverse edges, each node's rcap incoming edges of lowest rank
    let params = MLXArray([Int32(n)])
    let hist = BuildKernels.revCount(
        [pruned, params], template: [("D", d)], grid: (n * d, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[n, d]], outputDTypes: [.int32], initValue: 0)[0]
    let offset = BuildKernels.revOffset(
        [hist, params], template: [("D", d)], grid: (n, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[n, d + 1]], outputDTypes: [.int32])[0]
    let rev = BuildKernels.revFill(
        [pruned, offset, params], template: [("D", d), ("RCAP", rcap)], grid: (n * d, 1, 1),
        threadGroup: (256, 1, 1), outputShapes: [[n, d], [n, 2 * rcap]],
        outputDTypes: [.int32, .int32], initValue: 0)[1]
    return BuildKernels.merge(
        [pruned, rev, offset, params], template: [("D", d), ("RCAP", rcap)], grid: (n, 1, 1),
        threadGroup: (256, 1, 1), outputShapes: [[n, d]], outputDTypes: [.int32], initValue: -1)[0]
}

extension CagraIndex {
    /// Builds a CAGRA index of rows `[n, d]` on the GPU: the exact
    /// `intermediateGraphDegree`-NN graph of the rows (``bruteForceKNN``, so
    /// best for up to a few hundred thousand rows; build larger indexes with
    /// the Python package's NN-Descent), pruned by ``cagraOptimize``.
    public static func build(
        rows: MLXArray, graphDegree: Int = 32, metric: Metric = .l2,
        intermediateGraphDegree: Int = 64
    ) throws -> CagraIndex {
        var data = rows.asType(.float32)
        if metric == .cosine {
            data = data / maximum(sqrt((data * data).sum(axis: 1, keepDims: true)), MLXArray(Float(1e-30)))
        }
        let n = data.dim(0)
        let k = min(intermediateGraphDegree, max(1, n - 1))
        guard graphDegree <= k else {
            throw MLXVSError.invalidArgument("graphDegree must be at most intermediateGraphDegree and n - 1")
        }
        // each row's k + 1 nearest, itself (normally first) left out
        let knnMetric: Metric = metric == .l2 ? .l2 : .innerProduct
        let nearest = try bruteForceKNN(base: data, queries: data, k: min(k + 1, n), metric: knnMetric).ids
        let selfFirst = nearest[0..., 0] .== MLXArray(0 ..< n).asType(.int32)
        let shifted = which(
            selfFirst.expandedDimensions(axis: 1), nearest[0..., 1...], nearest[0..., 0 ..< k])
        let graph = try cagraOptimize(knnGraph: shifted, graphDegree: graphDegree)
        eval(data, graph)
        return CagraIndex(dataset: data, graph: graph, metric: metric)
    }
}
