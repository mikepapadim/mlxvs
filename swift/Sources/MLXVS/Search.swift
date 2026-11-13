// Copyright © 2026 mikepapadim

import Foundation
import MLX
import MLXFast

#if os(macOS)
    import IOKit
#endif

/// Distance of a search: squared Euclidean, inner product or cosine.
public enum Metric: String, Sendable {
    case l2
    case innerProduct = "inner_product"
    case cosine
}

/// How a graph search runs: one threadgroup per query (many queries), several
/// per query (few queries), or chosen as cuVS chooses.
public enum SearchAlgo: Sendable {
    case auto
    case singleCTA
    case multiCTA
}

/// Options of ``graphSearch(base:graph:queries:k:metric:options:)``, as cuVS's
/// CAGRA search parameters.
public struct SearchOptions {
    /// Length of each query's candidate list: longer is more accurate and
    /// slower. At least k.
    public var itopkSize = 64
    /// Nodes expanded per iteration.
    public var searchWidth = 1
    public var algo = SearchAlgo.auto
    /// Lanes per distance computation: 0 picks by dimension (8, 16 or 32).
    public var teamSize = 0
    /// 0: cuVS's automatic iteration count.
    public var maxIterations = 0
    public var minIterations = 0
    public var numRandomSamplings = 1
    public var seed: UInt64 = 0x128394
    /// Rows that may be returned: a bool mask `[n]` (true: allowed).
    public var filter: MLXArray? = nil

    public init() {}
}

/// Neighbors of each query: ids `int32 [nq, k]` (-1 where missing) and
/// distances `float32 [nq, k]`, nearest first (squared distances for L2,
/// similarities for inner product and cosine).
public struct Neighbors {
    public let ids: MLXArray
    public let distances: MLXArray
}

public enum MLXVSError: Error, CustomStringConvertible {
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .invalidArgument(let message): return message
        }
    }
}

enum KernelCache {
    static let search = MLXFast.metalKernel(
        name: "mlxvs_graph_search",
        inputNames: ["base", "graph", "queries", "inv_norm", "filter", "seeds", "params"],
        outputNames: ["out_ids", "out_scores"],
        source: Kernels.searchSource,
        header: Kernels.searchHeader)
    static let searchMC = MLXFast.metalKernel(
        name: "mlxvs_graph_search_mc",
        inputNames: ["base", "graph", "queries", "inv_norm", "filter", "seeds", "params"],
        outputNames: ["out_ids", "out_scores", "traversed"],
        source: Kernels.searchMcSource,
        header: Kernels.searchHeader,
        ensureRowContiguous: true,
        atomicOutputs: true)
    static let topk = MLXFast.metalKernel(
        name: "mlxvs_topk_merge",
        inputNames: ["scores", "prev_ids", "prev_scores", "params"],
        outputNames: ["out_ids", "out_scores"],
        source: Kernels.topkSource,
        header: Kernels.topkHeader)
}

/// The GPU's core count (cuVS's number of SMs), 8 where it cannot be read.
let gpuCores: Int = {
    #if os(macOS)
        var found: Int32 = 0
        var iterator: io_iterator_t = 0
        if IOServiceGetMatchingServices(
            kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS
        {
            var service = IOIteratorNext(iterator)
            while service != 0 && found == 0 {
                if let value = IORegistryEntryCreateCFProperty(
                    service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                    .takeRetainedValue() as? NSNumber
                {
                    found = value.int32Value
                }
                IOObjectRelease(service)
                service = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }
        return found > 0 ? Int(found) : 8
    #else
        return 8
    #endif
}()

func pow2AtLeast(_ x: Int) -> Int {
    var p = 1
    while p < x { p <<= 1 }
    return p
}

/// One more hop per power of max(2, degree / 2) below n: cuVS's estimate of the
/// hops needed to reach every node.
func reachHops(_ n: Int, _ degree: Int) -> Int {
    var hops = 0
    var reach = 1
    while reach < n {
        reach *= max(2, degree / 2)
        hops += 1
    }
    return hops
}

/// Lanes per distance, as cuVS picks its team size.
func teamFor(_ d: Int, _ requested: Int) -> Int {
    if requested != 0 { return requested }
    let dims = [128, 256, 512]
    let teams = [8, 16, 32]
    var best = 0
    for i in 1 ..< 3 where abs(d - dims[i]) < abs(d - dims[best]) {
        best = i
    }
    return teams[best]
}

/// The filter mask as one bitset of uint32 words.
func bitset(_ mask: MLXArray, rows n: Int) throws -> MLXArray {
    guard mask.ndim == 1, mask.dim(0) == n else {
        throw MLXVSError.invalidArgument("filter must be a bool mask [n]")
    }
    let words = (n + 31) / 32
    let u = padded(mask.asType(.uint32), widths: [IntOrPair((0, words * 32 - n))])
    let bits = leftShift(MLXArray(UInt32(1)), MLXArray(0 ..< 32).asType(.uint32))
    return (u.reshaped([words, 32]) * bits).sum(axis: -1).asType(.uint32)
}

/// The k nearest base rows of every query by beam search of a graph (CAGRA's
/// single- and multi-CTA search), on the GPU.
///
/// - Parameters:
///   - base: rows `[n, d]`: float32, float16, int8 or uint8.
///   - graph: neighbor lists `[n, degree]` (integer ids).
///   - queries: rows `[nq, d]`.
///   - k: neighbors per query.
public func graphSearch(
    base: MLXArray, graph: MLXArray, queries queriesIn: MLXArray, k: Int,
    metric: Metric = .l2, options: SearchOptions = SearchOptions()
) throws -> Neighbors {
    let asIs: Set<DType> = [.float32, .float16, .int8, .uint8]
    let base = asIs.contains(base.dtype) ? base : base.asType(.float32)
    var queries = queriesIn.ndim == 1 ? queriesIn.reshaped([1, -1]) : queriesIn
    queries = queries.asType(.float32)
    guard base.ndim == 2, graph.ndim == 2, graph.dim(0) == base.dim(0) else {
        throw MLXVSError.invalidArgument("base must be [n, d] and graph [n, degree]")
    }
    guard queries.dim(1) == base.dim(1) else {
        throw MLXVSError.invalidArgument("base and queries must have the same columns")
    }
    let n = base.dim(0)
    let d = base.dim(1)
    let degree = graph.dim(1)
    let nq = queries.dim(0)
    let width = options.searchWidth
    guard width >= 1, width <= Kernels.searchWidthMax,
        width * degree <= Kernels.searchCandidatesMax
    else {
        throw MLXVSError.invalidArgument(
            "searchWidth must be in [1, 32] and searchWidth * degree at most \(Kernels.searchCandidatesMax)")
    }
    var itopk = max(options.itopkSize, k)
    let m = (itopk + 31) / 32 * 32
    let mcFits =
        (Kernels.searchMcItopk + degree + 31) / 32 * 32 <= Kernels.searchMcBufferMax
    let multi =
        options.algo == .multiCTA
        || (options.algo == .auto && (m > Kernels.searchItopkMax || nq < 2 * gpuCores) && mcFits)
    let itopkMax = multi ? 1024 : Kernels.searchItopkMax
    guard k >= 1, m <= itopkMax else {
        throw MLXVSError.invalidArgument("k and itopkSize must be in [1, \(itopkMax)]")
    }
    let graph32 = graph.dtype == .int32 ? graph : graph.asType(.int32)
    var invNorm = MLXArray([Float(1)])
    if metric == .cosine {
        let qn = sqrt((queries * queries).sum(axis: 1, keepDims: true))
        queries = queries / maximum(qn, MLXArray(Float(1e-30)))
        let b = base.asType(.float32)
        invNorm = rsqrt(maximum((b * b).sum(axis: 1), MLXArray(Float(1e-30))))
    }
    let metricCode = metric == .innerProduct ? 0 : (metric == .l2 ? 1 : 2)
    if nq == 0 {
        return Neighbors(ids: MLXArray.zeros([0, k], type: Int32.self), distances: MLXArray.zeros([0, k]))
    }
    guard [0, 4, 8, 16, 32].contains(options.teamSize) else {
        throw MLXVSError.invalidArgument("teamSize must be 0, 4, 8, 16 or 32")
    }
    var filter = MLXArray([UInt32(0xffff_ffff)])
    var filtered = 0
    if let mask = options.filter {
        filter = try bitset(mask, rows: n)
        filtered = 1
        let rate = 1 - mask.asType(.float32).mean().item(Float.self)
        if multi && rate > 0 && rate < 1 {
            // as cuVS's search plan: a longer list when the filter rejects rows
            var adjusted = Int(
                Float(k) / (1 - rate) + Float(itopk - k) / (1 - rate).squareRoot())
            adjusted = (adjusted + 31) / 32 * 32
            itopk = min(max(itopk, adjusted), 1024)
        }
    }
    let prepared = Prepared(
        base: base, graph: graph32, queries: queries, invNorm: invNorm, metricCode: metricCode,
        team: teamFor(d, options.teamSize), filter: filter, filtered: filtered)
    return multi
        ? searchMultiCTA(prepared, k: k, itopk: itopk, options: options)
        : searchSingleCTA(prepared, k: k, itopk: itopk, options: options)
}

struct Prepared {
    let base: MLXArray
    let graph: MLXArray
    let queries: MLXArray
    let invNorm: MLXArray
    let metricCode: Int
    let team: Int
    let filter: MLXArray
    let filtered: Int
}

/// Queries per kernel launch, within the GPU's command time limits.
let searchQueryBlock = 65536

/// Scores (larger is closer) as distances: squared distances for L2.
func toDistances(_ scores: MLXArray, _ metricCode: Int) -> MLXArray {
    metricCode == 1 ? -scores : scores
}

func searchSingleCTA(_ p: Prepared, k: Int, itopk: Int, options: SearchOptions) -> Neighbors {
    let n = p.base.dim(0)
    let d = p.base.dim(1)
    let degree = p.graph.dim(1)
    let nq = p.queries.dim(0)
    let width = options.searchWidth
    let m = (itopk + 31) / 32 * 32
    let cp = pow2AtLeast(width * degree)
    var iterations = options.maxIterations
    if iterations <= 0 {
        iterations = itopk / width + reachHops(n, degree)
    }
    iterations = max(iterations, options.minIterations)
    let maxIter = max(1, iterations - 1)
    let nRand = max(1, options.numRandomSamplings)
    let randSlots = m + width * degree
    // threadgroup memory within 16 KB, the visited set taking what is left
    var qshared = d <= Kernels.searchQuerySharedMax
    var qcap = qshared ? (d + 31) / 32 * 32 : 1
    func fixed() -> Int { 16 * m + 8 * cp + 4 * qcap + 256 }
    var hmin = 9
    while (1 << hmin) < 2 * (m + cp) { hmin += 1 }
    if qshared && fixed() + (4 << hmin) > 16384 {
        qshared = false
        qcap = 1
    }
    var hbits = hmin
    while hbits < 12 && fixed() + (4 << (hbits + 1)) <= 16384 { hbits += 1 }

    var ids: [MLXArray] = []
    var dist: [MLXArray] = []
    for q0 in stride(from: 0, to: nq, by: searchQueryBlock) {
        let q1 = min(nq, q0 + searchQueryBlock)
        let bq = q1 - q0
        let params = MLXArray(
            [
                n, d, degree, k, width, maxIter, p.metricCode,
                Int(options.seed & 0x7fff_ffff) ^ q0, nRand, randSlots,
            ].map { Int32($0) })
        let out = KernelCache.search(
            [p.base, p.graph, p.queries[q0 ..< q1], p.invNorm, p.filter, MLXArray([Int32(-1)]), params],
            template: [
                ("M", m), ("CP", cp), ("QCAP", qcap), ("QSHARED", qshared), ("HBITS", hbits),
                ("TEAM", p.team), ("FILTER", p.filtered), ("NSEEDS", 0), ("PQ_LEN", 1),
                ("PQ_DIM", 1),
            ],
            grid: (bq * Kernels.searchThreads, 1, 1),
            threadGroup: (Kernels.searchThreads, 1, 1),
            outputShapes: [[bq, k], [bq, k]],
            outputDTypes: [.int32, .float32])
        ids.append(out[0])
        dist.append(out[1])
    }
    return Neighbors(
        ids: ids.count == 1 ? ids[0] : concatenated(ids, axis: 0),
        distances: dist.count == 1 ? dist[0] : concatenated(dist, axis: 0))
}

func searchMultiCTA(_ p: Prepared, k: Int, itopk: Int, options: SearchOptions) -> Neighbors {
    let n = p.base.dim(0)
    let d = p.base.dim(1)
    let degree = p.graph.dim(1)
    let nq = p.queries.dim(0)
    let list = Kernels.searchMcItopk
    let buffer = (list + degree + 31) / 32 * 32
    // as cuVS's search plan
    let numCTA = max(options.searchWidth, (itopk + list - 1) / list)
    var iterations = options.maxIterations
    if iterations <= 0 {
        let minimumDepth = 16
        iterations =
            minimumDepth + (list - minimumDepth + numCTA - 1) / numCTA + (k + list - 1) / list
            + reachHops(n, degree)
    }
    iterations = max(iterations, options.minIterations)
    var vbits = 8
    while (1 << vbits) / 2 < list + 2 * degree { vbits += 1 }
    let traversedNodes = numCTA * max(list, iterations)
    var tbits = 11
    while (1 << tbits) / 2 < traversedNodes { tbits += 1 }
    var qshared = d <= Kernels.searchQuerySharedMax
    var qcap = qshared ? (d + 31) / 32 * 32 : 1
    let bp = pow2AtLeast(buffer)
    if qshared && 4 * qcap + (4 << vbits) + 8 * bp + 64 > 16384 {
        qshared = false
        qcap = 1
    }
    let block = max(1, min(searchQueryBlock, (64 << 20) >> tbits))
    let cols = numCTA * list

    var ids: [MLXArray] = []
    var dist: [MLXArray] = []
    for q0 in stride(from: 0, to: nq, by: block) {
        let q1 = min(nq, q0 + block)
        let bq = q1 - q0
        let params = MLXArray(
            [
                n, d, degree, iterations, options.minIterations, p.metricCode,
                Int(options.seed & 0x7fff_ffff) ^ q0, max(1, options.numRandomSamplings), numCTA,
                bq, tbits,
            ].map { Int32($0) })
        let out = KernelCache.searchMC(
            [p.base, p.graph, p.queries[q0 ..< q1], p.invNorm, p.filter, MLXArray([Int32(-1)]), params],
            template: [
                ("B", buffer), ("BP", bp), ("QCAP", qcap), ("QSHARED", qshared), ("VBITS", vbits),
                ("TEAM", p.team), ("FILTER", p.filtered), ("NSEEDS", 0), ("PQ_LEN", 1),
                ("PQ_DIM", 1),
            ],
            grid: (numCTA * Kernels.searchMcThreads, bq, 1),
            threadGroup: (Kernels.searchMcThreads, 1, 1),
            outputShapes: [[bq, cols], [bq, cols], [bq, 1 << tbits]],
            outputDTypes: [.int32, .float32, .int32],
            initValue: -1)
        // the best k of the threadgroups' lists
        let order = argSort(-out[1], axis: 1)[0..., 0 ..< k]
        ids.append(takeAlong(out[0], order, axis: 1))
        dist.append(toDistances(takeAlong(out[1], order, axis: 1), p.metricCode))
    }
    return Neighbors(
        ids: ids.count == 1 ? ids[0] : concatenated(ids, axis: 0),
        distances: dist.count == 1 ? dist[0] : concatenated(dist, axis: 0))
}
