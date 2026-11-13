// Copyright © 2026 mikepapadim

// Embeds the paragraphs of a folder's text files and answers queries with the
// nearest paragraphs, searched on the GPU with MLXVS:
//
//   swift run semantic-search <folder> "query" ["query" ...]
//
// Corpora of more than 2,000 paragraphs are searched with a CAGRA graph index
// built on the GPU (CagraIndex.build), smaller ones exactly (bruteForceKNN).

import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import MLXVS
import Tokenizers

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count >= 2 else {
    print("usage: semantic-search <folder> \"query\" [\"query\" ...]")
    exit(1)
}
let folder = URL(fileURLWithPath: arguments[0])
let queries = Array(arguments.dropFirst())

// the paragraphs of the folder's .md and .txt files
var passages: [(source: String, text: String)] = []
let files = FileManager.default.enumerator(
    at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
while let file = files?.nextObject() as? URL {
    guard ["md", "txt"].contains(file.pathExtension.lowercased()),
        let contents = try? String(contentsOf: file, encoding: .utf8)
    else { continue }
    for paragraph in contents.components(separatedBy: "\n\n") {
        let text = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.count >= 40 {
            passages.append((file.lastPathComponent, text))
        }
    }
}
guard !passages.isEmpty else {
    print("no .md or .txt paragraphs under \(folder.path)")
    exit(1)
}

let model = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: EmbedderRegistry.bge_small
)

/// Unit-length embeddings of texts, [count, dim].
func embed(_ texts: [String]) async -> MLXArray {
    var rows: [[Float]] = []
    for start in stride(from: 0, to: texts.count, by: 32) {
        let batch = Array(texts[start ..< min(texts.count, start + 32)])
        let vectors = await model.perform { context -> [[Float]] in
            let tokens = batch.map {
                Array(context.tokenizer.encode(text: $0, addSpecialTokens: true).prefix(512))
            }
            let length = tokens.map(\.count).max() ?? 1
            let pad = context.tokenizer.eosTokenId ?? 0
            let ids = stacked(tokens.map { MLXArray($0 + Array(repeating: pad, count: length - $0.count)) })
            let mask = stacked(
                tokens.map {
                    MLXArray(Array(repeating: true, count: $0.count) + Array(repeating: false, count: length - $0.count))
                })
            let pooled = context.pooling(
                context.model(ids, positionIds: nil, tokenTypeIds: MLXArray.zeros(like: ids), attentionMask: mask),
                normalize: true, applyLayerNorm: false)
            pooled.eval()
            return pooled.map { $0.asArray(Float.self) }
        }
        rows.append(contentsOf: vectors)
    }
    return MLXArray(rows.flatMap { $0 }, [rows.count, rows[0].count])
}

var start = Date()
let corpus = await embed(passages.map(\.text))
print(String(format: "embedded %d paragraphs in %.1f s", passages.count, Date().timeIntervalSince(start)))

let index: CagraIndex? = passages.count > 2000
    ? try CagraIndex.build(rows: corpus, graphDegree: 32, metric: .cosine)
    : nil
let questions = await embed(queries)
let k = min(3, passages.count)
var result = Neighbors(ids: MLXArray([Int32(0)]), distances: MLXArray([Float(0)]))
for round in 0 ..< 2 {  // the first round compiles the Metal kernels
    if round == 1 { start = Date() }
    result = try index?.search(questions, k: k) ?? bruteForceKNN(base: corpus, queries: questions, k: k, metric: .cosine)
    eval(result.ids, result.distances)
}
print(String(format: "searched in %.1f ms (%@)", Date().timeIntervalSince(start) * 1000, index == nil ? "exact" : "CAGRA graph"))

let ids = result.ids.asArray(Int32.self)
let scores = result.distances.asArray(Float.self)
for (q, query) in queries.enumerated() {
    print("\n\(query)")
    for j in 0 ..< k where ids[q * k + j] >= 0 {
        let passage = passages[Int(ids[q * k + j])]
        let preview = passage.text.replacingOccurrences(of: "\n", with: " ").prefix(110)
        print(String(format: "  %.3f  %@: %@", scores[q * k + j], passage.source, String(preview)))
    }
}
