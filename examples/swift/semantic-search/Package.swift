// swift-tools-version: 6.0
// Copyright © 2026 mikepapadim

// semantic-search: embed text with an embedding model (mlx-swift-lm's
// MLXEmbedders) and search it on the GPU with MLXVS.

import PackageDescription

let package = Package(
    name: "semantic-search",
    platforms: [.macOS("14.0")],
    dependencies: [
        .package(path: "../../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "semantic-search",
            dependencies: [
                .product(name: "MLXVS", package: "mlxvs"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        )
    ]
)
