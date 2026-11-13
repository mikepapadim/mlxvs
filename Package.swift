// swift-tools-version: 5.9
// Copyright © 2026 mikepapadim

// MLXVS for Swift: GPU vector search on Apple silicon (macOS and iOS), with the
// Metal kernels of the C++ / Python library. Search indexes built with the
// Python package (mlxvs.CagraIndex.save), or exact k-NN.

import PackageDescription

let package = Package(
    name: "mlxvs",
    platforms: [.macOS("14.0"), .iOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "MLXVS", targets: ["MLXVS"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.0" ..< "0.34.0")
    ],
    targets: [
        .target(
            name: "MLXVS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "swift/Sources/MLXVS"
        ),
        .testTarget(
            name: "MLXVSTests",
            dependencies: ["MLXVS"],
            path: "swift/Tests/MLXVSTests",
            resources: [.copy("Resources")]
        ),
    ]
)
