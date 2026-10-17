// swift-tools-version: 6.2
// flux3-klein-swift — Swift/MLX port of black-forest-labs/flux-3-image-klein (FLUX Kommunity License v1.0 for the
// weights — see PORTING-SPEC.md §1; the port code here is MIT): a 5.55B flow-map DiT with per-mode early block
// stacks and shared single-stream blocks, conditioned on 8 hidden-state taps of Qwen3-VL-4B-Instruct, decoded by
// the FLUX.2 VAE (identical weights, AB-R-0571).
// Reference = BFL's own code (github.com/black-forest-labs/flux-3-image-klein @ bfc8889, Apache-2.0); goldens from
// mlxengine-image/WIP/flux3-klein-oracle (fp32 CPU torch).

import PackageDescription

let package = Package(
    name: "Flux3Klein",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Flux3Klein", targets: ["Flux3Klein"]),
        .executable(name: "Flux3KleinGate", targets: ["Flux3KleinGate"]),
    ],
    dependencies: [
        // 0.32.3 carries the NAX split-K GEMM fix (mlx#3810); the DiT's MLP down-projection has K = 9216.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
    ],
    targets: [
        .target(
            name: "Flux3Klein",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/Flux3Klein"
        ),
        .executableTarget(
            name: "Flux3KleinGate",
            dependencies: ["Flux3Klein", .product(name: "MLX", package: "mlx-swift")],
            path: "Sources/Flux3KleinGate"
        ),
    ]
)
