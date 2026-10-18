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
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // Qwen3-VL conditioner backbone; ≥ 0.5.1 for hiddenStates(inputIds:layers:mask:positionIds:).
        .package(url: "https://github.com/xocialize/qwen3vl-mlx-swift", from: "0.5.1"),
        // The FLUX.2 VAE (FLUX 3 ships identical weights, AB-R-0571); ≥ 0.5.0 for Flux2VAEEncoder.
        .package(url: "https://github.com/xocialize/flux2-vae-mlx-swift", from: "0.5.0"),
    ],
    targets: [
        .target(
            name: "Flux3Klein",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
            ],
            path: "Sources/Flux3Klein"
        ),
        .executableTarget(
            name: "Flux3KleinGate",
            dependencies: ["Flux3Klein", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift")],
            path: "Sources/Flux3KleinGate"
        ),
    ]
)
