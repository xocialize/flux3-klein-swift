// Weight loading for FLUX 3 Image Klein: the DiT loads from BFL's native `model.safetensors` (the key set pinned by
// the snapshot's `manifest.json`: 359 tensors) under its own key names, two-way strict. Quantized tiers are made on
// load from the bf16 file (affine, group 64), block by block from the lazy loads, so the bf16 copy is never resident
// in full.

import Foundation
import MLX
import MLXNN

/// DiT weight tier.
public enum Flux3Quant: String, Codable, Sendable, CaseIterable {
    case bf16, int8, int4

    public var bits: Int? {
        switch self {
        case .bf16: nil
        case .int8: 8
        case .int4: 4
        }
    }
}

public enum Flux3Weights {

    /// `final_layer.image.adaLN_modulation.1.weight` (Sequential(SiLU, Linear)) → `…adaLN_modulation.weight`.
    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: ".adaLN_modulation.1.", with: ".adaLN_modulation.")
    }

    /// The DiT from a snapshot root holding `model.safetensors` (the distilled flow-map checkpoint), cast to `dtype`,
    /// quantized on load when `quant` asks for it.
    public static func loadTransformer(snapshot: URL, dtype: DType = .bfloat16, quant: Flux3Quant = .bf16,
                                       groupSize: Int = 64) throws -> Flux3Transformer {
        let file = snapshot.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw Flux3KleinError.loading("no model.safetensors under \(snapshot.path)")
        }
        var weights: [String: MLXArray] = [:]
        for (k, v) in try MLX.loadArrays(url: file) { weights[sanitizeDiTKey(k)] = v.asType(dtype) }
        guard weights["ref_time_in.in_layer.weight"] != nil else {
            throw Flux3KleinError.loading("\(file.path) is not a flow-map checkpoint (no ref_time_in): "
                                          + "the base (CFG) checkpoint needs the CFG sampler")
        }
        let model = Flux3Transformer()
        try verifyAndLoad(model: model, weights: weights, label: "FLUX 3 klein DiT", evaluate: quant.bits == nil)
        model.dtype = dtype
        if let bits = quant.bits {
            quantizeTransformer(model, bits: bits, groupSize: groupSize,
                                eightBit: quant == .int4 ? int4EightBitLayers : [])
        }
        return model
    }

    /// The int4 tier's 8-bit layers: every block's attention projections. Uniform 4-bit (group 64) drops the last
    /// single block's hidden state to cos 0.84–0.86 against fp32 and moves 1024² renders to ~20 dB from bf16. Keeping
    /// q/k/v/attn_out at 8 bits recovers cos 0.94–0.997, the best of six recipes probed (group 32, `mlp_out` at 8,
    /// late single blocks at 8, q/k only, v/o only, early stacks too). ≈31 % of the block weights stay 8-bit.
    public static let int4EightBitLayers = ["q_proj", "k_proj", "v_proj", "attn_out"]

    /// Layers kept at the load dtype in a quantized tier: the token embeddings (`emb_in`, `txt_in`), the timestep
    /// MLPs, the six per-mode modulation producers (shared by every block of a stream) and the final layer. About
    /// 0.27B of the 5.55B parameters.
    static func keepHiPrecision(path: String) -> Bool {
        ["emb_in", "txt_in", "time_in", "ref_time_in", "early_stream_modulations", "single_stream_modulations",
         "final_layer"].contains { path.hasPrefix($0) }
    }

    /// Quantize every block Linear in place (affine). `eightBit`: path substrings kept at 8 bits inside a 4-bit tier
    /// (a mixed recipe). `FLUX3_QUANT_GROUP` / `FLUX3_QUANT_EIGHT_BIT` (comma-separated substrings) override both for
    /// recipe work. Evaluated one block at a time, so a model still holding lazy loads is read, quantized and dropped
    /// block by block.
    public static func quantizeTransformer(_ model: Flux3Transformer, bits: Int, groupSize: Int = 64,
                                           eightBit: [String] = []) {
        let env = ProcessInfo.processInfo.environment
        let group = env["FLUX3_QUANT_GROUP"].flatMap(Int.init) ?? groupSize
        let keep8 = env["FLUX3_QUANT_EIGHT_BIT"].map { $0.split(separator: ",").map(String.init) } ?? eightBit
        quantize(model: model, filter: { path, module in
            guard module is Linear, !keepHiPrecision(path: path) else { return nil }
            let b = keep8.contains { path.contains($0) } ? 8 : bits
            return (group, b, .affine)
        })
        for block in model.contentBlocks.image + model.contentBlocks.imageCond + model.textBlocks + model.singleBlocks {
            eval(block)
        }
        eval(model)
    }

    /// Two-way strict load: every module key filled, every checkpoint key consumed, shapes equal. `evaluate: false`
    /// leaves the lazy loads in place for a quantizer to consume.
    public static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String,
                                     evaluate: Bool = true) throws {
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        let missing = Set(shapes.keys).subtracting(weights.keys).sorted()
        guard missing.isEmpty else {
            throw Flux3KleinError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. " + missing.prefix(4).joined(separator: ", "))
        }
        let unused = Set(weights.keys).subtracting(shapes.keys).sorted()
        guard unused.isEmpty else {
            throw Flux3KleinError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. " + unused.prefix(4).joined(separator: ", "))
        }
        for (k, v) in weights where shapes[k] != v.shape {
            throw Flux3KleinError.loading("\(label): shape mismatch at \(k): checkpoint \(v.shape) vs module \(shapes[k] ?? [])")
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        if evaluate { eval(model) }
    }
}
