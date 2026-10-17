// Weight loading for FLUX 3 Image Klein: the DiT loads from BFL's native `model.safetensors` (the key set pinned by
// the snapshot's `manifest.json`: 359 tensors) under its own key names, two-way strict.

import Foundation
import MLX
import MLXNN

public enum Flux3Weights {

    /// `final_layer.image.adaLN_modulation.1.weight` (Sequential(SiLU, Linear)) → `…adaLN_modulation.weight`.
    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: ".adaLN_modulation.1.", with: ".adaLN_modulation.")
    }

    /// The DiT from a snapshot root holding `model.safetensors` (the distilled flow-map checkpoint), cast to `dtype`.
    public static func loadTransformer(snapshot: URL, dtype: DType = .bfloat16) throws -> Flux3Transformer {
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
        try verifyAndLoad(model: model, weights: weights, label: "FLUX 3 klein DiT")
        model.dtype = dtype
        return model
    }

    /// Two-way strict load: every module key filled, every checkpoint key consumed, shapes equal.
    public static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
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
        eval(model)
    }
}
