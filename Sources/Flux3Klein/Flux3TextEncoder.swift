// Qwen3-VL-4B conditioner for FLUX 3 Image Klein — the reference `text_encoder.py` on the fleet's qwen3vl-mlx-swift
// backbone. The snapshot's `text_encoder/` is byte-identical to Qwen/Qwen3-VL-4B-Instruct (AB-R-0571).
//
// Reference facts (pinned by the oracle's encoder goldens):
//   - `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)` with no system turn:
//     "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
//   - truncated at 4096 tokens, right-padded with <|endoftext|> to a multiple of 80 (capped at 4096);
//   - attention mask on the padding; positions arange for real tokens and 1 for pads, on all M-RoPE rows;
//   - features = `hidden_states[k]` for k = 4, 8, …, 32 (all before the final norm), concatenated per token in layer
//     order → [L, 20480]; text ids (0, 0, 0, position) over the padded length.
//
// Padding is not free: the DiT attends to every padded row. Under the causal mask the pads never change the real
// tokens, and every pad row is the same token at position 1 attending to the same real keys, so every pad row is
// identical. The encoder therefore runs the real tokens plus ONE pad row (mask: the pad attends to real keys only)
// and repeats that row over the padding.

import Foundation
import MLX
import MLXNN
import Qwen3VL
import Tokenizers

/// Text-encoder weight tier.
public enum Flux3TextEncoderQuant: String, Sendable, CaseIterable {
    case bf16, int8

    public var bits: Int? { self == .int8 ? 8 : nil }
}

public final class Flux3TextEncoder {
    public let model: Qwen3VL
    public let tokenizer: any Tokenizers.Tokenizer
    public let padId: Int

    public static let outputLayers = [4, 8, 12, 16, 20, 24, 28, 32]
    public static let padMultiple = 80
    public static let maxLength = 4096

    public static func template(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    public init(model: Qwen3VL, tokenizer: any Tokenizers.Tokenizer) throws {
        self.model = model
        self.tokenizer = tokenizer
        guard let pad = tokenizer.convertTokenToId("<|endoftext|>") else {
            throw Flux3KleinError.loading("tokenizer lacks <|endoftext|> (the pad token)")
        }
        self.padId = pad
    }

    /// Load `text_encoder/` from a FLUX 3 klein snapshot (or a stock Qwen3-VL-4B-Instruct folder), language model
    /// only: the conditioner never sees images, so the vision tower is not built or read. `quant: .int8` quantizes the
    /// language model's Linears after load (affine, group 64); the embedding table stays at `dtype`.
    public static func load(directory: URL, dtype: DType = .bfloat16, quant: Flux3TextEncoderQuant = .bf16,
                            groupSize: Int = 64) async throws -> Flux3TextEncoder {
        let model = try Qwen3VLLoader.load(directory: directory, dtype: dtype, includeVision: false)
        if let bits = quant.bits {
            quantize(model: model, filter: { _, module in module is Linear ? (groupSize, bits, .affine) : nil })
            eval(model)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return try Flux3TextEncoder(model: model, tokenizer: tokenizer)
    }

    public struct Encoded {
        /// [L, 20480] features over the padded length.
        public let features: MLXArray
        /// [L, 4] text ids (0, 0, 0, position), float32.
        public let ids: MLXArray
        /// Template token ids before padding.
        public let tokenIds: [Int]
        public var paddedLength: Int { ids.dim(0) }
    }

    public func encode(_ prompt: String) -> Encoded {
        var ids = tokenizer.encode(text: Self.template(prompt), addSpecialTokens: false)
        if ids.count > Self.maxLength { ids = Array(ids.prefix(Self.maxLength)) }
        let real = ids.count
        let length = min((real + Self.padMultiple - 1) / Self.padMultiple * Self.padMultiple, Self.maxLength)

        let seq = length > real ? ids + [padId] : ids
        let n = seq.count
        let positions = (0..<real).map { Int32($0) } + (length > real ? [1] : [])
        let positionIds = broadcast(MLXArray(positions).reshaped(1, 1, n), to: [3, 1, n])
        var mask: MLXArray? = nil
        if length > real {
            // rows < real: causal; the pad row (index real) sees keys 0..<real.
            let i = MLXArray(0..<n).reshaped(n, 1)
            let j = MLXArray(0..<n).reshaped(1, n)
            let isPad = i .== Int32(real)
            mask = MLX.where(isPad, j .< Int32(real), j .<= i).reshaped(1, 1, n, n)
        }
        let states = model.hiddenStates(
            inputIds: MLXArray(seq.map { Int32($0) }).reshaped(1, n), layers: Self.outputLayers, mask: mask,
            positionIds: positionIds)
        var features = concatenated(states.map { $0[0] }, axis: -1)  // [n, 8·2560], layer-major per token
        if length > real {
            let padRow = features[real ..< real + 1]
            features = concatenated([features[..<real], broadcast(padRow, to: [length - real, features.dim(1)])],
                                    axis: 0)
        }
        let textIds: [Float] = (0..<length).flatMap { p -> [Float] in [0, 0, 0, Float(p)] }
        return Encoded(features: features, ids: MLXArray(textIds, [length, 4]), tokenIds: ids)
    }
}
