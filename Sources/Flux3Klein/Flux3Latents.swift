// Latent tokens and ids (reference `autoencoder.py` `latent_ids` / `encode` flatten, `pipeline.py` `scaled_ids`).

import Foundation
import MLX

public enum Flux3Latents {
    /// `IMAGE_ROPE_SCALE`: image and reference (h, w) ids are halved before RoPE.
    public static let imageRopeScale: Float = 0.5

    /// Packed latents [1, 128, h, w] → tokens [h·w, 128], row-major (the reference's `permute(1, 2, 0).reshape`).
    public static func tokens(fromPacked packed: MLXArray) -> MLXArray {
        let (c, h, w) = (packed.dim(1), packed.dim(2), packed.dim(3))
        return packed[0].transposed(1, 2, 0).reshaped(h * w, c)
    }

    /// Tokens [h·w, 128] → packed latents [1, 128, h, w] (the decoder's input).
    public static func packed(fromTokens tokens: MLXArray, height h: Int, width w: Int) -> MLXArray {
        tokens.reshaped(h, w, tokens.dim(1)).transposed(2, 0, 1)[.newAxis]
    }

    /// ids (t, h·scale, w·scale, l) for an h × w latent grid, float32 [h·w, 4]. l = 0 for the target, −(1 + n) for
    /// reference n.
    public static func ids(height h: Int, width w: Int, l: Float = 0, scale: Float = imageRopeScale) -> MLXArray {
        var values = [Float]()
        values.reserveCapacity(h * w * 4)
        for r in 0..<h { for c in 0..<w { values += [0, Float(r) * scale, Float(c) * scale, l] } }
        return MLXArray(values, [h * w, 4])
    }
}
