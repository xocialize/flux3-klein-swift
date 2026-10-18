// The NSFW image classifier BFL's safety checks use (`flux3/safety.py`, bfc8889): Falconsai/nsfw_image_detection, a
// ViT-B/16 at 224² (Apache-2.0, 85.8M parameters, `ViTForImageClassification`), labels 0 = normal, 1 = nsfw. An image
// is flagged when P(nsfw) > 0.85.
//
// Reference facts:
//   - `ViTImageProcessor`: resize to 224×224 with PIL BILINEAR (`resample: 2`), rescale 1/255, normalize mean 0.5 /
//     std 0.5;
//   - patch embedding = a 16×16 stride-16 conv; a CLS token; learned position embeddings (197);
//   - 12 pre-norm blocks (LayerNorm eps 1e-12, 12 heads, exact GELU MLP 3072), a final LayerNorm, the classifier on
//     the CLS token;
//   - run in fp32, as the reference pipeline does.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Immutable after `load`, so it can cross tasks.
public final class Flux3NSFWClassifier: @unchecked Sendable {
    public static let threshold: Float = 0.85
    public static let imageSize = 224
    static let patch = 16
    static let hidden = 768
    static let heads = 12
    static let layers = 12
    static let eps: Float = 1e-12

    let w: [String: MLXArray]
    let patchWeight: MLXArray  // [768, 3·16·16], (c, kh, kw) order

    init(weights: [String: MLXArray]) {
        w = weights
        patchWeight = weights["vit.embeddings.patch_embeddings.projection.weight"]!.reshaped(Self.hidden, -1)
    }

    /// Load `model.safetensors` from a snapshot of Falconsai/nsfw_image_detection.
    public static func load(directory: URL) throws -> Flux3NSFWClassifier {
        let url = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux3KleinError.loading("no NSFW classifier weights at \(url.path)")
        }
        let weights = try loadArrays(url: url).mapValues { $0.asType(.float32) }
        let classifier = Flux3NSFWClassifier(weights: weights)
        eval(Array(classifier.w.values))
        return classifier
    }

    /// `ViTImageProcessor`: BILINEAR to 224², then (x/255 − 0.5)/0.5. NHWC [1, 224, 224, 3].
    public static func preprocess(_ image: Flux3RGBImage) -> MLXArray {
        let s = imageSize
        let resized = image.resized(width: s, height: s, filter: .bilinear)
        let x = MLXArray(resized.rgb, [1, s, s, 3]).asType(.float32) / 255
        return (x - 0.5) / 0.5
    }

    /// The classifier's logits [normal, nsfw] for a preprocessed batch [B, 224, 224, 3].
    public func logits(_ pixels: MLXArray) -> MLXArray {
        let b = pixels.dim(0), g = Self.imageSize / Self.patch, p = Self.patch
        // Patches in (c, kh, kw) order, matching the conv weight's flattening.
        let patches = pixels.reshaped(b, g, p, g, p, 3).transposed(0, 1, 3, 5, 2, 4).reshaped(b, g * g, 3 * p * p)
        var x = matmul(patches, patchWeight.T) + w["vit.embeddings.patch_embeddings.projection.bias"]!
        let cls = broadcast(w["vit.embeddings.cls_token"]!, to: [b, 1, Self.hidden])
        x = concatenated([cls, x], axis: 1) + w["vit.embeddings.position_embeddings"]!
        for i in 0..<Self.layers { x = block(x, prefix: "vit.encoder.layer.\(i).") }
        x = layerNorm(x, "vit.layernorm")
        return linear(x[0..., 0, 0...], "classifier")
    }

    /// P(nsfw) for one image.
    public func nsfwScore(_ image: Flux3RGBImage) -> Float {
        let p = softmax(logits(Self.preprocess(image)), axis: -1)
        return p[0, 1].item(Float.self)
    }

    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        matmul(x, w[name + ".weight"]!.T) + w[name + ".bias"]!
    }

    func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w[name + ".weight"]!, bias: w[name + ".bias"]!, eps: Self.eps)
    }

    func block(_ x: MLXArray, prefix: String) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let d = Self.hidden / Self.heads
        let h = layerNorm(x, prefix + "layernorm_before")
        func heads(_ name: String) -> MLXArray {
            linear(h, prefix + "attention.attention." + name).reshaped(b, n, Self.heads, d).transposed(0, 2, 1, 3)
        }
        let attn = MLXFast.scaledDotProductAttention(
            queries: heads("query"), keys: heads("key"), values: heads("value"), scale: 1 / Float(d).squareRoot(),
            mask: nil)
        var x = x + linear(attn.transposed(0, 2, 1, 3).reshaped(b, n, Self.hidden), prefix + "attention.output.dense")
        let m = gelu(linear(layerNorm(x, prefix + "layernorm_after"), prefix + "intermediate.dense"))
        x = x + linear(m, prefix + "output.dense")
        return x
    }
}
