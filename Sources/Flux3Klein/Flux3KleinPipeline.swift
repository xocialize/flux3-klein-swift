// FLUX 3 Image Klein pipeline (distilled, flow map): prompt + up to 10 references → one image. Isomorphic to the
// reference `Pipeline.generate` + `generate.generate` minus the safety checks and the prompt upsampler, which callers
// run first: the model wants a full scene caption (see BFL's `system_prompts.json`).
//
// Precision follows the reference under CUDA autocast: the DiT runs in bf16 with bf16 text features, bf16 reference
// tokens and bf16-rounded timesteps; the image tokens stay fp32 across the Euler steps (bf16-rounded noise); the VAE
// runs in fp32.

import CryptoKit
import Flux2VAE
import Foundation
import MLX
import MLXRandom

public final class Flux3KleinPipeline {
    public let transformer: Flux3Transformer
    public private(set) var textEncoder: Flux3TextEncoder?
    public let vaeEncoder: Flux2VAEEncoder
    public let vae: Flux2VAE
    public let snapshot: URL

    public init(transformer: Flux3Transformer, textEncoder: Flux3TextEncoder, vaeEncoder: Flux2VAEEncoder,
                vae: Flux2VAE, snapshot: URL) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.vaeEncoder = vaeEncoder
        self.vae = vae
        self.snapshot = snapshot
    }

    /// Load from a snapshot holding `model.safetensors`, `text_encoder/` and `vae/`.
    public static func load(snapshot: URL, dtype: DType = .bfloat16) async throws -> Flux3KleinPipeline {
        let transformer = try Flux3Weights.loadTransformer(snapshot: snapshot, dtype: dtype)
        let textEncoder = try await Flux3TextEncoder.load(
            directory: snapshot.appendingPathComponent("text_encoder"), dtype: dtype)
        let vaeDir = snapshot.appendingPathComponent("vae")
        let vaeEncoder = try Flux2VAEWeights.loadEncoder(directory: vaeDir, dtype: .float32)
        let vae = try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .float32)
        return Flux3KleinPipeline(transformer: transformer, textEncoder: textEncoder, vaeEncoder: vaeEncoder,
                                  vae: vae, snapshot: snapshot)
    }

    public struct Request: Sendable {
        /// The full scene caption (upsampled), optionally with a layout box table appended.
        public var prompt: String
        public var references: [Flux3RGBImage]
        /// Output size; nil = the first prepared reference's size (cropped back to its fitted box), else 1024².
        public var width: Int?
        public var height: Int?
        public var steps: Int
        public var alpha: Double
        /// nil = sha256(prompt) mod 2³², as the reference does.
        public var seed: UInt32?

        public init(prompt: String, references: [Flux3RGBImage] = [], width: Int? = nil, height: Int? = nil,
                    steps: Int = Flux3Schedule.defaultSteps, alpha: Double = Flux3Schedule.defaultAlpha,
                    seed: UInt32? = nil) {
            self.prompt = prompt
            self.references = references
            self.width = width
            self.height = height
            self.steps = steps
            self.alpha = alpha
            self.seed = seed
        }
    }

    public struct Result {
        public var image: Flux3RGBImage
        public var seed: UInt32
        public var width: Int
        public var height: Int
        public var timings: [String: Double]
    }

    /// The reference's default seed: int(sha256(prompt).hexdigest(), 16) % 2**32 — the digest's last 4 bytes.
    public static func defaultSeed(_ prompt: String) -> UInt32 {
        let digest = Array(SHA256.hash(data: Data(prompt.utf8)))
        return digest.suffix(4).reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
    }

    /// Generation noise [h·w, 128]: standard normal, rounded to bf16, kept in fp32. MLX's generator, not torch's:
    /// pass `noise` to `generate` to replay a reference run.
    public static func noise(seed: UInt32, latentHeight h: Int, latentWidth w: Int) -> MLXArray {
        MLXRandom.normal([h * w, 128], key: MLXRandom.key(UInt64(seed))).asType(.bfloat16).asType(.float32)
    }

    /// Drop the text encoder (~8.9 GB) after the last prompt has been encoded.
    public func releaseTextEncoder() { textEncoder = nil }

    public func generate(_ request: Request, noise providedNoise: MLXArray? = nil,
                         progress: ((Int, Int) -> Void)? = nil) throws -> Result {
        guard request.references.count <= Flux3References.maxReferences else {
            throw Flux3KleinError.invalidInput("at most \(Flux3References.maxReferences) references")
        }
        guard let textEncoder else { throw Flux3KleinError.invalidInput("the text encoder was released") }
        var timings: [String: Double] = [:]
        var clock = Date()
        func lap(_ name: String) { timings[name] = Date().timeIntervalSince(clock); clock = Date() }
        let dtype = transformer.dtype

        // References.
        var references: [Flux3RGBImage]
        var crop: Flux3References.Box? = nil
        if request.references.count > 1 {
            references = request.references.map(Flux3References.prepareMulti)
        } else {
            let prepared = request.references.map(Flux3References.prepareSingle)
            references = prepared.map(\.image)
            if request.width == nil && request.height == nil { crop = prepared.first?.box }
        }
        let width = request.width ?? references.first?.width ?? 1024
        let height = request.height ?? references.first?.height ?? 1024
        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux3KleinError.invalidInput("width and height must be multiples of 16 (got \(width)×\(height))")
        }
        var condTokens: [MLXArray] = []
        var condIDs: [MLXArray] = []
        for (n, reference) in references.enumerated() {
            let packed = vaeEncoder.encodePacked(reference.pixelsNCHW())
            condTokens.append(Flux3Latents.tokens(fromPacked: packed).asType(dtype))
            condIDs.append(Flux3Latents.ids(height: packed.dim(2), width: packed.dim(3), l: -Float(1 + n)))
        }
        lap("references")

        // Text.
        let text = textEncoder.encode(request.prompt)
        let features = text.features.asType(dtype)
        eval(features)
        lap("text")

        // Noise and tokens.
        let (lh, lw) = (height / 16, width / 16)
        let seed = request.seed ?? Self.defaultSeed(request.prompt)
        var x = providedNoise?.asType(.float32) ?? Self.noise(seed: seed, latentHeight: lh, latentWidth: lw)
        guard x.shape == [lh * lw, 128] else {
            throw Flux3KleinError.invalidInput("noise shape \(x.shape) ≠ [\(lh * lw), 128]")
        }
        var tokens = Flux3Tokens(
            image: x, imageIDs: Flux3Latents.ids(height: lh, width: lw),
            cond: condTokens.isEmpty ? nil : concatenated(condTokens, axis: 0),
            condIDs: condIDs.isEmpty ? nil : concatenated(condIDs, axis: 0),
            text: features, textIDs: text.ids)
        let rope = Flux3RopeSet(tokens, config: transformer.config)

        // Flow-map Euler.
        let schedule = Flux3Schedule.timesteps(steps: request.steps, alpha: request.alpha)
        for step in 0..<request.steps {
            tokens.image = x
            let velocity = transformer(tokens, rope: rope, t: schedule[step], tNext: schedule[step + 1])
            x = Flux3Schedule.step(x, velocity: velocity, t: schedule[step], tNext: schedule[step + 1])
            eval(x)
            progress?(step + 1, request.steps)
        }
        lap("denoise")

        // Decode (fp32), crop back to a padded single reference's box.
        let decoded = vae.decodePackedLatents(Flux3Latents.packed(fromTokens: x, height: lh, width: lw))
        var image = Flux3RGBImage.fromPixels(decoded[0])
        if let crop { image = image.cropped(left: crop.left, top: crop.top, width: crop.width, height: crop.height) }
        lap("decode")
        return Result(image: image, seed: seed, width: image.width, height: image.height, timings: timings)
    }
}
