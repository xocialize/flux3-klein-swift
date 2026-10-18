// FLUX 3 Image Klein pipeline (distilled, flow map): prompt + up to 10 references → one image. Isomorphic to the
// reference `Pipeline.generate` + `generate.generate` minus the safety checks and the prompt upsampler, which callers
// run first: the model wants a full scene caption (see BFL's `system_prompts.json`).
//
// Precision follows the reference under CUDA autocast: the DiT runs in bf16 with bf16 text features, bf16 reference
// tokens and bf16-rounded timesteps; the image tokens stay fp32 across the Euler steps (bf16-rounded noise); the VAE
// runs in fp32, in row bands by default (flux2-vae-mlx-swift 0.6.0, AB-R-0577): the same image as the untiled decode
// to float reassociation, at a fraction of its peak (10.7 → 2.4 GB at 1024²).

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
    /// Row-band size for the fp32 VAE decode; nil decodes the whole latent at once (about 4× the peak).
    public var vaeTile: Flux2VAETile? = .standard

    public init(transformer: Flux3Transformer, textEncoder: Flux3TextEncoder?, vaeEncoder: Flux2VAEEncoder,
                vae: Flux2VAE, snapshot: URL) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.vaeEncoder = vaeEncoder
        self.vae = vae
        self.snapshot = snapshot
    }

    /// Load from a snapshot holding `model.safetensors`, `text_encoder/` and `vae/`.
    /// `quant` picks the DiT tier (quantized on load), `textEncoderQuant` the conditioner's.
    public static func load(snapshot: URL, dtype: DType = .bfloat16, quant: Flux3Quant = .bf16,
                            textEncoderQuant: Flux3TextEncoderQuant = .bf16) async throws -> Flux3KleinPipeline {
        let transformer = try Flux3Weights.loadTransformer(snapshot: snapshot, dtype: dtype, quant: quant)
        let textEncoder = try await Flux3TextEncoder.load(
            directory: snapshot.appendingPathComponent("text_encoder"), dtype: dtype, quant: textEncoderQuant)
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
        /// A single reference is fitted to this area (the reference's 4 MiP by default; engine packages cap it to
        /// their measured envelope).
        public var referenceMaxPixels: Int = Flux3References.matchMaxPixels

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

    /// `conditioning`: the prompt already encoded (`textEncoder.encode(request.prompt)`), e.g. so the encoder can be
    /// released before denoising; nil encodes here.
    public func generate(_ request: Request, conditioning: Flux3TextEncoder.Encoded? = nil,
                         noise providedNoise: MLXArray? = nil,
                         progress: ((Int, Int) throws -> Void)? = nil) throws -> Result {
        guard request.references.count <= Flux3References.maxReferences else {
            throw Flux3KleinError.invalidInput("at most \(Flux3References.maxReferences) references")
        }
        guard conditioning != nil || textEncoder != nil else {
            throw Flux3KleinError.invalidInput("the text encoder was released and no conditioning was given")
        }
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
            let prepared = request.references.map { Flux3References.prepareSingle($0, maxPixels: request.referenceMaxPixels) }
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
            eval(condTokens[n])  // one encoder transient at a time, never stacked into the first DiT step
            condIDs.append(Flux3Latents.ids(height: packed.dim(2), width: packed.dim(3), l: -Float(1 + n)))
        }
        lap("references")

        // Text.
        let text = conditioning ?? textEncoder!.encode(request.prompt)
        eval(text.features)
        lap("text")

        var result = try denoiseAndDecode(
            features: text.features, textIDs: text.ids, condTokens: condTokens, condIDs: condIDs, width: width,
            height: height, steps: request.steps, alpha: request.alpha,
            seed: request.seed ?? Self.defaultSeed(request.prompt), noise: providedNoise, crop: crop, progress: progress)
        result.timings.merge(timings) { a, _ in a }
        return result
    }

    /// Text-to-image from pre-computed conditioning (`Flux3TextEncoder.Encoded`'s features and ids), e.g. a prompt
    /// encoded once and rendered on several DiT tiers.
    public func generate(conditioned features: MLXArray, textIDs: MLXArray, width: Int = 1024, height: Int = 1024,
                         steps: Int = Flux3Schedule.defaultSteps, alpha: Double = Flux3Schedule.defaultAlpha,
                         seed: UInt32, noise: MLXArray? = nil) throws -> Result {
        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux3KleinError.invalidInput("width and height must be multiples of 16 (got \(width)×\(height))")
        }
        return try denoiseAndDecode(features: features, textIDs: textIDs, condTokens: [], condIDs: [], width: width,
                                    height: height, steps: steps, alpha: alpha, seed: seed, noise: noise, crop: nil,
                                    progress: nil)
    }

    func denoiseAndDecode(features rawFeatures: MLXArray, textIDs: MLXArray, condTokens: [MLXArray],
                          condIDs: [MLXArray], width: Int, height: Int, steps: Int, alpha: Double, seed: UInt32,
                          noise providedNoise: MLXArray?, crop: Flux3References.Box?,
                          progress: ((Int, Int) throws -> Void)?) throws -> Result {
        var timings: [String: Double] = [:]
        var clock = Date()
        func lap(_ name: String) { timings[name] = Date().timeIntervalSince(clock); clock = Date() }
        let dtype = transformer.dtype
        let features = rawFeatures.asType(dtype)

        // Noise and tokens.
        let (lh, lw) = (height / 16, width / 16)
        var x = providedNoise?.asType(.float32) ?? Self.noise(seed: seed, latentHeight: lh, latentWidth: lw)
        guard x.shape == [lh * lw, 128] else {
            throw Flux3KleinError.invalidInput("noise shape \(x.shape) ≠ [\(lh * lw), 128]")
        }
        var tokens = Flux3Tokens(
            image: x, imageIDs: Flux3Latents.ids(height: lh, width: lw),
            cond: condTokens.isEmpty ? nil : concatenated(condTokens, axis: 0),
            condIDs: condIDs.isEmpty ? nil : concatenated(condIDs, axis: 0),
            text: features, textIDs: textIDs)
        let rope = Flux3RopeSet(tokens, config: transformer.config)

        // Flow-map Euler.
        let schedule = Flux3Schedule.timesteps(steps: steps, alpha: alpha)
        for step in 0..<steps {
            tokens.image = x
            let velocity = transformer(tokens, rope: rope, t: schedule[step], tNext: schedule[step + 1])
            x = Flux3Schedule.step(x, velocity: velocity, t: schedule[step], tNext: schedule[step + 1])
            eval(x)
            try progress?(step + 1, steps)  // a throwing callback cancels between steps
        }
        lap("denoise")

        // Decode (fp32), crop back to a padded single reference's box.
        let packed = Flux3Latents.packed(fromTokens: x, height: lh, width: lw)
        let decoded = vaeTile.map { vae.decodePackedLatents(packed, tile: $0) } ?? vae.decodePackedLatents(packed)
        var image = Flux3RGBImage.fromPixels(decoded[0])
        if let crop { image = image.cropped(left: crop.left, top: crop.top, width: crop.width, height: crop.height) }
        lap("decode")
        return Result(image: image, seed: seed, width: image.width, height: image.height, timings: timings)
    }
}
