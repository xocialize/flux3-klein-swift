// MLXEngine package for FLUX 3 Image Klein: `textToImage` + `imageEdit` on one core (the model is a unified
// generator/editor; `run(_:)` dispatches on the request).
//
// LICENCE (C7): the DiT and VAE are under the FLUX Kommunity License v1.0.
//   - It permits non-commercial use, and commercial use of OUTPUTS by Qualifying Users (under US$5M revenue) who
//     filter content and disclose AI generation. Outputs must never train a similar model (§4(b), AB-B-0041).
//   - It is declared package-locally as `LicenseRef-FLUX-Kommunity-1.0` and is deliberately NOT allowlisted, so a
//     `.permissiveOnly` engine refuses or flags this package by construction (AB-D-0151). It is a restricted,
//     named-only tier for Pixel Agency Image Engine, not a product asset.
// C8: the port code is MIT.

import Flux2VAE
import Flux3Klein
import Foundation
import MLX
import MLXToolKit

extension SPDXLicense {
    /// FLUX Kommunity License v1.0 (black-forest-labs/flux-3-image-klein `LICENSE.md`): no SPDX identifier exists, so
    /// it takes the `LicenseRef-` convention. Intentionally absent from `permissiveAllowlist`.
    public static let fluxKommunity: SPDXLicense = "LicenseRef-FLUX-Kommunity-1.0"
}

public enum Flux3KleinPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    case unsupportedQuant(Quant)
    /// The request would run outside the envelope the declared footprint was MEASURED at, so the memory governor's
    /// reservation would under-state it. Refused rather than clamped: silently shrinking a size changes the output.
    case outsideMeasuredEnvelope(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "FLUX 3 Image Klein snapshot not readable at \(p)."
        case .imageDecode: return "Could not decode a reference image."
        case .pngEncode: return "PNG encoding failed."
        case .unsupportedQuant(let q): return "FLUX 3 Image Klein has no \(q) tier (bf16, int8 and int4 are declared)."
        case .outsideMeasuredEnvelope(let why): return why
        }
    }
}

/// The input envelope the manifest's footprint was measured at (Flux3KleinFootprints). `run()` refuses anything
/// outside it, so the declaration stays an upper bound (AB-L-0202).
public enum Flux3KleinEnvelope {
    /// Output area cap (2048², 4.19 MP). The row-band decode made the output cheap (~1.7 GB per MP, AB-R-0577).
    public static let maxOutputPixels = Flux3KleinFootprints.envelopeMaxPixels
    /// A single reference is fitted to this area (2048², 4.19 MP), the reference's own 4 MiP: since v0.4.0 its fp32 VAE
    /// encode runs in row bands, so a 2048² reference costs 7.22 GB activation, what a 2048² output does (AB-R-0580).
    /// Several references are each fitted under 1 MiP by the model itself.
    public static let maxReferencePixels = Flux3KleinFootprints.envelopeReferenceMaxPixels
    public static let minSide = 64
    /// The model's maximum, measured at 1024² references.
    public static let maxReferenceImages = Flux3References.maxReferences

    /// nil when inside the envelope, else a user-facing reason. `width`/`height` are the requested size (nil = the
    /// first reference's prepared size, which `maxReferencePixels` keeps inside the cap, or 1024²).
    public static func violation(width: Int?, height: Int?, referenceCount: Int) -> String? {
        if referenceCount > maxReferenceImages {
            return "FLUX 3 Image Klein accepts at most \(maxReferenceImages) reference images; got \(referenceCount)."
        }
        guard let w = width, let h = height else {
            return (width == nil) == (height == nil) ? nil : "FLUX 3 Image Klein needs both width and height, or neither."
        }
        if w % 16 != 0 || h % 16 != 0 { return "FLUX 3 Image Klein sizes must be multiples of 16 (got \(w)×\(h))." }
        if w < minSide || h < minSide { return "FLUX 3 Image Klein sides must be at least \(minSide) px (got \(w)×\(h))." }
        if w * h > maxOutputPixels {
            let side = Int(Double(maxOutputPixels).squareRoot())
            return "FLUX 3 Image Klein: \(w)×\(h) exceeds the measured envelope (output area ≤ \(side)×\(side))."
        }
        return nil
    }
}

/// The DiT and VAE stay resident. The conditioner loads per request and is released before denoising, unless
/// `keepEncoderResident` is set.
@InferenceActor
final class Flux3KleinGenerator {
    let pipeline: Flux3KleinPipeline
    let snapshot: URL
    let keepEncoderResident: Bool
    let textEncoderQuant: Flux3TextEncoderQuant
    private var residentEncoder: Flux3TextEncoder?

    init(pipeline: Flux3KleinPipeline, snapshot: URL, keepEncoderResident: Bool, textEncoderQuant: Flux3TextEncoderQuant) {
        self.pipeline = pipeline
        self.snapshot = snapshot
        self.keepEncoderResident = keepEncoderResident
        self.textEncoderQuant = textEncoderQuant
    }

    static func load(configuration c: Flux3KleinConfiguration, snapshot: URL) throws -> Flux3KleinGenerator {
        let quant: Flux3Quant
        switch c.quant {
        case .bf16: quant = .bf16
        case .int8: quant = .int8
        case .int4: quant = .int4
        default: throw Flux3KleinPackageError.unsupportedQuant(c.quant)
        }
        let transformer = try Flux3Weights.loadTransformer(snapshot: snapshot, quant: quant)
        let vaeDir = snapshot.appendingPathComponent("vae")
        let vaeEncoder = try Flux2VAEWeights.loadEncoder(directory: vaeDir, dtype: .float32)
        let vae = try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .float32)
        MLX.Memory.clearCache()
        let pipeline = Flux3KleinPipeline(transformer: transformer, textEncoder: nil, vaeEncoder: vaeEncoder, vae: vae,
                                          snapshot: snapshot)
        return Flux3KleinGenerator(pipeline: pipeline, snapshot: snapshot, keepEncoderResident: c.keepEncoderResident,
                                   textEncoderQuant: c.textEncoderQuant)
    }

    /// The prompt's conditioning. A per-request encoder is created and dropped inside this scope, so it never lives
    /// into the denoise loop.
    private func encode(_ prompt: String) async throws -> Flux3TextEncoder.Encoded {
        if residentEncoder == nil && keepEncoderResident { residentEncoder = try await loadEncoder() }
        let encoder: Flux3TextEncoder
        if let residentEncoder { encoder = residentEncoder } else { encoder = try await loadEncoder() }
        let encoded = encoder.encode(prompt)
        eval(encoded.features)
        return encoded
    }

    private func loadEncoder() async throws -> Flux3TextEncoder {
        try await Flux3TextEncoder.load(directory: snapshot.appendingPathComponent("text_encoder"),
                                        quant: textEncoderQuant)
    }

    func generate(_ request: Flux3KleinPipeline.Request) async throws -> Flux3KleinPipeline.Result {
        let conditioning = try await encode(request.prompt)
        MLX.Memory.clearCache()
        try Task.checkCancellation()  // post-encode seam
        let result = try pipeline.generate(request, conditioning: conditioning) { step, total in
            RunProgress.report(.denoise, step: step, totalSteps: total)
            try Task.checkCancellation()  // between denoise steps
        }
        MLX.Memory.clearCache()
        return result
    }
}

@InferenceActor
enum Flux3KleinServing {
    static func load(_ c: Flux3KleinConfiguration) throws -> Flux3KleinGenerator {
        guard let snapshot = c.resolvedSnapshotDirectory(storeRoot: c.modelsRootDirectory),
              Flux3KleinConfiguration.complete(snapshot)
        else { throw Flux3KleinPackageError.unreadableSnapshot(c.snapshotPath.isEmpty ? Flux3KleinConfiguration.repo : c.snapshotPath) }
        return try Flux3KleinGenerator.load(configuration: c, snapshot: snapshot)
    }

    static func run(_ request: any CapabilityRequest, generator: Flux3KleinGenerator,
                    configuration c: Flux3KleinConfiguration) async throws -> any CapabilityResponse {
        let prompt: String
        let references: [Flux3RGBImage]
        let width: Int?, height: Int?, steps: Int?, seed: UInt64?
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
            prompt = t2i.prompt; references = []
            width = t2i.width ?? 1024; height = t2i.height ?? 1024
            steps = t2i.steps; seed = t2i.seed
        case .imageEdit:
            guard let edit = request as? IEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            guard !edit.images.isEmpty else { throw Flux3KleinPackageError.imageDecode }
            prompt = edit.prompt
            references = try edit.images.map { img in
                do { return try Flux3RGBImage.read(data: img.data) } catch { throw Flux3KleinPackageError.imageDecode }
            }
            width = edit.width; height = edit.height
            steps = edit.steps; seed = edit.seed
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
        if let why = Flux3KleinEnvelope.violation(width: width, height: height, referenceCount: references.count) {
            throw Flux3KleinPackageError.outsideMeasuredEnvelope(why)
        }
        try Task.checkCancellation()
        var req = Flux3KleinPipeline.Request(
            prompt: prompt, references: references, width: width, height: height, steps: steps ?? c.defaultSteps,
            seed: seed.map { UInt32(truncatingIfNeeded: $0) })
        req.referenceMaxPixels = Flux3KleinEnvelope.maxReferencePixels
        let result = try await generator.generate(req)
        try Task.checkCancellation()
        let png: Data
        do { png = try Flux3PNG.pngData(result.image.rgba()) } catch { throw Flux3KleinPackageError.pngEncode }
        let artifact = Image(format: .png, data: png, width: result.width, height: result.height)
        return request.capability == .textToImage ? T2IResponse(image: artifact) : IEditResponse(image: artifact)
    }

    /// What a caller (or an agent writing the prompt) must know: the model wants a full scene caption.
    nonisolated static let promptGuidance =
        "Write a FULL SCENE CAPTION, not a short request: medium and style, the scene, lighting and camera, and every "
        + "subject with its colours, materials and place in the frame. Put any text to render in quotes. A short "
        + "prompt gives generic output. For layout, append a JSON list of boxes after the caption: "
        + "[{\"id\": \"name_1\", \"bbox\": [top, left, bottom, right] on a 0–1000 grid, \"desc\": \"…\"}], and tag "
        + "each element in the caption as <name_1>. BFL's prompt upsampler system prompts: "
        + "huggingface.co/black-forest-labs/flux-3-image-klein-pup (system_prompts.json)."

    nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .fluxKommunity, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: Flux3KleinConfiguration.repo, revision: Flux3KleinConfiguration.revision,
                                   tier: 1),
            requirements: RequirementsManifest(
                footprints: Flux3KleinFootprints.declared,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil  // memory admission gates
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "flux-3-klein",
                    summary: "FLUX 3 Image Klein text-to-image (Black Forest Labs, 5.55B flow-map DiT on Qwen3-VL-4B; "
                        + "16 steps, no guidance): strong English text rendering, bounding-box layout in the prompt. "
                        + promptGuidance + " FLUX KOMMUNITY LICENCE: non-commercial, or commercial outputs only for "
                        + "businesses under US$5M revenue that filter content and disclose AI generation; never train on "
                        + "outputs.",
                    modes: []),
                IEditContract.descriptor(
                    name: "flux-3-klein",
                    summary: "FLUX 3 Image Klein editing and composition with up to 10 reference images. One reference "
                        + "keeps its own size, fitted under 4.19 MP (2048²); several are each fitted under 1 MP. Write the "
                        + "caption as a complete description of the result, the edit folded in. Refer to the n-th "
                        + "reference as 'the n-th image', or as <ref_image_n> in box edits. " + promptGuidance
                        + " FLUX KOMMUNITY LICENCE (see text-to-image).",
                    modes: []),
            ]
        )
    }
}

@InferenceActor
public final class Flux3KleinPackage: ModelPackage {
    public typealias Configuration = Flux3KleinConfiguration

    public nonisolated static var manifest: PackageManifest { Flux3KleinServing.manifest }

    let configuration: Configuration
    private var generator: Flux3KleinGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        generator = try Flux3KleinServing.load(configuration)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()  // CAN-1: first act of run()
        guard let generator else { throw PackageError.notLoaded }
        return try await Flux3KleinServing.run(request, generator: generator, configuration: configuration)
    }
}

extension Flux3KleinPackage {
    public nonisolated static var registration: PackageRegistration { .of(Flux3KleinPackage.self) }
}
