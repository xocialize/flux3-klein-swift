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
import Flux3KleinUpsampler
import Foundation
import MLX
import MLXToolKit

extension SPDXLicense {
    /// FLUX Kommunity License v1.0 (black-forest-labs/flux-3-image-klein `LICENSE.md`): no SPDX identifier exists, so
    /// it takes the `LicenseRef-` convention. Intentionally absent from `permissiveAllowlist`.
    public static let fluxKommunity: SPDXLicense = "LicenseRef-FLUX-Kommunity-1.0"
}

/// `description` is the readable message too: engines that report `"\(error)"` (Pixel Agency Image Engine's MES
/// `engine_error`) show the sentence, not the enum case.
public enum Flux3KleinPackageError: Error, LocalizedError, CustomStringConvertible {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    case unsupportedQuant(Quant)
    /// The request would run outside the envelope the declared footprint was MEASURED at, so the memory governor's
    /// reservation would under-state it. Refused rather than clamped: silently shrinking a size changes the output.
    case outsideMeasuredEnvelope(String)
    /// A safety check flagged the prompt, a reference or the output (the reference's message). Only with
    /// `safetyChecks` on.
    case contentRefused(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "FLUX 3 Image Klein snapshot not readable at \(p)."
        case .imageDecode: return "Could not decode a reference image."
        case .pngEncode: return "PNG encoding failed."
        case .unsupportedQuant(let q): return "FLUX 3 Image Klein has no \(q) tier (bf16, int8 and int4 are declared)."
        case .outsideMeasuredEnvelope(let why): return why
        case .contentRefused(let why): return "FLUX 3 Image Klein safety check refused the request: \(why)"
        }
    }

    public var description: String { errorDescription ?? "FLUX 3 Image Klein error" }
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

/// Package-specific `metaData` levers (contract C5).
public enum Flux3KleinMetaKeys {
    /// Bool. `false` sends the prompt to the image model as typed (for a caller that wrote a full scene caption).
    /// Absent = upsample when the configuration has an upsampler.
    public static let upsample = "upsample"
    /// Array of objects: layout boxes on 0–1 coordinates, `{"id", "bbox": [y_min, x_min, y_max, x_max], "desc"}`, or
    /// box edits `{"id", "from": "ref_image_n" | "target", "src_bbox", "tgt_bbox" (null removes), "desc"}`.
    public static let boxes = "boxes"

    static func upsample(_ meta: MetaData) throws -> Bool? {
        switch meta[upsample] {
        case nil, .null?: return nil
        case .bool(let b)?: return b
        default: throw PackageError.unsupportedRequestFeature("metaData[\"\(upsample)\"] must be a boolean")
        }
    }

    static func boxes(_ meta: MetaData) throws -> [Flux3Box]? {
        guard let value = meta[boxes], value != .null else { return nil }
        guard case .array(let items) = value else {
            throw PackageError.unsupportedRequestFeature("metaData[\"\(boxes)\"] must be an array of box objects")
        }
        func numbers(_ v: MetaValue?) throws -> [Double]? {
            guard let v, v != .null else { return nil }
            guard case .array(let a) = v else { throw PackageError.unsupportedRequestFeature("a box is a list of 4 numbers") }
            return try a.map {
                switch $0 {
                case .double(let d): return d
                case .int(let i): return Double(i)
                default: throw PackageError.unsupportedRequestFeature("a box is a list of 4 numbers")
                }
            }
        }
        return try items.map { item in
            guard case .object(let o) = item else { throw PackageError.unsupportedRequestFeature("each box is an object") }
            func str(_ k: String) -> String? { if case .string(let s)? = o[k] { return s }; return nil }
            return Flux3Box(id: str("id"), desc: str("desc"), bbox: try numbers(o["bbox"]), from: str("from"),
                            srcBBox: try numbers(o["src_bbox"]), tgtBBox: try numbers(o["tgt_bbox"]))
        }
    }
}

/// The DiT and VAE stay resident. The prompt upsampler (when configured) and then the conditioner load per request and
/// are released before denoising, unless `keepEncoderResident` is set. With the safety checks on, the NSFW classifier
/// stays resident and the yes/no judge loads per screening pass: the upsampler when one is configured (the input pass
/// hands it on to the caption), else Qwen3-VL-4B from the snapshot's text encoder.
@InferenceActor
final class Flux3KleinGenerator {
    let pipeline: Flux3KleinPipeline
    let snapshot: URL
    let keepEncoderResident: Bool
    let textEncoderQuant: Flux3TextEncoderQuant
    let upsamplerDirectory: URL?
    let safetyClassifier: Flux3NSFWClassifier?
    private var residentEncoder: Flux3TextEncoder?

    init(pipeline: Flux3KleinPipeline, snapshot: URL, keepEncoderResident: Bool, textEncoderQuant: Flux3TextEncoderQuant,
         upsamplerDirectory: URL?, safetyClassifier: Flux3NSFWClassifier? = nil) {
        self.pipeline = pipeline
        self.snapshot = snapshot
        self.keepEncoderResident = keepEncoderResident
        self.textEncoderQuant = textEncoderQuant
        self.upsamplerDirectory = upsamplerDirectory
        self.safetyClassifier = safetyClassifier
    }

    static func load(configuration c: Flux3KleinConfiguration, snapshot: URL, upsampler: URL?,
                     safety: URL? = nil) throws -> Flux3KleinGenerator {
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
        let classifier = try safety.map { try Flux3NSFWClassifier.load(directory: $0) }
        return Flux3KleinGenerator(pipeline: pipeline, snapshot: snapshot, keepEncoderResident: c.keepEncoderResident,
                                   textEncoderQuant: c.textEncoderQuant, upsamplerDirectory: upsampler,
                                   safetyClassifier: classifier)
    }

    // MARK: Safety checks (the reference's `generate.generate` with `safety=True`)

    /// The judge for one screening pass, created in the caller's scope: the upsampler when one is configured (the
    /// reference checks on it whenever it is loaded, even for a request that skips upsampling), else Qwen3-VL-4B-Instruct
    /// from the snapshot's `text_encoder/`, as a VLM.
    private func loadJudge() async throws -> (Flux3SafetyJudge, Flux3PromptUpsampler?) {
        if let dir = upsamplerDirectory {
            let upsampler = try await Flux3PromptUpsampler.load(directory: dir)
            return (try await Flux3SafetyJudge(upsampler: upsampler), upsampler)
        }
        let judge = try await Flux3SafetyJudge.load(textEncoderDirectory: snapshot.appendingPathComponent("text_encoder"))
        return (judge, nil)
    }

    /// The input checks: the prompt with every box description, then each reference as given. Returns the upsampler it
    /// loaded as the judge, for the caption to reuse; nil (and no work) with the checks off.
    func screenInputs(prompt: String, boxes: [Flux3Box]?, references: [Flux3RGBImage]) async throws -> Flux3PromptUpsampler? {
        guard let classifier = safetyClassifier else { return nil }
        RunProgress.report(.screen)
        let (judge, upsampler) = try await loadJudge()
        let checker = Flux3SafetyChecker(classifier: classifier, judge: judge)
        if try await checker.checkPrompt(Flux3SafetyPrompts.screened(prompt, boxes: boxes)).flagged {
            throw Flux3KleinPackageError.contentRefused(
                "The prompt may name a copyrighted character, a brand or a public figure.")
        }
        for (i, image) in references.enumerated() {
            try Task.checkCancellation()
            if let reason = try await checker.checkImage(image).reason {
                throw Flux3KleinPackageError.contentRefused("Reference image \(i + 1) may show \(reason).")
            }
        }
        return upsampler
    }

    /// The input checks, then the caption, in one scope: an upsampler loaded as the judge is reused for the caption and
    /// released when this returns, before the encoder and the denoise loop.
    func screenedCaption(for prompt: String, originals: [Flux3RGBAImage], references: [Flux3RGBImage], boxes: [Flux3Box]?,
                         upsample: Bool) async throws -> String {
        let judge = try await screenInputs(prompt: prompt, boxes: boxes, references: references)
        let caption = try await caption(for: prompt, references: originals, boxes: boxes, upsample: upsample, loaded: judge)
        MLX.Memory.clearCache()
        return caption
    }

    /// The output check: the classifier first; the judge loads only when the classifier passes the image.
    func screenOutput(_ image: Flux3RGBImage) async throws {
        guard let classifier = safetyClassifier else { return }
        RunProgress.report(.screen)
        func refuse(_ reason: String) -> Flux3KleinPackageError {
            .contentRefused("The generated image may show \(reason), so it was not returned. Try another prompt or seed.")
        }
        if classifier.nsfwScore(image) > Flux3NSFWClassifier.threshold { throw refuse(Flux3SafetyChecker.nsfwReason) }
        let flagged = try await loadJudge().0.imageMargin(image) > 0
        MLX.Memory.clearCache()
        if flagged { throw refuse(Flux3SafetyChecker.promptReason) }
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

    /// The caption the image model gets, as the reference's `generate.generate` builds it: a finished caption (box
    /// table cited in full) goes as it is; otherwise the upsampler writes it (boxes reconciled); without one, boxes are
    /// appended to the prompt on the 0–1000 canvas.
    func caption(for prompt: String, references: [Flux3RGBAImage], boxes: [Flux3Box]?, upsample: Bool,
                 loaded: Flux3PromptUpsampler? = nil) async throws -> String {
        let finished = (boxes?.isEmpty ?? true) && Flux3Boxes.isFinished(prompt)
        if upsample, !finished, let dir = upsamplerDirectory {
            // Loaded (or handed on by the input checks) and dropped inside this scope, so it never lives into the
            // encoder or the denoise loop.
            let upsampler: Flux3PromptUpsampler
            if let loaded { upsampler = loaded } else { upsampler = try await Flux3PromptUpsampler.load(directory: dir) }
            let caption = try await upsampler.upsample(prompt, references: references, boxes: boxes)
            MLX.Memory.clearCache()
            return caption
        }
        if let boxes, !boxes.isEmpty {
            return "\(prompt.trimmingTrailingSpace()) \(Flux3JSON.array(try Flux3Boxes.toCanvas(boxes)).dumps())"
        }
        return prompt
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
        var upsampler: URL? = nil
        if c.upsampler != .off {
            guard let dir = c.resolvedUpsamplerDirectory(storeRoot: c.modelsRootDirectory),
                  Flux3KleinConfiguration.upsamplerComplete(dir)
            else { throw Flux3KleinPackageError.unreadableSnapshot(c.upsamplerPath.isEmpty ? c.upsampler.repo ?? "" : c.upsamplerPath) }
            upsampler = dir
        }
        var safety: URL? = nil
        if c.safetyChecks {
            guard let dir = c.resolvedSafetyDirectory(storeRoot: c.modelsRootDirectory), Flux3KleinConfiguration.safetyComplete(dir)
            else { throw Flux3KleinPackageError.unreadableSnapshot(c.safetyClassifierPath.isEmpty ? Flux3KleinConfiguration.safetyRepo : c.safetyClassifierPath) }
            safety = dir
        }
        return try Flux3KleinGenerator.load(configuration: c, snapshot: snapshot, upsampler: upsampler, safety: safety)
    }

    static func run(_ request: any CapabilityRequest, generator: Flux3KleinGenerator,
                    configuration c: Flux3KleinConfiguration) async throws -> any CapabilityResponse {
        let prompt: String
        let originals: [Flux3RGBAImage]
        let width: Int?, height: Int?, steps: Int?, seed: UInt64?
        let meta: MetaData
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
            prompt = t2i.prompt; originals = []
            width = t2i.width ?? 1024; height = t2i.height ?? 1024
            steps = t2i.steps; seed = t2i.seed; meta = t2i.metaData
        case .imageEdit:
            guard let edit = request as? IEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            guard !edit.images.isEmpty else { throw Flux3KleinPackageError.imageDecode }
            prompt = edit.prompt
            originals = try edit.images.map { img in
                do { return try Flux3RGBAImage.read(data: img.data) } catch { throw Flux3KleinPackageError.imageDecode }
            }
            width = edit.width; height = edit.height
            steps = edit.steps; seed = edit.seed; meta = edit.metaData
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
        let references = originals.map(Flux3RGBImage.init)
        if let why = Flux3KleinEnvelope.violation(width: width, height: height, referenceCount: references.count) {
            throw Flux3KleinPackageError.outsideMeasuredEnvelope(why)
        }
        let boxes = try Flux3KleinMetaKeys.boxes(meta)
        let upsample = try Flux3KleinMetaKeys.upsample(meta) ?? true
        try Task.checkCancellation()
        let caption = try await generator.screenedCaption(for: prompt, originals: originals, references: references,
                                                          boxes: boxes, upsample: upsample)
        try Task.checkCancellation()  // post-upsample seam
        var req = Flux3KleinPipeline.Request(
            prompt: caption, references: references, width: width, height: height, steps: steps ?? c.defaultSteps,
            seed: seed.map { UInt32(truncatingIfNeeded: $0) })
        req.referenceMaxPixels = Flux3KleinEnvelope.maxReferencePixels
        let result = try await generator.generate(req)
        try Task.checkCancellation()
        try await generator.screenOutput(result.image)
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
