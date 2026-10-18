// Init-time configuration (C9) for the FLUX 3 Image Klein package.
//
// Weights materialize from the mlx-community mirror, pinned by commit. The mirror is a byte-identical copy of BFL's
// bf16 package: 22 files, sha256 compared on the hub after upload, 2026-10-17. It is GATED like upstream, so a first
// download needs a Hugging Face token whose account has accepted the FLUX Kommunity License; the engine resolves
// HF_TOKEN, then the Keychain, then the CLI token file. Quant tiers are made at load from the bf16 files.

import Flux3Klein
import Foundation
import MLXToolKit

/// The prompt upsampler's tier (black-forest-labs/flux-3-image-klein-pup, Apache-2.0; mlx-community mirrors, pinned).
/// Every tier keeps quoted text verbatim and box tables complete on BFL's 19 example requests (AB-R receipt, P4); lower
/// tiers paraphrase more. `.off` sends the prompt as typed.
public enum Flux3UpsamplerTier: String, Codable, Sendable, CaseIterable {
    case off, bf16, int8, int4

    public var repo: String? {
        switch self {
        case .off: nil
        case .bf16: "mlx-community/flux-3-image-klein-pup-bf16"
        case .int8: "mlx-community/flux-3-image-klein-pup-8bit"
        case .int4: "mlx-community/flux-3-image-klein-pup-4bit"
        }
    }

    /// Mirror commits, pinned (bf16 byte-identical to upstream 3f3edb52; 8/4-bit mlx-vlm 0.7.8 conversions with the
    /// upstream tokenizer, template, processor configs and system prompts).
    public var revision: String? {
        switch self {
        case .off: nil
        case .bf16: "cf0e42c70a05ad6f6fdef75e46267299ff12554e"
        case .int8: "3391b4504c1327229665a2e7a6b1ecd71e673c30"
        case .int4: "34abfa0769000d6a542169b0e61a04059697467c"
        }
    }
}

public struct Flux3KleinConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, FootprintConfigured,
    WeightSourcing {
    /// DiT tier: `.bf16`, `.int8`, or `.int4` (attention projections at 8 bits). All are made at load, and all are
    /// within the OCR tolerance of bf16 (AB-R-0576).
    public var quant: Quant
    /// The Qwen3-VL-4B conditioner tier (text-only), orthogonal to `quant`. It only moves the activation term.
    public var textEncoderQuant: Flux3TextEncoderQuant
    /// Explicit snapshot root (`model.safetensors`, `text_encoder/`, `vae/`). Empty = the store.
    public var snapshotPath: String
    public var defaultSteps: Int
    /// Keep the conditioner (8.0 GB bf16, 4.6 GB int8) resident between requests. Off: it loads per request and is
    /// released before denoising, which is what the declared footprint measures.
    public var keepEncoderResident: Bool
    /// The prompt upsampler, loaded per request and released before the encoder. FLUX 3 Klein wants full scene
    /// captions, so it is on by default (int8: 11.1 GB transient, under the declared activation). A request skips it
    /// with `metaData["upsample"] = false`, or by sending a finished caption with a cited box table.
    public var upsampler: Flux3UpsamplerTier
    /// Explicit upsampler snapshot (`system_prompts.json`, `config.json`, weights). Empty = the store.
    public var upsamplerPath: String
    /// BFL's safety checks (`flux3/safety.py`), OFF by default as upstream: the prompt (with every box description),
    /// each reference and the output are checked for copyrighted characters, brands and public figures (a yes/no on
    /// the upsampler, or on Qwen3-VL-4B when the upsampler is off) and images for adult content (an NSFW classifier).
    /// A flag fails the job with the reference's message. This is the licence's §2(e) content-filter route for a
    /// Qualifying User's commercial outputs; reviewing each output before it is distributed is the other route.
    public var safetyChecks: Bool
    /// Explicit NSFW classifier snapshot (Falconsai/nsfw_image_detection). Empty = the store.
    public var safetyClassifierPath: String
    public var modelsRootDirectory: URL?

    public static let repo = "mlx-community/flux-3-image-klein-bf16"
    /// Mirror commit (pinned; byte-identical to `upstreamRevision`).
    public static let revision = "d1fc3059ae6b06a09221fd6ee67e6100e0b1ec75"
    public static let upstreamRepo = "black-forest-labs/flux-3-image-klein"
    /// The upstream revision the port was gated against (oracle goldens, AB-R-0571).
    public static let upstreamRevision = "337337621b8526e120f2275d250242f4a6bedd75"
    /// The NSFW classifier the reference's safety checks load (Apache-2.0, ViT-B/16, 343 MB fp32), pinned.
    public static let safetyRepo = "Falconsai/nsfw_image_detection"
    public static let safetyRevision = "96cb0d0342c7afb80cab76ecc58b265fa44da256"

    public init(quant: Quant = .bf16, textEncoderQuant: Flux3TextEncoderQuant = .bf16, snapshotPath: String = "",
                defaultSteps: Int = Flux3Schedule.defaultSteps, keepEncoderResident: Bool = false,
                upsampler: Flux3UpsamplerTier = .int8, upsamplerPath: String = "", safetyChecks: Bool = false,
                safetyClassifierPath: String = "", modelsRootDirectory: URL? = nil) {
        self.quant = quant
        self.textEncoderQuant = textEncoderQuant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.keepEncoderResident = keepEncoderResident
        self.upsampler = upsampler
        self.upsamplerPath = upsamplerPath
        self.safetyChecks = safetyChecks
        self.safetyClassifierPath = safetyClassifierPath
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case quant, textEncoderQuant, snapshotPath, defaultSteps, keepEncoderResident, upsampler, upsamplerPath
        case safetyChecks, safetyClassifierPath
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decode(Quant.self, forKey: .quant)
        textEncoderQuant = try c.decodeIfPresent(Flux3TextEncoderQuant.self, forKey: .textEncoderQuant) ?? .bf16
        snapshotPath = try c.decodeIfPresent(String.self, forKey: .snapshotPath) ?? ""
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? Flux3Schedule.defaultSteps
        keepEncoderResident = try c.decodeIfPresent(Bool.self, forKey: .keepEncoderResident) ?? false
        upsampler = try c.decodeIfPresent(Flux3UpsamplerTier.self, forKey: .upsampler) ?? .int8
        upsamplerPath = try c.decodeIfPresent(String.self, forKey: .upsamplerPath) ?? ""
        safetyChecks = try c.decodeIfPresent(Bool.self, forKey: .safetyChecks) ?? false
        safetyClassifierPath = try c.decodeIfPresent(String.self, forKey: .safetyClassifierPath) ?? ""
        modelsRootDirectory = nil
    }

    /// The DiT tier keys the resident bytes (QuantFootprint); a resident encoder adds its weights to them. A
    /// per-request encoder is a transient under the declared activation at every size in the envelope, so neither
    /// encoder tier moves the activation term (Flux3KleinFootprints).
    /// The safety checks keep their classifier resident (0.34 GB fp32); their judge is a per-check transient.
    public var residentBytesHint: UInt64? {
        guard keepEncoderResident || safetyChecks else { return nil }
        let dit = Flux3KleinFootprints.declared.first { $0.quant == quant }?.residentBytes ?? 0
        return dit + (keepEncoderResident ? Flux3KleinFootprints.encoderBytes(textEncoderQuant) : 0)
            + (safetyChecks ? Flux3KleinFootprints.safetyClassifierBytes : 0)
    }

    /// Only the bf16 upsampler's transient (20.2 GB peak) exceeds the declared activation; int8 / int4 stay under it.
    public var peakActivationBytesHint: UInt64? {
        upsampler == .bf16 ? Flux3KleinFootprints.bf16UpsamplerPeakActivationBytes : nil
    }

    public var weightSources: [WeightSource] {
        var sources = [WeightSource(role: "snapshot", repo: Self.repo, revision: Self.revision,
                                    matching: ["model.safetensors", "config.json", "manifest.json", "text_encoder/*",
                                               "vae/*", "LICENSE.md", "NOTICE.md"])]
        if let repo = upsampler.repo {
            sources.append(WeightSource(role: "upsampler", repo: repo, revision: upsampler.revision))
        }
        if safetyChecks {
            sources.append(WeightSource(role: "safety", repo: Self.safetyRepo, revision: Self.safetyRevision,
                                        matching: ["config.json", "preprocessor_config.json", "model.safetensors"]))
        }
        return sources
    }

    static func safetyComplete(_ dir: URL) -> Bool {
        FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path)
    }

    /// The NSFW classifier root: the explicit path, else the store's flat or revision directory; nil when the checks
    /// are off.
    public func resolvedSafetyDirectory(storeRoot: URL?) -> URL? {
        guard safetyChecks else { return nil }
        if !safetyClassifierPath.isEmpty { return URL(fileURLWithPath: safetyClassifierPath) }
        let store = ModelStore(root: storeRoot)
        if let flat = store.directory(for: Self.safetyRepo), Self.safetyComplete(flat) { return flat }
        if let snap = store.snapshotDirectory(for: Self.safetyRepo, revision: Self.safetyRevision), Self.safetyComplete(snap) {
            return snap
        }
        return store.directory(for: Self.safetyRepo)
    }

    static func upsamplerComplete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ["system_prompts.json", "config.json", "tokenizer.json"].allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    /// The upsampler root: the explicit path, else the store's flat or revision directory; nil when `.off`.
    public func resolvedUpsamplerDirectory(storeRoot: URL?) -> URL? {
        guard let repo = upsampler.repo else { return nil }
        if !upsamplerPath.isEmpty { return URL(fileURLWithPath: upsamplerPath) }
        let store = ModelStore(root: storeRoot)
        if let flat = store.directory(for: repo), Self.upsamplerComplete(flat) { return flat }
        if let snap = store.snapshotDirectory(for: repo, revision: upsampler.revision), Self.upsamplerComplete(snap) {
            return snap
        }
        return store.directory(for: repo)
    }

    static func complete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ["model.safetensors", "text_encoder/config.json", "vae/config.json"].allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        weightSources.filter { source in
            if source.role == "upsampler" {
                guard let dir = resolvedUpsamplerDirectory(storeRoot: storeRoot) else { return true }
                return !Self.upsamplerComplete(dir)
            }
            if source.role == "safety" {
                guard let dir = resolvedSafetyDirectory(storeRoot: storeRoot) else { return true }
                return !Self.safetyComplete(dir)
            }
            if !snapshotPath.isEmpty, Self.complete(URL(fileURLWithPath: snapshotPath)) { return false }
            guard let dir = resolvedSnapshotDirectory(storeRoot: storeRoot) else { return true }
            return !Self.complete(dir)
        }
    }

    /// The snapshot root: the explicit path, else the store's flat or revision directory.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        if let flat = store.directory(for: Self.repo), Self.complete(flat) { return flat }
        if let snap = store.snapshotDirectory(for: Self.repo, revision: Self.revision), Self.complete(snap) {
            return snap
        }
        return store.directory(for: Self.repo)
    }
}
