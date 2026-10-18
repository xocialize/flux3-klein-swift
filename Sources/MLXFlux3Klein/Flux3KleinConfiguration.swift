// Init-time configuration (C9) for the FLUX 3 Image Klein package.
//
// Weights materialize from the mlx-community mirror, pinned by commit. The mirror is a byte-identical copy of BFL's
// bf16 package: 22 files, sha256 compared on the hub after upload, 2026-10-17. It is GATED like upstream, so a first
// download needs a Hugging Face token whose account has accepted the FLUX Kommunity License; the engine resolves
// HF_TOKEN, then the Keychain, then the CLI token file. Quant tiers are made at load from the bf16 files.

import Flux3Klein
import Foundation
import MLXToolKit

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
    public var modelsRootDirectory: URL?

    public static let repo = "mlx-community/flux-3-image-klein-bf16"
    /// Mirror commit (pinned; byte-identical to `upstreamRevision`).
    public static let revision = "d1fc3059ae6b06a09221fd6ee67e6100e0b1ec75"
    public static let upstreamRepo = "black-forest-labs/flux-3-image-klein"
    /// The upstream revision the port was gated against (oracle goldens, AB-R-0571).
    public static let upstreamRevision = "337337621b8526e120f2275d250242f4a6bedd75"

    public init(quant: Quant = .bf16, textEncoderQuant: Flux3TextEncoderQuant = .bf16, snapshotPath: String = "",
                defaultSteps: Int = Flux3Schedule.defaultSteps, keepEncoderResident: Bool = false,
                modelsRootDirectory: URL? = nil) {
        self.quant = quant
        self.textEncoderQuant = textEncoderQuant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.keepEncoderResident = keepEncoderResident
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case quant, textEncoderQuant, snapshotPath, defaultSteps, keepEncoderResident
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decode(Quant.self, forKey: .quant)
        textEncoderQuant = try c.decodeIfPresent(Flux3TextEncoderQuant.self, forKey: .textEncoderQuant) ?? .bf16
        snapshotPath = try c.decodeIfPresent(String.self, forKey: .snapshotPath) ?? ""
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? Flux3Schedule.defaultSteps
        keepEncoderResident = try c.decodeIfPresent(Bool.self, forKey: .keepEncoderResident) ?? false
        modelsRootDirectory = nil
    }

    /// The DiT tier keys the resident bytes (QuantFootprint); a resident encoder adds its weights to them. A
    /// per-request encoder is a transient under the declared activation at every size in the envelope, so neither
    /// encoder tier moves the activation term (Flux3KleinFootprints).
    public var residentBytesHint: UInt64? {
        guard keepEncoderResident else { return nil }
        let dit = Flux3KleinFootprints.declared.first { $0.quant == quant }?.residentBytes ?? 0
        return dit + Flux3KleinFootprints.encoderBytes(textEncoderQuant)
    }

    public var peakActivationBytesHint: UInt64? { nil }

    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: Self.repo, revision: Self.revision,
                      matching: ["model.safetensors", "config.json", "manifest.json", "text_encoder/*", "vae/*",
                                 "LICENSE.md", "NOTICE.md"])]
    }

    static func complete(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return ["model.safetensors", "text_encoder/config.json", "vae/config.json"].allSatisfy {
            fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if !snapshotPath.isEmpty, Self.complete(URL(fileURLWithPath: snapshotPath)) { return [] }
        guard let dir = resolvedSnapshotDirectory(storeRoot: storeRoot), Self.complete(dir) else { return weightSources }
        return []
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
