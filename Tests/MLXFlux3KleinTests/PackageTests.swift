// Offline manifest / configuration / materialization tests for the FLUX 3 Image Klein engine package.
import Flux3Klein
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXFlux3Klein

final class Flux3KleinPackageTests: XCTestCase {
    static let weights = "/Volumes/Satechi/Development/mlxengine-image/weights/flux-3-image-klein"
    static let classifier = "/Volumes/Satechi/Development/mlxengine-image/weights/nsfw-classifier"

    /// C7: the FLUX Kommunity License, package-local and NOT allowlisted (AB-D-0151); C8: MIT.
    func testManifestIsKommunityAndNotPermissive() {
        let m = Flux3KleinPackage.manifest
        XCTAssertEqual(m.license.weightLicense, .fluxKommunity)
        XCTAssertEqual(m.license.weightLicense.identifier, "LicenseRef-FLUX-Kommunity-1.0")
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.contractVersion, ContractVersion.current)
        XCTAssertEqual(Set(m.surfaces.map(\.capability)), [.textToImage, .imageEdit])
        XCTAssertTrue(m.surfaces.allSatisfy { $0.name == "flux-3-klein" })
        XCTAssertTrue(m.surfaces.allSatisfy { $0.summary.contains("FULL SCENE CAPTION") })
        XCTAssertEqual(m.provenance.sourceRepo, "mlx-community/flux-3-image-klein-bf16")
    }

    func testFootprintsCoverEveryTier() {
        let fp = Flux3KleinPackage.manifest.requirements.footprints
        XCTAssertEqual(Set(fp.map(\.quant)), [.bf16, .int8, .int4])
        XCTAssertTrue(fp.allSatisfy { $0.residentBytes > 0 && $0.peakActivationBytes > 0 })
        let r = Dictionary(uniqueKeysWithValues: fp.map { ($0.quant, $0.residentBytes) })
        XCTAssertGreaterThan(r[.bf16]!, r[.int8]!)
        XCTAssertGreaterThan(r[.int8]!, r[.int4]!)
    }

    func testEnvelope() {
        XCTAssertNil(Flux3KleinEnvelope.violation(width: 1024, height: 1024, referenceCount: 0))
        XCTAssertNil(Flux3KleinEnvelope.violation(width: nil, height: nil, referenceCount: 1))
        XCTAssertNil(Flux3KleinEnvelope.violation(width: 1024, height: 1024, referenceCount: 10))
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 1024, height: 1024, referenceCount: 11))
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 1000, height: 1024, referenceCount: 0))  // not ×16
        XCTAssertNil(Flux3KleinEnvelope.violation(width: 2048, height: 2048, referenceCount: 10))  // measured, AB-R-0579
        XCTAssertNil(Flux3KleinEnvelope.violation(width: 1440, height: 1920, referenceCount: 1))
        XCTAssertNil(Flux3KleinEnvelope.violation(width: 1536, height: 2720, referenceCount: 0))  // 4.18 MP, non-square
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 2064, height: 2048, referenceCount: 0))
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 1024, height: nil, referenceCount: 0))
    }

    /// v0.4.0 (row-band decode and reference encode): output and a single reference both up to 2048², the reference's
    /// own 4 MiP.
    func testEnvelopeCaps() {
        XCTAssertEqual(Flux3KleinEnvelope.maxOutputPixels, 2048 * 2048)
        XCTAssertEqual(Flux3KleinEnvelope.maxReferencePixels, 2048 * 2048)
        XCTAssertEqual(Flux3KleinEnvelope.maxReferencePixels, Flux3References.matchMaxPixels)
    }

    /// The declaration keeps a 24 GB Mac's budget (19.07 GB, recommendedMaxWorkingSetSize on an M5 Pro, AB-R-0551) on
    /// the int4 tier, and bf16 under 32 GB.
    func testDeclaredFitsTheTargetMacs() {
        let fp = Dictionary(uniqueKeysWithValues: Flux3KleinFootprints.declared.map { ($0.quant, $0) })
        let total = { (q: Quant) in fp[q]!.residentBytes + fp[q]!.peakActivationBytes }
        XCTAssertLessThanOrEqual(total(.int4), 19_070_000_000)
        XCTAssertLessThanOrEqual(total(.bf16), 32_000_000_000)
        XCTAssertGreaterThanOrEqual(fp[.bf16]!.peakActivationBytes, 11_340_000_000)  // the measured worst request
    }

    /// A single reference is fitted to the envelope, so an un-sized edit never leaves it.
    func testSingleReferenceFitsTheEnvelope() {
        let (_, canvas) = Flux3References.canvas(width: 4000, height: 3000, maxPixels: Flux3KleinEnvelope.maxReferencePixels)
        XCTAssertLessThanOrEqual(canvas.0 * canvas.1, Flux3KleinEnvelope.maxReferencePixels)
        XCTAssertNil(Flux3KleinEnvelope.violation(width: canvas.0, height: canvas.1, referenceCount: 1))
        XCTAssertEqual(canvas.0 % 16, 0)
        XCTAssertEqual(canvas.1 % 16, 0)
    }

    func testWeightSourceIsThePinnedMirror() {
        let s = Flux3KleinConfiguration(upsampler: .off).weightSources
        XCTAssertEqual(s.map(\.repo), ["mlx-community/flux-3-image-klein-bf16"])
        XCTAssertEqual(s.first?.revision?.count, 40)  // a commit, never a moving "main"
        XCTAssertTrue(s.first?.matching?.contains("NOTICE.md") ?? false)  // §3: the notice travels with the weights
    }

    /// The upsampler is on by default (int8), from its own pinned mirror; `.off` drops the source.
    func testUpsamplerSources() {
        let s = Flux3KleinConfiguration().weightSources
        XCTAssertEqual(s.map(\.role), ["snapshot", "upsampler"])
        XCTAssertEqual(s.last?.repo, "mlx-community/flux-3-image-klein-pup-8bit")
        for tier in Flux3UpsamplerTier.allCases where tier != .off {
            XCTAssertEqual(tier.revision?.count, 40, "\(tier) is pinned to a commit")
        }
        XCTAssertNil(Flux3UpsamplerTier.off.repo)
        // Only the bf16 upsampler's transient exceeds the declared activation.
        XCTAssertNil(Flux3KleinConfiguration(upsampler: .int8).peakActivationBytesHint)
        XCTAssertNil(Flux3KleinConfiguration(upsampler: .int4).peakActivationBytesHint)
        XCTAssertGreaterThan(Flux3KleinConfiguration(upsampler: .bf16).peakActivationBytesHint!,
                             Flux3KleinFootprints.declared.map(\.peakActivationBytes).max()!)
    }

    /// BFL's safety checks are off by default (as upstream); on, they add the pinned NSFW classifier as a weight source
    /// and its 0.35 GB to the resident term.
    func testSafetyChecksOffByDefault() throws {
        let off = Flux3KleinConfiguration()
        XCTAssertFalse(off.safetyChecks)
        XCTAssertFalse(off.weightSources.contains { $0.role == "safety" })
        XCTAssertNil(off.resolvedSafetyDirectory(storeRoot: nil))
        let on = Flux3KleinConfiguration(safetyChecks: true)
        let source = try XCTUnwrap(on.weightSources.last)
        XCTAssertEqual(source.role, "safety")
        XCTAssertEqual(source.repo, "Falconsai/nsfw_image_detection")
        XCTAssertEqual(source.revision?.count, 40)
        XCTAssertTrue(source.matching?.contains("model.safetensors") ?? false)
        let bf16 = Flux3KleinFootprints.declared.first { $0.quant == .bf16 }!.residentBytes
        XCTAssertEqual(on.residentBytesHint, bf16 + Flux3KleinFootprints.safetyClassifierBytes)
        let old = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: Data(#"{"quant":"bf16"}"#.utf8))
        XCTAssertFalse(old.safetyChecks, "configurations saved before P6 keep the checks off")
        let back = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: JSONEncoder().encode(on))
        XCTAssertTrue(back.safetyChecks)
    }

    func testMATGateWithSafety() {
        let haveLocal = FileManager.default.fileExists(atPath: Self.weights + "/model.safetensors")
            && FileManager.default.fileExists(atPath: Self.classifier + "/model.safetensors")
        let report = MaterializationConformance.check(
            freshConfiguration: Flux3KleinConfiguration(safetyChecks: true),
            satisfiedConfiguration: haveLocal
                ? Flux3KleinConfiguration(snapshotPath: Self.weights, upsamplerPath: Self.weights + "-pup",
                                          safetyChecks: true, safetyClassifierPath: Self.classifier) : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testRefusalMessage() {
        let e = Flux3KleinPackageError.contentRefused("Reference image 2 may show nudity or other adult content.")
        XCTAssertEqual(e.errorDescription,
                       "FLUX 3 Image Klein safety check refused the request: Reference image 2 may show nudity or other adult content.")
        XCTAssertEqual("\(e)", e.errorDescription, "an engine that interpolates the error shows the sentence")
    }

    func testMetaDataLevers() throws {
        XCTAssertNil(try Flux3KleinMetaKeys.upsample([:]))
        XCTAssertEqual(try Flux3KleinMetaKeys.upsample(["upsample": .bool(false)]), false)
        XCTAssertThrowsError(try Flux3KleinMetaKeys.upsample(["upsample": .string("no")]))
        let boxes = try Flux3KleinMetaKeys.boxes(["boxes": .array([
            .object(["id": .string("sun"), "bbox": .array([.double(0.1), .double(0.2), .int(0), .double(0.5)]),
                     "desc": .string("a low sun")]),
            .object(["from": .string("ref_image_0"), "src_bbox": .array([.double(0.5), .double(0.5), .double(0.9), .double(0.9)]),
                     "tgt_bbox": .null]),
        ])])
        XCTAssertEqual(boxes?.count, 2)
        XCTAssertEqual(boxes?.first?.id, "sun")
        XCTAssertEqual(boxes?.last?.from, "ref_image_0")
        XCTAssertNil(boxes?.last?.tgtBBox)
        XCTAssertThrowsError(try Flux3KleinMetaKeys.boxes(["boxes": .string("x")]))
    }

    func testMATGate() {
        let haveLocal = FileManager.default.fileExists(atPath: Self.weights + "/model.safetensors")
        let report = MaterializationConformance.check(
            freshConfiguration: Flux3KleinConfiguration(),
            satisfiedConfiguration: haveLocal
                ? Flux3KleinConfiguration(snapshotPath: Self.weights, upsamplerPath: Self.weights + "-pup") : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testConfigurationRoundTrips() throws {
        let cfg = Flux3KleinConfiguration(quant: .int4, textEncoderQuant: .int8, snapshotPath: "/tmp/x", defaultSteps: 12,
                                          upsampler: .int4)
        let back = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(back.quant, .int4)
        XCTAssertEqual(back.textEncoderQuant, .int8)
        XCTAssertEqual(back.snapshotPath, "/tmp/x")
        XCTAssertEqual(back.defaultSteps, 12)
        XCTAssertFalse(back.keepEncoderResident)
        XCTAssertEqual(back.upsampler, .int4)
        let minimal = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: Data(#"{"quant":"int8"}"#.utf8))
        XCTAssertEqual(minimal.textEncoderQuant, .bf16)
        XCTAssertEqual(minimal.defaultSteps, 16)
        XCTAssertEqual(minimal.upsampler, .int8, "configurations saved before P4 get the default upsampler")
    }

    /// A per-request encoder stays under the declared activation (so no hint); a resident one moves into the
    /// resident term.
    func testEncoderHints() {
        XCTAssertNil(Flux3KleinConfiguration().peakActivationBytesHint)
        XCTAssertNil(Flux3KleinConfiguration(textEncoderQuant: .int8).peakActivationBytesHint)
        XCTAssertNil(Flux3KleinConfiguration().residentBytesHint)
        XCTAssertLessThan(Flux3KleinFootprints.encoderBytes(.bf16),
                          Flux3KleinFootprints.declared.map(\.peakActivationBytes).min()!)
        let resident = Flux3KleinConfiguration(quant: .int8, keepEncoderResident: true)
        XCTAssertEqual(resident.residentBytesHint,
                       Flux3KleinFootprints.declared.first { $0.quant == .int8 }!.residentBytes
                           + Flux3KleinFootprints.encoderBytes(.bf16))
    }
}
