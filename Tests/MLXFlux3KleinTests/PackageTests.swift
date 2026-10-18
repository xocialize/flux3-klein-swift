// Offline manifest / configuration / materialization tests for the FLUX 3 Image Klein engine package.
import Flux3Klein
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXFlux3Klein

final class Flux3KleinPackageTests: XCTestCase {
    static let weights = "/Volumes/Satechi/Development/mlxengine-image/weights/flux-3-image-klein"

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
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 2048, height: 2048, referenceCount: 0))
        XCTAssertNotNil(Flux3KleinEnvelope.violation(width: 1024, height: nil, referenceCount: 0))
    }

    /// A single reference is fitted to the envelope, so an un-sized edit never leaves it.
    func testSingleReferenceFitsTheEnvelope() {
        let (_, canvas) = Flux3References.canvas(width: 4000, height: 3000, maxPixels: Flux3KleinEnvelope.maxOutputPixels)
        XCTAssertLessThanOrEqual(canvas.0 * canvas.1, Flux3KleinEnvelope.maxOutputPixels)
        XCTAssertEqual(canvas.0 % 16, 0)
        XCTAssertEqual(canvas.1 % 16, 0)
    }

    func testWeightSourceIsThePinnedMirror() {
        let s = Flux3KleinConfiguration().weightSources
        XCTAssertEqual(s.map(\.repo), ["mlx-community/flux-3-image-klein-bf16"])
        XCTAssertEqual(s.first?.revision?.count, 40)  // a commit, never a moving "main"
        XCTAssertTrue(s.first?.matching?.contains("NOTICE.md") ?? false)  // §3: the notice travels with the weights
    }

    func testMATGate() {
        let haveLocal = FileManager.default.fileExists(atPath: Self.weights + "/model.safetensors")
        let report = MaterializationConformance.check(
            freshConfiguration: Flux3KleinConfiguration(),
            satisfiedConfiguration: haveLocal ? Flux3KleinConfiguration(snapshotPath: Self.weights) : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testConfigurationRoundTrips() throws {
        let cfg = Flux3KleinConfiguration(quant: .int4, textEncoderQuant: .int8, snapshotPath: "/tmp/x", defaultSteps: 12)
        let back = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(back.quant, .int4)
        XCTAssertEqual(back.textEncoderQuant, .int8)
        XCTAssertEqual(back.snapshotPath, "/tmp/x")
        XCTAssertEqual(back.defaultSteps, 12)
        XCTAssertFalse(back.keepEncoderResident)
        let minimal = try JSONDecoder().decode(Flux3KleinConfiguration.self, from: Data(#"{"quant":"int8"}"#.utf8))
        XCTAssertEqual(minimal.textEncoderQuant, .bf16)
        XCTAssertEqual(minimal.defaultSteps, 16)
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
