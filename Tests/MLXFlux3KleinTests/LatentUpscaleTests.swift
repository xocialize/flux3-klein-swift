// metaData["latentUpscale"] (AB-A-0257): parsing, the draw limit, the weight source, workload units, the footprint line.

import Foundation
import Flux3Klein
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXFlux3Klein

final class LatentUpscaleTests: XCTestCase {
    func testLever() throws {
        XCTAssertEqual(try Flux3KleinMetaKeys.latentUpscale([:]), 0)
        XCTAssertEqual(try Flux3KleinMetaKeys.latentUpscale(["latentUpscale": .int(1)]), 1)
        XCTAssertEqual(try Flux3KleinMetaKeys.latentUpscale(["latentUpscale": .double(2)]), 2)
        for bad: MetaValue in [.int(3), .int(-1), .double(1.5), .bool(true)] {
            XCTAssertThrowsError(try Flux3KleinMetaKeys.latentUpscale(["latentUpscale": bad]), "\(bad)")
        }
    }

    func testDrawLimit() {
        XCTAssertNil(Flux3KleinEnvelope.latentUpscaleViolation(width: 1024, height: 1024, passes: 2))
        XCTAssertNil(Flux3KleinEnvelope.latentUpscaleViolation(width: 2048, height: 2048, passes: 0))
        XCTAssertNil(Flux3KleinEnvelope.latentUpscaleViolation(width: nil, height: nil, passes: 1))
        XCTAssertNotNil(Flux3KleinEnvelope.latentUpscaleViolation(width: 1280, height: 1024, passes: 1))
    }

    func testWeightSource() {
        let on = Flux3KleinConfiguration(upsampler: .off)
        XCTAssertEqual(on.weightSources.map(\.role), ["snapshot", "latent-upscaler"])
        XCTAssertEqual(on.weightSources.last?.repo, "mlx-community/FlowUpscaler-bf16")
        let root = FileManager.default.temporaryDirectory.appending(path: "f3-up-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(on.missingWeightSources(storeRoot: root).contains { $0.role == "latent-upscaler" })
        let dir = ModelStore(root: root).directory(for: Flux3KleinConfiguration.latentUpscalerRepo)!
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: dir.appending(path: "flow_upscaler.safetensors").path, contents: Data())
        XCTAssertFalse(on.missingWeightSources(storeRoot: root).contains { $0.role == "latent-upscaler" })
    }

    func testWorkloadUnits() {
        let c = Flux3KleinConfiguration()
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x")), 1024 * 1024)
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x", width: 1024, height: 768,
                                                        metaData: ["latentUpscale": .int(2)])), 4096 * 3072)
        XCTAssertNil(c.workloadUnits(for: IEditRequest(images: [], prompt: "x")))
        XCTAssertEqual(c.workloadUnits(for: IEditRequest(images: [], prompt: "x",
                                                          metaData: ["latentUpscale": .int(1)])), 2048 * 2048)
    }

    func testFootprintLine() {
        for f in Flux3KleinFootprints.declared {
            let line = try! XCTUnwrap(f.activationScaling)
            XCTAssertEqual(line.axis, .outputPixels)
            XCTAssertEqual(line.measuredCeiling, 4096 * 4096)
            // Above every measured native point (AB-R-0579/0580) and the 4096² upscale (28.77 GB over declared resident).
            for (pixels, gb) in [(1024 * 1024, 8.11), (1152 * 1152, 8.31), (1536 * 1536, 9.03), (1440 * 1920, 9.31),
                                 (2048 * 2048, 10.32), (4096 * 4096, 28.77)] {
                XCTAssertGreaterThanOrEqual(Double(line.projectedBytes(at: Double(pixels))), gb * 1e9, "\(pixels)")
            }
        }
        // Native 2048² on int4 still fits a 24 GB Mac's 19.07 GB budget.
        let int4 = Flux3KleinFootprints.declared.first { $0.quant == .int4 }!
        XCTAssertLessThan(int4.residentBytes + int4.activationScaling!.projectedBytes(at: 2048 * 2048), 19_070_000_000)
        for c in [Flux3KleinConfiguration(quant: .int4), Flux3KleinConfiguration(upsampler: .bf16)] {
            let report = FootprintConformance.check(manifest: Flux3KleinPackage.manifest, configuration: c)
            XCTAssertTrue(report.passed, report.summary)
        }
    }
}
