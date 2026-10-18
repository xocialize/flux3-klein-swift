// CAN gate (offline, no kernels, no weights). CAN-1/2: run() pre-cancelled on both surfaces; the entry checkpoint is
// the first act of run(), before notLoaded. CAN-3: the cadence of record is once per denoise step (the pipeline's
// progress callback throws on cancellation), with post-encode and post-denoise seams around it.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXFlux3Klein

final class Flux3KleinCancellationTests: XCTestCase {
    func testCANGatePreCancelledTextToImage() async {
        let package = Flux3KleinPackage(configuration: Flux3KleinConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledImageEdit() async {
        let package = Flux3KleinPackage(configuration: Flux3KleinConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package, request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Flux3KleinPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: Flux3KleinPackage.manifest, posture: .cadence([.init(phase: .denoise, unit: .step)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
