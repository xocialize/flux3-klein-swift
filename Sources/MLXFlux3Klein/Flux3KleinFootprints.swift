// Declared memory footprints (efficiency contract: resident + peak activation), MEASURED with
// `Flux3KleinGate --oneshot … --drop-te` (outputs/envelope.sh, outputs/envelope-v030.log + envelope-v040.log): M5 Max, mlx-swift 0.32.3,
// Release, MLX pool capped at the engine's 2 GB (.automatic), 2026-10-17. This is the package's own memory path: DiT
// (bf16 / quantized at load) plus the fp32 VAE encoder and decoder resident; per request the text-only Qwen3-VL-4B
// encoder loads, encodes and is released, then the references are encoded one at a time, then denoise and decode.
//
// v0.3.0: the VAE decodes in row bands (flux2-vae-mlx-swift 0.6.0, AB-R-0577) and each reference is evaluated as it
// is encoded, so ten encoder transients no longer stack into the first DiT step. Activation above resident, GB
// (v0.2.0 → v0.3.0, bf16; int4 is the same to 0.01):
//
//   output       T2I            1 ref (own size)    10 refs (1024²)
//   1024²        11.27 → 2.34   11.27 → 4.29         14.81 → 8.11
//   1152²        14.23 → 2.76   14.23 → 5.43         14.86 → 8.31
//   1536²        25.28 → 4.38   25.28 → 9.66                 9.03
//   1440×1920    29.62 → 5.01          11.34 (1664²)         9.31
//   2048²        45.02 → 7.21          18.66 (2048²)        10.32
//                                      11.34 (1664² ref → 2048² out)
//
// v0.4.0: the reference ENCODE runs in row bands too (flux2-vae-mlx-swift 0.7.0, AB-R-0580); it was ~4.1 GB per
// reference MP untiled. A single reference now costs what the output does (envelope-v040.log, bf16; int4 equal):
//
//   1 ref (own size)   1024² 4.29 → 2.35   1152² 5.43 → 2.77   1536² 9.66 → 4.38   1664² → 1440×1920 11.34 → 5.01
//                      2048² 18.66 → 7.22  (2048² ref → 1024² out: 4.62)            10 refs → 2048²: 10.32 (unchanged)
//
// So a single reference follows the reference's own cap (4 MiP = 2048²) and the envelope is output ≤ 2048² area
// (4.19 MP), a single reference fitted to 2048², up to 10 references. The worst request inside it is ten references
// into 2048², 10.32 GB; the encoder transient (8.07 GB bf16, 4.72 GB int8) stays under it.
//
//   DiT tier   resident   encode phase (+ encoder)   worst request in the envelope
//   bf16       11.44 GB   19.51 GB (+8.07)           22.78 GB
//   int8        6.49 GB   14.57 GB                   (17.83 GB, resident + the same activation)
//   int4        4.67 GB    9.39 GB (int8 encoder)    16.01 GB
//
// Declared (AB-R-0579, unchanged in v0.4.0): resident rounded up about 2–5 %, activation 12.0 GB (+16 % over the
// 10.32 GB worst; was 15.5 GB at the 1152² cap). bf16 declares 23.7 GB, int4 16.9 GB — inside a 24 GB Mac's 19.07 GB
// budget.
// [MLX-pool numbers. phys_footprint peaks (v0.4.0): bf16 22.8 GB (10 refs into 2048²), 20.0 GB (a 2048² reference);
// int4 16.0 GB, with a 15.0 GB floor set by quantize-on-load. Was 24.5 / 17.8 GB with the 1664² cap (v0.3.0).]

import Flux3Klein
import MLXToolKit

enum Flux3KleinFootprints {
    /// Output area the footprints were measured at (Flux3KleinEnvelope).
    static let envelopeMaxPixels = 2048 * 2048
    /// Area a single reference is fitted to: the reference's own 4 MiP, now that its VAE encode runs in row bands.
    static let envelopeReferenceMaxPixels = 2048 * 2048

    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 11_700_000_000, peakActivationBytes: 12_000_000_000),
        QuantFootprint(quant: .int8, residentBytes: 6_700_000_000, peakActivationBytes: 12_000_000_000),
        QuantFootprint(quant: .int4, residentBytes: 4_900_000_000, peakActivationBytes: 12_000_000_000),
    ]

    /// The text-only conditioner's weights, added to the resident term when it is kept resident.
    /// The bf16 prompt upsampler's per-request transient (20.16 GB MLX peak, pool capped at 2 GB, P4), declared +3.7 %.
    /// int8 (11.06 GB) and int4 (6.79 GB) stay under the declared activation.
    static let bf16UpsamplerPeakActivationBytes: UInt64 = 20_900_000_000

    static func encoderBytes(_ quant: Flux3TextEncoderQuant) -> UInt64 {
        quant == .int8 ? 4_900_000_000 : 8_300_000_000
    }
}
