// Declared memory footprints (efficiency contract: resident + peak activation), MEASURED with
// `Flux3KleinGate --oneshot … --drop-te` (outputs/envelope.sh, outputs/envelope-v030.log): M5 Max, mlx-swift 0.32.3,
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
// The decode now costs ~1.7 GB per output MP. The largest term left is the untiled VAE ENCODE of a single reference,
// ~4.1 GB per reference MP, so the envelope caps the two separately: output ≤ 2048² area (4.19 MP), a single
// reference fitted to 1664² (2.77 MP). The worst request inside it is 11.34 GB (a 1664² reference, any output up to
// 2048²); ten references into 2048² is 10.32. The encoder transient (8.07 GB bf16, 4.72 GB int8) stays under it.
//
//   DiT tier   resident   encode phase (+ encoder)   worst request in the envelope
//   bf16       11.44 GB   19.51 GB (+8.07)           22.78 GB
//   int8        6.49 GB   14.57 GB                   (17.83 GB, resident + the same activation)
//   int4        4.67 GB    9.39 GB (int8 encoder)    16.01 GB
//
// Declared (AB-R-0579): resident rounded up about 2–5 %, activation 12.0 GB (+5.8 %; was 15.5 GB at the 1152² cap). bf16 declares
// 23.7 GB, int4 16.9 GB — inside a 24 GB Mac's 19.07 GB budget.
// [MLX-pool numbers. phys_footprint peaks: bf16 24.5 GB (1664² reference), 22.8 GB (10 refs into 2048²), 20.1 GB
// (2048² T2I); int4 17.8 GB, with a 15.0 GB floor set by quantize-on-load. Was 28.2 / 21.4 GB at the 1152² cap.]

import Flux3Klein
import MLXToolKit

enum Flux3KleinFootprints {
    /// Output area the footprints were measured at (Flux3KleinEnvelope).
    static let envelopeMaxPixels = 2048 * 2048
    /// Area a single reference is fitted to: its fp32 VAE encode is untiled (~4.1 GB per MP).
    static let envelopeReferenceMaxPixels = 1664 * 1664

    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 11_700_000_000, peakActivationBytes: 12_000_000_000),
        QuantFootprint(quant: .int8, residentBytes: 6_700_000_000, peakActivationBytes: 12_000_000_000),
        QuantFootprint(quant: .int4, residentBytes: 4_900_000_000, peakActivationBytes: 12_000_000_000),
    ]

    /// The text-only conditioner's weights, added to the resident term when it is kept resident.
    static func encoderBytes(_ quant: Flux3TextEncoderQuant) -> UInt64 {
        quant == .int8 ? 4_900_000_000 : 8_300_000_000
    }
}
