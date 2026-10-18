// Declared memory footprints (efficiency contract: resident + peak activation), MEASURED with
// `Flux3KleinGate --oneshot … --drop-te` (outputs/envelope.sh): M5 Max, mlx-swift 0.32.3, Release, MLX pool capped at
// the engine's 2 GB (.automatic), 2026-10-17. This is the package's own memory path: DiT (bf16 / quantized at load)
// plus the fp32 VAE encoder and decoder resident; per request the text-only Qwen3-VL-4B encoder loads, encodes and is
// released, then denoise and decode.
//
//   DiT tier   resident   encode phase (+ encoder)   peak at the 1152² cap (worst: 10 refs)
//   bf16       11.44 GB   19.51 GB (+8.07)           26.30 GB (activation 14.86)
//   int8        6.49 GB   14.57 GB                   21.31 GB at 1024² / 10 refs
//   int4        4.67 GB    9.39 GB (int8 encoder)    19.52 GB (activation 14.85)
//
// Activation above resident, bf16, by request (GB): 1024² 11.27 (T2I, 1 ref and 4 refs alike) · 1024² + 10 refs
// 14.81 · 1152² 14.23 · 1152² + 10 refs 14.86 · 1536² 25.28 · 1440×1920 29.62 · 2048² 45.02. It is mostly the fp32
// VAE decode (about 10.7 GB per output MP), so the envelope caps the output at 1152² area (1.33 MP: 1024², 1184×896,
// 1216×832, 896×1120…). A single reference is fitted to the same area. The encoder transient (8.07 GB bf16,
// 4.72 GB int8) stays under the activation at every size, so the encoder tier never sets the peak; it matters only
// when kept resident.
//
// Declared: resident rounded up about 2–5 %, activation 15.5 GB (+4.3 %).
// [MLX-pool numbers. phys_footprint peaks at the cap: 26.6 GB T2I, 28.2 GB with 10 refs (bf16); 21.4 GB int4. The
// in-app re-baseline is owed (AB-T-0019 pattern).]

import Flux3Klein
import MLXToolKit

enum Flux3KleinFootprints {
    /// Output area the footprints were measured at (Flux3KleinEnvelope).
    static let envelopeMaxPixels = 1152 * 1152

    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 11_700_000_000, peakActivationBytes: 15_500_000_000),
        QuantFootprint(quant: .int8, residentBytes: 6_700_000_000, peakActivationBytes: 15_500_000_000),
        QuantFootprint(quant: .int4, residentBytes: 4_900_000_000, peakActivationBytes: 15_500_000_000),
    ]

    /// The text-only conditioner's weights, added to the resident term when it is kept resident.
    static func encoderBytes(_ quant: Flux3TextEncoderQuant) -> UInt64 {
        quant == .int8 ? 4_900_000_000 : 8_300_000_000
    }
}
