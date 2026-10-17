// Flow-map sampling for the distilled checkpoint (reference `sampling.py`): Euler over a warped linspace, the DiT
// told where each step lands (`timesteps_ref`), no CFG.

import Foundation
import MLX

public enum Flux3Schedule {
    /// The checkpoint's `config.json` `sampler.alpha`. The reference CLI's `Pipeline.generate` defaults to 16.0 and
    /// never reads the config, but only 6.93 reproduces BFL's own example renders (AB-R-0571).
    public static let defaultAlpha: Double = 6.93
    public static let defaultSteps = 16

    /// `timesteps`: t' = α·t / (1 + (α − 1)·t) over linspace(1, 0, steps + 1). Computed in Double; the reference
    /// computes in fp32 and calls `.item()`, so values agree to fp32 rounding.
    public static func timesteps(steps: Int = defaultSteps, alpha: Double = defaultAlpha) -> [Float] {
        (0...steps).map { i in
            let t = Float(1.0 - Double(i) / Double(steps))  // torch.linspace in fp32
            let tf = Double(t)
            return Float(alpha * tf / (1.0 + (alpha - 1.0) * tf))
        }
    }

    /// One flow-map Euler update: x + (t_next − t) · velocity, with x kept in fp32 across steps (the reference keeps
    /// the image tokens in their bf16-rounded-noise fp32 state).
    public static func step(_ x: MLXArray, velocity: MLXArray, t: Float, tNext: Float) -> MLXArray {
        x + (tNext - t) * velocity.asType(.float32)
    }
}
