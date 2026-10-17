// FLUX 3 Image Klein denoising transformer — Swift/MLX port.
//
// Isomorphic to BFL's reference `src/flux3/dit.py` (github.com/black-forest-labs/flux-3-image-klein @ bfc8889):
// three token modes — `image` (the target latents), `image_cond` (reference latents) and `text` — each run through
// its OWN stack of 5 blocks attending only within the mode, then are packed per sample in mode order
// [image, image_cond, text] and run through 28 shared single-stream blocks with PER-TOKEN modulation (each token
// keeps its mode's shift/scale/gate). One block = LayerNorm modulation → q/k/v + QK-RMSNorm fused with 4-axis RoPE →
// attention, in parallel with a SwiGLU MLP (×3), one gate on the sum. No biases anywhere.
//
// Flow map (the distilled checkpoint): every mode's conditioning vector is time_in(t_mode) + ref_time_in(t_next),
// with t_mode = t for `image` and 0 for `image_cond` / `text`, and t_next the time the step jumps to.
//
// Single-sample inference only (the reference's per-sample `packed_attention` bounds collapse to one full
// attention), and an empty mode is skipped instead of carrying the reference's never-attended phantom token.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum Flux3KleinError: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .loading(let m): return "Flux3Klein loading error: \(m)"
        case .invalidInput(let m): return "Flux3Klein input error: \(m)"
        }
    }
}

/// `config.json` `dit` block of the distilled checkpoint.
public struct Flux3DiTConfig: Sendable {
    public var hidden = 3072
    public var heads = 24
    public var depth = 5
    public var depthSingle = 28
    public var mlpRatio: Float = 3.0
    public var axesDim = [32, 32, 32, 32]
    public var theta: Double = 10000
    public var imageChannels = 128
    public var textChannels = 20480
    public var timestepChannels = 256

    public var headDim: Int { hidden / heads }
    public var mlpHidden: Int { Int(Float(hidden) * mlpRatio) }
    public init() {}
}

// MARK: - Timestep embedding

/// `timestep_embedding`: t·1000, freqs exp(−ln(10000)·i/half), [cos | sin], computed in fp32 and cast to `dtype`
/// (the reference returns `.to(t)`, i.e. the timesteps' own dtype — bf16 in the pipeline).
func flux3TimestepEmbedding(_ t: MLXArray, dim: Int = 256, maxPeriod: Float = 10000, dtype: DType) -> MLXArray {
    let half = dim / 2
    let freqs = exp(-log(maxPeriod) * MLXArray(0..<half).asType(.float32) / Float(half))
    let args = (t.asType(.float32) * 1000)[0..., .newAxis] * freqs[.newAxis, 0...]
    return concatenated([cos(args), sin(args)], axis: -1).asType(dtype)
}

/// `MLPEmbedder`: out_layer(silu(in_layer(x))), bias-free.
public final class Flux3MLPEmbedder: Module {
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear

    public init(inDim: Int, hidden: Int) {
        self._inLayer.wrappedValue = Linear(inDim, hidden, bias: false)
        self._outLayer.wrappedValue = Linear(hidden, hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { outLayer(silu(inLayer(x))) }
}

// MARK: - Modulation

public struct Flux3Mod {
    public var shift: MLXArray
    public var scale: MLXArray
    public var gate: MLXArray
}

/// `ModeModulation`: lin(silu(vec)) → (shift, scale, gate).
public final class Flux3ModeModulation: Module {
    @ModuleInfo(key: "lin") var lin: Linear

    public init(hidden: Int) {
        self._lin.wrappedValue = Linear(hidden, 3 * hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ vec: MLXArray) -> Flux3Mod {
        let p = split(lin(silu(vec)), parts: 3, axis: -1)
        return Flux3Mod(shift: p[0], scale: p[1], gate: p[2])
    }
}

/// The three per-mode modulations of one stream (`early_stream_modulations` / `single_stream_modulations`).
public final class Flux3ModeModulations: Module {
    @ModuleInfo(key: "image") var image: Flux3ModeModulation
    @ModuleInfo(key: "image_cond") var imageCond: Flux3ModeModulation
    @ModuleInfo(key: "txt") var text: Flux3ModeModulation

    public init(hidden: Int) {
        self._image.wrappedValue = Flux3ModeModulation(hidden: hidden)
        self._imageCond.wrappedValue = Flux3ModeModulation(hidden: hidden)
        self._text.wrappedValue = Flux3ModeModulation(hidden: hidden)
        super.init()
    }

    func callAsFunction(_ mode: Flux3Mode, _ vec: MLXArray) -> Flux3Mod {
        switch mode {
        case .image: return image(vec)
        case .imageCond: return imageCond(vec)
        case .text: return text(vec)
        }
    }
}

/// `layernorm_modulation`: (1 + scale)·LayerNorm(x, no affine, eps 1e-6) + shift. The norm and the modulation run in
/// fp32 (as torch autocast keeps `layer_norm` in fp32); the result goes back to the activation dtype.
func flux3LayerNormModulation(_ x: MLXArray, scale: MLXArray, shift: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let normed = MLXFast.layerNorm(xf, weight: nil, bias: nil, eps: 1e-6)
    return ((1 + scale.asType(.float32)) * normed + shift.asType(.float32)).asType(x.dtype)
}

// MARK: - RoPE

/// Rotation tables for a token list: cos/sin [L, headDim/2] in fp32. The reference builds the angles in float64
/// (`rope`: pos · θ^(−2k/dim) per axis, axes concatenated) and casts the rotation matrices to fp32; this does the same
/// on the CPU in Double, so long text positions keep full precision.
public struct Flux3Rope {
    public var cos: MLXArray
    public var sin: MLXArray

    public init(cos: MLXArray, sin: MLXArray) {
        self.cos = cos
        self.sin = sin
    }

    /// ids: [L, 4] positions (float; image h/w already scaled by `image_rope_scale`).
    public init(ids: MLXArray, axesDim: [Int], theta: Double) {
        let n = ids.dim(0)
        let pos = ids.asType(.float32).asArray(Float.self)
        let pairs = axesDim.reduce(0) { $0 + $1 / 2 }
        var c = [Float](repeating: 0, count: n * pairs)
        var s = [Float](repeating: 0, count: n * pairs)
        var inv: [(axis: Int, freq: Double)] = []
        for (axis, dim) in axesDim.enumerated() {
            for k in stride(from: 0, to: dim, by: 2) {
                inv.append((axis, 1.0 / pow(theta, Double(k) / Double(dim))))
            }
        }
        for i in 0..<n {
            for (j, f) in inv.enumerated() {
                let angle = Double(pos[i * 4 + f.axis]) * f.freq
                c[i * pairs + j] = Float(Foundation.cos(angle))
                s[i * pairs + j] = Float(Foundation.sin(angle))
            }
        }
        self.cos = MLXArray(c, [n, pairs])
        self.sin = MLXArray(s, [n, pairs])
    }

    static func concatenating(_ parts: [Flux3Rope]) -> Flux3Rope {
        parts.count == 1 ? parts[0]
            : Flux3Rope(cos: concatenated(parts.map(\.cos), axis: 0), sin: concatenated(parts.map(\.sin), axis: 0))
    }
}

/// `rmsnorm_rope`: fp32 RMSNorm over the head dim (eps 1e-6) times `scale`, then the interleaved-pair rotation
/// (x0, x1) → (cos·x0 − sin·x1, sin·x0 + cos·x1). x: [1, H, L, D].
func flux3RMSNormRope(_ x: MLXArray, scale: MLXArray, rope: Flux3Rope, eps: Float = 1e-6) -> MLXArray {
    let (b, h, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let xf = x.asType(.float32)
    let normed = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps) * scale.asType(.float32)
    let pairs = normed.reshaped(b, h, l, d / 2, 2)
    let x0 = pairs[.ellipsis, 0]
    let x1 = pairs[.ellipsis, 1]
    let c = rope.cos, s = rope.sin
    let rotated = stacked([c * x0 - s * x1, s * x0 + c * x1], axis: -1)
    return rotated.reshaped(b, h, l, d).asType(x.dtype)
}

// MARK: - Block

/// `RMSNorm` holding only `scale` (the math lives in `flux3RMSNormRope`).
public final class Flux3ScaleNorm: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    public init(dim: Int) {
        self._scale.wrappedValue = MLXArray.ones([dim])
        super.init()
    }
}

public final class Flux3QKNorm: Module {
    @ModuleInfo(key: "query_norm") var queryNorm: Flux3ScaleNorm
    @ModuleInfo(key: "key_norm") var keyNorm: Flux3ScaleNorm

    public init(dim: Int) {
        self._queryNorm.wrappedValue = Flux3ScaleNorm(dim: dim)
        self._keyNorm.wrappedValue = Flux3ScaleNorm(dim: dim)
        super.init()
    }
}

/// `Block`: x + gate · (attn_out(attention) + mlp_out(silu(gate_in) · up)) on one modulated input.
public final class Flux3Block: Module {
    let heads: Int
    let mlpHidden: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "attn_out") var attnOut: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear
    @ModuleInfo(key: "norm") var norm: Flux3QKNorm

    public init(hidden: Int, heads: Int, mlpHidden: Int) {
        self.heads = heads
        self.mlpHidden = mlpHidden
        self._qProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._vProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._mlpIn.wrappedValue = Linear(hidden, 2 * mlpHidden, bias: false)
        self._attnOut.wrappedValue = Linear(hidden, hidden, bias: false)
        self._mlpOut.wrappedValue = Linear(mlpHidden, hidden, bias: false)
        self._norm.wrappedValue = Flux3QKNorm(dim: hidden / heads)
        super.init()
    }

    /// x: [L, hidden]; mod fields broadcast against x ([1, hidden] per mode, or [L, hidden] per token).
    public func callAsFunction(_ x: MLXArray, rope: Flux3Rope, mod: Flux3Mod) -> MLXArray {
        let l = x.dim(0)
        let xMod = flux3LayerNormModulation(x, scale: mod.scale, shift: mod.shift)
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped(1, l, self.heads, -1).transposed(0, 2, 1, 3) }
        let q = flux3RMSNormRope(heads(qProj(xMod)), scale: norm.queryNorm.scale, rope: rope)
        let k = flux3RMSNormRope(heads(kProj(xMod)), scale: norm.keyNorm.scale, rope: rope)
        let v = heads(vProj(xMod)).asType(q.dtype)
        let headDim = q.dim(3)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / sqrt(Float(headDim)), mask: .none)
            .transposed(0, 2, 1, 3).reshaped(l, -1)
        let gu = mlpIn(xMod)
        let gateIn = gu[0..., ..<mlpHidden]
        let up = gu[0..., mlpHidden...]
        return x + mod.gate * (attnOut(attn) + mlpOut(silu(gateIn) * up))
    }
}

// MARK: - Containers keyed like the reference's ModuleDicts

public final class Flux3ContentEmbed: Module {
    @ModuleInfo(key: "image") var image: Linear
    @ModuleInfo(key: "image_cond") var imageCond: Linear

    public init(inChannels: Int, hidden: Int) {
        self._image.wrappedValue = Linear(inChannels, hidden, bias: false)
        self._imageCond.wrappedValue = Linear(inChannels, hidden, bias: false)
        super.init()
    }
}

public final class Flux3ContentBlocks: Module {
    @ModuleInfo(key: "image") var image: [Flux3Block]
    @ModuleInfo(key: "image_cond") var imageCond: [Flux3Block]

    public init(_ config: Flux3DiTConfig) {
        func stack() -> [Flux3Block] {
            (0..<config.depth).map { _ in
                Flux3Block(hidden: config.hidden, heads: config.heads, mlpHidden: config.mlpHidden)
            }
        }
        self._image.wrappedValue = stack()
        self._imageCond.wrappedValue = stack()
        super.init()
    }
}

/// `LastLayer`: linear(layernorm_modulation(x, scale, shift)) with (shift, scale) = adaLN(silu(vec)).
public final class Flux3LastLayer: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLN: Linear  // checkpoint `adaLN_modulation.1` (Sequential index)

    public init(hidden: Int, outChannels: Int) {
        self._linear.wrappedValue = Linear(hidden, outChannels, bias: false)
        self._adaLN.wrappedValue = Linear(hidden, 2 * hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, vec: MLXArray) -> MLXArray {
        let p = split(adaLN(silu(vec)), parts: 2, axis: -1)
        return linear(flux3LayerNormModulation(x, scale: p[1], shift: p[0]))
    }
}

public final class Flux3FinalLayers: Module {
    @ModuleInfo(key: "image") var image: Flux3LastLayer

    public init(hidden: Int, outChannels: Int) {
        self._image.wrappedValue = Flux3LastLayer(hidden: hidden, outChannels: outChannels)
        super.init()
    }
}

// MARK: - DiT

public enum Flux3Mode: String, CaseIterable, Sendable {
    case image, imageCond = "image_cond", text
}

/// One sample's token sets. ids are the reference's (t, h, w, l) positions as floats: image and reference ids with
/// h/w already scaled by `image_rope_scale` (0.5) and reference n on l = −(1 + n); text ids (0, 0, 0, position).
public struct Flux3Tokens {
    public var image: MLXArray
    public var imageIDs: MLXArray
    public var cond: MLXArray?
    public var condIDs: MLXArray?
    public var text: MLXArray
    public var textIDs: MLXArray

    public init(image: MLXArray, imageIDs: MLXArray, cond: MLXArray? = nil, condIDs: MLXArray? = nil,
                text: MLXArray, textIDs: MLXArray) {
        self.image = image
        self.imageIDs = imageIDs
        self.cond = cond
        self.condIDs = condIDs
        self.text = text
        self.textIDs = textIDs
    }
}

/// RoPE tables for the token sets, built once per generation (ids do not change across steps).
public struct Flux3RopeSet {
    public var image: Flux3Rope
    public var cond: Flux3Rope?
    public var text: Flux3Rope

    public init(_ tokens: Flux3Tokens, config: Flux3DiTConfig) {
        image = Flux3Rope(ids: tokens.imageIDs, axesDim: config.axesDim, theta: config.theta)
        cond = tokens.condIDs.map { Flux3Rope(ids: $0, axesDim: config.axesDim, theta: config.theta) }
        text = Flux3Rope(ids: tokens.textIDs, axesDim: config.axesDim, theta: config.theta)
    }
}

public final class Flux3Transformer: Module {
    public let config: Flux3DiTConfig
    @ModuleInfo(key: "emb_in") var embIn: Flux3ContentEmbed
    @ModuleInfo(key: "txt_in") var txtIn: Linear
    @ModuleInfo(key: "time_in") var timeIn: Flux3MLPEmbedder
    @ModuleInfo(key: "ref_time_in") var refTimeIn: Flux3MLPEmbedder
    @ModuleInfo(key: "early_stream_modulations") var earlyModulations: Flux3ModeModulations
    @ModuleInfo(key: "single_stream_modulations") var singleModulations: Flux3ModeModulations
    @ModuleInfo(key: "content_mode_blocks") var contentBlocks: Flux3ContentBlocks
    @ModuleInfo(key: "txt_mode_blocks") var textBlocks: [Flux3Block]
    @ModuleInfo(key: "single_blocks") var singleBlocks: [Flux3Block]
    @ModuleInfo(key: "final_layer") var finalLayer: Flux3FinalLayers

    /// Activation dtype (the weights' dtype); timesteps are rounded to it before the sinusoid, as in the reference.
    public var dtype: DType = .bfloat16

    public init(config: Flux3DiTConfig = Flux3DiTConfig()) {
        self.config = config
        let c = config
        self._embIn.wrappedValue = Flux3ContentEmbed(inChannels: c.imageChannels, hidden: c.hidden)
        self._txtIn.wrappedValue = Linear(c.textChannels, c.hidden, bias: false)
        self._timeIn.wrappedValue = Flux3MLPEmbedder(inDim: c.timestepChannels, hidden: c.hidden)
        self._refTimeIn.wrappedValue = Flux3MLPEmbedder(inDim: c.timestepChannels, hidden: c.hidden)
        self._earlyModulations.wrappedValue = Flux3ModeModulations(hidden: c.hidden)
        self._singleModulations.wrappedValue = Flux3ModeModulations(hidden: c.hidden)
        self._contentBlocks.wrappedValue = Flux3ContentBlocks(c)
        self._textBlocks.wrappedValue = (0..<c.depth).map { _ in
            Flux3Block(hidden: c.hidden, heads: c.heads, mlpHidden: c.mlpHidden)
        }
        self._singleBlocks.wrappedValue = (0..<c.depthSingle).map { _ in
            Flux3Block(hidden: c.hidden, heads: c.heads, mlpHidden: c.mlpHidden)
        }
        self._finalLayer.wrappedValue = Flux3FinalLayers(hidden: c.hidden, outChannels: c.imageChannels)
        super.init()
    }

    func embedTime(_ t: Float, _ embedder: Flux3MLPEmbedder) -> MLXArray {
        let ts = MLXArray([t]).asType(dtype)
        return embedder(flux3TimestepEmbedding(ts, dim: config.timestepChannels, dtype: dtype))
    }

    /// One flow-map velocity prediction for the image tokens. `t` is the current time, `tNext` the time this step
    /// jumps to (`timesteps_ref`). `capture` receives named intermediates (gate use; names match the oracle goldens).
    public func callAsFunction(_ tokens: Flux3Tokens, rope: Flux3RopeSet, t: Float, tNext: Float,
                               capture: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let ref = embedTime(tNext, refTimeIn)
        capture?("ref_time_in", ref)
        let timeZero = embedTime(0, timeIn)
        let vecImage = embedTime(t, timeIn) + ref
        let vecRest = timeZero + ref  // image_cond and text both sit at t = 0
        if let capture {
            capture("time_in", broadcast(timeZero, to: [tokens.text.dim(0), config.hidden]))
        }

        var xImage = embIn.image(tokens.image.asType(dtype))
        capture?("emb_in.image", xImage)
        var xCond = tokens.cond.map { embIn.imageCond($0.asType(dtype)) }
        if let xCond { capture?("emb_in.image_cond", xCond) }
        var xText = txtIn(tokens.text.asType(dtype))
        capture?("txt_in", xText)

        let earlyImage = earlyModulations(.image, vecImage)
        let earlyCond = earlyModulations(.imageCond, vecRest)
        let earlyText = earlyModulations(.text, vecRest)
        for d in 0..<config.depth {
            xImage = contentBlocks.image[d](xImage, rope: rope.image, mod: earlyImage)
            capture?("early.image.\(d)", xImage)
            if let cond = xCond, let condRope = rope.cond {
                xCond = contentBlocks.imageCond[d](cond, rope: condRope, mod: earlyCond)
                capture?("early.image_cond.\(d)", xCond!)
            }
            xText = textBlocks[d](xText, rope: rope.text, mod: earlyText)
            capture?("early.text.\(d)", xText)
        }

        // Pack in mode order [image, image_cond, text] with per-token modulation.
        let n = xImage.dim(0)
        var segments: [(MLXArray, Flux3Mod, Flux3Rope)] = [(xImage, singleModulations(.image, vecImage), rope.image)]
        if let cond = xCond, let condRope = rope.cond {
            segments.append((cond, singleModulations(.imageCond, vecRest), condRope))
        }
        segments.append((xText, singleModulations(.text, vecRest), rope.text))
        func perToken(_ field: (Flux3Mod) -> MLXArray) -> MLXArray {
            concatenated(segments.map { broadcast(field($0.1), to: [$0.0.dim(0), config.hidden]) }, axis: 0)
        }
        let mod = Flux3Mod(shift: perToken(\.shift), scale: perToken(\.scale), gate: perToken(\.gate))
        let packedRope = Flux3Rope.concatenating(segments.map(\.2))
        var h = concatenated(segments.map(\.0), axis: 0)
        for (d, block) in singleBlocks.enumerated() {
            h = block(h, rope: packedRope, mod: mod)
            if let capture, [0, 7, 14, 27].contains(d) { capture("single.\(d)", h) }
        }
        let out = finalLayer.image(h[..<n], vec: vecImage)
        capture?("final", out)
        return out
    }
}
