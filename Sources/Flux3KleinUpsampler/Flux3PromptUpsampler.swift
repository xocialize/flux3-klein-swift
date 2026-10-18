// FLUX 3 Image Klein's prompt upsampler (black-forest-labs/flux-3-image-klein-pup, Apache-2.0): a Qwen3.5-9B
// vision-language fine-tune that turns a short request (+ up to 10 references, + optional boxes) into the full scene
// caption the DiT expects. A port of the reference `flux3/pup.py` (bfc8889) on MLXVLM's Qwen3.5.
//
// Reference facts:
//   - system prompt by mode: `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox` (`system_prompts.json`, shipped with the weights);
//   - text-to-image: the user turn is the request (or the LOCKED BOUNDING-BOX scaffold, Flux3Boxes.plan);
//   - edits: the references first, then a three-line scaffold (I2I_SCAFFOLD);
//   - chat template with enable_thinking = False;
//   - greedy decoding, presence penalty 1.5 (t2i) / 0 (i2i) on GENERATED tokens only (vLLM semantics; MLXLMCommon's
//     presence penalty would also count the ~3,400-character system prompt), at most 3,584 new tokens, stop on
//     <|im_end|> / <|endoftext|>;
//   - references: EXIF transpose, alpha composited over white, LANCZOS to a budget by reference count (1 MP for one,
//     724² for two, 512² for 3–10; sides multiples of 32), then a JPEG q92 round trip;
//   - with boxes, the answer is reconciled against the caller's (Flux3Boxes.reconcile).

import CoreGraphics
import CoreImage
import Flux3Klein
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers
import UniformTypeIdentifiers

/// `swift-transformers` tokenizer → MLXLMCommon's protocol (what `#adaptHuggingFaceTokenizer` expands to).
struct Flux3TokenizerAdapter: MLXLMCommon.Tokenizer, @unchecked Sendable {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { upstream.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] {
        try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

struct Flux3TokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        Flux3TokenizerAdapter(upstream: try await AutoTokenizer.from(modelFolder: directory))
    }
}

/// vLLM's `presence_penalty`: subtract `penalty` once from the logit of every token already GENERATED.
struct OutputPresencePenalty: LogitProcessor {
    let penalty: Float
    var seen: Set<Int32> = []

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard penalty != 0, !seen.isEmpty else { return logits }
        let indices = MLXArray(Array(seen))[.newAxis, 0...].asType(.uint32)
        let penalized = takeAlong(logits, indices, axis: -1) - penalty
        return putAlong(logits, indices, values: penalized, axis: -1)
    }

    mutating func didSample(token: MLXArray) { seen.insert(token.asType(.int32).item(Int32.self)) }
}

public final class Flux3PromptUpsampler: Sendable {
    public let container: ModelContainer
    public let systemPrompts: [String: String]
    public let maxNewTokens: Int

    public static let maxReferences = 10
    static let registerProcessor: Void = VLMProcessorLoadingRegistry.shared.register(
        ModelTypeProcessorResolver(processorTypes: ["qwen3_5": "Qwen3VLProcessor"]))
    static let presencePenalty: [String: Float] = ["t2i": 1.5, "i2i": 0.0]
    static let stopTokens = ["<|im_end|>", "<|endoftext|>"]
    static let i2iScaffold = [
        false: "Reference images follow in order (the first image, the second image, ...).\nUser request: {instruction}\nOutput the caption.",
        true: "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: {instruction}\n"
            + "Decide the edit, imagine the target image, and output the annotation record.",
    ]

    init(container: ModelContainer, systemPrompts: [String: String], maxNewTokens: Int) {
        self.container = container
        self.systemPrompts = systemPrompts
        self.maxNewTokens = maxNewTokens
    }

    /// Load from a snapshot of black-forest-labs/flux-3-image-klein-pup (or an MLX conversion of it) holding
    /// `system_prompts.json`.
    public static func load(directory: URL, maxNewTokens: Int = 3584) async throws -> Flux3PromptUpsampler {
        let prompts = try JSONDecoder().decode([String: String].self,
                                               from: Data(contentsOf: directory.appendingPathComponent("system_prompts.json")))
        // BFL's processor configs name no `processor_class`, and MLXVLM's Qwen3.5 fallback only applies when there are
        // no processor configs at all; name the processor MLXVLM itself picks for Qwen3.5.
        _ = registerProcessor
        let container = try await VLMModelFactory.shared.loadContainer(from: directory, using: Flux3TokenizerLoader())
        return Flux3PromptUpsampler(container: container, systemPrompts: prompts, maxNewTokens: maxNewTokens)
    }

    /// One upsampled caption for `request` (any language; the caption is English). With `boxes` (or a box table already
    /// in the request) the answer carries the reconciled box table. Greedy, so the same input gives the same caption.
    public func upsample(_ request: String, references: [Flux3RGBAImage] = [], boxes: [Flux3Box]? = nil) async throws -> String {
        guard references.count <= Self.maxReferences else {
            throw Flux3KleinError.invalidInput("the upsampler takes at most \(Self.maxReferences) references")
        }
        let prepared = try references.map { try Self.policyReference($0, count: references.count) }
        return try await upsample(request, preparedReferences: prepared, boxes: boxes)
    }

    /// As `upsample(_:references:boxes:)`, with references already prepared by `policyReference` (or by the
    /// reference's own PIL path, for parity work).
    public func upsample(_ request: String, preparedReferences prepared: [CGImage], boxes: [Flux3Box]? = nil) async throws -> String {
        let task = prepared.isEmpty ? "t2i" : "i2i"
        let (key, text, locked) = try Flux3Boxes.plan(request, boxes: boxes, imageCount: prepared.count)
        guard let system = systemPrompts[key] else { throw Flux3KleinError.loading("system_prompts.json lacks \(key)") }
        let user = prepared.isEmpty ? text
            : Self.i2iScaffold[locked != nil]!.replacingOccurrences(of: "{instruction}", with: text)
        let caption = try await generate(system: system, user: user, images: prepared,
                                         presencePenalty: Self.presencePenalty[task]!)
        guard !caption.isEmpty else { throw Flux3KleinError.invalidInput("the upsampler returned an empty caption") }
        if let locked { return Flux3Boxes.reconcile(caption, locked: locked) }
        return caption
    }

    /// The prompt token ids exactly as generation sees them (the gate compares them with the HF processor's).
    public func promptTokens(system: String, user: String, images: [CGImage]) async throws -> [Int] {
        let input = try await prepare(system: system, user: user, images: images)
        return input.text.tokens.asArray(Int32.self).map(Int.init)
    }

    func prepare(system: String, user: String, images: [CGImage]) async throws -> LMInput {
        let userInput = UserInput(
            chat: [.system(system), .user(user, images: images.map { .ciImage(CIImage(cgImage: $0)) })],
            additionalContext: ["enable_thinking": false])
        return try await container.prepare(input: userInput)
    }

    func generate(system: String, user: String, images: [CGImage], presencePenalty: Float) async throws -> String {
        let input = try await prepare(system: system, user: user, images: images)
        let maxNewTokens = self.maxNewTokens
        return try await container.perform(nonSendable: input) { context, input in
            let stops = Set(Self.stopTokens.compactMap { context.tokenizer.convertTokenToId($0) })
            var iterator = try TokenIterator(
                input: input, model: context.model, processor: OutputPresencePenalty(penalty: presencePenalty),
                sampler: ArgMaxSampler(), maxTokens: maxNewTokens)
            var tokens: [Int] = []
            while let token = iterator.next() {
                if stops.contains(token) { break }
                tokens.append(token)
                if tokens.count % 64 == 0 { try Task.checkCancellation() }
            }
            return context.tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    // MARK: - Reference preparation (`policy_reference`, `reference_size`)

    static let referenceMultiple = 32
    static let maxAspectRatio = 200.0

    static func maxPixels(references n: Int) -> Int { n == 1 ? 1024 * 1024 : n == 2 ? 724 * 724 : 512 * 512 }

    /// `reference_size`.
    public static func referenceSize(width: Int, height: Int, count: Int) throws -> (Int, Int) {
        guard Double(max(width, height)) / Double(min(width, height)) <= maxAspectRatio else {
            throw Flux3KleinError.invalidInput("aspect ratio of \(width)x\(height) exceeds \(Int(maxAspectRatio)):1")
        }
        let m = referenceMultiple
        func floor32(_ v: Double) -> Int { max(m, Int((v / Double(m)).rounded(.down)) * m) }
        let cap = maxPixels(references: count)
        var w = max(m, Int((Double(width) / Double(m)).rounded(.toNearestOrEven)) * m)
        var h = max(m, Int((Double(height) / Double(m)).rounded(.toNearestOrEven)) * m)
        if w * h > cap {
            let beta = (Double(width * height) / Double(cap)).squareRoot()
            (w, h) = (floor32(Double(width) / beta), floor32(Double(height) / beta))
        }
        if w * h > cap {
            if w == m { h = floor32(Double(cap) / Double(w)) } else { w = floor32(Double(cap) / Double(h)) }
        }
        return (w, h)
    }

    /// `policy_reference`: alpha over white, RGB, LANCZOS to `reference_size`, JPEG q92 round trip.
    public static func policyReference(_ image: Flux3RGBAImage, count: Int) throws -> CGImage {
        var rgb = [UInt8](repeating: 255, count: image.width * image.height * 3)
        for i in 0..<(image.width * image.height) {
            let a = Int(image.rgba[i * 4 + 3])
            for c in 0..<3 {
                // PIL alpha_composite over opaque white: (src·a + 255·(255 − a)) / 255, rounded.
                let v = Int(image.rgba[i * 4 + c])
                rgb[i * 3 + c] = UInt8((v * a + 255 * (255 - a) + 127) / 255)
            }
        }
        var img = Flux3RGBImage(rgb: rgb, width: image.width, height: image.height)
        let (w, h) = try referenceSize(width: img.width, height: img.height, count: count)
        if (w, h) != (img.width, img.height) { img = img.resized(width: w, height: h) }
        return try jpegRoundTrip(img, quality: 0.92)
    }

    static func jpegRoundTrip(_ image: Flux3RGBImage, quality: Double) throws -> CGImage {
        guard let cg = cgImage(image) else { throw Flux3KleinError.invalidInput("cannot build a reference CGImage") }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw Flux3KleinError.invalidInput("cannot create a JPEG destination")
        }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest), let src = CGImageSourceCreateWithData(data as CFData, nil),
              let decoded = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw Flux3KleinError.invalidInput("JPEG round trip failed")
        }
        return decoded
    }

    static func cgImage(_ image: Flux3RGBImage) -> CGImage? {
        let rgba = image.rgba()
        guard let provider = CGDataProvider(data: Data(rgba.rgba) as CFData) else { return nil }
        return CGImage(width: rgba.width, height: rgba.height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: rgba.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
