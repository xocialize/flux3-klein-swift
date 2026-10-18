// FLUX 3 Image Klein's safety checks: a port of the reference `flux3/safety.py` + its use in `flux3/generate.py`
// (bfc8889, Apache-2.0). Off by default upstream (`--safety`, "Advanced, Safety checks" in the demo) and here.
//
// Reference facts:
//   - the prompt check: the request plus every box `desc`, joined by spaces, asked as PROMPT_TEXT_INTEGRITY;
//   - the image check (each reference as given, and the output): the NSFW classifier first (P(nsfw) > 0.85 =
//     "nudity or other adult content"), then the image resized to ~512² area (PIL `resize`, default BICUBIC) and
//     asked as PROMPT_IMAGE_INTEGRITY + image + PROMPT_IMAGE_INTEGRITY_FOLLOW_UP ("a copyrighted character, a brand or
//     a public figure");
//   - the verdict is the next token restricted to "yes" / "no" (logit(yes) > logit(no)), on the prompt upsampler when
//     one is loaded, else on the text encoder's Qwen3-VL-4B-Instruct (as a full VLM);
//   - a flag refuses the request with the reference's message.

import CoreImage
import Flux3Klein
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

public enum Flux3SafetyPrompts {
    public static let system =
        "Your task is identifying images and text for copyright concerns and depictions of public personas."

    public static let imageIntegrity = """
        Task: Analyze an image to identify potential copyright concerns or depictions of public figures.

        Output: Respond with only "yes" or "no"

        Criteria for "yes":
        - The image contains a recognizable character from copyrighted media (movies, TV, comics, games, etc.)
        - The image displays a trademarked logo or brand
        - The image depicts a recognizable public figure (celebrities, politicians, athletes, influencers, historical figures, etc.)

        Criteria for "no":
        - All other cases
        - When you cannot identify the specific copyrighted work or named individual

        Critical Requirements:
        1. You must be able to name the exact copyrighted work or specific person depicted
        2. General references to demographics or characteristics are not sufficient
        3. Base your decision solely on visual content, not interpretation
        4. Provide only the one-word answer: "yes" or "no"
        """

    public static let imageFollowUp = "Does this image have copyright concerns or includes public figures?"

    public static let textIntegrity = """
        Task: Analyze a text prompt to identify potential copyright concerns or requests to depict living public figures.

        Output: Respond with only "yes" or "no"

        Criteria for "Yes":
        - The prompt explicitly names a character from copyrighted media (movies, TV, comics, games, etc.)
        - The prompt explicitly mentions a trademarked logo or brand
        - The prompt names or describes a specific living public figure (celebrities, politicians, athletes, influencers, etc.)

        Criteria for "No":
        - All other cases
        - When you cannot identify the specific copyrighted work or named individual

        Critical Requirements:
        1. You must be able to name the exact copyrighted work or specific person referenced
        2. General demographic descriptions or characteristics are not sufficient
        3. Analyze only the prompt text, not potential image outcomes
        4. Provide only the one-word answer: "yes" or "no"

        The prompt to check is:
        -----
        {prompt}
        -----

        Does this prompt have copyright concerns or includes public figures?
        """

    /// The text the prompt check reads: the request and every box description.
    public static func screened(_ prompt: String, boxes: [Flux3Box]?) -> String {
        ([prompt] + (boxes ?? []).map { $0.desc ?? "" }).joined(separator: " ")
    }
}

/// Records the restricted next-token logits at prefill.
final class Flux3LogitTap: @unchecked Sendable {
    var yes: Float = 0
    var no: Float = 0
    var fired = false
}

struct Flux3YesNoTap: LogitProcessor {
    let yes: Int
    let no: Int
    let tap: Flux3LogitTap

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let row = logits.reshaped(-1).asType(.float32)
        tap.yes = row[yes].item(Float.self)
        tap.no = row[no].item(Float.self)
        tap.fired = true
        return logits
    }

    mutating func didSample(token: MLXArray) {}
}

/// The yes/no judge: the upsampler's Qwen3.5-9B, or Qwen3-VL-4B-Instruct (the text encoder's checkpoint, loaded as a
/// VLM). Margins are logit(yes) − logit(no); positive = flagged.
public final class Flux3SafetyJudge: Sendable {
    public let container: ModelContainer
    public let yes: Int
    public let no: Int
    /// The image check's area (`CHECK_AREA`).
    public static let checkArea = 512 * 512

    init(container: ModelContainer) async throws {
        self.container = container
        (yes, no) = try await container.perform { context in
            let ids = ["yes", "no"].map { context.tokenizer.encode(text: $0, addSpecialTokens: false).first }
            guard let y = ids[0], let n = ids[1] else { throw Flux3KleinError.loading("no yes/no token ids") }
            return (y, n)
        }
    }

    /// The upsampler, when one is loaded (the reference's vLLM path).
    public convenience init(upsampler: Flux3PromptUpsampler) async throws {
        try await self.init(container: upsampler.container)
    }

    /// Qwen3-VL-4B-Instruct as a VLM from the snapshot's `text_encoder/` (the reference's text-encoder path).
    public static func load(textEncoderDirectory directory: URL) async throws -> Flux3SafetyJudge {
        let container = try await VLMModelFactory.shared.loadContainer(from: directory, using: Flux3TokenizerLoader())
        return try await Flux3SafetyJudge(container: container)
    }

    public func promptMargin(_ prompt: String) async throws -> Float {
        let text = Flux3SafetyPrompts.textIntegrity.replacingOccurrences(of: "{prompt}", with: prompt)
        return try await margin(content: [["type": "text", "text": text]], images: [])
    }

    /// The image is resized to `checkArea` first, as the reference does (BICUBIC, sides truncated).
    public func imageMargin(_ image: Flux3RGBImage) async throws -> Float {
        let scale = (Double(Self.checkArea) / Double(image.width * image.height)).squareRoot()
        let (w, h) = (max(1, Int(scale * Double(image.width))), max(1, Int(scale * Double(image.height))))
        let resized = image.resized(width: w, height: h, filter: .bicubic)
        guard let cg = Flux3PromptUpsampler.cgImage(resized) else { throw Flux3KleinError.invalidInput("image encode failed") }
        return try await margin(content: [["type": "text", "text": Flux3SafetyPrompts.imageIntegrity], ["type": "image"],
                                          ["type": "text", "text": Flux3SafetyPrompts.imageFollowUp]],
                                images: [cg])
    }

    func margin(content: [[String: any Sendable]], images: [CGImage]) async throws -> Float {
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": [["type": "text", "text": Flux3SafetyPrompts.system]] as [[String: any Sendable]]],
            ["role": "user", "content": content],
        ]
        let input = try await container.prepare(input: UserInput(
            messages: messages, images: images.map { .ciImage(CIImage(cgImage: $0)) },
            additionalContext: ["enable_thinking": false]))
        let (yes, no) = (self.yes, self.no)
        return try await container.perform(nonSendable: input) { context, input in
            let tap = Flux3LogitTap()
            // The iterator prefills at init and hands the last position's logits to the processor.
            _ = try TokenIterator(input: input, model: context.model, processor: Flux3YesNoTap(yes: yes, no: no, tap: tap),
                                  sampler: ArgMaxSampler(), maxTokens: 1)
            // Fail closed: a check that never saw the logits must not read as a pass.
            guard tap.fired else { throw Flux3KleinError.invalidInput("the safety judge produced no logits") }
            return tap.yes - tap.no
        }
    }
}

/// What a check found.
public struct Flux3SafetyVerdict: Sendable, Codable {
    /// P(nsfw), when the classifier ran (images only).
    public var nsfw: Float?
    /// logit(yes) − logit(no) from the judge, when it ran (an image the classifier flagged skips it).
    public var margin: Float?
    /// nil = passed; else the reference's reason.
    public var reason: String?

    public var flagged: Bool { reason != nil }
}

/// The reference's `Safety`: the classifier + the judge.
public struct Flux3SafetyChecker: Sendable {
    public let classifier: Flux3NSFWClassifier
    public let judge: Flux3SafetyJudge

    public static let promptReason = "a copyrighted character, a brand or a public figure"
    public static let nsfwReason = "nudity or other adult content"

    public init(classifier: Flux3NSFWClassifier, judge: Flux3SafetyJudge) {
        self.classifier = classifier
        self.judge = judge
    }

    public func checkPrompt(_ prompt: String) async throws -> Flux3SafetyVerdict {
        let m = try await judge.promptMargin(prompt)
        return Flux3SafetyVerdict(nsfw: nil, margin: m, reason: m > 0 ? Self.promptReason : nil)
    }

    /// The image check: `image_flagged` (the classifier, then the judge).
    public func checkImage(_ image: Flux3RGBImage) async throws -> Flux3SafetyVerdict {
        let p = classifier.nsfwScore(image)
        if p > Flux3NSFWClassifier.threshold { return Flux3SafetyVerdict(nsfw: p, margin: nil, reason: Self.nsfwReason) }
        let m = try await judge.imageMargin(image)
        return Flux3SafetyVerdict(nsfw: p, margin: m, reason: m > 0 ? Self.promptReason : nil)
    }
}
