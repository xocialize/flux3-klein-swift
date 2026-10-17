// Flux3KleinGate — parity gates against the oracle goldens (mlxengine-image/WIP/flux3-klein-oracle).
//
//   Flux3KleinGate --sched <goldensDir>
//   Flux3KleinGate --dit <snapshot> <goldensDir> [--case t2i|ref1|ref3] [--gpu]
//   Flux3KleinGate --run <snapshot> <goldensDir> [--gpu]
//
// Parity runs fp32 on the CPU stream by default (the fleet's regime); `--gpu` runs the same fp32 math on the GPU.
// `--dit`: one flow-map forward per layout, every hooked stage. `--run`: the 16-step 512² T2I trajectory from the
// golden noise (text features from the dit_t2i golden — the same prompt and encoder), every step's latents.

import Foundation
import Flux3Klein
import MLX

setbuf(stdout, nil)

struct Cmp { let cos: Float; let maxAbs: Float; let relMax: Float; let shape: [Int] }

func compare(_ a: MLXArray, _ b: MLXArray) -> Cmp {
    let af = a.asType(.float32).flattened()
    let bf = b.asType(.float32).flattened()
    precondition(af.size == bf.size, "shape mismatch \(a.shape) vs \(b.shape)")
    let dot = sum(af * bf).item(Float.self)
    let na = sqrt(sum(af * af)).item(Float.self)
    let nb = sqrt(sum(bf * bf)).item(Float.self)
    let maxAbs = abs(af - bf).max().item(Float.self)
    let refMax = abs(bf).max().item(Float.self)
    return Cmp(cos: dot / max(na * nb, 1e-30), maxAbs: maxAbs, relMax: maxAbs / max(refMax, 1e-30), shape: a.shape)
}

nonisolated(unsafe) var failures = 0
func report(_ name: String, _ c: Cmp, cosGate: Float = 0.9999, relGate: Float = 2e-2) {
    let ok = c.cos >= cosGate && c.relMax <= relGate
    if !ok { failures += 1 }
    print(String(format: "  %@ %-26@ cos %.8f  maxAbs %.3e  relMax %.3e  %@",
                 ok ? "PASS" : "FAIL", name, c.cos, c.maxAbs, c.relMax, "\(c.shape)"))
}

func loadJSON(_ url: URL) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
}
func arg(_ flag: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
    return a[i + 1]
}
func has(_ flag: String) -> Bool { CommandLine.arguments.contains(flag) }
func positional(after flag: String, _ n: Int) -> [URL] {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: flag), i + n < a.count else { fatalError("\(flag) needs \(n) paths") }
    return (1...n).map { URL(fileURLWithPath: a[i + $0]) }
}

// MARK: - gates

func gateSched(_ goldens: URL) throws {
    let j = try loadJSON(goldens.appendingPathComponent("schedule.json"))
    let values = j["values"] as! [String: [Double]]
    for (key, alpha) in [("alpha_16_0", 16.0), ("alpha_6_93", 6.93)] {
        let ours = Flux3Schedule.timesteps(steps: 16, alpha: alpha)
        let ref = values[key]!.map { Float($0) }
        let maxAbs = zip(ours, ref).map { abs($0 - $1) }.max() ?? .infinity
        let ok = ours.count == ref.count && maxAbs < 1e-6
        if !ok { failures += 1 }
        print(String(format: "  %@ schedule α=%.2f maxAbs %.2e  %@", ok ? "PASS" : "FAIL", alpha, maxAbs,
                     "\(ours.suffix(3))"))
    }
}

func tokens(_ g: [String: MLXArray]) -> Flux3Tokens {
    Flux3Tokens(image: g["in.image"]!, imageIDs: g["in.image_ids"]!, cond: g["in.cond"], condIDs: g["in.cond_ids"],
                text: g["in.text"]!, textIDs: g["in.text_ids"]!)
}

func gateDiT(_ model: Flux3Transformer, goldens: URL, only: String?) throws {
    for name in ["t2i", "ref1", "ref3"] where only == nil || name == only {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("dit_\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("dit_\(name).json"))
        let t = Float(j["t"] as! Double), tNext = Float(j["t_next"] as! Double)
        print("== dit_\(name): tokens \(j["tokens"]!)  t \(t) → \(tNext)")
        let tk = tokens(g)
        let rope = Flux3RopeSet(tk, config: model.config)
        var stages: [String: MLXArray] = [:]
        let start = Date()
        let out = model(tk, rope: rope, t: t, tNext: tNext) { name, x in stages[name] = x }
        eval(out)
        print(String(format: "  forward %.1fs", Date().timeIntervalSince(start)))
        let order = ["ref_time_in", "time_in", "emb_in.image", "emb_in.image_cond", "txt_in"]
            + (0..<5).flatMap { ["early.image.\($0)", "early.image_cond.\($0)", "early.text.\($0)"] }
            + [0, 7, 14, 27].map { "single.\($0)" } + ["final"]
        for stage in order {
            guard let golden = g["stage.\(stage)"] else { continue }
            guard let ours = stages[stage] else {
                // the reference runs a never-attended phantom token through an empty mode; we skip the mode
                if tk.cond == nil && stage.contains("image_cond") { continue }
                print("  FAIL \(stage): not captured"); failures += 1; continue
            }
            report(stage, compare(ours, golden), cosGate: stage == "final" ? 0.9999 : 0.99999)
        }
        report("velocity", compare(out, g["out.velocity"]!))
    }
}

func gateRun(_ model: Flux3Transformer, goldens: URL) throws {
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("run_512.safetensors"))
    let j = try loadJSON(goldens.appendingPathComponent("run_512.json"))
    let text = try MLX.loadArrays(url: goldens.appendingPathComponent("dit_t2i.safetensors"))
    let size = j["size"] as! Int, alpha = j["alpha"] as! Double
    let side = size / 16
    // latent ids (0, row·0.5, col·0.5, 0)
    var ids = [Float]()
    for r in 0..<side { for c in 0..<side { ids += [0, Float(r) * 0.5, Float(c) * 0.5, 0] } }
    let tk = Flux3Tokens(image: g["noise"]!, imageIDs: MLXArray(ids, [side * side, 4]),
                         text: text["in.text"]!, textIDs: text["in.text_ids"]!)
    let rope = Flux3RopeSet(tk, config: model.config)
    let schedule = Flux3Schedule.timesteps(steps: 16, alpha: alpha)
    print("== run_512: α \(alpha), \(side * side) image tokens, 16 steps")
    var x = g["noise"]!.asType(.float32)
    let start = Date()
    for step in 0..<16 {
        var cur = tk
        cur.image = x
        let v = model(cur, rope: rope, t: schedule[step], tNext: schedule[step + 1])
        x = Flux3Schedule.step(x, velocity: v, t: schedule[step], tNext: schedule[step + 1])
        eval(x)
        report("latents.\(step)", compare(x, g["latents.\(step)"]!), cosGate: step < 8 ? 0.9999 : 0.999)
    }
    print(String(format: "  16 steps %.1fs", Date().timeIntervalSince(start)))
}

// MARK: - main

do {
    if has("--sched") {
        try gateSched(positional(after: "--sched", 1)[0])
    } else if has("--dit") || has("--run") {
        let flag = has("--dit") ? "--dit" : "--run"
        let p = positional(after: flag, 2)
        if !has("--gpu") { Device.setDefault(device: Device(.cpu)) }
        let start = Date()
        let model = try Flux3Weights.loadTransformer(snapshot: p[0], dtype: .float32)
        print(String(format: "DiT loaded fp32 on %@ in %.1fs", "\(Device.defaultDevice())", Date().timeIntervalSince(start)))
        if flag == "--dit" { try gateDiT(model, goldens: p[1], only: arg("--case")) } else { try gateRun(model, goldens: p[1]) }
    } else {
        print("usage: Flux3KleinGate --sched <goldens> | --dit <snapshot> <goldens> [--case n] [--gpu] | --run <snapshot> <goldens> [--gpu]")
        exit(2)
    }
} catch {
    print("ERROR: \(error)")
    exit(1)
}
print(failures == 0 ? "ALL PASS" : "\(failures) FAILURE(S)")
exit(failures == 0 ? 0 : 1)
