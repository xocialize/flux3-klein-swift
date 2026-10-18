// Flux3KleinGate — parity gates against the oracle goldens (mlxengine-image/WIP/flux3-klein-oracle).
//
//   Flux3KleinGate --sched <goldensDir>
//   Flux3KleinGate --prep <prepGoldensDir>
//   Flux3KleinGate --e2e <snapshot> <oracleDir> <example.json>... [--mlx-noise]
//   Flux3KleinGate --oneshot <snapshot> <oracleDir> --dit bf16|int8|int4 --te bf16|int8 --case t2i|ref1|ref4|ref4mp
//                  [--steps 2] [--drop-te] [--hold 6]   (footprint ledger: load, one request, hold for `footprint`)
//   Flux3KleinGate --vae <snapshot> <goldensDir> [--gpu]
//   Flux3KleinGate --encoder <snapshot> <goldensDir> [--gpu] [--bf16]
//   Flux3KleinGate --dit <snapshot> <goldensDir> [--case t2i|ref1|ref3] [--gpu] [--quant int8|int4]
//   Flux3KleinGate --encode-cache <snapshot> <prompts.json> <condDir> --te bf16|int8
//   Flux3KleinGate --render-cache <snapshot> <prompts.json> <condDir> <outDir> --dit bf16|int8|int4 [--seeds 7,1234]
//                  [--size 1024]   (the OCR lanes: encode once per encoder tier, denoise per DiT tier; MLX noise per
//                  seed, shared by every lane; existing renders are skipped)
//   Flux3KleinGate --run <snapshot> <goldensDir> [--gpu]
//
// Parity runs fp32 on the CPU stream by default (the fleet's regime); `--gpu` runs the same fp32 math on the GPU.
// `--dit`: one flow-map forward per layout, every hooked stage. `--run`: the 16-step 512² T2I trajectory from the
// golden noise (text features from the dit_t2i golden — the same prompt and encoder), every step's latents.

import Foundation
import Flux2VAE
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

func gateEncoder(snapshot: URL, goldens: URL, dtype: DType) async throws {
    let start = Date()
    let enc = try await Flux3TextEncoder.load(directory: snapshot.appendingPathComponent("text_encoder"), dtype: dtype)
    print(String(format: "text encoder loaded (%@) in %.1fs", "\(dtype)", Date().timeIntervalSince(start)))
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("encoder.safetensors"))
    let j = try loadJSON(goldens.appendingPathComponent("encoder.json"))
    let cosGate: Float = dtype == .float32 ? 0.9999 : 0.995
    for name in ["short", "caption", "bbox"] {
        let meta = j[name] as! [String: Any]
        let real = meta["real_tokens"] as! Int, padded = meta["padded"] as! Int
        let e = enc.encode(meta["prompt"] as! String)
        eval(e.features)
        let goldenIds = g["\(name).input_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let ours = e.tokenIds + Array(repeating: enc.padId, count: e.paddedLength - e.tokenIds.count)
        let idsOK = ours == goldenIds && e.tokenIds.count == real && e.paddedLength == padded
        if !idsOK { failures += 1 }
        print("== \(name): \(idsOK ? "PASS" : "FAIL") token ids (real \(e.tokenIds.count)/\(real), padded \(e.paddedLength)/\(padded))")
        let gf = g["\(name).features"]!
        report("features (all rows)", compare(e.features, gf), cosGate: cosGate, relGate: dtype == .float32 ? 2e-2 : 0.2)
        report("features real rows", compare(e.features[..<real], gf[..<real]), cosGate: cosGate,
               relGate: dtype == .float32 ? 2e-2 : 0.2)
        if padded > real {
            report("features pad rows", compare(e.features[real...], gf[real...]), cosGate: cosGate,
                   relGate: dtype == .float32 ? 2e-2 : 0.2)
        }
        report("text ids", compare(e.ids, g["\(name).ids"]!), cosGate: 0.9999999, relGate: 1e-7)
        for (i, layer) in Flux3TextEncoder.outputLayers.enumerated() where [0, 7].contains(i) {
            let slice = 2560 * i ..< 2560 * (i + 1)
            report("  tap k=\(layer)", compare(e.features[0..., slice], gf[0..., slice]), cosGate: cosGate,
                   relGate: dtype == .float32 ? 2e-2 : 0.2)
        }
    }
}

func gateVAE(snapshot: URL, goldens: URL) throws {
    let vaeDir = snapshot.appendingPathComponent("vae")
    let encoder = try Flux2VAEWeights.loadEncoder(directory: vaeDir, dtype: .float32)
    let decoder = try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .float32)
    decoder.convRoute = .conv3d  // exact, for parity
    for name in ["ref1", "ref3"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("dit_\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("dit_\(name).json"))
        var parts: [MLXArray] = []
        for i in 0..<(j["refs"] as! [Any]).count {
            parts.append(Flux3Latents.tokens(fromPacked: encoder.encodePacked(g["in.ref_pixels.\(i)"]![.newAxis])))
        }
        report("encode \(name) tokens", compare(concatenated(parts, axis: 0), g["in.cond"]!), cosGate: 0.99999, relGate: 5e-3)
    }
    let run = try MLX.loadArrays(url: goldens.appendingPathComponent("run_512.safetensors"))
    let latents = Flux3Latents.packed(fromTokens: run["latents.15"]!, height: 32, width: 32)
    report("decode 512²", compare(decoder.decodePackedLatents(latents)[0], run["decoded"]!), cosGate: 0.99999, relGate: 5e-3)
}

func psnr(_ a: Flux3RGBImage, _ b: Flux3RGBImage) -> Double {
    guard a.width == b.width, a.height == b.height else { return -1 }
    var se = 0.0
    for i in 0..<a.rgb.count { let d = Double(a.rgb[i]) - Double(b.rgb[i]); se += d * d }
    let mse = se / Double(a.rgb.count)
    return mse == 0 ? .infinity : 10 * log10(255 * 255 / mse)
}

func gatePrep(_ dir: URL) throws {
    let j = try loadJSON(dir.appendingPathComponent("prep.json"))
    for name in j.keys.sorted() {
        let meta = j[name] as! [String: Any]
        let src = Flux3RGBImage(try Flux3PNG.read(url: dir.appendingPathComponent("\(name).src.png")))
        let golden = Flux3RGBImage(try Flux3PNG.read(url: dir.appendingPathComponent("\(name).prep.png")))
        let ours: Flux3RGBImage
        var boxOK = true
        if name.hasPrefix("single") {
            let (img, box) = Flux3References.prepareSingle(src)
            ours = img
            let ref = meta["box"] as? [Int]
            let mine = box.map { [$0.left, $0.top, $0.left + $0.width, $0.top + $0.height] }
            boxOK = ref == mine
        } else {
            ours = Flux3References.prepareMulti(src)
        }
        let same = ours.width == golden.width && ours.height == golden.height && ours.rgb == golden.rgb
        var maxDiff = 0
        if ours.rgb.count == golden.rgb.count {
            for i in 0..<ours.rgb.count { maxDiff = max(maxDiff, abs(Int(ours.rgb[i]) - Int(golden.rgb[i]))) }
        }
        let ok = same && boxOK
        if !ok { failures += 1 }
        print("  \(ok ? "PASS" : "FAIL") prep \(name): \(src.width)×\(src.height) → \(ours.width)×\(ours.height) "
              + "(golden \(golden.width)×\(golden.height)) \(same ? "byte-identical" : "max |Δ| \(maxDiff)") box \(boxOK ? "ok" : "MISMATCH")")
    }
}

func gateE2E(snapshot: URL, oracle: URL, examples: [URL]) async throws {
    let start = Date()
    let quant = arg("--quant").flatMap(Flux3Quant.init(rawValue:)) ?? .bf16
    let pipe = try await Flux3KleinPipeline.load(snapshot: snapshot, quant: quant)
    print(String(format: "pipeline loaded (%@ DiT, bf16 TE, fp32 VAE) in %.1fs, MLX active %.2f GB", quant.rawValue,
                 Date().timeIntervalSince(start), Double(Memory.activeMemory) / 1e9))
    let outDir = URL(fileURLWithPath: "outputs/e2e")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    for example in examples {
        let r = try loadJSON(example)
        let name = "\(example.deletingLastPathComponent().lastPathComponent)__\(example.deletingPathExtension().lastPathComponent)"
        let refRoot = oracle.appendingPathComponent("ref")
        let refs = try ((r["references"] as? [String]) ?? []).map { try Flux3RGBImage.read(url: refRoot.appendingPathComponent($0)) }
        let req = Flux3KleinPipeline.Request(prompt: r["prompt"] as! String, references: refs,
                                             width: r["width"] as? Int, height: r["height"] as? Int,
                                             steps: r["steps"] as? Int ?? 16, seed: UInt32(r["seed"] as! Int))
        let noise = has("--mlx-noise") ? nil
            : try MLX.loadArrays(url: oracle.appendingPathComponent("e2e/noise/\(name).noise.safetensors"))["noise"]!
        let t0 = Date()
        let result = try pipe.generate(req, noise: noise)
        let secs = Date().timeIntervalSince(t0)
        let tag = (has("--mlx-noise") ? "mlxnoise" : "torchnoise") + (quant == .bf16 ? "" : "-\(quant.rawValue)")
        try result.image.write(to: outDir.appendingPathComponent("\(name)__\(tag).png"))
        let bfl = Flux3RGBImage(try Flux3PNG.read(url: example.deletingPathExtension().appendingPathExtension("png")))
        let mps = quant == .bf16 ? oracle.appendingPathComponent("e2e/\(name)__a6.93.png")
            : outDir.appendingPathComponent("\(name)__torchnoise.png")  // quant tiers: vs our bf16 render
        let vsOracle = FileManager.default.fileExists(atPath: mps.path)
            ? psnr(result.image, Flux3RGBImage(try Flux3PNG.read(url: mps))) : -1
        let timing = result.timings.sorted { $0.key < $1.key }.map { String(format: "%@ %.1fs", $0.key, $0.value) }
            .joined(separator: ", ")
        print(String(format: "  %@ %@ %d×%d  %.1fs (%@)  PSNR vs BFL %.2f dB, vs %@ %.2f dB", name, tag,
                     result.width, result.height, secs, timing, psnr(result.image, bfl),
                     quant == .bf16 ? "oracle MPS" : "Swift bf16", vsOracle))
    }
}

struct PromptItem: Decodable { let id: String; let prompt: String }

func loadPrompts(_ url: URL) throws -> [PromptItem] {
    try JSONDecoder().decode([PromptItem].self, from: Data(contentsOf: url))
}

func encodeCache(snapshot: URL, prompts: URL, condDir: URL, te: Flux3TextEncoderQuant) async throws {
    let enc = try await Flux3TextEncoder.load(directory: snapshot.appendingPathComponent("text_encoder"), quant: te)
    let dir = condDir.appendingPathComponent(te.rawValue)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for item in try loadPrompts(prompts) {
        let e = enc.encode(item.prompt)
        try MLX.save(arrays: ["features": e.features, "ids": e.ids], url: dir.appendingPathComponent("\(item.id).safetensors"))
        print("  encoded \(item.id): \(e.tokenIds.count) tokens → \(e.paddedLength)")
    }
}

func renderCache(snapshot: URL, prompts: URL, condDir: URL, outDir: URL, dit: Flux3Quant) throws {
    let start = Date()
    let pipe = Flux3KleinPipeline(
        transformer: try Flux3Weights.loadTransformer(snapshot: snapshot, quant: dit),
        textEncoder: nil, vaeEncoder: try Flux2VAEWeights.loadEncoder(directory: snapshot.appendingPathComponent("vae")),
        vae: try Flux2VAEWeights.loadVAE(directory: snapshot.appendingPathComponent("vae")), snapshot: snapshot)
    print(String(format: "DiT %@ + VAE loaded in %.1fs, MLX active %.2f GB", dit.rawValue, Date().timeIntervalSince(start),
                 Double(Memory.activeMemory) / 1e9))
    let seeds = (arg("--seeds") ?? "7,1234").split(separator: ",").compactMap { UInt32($0) }
    let size = Int(arg("--size") ?? "1024")!
    let teTiers = try FileManager.default.contentsOfDirectory(atPath: condDir.path).filter { !$0.hasPrefix(".") }.sorted()
    let items = try loadPrompts(prompts)
    for te in teTiers {
        let lane = outDir.appendingPathComponent("\(dit.rawValue)-\(te)")
        try FileManager.default.createDirectory(at: lane, withIntermediateDirectories: true)
        for item in items {
            let cond = try MLX.loadArrays(url: condDir.appendingPathComponent("\(te)/\(item.id).safetensors"))
            for seed in seeds {
                let png = lane.appendingPathComponent("\(item.id)-s\(seed).png")
                if FileManager.default.fileExists(atPath: png.path) { continue }
                Memory.peakMemory = 0
                let t0 = Date()
                let r = try pipe.generate(conditioned: cond["features"]!, textIDs: cond["ids"]!, width: size, height: size,
                                          seed: seed)
                let tmp = lane.appendingPathComponent(".\(item.id)-s\(seed).png")
                try r.image.write(to: tmp)
                try FileManager.default.moveItem(at: tmp, to: png)  // atomic: a killed run leaves no partial PNG
                print(String(format: "  %@ %@ s%u  %.1fs  MLX peak %.2f GB", lane.lastPathComponent, item.id, seed,
                             Date().timeIntervalSince(t0), Double(Memory.peakMemory) / 1e9))
            }
        }
    }
}

func oneshot(snapshot: URL, oracle: URL) async throws {
    // The engine's default buffer-pool cap (MLXServeCore GPUCacheConfiguration .automatic: min(2 GB, 5 % of budget)).
    // Without it, MLX's pool keeps every freed buffer (the bf16 copies dropped by quantize-on-load included) and
    // phys_footprint over-reads. FLUX3_CACHE_LIMIT_GB overrides.
    Memory.cacheLimit = Int((Double(ProcessInfo.processInfo.environment["FLUX3_CACHE_LIMIT_GB"] ?? "2") ?? 2) * 1_073_741_824)
    let dit = Flux3Quant(rawValue: arg("--dit") ?? "bf16")!
    let te = Flux3TextEncoderQuant(rawValue: arg("--te") ?? "bf16")!
    let caseName = arg("--case") ?? "t2i"
    let steps = Int(arg("--steps") ?? "2")!
    let gb = { (b: Int) in Double(b) / 1e9 }
    var t0 = Date()
    let pipe = try await Flux3KleinPipeline.load(snapshot: snapshot, quant: dit, textEncoderQuant: te)
    eval(pipe.transformer)
    let loadSecs = Date().timeIntervalSince(t0)
    let residentAll = Memory.activeMemory
    let inputs = oracle.appendingPathComponent("ref/examples/inputs")
    let img = { (n: Int) in try Flux3RGBImage.read(url: inputs.appendingPathComponent("\(n).png")) }
    var refs: [Flux3RGBImage] = []
    switch caseName {
    case "ref1": refs = [try img(2)]
    case "ref4": refs = [try img(2), try img(3), try img(1), try img(2)]
    case "ref4mp": refs = [try img(2).resized(width: 2048, height: 2048)]
    default: break
    }
    let prompt = "A ceramic mug on a wooden desk by a window, morning light, a small sign that reads OPEN LATE."
    Memory.peakMemory = 0
    t0 = Date()
    let conditioning = pipe.textEncoder!.encode(prompt)
    eval(conditioning.features)
    let encodePeak = Memory.peakMemory
    if has("--drop-te") { pipe.releaseTextEncoder(); Memory.clearCache() }
    let residentDiT = Memory.activeMemory
    Memory.peakMemory = 0
    let req = Flux3KleinPipeline.Request(prompt: prompt, references: refs, width: refs.isEmpty ? 1024 : nil,
                                         height: refs.isEmpty ? 1024 : nil, steps: steps, seed: 1)
    let r = try pipe.generate(req, conditioning: conditioning)
    let secs = Date().timeIntervalSince(t0)
    print(String(format: "oneshot dit=%@ te=%@ case=%@ %dx%d steps=%d drop-te=%@ | load %.1fs | MLX resident (all) %.2f GB, "
                 + "resident for denoise %.2f GB, encode peak %.2f GB, request peak %.2f GB | %.1fs (%@)",
                 dit.rawValue, te.rawValue, caseName, r.width, r.height, steps, has("--drop-te") ? "yes" : "no", loadSecs,
                 gb(residentAll), gb(residentDiT), gb(encodePeak), gb(Memory.peakMemory), secs,
                 r.timings.sorted { $0.key < $1.key }.map { String(format: "%@ %.1fs", $0.key, $0.value) }.joined(separator: ", ")))
    if let hold = arg("--hold").flatMap(Double.init) { try await Task.sleep(for: .seconds(hold)) }
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
    } else if has("--encode-cache") {
        let p = positional(after: "--encode-cache", 3)
        try await encodeCache(snapshot: p[0], prompts: p[1], condDir: p[2],
                              te: Flux3TextEncoderQuant(rawValue: arg("--te") ?? "bf16")!)
    } else if has("--render-cache") {
        let p = positional(after: "--render-cache", 4)
        try renderCache(snapshot: p[0], prompts: p[1], condDir: p[2], outDir: p[3],
                        dit: Flux3Quant(rawValue: arg("--dit") ?? "bf16")!)
    } else if has("--oneshot") {
        let p = positional(after: "--oneshot", 2)
        try await oneshot(snapshot: p[0], oracle: p[1])
    } else if has("--prep") {
        try gatePrep(positional(after: "--prep", 1)[0])
    } else if has("--e2e") {
        let a = CommandLine.arguments
        let i = a.firstIndex(of: "--e2e")!
        let snapshot = URL(fileURLWithPath: a[i + 1]), oracle = URL(fileURLWithPath: a[i + 2])
        let examples = a[(i + 3)...].filter { $0.hasSuffix(".json") }.map { URL(fileURLWithPath: $0) }
        try await gateE2E(snapshot: snapshot, oracle: oracle, examples: examples)
    } else if has("--vae") {
        let p = positional(after: "--vae", 2)
        if !has("--gpu") { Device.setDefault(device: Device(.cpu)) }
        try gateVAE(snapshot: p[0], goldens: p[1])
    } else if has("--encoder") {
        let p = positional(after: "--encoder", 2)
        if !has("--gpu") { Device.setDefault(device: Device(.cpu)) }
        try await gateEncoder(snapshot: p[0], goldens: p[1], dtype: has("--bf16") ? .bfloat16 : .float32)
    } else if has("--dit") || has("--run") {
        let flag = has("--dit") ? "--dit" : "--run"
        let p = positional(after: flag, 2)
        if !has("--gpu") { Device.setDefault(device: Device(.cpu)) }
        let start = Date()
        let quant = arg("--quant").flatMap(Flux3Quant.init(rawValue:)) ?? .bf16
        let model = try Flux3Weights.loadTransformer(snapshot: p[0], dtype: quant == .bf16 ? .float32 : .bfloat16,
                                                     quant: quant)
        print(String(format: "DiT loaded (%@) on %@ in %.1fs", quant == .bf16 ? "fp32" : "bf16 + \(quant.rawValue)",
                     "\(Device.defaultDevice())", Date().timeIntervalSince(start)))
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
