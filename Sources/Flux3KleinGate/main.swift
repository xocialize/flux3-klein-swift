// Flux3KleinGate — parity gates against the oracle goldens (mlxengine-image/WIP/flux3-klein-oracle).
//
//   Flux3KleinGate --sched <goldensDir>
//   Flux3KleinGate --pup <pupDir> <oracleDir> [--tokens-only] [--case name]   (the prompt upsampler: prompt-token parity
//                  with the HF processor, then captions for the shipped records → outputs/pup/)
//   Flux3KleinGate --pup-tier <pupDir> <oracleDir> <tag>   (captions for BFL's 19 unique example requests → outputs/pup/<tag>/)
//   Flux3KleinGate --safety <set.json> <out.json> [--judge te|<pupDir>] [--snapshot dir] [--classifier dir]   (the safety
//                  checks' raw scores on a test set: P(nsfw), the judge's yes−no margin, wall time per check)
//   Flux3KleinGate --boxes <goldensDir>   (the bbox.py port: plan / detect / is_finished / reconcile / to_canvas)
//   Flux3KleinGate --prep <prepGoldensDir>
//   Flux3KleinGate --e2e <snapshot> <oracleDir> <example.json>... [--mlx-noise] [--untiled]
//   Flux3KleinGate --oneshot <snapshot> <oracleDir> --dit bf16|int8|int4 --te bf16|int8 --case t2i|ref1|ref4|ref4mp
//                  [--steps 2] [--drop-te] [--hold 6] [--untiled]   (footprint ledger: load, one request, hold for `footprint`)
//   Flux3KleinGate --package <snapshot> <oracleDir> [--quant int8] [--steps 16]   (the ModelPackage path: load, T2I and
//                  a 1-ref edit through run(), PNGs to outputs/package/, and an envelope refusal)
//   Flux3KleinGate --package-safety <snapshot> <oracleDir> [--quant bf16] [--pup-dir dir|off]   (the package with the
//                  safety checks on: a benign request passes, a named character is refused, a benign edit passes)
//   Flux3KleinGate --vae <snapshot> <goldensDir> [--gpu]   (decode untiled and in row bands)
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
import ImageIO
import Flux2VAE
import Flux3Klein
import Flux3KleinUpsampler
import MLX
import MLXFlux3Klein
import MLXToolKit

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
        let count = (j["refs"] as! [Any]).count
        func tokens(_ tile: Flux2VAETile?) -> MLXArray {
            concatenated((0..<count).map { i in
                let pixels = g["in.ref_pixels.\(i)"]![.newAxis]
                return Flux3Latents.tokens(fromPacked: tile.map { encoder.encodePacked(pixels, tile: $0) }
                    ?? encoder.encodePacked(pixels))
            }, axis: 0)
        }
        let untiled = tokens(nil)
        report("encode \(name) tokens", compare(untiled, g["in.cond"]!), cosGate: 0.99999, relGate: 5e-3)
        // The pipeline's default (row bands, flux2-vae-mlx-swift 0.7.0), and 1 MB bands so a 256² reference splits.
        for (label, tile) in [("standard", Flux2VAETile.standard), ("1 MB bands", Flux2VAETile(bandBytes: 1 << 20))] {
            let tiled = tokens(tile)
            report("encode \(name) tiled \(label)", compare(tiled, g["in.cond"]!), cosGate: 0.99999, relGate: 5e-3)
            report("  tiled vs untiled", compare(tiled, untiled), cosGate: 0.9999999, relGate: 2e-4)
        }
    }
    let run = try MLX.loadArrays(url: goldens.appendingPathComponent("run_512.safetensors"))
    let latents = Flux3Latents.packed(fromTokens: run["latents.15"]!, height: 32, width: 32)
    let untiled = decoder.decodePackedLatents(latents)[0]
    report("decode 512²", compare(untiled, run["decoded"]!), cosGate: 0.99999, relGate: 5e-3)
    // The pipeline's default: row bands (.standard), and a small band so the 512² latent really splits.
    for (label, tile) in [("standard", Flux2VAETile.standard), ("4 MB bands", Flux2VAETile(bandBytes: 4 << 20))] {
        let tiled = decoder.decodePackedLatents(latents, tile: tile)[0]
        report("decode 512² tiled \(label)", compare(tiled, run["decoded"]!), cosGate: 0.99999, relGate: 5e-3)
        // Float reassociation in the band-wise GroupNorm stats and GEMM tiling: ~6e-5, 35× below the port vs the golden.
        report("  tiled vs untiled", compare(tiled, untiled), cosGate: 0.9999999, relGate: 2e-4)
    }
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
    if has("--untiled") { pipe.vaeTile = nil }
    print(String(format: "pipeline loaded (%@ DiT, bf16 TE, fp32 VAE %@) in %.1fs, MLX active %.2f GB", quant.rawValue,
                 pipe.vaeTile == nil ? "untiled" : "tiled",
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
        let file = "\(name)__\(tag)\(has("--untiled") ? "-untiled" : "").png"
        try result.image.write(to: outDir.appendingPathComponent(file))
        let untiledPNG = outDir.appendingPathComponent("\(name)__\(tag)-untiled.png")
        if !has("--untiled"), FileManager.default.fileExists(atPath: untiledPNG.path) {
            print(String(format: "    tiled vs untiled render: %.2f dB", psnr(result.image, Flux3RGBImage(try Flux3PNG.read(url: untiledPNG)))))
        }
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
    if has("--untiled") { pipe.vaeTile = nil }
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
    // Generic envelope probes: --refs N (multi-ref when N > 1) at --ref-size S², output --size WxH.
    if let n = arg("--refs").flatMap(Int.init), n > 0 {
        let side = Int(arg("--ref-size") ?? "1024")!
        let pool = [try img(2), try img(3), try img(1)]
        refs = (0..<n).map { pool[$0 % pool.count].resized(width: side, height: side) }
    }
    var outW: Int? = refs.isEmpty ? 1024 : nil, outH: Int? = refs.isEmpty ? 1024 : nil
    if let size = arg("--size") {
        let parts = size.split(separator: "x").compactMap { Int($0) }
        outW = parts[0]; outH = parts[1]
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
    let req = Flux3KleinPipeline.Request(prompt: prompt, references: refs, width: outW, height: outH, steps: steps,
                                         seed: 1)
    let r = try pipe.generate(req, conditioning: conditioning)
    let secs = Date().timeIntervalSince(t0)
    print(String(format: "oneshot dit=%@ te=%@ case=%@ %dx%d steps=%d drop-te=%@ | load %.1fs | MLX resident (all) %.2f GB, "
                 + "resident for denoise %.2f GB, encode peak %.2f GB, request peak %.2f GB | %.1fs (%@)",
                 dit.rawValue, te.rawValue, caseName, r.width, r.height, steps, has("--drop-te") ? "yes" : "no", loadSecs,
                 gb(residentAll), gb(residentDiT), gb(encodePeak), gb(Memory.peakMemory), secs,
                 r.timings.sorted { $0.key < $1.key }.map { String(format: "%@ %.1fs", $0.key, $0.value) }.joined(separator: ", ")))
    if let hold = arg("--hold").flatMap(Double.init) { try await Task.sleep(for: .seconds(hold)) }
}

@InferenceActor
func packageSmoke(snapshot: URL, oracle: URL) async throws {
    let quant = Quant(rawValue: arg("--quant") ?? "bf16") ?? .bf16
    let steps = Int(arg("--steps") ?? "16")!
    let pup = arg("--pup-dir") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/publish/flux-3-image-klein-pup-8bit"
    let package = Flux3KleinPackage(configuration: Flux3KleinConfiguration(quant: quant, snapshotPath: snapshot.path,
                                                                           upsamplerPath: pup))
    var t0 = Date()
    try await package.load()
    print(String(format: "package loaded (%@) in %.1fs, MLX active %.2f GB", quant.rawValue, Date().timeIntervalSince(t0),
                 Double(Memory.activeMemory) / 1e9))
    let out = URL(fileURLWithPath: "outputs/package")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let caption = "A hand-painted wooden shop sign that reads \"OPEN LATE\" in cream serif letters on deep green, "
        + "hanging from a wrought-iron bracket on a brick wall, warm evening light from the left, shallow depth of field."
    t0 = Date()
    // A full caption, sent as typed (the upsampler is skipped).
    let t2i = try await package.run(T2IRequest(prompt: caption, width: 1184, height: 896, steps: steps, seed: 7,
                                               metaData: ["upsample": .bool(false)]))
    guard let image = (t2i as? T2IResponse)?.image else { failures += 1; print("FAIL t2i: no image"); return }
    try image.data.write(to: out.appendingPathComponent("t2i-\(quant.rawValue).png"))
    print(String(format: "  PASS t2i %dx%d in %.1fs", image.width ?? 0, image.height ?? 0, Date().timeIntervalSince(t0)))
    let photo = try Data(contentsOf: oracle.appendingPathComponent("ref/examples/inputs/2.png"))
    let editCaption = "A photograph of the same sunlit wooden windowsill and white mug, now filled with black coffee, "
        + "with a golden croissant beside the mug and a few crumbs on the wood; same camera position and light."
    t0 = Date()
    let edit = try await package.run(IEditRequest(images: [Image(format: .png, data: photo)], prompt: editCaption,
                                                  steps: steps, seed: 21))
    guard let edited = (edit as? IEditResponse)?.image else { failures += 1; print("FAIL edit: no image"); return }
    try edited.data.write(to: out.appendingPathComponent("edit-\(quant.rawValue).png"))
    print(String(format: "  PASS edit %dx%d in %.1fs", edited.width ?? 0, edited.height ?? 0, Date().timeIntervalSince(t0)))
    // A short request, upsampled first (the default path).
    Memory.cacheLimit = 2 * 1_073_741_824
    let residentBefore = Memory.activeMemory
    Memory.peakMemory = 0
    t0 = Date()
    let short = try await package.run(T2IRequest(prompt: "a hand-painted shop sign that says OPEN LATE", width: 1024,
                                                 height: 1024, steps: steps, seed: 7))
    guard let shortImage = (short as? T2IResponse)?.image else { failures += 1; print("FAIL short: no image"); return }
    try shortImage.data.write(to: out.appendingPathComponent("t2i-upsampled-\(quant.rawValue).png"))
    print(String(format: "  PASS upsampled t2i %dx%d in %.1fs; MLX resident %.2f GB, request peak %.2f GB (activation %.2f GB)",
                 shortImage.width ?? 0, shortImage.height ?? 0, Date().timeIntervalSince(t0),
                 Double(residentBefore) / 1e9, Double(Memory.peakMemory) / 1e9,
                 Double(Memory.peakMemory - residentBefore) / 1e9))
    do {
        _ = try await package.run(T2IRequest(prompt: "x", width: 2064, height: 2048, steps: 1))
        failures += 1; print("  FAIL envelope: 2064×2048 was not refused")
    } catch let error as Flux3KleinPackageError {
        print("  PASS envelope refusal: \(error.localizedDescription)")
    }
    await package.unload()
}

/// The package path with the safety checks on: a benign short request (upsampled), a named character (refused at the
/// prompt check), and a benign 1-ref edit (the reference and output checks). `--pup-dir off` puts the judge on
/// Qwen3-VL-4B instead of the upsampler. Per request: wall time and the MLX peak above resident (2 GB pool cap).
func packageSafetySmoke(snapshot: URL, oracle: URL) async throws {
    let quant = Quant(rawValue: arg("--quant") ?? "bf16") ?? .bf16
    let pupArg = arg("--pup-dir") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/publish/flux-3-image-klein-pup-8bit"
    let config = Flux3KleinConfiguration(
        quant: quant, snapshotPath: snapshot.path, upsampler: pupArg == "off" ? .off : .int8,
        upsamplerPath: pupArg == "off" ? "" : pupArg, safetyChecks: true,
        safetyClassifierPath: arg("--classifier") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/nsfw-classifier")
    let package = Flux3KleinPackage(configuration: config)
    Memory.cacheLimit = 2 * 1_073_741_824
    var t0 = Date()
    try await package.load()
    let resident = Memory.activeMemory
    print(String(format: "package loaded (%@, judge %@) in %.1fs, MLX resident %.2f GB", quant.rawValue,
                 pupArg == "off" ? "Qwen3-VL-4B" : "upsampler int8", Date().timeIntervalSince(t0), Double(resident) / 1e9))
    let out = URL(fileURLWithPath: "outputs/package-safety")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let tag = "\(quant.rawValue)-\(pupArg == "off" ? "te" : "pup")"
    final class Phases: @unchecked Sendable { var names: [String] = [] }
    func measure(_ label: String, _ body: () async throws -> Void) async throws {
        Memory.peakMemory = 0
        let phases = Phases()
        t0 = Date()
        try await RunProgress.$sink.withValue({ phases.names.append($0.phase.rawValue) }) { try await body() }
        let screens = phases.names.filter { $0 == "screen" }.count
        print(String(format: "    %@: %.1fs, activation %.2f GB, %d screen phase(s)", label, Date().timeIntervalSince(t0),
                     Double(Memory.peakMemory - resident) / 1e9, screens))
    }
    try await measure("benign t2i") {
        let r = try await package.run(T2IRequest(prompt: "a hand-painted shop sign that says OPEN LATE", width: 1024,
                                                 height: 1024, steps: 16, seed: 7))
        guard let image = (r as? T2IResponse)?.image else { failures += 1; print("  FAIL benign t2i: no image"); return }
        try image.data.write(to: out.appendingPathComponent("t2i-\(tag).png"))
        print("  PASS benign t2i passed every check")
    }
    try await measure("named character") {
        do {
            _ = try await package.run(T2IRequest(prompt: "Mickey Mouse waving in front of a castle", width: 1024,
                                                 height: 1024, steps: 16, seed: 7))
            failures += 1; print("  FAIL named character: not refused")
        } catch let error as Flux3KleinPackageError {
            print("  PASS refused: \(error.localizedDescription)")
        }
    }
    try await measure("benign edit") {
        let photo = try Data(contentsOf: oracle.appendingPathComponent("ref/examples/inputs/2.png"))
        let r = try await package.run(IEditRequest(images: [Image(format: .png, data: photo)],
                                                   prompt: "fill the mug with coffee and add a croissant", steps: 16, seed: 21))
        guard let image = (r as? IEditResponse)?.image else { failures += 1; print("  FAIL benign edit: no image"); return }
        try image.data.write(to: out.appendingPathComponent("edit-\(tag).png"))
        print("  PASS benign edit passed every check")
    }
    await package.unload()
}

func boxesFrom(_ any: Any?) -> [Flux3Box]? {
    guard let list = any as? [[String: Any]] else { return nil }
    func nums(_ v: Any?) -> [Double]? { (v as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue } }
    return list.map { d in
        Flux3Box(id: d["id"] as? String, desc: d["desc"] as? String, bbox: nums(d["bbox"]), from: d["from"] as? String,
                 srcBBox: nums(d["src_bbox"]), tgtBBox: nums(d["tgt_bbox"]))
    }
}

func gateBoxes(_ goldens: URL) throws {
    let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: goldens.appendingPathComponent("boxes.json"))) as! [[String: Any]]
    for (n, c) in cases.enumerated() {
        let kind = c["kind"] as! String
        var ok = false, note = ""
        switch kind {
        case "plan":
            let (key, text, locked) = try Flux3Boxes.plan(c["prompt"] as! String, boxes: boxesFrom(c["boxes"]),
                                                          imageCount: c["images"] as! Int)
            let lockedS = locked.map { Flux3JSON.array($0).dumps() } ?? "null"
            ok = key == c["key"] as! String && text == c["text"] as! String && lockedS == c["locked_s"] as! String
            note = "key \(key)"
        case "detect":
            let found = Flux3Boxes.detect(c["prompt"] as! String)
            let caption = found?.0, rowsS = found.map { Flux3JSON.array($0.1).dumps() }
            ok = caption == c["caption"] as? String && rowsS == c["rows_s"] as? String
                && Flux3Boxes.isFinished(c["prompt"] as! String) == c["finished"] as! Bool
            note = "found \(found != nil), finished \(Flux3Boxes.isFinished(c["prompt"] as! String))"
        case "reconcile":
            guard case .array(let locked)? = Flux3JSON.parse(c["locked_s"] as! String) else { fatalError("bad locked") }
            let out = Flux3Boxes.reconcile(c["text"] as! String, locked: locked)
            ok = out == c["out"] as! String
            if !ok { note = "\n      ours: \(out)\n      want: \(c["out"] as! String)" }
        case "canvas":
            let rows = try Flux3Boxes.toCanvas(boxesFrom(c["boxes"])!)
            let s = Flux3JSON.array(rows).dumps()
            ok = s == c["rows_s"] as! String
            if !ok { note = "\n      ours: \(s)\n      want: \(c["rows_s"] as! String)" }
        default: continue
        }
        if !ok { failures += 1 }
        print("  \(ok ? "PASS" : "FAIL") box case \(n) \(kind) \(note)")
    }
}

func gatePup(pup: URL, oracle: URL) async throws {
    let t0 = Date()
    let up = try await Flux3PromptUpsampler.load(directory: pup)
    print(String(format: "upsampler loaded in %.1fs, MLX active %.2f GB", Date().timeIntervalSince(t0),
                 Double(Memory.activeMemory) / 1e9))
    let goldens = oracle.appendingPathComponent("goldens/pup")
    let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: goldens.appendingPathComponent("prompts.json"))) as! [String: [String: Any]]
    let out = URL(fileURLWithPath: "outputs/pup")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    for name in cases.keys.sorted() where arg("--case") == nil || arg("--case") == name {
        let c = cases[name]!
        let key = c["key"] as! String, nRefs = c["n_refs"] as! Int
        // The golden policy_reference pixels, so the token check isolates the template + processor.
        let refs: [CGImage] = try (0..<nRefs).map { i in
            let url = goldens.appendingPathComponent("\(name).ref\(i).png")
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw Flux3KleinError.loading("no \(url.path)") }
            return cg
        }
        let userText = c["user_text"] as! String
        let user = nRefs == 0 ? userText
            : (key.hasSuffix("bbox")
                ? "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: \(userText)\nDecide the edit, imagine the target image, and output the annotation record."
                : "Reference images follow in order (the first image, the second image, ...).\nUser request: \(userText)\nOutput the caption.")
        let ours = try await up.promptTokens(system: up.systemPrompts[key]!, user: user, images: refs)
        let golden = (c["input_ids"] as! [Int])
        let firstDiff = zip(ours, golden).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        let same = ours == golden
        if !same { failures += 1 }
        print("  \(same ? "PASS" : "FAIL") \(name) prompt tokens: ours \(ours.count) / HF \(golden.count)"
              + (same ? "" : ", first difference at \(firstDiff ?? min(ours.count, golden.count))"))
        if has("--tokens-only") { continue }
        // The caption, through the public entry point (box planning, the scaffold, reconcile), from the record's inputs.
        let record = try loadJSON(oracle.appendingPathComponent("ref/examples/\(c["record"] as! String)"))
        let inputs = try ((record["references"] as? [String]) ?? []).map {
            try Flux3PNG.read(url: oracle.appendingPathComponent("ref").appendingPathComponent($0))
        }
        let boxes = boxesFrom(record["boxes"])
        let t1 = Date()
        let caption = has("--golden-refs")
            ? try await up.upsample(record["input_prompt"] as! String, preparedReferences: refs, boxes: boxes)
            : try await up.upsample(record["input_prompt"] as! String, references: inputs, boxes: boxes)
        let secs = Date().timeIntervalSince(t1)
        try caption.write(to: out.appendingPathComponent("\(name)\(has("--golden-refs") ? ".goldenrefs" : "").txt"),
                          atomically: true, encoding: .utf8)
        let bfl = c["bfl_caption"] as! String
        print(String(format: "    caption %d chars in %.1fs (BFL %d chars), exact match %@", caption.count, secs, bfl.count,
                     caption == bfl ? "YES" : "no"))
    }
}

let pupTierRecords = [
    "distilled/man_dog", "distilled/balloons", "distilled/2_coffee", "distilled/2_3_mugprint", "distilled/1_figurine",
    "distilled/3_bandana", "distilled/cactus", "base_vs_distilled/church_distilled", "base_vs_distilled/fashion_seamless_distilled",
    "base_vs_distilled/hiroshige_distilled", "base_vs_distilled/riso_lido_distilled", "base_vs_distilled/still_life_distilled",
    "inputs/2", "inputs/3", "typography/cereal_box", "typography/field_notes", "typography/gig_poster", "typography/tea_tin",
    "upsampling/double_exposure_s1",
]

func gatePupTier(pup: URL, oracle: URL, tag: String) async throws {
    Memory.cacheLimit = 2 * 1_073_741_824
    let t0 = Date()
    let up = try await Flux3PromptUpsampler.load(directory: pup)
    print(String(format: "upsampler %@ loaded in %.1fs, MLX active %.2f GB", tag, Date().timeIntervalSince(t0),
                 Double(Memory.activeMemory) / 1e9))
    let out = URL(fileURLWithPath: "outputs/pup/\(tag)")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    var timings: [String: Double] = [:]
    Memory.peakMemory = 0
    for rel in pupTierRecords {
        let name = rel.replacingOccurrences(of: "/", with: "__")
        let file = out.appendingPathComponent("\(name).txt")
        if FileManager.default.fileExists(atPath: file.path) { continue }
        let record = try loadJSON(oracle.appendingPathComponent("ref/examples/\(rel).json"))
        let inputs = try ((record["references"] as? [String]) ?? []).map {
            try Flux3PNG.read(url: oracle.appendingPathComponent("ref").appendingPathComponent($0))
        }
        let t1 = Date()
        let caption = try await up.upsample(record["input_prompt"] as! String, references: inputs, boxes: boxesFrom(record["boxes"]))
        timings[name] = Date().timeIntervalSince(t1)
        try caption.write(to: file, atomically: true, encoding: .utf8)
        print(String(format: "  %@ %@: %d chars in %.1fs", tag, name, caption.count, timings[name]!))
    }
    print(String(format: "%@ MLX peak %.2f GB", tag, Double(Memory.peakMemory) / 1e9))
    let data = try JSONSerialization.data(withJSONObject: ["timings": timings, "peak_gb": Double(Memory.peakMemory) / 1e9])
    try data.write(to: out.appendingPathComponent("run.json"))
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

// MARK: - safety (the reference's `flux3/safety.py`: NSFW classifier + yes/no judge)

/// Runs every prompt and image of a test set through the checks and records the raw scores: P(nsfw) for images, the
/// judge's logit(yes) − logit(no) for both (images always get the judge, even when the classifier flags them, so the
/// two can be compared with the reference), and the wall time of each check.
func gateSafety(set: URL, out: URL, judgeSpec: String, snapshot: String, classifierDir: String) async throws {
    struct Item: Decodable { let id: String; let `class`: String; let text: String?; let path: String? }
    struct TestSet: Decodable { let prompts: [Item]; let images: [Item] }
    let testSet = try JSONDecoder().decode(TestSet.self, from: Data(contentsOf: set))
    var t0 = Date()
    let classifier = try Flux3NSFWClassifier.load(directory: URL(fileURLWithPath: classifierDir))
    let classifierLoad = Date().timeIntervalSince(t0)
    t0 = Date()
    let judge: Flux3SafetyJudge
    if judgeSpec == "te" {
        judge = try await Flux3SafetyJudge.load(textEncoderDirectory: URL(fileURLWithPath: snapshot)
            .appendingPathComponent("text_encoder"))
    } else {
        judge = try await Flux3SafetyJudge(upsampler: try await Flux3PromptUpsampler.load(directory: URL(fileURLWithPath: judgeSpec)))
    }
    let judgeLoad = Date().timeIntervalSince(t0)
    print(String(format: "loaded classifier %.1fs, judge (%@) %.1fs, yes=%d no=%d", classifierLoad, judgeSpec, judgeLoad,
                 judge.yes, judge.no))
    var prompts: [[String: Any]] = []
    for item in testSet.prompts {
        let t = Date()
        let m = try await judge.promptMargin(item.text ?? "")
        let ms = Date().timeIntervalSince(t) * 1000
        prompts.append(["id": item.id, "class": item.class, "margin": Double(m), "ms": ms])
        print(String(format: "  prompt %-28@ %-9@ margin %+7.3f %@ %5.0f ms", item.id as NSString, item.class as NSString,
                     m, m > 0 ? "FLAG" : "    ", ms))
    }
    let base = set.deletingLastPathComponent()
    var images: [[String: Any]] = []
    for item in testSet.images {
        let url = URL(fileURLWithPath: item.path!, relativeTo: base)
        let image = try Flux3RGBImage.read(url: url)
        var t = Date()
        let p = classifier.nsfwScore(image)
        let nsfwMs = Date().timeIntervalSince(t) * 1000
        t = Date()
        let m = try await judge.imageMargin(image)
        let judgeMs = Date().timeIntervalSince(t) * 1000
        images.append(["id": item.id, "class": item.class, "nsfw": Double(p), "margin": Double(m), "nsfw_ms": nsfwMs,
                       "judge_ms": judgeMs, "width": image.width, "height": image.height])
        let flag = p > Flux3NSFWClassifier.threshold ? "NSFW" : m > 0 ? "FLAG" : "    "
        print(String(format: "  image  %-28@ %-9@ nsfw %.4f margin %+7.3f %@ %4.0f + %5.0f ms", item.id as NSString,
                     item.class as NSString, p, m, flag, nsfwMs, judgeMs))
    }
    let record: [String: Any] = ["judge": judgeSpec, "classifier_load_s": classifierLoad, "judge_load_s": judgeLoad,
                                 "yes": judge.yes, "no": judge.no, "prompts": prompts, "images": images,
                                 "strings": [Flux3SafetyPrompts.system, Flux3SafetyPrompts.imageIntegrity,
                                             Flux3SafetyPrompts.imageFollowUp, Flux3SafetyPrompts.textIntegrity]]
    try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(to: out)
    print("wrote \(out.path)")
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
    } else if has("--pup") {
        let p = positional(after: "--pup", 2)
        try await gatePup(pup: p[0], oracle: p[1])
    } else if has("--pup-tier") {
        let p = positional(after: "--pup-tier", 3)
        try await gatePupTier(pup: p[0], oracle: p[1], tag: p[2].lastPathComponent)
    } else if has("--safety") {
        let p = positional(after: "--safety", 2)
        try await gateSafety(set: p[0], out: p[1], judgeSpec: arg("--judge") ?? "te",
                             snapshot: arg("--snapshot") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/flux-3-image-klein",
                             classifierDir: arg("--classifier") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/nsfw-classifier")
    } else if has("--boxes") {
        try gateBoxes(positional(after: "--boxes", 1)[0])
    } else if has("--package-safety") {
        let p = positional(after: "--package-safety", 2)
        try await packageSafetySmoke(snapshot: p[0], oracle: p[1])
    } else if has("--package") {
        let p = positional(after: "--package", 2)
        try await packageSmoke(snapshot: p[0], oracle: p[1])
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
