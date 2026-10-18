# flux3-klein-swift — Porting spec: FLUX 3 Image Klein → Swift/MLX

Status: **P0–P5 done 2026-10-18** (served by Pixel Agency Image Engine 0.2.12, named-only). The engine-free pipeline is gated end to end; quant tiers have passed the OCR gate,
footprints are measured, and the weights are on mlx-community. P5 (engine package) comes next (the spec was written the same day, four days after release). It is a
**restricted, named-only tier for the Pixel Agency Image Engine** (`pixel-agency-image-engine`, with its MES HTTP and MCP
bindings), not a product
asset, following the Qwen-Image-2.1 precedent (AB-D-0085, AB-D-0116). The licence posture is in §1.
Bridge: task **AB-T-0315**, decision **AB-D-0151** (points 4–5 pending the operator), boundary **AB-B-0041**; P0 receipt **AB-R-0571**
(no FLUX 3 output in any training set).

Upstream:
- Model: https://huggingface.co/black-forest-labs/flux-3-image-klein (distilled, 16 steps). It was created
  2026-10-13 and is gated `auto`; our HF token already has access.
- Base: https://huggingface.co/black-forest-labs/flux-3-image-klein-base (undistilled, 50 steps, CFG).
- Prompt upsampler: https://huggingface.co/black-forest-labs/flux-3-image-klein-pup (**Apache-2.0**, ungated).
- Reference code: https://github.com/black-forest-labs/flux-3-image-klein (**Apache-2.0**, ~1.3k lines of torch).
  `src/flux3/` holds `dit.py`, `sampling.py`, `pipeline.py`, `text_encoder.py`, `autoencoder.py`, `pup.py`,
  `safety.py` and `bbox.py`.
- Diffusers: `Flux3Transformer2DModel` + `Flux3DistilledModularPipeline` (0.41.0.dev0, a modular pipeline).
  **The oracle is the BFL reference code, not diffusers.** It is small, Apache-2.0, and the checkpoint's own
  `config.json` and `manifest.json` describe that format.

## 0. What it is, against what we already have

FLUX 3 Klein is **not a FLUX.2 klein checkpoint bump**. The DiT is a new design. The VAE is the same
architecture, and the text encoder is a different but already-ported backbone.

| | FLUX.2 klein-4B (`PROD/flux2-klein-swift`, Apache-2.0) | **FLUX 3 Image Klein** |
|---|---|---|
| DiT size | 4B: 5 double-stream + 20 single-stream blocks, hidden 3072 | **5.55B** (manifest: 5,548,681,984 params, 359 tensors): **5 per-mode blocks + 28 shared single-stream blocks**, hidden 3072, 24 heads × 128 |
| Early blocks | joint img+txt MMDiT attention | **one block stack per mode** (`image`, `image_cond`, `text`) with its own weights. Each mode attends **only to itself** for the first 5 blocks |
| Block | separate attn and FFN sub-blocks with their own modulations | one **parallel** block (attention + SwiGLU MLP ×3, a single gate), bias-free, `LayerNorm(affine: false)`, QK-RMSNorm fused with RoPE |
| Modulation | per-stream shift/scale/gate | **per-mode** (`image` / `image_cond` / `txt`), one set for the early stack and one for the single stack, applied **per token** after the pack |
| Timestep | t | t for `image`; **t = 0** for `image_cond` and `text`; **plus `ref_time_in(t_next)`** added to every mode (flow map) |
| RoPE | 4-axis, theta 2000 | 4-axis (32,32,32,32), theta 10000, written as **rotation matrices**. Image (h, w) ids scaled ×**0.5** (`image_rope_scale`); reference *n* sits on axis l = −(1+n); text ids are (0, 0, 0, pos) |
| Text encoder | Qwen3-4B, 3 taps (9/18/27) → 7680 | **Qwen3-VL-4B-Instruct**, **text only**, 8 taps (4, 8, …, 32 of 36) → **20480**. Padded to a multiple of 80, max 4096 tokens |
| VAE | FLUX.2 VAE (32 ch, 8× + 2×2 patchify = 16×, BN latent norm) | the **same architecture** (`AutoencoderKLFlux2`, identical diffusers config), fp32. The weights hash differs from klein-4B's diffusers file; check it tensor by tensor in P0 |
| Sampler | Euler, 4 steps, power schedule | **flow-map Euler, 16 steps, no CFG**: `t' = α·t / (1 + (α−1)·t)` over `linspace(1, 0, 17)`. **α = 6.93**, read from `config.json` `sampler.alpha`. The reference code's `Pipeline.generate` defaults to α = 16.0 and never reads the config; that does **not** reproduce BFL's own example renders (P0, §3) |
| References | ≤4 refs (our edit path) | **≤10**. A single ref is fitted to ≤4 MP (2048²), reflect-padded to the 16-px grid, and the output is cropped back. Multi-ref: each ≤1 MP, centre-cropped to 16 px |
| Prompting | short prompts work | **needs full scene captions**. BFL's upsampler (or any LLM given `system_prompts.json`) expands the request. Bounding-box layout on a 0–1000 grid, box edits (`<ref_image_n>`, `src_bbox` → `tgt_bbox`), text rendering |
| Licence | Apache-2.0 | **FLUX Kommunity License v1.0** (§1) |

**What carries over:** the FLUX.2 VAE (decoder: `flux2-vae-mlx-swift`; encoder: `flux2-klein-swift`'s
`VAEEncoder`), the Qwen3-VL Swift backbone (`qwen3vl-mlx-swift`, which already has a 4B load probe and
the flat-positionIds override), and Qwen3.5 VLM in `mlx-swift-lm` (`MLXVLM/Models/Qwen35.swift`) for the
upsampler. The residency, cancellation, loader and NAX conventions carry over too.

**What is new:** the DiT (per-mode stacks, per-token modulation, the flow-map ref-time embedder,
packed per-sample attention), the 8-tap conditioner, the flow-map sampler, reference prep, the bbox
caption format, and the safety path (§6).

## 1. Licence: what decides where this can go

Read on the HF repo 2026-10-17 (`LICENSE.md`, 19,593 bytes; `NOTICE.md`).

- **FLUX Kommunity License v1.0 covers the DiT and VAE.** It is not the Qwen Research licence. It grants
  - §2(a)(i): non-commercial use;
  - §2(a)(ii): **commercial use of Outputs by a "Qualifying User"** (under US$5M gross annualized revenue,
    affiliates included), and production use solely to generate those Outputs.
- **§1(c): "in direct interactions with or that has impact on end users" is *not* a Non-Commercial Purpose.**
  Shipping the weights to customers inside a paid app is a commercial Distribution. It is not covered.
- **§2(e) conditions on the Qualifying-User grant:**
  - content filters, or a review of Outputs before distribution;
  - AI-generated disclosure where the law requires it.
- **§3 Distribution** is allowed with the licence text and the verbatim Attribution Notice
  (`NOTICE.md`). Derivatives (our MLX conversions and quants) must also say they are modified, and
  must not imply BFL endorsement.
- **§4(b): no Output may be used to improve another AI model "that performs any function similar to a
  FLUX Model".** §1(a)(iii) also makes a model trained on FLUX outputs a Derivative.
  **FLUX 3 outputs never go into any training set of ours.** That includes the drawn RealPLKSR `cg`
  bucket (AB-D-0129), the Klein de-light LoRA bootstrap (CLAUDE.md watchlist) and the removal LoRA
  (AB-T-0215). Filed as a boundary.
- §4(a) bans surveillance, biometric processing and digital-replica use. §8(b): **BFL can terminate on
  notice.**
- **Apache-2.0 parts:** the text encoder (Qwen3-VL-4B-Instruct, byte-identical; §3), the upsampler (a
  Qwen3.5-9B fine-tune) and the reference code. The upsampler is the one piece that is a real asset. It
  could also feed AB-T-0054 (the image prompt-enhancer).

**Posture (proposed; the operator confirms, as with AB-D-0085):**
1. New WIP package `flux3-klein-swift`, kept apart from Apache-2.0 `flux2-klein-swift` so the
   shipping package's licence story stays clean.
2. C7: `weightLicense = SPDXLicense("LicenseRef-FLUX-Kommunity-1.0")`, declared package-locally and
   **never** added to `permissiveAllowlist`. Forge (`.permissiveOnly` + `.blocking`) refuses it by
   construction.
3. **Engine + agent only:**
   - Pixel Agency Image Engine and ML[X] Image Server, `.restricted`, `.wip`, **named-only**, behind
     `--allow-research` / `allowRestricted`;
   - never in default routing (`preferredFor` empty), never in a MARKETING-HIGHLIGHTS entry;
   - not cleared for the Desktop App Store channel (AB-D-0115).
4. Offers say `commercial_use: false`. The engine cannot know whether its operator is a Qualifying User,
   so it does not claim the exception. The `notice` names it instead, as plain text:
   > FLUX Kommunity licence: non-commercial; commercial use of outputs only for businesses under
   > US$5M revenue that filter content and disclose AI generation. Read the terms.
5. **Weights:** we convert locally. Whether to publish an MLX mirror is an operator call:
   - legally allowed under §3, with LICENSE + NOTICE + "modified" statement;
   - default: a private xocialize mirror only, never mlx-community;
   - the memory's standing "publish weights" authorization was given for permissive models and does not
     extend to this licence.

## 2. Components and sizes

| Component | Source | bf16 | Plan |
|---|---|---|---|
| DiT | `model.safetensors` (11.1 GB, BFL-native key names; the `transformer/` folder holds the same in diffusers names) | 11.1 GB | Port from `dit.py`; load the native file (its key set is pinned by `manifest.json`). int8 ≈ 5.6 GB, int4 ≈ 3.0 GB |
| Text encoder | `text_encoder/` = Qwen3-VL-4B-Instruct (8.9 GB incl. vision) | ~7.5 GB LM only | `qwen3vl-mlx-swift`, **LM weights only** (the conditioner never sees images), plus a `hiddenStates(layers:)` tap API |
| VAE | `ae.safetensors` (336 MB, fp32) | 0.34 GB | Decoder in `flux2-vae-mlx-swift`. Move klein's `VAEEncoder` there, in its first encode release, so both klein packages share it |
| Upsampler | `flux-3-image-klein-pup` (Qwen3.5-9B VL, 18.8 GB) | 18.8 GB | `MLXVLM` Qwen35, converted with `mlx_vlm.convert` to int4/int8. **Optional at the engine** (§5) |
| Safety | `Falconsai/nsfw_image_detection` (ViT) + a yes/no prompt check on the Qwen3-VL-4B | small | §6, P6 |

Reference peak memory (B200, 1024², upsampler off): 31 GB in bf16. The upsampler adds ~28 GB under vLLM.
Expect the MLX resident set to be ≈ 19 GB in bf16 (DiT + LM + VAE) before activations. With an int8 DiT
it is ≈ 13 GB, before the upsampler.

**Sequence lengths matter more than weights here.**
- 1024² is 4,096 image tokens.
- A single-reference edit at the 4 MP reference cap is 16,384 cond tokens + 16,384 target tokens + text.
- Full-scene captions run to several hundred tokens (the upsampler allows 3,584 new tokens).
- Measure the envelope; do not inherit it (AB-T-0204, AB-L-0202).

## 3. Verifications already done (2026-10-17, read-only)

- **Text encoder = Qwen/Qwen3-VL-4B-Instruct, byte for byte.** Both shards' LFS sha256 match:
  - `30a01a05…39a9`
  - `046296a2…02a6`
- **The VAE *is* the FLUX.2 VAE** (P0, `WIP/flux3-klein-oracle/vae_diff.py`, `goldens/vae_diff.json`).
  - The config is identical to FLUX.2-klein-4B's, field for field.
  - All 250 float tensors of FLUX 3's fp32 `vae/diffusion_pytorch_model.safetensors` round **exactly** to
    the mlx-community klein-4B bf16 mirror. They are the same weights, at full fp32 precision.
  - The native `ae.safetensors` holds the same values under BFL names (242 of 251 match by fingerprint).
    The 9 that don't are the mid-block attention q/k/v/proj, stored as 1×1 convs instead of linears, plus
    `num_batches_tracked`.
  - So there is no VAE port: `flux2-vae-mlx-swift` loads FLUX 3's `vae/` folder as it is.
- **α is 6.93, not the code's 16.0** (P0, `e2e_examples.py`, `e2e/scores.json`). BFL's shipped example records,
  re-rendered with the reference code in bf16 on MPS, scored by PSNR against BFL's own PNGs:

  | Example | α = 16.0 | α = 6.93 |
  |---|---|---|
  | knitted diorama (T2I, boxes, 1184×896) | 23.1 dB | **35.8 dB** |
  | coffee + croissant (1-ref edit, 1024²) | 29.6 dB | **42.5 dB** |
  | mug print (multi-ref, 1024²) | 38.0 dB | **56.2 dB** |

  At 6.93 the renders match BFL's to bf16-backend noise; at 16 the content changes (`e2e/sheet_2_coffee.png`). The
  reference CLI as published (`Pipeline.generate(..., alpha=16.0)`) therefore drifts from the checkpoint's own
  setting. That could be reported upstream; it is outward-facing, so it is the operator's call.
- The reference code runs on CPU fp32 unmodified. On MPS it needs one shim, because its RoPE frequencies are float64:
  compute them on the CPU, as `e2e_examples.py` does.
- The `fp8r` and `nvfp4` variants are CUDA formats and unusable as such. We quantize from bf16 ourselves.
- The reference repo ships **external goldens**: `examples/{distilled,base,typography}/*.json` + `.png`
  hold caption, seed, size and steps, with BFL's own renders. These are the e2e targets.

## 4. Phases

**P0 — oracle and weights** (`WIP/flux3-klein-oracle`)
- Disk: freed to 188 GB (2026-10-17). This phase needs about 20 GB of weights, ~19 GB for the upsampler,
  plus MLX conversions.
- Download the bf16 package to `mlxengine-image/weights/flux-3-image-klein`: `model.safetensors`,
  `config.json`, `manifest.json`, `ae.safetensors`, `text_encoder/*`, `LICENSE.md`, `NOTICE.md`, plus
  `vae/` (336 MB, for the diffusers-vs-native VAE diff). No variants.
- Pin the reference repo at a commit. Run it fp32 on CPU, swapping the cuda autocast/cudnn paths.
- Goldens:
  - the 8-tap conditioner (token ids, padding, every tap);
  - VAE encode and decode, plus a tensor diff against the klein-4B VAE;
  - the schedule at α = 16 and α = 6.93 (6.93 is the right one, §3);
  - the DiT on three layouts (T2I, 1 ref, 3 refs), with the post-embed state, every early block per
    mode, the pack, single blocks 0/7/14/27 and the final layer;
  - one full 16-step run at 512² with fixed noise;
  - bf16 MPS e2e renders of four shipped examples, checked against BFL's PNGs.

**P1 — Swift core** (`WIP/flux3-klein-swift`, new SPM package, `Flux3Klein` + `MLXFlux3Klein`)
- Write `Flux3Transformer`:
  - per-mode `Block` stacks;
  - the pack/unpack with per-token modulation;
  - `MLXFast` SDPA per sample. Single-sample inference needs no masking: the early stacks are per mode,
    and the single stack is one sample;
  - RMSNorm+RoPE in fp32, as the reference does;
  - `time_in` / `ref_time_in`.
- Write the flow-map scheduler, reference prep (fit, reflect-pad, crop back; multi-ref fit and crop),
  and the latent ids.
- Add the `qwen3vl-mlx-swift` `hiddenStates(layers:)` tap. qwen3vl lives in mlxengine-think; send the
  change as an ask or a PR, as the 2.1 `applyFinalNorm` change was.
- Add the `flux2-vae-mlx-swift` encoder.

**P1 progress (2026-10-17):**
- `Flux3Transformer`, `Flux3Weights` (native `model.safetensors`, two-way strict, 359 tensors) and
  `Flux3Schedule` are in.
- `Flux3KleinGate --dit`, fp32 CPU: **73/73 PASS**. Every hooked stage matches on t2i, ref1 and ref3 at
  cos ≥ 0.9999998, relMax ≤ 2.1e-5. Peak 25.0 GB, ~3 s per forward on the CPU.
- `--run`, fp32 CPU: the 16-step 512² trajectory matches at every step (final relMax 1.6e-4).
- On the GPU in fp32 the same trajectory ends at cos 0.99957: MLX GPU fp32 matmul noise (~1e-5 per forward)
  compounding over 16 steps.
- Deviations from the reference, both equivalent:
  - an empty mode is skipped instead of carrying a phantom token;
  - RoPE tables are built in Double on the CPU (the reference uses float64).
- Still to do: the conditioner tap, reference prep, the pipeline, and the bf16 GPU e2e.

**P1/P2 done (2026-10-17):**
- Conditioner:
  - qwen3vl-mlx-swift 0.5.1 adds `hiddenStates(inputIds:layers:mask:positionIds:)`.
  - Pad rows: every pad row is the same token at position 1 attending to the real keys, so one masked pad row is
    run and broadcast. It is exact against the reference's padded forward.
- VAE: flux2-vae-mlx-swift 0.5.0 (`Flux2VAEEncoder`); 0.6.0 from v0.3.0 (row-band decode).
- Reference prep: byte-exact.
- `Flux3KleinPipeline`.
- bf16 GPU e2e vs BFL: 35.9 / 39.8 / 52.7 / 30.9 dB, where BFL's code on MPS scores 35.8 / 42.5 / 56.2.
- Speed at 1024² on an M5 Max: 1.2 s per step for T2I, 2.3 s per step with one reference.

**P2 — gates** (`Flux3KleinGate`)
- Conditioner: cos 1.0 on text cases.
- VAE: as the FLUX.2 VAE gate.
- DiT: every stage cos ≥ 0.9999, fp32 CPU, on all three layouts.
- Sampler: exact.
- e2e: bf16 GPU 1024² T2I + 1-ref edit, coherent beside the oracle and BFL's PNGs.
- Run `BooguGate --nax-probe`-style checks for the K = 9216 MLP down-projection: the NAX split-K class
  (mlx#3797/#3810) is fixed in mlx-swift 0.32.3, but confirm.

**P3 — performance and quantization**
- Release s/step at 1024² and 2048².
- int8 and int4 DiT, A/B against bf16 (per-pass cosine, SigLIP2 NR-IQA, OCR on the typography examples).
  AB-T-0312 shows int8 can drop letters: gate text with OCR, never by eye.
- Footprint MEASURED in-app: resident + activation, at 0, 1 and 4 references and at the 4 MP single-ref
  cap. Declare per tier.

**P3 done (2026-10-17):**
- **Tiers:**
  - DiT int8 and int4 are made on load. The int4 keeps the attention projections at 8 bits: uniform int4 put the last
    block's hidden state at cos 0.84–0.86.
  - The encoder loads text-only (qwen3vl 0.6.0) and can be int8.
- **OCR gate:** every lane is within the pre-registered tolerance (README).
- **Footprints:** measured under the engine's 2 GB pool cap. bf16 1024² peaks at 31.8 GB; int8 + int8 with the encoder
  dropped at 18.8 GB. The 4 MP single-reference case needs 64 GB.
- **Mirror:** [mlx-community/flux-3-image-klein-bf16](https://huggingface.co/mlx-community/flux-3-image-klein-bf16).
  Gated `auto` like upstream, byte-identical (22 files), card with the Attribution Notice.
- **Not done:**
  - Pre-quantized repos. The download-size optimization stays later, per the fleet convention.
  - Tiled or bf16 VAE decode. About 11 GB of the 1024² peak is the fp32 decode, the lever for 16–24 GB Macs.

**P5a done (2026-10-17): the package.**
- `MLXFlux3Klein` / `Flux3KleinPackage` (textToImage + imageEdit, surface `flux-3-klein`), C7 Kommunity LicenseRef.
- Envelope: 1152² area, measured; single references fitted to it; ≤ 10 refs.
- Footprints declared from the package-path ledger: resident bf16 11.7 / int8 6.7 / int4 4.9 GB, activation
  15.5 GB.
- 11 offline tests (MAT, CAN, manifest) pass; the live `--package` smoke passes at bf16 and int4.
- **P5b done:**
  - mlx-image-runtime 0.9.0: `flux-3-klein` entry, restricted, `.wip`, DiT tier by Mac (bf16 ≥ 48 GB, int8 ≥ 32 GB,
    int4 below; `MLXIMG_FLUX3_QUANT`).
  - Pixel Agency Image Engine 0.2.12: offered with `--allow-research`, named-only on image.generate / image.edit,
    with a Kommunity licence notice.
  - In-app (`WIP/flux3-klein-inapp`):
    - first download from the gated mirror works;
    - bf16 phys peak 22 GB (declared 27.2), int8 17 GB (declared 22.2);
    - a commercial-only request is refused `policy_unmet`;
    - conformance 35/0 warm.
  - Open: per-model prompting notes in MES (ask AB-A-0186 to mlx-apps); ~~the tiled VAE decode (flux2-vae 0.6.0) to
    widen the envelope~~ done in v0.3.0 (below).

**v0.3.0 done (2026-10-17): the row-band decode widens the envelope to 2048² (AB-R-0579).**
- `Flux3KleinPipeline` decodes with flux2-vae-mlx-swift 0.6.0's `decodePackedLatents(_:tile: .standard)` by default
  (`vaeTile`; nil = untiled). No flag: the quality is the same.
  - `--vae` (GPU fp32): the 512² decode golden at cos 1.0000000 / relMax 1.98e-3 tiled, as untiled; tiled vs untiled
    relMax 5.8e-5 (64 MB and 4 MB bands).
  - `--e2e`: PSNR vs BFL unchanged to 0.01 dB (35.88 / 39.80 / 52.70 / 30.91); tiled vs untiled renders
    59.9–62.6 dB.
- Each reference is evaluated as it is encoded. Lazily, ten 1024² encodes ran inside the first DiT step: 10 refs at
  1024² went 14.81 → 8.11 GB activation from this alone (exact: eval only schedules).
- Ledger (`outputs/envelope.sh`, package path, 2 GB pool, `outputs/envelope-v030.log`): 1024² … 2048² × 0 / 1 / 10
  refs, bf16 and int4. The decode is now ~1.7 GB per output MP; 2048² T2I is 7.21 GB activation (was 45.02).
  - What is left is the untiled fp32 VAE **encode** of a single reference, ~4.1 GB per reference MP: a 2048²
    reference is 18.66 GB, which would put int4 at 24.5 GB declared, over a 24 GB Mac's 19.07 GB budget.
- **Envelope:** output ≤ 2048² area; a single reference fitted to 1664² (2.77 MP); ≤ 10 refs. Worst request inside
  it: 11.34 GB (a 1664² reference into any output up to 2048²); 10 refs into 2048²: 10.32 GB.
- **Declared:** activation 12.0 GB on every tier (was 15.5). Totals: bf16 23.7, int8 18.7, int4 16.9 GB.
  phys_footprint peaks: bf16 24.5 GB, int4 17.8 GB (was 28.2 / 21.4 at the 1152² cap).
- Next lever: a row-band VAE **encode** (flux2-vae-mlx-swift) would let a single reference follow the output to 2048².
- **v0.4.0 (2026-10-17):** that encode landed (flux2-vae-mlx-swift 0.7.0, AB-R-0580: 2048² fp32 encode 22.72 → 4.63 GB,
  tiled vs untiled ≤ 1.4e-5). The pipeline's reference encode follows `vaeTile`.
  - `--vae` gate: tiled encode vs BFL is cos 0.999999 / relMax 1.90e-3, the same as untiled; tiled vs untiled 2.5–3.8e-5.
  - Ledger `outputs/envelope-v040.log`: 1 reference at its own size, 1024² 4.29 → 2.35, 1536² 9.66 → 4.38, 2048²
    18.66 → 7.22 GB (int4 the same); 10 refs into 2048² 10.32, unchanged.
  - **Envelope:** a single reference fitted to 2048² (the reference's 4 MiP). The worst request is 10 refs into 2048²,
    10.32 GB, so the declared 12.0 GB holds. phys peaks bf16 22.8 GB, int4 16.0 GB.

**P4 done (2026-10-18).**
- `Flux3KleinUpsampler` on MLXVLM. Prompt tokens are exact vs HF, `bbox.py` is ported (13/13), and one edit caption
  matches BFL's word for word.
- Tiers bf16 / int8 / int4 are on mlx-community; all keep quoted text and box tables. int8 is the package default:
  its 11.1 GB transient sits under the declared activation.
- In the package: per request, with the `upsample` / `boxes` metaData levers (README).

**P4 plan (kept for the record)**
- Convert the upsampler (Qwen3.5-9B VL) to MLX int4/int8 and run it through `MLXVLM`, using
  `system_prompts.json` (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`) with greedy decoding and thinking off.
- Gate the captions against the reference's vLLM captions on the shipped examples. Score the
  downstream image, not the text, as there will be drift.

**P5 — packages and registration** (target: `pixel-agency-image-engine`)
- `Flux3KleinPackage` (textToImage + imageEdit on one core, PackageID `flux-3-klein`), born-conformant to
  C0–C13, with the C7 posture from §1.
- `mlx-image-runtime`: a `CatalogEntry` (`.restricted`, `.wip`, empty `preferredFor`), with measured
  envelopes per capability. A runtime tag follows; ML[X] Image Server gets it through the same runtime.
- `pixel-agency-image-engine`:
  - an `OperationTable` row on `image.generate` and `image.edit`, best tier, by name only. MOSS-Image is the
    precedent (0.2.11).
  - a Kommunity line in `ImageNormalizer.notice`. Today every licence that is neither permissive nor
    eval-only falls through to "Research licence (…): not for commercial use", which misstates this one.
  - then the live `mes-conformance` run and a `mes run image.generate --model flux-3-klein` receipt.
- MES extension for boxes: `x-pixelagency.layout` (a list of `{id, bbox, desc}`), plus the box-edit form.
- **MCP:** the engine's MCP tools are generated from descriptors. The model's offer `summary` carries BFL's
  caption guidance, so a calling agent can *be* the upsampler. The card endorses "any LLM … with a system
  prompt that expands the request". That leaves the 18.8 GB upsampler as an engine-side option, not a
  requirement.

**P6 — safety (the §2(e) condition, if anyone uses the commercial exception)**
- An opt-in output filter: the NSFW classifier, plus the prompt check through the already-loaded TE.
- EXIF/XMP AI-provenance tags, as `safety.py` writes them.
- Default on whenever the offer is used outside research.

**P7 (optional) — Base checkpoint**
- 50 steps, CFG 4.0 (the code) / 3.5 (the card) on a two-sample pack. It is the parity reference for
  quality questions and the LoRA-training base, if that is ever wanted. Training is non-commercial only,
  and §4(b) still applies.

## 5. Open questions for the operator

1. **The licence posture in §1**: restricted and named-only, with `commercial_use: false` and a notice
   naming the Qualifying-User exception. Or should the offer claim the exception for this operator?
   (That would make P6 mandatory.)
2. **Publishing an MLX mirror**: private xocialize only (default), or not at all?
3. ~~Disk~~ — freed by the operator 2026-10-17 (188 GB free). ~~Target~~ — `pixel-agency-image-engine` (the
   AB-D-0151 amendment).
