# flux3-klein-swift

Swift/MLX port of [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein):
text-to-image, single-reference editing and multi-reference composition (up to 10 references) in one model.
- **DiT:** 5.55B parameters. Each mode (image, references, text) has its own stack of 5 blocks; then come 28 shared
  single-stream blocks with per-token modulation, sampled as a 16-step flow map.
- **Conditioner:** 8 hidden-state taps of Qwen3-VL-4B-Instruct.
- **VAE:** the FLUX.2 VAE.

> **Licence: the FLUX 3 Klein weights are under the FLUX Kommunity License v1.0.**
> - It allows non-commercial use.
> - It also allows commercial use of *outputs* by a "Qualifying User" (under US$5M annual revenue, affiliates
>   included) who filters content and discloses AI generation (§2(a), §2(e)).
> - Outputs must never be used to train a similar model (§4(b)).
>
> The Swift port code in this repository is MIT (`LICENSE`). That does not change the weights' terms.
>
> **Status: work in progress, a restricted tier, not a product asset.** It will be offered by
> [Pixel Agency Image Engine](https://github.com/xocialize/pixel-agency-image-engine) by name only, behind
> `--allow-research`, and never routed by default. See `PORTING-SPEC.md` §1 and fleet decision AB-D-0151.

The reference is BFL's own code ([github.com/black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein)
at `bfc8889`, Apache-2.0), not diffusers. The spec, the architecture delta from FLUX.2 klein and the plan are in
`PORTING-SPEC.md`. The oracle and goldens are in `mlxengine-image/WIP/flux3-klein-oracle`.

## Status

The engine-free pipeline is **done and gated** (`Flux3KleinPipeline`): T2I, single- and multi-reference edits, box
layouts in the caption, and three DiT tiers plus an int8 encoder tier, all within OCR tolerance of bf16. The MLXEngine
package is `MLXFlux3Klein` (below); its registration in Pixel Agency Image Engine is PORTING-SPEC.md P5.

**Weights:** [mlx-community/flux-3-image-klein-bf16](https://huggingface.co/mlx-community/flux-3-image-klein-bf16)
(gated, FLUX Kommunity License): a byte-identical copy of BFL's bf16 package. Quant tiers are made at load.

| Piece | Gate |
|---|---|
| DiT (`Flux3Transformer`) | fp32 CPU: every stage on t2i, 1-ref and 3-ref at cos ≥ 0.9999998 (relMax ≤ 2.1e-5); the 16-step 512² trajectory at relMax ≤ 1.7e-4 |
| Flow-map schedule | Exact. α = 6.93 from `config.json`; the reference CLI's 16.0 does not reproduce BFL's renders |
| Conditioner (`Flux3TextEncoder`, Qwen3-VL-4B, 8 taps → 20480) | fp32 CPU: token ids exact; features cos ≥ 0.99998 on real and padded rows (the pad rows come from one masked pad token, broadcast) |
| VAE (flux2-vae-mlx-swift ≥ 0.5.0, the same weights as FLUX.2) | Encoder tokens cos 0.999999; 512² decode cos 0.9999999 (GPU fp32) |
| Reference prep (`Flux3References`) | Byte-identical to the reference on 5 cases: reflect pad, LANCZOS fit > 4 MiP, symmetric tiny pad, multi-ref resize + crop, multi-ref crop |
| End to end, bf16 GPU (`--e2e`) | Replaying the reference's torch noise, PSNR against BFL's own example renders, the same level as BFL's own code on MPS (table below) |

| BFL example | Swift bf16 vs BFL | torch MPS oracle vs BFL | M5 Max, 16 steps |
|---|---|---|---|
| knitted diorama, T2I + boxes, 1184×896 | 35.9 dB | 35.8 dB | 18.9 s |
| coffee + croissant, 1-ref edit, 1024² | 39.8 dB | 42.5 dB | 37.1 s |
| mug print, 2-ref composition, 1024² | 52.7 dB | 56.2 dB | 61.7 s |
| figurine on a desk, 1-ref edit, 896×1120 | 30.9 dB | — | 44.3 s |

Prompting: the model wants a **full scene caption**. Run a prompt upsampler first: BFL's
[flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), or any LLM given its
`system_prompts.json`. Boxes go after the caption as a JSON list on a 0–1000 grid.

## MLXEngine package (`MLXFlux3Klein`)

`Flux3KleinPackage` serves `textToImage` and `imageEdit` (surface `flux-3-klein`) on one core. C0–C13:
- **C7 licence:** `LicenseRef-FLUX-Kommunity-1.0`, package-local, never allowlisted (AB-D-0151). A `.permissiveOnly`
  engine refuses or flags it.
- **C8:** the port code is MIT.
- **Weights:** `Flux3KleinConfiguration` materializes them from the gated mlx-community mirror, pinned by commit. Needs an
  HF token whose account accepted the licence.
- **Quant:** `quant` takes `.bf16`, `.int8` or `.int4`, made at load; `textEncoderQuant` sets the encoder tier.
- **Memory path:** the encoder loads per request and is released before denoising, unless `keepEncoderResident` is set.

**Envelope** (refused, not clamped, in `run()`):
- output area ≤ **1152²** (1.33 MP), sides multiples of 16, at least 64 px;
- a single reference is fitted to the same area;
- up to 10 references.

**Footprints** (MLX pool, measured on the package path at the cap, 2 GB pool cap; `Flux3KleinFootprints`):

| DiT tier | Resident | Activation |
|---|---|---|
| bf16 | 11.7 GB | 15.5 GB (worst: 10 refs at 1152²) |
| int8 | 6.7 GB | 15.5 GB |
| int4 | 4.9 GB | 15.5 GB |

Most of the activation is the fp32 VAE decode, about 10.7 GB per output MP, which is why the cap sits at 1.33 MP.
Tests: 11 offline (manifest licence posture, footprints, envelope, MAT, CAN-1/2/3, configuration). Live:
`Flux3KleinGate --package` (load, T2I, edit, envelope refusal) passes at bf16 and int4.

## Quant tiers (P3)

`Flux3KleinPipeline.load(snapshot:quant:textEncoderQuant:)` builds every tier on load from the bf16 files.
- DiT: `.bf16`, `.int8`, `.int4`, affine group 64. Embeddings, timestep MLPs, modulations and the final layer stay
  bf16. `.int4` keeps every block's attention projections at 8 bits (`Flux3Weights.int4EightBitLayers`); six recipes
  were probed by single-forward parity.
- Encoder: `.bf16` or `.int8`, always text-only (qwen3vl-mlx-swift ≥ 0.6.0).

**Text fidelity: an OCR gate** (`mlxengine-image/WIP/flux3-klein-ocr-gate`). 20 long-text EN/ZH prompts × 2 seeds per
lane, Apple Vision OCR, verdict rule pre-registered. Every lane is within tolerance of bf16:

| DiT + encoder | ΔCER vs bf16 [95 % CI] | exact strings |
|---|---|---|
| bf16 + bf16 | — (CER 3.00 %, EN 0.19 %) | 68.7 % (EN 95.8 %) |
| int8 + bf16 | −0.39 pp [−0.85, −0.05] | 72.4 % |
| int4 (8-bit attn) + bf16 | +0.10 pp [−0.69, +0.91] | 71.6 % |
| bf16 + int8 | −0.05 pp [−0.32, +0.21] | 69.4 % |
| int8 + int8 | +0.05 pp [−0.33, +0.46] | 68.7 % |

int4 keeps text, but at a given seed it renders a visibly different image from bf16. BFL's examples sit 21.9–35.5 dB
from bf16 under int4, against 31–50 dB under int8.

**Memory** (M5 Max; MLX with the engine's 2 GB buffer-pool cap; phys_footprint peak via `outputs/footprint.sh`):

| DiT + encoder | Resident (MLX) | 1024² peak | Peak, encoder dropped after encoding |
|---|---|---|---|
| bf16 + bf16 | 19.5 GB | 31.8 GB | 24.1 GB |
| int8 + bf16 | 14.5 GB | 26.9 GB | — |
| int4 + bf16 | 12.7 GB | 25.1 GB | — |
| bf16 + int8 | 16.1 GB | 28.4 GB | — |
| int8 + int8 | 11.1 GB | 23.5 GB | 18.8 GB |

- About 11 GB of each 1024² peak is the fp32 VAE decode, the same in every tier.
- 1 and 4 references at 1024² peak the same as T2I.
- A single reference at the 4 MP cap (2048² canvas and output) peaks at 51–58 GB.
- Speed is the same in every tier: about 1 s per step for T2I, 2.3 s with one reference, 10 s with four, 20 s at the
  4 MP cap.

## Gate

```bash
swift build -c release --product Flux3KleinGate
.build/release/Flux3KleinGate --sched <goldens>
.build/release/Flux3KleinGate --encoder <snapshot> <goldens>    # conditioner, fp32 CPU
.build/release/Flux3KleinGate --vae <snapshot> <goldens> --gpu  # encoder + decoder
.build/release/Flux3KleinGate --prep <goldens>/prep             # reference prep, byte-exact
.build/release/Flux3KleinGate --e2e <snapshot> <oracleDir> <example.json>...  # bf16 GPU vs BFL renders
.build/release/Flux3KleinGate --dit <snapshot> <goldens>        # fp32 CPU; --gpu for the same math on the GPU
.build/release/Flux3KleinGate --run <snapshot> <goldens>        # the 16-step trajectory
.build/release/Flux3KleinGate --encode-cache / --render-cache …  # the OCR-gate lanes (see the ocr-gate README)
.build/release/Flux3KleinGate --oneshot <snapshot> <oracleDir> --dit int8 --te bf16 --case t2i  # footprint ledger
```

`<snapshot>` is the HF repo's bf16 package: `model.safetensors` (the native DiT, 359 tensors, sha256 in `manifest.json`),
`text_encoder/` and `vae/`.

**On the GPU in fp32**, one forward is still exact (relMax ~1e-5). Over the 16 chained steps, though, MLX's GPU fp32
matmul drifts to cos 0.99957 at the last latents. That is compounding precision, not a port defect: the same code on
the CPU stays at 1.6e-4. bf16 on the GPU is the production regime, and it is judged at image level against BFL's
example renders.
