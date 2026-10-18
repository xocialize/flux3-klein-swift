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
layouts in the caption. Next comes the MLXEngine package and its registration in Pixel Agency Image Engine
(PORTING-SPEC.md P3–P5).

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
```

`<snapshot>` is the HF repo's bf16 package: `model.safetensors` (the native DiT, 359 tensors, sha256 in `manifest.json`),
`text_encoder/` and `vae/`.

**On the GPU in fp32**, one forward is still exact (relMax ~1e-5). Over the 16 chained steps, though, MLX's GPU fp32
matmul drifts to cos 0.99957 at the last latents. That is compounding precision, not a port defect: the same code on
the CPU stays at 1.6e-4. bf16 on the GPU is the production regime, and it is judged at image level against BFL's
example renders.
