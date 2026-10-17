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

| Piece | State |
|---|---|
| DiT (`Flux3Transformer`) | **Gate green, fp32 CPU.** Every stage on t2i, 1-ref and 3-ref matches at cos ≥ 0.9999998 (relMax ≤ 2.1e-5). The 16-step 512² trajectory matches at relMax ≤ 1.7e-4 |
| Flow-map schedule | Gate green (exact). α = 6.93 from `config.json`; the reference CLI's 16.0 does not reproduce BFL's renders |
| Conditioner (Qwen3-VL-4B, 8 taps → 20480) | P1, next. Needs a `hiddenStates(layers:)` tap in qwen3vl-mlx-swift |
| VAE | No port: FLUX 3's VAE is the FLUX.2 VAE, identical weights. Uses flux2-vae-mlx-swift |
| Pipeline, reference prep, MLXEngine package | P1–P5 |

## Gate

```bash
swift build -c release --product Flux3KleinGate
.build/release/Flux3KleinGate --sched <goldens>
.build/release/Flux3KleinGate --dit <snapshot> <goldens>        # fp32 CPU; --gpu for the same math on the GPU
.build/release/Flux3KleinGate --run <snapshot> <goldens>        # the 16-step trajectory
```

`<snapshot>` holds BFL's `model.safetensors` (the native DiT, 359 tensors, sha256 in `manifest.json`).

**On the GPU in fp32**, one forward is still exact (relMax ~1e-5). Over the 16 chained steps, though, MLX's GPU fp32
matmul drifts to cos 0.99957 at the last latents. That is compounding precision, not a port defect: the same code on
the CPU stays at 1.6e-4. bf16 on the GPU is the production regime, and it is judged at image level against BFL's
example renders.
