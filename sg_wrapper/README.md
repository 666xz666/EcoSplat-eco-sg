# sg_wrapper — Spherical-Gaussian colors for feed-forward 3DGS

Vendored, base-agnostic-ish package that swaps the SH color parametrization
of the EcoSplat/SPFSplat feed-forward head for MEGS²-style spherical
Gaussians (SG), and provides the partial stage-1 initialization used by the
representation-swap experiments. Mirrors the design of `ecosplat_wrapper/`:
all new logic lives here; the host repo is touched only through small,
config-gated hooks (see `INTEGRATION_SG.md`).

## What it contains

| Module | Purpose |
|---|---|
| `sg_config.py` | `SGAdapterCfg` — lobes per Gaussian, init scales, clamp switch |
| `sg_adapter.py` | `UnifiedSGGaussianAdapter` (drop-in for `UnifiedGaussianAdapter`), `SGGaussians`, `flatten_sg_gaussians` |
| `sg_colors.py` | `compute_sg_colors` — batched port of MEGS² `compute_colors_precomp` |
| `stage1_loader.py` | `init_encoder_from_stage1` — partial init from a stage-1 (PGT) checkpoint |
| `tests/test_sg_colors.py` | CPU sanity checks vs a literal transcription of the MEGS² formula |

## Method summary

Per-Gaussian view-dependent color, with v = normalize(camera_center − mean):

    color(v) = SH2RGB( c0 + Σ_l a_l · exp(s_l · (μ_l · v − 1)) )

* `c0` — diffuse term, in SH space (same convention as the SH path's DC
  term, which is what makes warm-starting from a stage-1 SH checkpoint
  possible).
* Per lobe: axis μ_l (normalized head output), sharpness s_l = |raw| (MEGS²
  convention), amplitude a_l = raw · `rgb_scale` (default 0.1 — the
  feed-forward analog of MEGS²'s `rgb × 0.1` initialization).
* Head output width: `1 + 7 + 3 + 7·L` = 32 channels for L = 3 (SH4: 83).
* Colors are evaluated in Python and passed to gsplat via
  `colors_precomp` — no custom CUDA kernels anywhere.

Deliberate deviations from MEGS² (both configurable in `SGAdapterCfg`):
* MEGS² clamps colors to [0, 1] during training; the SH path does not, so
  `clamp_colors` defaults to `False` for a fair comparison.
* MEGS² stores a per-point lobe count; the feed-forward head predicts a
  fixed count per point (no masking needed at this stage).

## Install

```bash
pip install -e sg_wrapper
```

## Usage

Enable in an experiment yaml (see `config/experiment/spfsplat/re10k_10view_sg.yaml`):

```yaml
model:
  encoder:
    pretrained_weights: pretrained_weights/re10k_10view.ckpt
    sg: { num_lobes: 3 }
train:
  freeze_pretrained: true   # only gaussian_param_head* (+intrinsic_encoder) train
  freeze_pose_head: true
```

`sg: null` (default) keeps stock SH behavior — every hook is inert.

## Experiments

Three variants share one recipe (stage-1 partial init, heads-only training,
100k steps, batch 1 × accum 4):

| Variant | Color params / Gaussian | Config |
|---|---|---|
| SG 3-lobe (main) | 24 | `re10k_10view_sg.yaml` |
| SH degree 1 (lower squeeze) | 12 | `re10k_10view_sh1.yaml` |
| SH degree 2 (upper squeeze) | 27 | `re10k_10view_sh2.yaml` |

```bash
./train_sg.sh sg3   # or sh1 / sh2
```

The SG parametrization must beat both SH baselines at matched training
budget for the representation claim to hold (its parameter count lies
between them).

## Partial stage-1 init (`stage1_loader`)

`init_encoder_from_stage1(encoder, ckpt)` loads a stage-1 (PGT) checkpoint:

1. every name- and shape-matching tensor is copied — backbone, center
   heads, pose head, and the DPT trunk of the parameter heads;
2. the parameter heads' final pointwise conv (the only layer whose width
   depends on the parametrization) gets its leading 11 output channels
   copied — density (1), scale (3), rotation (4), c0/SH-DC (3) — which are
   semantically aligned across SH and SG layouts; the remaining channels
   (higher SH degrees / SG lobes) keep their fresh initialization;
3. everything else is reported and left alone.

The same loader serves the SG head (83 → 32) and the SH baselines
(83 → 20 / 35), so all variants get the identical warm-start recipe.

## Known limitations

* Only the pose-free (`UnifiedGaussianAdapter`) path is supported; the
  pose-free config is the default for this base.
* PLY export and the orthographic render path still expect SH — they are
  unused by the training/eval protocol of these experiments.
* `intrinsic_encoder` remains trainable under `freeze_pretrained` (host
  behavior); identical across variants, so comparisons stay fair.
