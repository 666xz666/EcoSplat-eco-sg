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
| `sg_prune.py` | post-hoc lobe budget: importance, global top-K, closed-form c0 compensation, storage accounting |
| `tests/test_sg_colors.py` | CPU sanity checks vs a literal transcription of the MEGS² formula |
| `tests/test_sg_prune.py` | CPU checks for budget counts, top-K, compensation, edge cases, accounting |

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
  fixed count per point. A per-point budget is applied after the fact by
  `sg_prune` (see below).

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
  freeze_pretrained: true   # only gaussian_param_head* train
  freeze_intrinsic_encoder: true
  freeze_pose_head: true
```

`sg: null` (default) keeps stock SH behavior — every hook is inert.

## Experiments

These variants share one recipe (stage-1 partial init, heads-only training,
100k steps, batch 1 × accum 4):

| Variant | Color params / Gaussian | Config |
|---|---|---|
| SG 3-lobe (main) | 24 | `re10k_10view_sg.yaml` |
| SH degree 1 (lower squeeze) | 12 | `re10k_10view_sh1.yaml` |
| SH degree 2 (upper squeeze) | 27 | `re10k_10view_sh2.yaml` |
| SH degree 4 (same-recipe reference) | 75 | `re10k_10view_sh4.yaml` |

```bash
./train_sg.sh sg3   # or sh1 / sh2 / sh4
```

The SH4 run is optional: the released stage-1 checkpoint is a different-
recipe anchor, so this run exists only to put the whole SH family on equal
training footing.

The SG parametrization must beat both SH baselines at matched training
budget for the representation claim to hold (its parameter count lies
between them).

## Post-hoc lobe budget (ρ_lobe)

`sg_prune.py` is the inference-time half of the second control axis. For each
scene it ranks every lobe with MEGS²'s range metric and keeps a global budget:

    D_l = mean_c |a_lc| · (1 − e^{−2 s_l})
    keep round(L̄ · G) lobes with the largest D
    Δc0 = a_l · (1 − e^{−2 s_l}) / (2 s_l)      # exact sphere-mean compensation

```python
from sg_wrapper import prune_lobes, storage_report

pruned, stats = prune_lobes(g, avg_lobes=1.0)    # g: SGGaussians
stats["avg_lobes"], stats["histogram"]           # actual L̄, counts for d=0..L
storage_report(pruned)                           # 14 + 7·L̄ floats per point
```

Removed lobes get zero amplitude, so the padded tensors stay rectangular;
their sphere-average color is folded into `c0`, which preserves the average
view color exactly. Geometry is untouched.

Evaluation sweeps use the decoder hook, which reads `SG_LOBE_AVG` (or
`set_lobe_budget`) — one checkpoint yields the whole curve:

```bash
for b in 3 2 1 0; do
  SG_LOBE_AVG=$b ./eval_sg.sh sg3 CKPT=outputs/.../checkpoint.ckpt
done
```

`eval_sg.sh` defaults to `NUM_CONTEXT=24`, the official RE10K protocol: the
model is trained with 10 context views and evaluated with 24 (see
`eval_igf_rho0p02.sh` and the paper's 16/20/24-view tables). The index stores
2 context views; the sampler adds random interior views up to the requested
number. Set `NUM_CONTEXT=2` for ACID, or `10` for a training-matched ablation.

Storage is accounted from the kept count (`14 + 7L̄` floats: position 3,
opacity 1, scale 3, rotation 4, c0 3, plus 7 per lobe), which is what a
grouped export writes. Padded GPU tensors still occupy `3L` slots until the
packed layout below is enabled, so storage and render-VRAM savings are
reported separately.

### Byte budget and packed storage

The control quantity can be a target payload size instead of L̄:

    B = 4 · (14·N + 7·L_total)
    L_total = (B/4 − 14·N) / 7

`pack_lobes_to_bytes(g, B)` selects exactly `L_total` lobes globally and
returns a `PackedLobes` whose lobe tensors are `[K, …]` (only kept lobes)
plus a `point_index` map; `packed_bytes()` reports the actual payload. The
decoder hook accepts the same budget through the environment:

```bash
SG_PACKED=1 SG_BUDGET_MB=80 ./eval_sg.sh sg3 CKPT=...
```

With `SG_PACKED=1`, `apply_runtime_lobe_budget` drops the padded
`sg_directions/sg_sharpness/sg_rgb` tensors and evaluates colors with
`compute_packed_sg_colors`, so the VRAM reduction is real rather than only
accounting. `SG_LOBE_AVG` remains available as the average-lobe shorthand;
if both are set, `SG_LOBE_AVG` wins.

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
* `freeze_pretrained` alone still unfreezes `intrinsic_encoder`, whose
  gradient keeps the whole ViT graph alive (~23 GB at 256², 10 views). All
  variant configs therefore set `train.freeze_intrinsic_encoder: true`;
  the setting is identical across variants, so comparisons stay fair.
