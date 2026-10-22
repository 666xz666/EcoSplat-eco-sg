# Integrating `sg_wrapper` into the SPFSplat base

Exact list of host-repo changes on the `eco-sg` branch, in the spirit of
`INTEGRATION.md`: every change is gated on `model.encoder.sg` (or on
`model.encoder.pretrained_weights` for the loader) and is inert when the
gate is off — with `sg: null` the model behaves like stock SPFSplat.

All paths relative to the repo root.

## 1. Encoder config + module — `src/model/encoder/encoder_spfsplat.py`

**Config switch** (next to `igf`):

```python
sg: Optional[dict] = None  # sg_wrapper spherical-Gaussian color cfg; None = stock SH.
```

**Adapter selection** (`__init__`): when `cfg.sg` is set, build
`sg_wrapper.UnifiedSGGaussianAdapter` instead of `UnifiedGaussianAdapter`.
The SG adapter subclasses the stock adapter and keeps its
`forward(means, opacities, raw_gaussians)` signature, so neither head
construction (`raw_gs_dim = 1 + adapter.d_in`, 83 → 32) nor the two
`self.gaussian_adapter.forward(...)` call sites in `forward` change.
`sg` and `igf` are mutually exclusive (asserted).

**Partial stage-1 init** (`__init__`, after the IGF block): when
`cfg.pretrained_weights` is set and `cfg.igf is None`,
`sg_wrapper.init_encoder_from_stage1(self, cfg.pretrained_weights)` loads
backbone / center / pose heads plus the param-head trunk, and aligns the
leading 11 channels of the resized final convs. Serves the SG head and the
SH-degree baselines alike.

**Training guard** (`forward`): the stock branch refuses to train without
IGF; the guard now also accepts the partial-init path
(`pretrained_weights` set, `igf` None).

**Flattening** (`_flatten_gaussians` closure): routes to
`sg_wrapper.flatten_sg_gaussians` when `self.sg` is set; stock
`types.Gaussians` path untouched otherwise.

## 2. Decoder — `src/model/decoder/decoder_splatting_cuda.py`

`forward` gains an `isinstance(gaussians, SGGaussians)` branch: it repeats
the SG color fields to `(b v) g …`, evaluates per-view colors with
`sg_wrapper.compute_sg_colors` (view directions from the *unscaled*
extrinsics — uniform scale-invariant rescaling does not change directions),
and calls `render_cuda` with `gaussian_sh_coefficients=None` and
`gaussian_colors=…`. The SH branch is byte-identical to stock.

## 3. Rasterizer wrapper — `src/model/decoder/cuda_splatting.py`

`render_cuda` gains one optional kwarg:

```python
gaussian_colors: Float[Tensor, "batch gaussian 3"] | None = None
```

When set: `shs = None`, `degree = None`, and the gsplat call passes
`colors_precomp=gaussian_colors[i][None]` with `sh_degree=None`. The SH
path is unchanged (same arguments as before). `render_cuda_orthographic`
(figure-only escape hatch) is untouched and still expects SH.

## 4. Entry point — `src/main.py`

`main` loads `model.encoder.pretrained_weights` into the encoder itself when
training (strict=False). A second load after the partial init would raise
on the resized parameter heads, so the condition gains one guard:

```python
and not getattr(encoder, "_stage1_partial_loaded", False)
```

The flag is set by the encoder right after `init_encoder_from_stage1`
runs. Stock behavior (IGF configs, matching shapes) is unchanged.

## 5. Configs + scripts (new files, no edits)

* `config/experiment/spfsplat/re10k_10view_sg.yaml` — SG 3-lobe variant
* `config/experiment/spfsplat/re10k_10view_sh1.yaml` — SH degree-1 baseline
* `config/experiment/spfsplat/re10k_10view_sh2.yaml` — SH degree-2 baseline
* `train_sg.sh` — variant launcher (no Lightning resume; init happens in
  the encoder via `pretrained_weights`)

## 6. What is *not* changed

* `ecosplat_wrapper/` (IGF) — untouched; `sg` and `igf` never combine.
* Losses, dataset, optimizer, Lightning plumbing — untouched.
* `train.freeze_pretrained` / `train.freeze_pose_head` /
  `train.freeze_intrinsic_encoder` define what trains: only
  `gaussian_param_head*`. The third switch is required because
  `freeze_pretrained` unfreezes `intrinsic_encoder` at the ViT input, which
  keeps the whole backbone graph alive; it is identical across variants.
