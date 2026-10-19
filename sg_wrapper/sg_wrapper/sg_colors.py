"""Batched spherical-Gaussian color evaluation (feed-forward port of MEGS²).

Ported from MEGS-2 `scene/spherical_gaussian_model.py::compute_colors_precomp`
(the `is_training or not variable_sg_bands` branch), vectorized over an
arbitrary leading batch of cameras so it can run inside the EcoSplat decoder:

    color(v) = SH2RGB( c0 + sum_l a_l * exp(s_l * (mu_l . v - 1)) )

Conventions kept identical to MEGS²:
  * c0 / a_l live in "SH space"; the final color applies SH2RGB (x*C0 + 0.5),
    matching the DC-term convention of the SH path used by gsplat.
  * view direction v = normalize(camera_center - mean)  (point -> camera).
  * sharpness enters as |s| (abs activation), guaranteeing a decaying lobe.

Differences from MEGS², both configurable:
  * MEGS² clamps the final color to [0, 1] during training as well; the SH
    path in EcoSplat/gsplat does not clamp per-Gaussian colors. For a fair
    SG-vs-SH comparison the clamp is off by default here (`clamp_colors`).
  * MEGS² stores a per-point lobe count; the feed-forward head predicts a
    fixed number of lobes per point, so no masking is needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Same constant as 3DGS / MEGS² sh_utils.
C0 = 0.28209479177387814


def compute_sg_colors(
    means: Tensor,
    rgb_base: Tensor,
    sg_directions: Tensor,
    sg_sharpness: Tensor,
    sg_rgb: Tensor,
    extrinsics: Tensor,
    clamp_colors: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate per-Gaussian, per-camera colors from SG parameters.

    Args:
        means: (B, G, 3) Gaussian centers in world space.
        rgb_base: (B, G, 3) diffuse term c0, in SH space.
        sg_directions: (B, G, L, 3) raw lobe axes (any norm; normalized here).
        sg_sharpness: (B, G, L, 1) lobe sharpness, already activated (>= 0).
        sg_rgb: (B, G, L, 3) lobe amplitudes, in SH space.
        extrinsics: (B, 4, 4) camera-to-world matrices; camera center is
            extrinsics[:3, 3] (uniform scene scaling does not change view
            directions, so pre- or post-scale-invariant coordinates agree).
        clamp_colors: clamp output to [0, 1] (MEGS² behavior).

    Returns:
        (B, G, 3) RGB colors in [0, 1]-ish range (clamped if requested),
        ready for gsplat's `colors_precomp`.
    """
    camera_center = extrinsics[:, :3, 3]  # (B, 3)
    # view direction per Gaussian: point -> camera, normalized.
    view_dirs = camera_center[:, None, :] - means  # (B, G, 3)
    view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + eps)

    dirs = F.normalize(sg_directions, dim=-1, eps=eps)  # (B, G, L, 3)
    # cos_theta = mu . v, broadcast over lobes.
    cos_theta = torch.einsum("bgli,bgi->bgl", dirs, view_dirs)  # (B, G, L)
    directional_scale = torch.exp(sg_sharpness * (cos_theta.unsqueeze(-1) - 1.0))

    weighted_rgb = sg_rgb * directional_scale  # (B, G, L, 3)
    colors = rgb_base + weighted_rgb.sum(dim=2)  # (B, G, 3)

    colors = colors * C0 + 0.5  # SH2RGB
    if clamp_colors:
        colors = colors.clamp(0.0, 1.0)
    return colors
