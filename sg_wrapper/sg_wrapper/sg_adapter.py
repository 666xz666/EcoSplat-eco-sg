"""Spherical-Gaussian parametrization for the EcoSplat Gaussian adapter.

Drop-in replacement for ``UnifiedGaussianAdapter`` (the pose-free variant
used by the SPFSplat base): same ``forward(means, opacities, raw_gaussians)``
signature, same geometry, only the color parametrization changes —

    SH:  raw[..., 7:]  -> 3 x (2k+1)^2 spherical-harmonic coefficients
    SG:  raw[..., 7:]  -> c0 (3) + L x [direction (3), sharpness (1), rgb (3)]

so the encoder's head/adapter call sites do not change; only the head's
output width (``raw_gs_dim``) and the decoder's color path do.

Channel layout of the head output (shared prefix with the SH layout, which
is what makes stage-1 warm-starting possible — see ``stage1_loader``):

    [0]      density (opacity logit)
    [1:4]    scale features
    [4:8]    rotation features (quaternion)
    [8:11]   c0 / SH DC term (identical semantics in both parametrizations)
    [11:]    L lobes x (dir 3, sharpness 1, rgb 3)   [SH: higher-degree coeffs]

``SGGaussians`` is created as a subclass of the host's
``src.model.types.Gaussians``: every beartype/jaxtyping runtime check in the
host (decoder, losses, visualizer) annotates that type, and a subclass
passes all of them without touching a single host annotation. The class is
built lazily (creating it imports the host repo), cached, and exposed
through the package's ``__getattr__`` so that merely importing
``sg_wrapper`` stays torch-only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .sg_config import SGAdapterCfg

if TYPE_CHECKING:
    from jaxtyping import Float


_SG_GAUSSIANS_CLS = None


def _get_sg_gaussians_class():
    """Build (once) SGGaussians as a subclass of the host's Gaussians.

    Field order mirrors ``src/model/types.py::Gaussians`` and appends the SG
    color fields; ``harmonics`` (inherited) carries c0 as a pseudo degree-0
    SH term so visualization code reading ``harmonics[..., 0]`` shows the
    diffuse color unchanged. Leading dims are ``(b v r srf spp)`` right
    after the adapter and ``(b g)`` after :func:`flatten_sg_gaussians`.
    """
    global _SG_GAUSSIANS_CLS
    if _SG_GAUSSIANS_CLS is None:
        from src.model.types import Gaussians

        @dataclass
        class SGGaussians(Gaussians):
            rgb_base: Tensor = None  # (*batch, 3) diffuse term, SH space
            sg_directions: Tensor = None  # (*batch, L, 3) lobe axes
            sg_sharpness: Tensor = None  # (*batch, L, 1) activated (>= 0)
            sg_rgb: Tensor = None  # (*batch, L, 3) lobe amplitudes, SH space
            # Carried along so the decoder evaluates colors with the same
            # setting the adapter was configured with (MEGS2 clamps; the SH
            # path does not).
            clamp_colors: bool = False
            # Optional packed color payload (set by sg_prune.pack_into when a
            # byte budget is active). When present, the padded sg_* tensors
            # above are dropped and the decoder evaluates colors from these.
            sg_packed_directions: Tensor = None  # (K, 3)
            sg_packed_sharpness: Tensor = None  # (K, 1)
            sg_packed_rgb: Tensor = None  # (K, 3)
            sg_packed_index: Tensor = None  # (K,) long
            sg_lobe_counts: Tensor = None  # (G,) long
            sg_num_lobes: int = 0

        _SG_GAUSSIANS_CLS = SGGaussians
    return _SG_GAUSSIANS_CLS


_ADAPTER_BASE = None


def _get_base_adapter_class():
    # Imported lazily and cached: resolving the base class pulls in the host
    # repo (src/...) and its deps, which must happen only when an adapter is
    # actually built — importing this package stays torch-only.
    global _ADAPTER_BASE
    if _ADAPTER_BASE is None:
        from src.model.encoder.common.gaussian_adapter import (
            UnifiedGaussianAdapter,
        )

        _ADAPTER_BASE = UnifiedGaussianAdapter
    return _ADAPTER_BASE


def UnifiedSGGaussianAdapter(gaussian_adapter_cfg, sg_cfg: SGAdapterCfg):
    """Build a ``UnifiedGaussianAdapter`` whose SH readout is swapped for SG.

    Called exactly like the stock adapter class
    (``UnifiedSGGaussianAdapter(cfg.gaussian_adapter, sg_cfg)``); the class
    is created on demand so that importing this module does not require the
    host repo.
    """

    class _UnifiedSGGaussianAdapter(_get_base_adapter_class()):
        """``UnifiedGaussianAdapter`` with the SH color readout swapped for SG."""

        def __init__(self, gaussian_adapter_cfg, sg_cfg: SGAdapterCfg) -> None:
            # The parent builds an sh_mask buffer sized by sh_degree; force
            # degree 0 so the buffer is trivial (it is unused on the SG path).
            super().__init__(replace(gaussian_adapter_cfg, sh_degree=0))
            self.sg_cfg = sg_cfg
            self.num_lobes = sg_cfg.num_lobes

        @property
        def sg_dim(self) -> int:
            # c0 + per lobe: direction (3) + sharpness (1) + rgb (3).
            return 3 + self.num_lobes * 7

        @property
        def d_in(self) -> int:
            # scale (3) + rotation (4) + SG color channels.
            return 7 + self.sg_dim

        def forward(self, means, opacities, raw_gaussians, eps: float = 1e-8):
            scales, rotations, sg_raw = raw_gaussians.split(
                (3, 4, self.sg_dim), dim=-1
            )

            # Geometry: identical to UnifiedGaussianAdapter.
            scales = 0.001 * F.softplus(scales)
            scales = scales.clamp_max(0.3)
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

            # Color: SG readout.
            c0, dirs_raw, sharpness_raw, rgb_raw = sg_raw.split(
                (3, self.num_lobes * 3, self.num_lobes * 1, self.num_lobes * 3),
                dim=-1,
            )
            lobe_shape = (*opacities.shape, self.num_lobes)
            dirs = F.normalize(
                dirs_raw.reshape(*lobe_shape, 3), dim=-1, eps=eps
            )  # (*batch, L, 3)
            # |s| activation keeps lobes decaying (MEGS² convention).
            sharpness = (
                sharpness_raw.reshape(*lobe_shape, 1).abs()
                * self.sg_cfg.sharpness_scale
            )  # (*batch, L, 1)
            rgb = rgb_raw.reshape(*lobe_shape, 3) * self.sg_cfg.rgb_scale

            from src.model.encoder.common.gaussians import build_covariance

            covariances = build_covariance(scales, rotations)

            SGGaussians = _get_sg_gaussians_class()
            return SGGaussians(
                means=means,
                covariances=covariances,
                rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
                scales=scales,
                harmonics=c0.unsqueeze(-1),  # pseudo-DC for visualization
                opacities=opacities,
                rgb_base=c0,
                sg_directions=dirs,
                sg_sharpness=sharpness,
                sg_rgb=rgb,
                clamp_colors=self.sg_cfg.clamp_colors,
            )

    return _UnifiedSGGaussianAdapter(gaussian_adapter_cfg, sg_cfg)


def flatten_sg_gaussians(g):
    """(b v r srf spp ...) -> (b g ...).

    Equivalent to the encoder's einops flattening: every pattern is an
    adjacent-dim merge, so a plain reshape reproduces it.
    """
    b = g.means.shape[0]
    SGGaussians = _get_sg_gaussians_class()
    return SGGaussians(
        means=g.means.reshape(b, -1, 3),
        covariances=g.covariances.reshape(b, -1, 3, 3),
        rotations=g.rotations.reshape(b, -1, 4),
        scales=g.scales.reshape(b, -1, 3),
        harmonics=g.harmonics.reshape(b, -1, 3, 1),
        opacities=g.opacities.reshape(b, -1),
        rgb_base=g.rgb_base.reshape(b, -1, 3),
        sg_directions=g.sg_directions.reshape(b, -1, g.sg_directions.shape[-2], 3),
        sg_sharpness=g.sg_sharpness.reshape(b, -1, g.sg_sharpness.shape[-2], 1),
        sg_rgb=g.sg_rgb.reshape(b, -1, g.sg_rgb.shape[-2], 3),
        clamp_colors=g.clamp_colors,
    )
