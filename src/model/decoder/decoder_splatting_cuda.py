from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    enable_cov_grad: bool
    enable_sh_grad: bool

class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        # dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.enable_cov_grad = cfg.enable_cov_grad
        self.enable_sh_grad = cfg.enable_sh_grad
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        # sg_wrapper: SG-parametrized Gaussians carry lobe color parameters
        # instead of SH coefficients; evaluate per-view colors in Python and
        # hand them to the rasterizer as precomputed colors. SGGaussians
        # subclasses types.Gaussians, so the forward() type annotation holds
        # for both. The import is guarded so stock runs without sg_wrapper
        # installed are unaffected.
        sg_gaussians_cls = None
        try:
            from sg_wrapper import SGGaussians as sg_gaussians_cls
        except ImportError:
            pass

        if sg_gaussians_cls is not None and isinstance(gaussians, sg_gaussians_cls):
            from sg_wrapper import apply_runtime_lobe_budget, compute_sg_colors

            # Inference-time lobe budget (rho_lobe). Returns a pruned copy
            # with compensated c0 when a budget is configured; a no-op
            # otherwise, so training and the unpruned evaluation are
            # unchanged.
            gaussians = apply_runtime_lobe_budget(gaussians)

            sg_colors = compute_sg_colors(
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.rgb_base, "b g c -> (b v) g c", v=v),
                repeat(gaussians.sg_directions, "b g l xyz -> (b v) g l xyz", v=v),
                repeat(gaussians.sg_sharpness, "b g l s -> (b v) g l s", v=v),
                repeat(gaussians.sg_rgb, "b g l c -> (b v) g l c", v=v),
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                clamp_colors=getattr(gaussians, "clamp_colors", False),
            )
            color, depth, alpha = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                None,
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
                repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
                scale_invariant=self.make_scale_invariant,
                enable_cov_grad=self.enable_cov_grad,
                enable_sh_grad=self.enable_sh_grad,
                gaussian_colors=sg_colors,
            )
        else:
            color, depth, alpha = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                repeat(gaussians.rotations, "b g i -> (b v) g i", v=v),
                repeat(gaussians.scales, "b g i -> (b v) g i", v=v),
                scale_invariant=self.make_scale_invariant,
                enable_cov_grad=self.enable_cov_grad,
                enable_sh_grad=self.enable_sh_grad
            )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        
        depth = rearrange(depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) 1 h w -> b v h w", b=b, v=v)

        if self.make_scale_invariant:
            scale = near / 1
            depth = depth * scale[:, :, None, None]

        return DecoderOutput(color, depth, alpha=alpha)
        


  