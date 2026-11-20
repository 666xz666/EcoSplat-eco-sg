"""SG 推理颜色融合入口；训练沿用可微 PyTorch 路径，扩展仅在显式调用时加载。"""

from functools import lru_cache
from pathlib import Path
import math

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _extension():
    """按源码内容缓存编译扩展。

    Returns:
        module: 含 forward 的 CUDA 扩展；首次调用可能需要数分钟编译。
    """
    from torch.utils.cpp_extension import load

    return load(
        name="ecosplat_sg_colors",
        sources=[str(Path(__file__).parent / "cuda" / "sg_colors.cu")],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def compute_sg_colors_cuda(
    means: Tensor,
    rgb_base: Tensor,
    sg_directions: Tensor,
    sg_sharpness: Tensor,
    sg_rgb: Tensor,
    extrinsics: Tensor,
    clamp_colors: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """融合每相机每高斯的 SG 颜色求值，不重复存储相机共享的 lobe 参数。

    Args:
        means (Tensor[B,G,3]): 世界坐标中心，CUDA float32。
        rgb_base (Tensor[B,G,3]): 已完成可选剪枝补偿的基础颜色。
        sg_directions (Tensor[B,G,L,3]): 未归一化方向。
        sg_sharpness (Tensor[B,G,L,1]): 已激活锐度；此处不再次 abs。
        sg_rgb (Tensor[B,G,L,3]): 已缩放振幅；零振幅 lobe 跳过求值。
        extrinsics (Tensor[B,V,4,4]): camera-to-world 相机矩阵。
        clamp_colors (bool): 是否截断到 [0,1]。
        eps (float): 与参考实现相同的归一化稳定常数。

    Returns:
        Tensor[B*V,G,3]: 按 batch、view 顺序排列的颜色，无反向实现。
    """
    tensors = (means, rgb_base, sg_directions, sg_sharpness, sg_rgb, extrinsics)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("SG CUDA is inference-only; use compute_sg_colors for gradients")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if any(t.device != means.device or not t.is_cuda or t.dtype != torch.float32 for t in tensors):
        raise ValueError("SG CUDA requires float32 tensors on the same CUDA device")
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError("means must be [B,G,3]")
    b, g, _ = means.shape
    if sg_directions.ndim != 4 or sg_directions.shape[:2] != (b, g) or sg_directions.shape[-1] != 3:
        raise ValueError("directions must be [B,G,L,3]")
    if rgb_base.shape != means.shape or sg_rgb.shape != sg_directions.shape:
        raise ValueError("base/amplitude shape mismatch")
    if sg_sharpness.shape != (*sg_directions.shape[:-1], 1):
        raise ValueError("sharpness must be [B,G,L,1]")
    if extrinsics.ndim != 4 or extrinsics.shape[0] != b or extrinsics.shape[-2:] != (4, 4):
        raise ValueError("extrinsics must be [B,V,4,4]")
    # 步骤1：推理可读取 requires_grad 参数，但不得把无反向输出暴露给训练。
    inputs = [t.detach().contiguous() for t in tensors]
    # 步骤2：扩展使用当前 CUDA stream，保持与后续 gsplat 的依赖顺序。
    return _extension().forward(*inputs, clamp_colors, eps)
