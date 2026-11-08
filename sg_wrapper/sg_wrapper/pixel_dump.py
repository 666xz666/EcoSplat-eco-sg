"""把一次渲染所需的基元与结果落盘，供离线统计「每像素覆盖多少点/多少 lobe」。

只在设置了 ``SG_PIXEL_DUMP_DIR`` 时生效：不设变量时 ``maybe_dump`` 立即返回，
训练与评测热路径没有额外开销，也不改任何数值。
"""

from __future__ import annotations

import os
import re

import torch

_ALLOWED_NAME = re.compile(r"^[A-Za-z0-9_]+$")

_STATE = {"scene": None, "saved": {}}


def set_scene(scene: str | None) -> None:
    """记录当前 batch 的场景名，供渲染侧拼文件名。

    Args:
        scene: 场景 hash；None 表示清空。

    Returns:
        None
    """
    _STATE["scene"] = scene


def _enabled_for(scene: str) -> str | None:
    """判断该场景是否需要落盘。

    Args:
        scene: 场景 hash。

    Returns:
        落盘目录；不需要落盘时返回 None。
    """
    # 步骤1：目录未设置就整体关闭，保证默认路径零开销。
    out_dir = os.environ.get("SG_PIXEL_DUMP_DIR")
    if not out_dir:
        return None
    # 步骤2：可选白名单，方便只抓几个样例场景。
    allow = os.environ.get("SG_PIXEL_DUMP_SCENES", "").strip()
    if allow and scene not in {s.strip() for s in allow.split(",") if s.strip()}:
        return None
    # 步骤3：每个场景只抓前 N 次渲染（默认 1），避免 3 个目标视角 × 200 场景爆盘。
    cap = int(os.environ.get("SG_PIXEL_DUMP_MAX", "1"))
    if _STATE["saved"].get(scene, 0) >= cap:
        return None
    if not _ALLOWED_NAME.match(scene):
        return None
    return out_dir


def dump_target(scene: str, target_image: torch.Tensor, indices=None) -> None:
    """保存该场景的目标图像，供离线算逐像素误差。

    Args:
        scene: 场景 hash。
        target_image: (V, 3, H, W) 目标视角图像，取值 [0, 1]。
        indices: 目标视角帧号，可选，仅作记录。

    Returns:
        None
    """
    out_dir = _enabled_for(scene)
    if out_dir is None:
        return
    payload = {
        "scene": scene,
        "target": target_image.detach().float().cpu(),
    }
    if indices is not None:
        payload["target_index"] = indices
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene}_target.pt")
    torch.save(payload, path)


def maybe_dump(
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    color: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
) -> None:
    """按需保存单个视角的基元与渲染结果。

    Args:
        gaussians: 解码器实际用来渲染的 Gaussians（可能是剪枝后的对象）。
        extrinsics: (B, V, 4, 4) 世界到相机矩阵。
        intrinsics: (B, V, 3, 3) 归一化内参。
        image_shape: (H, W)。
        color: (B, V, 3, H, W) 渲染图。
        alpha: (B, V, 1, H, W) 累积不透明度。
        depth: (B, V, H, W) 渲染深度。

    Returns:
        None
    """
    scene = _STATE["scene"]
    if scene is None:
        return
    out_dir = _enabled_for(scene)
    if out_dir is None:
        return

    idx = _STATE["saved"].get(scene, 0)
    _STATE["saved"][scene] = idx + 1

    # 只落第一张图：离线分析按目标视角逐个看，多存没有信息增量。
    payload = {
        "scene": scene,
        "means": gaussians.means[0].detach().float().cpu(),
        "covariances": gaussians.covariances[0].detach().float().cpu(),
        "opacities": gaussians.opacities[0].detach().float().cpu(),
        "scales": gaussians.scales[0].detach().float().cpu(),
        "rotations": gaussians.rotations[0].detach().float().cpu(),
        "extrinsics": extrinsics[0, 0].detach().float().cpu(),
        "intrinsics": intrinsics[0, 0].detach().float().cpu(),
        "image_shape": tuple(int(x) for x in image_shape),
        "color": color[0, 0].detach().float().cpu(),
        "alpha": alpha[0, 0].detach().float().cpu(),
        "depth": depth[0, 0].detach().float().cpu(),
    }
    # SG 头才有 lobe 参数；SH 头缺这两个键，离线脚本按「每点一颗」处理。
    sg_rgb = getattr(gaussians, "sg_rgb", None)
    if sg_rgb is not None:
        payload["sg_rgb"] = sg_rgb[0].detach().float().cpu()
    rgb_base = getattr(gaussians, "rgb_base", None)
    if rgb_base is not None:
        payload["rgb_base"] = rgb_base[0].detach().float().cpu()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene}_t{idx}.pt")
    torch.save(payload, path)
    print(f"[pixel_dump] wrote {path} ({payload['means'].shape[0]} primitives)")
