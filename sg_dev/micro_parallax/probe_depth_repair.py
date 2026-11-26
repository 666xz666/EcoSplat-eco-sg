"""用少量 RGB-D 观测估计冻结 3DGS 的透明度修正，并用独立 RGB-D 验证。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bench_adaptive_raster import scene_paths
from probe_edit_transport import reduced
from probe_signed_repair import removal_delta, target_image
from readout import FrozenGaussians, ordered_alpha_samples


def depth_targets(depth, size, device):
    """构建实测表面带；剔除无深度值和缩放后仍不连续的边界。"""
    small = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
    radius = .04 + .015 * small
    lo = cv2.erode(small, np.ones((3, 3), np.uint8))
    hi = cv2.dilate(small, np.ones((3, 3), np.uint8))
    valid = (lo > 0) & (hi - lo < 2 * radius)
    return torch.as_tensor(np.stack((small - radius, small + radius, valid), axis=-1),
                           dtype=torch.float32, device=device).reshape(-1, 3)


def ray_terms(camera, metadata, band):
    """返回排序后的滤波贡献与实测深度的空间关系。

    Args:
        camera (Camera): 与 metadata 同分辨率和位姿的相机。
        metadata (dict): 单相机的 gsplat 光栅化数据。
        band (Tensor, (H * W, 3)): 每像素的近侧阈值、远侧阈值与有效掩码。

    Returns:
        tuple: 像素索引、高斯索引、alpha、合成权重，形状均为 (M,)，
            阈值分类形状为 (M, 2)，两列依次为空间中提前命中和在表面带内命中。
    """
    pixel, splat, alpha, cumulative, _ = ordered_alpha_samples(metadata, camera.width, camera.height)
    weight = alpha * torch.exp(-(cumulative + torch.log1p(-alpha))).to(alpha.dtype)
    low, high, valid = band[pixel].unbind(dim=-1)
    z = metadata["depths"][0, splat]
    labels = torch.stack((z < low, (z >= low) & (z <= high)), dim=-1)
    return pixel, splat, alpha, weight, labels * valid[:, None]


def center_carving(camera, metadata, band):
    """用二维投影中心采样作为透明度干预的几何基线。"""
    xy = metadata["means2d"][0].floor().long()
    x, y = xy.unbind(dim=-1)
    height, width = camera.height, camera.width
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    p = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    low, high, valid = band[p].unbind(dim=-1)
    available = inside & valid.bool()
    z = metadata["depths"][0]
    free = (z < low) & available
    surface = (z >= low) & (z <= high) & available
    return (free.float() - surface.float()) * metadata["opacities"][0]


def evidence(model, camera, rgb, depth, colors):
    """同时估计 RGB 和实测深度目标上的单高斯有符号收益。"""
    render, _, meta = model.render(camera, colors)
    band = depth_targets(depth, (camera.width, camera.height), model.xyz.device)
    pixel, splat, alpha, weight, labels = ray_terms(camera, meta, band)
    x = torch.cat((colors[splat], labels), dim=-1) * weight[:, None]
    delta = removal_delta(alpha, pixel, x)
    drgb, dfree, dsurface = delta[:, :3], delta[:, 3], delta[:, 4]
    actual = target_image(rgb, camera, model.xyz.device).reshape(-1, 3)
    residual = render.reshape(-1, 3) - actual
    dot = (drgb * residual[pixel]).sum(-1)
    rgb_gain = 2 * dot - drgb.square().sum(-1)
    contributions = torch.stack((rgb_gain, weight * labels[:, 0], dfree, dfree - dsurface), dim=-1)
    result = weight.new_zeros((len(model.xyz), contributions.shape[-1]))
    result.index_add_(0, splat, contributions)
    return torch.cat((result, center_carving(camera, meta, band)[:, None]), dim=-1)


def evaluated_views(source, loader, count):
    """按共同帧号读取训练时未使用的照片、测深和相机。"""
    cameras, depths, frames = loader(source)
    indices = np.rint(np.linspace(0, len(cameras) - 1, min(count, len(cameras)))).astype(int)
    views = []
    for i in indices:
        path = source / "images" / f"frame{frames[i]:06d}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        views.append((cameras[i], int(frames[i]), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depths[i]))
    return views


def evaluate(model, views, masks):
    """重渲染留出视角，记录原尺寸 RGB 误差及固定深度掩码上的几何指标。"""
    baseline = model.opacity
    measurements = {name: [] for name in masks}
    for native, frame, rgb, depth in views:
        camera = reduced(native, 144)
        band = depth_targets(depth, (camera.width, camera.height), model.xyz.device)
        colors = model.colors(camera)
        target = target_image(rgb, native, model.xyz.device)
        for name, scale in masks.items():
            model.opacity = baseline * scale
            raster, _, meta = model.render(camera, colors, with_depth=True)
            native_rgb = model.render(native, colors)[0][0].clamp(0, 1)
            pixel, _, _, weight, labels = ray_terms(camera, meta, band)
            prob = labels.new_zeros((len(band), 2))
            prob.index_add_(0, pixel, weight[:, None] * labels)
            use = band[:, 2].bool()
            depth_abs = (raster[..., 3].flatten() - band[:, :2].mean(-1)).abs()
            measurements[name].append({
                "frame": frame, "native_mse": float((native_rgb - target).square().mean()),
                "free_mass": float(prob[use, 0].mean()), "hit_mass": float(prob[use, 1].mean()),
                "depth_mae_m": float(depth_abs[use].mean()), "depth_pixels": int(use.sum()),
            })
    model.opacity = baseline
    return measurements


@torch.inference_mode()
def main():
    """比较匹配的硬移除预算，以独立实测深度和原尺寸 RGB 验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--scene", choices=("room0", "office0", "tum"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    root, checkpoint, loader = scene_paths(args.data, args.checkpoints, args.scene)
    model = FrozenGaussians.load(checkpoint)
    views = evaluated_views(root, loader, 10)
    calibration, evaluation = views[::2], views[1::2]
    evidence_frames = torch.stack([evidence(model, reduced(c, 144), rgb, depth, model.colors(c))
                                   for c, _, rgb, depth in calibration])
    score = evidence_frames.sum(dim=0).T
    values = {"center_carving": score[4], "free_blending_weight": score[1],
              "free_delta": score[2], "surface_delta": score[3], "rgb_delta": score[0],
              "rgb_surface_delta": score[3] + 8 * score[0]}
    masks = {"original": torch.ones_like(model.opacity)}
    selected = {}
    budget = int(.01 * len(model.xyz))
    budget = min(budget, *(int((value > 0).sum()) for value in values.values()))
    if budget == 0:
        raise ValueError("Common intervention budget is empty")
    for name, value in values.items():
        indices = torch.topk(value, budget).indices
        factor = torch.ones_like(model.opacity)
        factor[indices] = 0
        masks[name] = factor
        selected[name] = len(indices)
    print(f"{len(calibration)} RGB-D views scored; {budget} matched removals each", flush=True)
    result = {"scene": args.scene, "splats": len(model.xyz), "removed": selected,
              "calibration_frames": [v[1] for v in calibration], "evaluation_frames": [v[1] for v in evaluation],
              "measurements": evaluate(model, evaluation, masks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print("Held-out native RGB and depth evaluation complete", flush=True)


if __name__ == "__main__":
    main()
