"""用少量实测深度标定冻结 3DGS 的新视角深度，比较传感器直接重投影。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bench_adaptive_raster import scene_paths
from probe_depth_repair import depth_targets
from probe_edit_transport import reduced
from readout import FrozenGaussians, ordered_alpha_samples


def calibration_view(model, native, depth):
    """缓存实测有效射线及同一射线上的 RGB 光栅权重。"""
    camera = reduced(native, 240)
    dummy = model.opacity[:, None]
    attr, opacity, meta = model.render(camera, dummy, with_depth=True)
    band = depth_targets(depth, (camera.width, camera.height), model.xyz.device)
    truth = band[:, :2].mean(dim=1)
    estimate = attr[..., -1].flatten()
    valid = band[:, 2].bool() & (opacity.flatten() > .7) & ((estimate - truth).abs() < .65)
    pixel, splat, alpha, cumulative, _ = ordered_alpha_samples(meta, camera.width, camera.height)
    keep = valid[pixel]
    alpha, cumulative, pixel, splat = alpha[keep], cumulative[keep], pixel[keep], splat[keep]
    before = cumulative + torch.log1p(-alpha)
    weight = alpha * torch.exp(-before).to(alpha.dtype) / opacity.flatten()[pixel].clamp_min(.1)
    normal = torch.as_tensor(camera.c2w[:3, 2].copy(), device=weight.device, dtype=weight.dtype)
    return estimate, truth, valid, pixel, splat, weight, normal


def solve(model, sessions):
    """拟合全局度量基线和逐高斯残差，全部参数只使用标定视角。

    Args:
        model (FrozenGaussians): RGB-only 拟合所得的原始地图。
        sessions (list[tuple]): 标定相机的有效像素、测深和光栅权重。

    Returns:
        tuple: 全局尺度、偏移、每高斯标量偏移 (N,) 与世界坐标偏移 (N, 3)。
    """
    x = torch.cat([v[0][v[2]] for v in sessions])
    y = torch.cat([v[1][v[2]] for v in sessions])
    x_mean, y_mean = x.mean(), y.mean()
    coefficient = (((x - x_mean) * (y - y_mean)).sum() /
                   (x - x_mean).square().sum().clamp_min(1e-5)).clamp(.75, 1.25)
    intercept = (y_mean - coefficient * x_mean).clamp(-.4, .4)
    mass, rhs, normals = [], [], []
    for estimate, truth, _, pixel, splat, weight, normal in sessions:
        residual = (truth - (coefficient * estimate + intercept)).clamp(-.4, .4)
        accum = weight.new_zeros((len(model.xyz), 2))
        accum.index_add_(0, splat, torch.stack((weight, weight * residual[pixel]), dim=-1))
        mass.append(accum[:, 0])
        rhs.append(accum[:, 1])
        normals.append(normal)
    mass, rhs, normals = torch.stack(mass), torch.stack(rhs), torch.stack(normals)
    support = mass.sum(0)
    # 低支持区域优先退回到场景级校准，避免单个低权重像素产生整块漂移。
    regularization = .20 + .10 * support
    scalar = rhs.sum(0) / (support + regularization)
    normal_outer = (normals[:, :, None] * normals[:, None, :]).reshape(-1, 9)
    matrix = (mass.T @ normal_outer).reshape(-1, 3, 3)
    matrix.diagonal(dim1=1, dim2=2).add_(regularization[:, None])
    vector = torch.linalg.solve(matrix, (rhs.T @ normals).unsqueeze(-1)).squeeze(-1)
    return coefficient, intercept, scalar, vector


def sensor_points(camera, depth, device):
    """将实测深度投影到世界坐标，忽略无深度和局部不连续区域。"""
    band = depth_targets(depth, (camera.width, camera.height), device)
    depth = band[:, :2].mean(-1)
    idx = torch.arange(len(depth), device=device)[band[:, 2].bool()]
    intrinsic = camera.intrinsic
    x = (idx % camera.width - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (idx // camera.width - intrinsic[1, 2]) / intrinsic[1, 1]
    local = torch.stack((x, y, torch.ones_like(x)), dim=1) * depth[idx, None]
    pose = torch.as_tensor(camera.c2w.copy(), device=device, dtype=local.dtype)
    return local @ pose[:3, :3].T + pose[:3, 3]


def sensor_reprojection(camera, world):
    """用深度缓冲重投影标定数据，无测量覆盖的结果为正无穷。"""
    pose = torch.as_tensor(np.linalg.inv(camera.c2w), device=world.device, dtype=world.dtype)
    local = world @ pose[:3, :3].T + pose[:3, 3]
    z = local[:, 2]
    fx, fy = camera.intrinsic.diagonal()[:2]
    cx, cy = camera.intrinsic[:2, 2]
    x = (local[:, 0] / z * fx + cx).round().long()
    y = (local[:, 1] / z * fy + cy).round().long()
    w, h = camera.width, camera.height
    inside = (z > .10) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    output = torch.full((w * h,), torch.inf, device=world.device)
    output.scatter_reduce_(0, (y * w + x)[inside], z[inside], reduce="amin", include_self=True)
    return output


def metrics(model, camera, depth, sources, solution):
    """全部方案用原尺寸留出深度，并按实测重投影覆盖分别计分。"""
    coefficient, intercept, scalar, vector = solution
    n = torch.as_tensor(camera.c2w[:3, 2].copy(), dtype=scalar.dtype, device=scalar.device)
    attrs = torch.stack((scalar, vector @ n), dim=-1)
    z, opacity, _ = model.render(camera, attrs, with_depth=True)
    plain = z[..., -1].flatten()
    opacity = opacity.flatten()
    correction = z[..., :2].reshape(-1, 2) / opacity[:, None].clamp_min(.1)
    global_depth = plain * coefficient + intercept
    scalar_depth = global_depth + correction[:, 0]
    local_depth = global_depth + correction[:, 1]
    reprojected = sensor_reprojection(camera, sources)
    measured = reprojected.isfinite()
    band = depth_targets(depth, (camera.width, camera.height), scalar.device)
    valid = band[:, 2].bool() & (opacity > .1)
    truth = band[:, :2].mean(-1)
    estimates = {"expected": plain, "global_affine": global_depth, "scalar_adapter": scalar_depth,
                 "world_adapter": local_depth, "direct_plus_affine": torch.where(measured, reprojected, global_depth),
                 "direct_plus_local": torch.where(measured, reprojected, local_depth)}
    groups = {"all": valid, "sensor_covered": valid & measured, "sensor_uncovered": valid & ~measured}
    rows = {}
    for name, candidate in estimates.items():
        error = (candidate - truth).abs()
        rows[name] = {group: {"count": int(mask.sum()), "sum_abs": float(error[mask].sum()),
                              "within_10cm": int((error[mask] < .10).sum())}
                      for group, mask in groups.items()}
    return rows


@torch.inference_mode()
def main():
    """利用已有数据和检查点完成标定与原尺寸留出评测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--scene", choices=("room0", "office0", "tum"), required=True)
    parser.add_argument("--phase", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    root, checkpoint, loader = scene_paths(args.data, args.checkpoints, args.scene)
    model = FrozenGaussians.load(checkpoint)
    cameras, depths, frame_ids = loader(root)
    indices = np.rint(np.linspace(0, len(cameras) - 1, min(10, len(cameras)))).astype(int)
    calibration, evaluation = indices[args.phase::2], indices[1 - args.phase::2]
    data = [calibration_view(model, cameras[i], depths[i]) for i in calibration]
    solution = solve(model, data)
    points = torch.cat([sensor_points(cameras[i], depths[i], model.xyz.device) for i in calibration])
    print(f"Five measurements calibrated; testing {len(evaluation)} separate native views", flush=True)
    result = {"scene": args.scene, "calibration": [int(frame_ids[i]) for i in calibration],
              "evaluation": [int(frame_ids[i]) for i in evaluation],
              "affine": [float(x) for x in solution[:2]], "views": [
                  {"frame": int(frame_ids[i]), **metrics(model, cameras[i], depths[i], points, solution)}
                  for i in evaluation]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print("Held-out metric depth calibration complete", flush=True)


if __name__ == "__main__":
    main()
