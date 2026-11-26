"""在冻结 3DGS 上选择少量深度标定相机，验证其对指定新视角的用途。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bench_adaptive_raster import scene_paths
from probe_depth_adapter import calibration_view, local_sensor_fill, sensor_points, sensor_reprojection, solve
from probe_depth_repair import depth_targets
from probe_edit_transport import reduced
from readout import FrozenGaussians, ordered_alpha_samples


def view_support(model, native):
    """用光栅化的射线深度分散度估计源测量可归因性和目标的测量需求。"""
    camera = reduced(native, 108)
    _, opacity, meta = model.render(camera, model.opacity[:, None])
    pixels, splats, alpha, cumulative, _ = ordered_alpha_samples(meta, camera.width, camera.height)
    weight = alpha * torch.exp(-(cumulative + torch.log1p(-alpha))).to(alpha.dtype)
    z = meta["depths"][0, splats]
    moments = weight.new_zeros((camera.width * camera.height, 2))
    moments.index_add_(0, pixels, torch.stack((weight * z, weight * z.square()), dim=-1))
    moments /= opacity.reshape(-1, 1).clamp_min(.1)
    spread = (moments[:, 1] - moments[:, 0].square()).clamp(0, .36).sqrt()
    spread = spread[pixels]
    normalized = weight / (camera.width * camera.height)
    evidence = torch.stack((normalized, normalized * torch.exp(-spread / .14),
                            normalized * (.03 + spread)), dim=-1)
    totals = weight.new_zeros((len(model.xyz), 3))
    totals.index_add_(0, splats, evidence)
    return totals


def facility(scores, weight, total):
    """逐个选择让目标覆盖增量最大的姿态，已覆盖位置的收益递减。"""
    best = torch.zeros_like(weight)
    indices = []
    for _ in range(total):
        value = (torch.maximum(scores, best) * weight).sum(dim=1)
        value[indices] = -torch.inf
        choice = int(value.argmax())
        best = torch.maximum(best, scores[choice])
        indices.append(choice)
    return tuple(sorted(indices))


def pose_scores(pool, target, device):
    """按平移与朝向相似度做目标覆盖的独立基线。"""
    a = np.stack([c.c2w for c in pool])
    b = np.stack([c.c2w for c in target])
    displacement = np.linalg.norm(a[:, None, :3, 3] - b[None, :, :3, 3], axis=-1)
    agreement = np.clip(a[:, :3, 2] @ b[:, :3, 2].T, -1, 1)
    return torch.as_tensor(np.exp(-displacement / .45 - (1 - agreement) / .25),
                           device=device, dtype=torch.float32)


def evaluate(model, camera, depth, points, solution):
    """与传感器重投影结合后，记录独立相机的原尺寸深度误差。"""
    coefficient, intercept, scalar, _ = solution
    render, opacity, _ = model.render(camera, scalar[:, None], with_depth=True)
    weight = opacity.flatten()
    plain = render[..., 1].flatten()
    residual = render[..., 0].flatten() / weight.clamp_min(.1)
    prior = coefficient * plain + intercept
    direct = sensor_reprojection(camera, points)
    filled, distance = local_sensor_fill(camera, direct, prior)
    band = depth_targets(depth, (camera.width, camera.height), model.xyz.device)
    valid = band[:, 2].bool() & (weight > .1)
    truth = band[:, :2].mean(-1)
    values = {"original": plain, "filled_plus_affine": torch.where(filled.isfinite(), filled, prior),
              "filled_plus_adapter": torch.where(filled.isfinite(), filled, prior + residual)}
    return {"valid": int(valid.sum()), "filled": int((valid & filled.isfinite()).sum()),
            "far": int((valid & (distance > 8)).sum()),
            "methods": {name: {"sum_abs": float((value - truth).abs()[valid].sum()),
                               "within_10cm": int(((value - truth).abs()[valid] < .1).sum())}
                        for name, value in values.items()}}


@torch.inference_mode()
def main():
    """固定相机预算，源相机选择只用地图和位姿，测量在选定后才参与计算。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--checkpoints", required=True, type=Path)
    p.add_argument("--scene", choices=("room0", "office0", "tum"), required=True)
    p.add_argument("--phase", choices=(0, 1), type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    root, checkpoint, loader = scene_paths(args.data, args.checkpoints, args.scene)
    model = FrozenGaussians.load(checkpoint)
    camera, depths, frames = loader(root)
    pool = list(range(args.phase, len(camera), 2))
    query = list(range(1 - args.phase, len(camera), 2))
    query = [query[i] for i in np.rint(np.linspace(0, len(query) - 1, 5)).astype(int)]
    candidates = torch.stack([view_support(model, camera[i]) for i in pool])
    targets = torch.stack([view_support(model, camera[i]) for i in query]).mean(0)
    pixels = reduced(camera[pool[0]], 108)
    scale = .65 * pixels.width * pixels.height
    poses = pose_scores([camera[i] for i in pool], [camera[i] for i in query], model.xyz.device)
    total = 3
    picks = {
        "uniform": tuple(np.rint(np.linspace(0, len(pool) - 1, total)).astype(int).tolist()),
        "pose_cover": facility(poses, torch.ones_like(poses[0]), total),
        "gauss_visibility_cover": facility(-torch.expm1(-scale * candidates[:, :, 0]), targets[:, 0], total),
        "attribution_cover": facility(-torch.expm1(-scale * candidates[:, :, 1]), targets[:, 2], total),
    }
    print("Camera selection complete without sensor evidence", flush=True)
    cached = {}
    result = {"scene": args.scene, "pool_frames": [int(frames[i]) for i in pool],
              "query_frames": [int(frames[i]) for i in query], "methods": {}}
    for method, subset in picks.items():
        if subset not in cached:
            indices = [pool[i] for i in subset]
            sessions = [calibration_view(model, camera[i], depths[i]) for i in indices]
            solution = solve(model, sessions)
            points = torch.cat([sensor_points(camera[i], depths[i], model.xyz.device) for i in indices])
            cached[subset] = {"frames": [int(frames[i]) for i in indices],
                              "metrics": [evaluate(model, camera[i], depths[i], points, solution) for i in query]}
        result["methods"][method] = cached[subset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print("Matched sensor budget evaluated on separate cameras", flush=True)


if __name__ == "__main__":
    main()
