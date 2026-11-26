"""用冻结 3DGS 的双视角射线分位深度检验候选可见性。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate import (
    boundary_pixels,
    evaluation_pairs,
    load_replica,
    project_between_cameras,
    scores_for_pair,
    valid_pixels,
)
from readout import FrozenGaussians, opacity_quantiles


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
PIXEL_TOLERANCE = 1.0
RELATIVE_DEPTH_TOLERANCE = 0.015


def ray_depths(model, camera):
    """在原相机分辨率计算 alpha 与射线终止深度分位数。"""
    _, alpha, metadata = model.render(camera, model.colors(camera), with_depth=False)
    depths = opacity_quantiles(metadata, camera.width, camera.height, QUANTILES)
    return depths, alpha[0, :, :, 0].cpu().numpy()


def target_residual(depth, source, target, target_depth, target_alpha):
    """计算源候选在目标射线的深度偏差和反向像素偏差。

    Args:
        depth (ndarray, (H, W)): 源候选的 Z 深度。
        source (Camera): 源相机。
        target (Camera): 目标相机。
        target_depth (ndarray, (H, W)): 目标透明度中位深度。
        target_alpha (ndarray, (H, W)): 目标累计 alpha。

    Returns:
        tuple: 相对深度偏差及以像素计的反投影偏差 (H, W)。无有效目标观测处为无穷大。
    """
    uv, predicted_z = project_between_cameras(depth, source, target)
    u, v = uv.transpose(2, 0, 1)
    valid = np.isfinite(uv).all(axis=-1) & (predicted_z > 0)
    valid &= (u >= 0) & (u <= target.width - 1) & (v >= 0) & (v <= target.height - 1)
    x = np.rint(np.where(valid, u, 0)).astype(np.intp)
    y = np.rint(np.where(valid, v, 0)).astype(np.intp)
    measured = target_depth[y, x]
    valid &= np.isfinite(measured) & (measured > 0.15) & (target_alpha[y, x] > 0.25)
    valid &= np.isfinite(depth) & (depth > 0.15)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_error = np.abs(np.log(measured / predicted_z))
        # 沿原投影射线取目标观测，可避免整数采样导致的整像素循环误差。
        inverse = np.linalg.inv(source.c2w) @ target.c2w
        direction = (uv - target.intrinsic[:2, 2]) / target.intrinsic.diagonal()[:2]
        xyz = np.concatenate((direction, np.ones_like(u)[..., None]), axis=-1) * measured[..., None]
        xyz = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        uv_back = xyz[..., :2] / xyz[..., 2:] * source.intrinsic.diagonal()[:2] + source.intrinsic[:2, 2]
        yy, xx = np.indices(depth.shape)
        cycle_error = np.hypot(uv_back[..., 0] - xx, uv_back[..., 1] - yy)
    valid &= np.isfinite(cycle_error) & np.isfinite(depth_error) & (xyz[..., 2] > 0)
    return np.where(valid, depth_error, np.inf), np.where(valid, cycle_error, np.inf)


def predictions(source_ray, target_ray, source, target):
    """比较中位深度、单候选循环一致性和多候选射线迁移。"""
    depths, alpha = source_ray
    other_depths, other_alpha = target_ray
    rel, cycle = zip(*(target_residual(d, source, target, other_depths[2], other_alpha) for d in depths))
    rel, cycle = np.stack(rel), np.stack(cycle)
    valid = np.isfinite(depths[2]) & (depths[2] > 0.15) & (alpha > 0.25)
    p25, _ = project_between_cameras(depths[1], source, target)
    p75, _ = project_between_cameras(depths[3], source, target)
    width_px = np.linalg.norm(p25 - p75, axis=-1)
    depth_width = np.log(depths[-1] / depths[0])
    confidence = {
        "median": np.log(alpha.clip(1e-6)) - depth_width,
        "median_depth_agreement": -rel[2],
        "median_cycle": -cycle[2],
        "median_envelope": -np.hypot(cycle[2], width_px),
    }
    # 候选在像素与相对深度容差内均应一致。分位先验打破等分时的平局。
    error = np.hypot(rel / RELATIVE_DEPTH_TOLERANCE, cycle / PIXEL_TOLERANCE)
    prior = np.abs(np.asarray(QUANTILES) - 0.5)[:, None, None]
    cost = error + prior
    choice = cost.argmin(axis=0)[None]
    chosen = np.take_along_axis(depths, choice, axis=0)[0]
    min_cost = np.take_along_axis(cost, choice, axis=0)[0]
    estimates = {name: (depths[2], np.where(valid, score, -np.inf))
                 for name, score in confidence.items()}
    estimates["multi_quantile"] = (chosen, np.where(valid, -min_cost, -np.inf))
    return estimates, depths


def oracle_bad(depths, source, target, projected, valid, truth):
    """计算只有接触真值才能达到的候选集合误差下界。"""
    projected_candidates = [project_between_cameras(depth, source, target) for depth in depths]
    warped, z = (np.stack(values) for values in zip(*projected_candidates))
    inside = (z > 0) & (warped[..., 0] >= 0) & (warped[..., 0] < target.width)
    inside &= (warped[..., 1] >= 0) & (warped[..., 1] < target.height)
    errors = np.where(inside, np.linalg.norm(warped - projected[None], axis=-1), np.inf)
    error = np.min(errors, axis=0)
    edges = boundary_pixels(truth) & valid
    return {
        "all": float(np.mean(error[valid] > 1)),
        "boundary": float(np.mean(error[edges] > 1)) if edges.any() else None,
        "median_on_all": float(np.mean(~(errors[2][valid] <= 1))),
        "median_on_boundary": float(np.mean(~(errors[2][edges] <= 1))) if edges.any() else None,
    }


@torch.inference_mode()
def main():
    """评测同一场景的已有留出相机对，真值仅进入打分环节。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    model = FrozenGaussians.load(args.checkpoint)
    cameras, truth, frames = load_replica(args.replica)
    pairs = list(evaluation_pairs(cameras))
    assert pairs
    rays = {i: ray_depths(model, cameras[i]) for i in sorted({i for a, b, _ in pairs for i in (a, b)})}
    results = []
    for i, j, baseline in pairs:
        projected, z = project_between_cameras(truth[i], cameras[i], cameras[j])
        visible = valid_pixels(truth[i], truth[j], projected, z)
        methods, candidates = predictions(rays[i], rays[j], cameras[i], cameras[j])
        scores = {name: scores_for_pair(prediction, cameras[i], cameras[j], truth[i], projected, visible)
                  for name, prediction in methods.items()}
        oracle = oracle_bad(candidates, cameras[i], cameras[j], projected, visible, truth[i])
        results.append({"source": frames[i], "target": frames[j], "baseline": baseline,
                        "visible": int(visible.sum()), "scores": scores, "oracle_full_coverage": oracle})
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump({"scene": args.replica.name, "quantiles": QUANTILES, "pairs": results}, sink, allow_nan=False)
    print(f"{args.replica.name}: {len(results)} pairs complete", flush=True)


if __name__ == "__main__":
    main()
