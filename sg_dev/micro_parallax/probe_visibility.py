"""检验两条 3DGS 射线的终止深度分布能否识别跨视角遮挡。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate import boundary_pixels, evaluation_pairs, load_replica, project_between_cameras
from probe_two_rays import ray_depths, target_residual
from readout import FrozenGaussians


QUANTILE_WEIGHTS = np.array([.175, .2, .25, .2, .175], dtype=np.float64)


def projected_samples(projected, target):
    """将分辨率内的浮点像素投影转成近邻索引及有效掩码。"""
    u, v = np.moveaxis(projected, -1, 0)
    inside = (u >= 1) & (u < target.width - 1) & (v >= 1) & (v < target.height - 1)
    x = np.rint(np.where(inside, u, 0)).astype(np.intp)
    y = np.rint(np.where(inside, v, 0)).astype(np.intp)
    return y, x, inside


def evidence(source, target, source_ray, target_ray):
    """生成同一中位深度对应的可见性和可靠性分数。

    Args:
        source (Camera): 源相机。
        target (Camera): 目标相机。
        source_ray (tuple): 源分位深度 (5, H, W) 与 alpha (H, W)。
        target_ray (tuple): 目标分位深度和 alpha。

    Returns:
        tuple: 中位深度对应坐标 (H, W, 2) 与各置信度图 (H, W)。
    """
    source_depth, source_alpha = source_ray
    target_depth, target_alpha = target_ray
    center, z_center = project_between_cameras(source_depth[2], source, target)
    _, cycle = target_residual(source_depth[2], source, target, target_depth[2], target_alpha)
    y, x, inside = projected_samples(center, target)
    signed = (target_depth[2, y, x] - z_center) / np.maximum(z_center, .15)
    signed = np.where(inside & (target_alpha[y, x] > .25) & (z_center > .15), signed, -np.inf)
    weights = QUANTILE_WEIGHTS[:, None, None]
    front = np.zeros_like(z_center)
    hit = np.zeros_like(z_center)
    target_only = None
    for index, candidate in enumerate(source_depth):
        uv, z = project_between_cameras(candidate, source, target)
        y, x, inside = projected_samples(uv, target)
        target_z = target_depth[:, y, x]
        tol = np.maximum(.025, .025 * z)
        usable = inside & (z > .15) & (target_alpha[y, x] > .25)
        delta = (target_z - z) / tol
        # 前向支持允许目标表面更深。对应支持还需要目标射线经过同一个深度。
        forward = np.sum(weights * np.clip(.5 + .5 * delta, 0, 1), axis=0)
        same_surface = np.sum(weights * np.clip(1 - np.abs(delta), 0, 1), axis=0)
        forward = np.where(usable & np.isfinite(forward), forward, 0)
        same_surface = np.where(usable & np.isfinite(same_surface), same_surface, 0)
        displacement = np.linalg.norm(uv - center, axis=-1)
        match = np.exp(-np.nan_to_num(displacement, nan=100) ** 2 / 2)
        front += QUANTILE_WEIGHTS[index] * forward
        hit += QUANTILE_WEIGHTS[index] * same_surface * match
        if index == 2:
            target_only = forward
    good = np.isfinite(source_depth[2]) & (source_alpha > .25)
    spread = np.log(source_depth[-1] / source_depth[0])
    scores = {
        "median_signed": signed,
        "source_alpha_width": source_alpha * np.exp(-spread),
        "median_cycle": -cycle,
        "target_distribution": target_only,
        "two_ray_visibility": front,
        "two_ray_joint": hit,
    }
    return center, {key: np.where(good & np.isfinite(value), value, -np.inf) for key, value in scores.items()}


def rank_metrics(labels, scores):
    """逐分数阈值计算 AP、ROC AUC 以及 90% 召回率的最低假阳性率。"""
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives < 25 or negatives < 25:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_score = scores[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(sorted_score) - 1]
    tp = np.cumsum(labels[order])[ends]
    fp = ends + 1 - tp
    recall, fpr = tp / positives, fp / negatives
    precision = tp / (tp + fp)
    return {
        "ap": float(np.sum(np.diff(recall, prepend=0) * precision)),
        "roc_auc": float(np.sum(np.diff(fpr, prepend=0) * (recall + np.r_[0, recall[:-1]]) / 2)),
        "fpr_at_90_recall": float(fpr[np.searchsorted(recall, .9)]),
        "positives": int(positives), "negatives": int(negatives),
    }


def compare(source, target, source_truth, target_truth, source_ray, target_ray):
    """同时评估可见性和满足像素误差要求的可用对应。"""
    correspondence, scores = evidence(source, target, source_ray, target_ray)
    uv, z = project_between_cameras(source_truth, source, target)
    y, x, inside = projected_samples(uv, target)
    observed = target_truth[y, x]
    tolerance = np.maximum(.025, .025 * z)
    evaluation = inside & (z > .15) & (source_truth > .15) & (source_truth < 9.9)
    evaluation &= (observed > .15) & (observed < 9.9) & (z + tolerance >= observed)
    visible = z <= observed + tolerance
    error = np.linalg.norm(correspondence - uv, axis=-1)
    correct = visible & (error <= 1)
    edges = boundary_pixels(source_truth)
    return {
        f"{task}_{part}": {name: rank_metrics(labels[mask], score[mask]) for name, score in scores.items()}
        for task, labels in (("visibility", visible), ("usable", correct))
        for part, mask in (("all", evaluation), ("boundary", evaluation & edges))
    }


@torch.inference_mode()
def main():
    """在冻结模型上运行预留相机对的遮挡及可用对应检验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    model = FrozenGaussians.load(args.checkpoint)
    cameras, truth, frames = load_replica(args.replica)
    pairs = list(evaluation_pairs(cameras))
    rays = {i: ray_depths(model, cameras[i]) for i in sorted({i for a, b, _ in pairs for i in (a, b)})}
    result = [{"source": frames[i], "target": frames[j], "baseline": baseline,
               "scores": compare(cameras[i], cameras[j], truth[i], truth[j], rays[i], rays[j])}
              for i, j, baseline in pairs]
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump({"scene": args.replica.name, "pairs": result}, sink, allow_nan=False)
    print(f"{args.replica.name}: {len(result)} pairs complete", flush=True)


if __name__ == "__main__":
    main()
