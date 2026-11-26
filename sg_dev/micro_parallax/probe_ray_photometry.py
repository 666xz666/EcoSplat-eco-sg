"""用固定的虚拟图组在 3DGS 射线候选之间做光度选择。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from evaluate import evaluation_pairs, load_replica, project_between_cameras, scores_for_pair, valid_pixels
from probe_two_rays import QUANTILES
from readout import FrozenGaussians, box, opacity_quantiles


def photometric_cost(center, probes, candidate):
    """计算单个射线深度图的虚拟视图 RGB 一致性。

    Args:
        center (ndarray, (H, W)): 中心灰度图。
        probes (list): 位移焦距乘积，虚拟灰度图与虚拟期望深度。
        candidate (ndarray, (H, W)): 每条射线的候选深度。

    Returns:
        tuple: 直接光度残差和使用虚拟图深度抑制遮挡后的残差 (H, W)。
    """
    y, x = np.indices(center.shape, dtype=np.float32)
    raw = np.zeros_like(center)
    weighted = np.zeros_like(center)
    count = np.zeros_like(center)
    inverse = 1 / np.where(np.isfinite(candidate) & (candidate > .15), candidate, 1).astype(np.float32)
    for shift_x, shift_y, image, depth in probes:
        ux, uy = x - inverse * np.float32(shift_x), y - inverse * np.float32(shift_y)
        warped = cv2.remap(image, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sensed = cv2.remap(depth, ux, uy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        inside = (ux >= 0) & (ux <= center.shape[1] - 1) & (uy >= 0) & (uy <= center.shape[0] - 1)
        # 只压低落在遮挡物背后的方向，避免让前方深度估计直接决定候选。
        visibility = np.clip(1 - (candidate - sensed) / np.maximum(.03 * candidate, .015), 0, 1)
        weight = np.where(inside & np.isfinite(visibility) & (sensed > 0.15), visibility, 0).astype(np.float32)
        error = np.abs(warped - center)
        raw += np.where(inside, error, np.float32(.05))
        weighted += error * weight
        count += weight
    count = box(count, 3)
    raw = box(raw / len(probes), 3)
    adjusted = box(weighted, 3) / np.maximum(1, count) + .01 * (1 - count / len(probes))
    return raw, np.where(count >= len(probes) / 4, adjusted, np.float32(np.inf))


def select_depth(depths, costs, alpha):
    """返回最低代价深度及中心 RGB 和候选投影一致性的置信度。"""
    costs = np.where(np.isfinite(depths), costs, np.float32(np.inf))
    best = np.argmin(costs, axis=0)[None]
    depth = np.take_along_axis(depths, best, axis=0)[0]
    loss = np.take_along_axis(costs, best, axis=0)[0]
    high = np.sort(costs, axis=0)[max(1, len(costs) - 2)]
    valid = (alpha > .25) & np.isfinite(loss) & np.isfinite(high) & (depth > .15) & (depth < 12)
    high = np.where(valid, high, 0)
    loss = np.where(valid, loss, 0)
    # 成本间距衡量该射线是否具备可区分的光度证据。
    confidence = alpha * (high - loss) / (high + .002)
    return depth, np.where(valid, confidence, -np.inf)


def read_view(model, camera):
    """生成与上一轮虚拟平面扫描相同的平移图组，并比较同候选数控制。"""
    fixed = model.colors(camera)
    images, alpha, metadata = model.render(camera, fixed, with_depth=True)
    center = cv2.cvtColor(images[0, :, :, :3].cpu().numpy(), cv2.COLOR_RGB2GRAY)
    expected = images[0, :, :, 3].cpu().numpy()
    alpha = alpha[0, :, :, 0].cpu().numpy()
    focal = camera.intrinsic.diagonal()[:2]
    reference = expected[(alpha > .5) & (expected > .15) & (expected < 12)]
    assert reference.size
    step = .75 * np.median(reference) / max(focal)
    depths = opacity_quantiles(metadata, camera.width, camera.height, QUANTILES)
    probes = []
    for axis in range(2):
        for sign in (-1, 1):
            for scale in (1, 2):
                shift = [0., 0.]
                shift[axis] = sign * scale * step
                rendered = model.render(camera.translated(*shift), fixed, with_depth=True)[0][0].cpu().numpy()
                probes.append((shift[0] * focal[0], shift[1] * focal[1],
                               cv2.cvtColor(rendered[:, :, :3], cv2.COLOR_RGB2GRAY), rendered[:, :, 3]))
    inv_low, inv_high = 1 / depths[-1], 1 / depths[0]
    uniform = 1 / np.linspace(inv_high, inv_low, len(depths), dtype=np.float32)
    start = perf_counter()
    raw_costs, visible_costs = (np.stack(group) for group in zip(
        *(photometric_cost(center, probes, d) for d in depths)))
    uniform_costs = np.stack([photometric_cost(center, probes, d)[0] for d in uniform])
    return {
        "median": (depths[2], np.where((alpha > .25) & np.isfinite(depths[2]),
                                      alpha / (depths[-1] / depths[0]), -np.inf)),
        "quantile_photo": select_depth(depths, raw_costs, alpha),
        "quantile_visible_photo": select_depth(depths, visible_costs, alpha),
        "uniform_photo": select_depth(uniform, uniform_costs, alpha),
    }, perf_counter() - start


@torch.inference_mode()
def main():
    """使用原 Replica 留出相机对评估光度选择。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    model = FrozenGaussians.load(args.checkpoint)
    cameras, truth, frames = load_replica(args.replica)
    pairs = list(evaluation_pairs(cameras))
    views = {}
    results = []
    for i, j, baseline in pairs:
        if i not in views:
            views[i] = read_view(model, cameras[i])
        projected, z = project_between_cameras(truth[i], cameras[i], cameras[j])
        visible = valid_pixels(truth[i], truth[j], projected, z)
        variants, _ = views[i]
        scores = {name: scores_for_pair(prediction, cameras[i], cameras[j], truth[i], projected, visible)
                  for name, prediction in variants.items()}
        results.append({"source": frames[i], "target": frames[j], "baseline": baseline, "scores": scores})
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump({"scene": args.replica.name, "pairs": results,
                   "matcher_seconds": {frames[i]: item[1] for i, item in views.items()}}, sink, allow_nan=False)
    print(f"{args.replica.name}: {len(results)} pairs complete", flush=True)


if __name__ == "__main__":
    main()
