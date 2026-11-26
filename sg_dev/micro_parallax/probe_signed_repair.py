"""用实际 RGB 的跨视角留出验证冻结 3DGS 的有害高斯衰减。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bench_adaptive_raster import heldout_views, scene_paths
from probe_edit_transport import reduced
from readout import FrozenGaussians, ordered_alpha_samples


FRACTIONS = (.002, .01, .04)


def removal_delta(alpha, pixels, weighted_colors):
    """计算单项清除后的 RGB 差值，后方颜色由同一次前向合成恢复。

    Args:
        alpha (Tensor, (M,)): 光栅排序内每项的原始 opacity。
        pixels (LongTensor, (M,)): 像素索引，相同索引连续。
        weighted_colors (Tensor, (M, 3)): 已乘透过率与 opacity 的颜色贡献。

    Returns:
        Tensor: 原色减去单独清除该项后的 RGB，形状 (M, 3)。
    """
    _, count = torch.unique_consecutive(pixels, return_counts=True)
    end = count.cumsum(dim=0) - 1
    prefix = weighted_colors.cumsum(dim=0, dtype=torch.float64)
    suffix = (prefix[end].repeat_interleave(count, dim=0) - prefix).to(weighted_colors.dtype)
    return weighted_colors - (alpha / (1 - alpha))[:, None] * suffix


def view_signal(model, camera, colors, target):
    """对同一个实际观测估计每个高斯的移除收益和二阶项。"""
    render, _, meta = model.render(camera, colors)
    pi, gi, alpha, cumulative, _ = ordered_alpha_samples(meta, camera.width, camera.height)
    trans = torch.exp(-(cumulative + torch.log1p(-alpha))).to(alpha.dtype)
    weight = trans * alpha
    effect = removal_delta(alpha, pi, weight[:, None] * colors[gi])
    residual = render[0].reshape(-1, 3) - target.reshape(-1, 3)
    dot = (effect * residual[pi]).sum(dim=-1)
    curvature = effect.square().sum(dim=-1)
    n = len(model.xyz)
    totals = [torch.zeros(n, device=weight.device, dtype=weight.dtype) for _ in range(3)]
    for dest, values in zip(totals, (dot, curvature, weight)):
        dest.index_add_(0, gi, values)
    return torch.stack(totals)


def select(signals, fraction):
    """在相同预算用一阶打分、条件衰减和多视角确认选择贡献项。"""
    dot, square, _ = signals.sum(0)
    visible = signals[:, 2] > .12
    votes = (visible & (signals[:, 0] > 0)).sum(dim=0)
    held = visible.sum(dim=0)
    count = max(1, int(len(dot) * fraction))
    attenuation = (dot / (square + .004)).clamp(0, 1)
    gain = attenuation * (2 * dot - attenuation * square)
    eligible = held >= 1
    supported = (held >= 2) & (votes * 3 >= held * 2)
    methods = {
        "opacity_gradient": (dot, eligible, attenuation),
        "removal_loss": (2 * dot - square, eligible, torch.ones_like(attenuation)),
        "conditional": (gain, eligible, attenuation),
        "multiview_confirmed": (gain, supported, attenuation),
    }
    variants = {}
    for name, (score, pool, amount) in methods.items():
        ids = torch.topk(torch.where(pool, score, -torch.inf), count).indices
        ids = ids[pool[ids] & (score[ids] > 0)]
        mask = torch.ones_like(attenuation)
        mask[ids] = 1 - amount[ids]
        variants[name] = (mask, ids)
    return variants


def target_image(rgb, camera, device):
    """把公开数据中的实际照片缩放到统一评测分辨率。"""
    small = cv2.resize(rgb, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
    return torch.as_tensor(small, device=device, dtype=torch.float32) / 255


def prepare_views(model, views):
    """预存渲染条件与真实 RGB；标定和评测的照片从不交叉。"""
    return [(reduced(camera, 144), frame, rgb, model.colors(camera)) for camera, frame, rgb in views]


def measure(model, views, scales):
    """只用未参与评分的相机比较干预前后 RGB 误差。"""
    baseline = model.opacity
    results = {}
    for name, factor in scales.items():
        model.opacity = baseline * factor
        scores = []
        for camera, frame, rgb, colors in views:
            target = target_image(rgb, camera, model.xyz.device)
            render = model.render(camera, colors)[0][0].clamp(0, 1)
            error = (render - target).square()
            scores.append({"frame": frame, "mse": float(error.mean())})
        results[name] = scores
    model.opacity = baseline
    return results


def clone_distractors(model, camera, total=400):
    """构造来自真实场景分布的近相机浮点，仅用作受控压力测试。"""
    colors = model.colors(camera)
    _, _, meta = model.render(camera, colors)
    p, ids, alpha, cumulative, _ = ordered_alpha_samples(meta, camera.width, camera.height)
    height, width = camera.height, camera.width
    crop = ((p // width - height * .43).abs() < height * .2) & ((p % width - width * .57).abs() < width * .2)
    weight = alpha * torch.exp(-(cumulative + torch.log1p(-alpha))).to(alpha.dtype)
    scores = torch.zeros(len(model.xyz), dtype=weight.dtype, device=weight.device)
    scores.index_add_(0, ids, weight * crop)
    # 先从覆盖有证据的较大池取样，保持背景、大小和颜色多样性。
    pool = torch.topk(scores, total * 4).indices
    generator = torch.Generator(device=scores.device).manual_seed(11)
    selected = pool[torch.randperm(len(pool), generator=generator, device=scores.device)[:total]]
    origin = torch.as_tensor(camera.c2w[:3, 3].copy(), device=weight.device, dtype=torch.float32)
    offset = torch.empty(total, device=scores.device).uniform_(.69, .9, generator=generator)
    xyz = origin + (model.xyz[selected] - origin) * offset[:, None]
    old_count = len(model.xyz)
    model.xyz = torch.cat((model.xyz, xyz))
    model.rotation = torch.cat((model.rotation, model.rotation[selected]))
    model.scale = torch.cat((model.scale, model.scale[selected] * offset[:, None]))
    model.opacity = torch.cat((model.opacity, model.opacity[selected].clamp(.22, .88)))
    model.sh = torch.cat((model.sh, model.sh[selected]))
    return old_count


@torch.inference_mode()
def main():
    """跑现有场景的短测，不更新检查点。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--scene", choices=("room0", "office0", "tum"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    source, path, loader = scene_paths(args.data, args.checkpoints, args.scene)
    camera_rows = list(heldout_views(source, loader, 10))
    model = FrozenGaussians.load(path)
    original_count = len(model.xyz)
    if args.stress:
        clone_distractors(model, reduced(camera_rows[0][0], 144))
    views = prepare_views(model, camera_rows)
    control = views[::2]
    evaluation = views[1::2]
    assert {v[1] for v in control}.isdisjoint(v[1] for v in evaluation)
    signals = torch.stack([view_signal(model, c, colors, target_image(rgb, c, model.xyz.device))
                           for c, _, rgb, colors in control])
    print(f"{len(control)} scoring photos complete; evaluating {len(evaluation)} separate photos", flush=True)
    masks = {"original": torch.ones_like(model.opacity)}
    chosen = {}
    for rate in FRACTIONS:
        for method, (scale, ids) in select(signals, rate).items():
            tag = f"{method}_{rate:g}"
            masks[tag] = scale
            chosen[tag] = {"count": int(len(ids)), "injected": int((ids >= original_count).sum()),
                           "effective_removed": float((1 - scale).sum())}
    if args.stress:
        oracle = torch.ones_like(model.opacity)
        oracle[original_count:] = 0
        masks["oracle_clean"] = oracle
    result = {"scene": args.scene, "stress": args.stress, "scoring_frames": [v[1] for v in control],
              "evaluation_frames": [v[1] for v in evaluation], "decisions": chosen,
              "measurements": measure(model, evaluation, masks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print("Exclusive held-out evaluation complete", flush=True)


if __name__ == "__main__":
    main()
