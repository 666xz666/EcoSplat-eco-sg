"""在同一官方 EcoSplat 候选集上测量投影覆盖对单帧渲染的影响。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import fields
from importlib import import_module
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor

from profile_ecosplat_post_forward import wall
from profile_spfsplat_baseline import (
    PAPER_RELEASE,
    compile_observed_function,
    construct_input,
    make_encoder,
    select_scenes,
    to_device,
    verified_stage2,
)

COUNTS = (32_768, 65_536)
GROUPS = ("topk", "low_coverage", "high_coverage")
KERNELS = {
    "fully_fused_projection": "project_ms",
    "isect_tiles": "tile_sort_ms",
    "rasterize_to_pixels": "composite_ms",
}
CANDIDATES = (
    ("pts_list = []", "before_topk",
     "points=pts_all, parameters=merge_gaussian_parameters, densities=merge_densities,"
     " alpha=learn_merged_opacity, rates=primitive_ratio_per_view"),
)


def view_quotas(rates: np.ndarray, total: int, per_view: int) -> np.ndarray:
    """从 EcoSplat 高频比例按最大余数法得到精确的整数配额。

    Args:
        rates (Float[ndarray, "view"]): 官方前向计算的各源视图保留率。
        total (int): 目标场景高斯数。
        per_view (int): 每个源视图的候选数。

    Returns:
        Int[ndarray, "view"]: 按源视图排列的配额，总和等于 total。
    """
    if not np.isfinite(rates).all() or (rates < 0).any() or rates.sum() <= 0:
        raise ValueError(f"invalid official view ratios: {rates}")
    ideal = rates.astype(np.float64) * total / rates.astype(np.float64).sum()
    quotas = np.floor(ideal).astype(np.int64)
    if (quotas > per_view).any():
        raise ValueError(f"view quotas exceed {per_view} candidates: {quotas}")
    remainder = total - int(quotas.sum())
    order = np.argsort(-(ideal - quotas), kind="stable")
    eligible = order[quotas[order] < per_view]
    if not 0 <= remainder <= len(eligible):
        raise ValueError(f"cannot allocate {total} points with view quotas {quotas}")
    quotas[eligible[:remainder]] += 1
    return quotas


def opacity_topk(alpha: Tensor, quotas: np.ndarray) -> Tensor:
    """按源视图复用官方 torch.topk，并返回视图优先的平坦索引。"""
    per_view = alpha.shape[1]
    return torch.cat([
        torch.topk(scores, k=int(quota)).indices + view * per_view
        for view, (scores, quota) in enumerate(zip(alpha, quotas))
    ])


def matching_strata(means: np.ndarray, alpha: np.ndarray, grid: int, bins: int) -> np.ndarray:
    """按源视图、三维分块和不透明度区间分层。

    Args:
        means (Float[ndarray, "view pixel 3"]): 世界坐标中的高斯中心。
        alpha (Float[ndarray, "view pixel"]): 官方映射后的不透明度。
        grid (int): 场景每个坐标轴的分块数。
        bins (int): [0, 1] 内均匀划分的不透明度区间数。

    Returns:
        Int[ndarray, "candidate"]: 与官方展平顺序一致的分层编号。
    """
    flat = means.reshape(-1, 3)
    lower, upper = np.quantile(flat, (0.02, 0.98), axis=0)
    scaled = np.clip((flat - lower) / np.maximum(upper - lower, 1e-8), 0.0, 1.0)
    xyz = np.minimum((scaled * grid).astype(np.int64), grid - 1)
    block = (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]
    opacity = np.minimum((alpha.flatten() * bins).astype(np.int64), bins - 1)
    view = np.repeat(np.arange(alpha.shape[0]), alpha.shape[1])
    return ((view * grid**3 + block) * bins) + opacity


def coverage_extremes(
    strata: np.ndarray, tiles: np.ndarray, baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """保持基线各层点数，从每层取最低和最高源视角覆盖的候选。

    Args:
        strata (Int[ndarray, "candidate"]): 混合分层编号，包含源视图。
        tiles (Int[ndarray, "candidate"]): 候选在全部源视图的 tile 相交数之和。
        baseline (Int[ndarray, "selected"]): opacity Top-K 的平坦索引。

    Returns:
        tuple[Int[ndarray, "selected"], Int[ndarray, "selected"]]: 低、高覆盖索引。
    """
    num_strata = int(strata.max()) + 1
    wanted = np.bincount(strata[baseline], minlength=num_strata)
    population = np.bincount(strata, minlength=num_strata)
    start = np.cumsum(population) - population
    result = []
    for sign in (1, -1):
        # 步骤1：相同覆盖以原始索引打破平局，避免把随机差异当成处理效应。
        order = np.lexsort((np.arange(len(tiles)), sign * tiles, strata))
        rank = np.arange(len(order)) - start[strata[order]]
        picked = order[rank < wanted[strata[order]]]
        # 步骤2：三组渲染前统一原始索引顺序，保持同深度相交项的次序约定。
        if not np.array_equal(np.bincount(strata[picked], minlength=num_strata), wanted):
            raise AssertionError("matched strata counts changed")
        result.append(np.sort(picked))
    return result[0], result[1]


def align_cameras(truth: Tensor, predicted: Tensor) -> tuple[Tensor, float, float]:
    """仅用源相机求相似变换，并把所有相机映射到同一预测坐标系。

    Args:
        truth (Float[Tensor, "all_view 4 4"]): 数据集相机到世界矩阵，源帧在前。
        predicted (Float[Tensor, "source_view 4 4"]): 仅源图一次前向的相机。

    Returns:
        tuple: 对齐后的矩阵 (all_view, 4, 4)、尺度和源相机中心 RMSE。
    """
    observed = truth[:len(predicted)].double()
    predicted = predicted.double()
    # 步骤1：用相机朝向求旋转，避免近直线轨迹的点中心配准存在自由轴。
    covariance = (predicted[:, :3, :3] @ observed[:, :3, :3].transpose(-1, -2)).mean(dim=0)
    left, _, right = torch.linalg.svd(covariance)
    left[:, -1] *= torch.linalg.det(left @ right)
    rotation = left @ right
    source = observed[:, :3, 3] @ rotation.T
    dest = predicted[:, :3, 3]
    xs, ys = source - source.mean(dim=0), dest - dest.mean(dim=0)
    spread = xs.square().sum()
    if spread <= 1e-12:
        raise ValueError("source camera centers have zero baseline")
    scale = (xs * ys).sum() / spread
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid source-only camera alignment scale: {scale}")
    translation = dest.mean(dim=0) - scale * source.mean(dim=0)
    # 步骤2：目标图像不参与求解，所有组共享相同的目标坐标和内参。
    aligned = truth.double().clone()
    aligned[:, :3, :3] = rotation @ aligned[:, :3, :3]
    aligned[:, :3, 3] = scale * (aligned[:, :3, 3] @ rotation.T) + translation
    rmse = (aligned[:len(predicted), :3, 3] - dest).square().sum(dim=-1).mean().sqrt()
    return aligned.to(truth), float(scale), float(rmse)


def load_scene(chunk, scene, indices, size):
    """取同一场景的六张源图与三张留出目标，并读入数据集相机。"""
    start, end = indices[:2]
    targets = [start + (end - start) * k // 10 for k in (3, 5, 7)]
    if len(set(targets + indices)) != len(targets) + len(indices):
        raise ValueError(f"source/target overlap in {scene}: {indices}, {targets}")
    frames = indices + targets
    data, _ = construct_input(chunk, scene, frames, size)
    sample = next(x for x in torch.load(chunk, map_location="cpu", weights_only=True) if x["key"] == scene)
    w2c = torch.eye(4).repeat(len(frames), 1, 1)
    w2c[:, :3] = sample["cameras"][frames, 6:].reshape(-1, 3, 4)
    truth = torch.linalg.inv(w2c)
    source_baseline = float((truth[0, :3, 3] - truth[1, :3, 3]).norm())
    if source_baseline <= 1e-8:
        raise ValueError(f"zero reference baseline in {scene}")
    truth[:, :3, 3] /= source_baseline
    truth = torch.linalg.inv(truth[0]) @ truth
    v = len(indices)
    context = to_device({k: x[:, :v] for k, x in data.items()})
    context["extrinsics"] = truth[:v].unsqueeze(0).cuda()
    target = to_device({k: x[:, v:] for k, x in data.items()})
    return context, target, truth, source_baseline, targets


def subset(gaussians, indices: Tensor):
    """按同一 (selected,) 索引裁取所有高斯字段，保留官方张量布局。"""
    return type(gaussians)(**{
        field.name: getattr(gaussians, field.name).index_select(1, indices) for field in fields(gaussians)
    })


def encode_candidates(encoder, context, step: int):
    """一次前向捕获官方 Top-K 前的条件候选，并校验原生输出可重建。"""
    captured = {}

    def capture(_stage, **values):
        captured.update(values)

    observed = compile_observed_function(encoder.forward.__func__, CANDIDATES, capture)
    with patch.object(encoder, "forward", MethodType(observed, encoder)):
        stock = encoder(context, global_step=step)
    points, params, densities = (captured[key] for key in ("points", "parameters", "densities"))
    alpha = captured["alpha"]
    if alpha.ndim != 3 or alpha.shape[0] != 1 or points.shape[3] != 1:
        raise ValueError(f"expected batch 1 and one Gaussian per source pixel, got {alpha.shape}, {points.shape}")
    dense = encoder.gaussian_adapter.forward(
        points.unsqueeze(-2), encoder.map_pdf_to_opacity(densities, step), params.unsqueeze(-2)
    )
    compact = stock["gaussians"]
    candidates = type(compact)(**{
        field.name: getattr(dense, field.name).flatten(1, 4) for field in fields(dense)
    })
    flat_alpha = alpha[0]
    views, per_view = flat_alpha.shape
    if candidates.means.shape[:2] != (1, views * per_view):
        raise ValueError(f"unexpected condition-candidate shape: {candidates.means.shape}")
    torch.testing.assert_close(candidates.opacities[0], flat_alpha.flatten())
    rates = captured["rates"].detach().cpu().numpy()
    # 先核验未经余数修正的原生输出，再使用固定点数的同一套高频配额和 Top-K。
    native_counts = np.array([int(rate * per_view) for rate in captured["rates"]])
    native = subset(candidates, opacity_topk(flat_alpha, native_counts))
    for field in fields(compact):
        torch.testing.assert_close(getattr(native, field.name), getattr(compact, field.name))
    return candidates, flat_alpha, rates, stock["extrinsics"]["c"]


def source_footprint(gaussians, cameras: Tensor, intrinsics: Tensor, size: int, scale: float) -> Tensor:
    """按官方 gsplat 投影参数统计每点在六张源视图上的 tile 相交之和。"""
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles

    cameras = cameras[0].clone()
    cameras[:, :3, 3] *= scale
    intrinsics = intrinsics[0].clone()
    intrinsics[:, :2] *= size
    means, scales = gaussians.means[0] * scale, gaussians.scales[0] * scale
    rotations, alpha = gaussians.rotations[0], gaussians.opacities[0]
    tiles = torch.zeros_like(alpha, dtype=torch.int64)
    side = (size + 15) // 16
    for w2c, intr in zip(torch.linalg.inv(cameras), intrinsics):
        radii, means2d, depths, _, _ = fully_fused_projection(
            means, None, rotations, scales, w2c[None], intr[None], size, size,
            near_plane=1e-10, radius_clip=0.1, opacities=alpha,
        )
        per_point, intersection_ids, flatten_ids = isect_tiles(means2d, radii, depths, 16, side, side, sort=False)
        tiles += per_point[0]
        del intersection_ids, flatten_ids
    return tiles


def group_candidates(candidates, alpha, rates, tile_sum, total, grid, bins):
    """构造严格配对的三组，并记录源视图代理覆盖和不透明度平衡。"""
    views, per_view = alpha.shape
    quotas = view_quotas(rates, total, per_view)
    baseline = opacity_topk(alpha, quotas).sort().values.cpu().numpy()
    alpha_cpu = alpha.cpu().numpy()
    means_cpu = candidates.means[0].reshape(views, per_view, 3).cpu().numpy()
    if not np.isfinite(means_cpu).all() or not ((0 <= alpha_cpu) & (alpha_cpu <= 1)).all():
        raise ValueError("candidate geometry or opacity is invalid")
    strata = matching_strata(means_cpu, alpha_cpu, grid, bins)
    cost = tile_sum.cpu().numpy()
    low, high = coverage_extremes(strata, cost, baseline)
    totals = [int(cost[ids].sum()) for ids in (baseline, low, high)]
    if not totals[1] <= totals[0] <= totals[2]:
        raise AssertionError(f"coverage extrema failed: {totals}")
    rows = {}
    for name, ids, source_total in zip(GROUPS, (baseline, low, high), totals):
        if len(ids) != total or len(np.unique(ids)) != total:
            raise AssertionError(f"incorrect count for {name}")
        rows[name] = {
            "source_tiles_mean": source_total / views,
            "opacity_mean": float(alpha_cpu.flatten()[ids].mean()),
            "overlap_with_topk": int(np.intersect1d(ids, baseline, assume_unique=True).size),
        }
    indices = {name: torch.as_tensor(ids, device=alpha.device) for name, ids in zip(GROUPS, (baseline, low, high))}
    return indices, rows, quotas.tolist(), int((np.bincount(strata[baseline]) > 0).sum())


def measure_groups(decoder, gaussians, pose, intrinsics, near, far, image, repeats: int):
    """交错计时三组，独立采集官方 gsplat 元数据与分段 CUDA Event。"""
    rendering = import_module("gsplat.rendering")
    adapter = import_module("src.model.decoder.cuda_splatting")
    rows = {name: {"wall_ms": [], "stages_ms": []} for name in GROUPS}

    def render(name):
        return decoder(gaussians[name], pose, intrinsics, near, far, image.shape[-2:])

    meta = {}
    stock_rasterize = adapter.rasterization

    def inspect_rasterize(*args, **kwargs):
        out = stock_rasterize(*args, **kwargs)
        meta.update(out[2])
        return out

    for name in GROUPS:
        for _ in range(5):
            render(name)
        meta.clear()
        with patch.object(adapter, "rasterization", inspect_rasterize):
            output = render(name)
        radii = meta["radii"]
        visible = (radii > 0).all(dim=-1) if radii.ndim == 3 else radii > 0
        mse = (output.color.clamp(0, 1) - image).square().mean()
        if not torch.isfinite(mse):
            raise ValueError(f"non-finite rendered image for {name}")
        rows[name].update(
            visible_gaussians=int(visible.sum()),
            tile_intersections=meta["isect_ids"].numel(),
            psnr_db=float(-10 * torch.log10(mse.clamp_min(1e-12))),
        )
        del output, radii, visible
        meta.clear()

    def order(index):
        offset = index % len(GROUPS)
        return GROUPS[offset:] + GROUPS[:offset]

    for i in range(repeats):
        for name in order(i):
            elapsed, output = wall(lambda: render(name))
            rows[name]["wall_ms"].append(elapsed)
            del output

    events = {}

    def observe(label, function):
        def measured(*args, **kwargs):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = function(*args, **kwargs)
            stop.record()
            events.setdefault(label, []).append((start, stop))
            return output
        return measured

    with ExitStack() as stack:
        for symbol, label in KERNELS.items():
            stack.enter_context(patch.object(rendering, symbol, observe(label, getattr(rendering, symbol))))
        for i in range(repeats):
            for name in order(i):
                events.clear()
                render(name)
                torch.cuda.synchronize()
                if set(events) != set(KERNELS.values()):
                    raise RuntimeError(f"missing gsplat timing stages: {list(events)}")
                rows[name]["stages_ms"].append({
                    label: sum(start.elapsed_time(stop) for start, stop in spans)
                    for label, spans in events.items()
                })
    return rows


def run_scene(encoder, decoder, step, scene, chunk, indices, args):
    """复用场景级条件候选，逐预算、逐目标视图执行配对计时。"""
    context, target, truth, source_baseline, targets = load_scene(chunk, scene, indices, args.size)
    candidates, alpha, rates, pred_source = encode_candidates(encoder, context, step)
    aligned, scale, rmse = align_cameras(truth, pred_source[0].cpu())
    near_value = scale * 0.1 / source_baseline
    far_value = scale * 100.0 / source_baseline
    aligned = aligned[len(indices):][None].cuda()
    near = torch.full(aligned.shape[:2], near_value, device=aligned.device)
    far = torch.full_like(near, far_value)
    projection_scale = 1.0 / near_value if decoder.make_scale_invariant else 1.0
    tile_sum = source_footprint(candidates, pred_source, context["intrinsics"], args.size, projection_scale)
    cases = []
    for total in COUNTS:
        group_ids, selection, quotas, strata = group_candidates(
            candidates, alpha, rates, tile_sum, total, args.grid, args.opacity_bins
        )
        groups = {name: subset(candidates, ids) for name, ids in group_ids.items()}
        print(f"BUDGET {scene} {total} source_tiles="
              f"{[round(selection[name]['source_tiles_mean']) for name in GROUPS]}", flush=True)
        if selection["low_coverage"]["source_tiles_mean"] == selection["high_coverage"]["source_tiles_mean"]:
            print(f"NO_SOURCE_COVERAGE_CONTRAST {scene} {total}", flush=True)
        for j, frame in enumerate(targets):
            sl = slice(j, j + 1)
            metrics = measure_groups(
                decoder, groups, aligned[:, sl], target["intrinsics"][:, sl],
                near[:, sl], far[:, sl], target["image"][:, sl], args.repeats,
            )
            print(f"VIEW_DONE {scene} {total} target={frame}", flush=True)
            cases.append({
                "scene": scene, "gaussians": total, "target_index": frame, "source_indices": indices,
                "view_quotas": quotas, "occupied_strata": strata, "source_pose_rmse": rmse,
                "groups": {name: {**selection[name], **metrics[name]} for name in GROUPS},
            })
    return cases


def write_results(result, output):
    """保存原始重复样本和逐场景、逐目标、逐组的中位数 CSV。"""
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    rows = []
    for case in result["cases"]:
        for name, measurements in case["groups"].items():
            row = {k: case[k] for k in ("scene", "gaussians", "target_index")}
            row.update(group=name, wall_ms=median(measurements["wall_ms"]))
            row.update({k: v for k, v in measurements.items() if k not in ("stages_ms", "wall_ms")})
            row.update({
                stage: median(sample[stage] for sample in measurements["stages_ms"])
                for stage in KERNELS.values()
            })
            rows.append(row)
    csv_path = output.with_name("summary.csv")
    with csv_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".csv.tmp").replace(csv_path)


def main():
    """在已核验的官方检出上运行固定三场景和两档高斯数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "dataset", "stage1", "stage2", "output"):
        parser.add_argument(f"--{key}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--stage2-sha256", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--opacity-bins", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1 or args.size < 16 or args.size % 16 or min(args.grid, args.opacity_bins) < 1:
        parser.error("positive repetitions and bins, and a size divisible by 16, are required")
    revision = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    origin = subprocess.check_output(["git", "-C", args.repo, "remote", "get-url", "origin"], text=True).strip()
    if revision != PAPER_RELEASE or origin != "https://github.com/KAIST-VICLab/EcoSplat.git":
        raise RuntimeError(f"unexpected official checkout: {origin} {revision}")
    args.paper_release, args.rho = True, 0.4
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    from gsplat import __version__ as gsplat_version
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.model.decoder import get_decoder

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    print("VERIFY_STAGE2", flush=True)
    verified_stage2(args.stage2, args.stage2_sha256)
    print("LOAD_BASELINE", revision, flush=True)
    encoder, step = make_encoder(args)
    with initialize_config_dir(config_dir=str(args.repo / "config"), version_base=None):
        config = compose(config_name="main", overrides=["+experiment=ecosplat/re10k", "mode=test"])
    decoder = get_decoder(load_typed_root_config(config).model.decoder).cuda().eval()
    benchmark_revision = subprocess.check_output(
        ["git", "-C", Path(__file__).resolve().parents[1], "rev-parse", "HEAD"], text=True
    ).strip()
    result = dict(
        benchmark_commit=benchmark_revision, official_commit=revision, stage2_sha256=args.stage2_sha256, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, gsplat=gsplat_version, size=args.size, counts=COUNTS, rho=args.rho,
        grid=args.grid, opacity_bins=args.opacity_bins, repeats=args.repeats, cases=[],
    )
    with torch.no_grad():
        for scene, chunk, indices in select_scenes(args.repo, args.dataset, 3, paper_release=True):
            print(f"CASE_START {scene}", flush=True)
            result["cases"].extend(run_scene(encoder, decoder, step, scene, chunk, indices, args))
            write_results(result, args.output)
            print(f"CASE_DONE {scene}", flush=True)
    print("COVERAGE_PROFILE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
