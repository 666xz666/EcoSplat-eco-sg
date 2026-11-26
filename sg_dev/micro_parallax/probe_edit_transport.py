"""比较冻结 3DGS 的逐高斯回写和联合反解在留出视角的编辑泄漏。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from scipy import sparse

from evaluate import load_replica
from probe_visibility import load_tum
from readout import Camera, FrozenGaussians, ordered_alpha_samples


OBSERVED = (0, 2, 4, 6)
HELD_OUT = (1, 3, 5, 7)
BUDGET = 16
REGULARIZATION = 0.002


def select_views(cameras, frame_names):
    """只按位姿选视野相近的一组相机，按距离交错划分编辑和留出图。"""
    positions = np.stack([c.c2w[:3, 3] for c in cameras])
    optical_axes = np.stack([c.c2w[:3, 2] for c in cameras])
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    separation = distance + .65 * (1 - optical_axes @ optical_axes.T)
    count = len(OBSERVED) + len(HELD_OUT)
    anchor = np.argsort(separation, axis=1, kind="stable")[:, :count]
    neighborhood = np.take_along_axis(separation, anchor, axis=1)
    selection = anchor[neighborhood.mean(axis=1).argmin()]
    assert len(selection) == count
    return [cameras[i] for i in selection], [frame_names[i] for i in selection]


def reduced(camera, max_width):
    """缩小分辨率并保留原相机的有效视野和像素中心。"""
    divisor = 3 if camera.width >= 500 else 2
    width = min(camera.width // divisor, max_width)
    height = round(camera.height * width / camera.width)
    scale = np.array([width / camera.width, height / camera.height])
    intrinsic = camera.intrinsic.copy()
    intrinsic[:2, :2] *= scale[:, None]
    intrinsic[:2, 2] = (intrinsic[:2, 2] + .5) * scale - .5
    return Camera(camera.c2w, intrinsic, width, height)


def build_transport(model, camera):
    """生成固定几何下，高斯颜色改变量到像素改变量的稀疏矩阵。"""
    colors = model.colors(camera)
    rendered, pixel_alpha, metadata = model.render(camera, colors)
    pixels, splats, alpha, cumulative, _ = ordered_alpha_samples(metadata, camera.width, camera.height)
    weights = alpha * torch.exp(-(cumulative + torch.log1p(-alpha))).to(alpha.dtype)
    transport = sparse.csr_matrix(
        (weights.cpu().numpy(), (pixels.cpu().numpy(), splats.cpu().numpy())),
        shape=(camera.width * camera.height, len(model.xyz)),
        dtype=np.float32,
    )
    rgb = rendered[0].cpu().numpy().reshape(-1, 3)
    expected_alpha = pixel_alpha[0, ..., 0].cpu().numpy().reshape(-1)
    diff = np.max(np.abs((transport @ colors.cpu().numpy()) - rgb))
    alpha_diff = np.max(np.abs(np.asarray(transport.sum(axis=1)).ravel() - expected_alpha))
    if diff > 5e-4 or alpha_diff > 5e-4:
        raise ValueError(f"Sparse transport does not reproduce rendering: rgb {diff:.5g}, alpha {alpha_diff:.5g}")
    return transport, float(diff)


def patch_centers():
    """预定三个归一化编辑中心和亮度方向。"""
    return ((.4, .4, 1), (.63, .53, -1), (.52, .68, 1))


def known_edits(matrix, xyz, source):
    """构造三个相互独立的局部改色，用于检查同像素混合和场景编辑泄漏。"""
    edits = []
    regions = []
    for x_frac, y_frac, sign in patch_centers():
        width, height = source.width, source.height
        x, y = int(width * x_frac), int(height * y_frac)
        indices = np.arange((y - 2) * width, (y + 3) * width).reshape(-1, width)[:, x - 2:x + 3]
        contributions = np.asarray(matrix[indices.ravel()].sum(axis=0)).ravel()
        if contributions.max() <= 0:
            raise ValueError("Selected patch is not covered by a visible 3D region")
        seed = xyz[contributions.argmax()]
        radius = float(np.clip(.095 * np.linalg.norm(seed - source.c2w[:3, 3]), .08, .60))
        distances = np.linalg.norm(xyz - seed, axis=-1)
        magnitude = .12 * sign * np.clip((1 - distances / radius) * 3, 0, 1)
        edits.append(magnitude.astype(np.float32))
        regions.append({"seed": seed.tolist(), "radius": radius, "edited_splats": int((distances < radius).sum())})
    return np.stack(edits, axis=1), regions


def measured_edits(cameras, depths):
    """从独立 RGB-D 定义同一世界区域，评测时模拟用户提供的局部编辑。"""
    worlds, validities = [], []
    for camera, original in zip(cameras, depths):
        depth = cv2.resize(original, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST_EXACT)
        y, x = np.indices(depth.shape, dtype=np.float32)
        fx, fy = camera.intrinsic.diagonal()[:2]
        cx, cy = camera.intrinsic[:2, 2]
        local = np.stack(((x - cx) * depth / fx, (y - cy) * depth / fy, depth), axis=-1)
        world = local @ camera.c2w[:3, :3].T + camera.c2w[:3, 3]
        worlds.append(world.reshape(-1, 3).astype(np.float32))
        validities.append((np.isfinite(depth) & (depth > .15) & (depth < 9.9)).ravel())
    source = cameras[0]
    coordinates = worlds[0].reshape(source.height, source.width, 3)
    valid = validities[0].reshape(source.height, source.width)
    regions = []
    for fx, fy, sign in patch_centers():
        x, y = int(fx * source.width), int(fy * source.height)
        region = np.s_[y - 4:y + 5, x - 4:x + 5]
        points = coordinates[region][valid[region]]
        if len(points) < 8:
            raise ValueError("Too little measured depth around a predefined editing patch")
        seed = np.median(points, axis=0)
        radius = float(np.clip(.095 * np.linalg.norm(seed - source.c2w[:3, 3]), .08, .60))
        regions.append({"seed": seed.tolist(), "radius": radius, "sign": sign})
    outputs = []
    for points in worlds:
        rows = []
        for region in regions:
            dist = np.linalg.norm(points - np.asarray(region["seed"], dtype=np.float32), axis=1)
            rows.append((.12 * region["sign"] * np.clip((1 - dist / region["radius"]) * 3, 0, 1)).astype(np.float32))
        outputs.append(np.stack(rows, axis=1))
    return outputs, validities, regions


def solve_edits(transport, target, budget=BUDGET):
    """使用相同观测、正则化和矩阵乘预算比较两种回写更新。

    Args:
        transport (csr_matrix, (P, N)): alpha 混合的像素到高斯算子。
        target (ndarray, (P, K)): K 个独立的局部编辑指令。
        budget (int): 每个迭代解法使用的前向与伴随算子对数。

    Returns:
        tuple: 各解法的高斯改变量 (N, K)、求解时间和目标函数。
    """
    adjoint = transport.transpose().tocsr()
    mass = np.asarray(transport.sum(axis=0)).ravel()[:, None]
    ridge = REGULARIZATION * np.maximum(mass, .03)
    diag = np.asarray(transport.power(2).sum(axis=0)).ravel()[:, None] + ridge
    rhs = adjoint @ target

    def normal(values):
        return adjoint @ (transport @ values) + ridge * values

    observed_avg = rhs / np.maximum(mass + ridge, 1e-8)
    time_log = {}
    started = perf_counter()
    instant = observed_avg.copy()
    for _ in range(budget - 1):
        instant += (rhs - normal(instant)) / np.maximum(mass + ridge, 1e-8)
    time_log["instant_iterative"] = perf_counter() - started

    started = perf_counter()
    pcg = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual / diag
    residual_inner = np.sum(residual * direction, axis=0, dtype=np.float64)
    for _ in range(budget):
        product = normal(direction)
        curvature = np.sum(direction * product, axis=0, dtype=np.float64)
        step = (residual_inner / np.maximum(curvature, 1e-28)).astype(np.float32)
        pcg += direction * step
        residual -= product * step
        preconditioned = residual / diag
        next_inner = np.sum(residual * preconditioned, axis=0, dtype=np.float64)
        ratio = (next_inner / np.maximum(residual_inner, 1e-28)).astype(np.float32)
        direction = preconditioned + direction * ratio
        residual_inner = next_inner
    time_log["joint_pcg"] = perf_counter() - started

    signs = np.sign(np.sum(target, axis=0)).astype(np.float32)
    safe_step = 1 / np.maximum(normal(np.ones_like(mass)), 1e-8)
    directed_rhs = rhs * signs
    upper = np.full(rhs.shape, .12, dtype=np.float32)
    preserve_caps = []
    for channel in range(target.shape[1]):
        clean = np.abs(target[:, channel]) <= .0005
        if clean.any():
            keep_weight = transport[clean].max(axis=0).toarray().ravel()
            cap = .0015 / np.maximum(keep_weight, 1e-8)
        else:
            cap = upper[:, channel]
        preserve_caps.append(cap)
    bounded = np.minimum(upper, np.stack(preserve_caps, axis=1))

    def projected(limit):
        """在编辑方向和每个高斯上限内迭代求解，避免正负改动抵消。"""
        solution = np.zeros_like(rhs)
        extrapolated = solution
        momentum = 1.0
        for _ in range(budget):
            updated = np.clip(extrapolated + (directed_rhs - normal(extrapolated)) * safe_step, 0, limit)
            next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
            extrapolated = updated + np.float32((momentum - 1) / next_momentum) * (updated - solution)
            solution, momentum = updated, next_momentum
        return solution * signs

    started = perf_counter()
    monotone = projected(upper)
    time_log["joint_monotone"] = perf_counter() - started
    started = perf_counter()
    protected = projected(bounded)
    time_log["joint_preserve"] = perf_counter() - started

    estimates = {"instant_one_pass": observed_avg, "instant_iterative": instant, "joint_pcg": pcg,
                 "joint_pcg_clipped": np.clip(pcg * signs, 0, upper) * signs,
                 "joint_monotone": monotone, "joint_preserve": protected}
    objectives = {name: float(np.sum((transport @ x - target) ** 2) + np.sum(ridge * x ** 2))
                  for name, x in estimates.items()}
    return estimates, time_log, objectives


def evaluation(truth, prediction, min_pixels=30):
    """比较留出视角编辑区域保真度和区域外误改量。"""
    support = np.abs(truth) > .001
    if support.sum() < min_pixels or np.abs(truth).sum() < .1:
        return None
    magnitude = np.abs(truth).sum()
    error = prediction - truth
    return {
        "normalized_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
        "relative_spill": float(np.abs(prediction[~support]).sum() / magnitude),
        "relative_edit_l1": float(np.abs(error[support]).sum() / magnitude),
        "edited_pixels": int(support.sum()),
    }


@torch.inference_mode()
def main():
    """在一场已有 3DGS 地图上做受控局部改色短验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replica", type=Path)
    source.add_argument("--tum", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=200)
    parser.add_argument("--independent-depth", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    loader = load_tum if args.tum else load_replica
    cameras, depths, names = loader(args.tum or args.replica)
    depth_by_name = dict(zip(names, depths))
    cameras, names = select_views(cameras, names)
    cameras = [reduced(camera, args.max_width) for camera in cameras]
    model = FrozenGaussians.load(args.checkpoint)
    views = [build_transport(model, camera) for camera in cameras]
    matrices, discrepancies = zip(*views)
    if args.independent_depth:
        references, valid, regions = measured_edits(cameras, [depth_by_name[name] for name in names])
    else:
        truth, regions = known_edits(matrices[0], model.xyz.cpu().numpy(), cameras[0])
        references = [matrix @ truth for matrix in matrices]
        valid = [np.ones(matrix.shape[0], dtype=bool) for matrix in matrices]
    observed = sparse.vstack([matrices[i][valid[i]] for i in OBSERVED], format="csr")
    targets = np.concatenate([references[i][valid[i]] for i in OBSERVED], axis=0)
    active = np.flatnonzero(np.asarray(observed.sum(axis=0)).ravel())
    fit = observed[:, active].tocsr()
    edits, seconds, objectives = solve_edits(fit, targets)
    cancellations = {}
    for name, update in edits.items():
        cancellation = (fit @ np.abs(update)) - np.abs(fit @ update)
        cancellations[name] = float(np.maximum(cancellation, 0).sum() / np.abs(targets).sum())
    tested = []
    for i in HELD_OUT:
        reference = references[i][valid[i]]
        test = matrices[i][valid[i]][:, active]
        predicted = {method: test @ edit for method, edit in edits.items()}
        for k in range(reference.shape[1]):
            metrics = {method: evaluation(reference[:, k], values[:, k], min_pixels=12)
                       for method, values in predicted.items()}
            if all(row is not None for row in metrics.values()):
                tested.append({"frame": names[i], "edit": k, "metrics": metrics})
    result = {
        "edits": regions, "observed": [names[i] for i in OBSERVED], "held_out": [names[i] for i in HELD_OUT],
        "image_size": [cameras[0].width, cameras[0].height], "active_splats": len(active),
        "maximum_rgb_forward_error": float(max(discrepancies)), "budget": BUDGET,
        "independent_depth": args.independent_depth, "source_cancellation": cancellations,
        "elapsed_solve_seconds": seconds, "training_objective": objectives, "held_out_scores": tested,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print(f"{len(tested)} held-out view-edit pairs evaluated; forward error={max(discrepancies):.3g}", flush=True)


if __name__ == "__main__":
    main()
