"""在冻结 3DGS 上选择少量深度标定相机，验证其对指定新视角的用途。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from bench_adaptive_raster import scene_paths
from probe_depth_adapter import calibration_view, local_sensor_fill, sensor_points, sensor_reprojection, solve
from probe_depth_repair import depth_targets
from probe_edit_transport import reduced
from prepare_tum import nearest, timed_files, warped_image
from readout import Camera, FrozenGaussians, ordered_alpha_samples


def fresh_tum_depth(prepared, fold):
    """从同一 TUM 序列读取未用于训练和此前留出的深度帧。"""
    metadata = json.loads((prepared / "tum-metadata.json").read_text())
    root = Path(metadata["source"])
    rgb_time, _ = timed_files(root / "rgb.txt")
    depth_time, depth_files = timed_files(root / "depth.txt")
    association = nearest(depth_time, rgb_time)
    known = set(metadata["training_frames"]) | {item["frame"] for item in metadata["evaluation"]}
    old_depth = set(association[sorted(known)].tolist())
    trajectory = np.loadtxt(root / "groundtruth.txt")
    stamps = trajectory[:, 0]
    observed = depth_time[association]
    after = np.searchsorted(stamps, observed).clip(1, len(stamps) - 1)
    valid = ((np.abs(rgb_time - observed) <= .012) & (observed >= stamps[0]) & (observed <= stamps[-1]) &
             (stamps[after] - stamps[after - 1] <= .05))
    pool = np.array([i for i in np.flatnonzero(valid) if i not in known and association[i] not in old_depth], dtype=np.int64)
    _, unique = np.unique(association[pool], return_index=True)
    pool = pool[np.sort(unique)]
    if len(pool) < 48:
        raise ValueError(f"Only {len(pool)} independent time-aligned depth frames")
    frames = pool[np.rint(np.linspace(0, len(pool) - 1, 48)).astype(int)][fold::2]
    timestamps = observed[frames]
    poses = np.repeat(np.eye(4)[None], len(frames), axis=0)
    poses[:, :3, :3] = Slerp(stamps, Rotation.from_quat(trajectory[:, 4:]))(timestamps).as_matrix()
    poses[:, :3, 3] = np.stack([np.interp(timestamps, stamps, trajectory[:, i]) for i in (1, 2, 3)], axis=1)
    width, height = metadata["image_size"]
    intrinsic = np.asarray(metadata["intrinsics"])
    cameras = [Camera(pose, intrinsic, width, height) for pose in poses]
    depths = [warped_image(root / depth_files[association[i]], cv2.INTER_NEAREST, cv2.IMREAD_UNCHANGED)
              .astype(np.float32) / 5000. for i in frames]
    return cameras, depths, frames


def view_support(model, native):
    """用光栅化的射线深度分散度估计源测量可归因性和目标的测量需求。"""
    camera = reduced(native, 108)
    _, opacity, meta = model.render(camera, model.opacity[:, None])
    pixels, splats, alpha, cumulative, total = ordered_alpha_samples(meta, camera.width, camera.height)
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
    # 同一排序内读取三个分位数，避免再次光栅化，并保留多表面射线的差异。
    accumulated = -torch.expm1(-cumulative)
    percentiles = z.new_tensor([.25, .5, .75])
    size = camera.width * camera.height
    output = torch.full((size, 3), torch.inf, device=z.device)
    output.scatter_reduce_(0, pixels[:, None].expand(-1, 3),
                           torch.where(accumulated[:, None] >= total[:, None] * percentiles, z[:, None], torch.inf),
                           reduce="amin", include_self=True)
    return totals, (camera, output, opacity.flatten() > .5)


def range_scores(source, target):
    """估计候选测深能够命中目标相机三个可信表面假设的程度。

    Args:
        source (tuple): 候选相机、(H_s * W_s, 3) 深度分位数、有效掩码。
        target (tuple): 查询相机、(H_t * W_t, 3) 深度分位数、有效掩码。

    Returns:
        Tensor: 目标像素的预测测量覆盖，形状 (H_t * W_t,)。
    """
    a, source_depths, source_valid = source
    b, target_depths, target_valid = target
    flat = torch.arange(len(target_depths), device=target_depths.device)
    bx, by = b.intrinsic[0, 2], b.intrinsic[1, 2]
    bfx, bfy = b.intrinsic.diagonal()[:2]
    x = (flat % b.width - bx) / bfx
    y = (flat // b.width - by) / bfy
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    depth = torch.where(target_valid[:, None], target_depths, 0.)
    transform = np.linalg.inv(a.c2w) @ b.c2w
    transform = torch.as_tensor(transform, device=depth.device, dtype=depth.dtype)
    xyz = (rays[:, None, :] * depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    z = xyz[..., 2]
    cx, cy = a.intrinsic[0, 2], a.intrinsic[1, 2]
    fx, fy = a.intrinsic.diagonal()[:2]
    u = (xyz[..., 0] / z.clamp_min(.1) * fx + cx).round().long()
    v = (xyz[..., 1] / z.clamp_min(.1) * fy + cy).round().long()
    hit = (z > .1) & (u >= 0) & (u < a.width) & (v >= 0) & (v < a.height)
    index = v.clamp(0, a.height - 1) * a.width + u.clamp(0, a.width - 1)
    center = source_depths[index, 1]
    width = (source_depths[index, 2] - source_depths[index, 0]).clamp(0., .35)
    confidence = torch.exp(-((center - z).abs() + .35 * width) / (.08 + .04 * z.clamp_min(.1)))
    confidence = torch.where(hit & source_valid[index] & target_valid[:, None], confidence, 0.)
    return confidence @ confidence.new_tensor([.25, .5, .25])


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


def reject_free_space_conflicts(cloud, sources, source_index):
    """检查该帧点是否被其他实测深度表明位于自由空间。"""
    support = torch.zeros(len(cloud), dtype=torch.int32, device=cloud.device)
    conflict = torch.zeros_like(support)
    for i, (camera, band) in enumerate(sources):
        if i == source_index:
            continue
        pose = torch.as_tensor(np.linalg.inv(camera.c2w), device=cloud.device, dtype=cloud.dtype)
        local = cloud @ pose[:3, :3].T + pose[:3, 3]
        z = local[:, 2]
        fx, fy = camera.intrinsic.diagonal()[:2]
        cx, cy = camera.intrinsic[:2, 2]
        u = (local[:, 0] / z.clamp_min(.1) * fx + cx).round().long()
        v = (local[:, 1] / z.clamp_min(.1) * fy + cy).round().long()
        inside = (z > .1) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        index = v.clamp(0, camera.height - 1) * camera.width + u.clamp(0, camera.width - 1)
        lo, hi, valid = band[index].unbind(-1)
        available = inside & valid.bool()
        support += available & (z >= lo) & (z <= hi)
        conflict += available & (z < lo)
    # 仅看到更前面的东西时，该测量仍可能在另一张图中被遮挡。
    return (support > 0) | (conflict == 0)


def view_directions(model, cameras):
    """返回各相机对同一高斯的世界坐标观测方向，形状 (V, N, 3)。"""
    centers = np.stack([c.c2w[:3, 3] for c in cameras])
    centers = torch.as_tensor(centers, device=model.xyz.device, dtype=model.xyz.dtype)
    return torch.nn.functional.normalize(model.xyz[None, :, :] - centers[:, None, :], dim=-1)


def angular_novelty(xyz, cameras, evidence, directions, total):
    """把 COVER 的角度新颖度适配到冻结地图和零张预采测深的情形。"""
    # 首张测深选择光栅覆盖最大的相机，随后按论文的角度相似度选新方向。
    picks = [int(evidence.sum(dim=1).argmax())]
    visible = []
    for c in cameras:
        pose = torch.as_tensor(np.linalg.inv(c.c2w), device=directions.device, dtype=directions.dtype)
        local = xyz @ pose[:3, :3].T + pose[:3, 3]
        z = local[:, 2]
        fx, fy = c.intrinsic.diagonal()[:2]
        cx, cy = c.intrinsic[0, 2], c.intrinsic[1, 2]
        u = local[:, 0] / z.clamp_min(.1) * fx + cx
        v = local[:, 1] / z.clamp_min(.1) * fy + cy
        visible.append((z > .1) & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height))
    visible = torch.stack(visible)
    similar = torch.zeros_like(evidence)
    for _ in range(1, total):
        last = picks[-1]
        part = ((directions * directions[last]).sum(-1).clamp(-1, 1) + 1) / 2
        similar = torch.maximum(similar, part * visible[last])
        score = (evidence * (1 - similar)).sum(dim=1) / evidence.sum(dim=1).clamp_min(1e-7)
        score[picks] = -torch.inf
        picks.append(int(score.argmax()))
    return tuple(sorted(picks))


def target_angular_cover(source_weight, target_weight, source, target, total):
    """最大化目标视角实际可见表面的测深支持，并惩罚大角度转移。"""
    agreement = torch.einsum("vnc,tnc->vtn", source, target).clamp(-1, 1)
    reliability = torch.exp((agreement - 1) / .125)
    metric_support = source_weight[:, None, :] * reliability
    return facility(metric_support.flatten(1), target_weight.flatten(), total)


def evaluate(model, camera, depth, points, checked_points, solution):
    """与传感器重投影结合后，记录独立相机的原尺寸深度误差。"""
    coefficient, intercept, scalar, _ = solution
    render, opacity, _ = model.render(camera, scalar[:, None], with_depth=True)
    weight = opacity.flatten()
    plain = render[..., 1].flatten()
    residual = render[..., 0].flatten() / weight.clamp_min(.1)
    prior = coefficient * plain + intercept
    direct = sensor_reprojection(camera, points)
    filled, distance = local_sensor_fill(camera, direct, prior)
    checked = sensor_reprojection(camera, checked_points)
    checked_fill, _ = local_sensor_fill(camera, checked, prior)
    band = depth_targets(depth, (camera.width, camera.height), model.xyz.device)
    valid = band[:, 2].bool() & (weight > .1)
    truth = band[:, :2].mean(-1)
    values = {"original": plain, "filled_plus_affine": torch.where(filled.isfinite(), filled, prior),
              "filled_plus_adapter": torch.where(filled.isfinite(), filled, prior + residual),
              "consensus_plus_adapter": torch.where(checked_fill.isfinite(), checked_fill, prior + residual)}
    return {"valid": int(valid.sum()), "filled": int((valid & filled.isfinite()).sum()),
            "far": int((valid & (distance > 8)).sum()),
            "consensus_filled": int((valid & checked_fill.isfinite()).sum()),
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
    p.add_argument("--split", choices=("interleaved", "blocked"), default="interleaved")
    p.add_argument("--fresh-tum-fold", choices=(0, 1), type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    root, checkpoint, loader = scene_paths(args.data, args.checkpoints, args.scene)
    model = FrozenGaussians.load(checkpoint)
    if args.fresh_tum_fold is not None:
        if args.scene != "tum":
            p.error("--fresh-tum-fold requires --scene tum")
        camera, depths, frames = fresh_tum_depth(root, args.fresh_tum_fold)
    else:
        camera, depths, frames = loader(root)
    if args.split == "blocked":
        blocks = np.array_split(np.arange(len(camera)), 2)
        pool, query = blocks[args.phase].tolist(), blocks[1 - args.phase].tolist()
    else:
        pool = list(range(args.phase, len(camera), 2))
        query = list(range(1 - args.phase, len(camera), 2))
    query = [query[i] for i in np.rint(np.linspace(0, len(query) - 1, 5)).astype(int)]
    source_views = [view_support(model, camera[i]) for i in pool]
    target_views = [view_support(model, camera[i]) for i in query]
    candidates = torch.stack([stats for stats, _ in source_views])
    targets = torch.stack([stats for stats, _ in target_views]).mean(0)
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
    by_ray = torch.stack([torch.cat([range_scores(a, b) for _, b in target_views])
                          for _, a in source_views])
    picks["metric_surface_cover"] = facility(by_ray, torch.ones_like(by_ray[0]), total)
    source_cameras = [camera[i] for i in pool]
    source_dirs = view_directions(model, source_cameras)
    target_dirs = view_directions(model, [camera[i] for i in query])
    picks["cover_angular_adapted"] = angular_novelty(model.xyz, source_cameras, candidates[:, :, 0], source_dirs, total)
    picks["target_angular_cover"] = target_angular_cover(-torch.expm1(-scale * candidates[:, :, 0]),
                                                        torch.stack([s[:, 0] for s, _ in target_views]),
                                                        source_dirs, target_dirs, total)
    print("Camera selection complete without sensor evidence", flush=True)
    cached = {}
    result = {"scene": args.scene, "split": args.split, "fresh_tum_fold": args.fresh_tum_fold,
              "pool_frames": [int(frames[i]) for i in pool],
              "query_frames": [int(frames[i]) for i in query], "methods": {}}
    for method, subset in picks.items():
        if subset not in cached:
            indices = [pool[i] for i in subset]
            sessions = [calibration_view(model, camera[i], depths[i]) for i in indices]
            solution = solve(model, sessions)
            clouds = [sensor_points(camera[i], depths[i], model.xyz.device) for i in indices]
            sources = [(camera[i], depth_targets(depths[i], (camera[i].width, camera[i].height), model.xyz.device))
                       for i in indices]
            points = torch.cat(clouds)
            checked = torch.cat([cloud[reject_free_space_conflicts(cloud, sources, j)]
                                 for j, cloud in enumerate(clouds)])
            cached[subset] = {"frames": [int(frames[i]) for i in indices],
                              "sensor_samples": len(points), "retained_samples": len(checked),
                              "metrics": [evaluate(model, camera[i], depths[i], points, checked, solution)
                                          for i in query]}
        result["methods"][method] = cached[subset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    print("Matched sensor budget evaluated on separate cameras", flush=True)


if __name__ == "__main__":
    main()
