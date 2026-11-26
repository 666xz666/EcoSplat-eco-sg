"""在 Replica 留出相机上评估 3DGS 的微视差表面对应。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from readout import Camera, FrozenGaussians, opacity_quantiles, readouts


COVERAGES = (0.3, 0.5, 0.7)
PIXEL_THRESHOLDS = (1.0, 3.0)


def project_between_cameras(depth, source, target):
    """把源相机的 Z 深度 (H, W) 投影到目标相机。"""
    y, x = np.indices(depth.shape, dtype=np.float32)
    focal = source.intrinsic.diagonal()[:2]
    center = source.intrinsic[:2, 2]
    rays = np.stack(((x - center[0]) / focal[0], (y - center[1]) / focal[1], np.ones_like(x)), axis=-1)
    relative = np.linalg.inv(target.c2w) @ source.c2w
    points = rays * depth[..., None]
    transformed = points @ relative[:3, :3].T + relative[:3, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        image = transformed[..., :2] / transformed[..., 2:]
    image = image * target.intrinsic.diagonal()[:2] + target.intrinsic[:2, 2]
    return image, transformed[..., 2]


def valid_pixels(depth, other_depth, projected, target_z):
    """评测像素必须在两张留出真值深度中可见。"""
    y, x = np.indices(depth.shape)
    width = depth.shape[1]
    height = depth.shape[0]
    good = (depth > 0.15) & (depth < 9.9) & np.isfinite(target_z) & (target_z > 0.15)
    good &= (x >= 4) & (x < width - 4) & (y >= 4) & (y < height - 4)
    u, v = np.moveaxis(projected, -1, 0)
    good &= (u >= 3) & (u < width - 3) & (v >= 3) & (v < height - 3)
    x_other = np.rint(np.where(good, u, 0)).astype(int)
    y_other = np.rint(np.where(good, v, 0)).astype(int)
    target_depth = other_depth[y_other, x_other]
    good &= (target_depth > 0.15) & (target_depth < 9.9)
    good &= np.abs(target_depth - target_z) <= np.maximum(0.025, 0.025 * target_z)
    return good


def boundary_pixels(depth):
    """在真值深度跳变邻域衡量边界投影，容忍小量深度噪声。"""
    neighborhood = np.ones((5, 5), dtype=np.uint8)
    depth = depth.astype(np.float32)
    front = cv2.erode(np.where(depth > 0.15, depth, np.float32(99.0)), neighborhood)
    back = cv2.dilate(np.where(depth < 9.9, depth, np.float32(0.0)), neighborhood)
    return (front < 9.9) & (back - front > np.maximum(0.1, 0.05 * depth))


def evaluation_pairs(cameras):
    """按已知相机几何系统选择目标，测试深度只在评测中使用。"""
    positions = np.stack([camera.c2w[:3, 3] for camera in cameras])
    directions = np.stack([camera.c2w[:3, 2] for camera in cameras])
    for i, source in enumerate(cameras):
        lengths = np.linalg.norm(positions - source.c2w[:3, 3], axis=1)
        choices = np.flatnonzero((lengths >= 0.04) & (lengths <= 0.75) & (directions @ directions[i] >= 0.75))
        selected = choices[np.argsort(np.abs(lengths[choices] - 0.22))[:2]]
        yield from ((i, int(j), float(lengths[j])) for j in selected)


def scores_for_pair(estimate, source, target, truth, projected, valid):
    """在全体及边缘像素上单独匹配接受率，避免靠丢弃边缘提高指标。"""
    depth, score = estimate
    uv, z = project_between_cameras(depth, source, target)
    pixel_error = np.linalg.norm(uv - projected, axis=-1)
    # 离开目标图或落到相机背后的预测在像素对应判定中算错误。
    inside = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < target.width)
    inside &= (uv[..., 1] >= 0) & (uv[..., 1] < target.height) & np.isfinite(pixel_error)
    pixel_error = np.where(inside, pixel_error, np.hypot(target.width, target.height))
    relative_depth_error = np.abs(depth - truth) / np.maximum(truth, 1e-8)
    boundary = boundary_pixels(truth)
    rows = []
    for label, mask in (("all", valid), ("boundary", valid & boundary), ("interior", valid & ~boundary)):
        population = np.flatnonzero(mask)
        if len(population) < 24:
            continue
        flat_score = score.ravel()
        usable = population[np.isfinite(flat_score[population]) & np.isfinite(depth.ravel()[population])]
        order = usable[np.argsort(-flat_score[usable], kind="stable")]
        for coverage in COVERAGES:
            count = max(1, round(coverage * len(population)))
            row = {"stratum": label, "coverage": coverage, "available": len(order), "population": len(population)}
            if len(order) >= count:
                keep = order[:count]
                error = pixel_error.ravel()[keep]
                row.update({
                    "selected": count,
                    "mean_epe": float(np.mean(error)),
                    "median_epe": float(np.median(error)),
                    "bad_1px": float(np.mean(error > PIXEL_THRESHOLDS[0])),
                    "bad_3px": float(np.mean(error > PIXEL_THRESHOLDS[1])),
                    "abs_rel": float(np.mean(relative_depth_error.ravel()[keep])),
                })
            rows.append(row)
    return rows


def load_replica(prepared):
    """加载训练时使用的相机和单独存储的留出深度。"""
    meta = json.loads((prepared / "replica-metadata.json").read_text())
    width, height = meta["image_size"]
    intrinsic = np.asarray(meta["intrinsics"], dtype=np.float64)
    source = Path(meta["source"]) / "results"
    depth_files = {int(path.stem[5:]): path for path in source.glob("depth*.png")}
    heldout_set = set(meta["evaluation_indices"])
    assert set(depth_files) == heldout_set
    cameras, depths, names = [], [], []
    for frame, pose in zip(meta["indices"], meta["poses_c2w"]):
        if frame in heldout_set:
            camera = Camera(np.asarray(pose, dtype=np.float64), intrinsic, width, height)
            depth = cv2.imread(str(depth_files[frame]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 6553.5
            depths.append(cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST_EXACT))
            cameras.append(camera)
            names.append(frame)
    return cameras, depths, names


def run_replica(model, prepared, output):
    """输出完整逐相机对的评测记录，不读取留出 RGB 参与微视差。"""
    cameras, truth, names = load_replica(prepared)
    pairs = list(evaluation_pairs(cameras))
    if not pairs:
        raise ValueError("No stereo pair found in the held-out poses")
    timings = {}
    prepared_outputs = {}
    with output.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps({"type": "settings", "frames": names, "pairs": len(pairs), "coverages": COVERAGES}) + "\n")
        for i, j, baseline in pairs:
            if i not in prepared_outputs:
                start = time.perf_counter()
                prediction, _, displacement, timing = readouts(model, cameras[i])
                prepared_outputs[i] = prediction
                timings[i] = {**timing, "end_to_end": time.perf_counter() - start}
                print(f"View {names[i]} of {len(names)} held-out frames: h={displacement:.6g} m, {timings[i]['end_to_end']:.1f}s", flush=True)
            projected, z = project_between_cameras(truth[i], cameras[i], cameras[j])
            valid = valid_pixels(truth[i], truth[j], projected, z)
            pair = {"type": "pair", "source": names[i], "target": names[j], "baseline": baseline, "gt_valid": int(valid.sum())}
            for name, estimate in prepared_outputs[i].items():
                pair[name] = scores_for_pair(estimate, cameras[i], cameras[j], truth[i], projected, valid)
            sink.write(json.dumps(pair, allow_nan=False) + "\n")
            sink.flush()
            print(f"Pair {names[i]}->{names[j]}, verified pixels={valid.sum()}", flush=True)
        sink.write(json.dumps({"type": "complete", "render_and_readout_seconds_per_source": timings}) + "\n")


@torch.inference_mode()
def synthetic_smoke():
    """用已知 Z=2 的单高斯验证渲染投影、平移符号和深度分位数。"""
    intrinsic = np.array([[80.0, 0, 47.5], [0, 80.0, 35.5], [0, 0, 1.0]])
    camera = Camera(np.eye(4), intrinsic, 96, 72)
    gauss = FrozenGaussians((
        0, torch.tensor([[0.0, 0.0, 2.0]]), torch.ones((1, 1, 3)), torch.zeros((1, 0, 3)),
        torch.full((1, 3), np.log(0.18)), torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([[2.2]]),
    ))
    colors = gauss.colors(camera)
    center, alpha, meta = gauss.render(camera, colors, with_depth=True)
    other = gauss.render(camera.translated(0.025, 0), colors)[0]
    x = torch.arange(camera.width, device="cuda")
    center_col = center[0, :, :, 0].sum(dim=0)
    moved_col = other[0, :, :, 0].sum(dim=0)
    measured = ((x * center_col).sum() / center_col.sum() - (x * moved_col).sum() / moved_col.sum()).item()
    if abs(measured - 1) > 0.05:
        raise AssertionError(f"Known one-pixel shift was {measured}")
    valid = alpha[0, :, :, 0].cpu().numpy() > 0.15
    median_depth = opacity_quantiles(meta, camera.width, camera.height)[1][valid]
    if not np.allclose(median_depth, 2, atol=1e-4):
        raise AssertionError("Optical median depth disagrees with the single Gaussian")
    return {"synthetic_shift_px": measured, "synthetic_expected_px": 1, "synthetic_median_depth": float(np.median(median_depth))}


def run_room_smoke(model, cameras_json, output):
    """在上轮的 room 模型验证微视差输出范围并记录可用像素。"""
    result = synthetic_smoke()
    records = json.loads(cameras_json.read_text())
    room = records[0]
    old_w, old_h = room["width"], room["height"]
    width = min(600, old_w)
    height = round(old_h * width / old_w)
    intrinsic = np.array([[room["fx"] * width / old_w, 0, (width - 1) / 2],
                          [0, room["fy"] * height / old_h, (height - 1) / 2], [0, 0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(room["rotation"])
    pose[:3, 3] = np.asarray(room["position"])
    estimates, center, displacement, timing = readouts(model, Camera(pose, intrinsic, width, height), smoke=True)
    depth, confidence = estimates["micro_fixed"]
    valid = np.isfinite(confidence)
    if valid.mean() < 0.1:
        raise AssertionError(f"Too few valid room pixels: {valid.mean():.1%}")
    result.update(
        room_frame=room["img_name"], resolution=[width, height], translation=displacement,
        room_valid_fraction=float(valid.mean()), room_median_depth=float(np.nanmedian(depth)),
        center_image_is_finite=bool(np.isfinite(center).all()), seconds=timing,
    )
    print(json.dumps(result), flush=True)
    with output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2)


def main():
    """验证 room 或运行 Replica 冻结几何评测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--room-cameras", type=Path)
    inputs.add_argument("--replica", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    model = FrozenGaussians.load(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.room_cameras is not None:
        run_room_smoke(model, args.room_cameras, args.output)
    else:
        run_replica(model, args.replica, args.output)


if __name__ == "__main__":
    main()
