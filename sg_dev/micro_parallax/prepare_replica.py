"""用已标定的 Replica RGB 帧准备 3DGS，测试深度留在原始数据目录。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


INTRINSICS = np.array([[300.0, 0.0, 299.5], [0.0, 300.0, 169.5], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (600, 340)


def choose_pairs(transforms):
    """按视点位置和朝向选择有限的相邻相机对。"""
    centers = transforms[:, :3, 3]
    forward = transforms[:, :3, 2]
    pairs = set()
    for i, center in enumerate(centers):
        distance = np.linalg.norm(centers - center, axis=1)
        candidates = np.flatnonzero((distance >= 0.035) & (distance <= 0.6) & (forward @ forward[i] >= 0.8))
        preferred = candidates[np.argsort(np.abs(distance[candidates] - 0.12))[:4]]
        pairs.update(tuple(sorted((i, int(j)))) for j in preferred)
    return sorted(pairs)


def reciprocal_matches(matcher, first, second):
    """用双向最近邻和比率阈值限制错误的特征关联。"""
    forward = matcher.knnMatch(first, second, k=2)
    backward = matcher.knnMatch(second, first, k=2)
    keep_forward = [(m.queryIdx, m.trainIdx) for m, n in forward if m.distance < 0.8 * n.distance]
    keep_backward = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < 0.8 * n.distance}
    return [(i, j) for i, j in keep_forward if keep_backward.get(j) == i]


def triangulate(first, second, projections, poses):
    """三角化 RGB 对应，并剔除背面、低视差及高重投影误差点。

    Args:
        first (ndarray, (M, 2)): 第一张图的像素坐标。
        second (ndarray, (M, 2)): 第二张图的配对像素坐标。
        projections (ndarray, (2, 3, 4)): 带内参的投影矩阵。
        poses (ndarray, (2, 4, 4)): OpenCV 相机到世界矩阵。

    Returns:
        tuple: 三维点 (M, 3) 和有效掩码 (M,)。
    """
    xyzw = cv2.triangulatePoints(*projections, first.T, second.T).T
    valid = np.abs(xyzw[:, 3]) > 1e-8
    safe_w = np.where(valid, xyzw[:, 3], 1)
    xyz = xyzw[:, :3] / safe_w[:, None]
    homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
    for projection, observed in zip(projections, (first, second)):
        projected = homogeneous @ projection.T
        valid &= (projected[:, 2] > 0.15) & (projected[:, 2] < 10)
        error = np.linalg.norm(projected[:, :2] / np.maximum(projected[:, 2:], 1e-8) - observed, axis=1)
        valid &= error < 1.75
    rays = [xyz - pose[:3, 3] for pose in poses]
    numerator = np.einsum("ij,ij->i", *rays)
    denominator = np.linalg.norm(rays[0], axis=1) * np.linalg.norm(rays[1], axis=1)
    valid &= numerator / np.maximum(denominator, 1e-8) < np.cos(np.deg2rad(0.7))
    return xyz, valid


def initialize_points(frames, poses):
    """只用训练 RGB 和已知相机位姿构造初始化点云。"""
    extractor = cv2.SIFT_create(nfeatures=1400)
    matcher = cv2.BFMatcher()
    features = [extractor.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None) for frame in frames]
    projections = INTRINSICS @ np.linalg.inv(poses)[:, :3]
    points = []
    colors = []
    for i, j in choose_pairs(poses):
        key_i, desc_i = features[i]
        key_j, desc_j = features[j]
        if min(len(key_i), len(key_j)) < 2:
            continue
        pairs = reciprocal_matches(matcher, desc_i, desc_j)
        if not pairs:
            continue
        image_i = np.asarray([key_i[m].pt for m, _ in pairs])
        image_j = np.asarray([key_j[n].pt for _, n in pairs])
        xyz, valid = triangulate(image_i, image_j, projections[[i, j]], poses[[i, j]])
        pixels = np.rint(image_i[valid]).astype(int)
        points.append(xyz[valid])
        colors.append(frames[i][pixels[:, 1], pixels[:, 0], ::-1])
    if not points:
        raise RuntimeError("No calibrated RGB matches could be triangulated")
    points = np.concatenate(points)
    colors = np.concatenate(colors)
    _, keep = np.unique(np.floor(points / 0.015).astype(np.int64), axis=0, return_index=True)
    if len(keep) < 500:
        raise RuntimeError(f"Only {len(keep)} usable RGB points")
    return points[keep], colors[keep]


def write_point_cloud(path, xyz, rgb):
    """写入原版和 MCMC 3DGS 都能读的二进制 PLY。"""
    keys = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")
    dtype = np.dtype([(key, "<f4") for key in keys[:6]] + [(key, "u1") for key in keys[6:]])
    rows = np.zeros(len(xyz), dtype=dtype)
    for i in range(3):
        rows[keys[i]] = xyz[:, i]
        rows[keys[i + 6]] = rgb[:, i]
    types = ("float",) * 6 + ("uchar",) * 3
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header.extend(f"property {kind} {name}" for kind, name in zip(types, keys))
    with path.open("wb") as output:
        output.write(("\n".join(header) + "\nend_header\n").encode("ascii"))
        rows.tofile(output)


def prepare(source, destination):
    """构造校准的 COLMAP 输入和留出帧名单，成功后再发布目录。"""
    selection = json.loads((source / "selection.json").read_text())
    indices = selection["rgb_frame_ids"]
    heldout = set(selection["evaluation_frame_ids"])
    if indices[::8] != selection["evaluation_frame_ids"]:
        raise ValueError("Replica selection and the MCMC --eval split disagree")
    trajectory = np.loadtxt(source / "traj.txt").reshape(-1, 4, 4)
    poses = trajectory[indices]
    paths = {int(path.stem[5:]): path for path in (source / "results").glob("frame*.jpg")}
    if sorted(paths) != indices:
        raise ValueError("RGB files and selected frame list disagree")
    frames = [cv2.resize(cv2.imread(str(paths[i])), IMAGE_SIZE, interpolation=cv2.INTER_AREA) for i in indices]
    train = [j for j, i in enumerate(indices) if i not in heldout]
    xyz, rgb = initialize_points([frames[j] for j in train], poses[train])
    with TemporaryDirectory(dir=destination.parent) as temporary:
        prepared = Path(temporary) / "scene"
        sparse = prepared / "sparse" / "0"
        images = prepared / "images"
        sparse.mkdir(parents=True)
        images.mkdir()
        (sparse / "cameras.txt").write_text("1 PINHOLE 600 340 300 300 299.5 169.5\n")
        with (sparse / "images.txt").open("w") as camera_file:
            for position, (index, pose, image) in enumerate(zip(indices, poses, frames), 1):
                name = paths[index].name
                if not cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    raise OSError(f"Could not write {name}")
                world_to_camera = np.linalg.inv(pose)
                qx, qy, qz, qw = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
                values = [qw, qx, qy, qz, *world_to_camera[:3, 3]]
                pose_text = " ".join(f"{value:.12g}" for value in values)
                camera_file.write(f"{position} {pose_text} 1 {name}\n\n")
        write_point_cloud(sparse / "points3D.ply", xyz, rgb)
        metadata = {
            "source": str(source.resolve()), "image_size": IMAGE_SIZE,
            "intrinsics": INTRINSICS.tolist(), "indices": indices,
            "evaluation_indices": indices[::8], "poses_c2w": poses.tolist(),
            "training_image_count": len(train), "initial_points": len(xyz),
        }
        (prepared / "replica-metadata.json").write_text(json.dumps(metadata) + "\n")
        prepared.rename(destination)
    print(f"{destination.name}: {len(train)} train RGB, {len(heldout)} held-out RGB-D, {len(xyz):,} RGB-only points")


def main():
    """从命令行准备一个或两个已下载场景。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("scenes", nargs="+", choices=("room0", "office0"))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    args.destination.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        prepare(args.source / scene, args.destination / scene)


if __name__ == "__main__":
    main()
