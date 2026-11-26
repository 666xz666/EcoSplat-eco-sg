"""用 TUM Freiburg 3 的 RGB 和动捕位姿构造带留出的 3DGS 数据。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from prepare_replica import initialize_points, write_point_cloud


SIZE = (320, 240)
KINECT_K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.]])
INTRINSIC = np.array([[267.7, 0, 159.5], [0, 269.6, 119.5], [0, 0, 1.]])
WARP = (INTRINSIC @ np.linalg.inv(KINECT_K))[:2]
FRAME_COUNT = 160


def timed_files(path):
    """读取按时间排序的 TUM RGB 或深度文件索引。"""
    entries = [line.split() for line in path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    entries.sort(key=lambda entry: float(entry[0]))
    return np.array([float(stamp) for stamp, _ in entries]), [file for _, file in entries]


def nearest(stamps, queries):
    """找时间轴上的最近邻索引，输入时间戳以秒计。"""
    after = np.searchsorted(stamps, queries).clip(1, len(stamps) - 1)
    before = after - 1
    return np.where(queries - stamps[before] < stamps[after] - queries, before, after)


def frame_selection(root):
    """选择时间同步的观测，按均匀时间间隔抽样并插值位姿。"""
    rgb_t, rgb = timed_files(root / "rgb.txt")
    depth_t, depth = timed_files(root / "depth.txt")
    ground_truth = np.loadtxt(root / "groundtruth.txt")
    stamps = ground_truth[:, 0]
    depth_ids = nearest(depth_t, rgb_t)
    depth_diff = np.abs(depth_t[depth_ids] - rgb_t)
    right = np.searchsorted(stamps, rgb_t).clip(1, len(stamps) - 1)
    valid = (rgb_t >= stamps[0]) & (rgb_t <= stamps[-1])
    valid &= (stamps[right] - stamps[right - 1] <= .05) & (depth_diff <= .02)
    pool = np.flatnonzero(valid)
    if len(pool) < FRAME_COUNT:
        raise ValueError(f"Only {len(pool)} time-aligned frames")
    frames = pool[np.rint(np.linspace(0, len(pool) - 1, FRAME_COUNT)).astype(int)]
    timestamps = rgb_t[frames]
    poses = np.repeat(np.eye(4)[None], len(frames), axis=0)
    poses[:, :3, :3] = Slerp(stamps, Rotation.from_quat(ground_truth[:, 4:]))(timestamps).as_matrix()
    poses[:, :3, 3] = np.stack([np.interp(timestamps, stamps, ground_truth[:, coordinate])
                               for coordinate in (1, 2, 3)], axis=1)
    measurements = [(rgb[i], depth[depth_ids[i]], float(depth_diff[i])) for i in frames]
    return frames.tolist(), poses, measurements


def warped_image(path, interpolation, flag):
    """把同一 RGB 相机像素变换应用到彩色图和已配准深度。"""
    image = cv2.imread(str(path), flag)
    if image is None or image.shape[:2] != (480, 640):
        raise ValueError(f"Expected 640x480 RGB or depth at {path}")
    return cv2.warpAffine(image, WARP, SIZE, flags=interpolation, borderMode=cv2.BORDER_CONSTANT)


def prepare(source, destination):
    """原子写入 COLMAP 格式。初始化只使用训练 RGB。"""
    indices, poses, measurements = frame_selection(source)
    frames = [warped_image(source / item[0], cv2.INTER_LINEAR, cv2.IMREAD_COLOR) for item in measurements]
    train = [i for i in range(len(indices)) if i % 8]
    eval_positions = range(0, len(indices), 8)
    xyz, colors = initialize_points([frames[i] for i in train], poses[train], intrinsics=INTRINSIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as scratch:
        scene = Path(scratch) / "scene"
        sparse = scene / "sparse" / "0"
        images = scene / "images"
        sparse.mkdir(parents=True)
        images.mkdir()
        intr = (INTRINSIC[0, 0], INTRINSIC[1, 1], INTRINSIC[0, 2], INTRINSIC[1, 2])
        (sparse / "cameras.txt").write_text("1 PINHOLE " + " ".join(map(str, (*SIZE, *intr))) + "\n")
        with (sparse / "images.txt").open("w") as sink:
            for i, (frame, c2w, image) in enumerate(zip(indices, poses, frames), 1):
                name = f"frame{frame:06d}.jpg"
                if not cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    raise OSError(f"Could not write {name}")
                w2c = np.linalg.inv(c2w)
                qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
                values = [qw, qx, qy, qz, *w2c[:3, 3]]
                sink.write(f"{i} " + " ".join(f"{x:.12g}" for x in values) + f" 1 {name}\n\n")
        write_point_cloud(sparse / "points3D.ply", xyz, colors)
        metadata = {
            "source": str(source.resolve()), "intrinsics": INTRINSIC.tolist(), "image_size": SIZE,
            "training_frames": [indices[i] for i in train], "initial_points": len(xyz),
            "evaluation": [{"frame": indices[i], "pose_c2w": poses[i].tolist(), "depth": measurements[i][1],
                            "rgb_depth_delta_s": measurements[i][2]} for i in eval_positions],
        }
        (scene / "tum-metadata.json").write_text(json.dumps(metadata) + "\n")
        scene.rename(destination)
    print(f"{destination.name}: {len(train)} train RGB, {len(eval_positions)} held-out RGB-D, {len(xyz):,} RGB-only points")


def main():
    """从已解压的 Freiburg 3 序列准备训练与留出数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
