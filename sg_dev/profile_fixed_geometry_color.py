"""固定视点和场景几何，测量高斯数与颜色容量对单帧渲染的影响。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack
import functools
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg_wrapper"))
from sg_wrapper.sg_colors import C0, compute_sg_colors

GAUSSIAN_FIELDS = ("means", "quats", "scales", "opacities")
DEFAULT_LOBES = (3, 0)
KERNELS = {
    "fully_fused_projection": "project",
    "spherical_harmonics": "color",
    "isect_tiles": "tile_intersect_sort",
    "isect_offset_encode": "tile_index",
    "rasterize_to_pixels": "composite",
}


def load_geometry(path: Path, size: int) -> tuple[dict, Tensor, Tensor]:
    """从官方 PLY 读入共享几何，并在参考相机方向构造固定测试视点。

    Args:
        path (Path): 保存原始 DC、正常不透明度、对数尺度及 WXYZ 旋转的场景文件。
        size (int): 单帧的高和宽。

    Returns:
        tuple: gsplat 场景参数与相机内参，c0 (G, 3) 和相机到世界矩阵 (1, 4, 4)。
    """
    vertices = PlyData.read(path)["vertex"].data

    def columns(*names):
        return np.stack([vertices[name] for name in names], axis=-1)

    # 已导出的 PLY 经过中心化和归一化。只去掉固定的显示旋转，保留相机轴方向。
    axis_transform = Rotation.from_rotvec([0, 0, -45], degrees=True).as_matrix() @ np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    )
    xyz = columns("x", "y", "z").astype(np.float64) @ axis_transform
    xyzw = columns("rot_1", "rot_2", "rot_3", "rot_0")
    rotations = (Rotation.from_matrix(axis_transform.T) * Rotation.from_quat(xyzw)).as_quat()
    depth = max(0.85, 0.3 - float(np.quantile(xyz[:, 2], 0.15)))
    extrinsics = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0)
    extrinsics[0, 2, 3] = -depth
    intrinsics = torch.tensor(
        [[[size * 0.85, 0, size * 0.5], [0, size * 0.85, size * 0.5], [0, 0, 1]]],
        device="cuda", dtype=torch.float32,
    )

    def gpu(array):
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device="cuda")

    scene = {
        "means": gpu(xyz), "quats": gpu(rotations[:, [3, 0, 1, 2]]),
        "scales": gpu(np.exp(columns("scale_0", "scale_1", "scale_2"))),
        "opacities": gpu(np.array(vertices["opacity"], copy=True)),
        "viewmats": torch.linalg.inv(extrinsics), "Ks": intrinsics, "height": size, "width": size,
    }
    for field in GAUSSIAN_FIELDS:
        if not torch.isfinite(scene[field]).all():
            raise ValueError(f"non-finite scene geometry: {path}")
    if not bool(((0 <= scene["opacities"]) & (scene["opacities"] <= 1)).all()):
        raise ValueError(f"unexpected opacity convention: {path}")
    return scene, gpu(columns("f_dc_0", "f_dc_1", "f_dc_2")), extrinsics


def color_parameters(c0: Tensor, lobe_counts: tuple[int, ...]) -> dict:
    """用固定随机种子生成方向项，较小瓣数使用同一组参数的前缀。

    Args:
        c0 (Tensor, "gaussian 3"): PLY 的原始 SH DC 系数。
        lobe_counts (tuple[int, ...]): 每颗高斯实际存储的瓣数，0 对应直接传入 RGB。

    Returns:
        dict: 颜色路径名到实际瓣数（SH4 为 None）和对应输入张量的映射。
    """
    count = len(c0)
    generator = torch.Generator(device=c0.device).manual_seed(23)
    rand = functools.partial(torch.randn, device=c0.device, generator=generator)
    sh = torch.cat((c0[:, None], rand(count, 24, 3) * 0.025), dim=1)
    max_lobes = max(lobe_counts)
    directions = torch.nn.functional.normalize(rand(count, max_lobes, 3), dim=-1)
    sharpness = rand(count, max_lobes, 1).abs() + 0.5
    amplitudes = rand(count, max_lobes, 3) * 0.05

    variants = {"SH4": (None, (sh,))}
    for lobes in lobe_counts:
        if lobes == 0:
            variants["RGB/DC"] = (0, (c0 * C0 + 0.5,))
        else:
            params = (directions, sharpness, amplitudes)
            variants[f"SG-{lobes}"] = (lobes, (c0, *(t[:, :lobes].contiguous() for t in params)))
    return variants


def measure_scene(scene: dict, color: dict, extrinsics: Tensor, label: str, repeats: int) -> dict:
    """交错测量同场景同点数的颜色路径，并核验可见集和覆盖一致。

    Args:
        scene (dict): 已选定的 gsplat 几何张量、相机和图像大小。
        color (dict): 各颜色路径的实际瓣数和输入张量。
        extrinsics (Tensor, "1 4 4"): 相机到世界变换。
        label (str): 用于定位失败样本的场景和点数。
        repeats (int): 整体和分段计时各自重复的次数。

    Returns:
        dict: 可见高斯、tile 相交数、逐次耗时及额外显存峰值。
    """
    from gsplat import rendering

    count = len(scene["means"])
    outputs = {}
    for name, (lobes, tensors) in color.items():
        color_dims = 75 if lobes is None else 3 + 7 * lobes
        color_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        assert color_bytes == 4 * color_dims * count
        outputs[name] = {
            "wall_ms": [], "gpu_ms": [], "events": [], "color_bytes": color_bytes,
            "lobes_per_gaussian": lobes, "total_sg_lobes": None if lobes is None else lobes * count,
        }

    def render(name, invoke):
        lobes, tensors = color[name]
        if lobes:
            args = (scene["means"][None], *(x[None] for x in tensors), extrinsics)
            colors = invoke("color", compute_sg_colors, *args).squeeze(0)
        else:
            colors = tensors[0]
        return invoke(
            "raster", rendering.rasterization, **scene, colors=colors, sh_degree=4 if name == "SH4" else None,
            render_mode="RGB+D", packed=False, radius_clip=0.1,
            near_plane=0.01, far_plane=100.0, tile_size=16,
        )

    def direct(_name, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    reference = None
    names = tuple(color)
    for name in names:
        for _ in range(3):
            frame, alpha, meta = render(name, direct)
        if not torch.isfinite(frame).all():
            raise ValueError(f"non-finite output: {label} {name}")
        radii = meta["radii"]
        if radii.ndim == 3:
            radii = radii.amax(-1)
        coverage = (int((radii > 0).sum()), int(meta["isect_ids"].numel()))
        if coverage[0] == 0 or coverage[1] == 0 or (reference is not None and coverage != reference):
            raise ValueError(f"invalid or mismatched coverage: {label} {name} {coverage} {reference}")
        reference = coverage
        del frame, alpha, meta, radii

    for index in range(repeats):
        for name in names[index % len(names):] + names[:index % len(names)]:
            torch.cuda.synchronize()
            wall_start = perf_counter_ns()
            render(name, direct)
            torch.cuda.synchronize()
            outputs[name]["wall_ms"].append((perf_counter_ns() - wall_start) / 1e6)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            render(name, direct)
            stop.record()
            stop.synchronize()
            outputs[name]["gpu_ms"].append(start.elapsed_time(stop))

    events = defaultdict(list)

    def traced(stage, fn, *args, **kwargs):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        stop.record()
        events[stage].append((start, stop))
        return result

    with ExitStack() as active:
        for symbol, stage in KERNELS.items():
            instrumented = functools.partial(traced, stage, getattr(rendering, symbol))
            active.enter_context(patch.object(rendering, symbol, instrumented))
        for index in range(repeats):
            for name in names[index % len(names):] + names[:index % len(names)]:
                events.clear()
                render(name, traced)
                torch.cuda.synchronize()
                values = {
                    stage: sum(begin.elapsed_time(end) for begin, end in spans)
                    for stage, spans in events.items()
                }
                if ("color" in values) != (name != "RGB/DC"):
                    raise RuntimeError(f"color stage was not measured correctly: {name}")
                required = set(KERNELS.values()) - {"color"}
                if not required <= values.keys():
                    raise RuntimeError(f"missing rasterization stages: {name}")

                # 光栅调用首尾与首个投影、末次合成对齐，独立记录输入及输出整理。
                raster_start, raster_stop = events["raster"][0]
                values["raster_input"] = raster_start.elapsed_time(events["project"][0][0])
                values["raster_output"] = events["composite"][-1][1].elapsed_time(raster_stop)
                outputs[name]["events"].append(values)

    # 参数输入都提前就位。额外峰值单独测量，避免把其他变体的驻留张量算进差值。
    for name in names:
        torch.cuda.synchronize()
        initial = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        render(name, direct)
        torch.cuda.synchronize()
        outputs[name]["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - initial
    return {"visible": reference[0], "tile_intersections": reference[1], "methods": outputs}


def selected_scenes(
    path: Path, size: int, counts: list[int] | None, lobe_counts: tuple[int, ...],
) -> Iterator[tuple[int, dict, dict, Tensor]]:
    """从一次读入的场景生成相机固定、几何逐档嵌套的测试输入。

    Args:
        path (Path): 场景 PLY。
        size (int): 所有档位的渲染边长。
        counts (list[int] | None): 待测点数。省略时保留全场景。
        lobe_counts (tuple[int, ...]): 每颗高斯的待测瓣数。

    Returns:
        Iterator: 逐档产出点数、gsplat 几何、颜色参数和相机外参。
    """
    scene, c0, extrinsics = load_geometry(path, size)
    full_count = len(c0)
    if counts and max(counts) > full_count:
        raise ValueError(f"{path} contains {full_count} Gaussians, requested {max(counts)}")

    # 步骤1：整场生成一次颜色。不同点数或瓣数共享已有高斯上的已有系数。
    color = color_parameters(c0, lobe_counts)
    if counts is None:
        yield full_count, scene, color, extrinsics
        return

    # 步骤2：随机排列产生嵌套子集，避免 PLY 的视图顺序影响小档位的空间分布。
    generator = torch.Generator(device="cpu").manual_seed(23)
    order = torch.randperm(full_count, generator=generator).to(c0.device)
    for count in counts:
        indices = order[:count].sort().values
        subset = scene | {field: scene[field].index_select(0, indices) for field in GAUSSIAN_FIELDS}
        subset_color = {
            name: (lobes, tuple(t.index_select(0, indices) for t in tensors))
            for name, (lobes, tensors) in color.items()
        }
        yield count, subset, subset_color, extrinsics


def main() -> None:
    """以原先导出的 RE10K 场景为几何输入，逐样本保存计时记录。"""
    from gsplat import __version__ as gsplat_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", default=12, type=int)
    parser.add_argument("--counts", nargs="+", type=int, help="Keep these nested Gaussian counts.")
    parser.add_argument("--lobes", nargs="+", choices=range(4), type=int, default=DEFAULT_LOBES)
    parser.add_argument("--render-size", type=int, help="Override the source-specific render size.")
    parser.add_argument("--scene-prefix", default="", help="Only use PLY directories with this prefix.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.render_size is not None and args.render_size < 1:
        parser.error("--render-size must be positive")
    if args.counts is not None and (min(args.counts) < 1 or len(set(args.counts)) != len(args.counts)):
        parser.error("--counts must contain unique positive integers")
    if len(set(args.lobes)) != len(args.lobes):
        parser.error("--lobes must contain unique integers")
    paths = sorted(
        path for path in args.ply_root.glob("*/scene.ply")
        if path.parent.name.startswith(args.scene_prefix)
    )
    if not paths:
        raise ValueError(f"no scene.ply files in {args.ply_root}")
    torch.set_num_threads(4)
    worktree = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    counts = sorted(args.counts) if args.counts is not None else None
    lobes = tuple(args.lobes)
    result = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "gsplat": gsplat_version,
        "repeats": args.repeats, "script_commit": revision, "counts": counts, "lobes": lobes,
        "input_ply_root": str(args.ply_root.resolve()), "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    print("PROFILE_START", "scenes", len(paths), "counts", counts, "lobes", lobes, flush=True)
    with torch.inference_mode():
        for path in paths:
            label = path.parent.name
            size = args.render_size if args.render_size else int(label.split("_", 1)[0])
            for count, scene, color, extrinsics in selected_scenes(path, size, counts, lobes):
                print("CASE_START", label, "gaussians", count, flush=True)
                measurements = measure_scene(scene, color, extrinsics, f"{label}/{count}", args.repeats)
                result["cases"].append({"geometry": label, "size": size, "count": count, **measurements})
                temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                temporary.replace(args.output)
                print("CASE_DONE", label, "gaussians", count, flush=True)
    print("COLOR_PROFILE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
