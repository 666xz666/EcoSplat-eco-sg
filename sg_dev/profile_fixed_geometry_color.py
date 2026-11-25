"""固定场景几何，测量 SH4、SG-3 和预存 RGB 的单帧渲染开销。"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

COLOR_DIMS = {"SH4": 75, "SG-3": 24, "RGB/DC": 3}
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
    for tensor in (scene["means"], scene["quats"], scene["scales"], scene["opacities"]):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite scene geometry: {path}")
    if not bool(((0 <= scene["opacities"]) & (scene["opacities"] <= 1)).all()):
        raise ValueError(f"unexpected opacity convention: {path}")
    return scene, gpu(columns("f_dc_0", "f_dc_1", "f_dc_2")), extrinsics


def color_parameters(c0: Tensor) -> dict:
    """为相同基础颜色构造三档 FP32 张量，方向项用固定随机种子产生。

    Args:
        c0 (Tensor, "gaussian 3"): PLY 的原始 SH DC 系数。

    Returns:
        dict: SH (G, 25, 3)，SG (G, 3)、(G, 3, 3)、(G, 3, 1)、(G, 3, 3)，RGB (G, 3)。
    """
    count = len(c0)
    generator = torch.Generator(device=c0.device).manual_seed(23)
    rand = functools.partial(torch.randn, device=c0.device, generator=generator)
    sh = torch.cat((c0[:, None], rand(count, 24, 3) * 0.025), dim=1)
    directions = torch.nn.functional.normalize(rand(count, 3, 3), dim=-1)
    sharpness = rand(count, 3, 1).abs() + 0.5
    amplitudes = rand(count, 3, 3) * 0.05
    return {"SH4": (sh,), "SG-3": (c0, directions, sharpness, amplitudes),
            "RGB/DC": (c0 * C0 + 0.5,)}


def measure_scene(path: Path, size: int, repeats: int) -> dict:
    """交错测量同个场景上的三条渲染路径并核验可见集和覆盖一致。

    Args:
        path (Path): 要复用的同场景几何。
        size (int): 渲染分辨率。
        repeats (int): 每条路径的整体和分段重复数。

    Returns:
        dict: 可见高斯、tile 相交数、逐次耗时及额外显存峰值。
    """
    from gsplat import rendering

    scene, c0, extrinsics = load_geometry(path, size)
    color = color_parameters(c0)
    count = len(c0)
    outputs = {}
    for name, tensors in color.items():
        color_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        assert color_bytes == 4 * COLOR_DIMS[name] * count
        outputs[name] = {"wall_ms": [], "gpu_ms": [], "events": [], "color_bytes": color_bytes}

    def render(name, invoke):
        if name == "SG-3":
            args = (scene["means"][None], *(x[None] for x in color[name]), extrinsics)
            colors = invoke("color", compute_sg_colors, *args).squeeze(0)
        else:
            colors = color[name][0]
        return rendering.rasterization(
            **scene, colors=colors, sh_degree=4 if name == "SH4" else None,
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
            raise ValueError(f"non-finite output: {path} {name}")
        radii = meta["radii"]
        if radii.ndim == 3:
            radii = radii.amax(-1)
        coverage = (int((radii > 0).sum()), int(meta["isect_ids"].numel()))
        if coverage[0] == 0 or coverage[1] == 0 or (reference is not None and coverage != reference):
            raise ValueError(f"invalid or mismatched coverage: {path} {name} {coverage} {reference}")
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
                if not {"project", "tile_intersect_sort", "composite"} <= values.keys():
                    raise RuntimeError(f"missing rasterization stages: {name}")
                outputs[name]["events"].append(values)

    # 参数输入都提前就位。额外峰值单独测量，避免把其他变体的驻留张量算进差值。
    for name in names:
        torch.cuda.synchronize()
        initial = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        render(name, direct)
        torch.cuda.synchronize()
        outputs[name]["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - initial
    return {"geometry": path.parent.name, "size": size, "count": count,
            "visible": reference[0], "tile_intersections": reference[1], "methods": outputs}


def main() -> None:
    """以原先导出的 RE10K 场景为几何输入，逐样本保存计时记录。"""
    from gsplat import __version__ as gsplat_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", default=12, type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    paths = sorted(args.ply_root.glob("*/scene.ply"))
    if not paths:
        raise ValueError(f"no scene.ply files in {args.ply_root}")
    torch.set_num_threads(4)
    worktree = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip()
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "gsplat": gsplat_version, "repeats": args.repeats, "script_commit": revision, "cases": []}
    with torch.inference_mode():
        for path in paths:
            print("CASE_START", path.parent.name, flush=True)
            size = int(path.parent.name.split("_", 1)[0])
            result["cases"].append(measure_scene(path, size, args.repeats))
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
            print("CASE_DONE", path.parent.name, flush=True)
    print("COLOR_PROFILE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
