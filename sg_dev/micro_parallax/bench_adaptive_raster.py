"""用同一冻结检查点测端到端光栅延迟与画质。"""

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
from time import perf_counter

import cv2
import numpy as np
import torch

from adaptive_raster.renderer import RAW_EPSILON, Renderer, View, adaptive_extension
from evaluate import load_replica
from probe_visibility import load_tum
from readout import Camera, FrozenGaussians


SCENES = ("room0", "office0", "tum")
SSAA = (1., 1.125, 1.25, 1.5, 1.75, 2., 2.5, 3., 4.)
ADAPTIVE = (0., .0625, .125, .25, .5, .75)


def scene_paths(dataset_root, checkpoint_root, name):
    """返回固定场景的当前数据和原短测检查点。"""
    if name == "tum":
        sequence = "tum-fr3-structure-texture-near"
        return (dataset_root / sequence / "prepared", checkpoint_root / sequence / "chkpnt7000.pth", load_tum)
    return (dataset_root / "replica-rgb" / name, checkpoint_root / f"replica-{name}" / "chkpnt7000.pth",
            load_replica)


def heldout_views(prepared, load, count):
    """等间隔选择留出位姿，仅加载每张对应真实 RGB 用于测图。"""
    cameras, _, names = load(prepared)
    indices = np.rint(np.linspace(0, len(cameras) - 1, min(count, len(cameras)))).astype(int)
    for i in indices:
        frame = names[i]
        path = prepared / "images" / f"frame{frame:06d}.jpg"
        observed = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if observed is None:
            raise FileNotFoundError(path)
        yield cameras[i], int(frame), cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)


def variants(renderer):
    """枚举基本合成、选择性与整图高采样，各点均执行真正的 GPU 渲染。"""
    configs = {}
    for scale in SSAA:
        configs[f"ssaa_{scale:g}"] = ("ssaa", scale, partial(renderer.supersample, scale))
    configs["gaussian_blending"] = ("gaussian_blending", None, renderer.gaussian_blending)
    for fraction in ADAPTIVE:
        configs[f"adaptive_{fraction:g}"] = ("adaptive", fraction, partial(renderer.adaptive, fraction))
        configs[f"corrected_{fraction:g}"] = (
            "adaptive_corrected", fraction, partial(renderer.adaptive, fraction, corrected=True),
        )
    for fraction in (.125, .25, .5):
        configs[f"gradient_{fraction:g}"] = (
            "adaptive_gradient", fraction, partial(renderer.adaptive, fraction, selector="gradient"),
        )
    configs["shared_quad"] = ("shared_quad", 1., partial(renderer.adaptive, 1.))
    return configs


def latency(configs, repetitions, seed):
    """方法交错测量；GPU 事件包住每帧全部调用，首次初始化预热。"""
    operations = {name: value[2] for name, value in configs.items()}
    keys = list(operations)
    random = np.random.default_rng(seed)
    for _ in range(3):
        for name in random.permutation(keys):
            operations[name]()
    torch.cuda.synchronize()
    records = {name: [] for name in keys}
    wall_start = perf_counter()
    for _ in range(repetitions):
        for name in random.permutation(keys):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[name]()
            finish.record()
            records[name].append((start, finish))
    torch.cuda.synchronize()
    wall = perf_counter() - wall_start
    seconds = {name: [start.elapsed_time(finish) for start, finish in events] for name, events in records.items()}
    return seconds, wall


def color_metrics(image, reference, real, edge):
    """记录参考积分误差、边缘误差和对真实 RGB 的保真度。"""
    image = image.clamp(0, 1)
    ref_error = (image - reference).square().mean(-1)
    photo = (image - real).square().mean(-1)
    return {
        "integral_mse": float(ref_error.mean()),
        "integral_edge_mse": float(ref_error[edge].mean()),
        "capture_mse": float(photo.mean()),
        "capture_edge_mse": float(photo[edge].mean()),
    }


def reference_edge(reference):
    """用公共参考图梯度的上 20% 定义同一评测边缘集合。"""
    delta = torch.zeros_like(reference)
    delta[:, 1:] += (reference[:, 1:] - reference[:, :-1]).square()
    delta[1:] += (reference[1:] - reference[:-1]).square()
    gradient = delta.sum(-1)
    return gradient >= torch.quantile(gradient, .8)


def measure(model, pose, photo, width, height, order, repetitions, seed):
    """为一张相机输出各实际渲染的延迟分布与画质。"""
    renderer = Renderer(model, View.from_camera(pose, width, height, model.xyz.device))
    real = torch.as_tensor(cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA),
                           dtype=torch.float32, device=model.xyz.device) / 255
    ref = renderer.reference(order).clamp(0, 1)
    edge = reference_edge(ref)
    configs = variants(renderer)
    latencies, wall = latency(configs, repetitions, seed)
    results = {}
    for name, (family, setting, operation) in configs.items():
        times = latencies[name]
        image = operation()
        scores = color_metrics(image, ref, real, edge)
        if not np.isfinite(list(scores.values())).all():
            raise ValueError(f"Non-finite output from {name}")
        results[name] = {
            "family": family, "setting": setting, "median_ms": statistics.median(times),
            "p90_ms": float(np.quantile(times, .9)), "durations_ms": times, **scores,
        }
    return results, wall


def synthetic_check():
    """验证两个后端的主点和四点共享采样，以及非整数面积滤波。"""
    K = np.array([[91., 0, 35.9], [0, 97., 25.7], [0, 0, 1.]], dtype=np.float64)
    camera = Camera(np.eye(4), K, 80, 60)
    xyz = torch.tensor([[.17, -.06, 2.]], dtype=torch.float32)
    model = FrozenGaussians((0, xyz, torch.full((1, 1, 3), .7), torch.zeros((1, 0, 3)),
                              torch.full((1, 3), float(np.log(.12))), torch.tensor([[1., 0., 0., 0.]]),
                              torch.tensor([[2.1]])))
    render = Renderer(model, View.from_camera(camera, 80, 60, model.xyz.device))
    reference = render.reference(2)
    shared = render.adaptive(1.)
    official = render.gaussian_blending()
    reference_error = float((shared - reference).abs().max())
    if reference_error > .001:
        raise ValueError(f"Shared four-node integration disagrees with separate passes: {reference_error:.6g}")
    expected = np.array([K[0, 0] * .17 / 2 + K[0, 2], K[1, 1] * -.06 / 2 + K[1, 2]])
    errors = {}
    yy, xx = torch.meshgrid(torch.arange(60, device="cuda"), torch.arange(80, device="cuda"), indexing="ij")
    for name, input in (("shared", shared), ("official", official)):
        gray = input.mean(-1)
        centroid = np.array([float((gray * xx).sum() / gray.sum()), float((gray * yy).sum() / gray.sum())])
        errors[name] = float(np.max(np.abs(centroid - expected)))
        if errors[name] > .12:
            raise ValueError(f"{name} off-center projection mismatch: {centroid} vs {expected}")
    ext = adaptive_extension()
    image = torch.arange(6., device="cuda").reshape(1, 6, 1).expand(2, 6, 3).contiguous()
    result = ext.downsample_box(image, 4, 1).cpu().numpy()[0, :, 0]
    if not np.allclose(result, [1/3, 5/3, 10/3, 14/3], atol=1e-6):
        raise AssertionError(f"Wrong fractional box filter: {result}")
    return {"four_node_max_error": reference_error, "centroid_max_error_px": errors,
            "fractional_box_filter": result.tolist()}


@torch.inference_mode()
def main():
    """运行 GPU 冒烟或各场景的成组计时，增量写入 JSONL。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--sizes", nargs="+", type=float, default=(.5, 1.))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    tests = synthetic_check()
    print(json.dumps({"synthetic": tests}), flush=True)
    if args.smoke:
        return
    if args.output is None or args.data is None or args.checkpoints is None:
        parser.error("--data, --checkpoints and --output are required for benchmarking")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        sink.write(json.dumps({
            "type": "settings", "scenes": args.scenes, "sizes": args.sizes, "views": args.views,
            "order": args.order, "repetitions": args.repetitions, "epsilon": RAW_EPSILON, "tests": tests,
            "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        }) + "\n")
        for name in args.scenes:
            prepared, checkpoint, load = scene_paths(args.data, args.checkpoints, name)
            model = FrozenGaussians.load(checkpoint)
            model.rotation = torch.nn.functional.normalize(model.rotation, dim=-1)
            cameras = list(heldout_views(prepared, load, args.views))
            for size in args.sizes:
                for camera, frame, photo in cameras:
                    width, height = round(size * camera.width), round(size * camera.height)
                    results, elapsed = measure(model, camera, photo, width, height, args.order, args.repetitions, frame)
                    record = {"type": "view", "scene": name, "scale": size, "frame": frame,
                              "size": [width, height], "measurements": results, "measurement_wall_seconds": elapsed}
                    sink.write(json.dumps(record, allow_nan=False) + "\n")
                    sink.flush()
                    print(f"{name} frame {frame} {width}x{height}: {elapsed:.1f}s, GB "
                          f"{results['gaussian_blending']['median_ms']:.3f} ms, SSAA2 "
                          f"{results['ssaa_2']['median_ms']:.3f} ms, adaptive25 "
                          f"{results['adaptive_0.25']['median_ms']:.3f} ms", flush=True)
            del model
        sink.write(json.dumps({"type": "complete"}) + "\n")


if __name__ == "__main__":
    main()
