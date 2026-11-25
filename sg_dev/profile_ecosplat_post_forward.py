"""测量 EcoSplat 官方论文版场景高斯返回后的渲染、评估及导出耗时。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib import import_module
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from profile_spfsplat_baseline import (
    PAPER_RELEASE,
    allocator_end,
    allocator_start,
    compile_observed_function,
    construct_input,
    make_encoder,
    milliseconds,
    select_scenes,
    to_device,
    verified_stage2,
)

RENDER = (
    ("assert use_sh", "相机变换与球谐布局", ""),
    ("for i in range(b):", "光栅化参数准备", ""),
    ("image, rendered_alpha, _ = rasterization(", "gsplat 输入准备", ""),
    ("image, rendered_depth = torch.split(", "RGB 深度与 alpha 输出整理", ""),
)
GSPLAT = (
    ("proj_results = fully_fused_projection(", "三维高斯投影", ""),
    ("# Turn colors into", "球谐颜色与投影结果整理", ""),
    ("# Identify intersecting tiles", "Tile 相交与排序", ""),
    ("# print(\"rank\", world_rank, \"Before rasterize_to_pixels\")", "像素 alpha 合成", ""),
)
ALIGN = (
    ("pose_optimizer.zero_grad()", "Adam 梯度清零", ""),
    ("output = self.decoder.forward(", "对齐渲染", ""),
    ("total_loss = 0", "对齐损失准备", ""),
    ("total_loss.backward()", "反向传播", ""),
    ("pose_optimizer.step()", "Adam 参数更新", ""),
)
PLY = (
    ("rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()", "四元数搬运与 SciPy 变换", ""),
    ("dtype_full =", "顶点字段转换与 D2H", ""),
    ("attributes = np.concatenate", "CPU 顶点数组装配", ""),
    ("path.parent.mkdir(exist_ok=True, parents=True)", "构造 PLY 顶点布局", ""),
)


class Timing:
    """记录顺序执行的阶段。GPU 计时使用 event，文件路径使用同步墙钟。"""

    def __init__(self, use_events):
        self.use_events = use_events
        self.active = None
        self.rows = []

    def stamp(self):
        if self.use_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        torch.cuda.synchronize()
        return perf_counter_ns()

    def mark(self, next_stage, **unused):
        stamp = self.stamp()
        if self.active is not None:
            name, start = self.active
            self.rows.append((name, start, stamp))
        self.active = (next_stage, stamp) if next_stage is not None else None

    def finish(self):
        self.mark(None)
        torch.cuda.synchronize()
        return [
            dict(stage=name, ms=start.elapsed_time(end) if self.use_events else (end - start) / 1e6)
            for name, start, end in self.rows
        ]


def wall(function):
    torch.cuda.synchronize()
    start = perf_counter_ns()
    result = function()
    torch.cuda.synchronize()
    return milliseconds(start), result


def load_components(args):
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.misc.benchmarker import Benchmarker

    encoder, step = make_encoder(args)
    with initialize_config_dir(config_dir=str(args.repo / "config"), version_base=None):
        cfg = load_typed_root_config(compose(
            config_name="main", overrides=["+experiment=ecosplat/re10k", "mode=test", "test.align_pose=true"]
        ))
    model = SimpleNamespace(
        encoder=encoder, global_step=step, decoder=get_decoder(cfg.model.decoder).cuda().eval(),
        losses=[loss.cuda().eval() for loss in get_losses(cfg.loss) if loss.name in ("mse", "lpips")],
        test_cfg=cfg.test, benchmarker=Benchmarker(),
    )
    return model


def make_case(chunk, scene, indices, size):
    """用同一场景中的固定新视图构造官方评估输入。"""
    target_index = (indices[0] + indices[1]) // 2
    assert target_index not in indices
    image_data, _ = construct_input(chunk, scene, indices + [target_index], size)
    example = next(item for item in torch.load(chunk, map_location="cpu", weights_only=True)
                   if item["key"] == scene)
    camera_rows = example["cameras"][indices + [target_index]]
    poses = torch.eye(4).repeat(len(camera_rows), 1, 1)
    poses[:, :3] = camera_rows[:, 6:].reshape(-1, 3, 4)
    poses = poses.inverse()
    scale = (poses[0, :3, 3] - poses[len(indices) - 1, :3, 3]).norm()
    poses[:, :3, 3] /= scale
    poses = poses[0].inverse() @ poses
    context = to_device({key: value[:, :-1] for key, value in image_data.items()})
    context["extrinsics"] = poses[:-1][None].cuda()
    target = to_device({key: value[:, -1:] for key, value in image_data.items()})
    target.update(extrinsics=poses[-1:][None].cuda(),
                  near=(0.1 / scale).reshape(1, 1).cuda(),
                  far=(100.0 / scale).reshape(1, 1).cuda())
    return context, target, target_index


@contextmanager
def trace_raster(timing):
    adapter = import_module("src.model.decoder.cuda_splatting")
    decoder = import_module("src.model.decoder.decoder_splatting_cuda")
    gsplat = compile_observed_function(adapter.rasterization, GSPLAT, timing.mark)
    observed_render = compile_observed_function(adapter.render_cuda, RENDER, timing.mark, {"rasterization": gsplat})
    with patch.object(decoder, "render_cuda", observed_render):
        yield


def measure_renderer(render, repeats):
    gpu_rows, wall_ms = [], []
    for _ in range(3):
        render()
    for _ in range(repeats):
        took, output = wall(render)
        wall_ms.append(took)
        del output

    last = render()
    for _ in range(repeats):
        timing = Timing(True)
        with trace_raster(timing):
            timing.mark("Decoder 输入组织")
            output = render()
            stages = timing.finish()
        for field in ("color", "depth", "alpha"):
            torch.testing.assert_close(getattr(output, field), getattr(last, field), rtol=0, atol=0)
        gpu_rows.append(stages)
    del last, output

    start = allocator_start()
    output = render()
    peak = allocator_end(start)
    output_bytes = sum(t.numel() * t.element_size() for t in (output.color, output.depth, output.alpha))
    return dict(wall_ms=wall_ms, event_stages=gpu_rows, memory=peak, output_bytes=output_bytes)


def measure_alignment(model, target, gaussians, initial):
    from src.model.model_wrapper import ModelWrapper

    fn = ModelWrapper.test_step_align
    inputs = (model, target, gaussians, target["intrinsics"], initial.clone())
    elapsed, exact = wall(lambda: fn(*inputs))
    timing = Timing(True)
    observed = compile_observed_function(fn, ALIGN, timing.mark)
    from contextlib import ExitStack

    with ExitStack() as patches:
        for loss in model.losses:
            forward = loss.forward

            def measured_loss(*args, original=forward, name=loss.name):
                timing.mark("对齐 " + name.upper())
                return original(*args)

            patches.enter_context(patch.object(loss, "forward", measured_loss))
        timing.mark("目标位姿与优化器初始化")
        output, pose = observed(model, target, gaussians, target["intrinsics"], initial.clone())
        rows = timing.finish()

    torch.testing.assert_close(output.color, exact[0].color, rtol=5e-4, atol=3e-4)
    torch.testing.assert_close(pose, exact[1], rtol=5e-4, atol=3e-4)
    return dict(wall_ms=elapsed, event_stages=rows, iterations=model.test_cfg.pose_align_steps), output


def measure_scores(target, context, predicted_cameras, color, repeats):
    from src.evaluation.metrics import compute_lpips, compute_pose_error_for_batch, compute_psnr, compute_ssim

    views = context["image"].shape[1]
    pairs = (
        ("PSNR", lambda: compute_psnr(target["image"][0], color[0])),
        ("SSIM", lambda: compute_ssim(target["image"][0], color[0])),
        ("LPIPS", lambda: compute_lpips(target["image"][0], color[0])),
        ("位姿误差", lambda: (
            compute_pose_error_for_batch(predicted_cameras[:, views:], target["extrinsics"]),
            compute_pose_error_for_batch(predicted_cameras[:, 1:views], context["extrinsics"][:, 1:]),
        )),
    )
    results = {}
    for name, function in pairs:
        function()
        results[name] = [wall(function)[0] for _ in range(repeats)]
    return results


def measure_png(image, directory, repeats):
    from src.misc.image_io import prep_image, save_image

    path = directory / "target.png"
    save_image(image, path)
    reference = path.read_bytes()
    result = []
    for _ in range(repeats):
        total, _ = wall(lambda: save_image(image, path))
        d2h, rgb = wall(lambda: prep_image(image))
        buffer = BytesIO()
        encode, _ = wall(lambda: Image.fromarray(rgb).save(buffer, format="PNG"))
        raw = buffer.getvalue()
        write, _ = wall(lambda: path.write_bytes(raw))
        assert raw == reference
        result.append(dict(wall_ms=total, prep_d2h_ms=d2h, encode_ms=encode, write_ms=write, bytes=len(raw)))
    return result


def measure_ply(reference_camera, gaussians, directory, repeats):
    from src.model.ply_export import export_ply
    from plyfile import PlyData

    path = directory / "scene.ply"
    fields = (reference_camera, gaussians.means[0], gaussians.scales[0], gaussians.rotations[0],
              gaussians.harmonics[0], gaussians.opacities[0], path)
    export_ply(*fields)
    reference = path.read_bytes()
    rows = []
    for _ in range(repeats):
        total, _ = wall(lambda: export_ply(*fields))
        timing = Timing(False)

        class TimedPlyData:
            def __init__(self, *args):
                self.data = PlyData(*args)

            def write(self, filename):
                timing.mark("PLY 二进制编码")
                buffer = BytesIO()
                self.data.write(buffer)
                timing.mark("PLY 文件写入")
                filename.write_bytes(buffer.getvalue())

        observed = compile_observed_function(export_ply, PLY, timing.mark, {"PlyData": TimedPlyData})
        timing.mark("归一化与坐标变换")
        observed(*fields)
        stages = timing.finish()
        assert path.read_bytes() == reference
        rows.append(dict(wall_ms=total, stages=stages, bytes=len(reference)))
    return rows


def run_case(model, scene, chunk, indices, size, settings, directory):
    from src.evaluation.metrics import get_lpips

    directory.mkdir(parents=True, exist_ok=True)
    context, target, target_index = make_case(chunk, scene, indices, size)
    with torch.no_grad():
        base = model.encoder(context, global_step=model.global_step)
        conditional = lambda: model.encoder(context, global_step=model.global_step, target=target)
        conditional()
        target_ms = []
        for _ in range(settings.repeats):
            elapsed, eval_result = wall(conditional)
            target_ms.append(elapsed)
        views = len(indices)
        all_cameras = eval_result["extrinsics"]["cwt"]
        camera = all_cameras[:, views:]
        gaussians = base["gaussians"]
        eval_gaussians = eval_result["gaussians"]
        render = lambda: model.decoder.forward(
            gaussians, camera, target["intrinsics"], target["near"], target["far"], (size, size)
        )
        print("  RENDER", flush=True)
        render_result = measure_renderer(render, settings.repeats)
        print("  PLY", flush=True)
        ply_result = measure_ply(base["extrinsics"]["c"][0, 0], gaussians, directory, settings.repeats)
        # 评分网络冷加载不计入后续各次评估。
        get_lpips(torch.device("cuda", 0))

    print("  ALIGN", flush=True)
    # 原官方对齐函数内部重新开启梯度。短预热使用同样的 decoder 和 loss。
    align_steps = model.test_cfg.pose_align_steps
    try:
        model.test_cfg.pose_align_steps = 2
        measure_alignment(model, target, eval_gaussians, camera)
    finally:
        model.test_cfg.pose_align_steps = align_steps
    alignment, aligned = measure_alignment(model, target, eval_gaussians, camera)
    print("  SCORES_PNG", flush=True)
    with torch.no_grad():
        scores = measure_scores(target, context, all_cameras, aligned.color, settings.repeats)
        png = measure_png(aligned.color[0, 0], directory, settings.repeats)
    return dict(
        scene=scene, size=size, views=views, context_indices=indices, target_index=target_index,
        scene_gaussians=gaussians.means.shape[1], eval_gaussians=eval_gaussians.means.shape[1],
        render=render_result, eval_encoder_ms=target_ms, alignment=alignment, scores=scores, png=png, ply=ply_result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "dataset", "stage1", "stage2", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--stage2-sha256", required=True)
    parser.add_argument("--scene-limit", type=int, default=3)
    parser.add_argument("--views", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.scene_limit not in range(1, 4) or args.repeats < 1 or not set(args.views) <= {2, 6}:
        parser.error("Use up to three scenes and 2 or 6 views.")
    revision = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    origin = subprocess.check_output(["git", "-C", args.repo, "remote", "get-url", "origin"], text=True).strip()
    if revision != PAPER_RELEASE or origin != "https://github.com/KAIST-VICLab/EcoSplat.git":
        raise RuntimeError(f"Unexpected baseline {origin} {revision}")

    args.paper_release, args.rho = True, 0.4
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    print("VERIFY_STAGE2", flush=True)
    verified_stage2(args.stage2, args.stage2_sha256)
    print("LOAD_BASELINE", revision, flush=True)
    model = load_components(args)
    scenes = select_scenes(args.repo, args.dataset, args.scene_limit, paper_release=True)
    result = dict(
        upstream=origin, commit=revision, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        stage2_sha256=args.stage2_sha256, cases=[],
    )
    for views in args.views:
        size = 256 if views == 2 else 224
        for scene, path, ids in scenes:
            print(f"CASE_START {scene} {size} views={views}", flush=True)
            directory = args.output.parent / "post_artifacts" / f"{size}_{views}_{scene}"
            result["cases"].append(run_case(model, scene, path, ids[:views], size, args, directory))
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temporary.replace(args.output)
            print(f"CASE_DONE {scene} {size} views={views}", flush=True)
    print("POST_PROFILE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()