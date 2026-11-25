"""在固定的官方 EcoSplat 提交上测量源图像到场景高斯的开销。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from hashlib import sha256
from inspect import getsource
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from time import perf_counter_ns
from types import MethodType

import torch

SPFSPLAT_RELEASE = "544eedc74c445c49d94a0016f63b2433a74107b3"
PAPER_RELEASE = "f3926dcaefea290c5fe23bddd884af76251e0aeb"
BACKBONE_BOUNDARIES = (
    ("dec_feat = self._decoder(", "跨视图融合", "feat=feat, pos=pos"),
)
SPFSPLAT_BOUNDARIES = (
    ("igf_active = self.igf is not None", "预算条件编码", "decoder_features=dec_feat, shape=shape"),
    ("with torch.amp.autocast('cuda', enabled=False):", "三维中心预测", "rate_feat=rate_feat"),
    ("# Gaussian parameter head (context only)", "密集高斯属性预测", "points=all_mean_res"),
    ("# Pose head", "相机预测与深度投影", "original=all_other_params, conditional=merged_other_params"),
    ("gaussians = self.gaussian_adapter.forward(", "稠密高斯参数化", "points=pts_all, raw=gaussian_parameters, density=densities, depth=depths_per_view, cameras=context_extrinsics"),
    ("# IGF: build merged Gaussians", "基元选择", "full_gaussians=gaussians"),
    ("merged_gaussians = self.gaussian_adapter.forward(", "紧凑高斯参数化", "candidates=merged_raw, opacities=merged_opacities_per_pixel, indices=top_idx, points=pts_topk, params=params_topk, density=densities_topk"),
    ("# Dump visualizations if needed.", "场景组装", "compact_gaussians=merged_gaussians"),
)


PAPER_BOUNDARIES = (
    ("context_input = {", "图像与内参编码", "scores=high_freq_scores, weights=high_freq_score_tensor"),
    ("with torch.amp.autocast('cuda', enabled=False):", "预算条件编码", "decoder_features=dec_feat, shape=shape"),
    ("# Pts3d head (context only)", "三维中心预测", "rate_features=rate_feat, rates=primitive_ratio_per_view"),
    ("# Gaussian parameter head (context only)", "密集高斯属性预测", "points=all_mean_res"),
    ("# Pose head", "相机预测与深度投影", "original=all_other_params, conditional=all_other_merge_params"),
    ("# pseudo merging", "伪合并与重投影",
     "points=pts_all, depth=depths_per_view, cameras=context_extrinsics, original=gaussian_parameters, conditional=merge_gaussian_parameters"),
    ("learn_merged_pts_all =  pts_all", "基元选择",
     "normals=normal_map, patch_gradient=patch_grad_map, mask=full_merge_mask, metric=agg_features, points=merged_pts_all, pixel_coords=pixel_pts, projection=projection"),
    ("gaussians = self.gaussian_adapter.forward(pts, opa, gs_param)", "紧凑高斯参数化",
     "opacities=learn_merged_opacity, last_view_indices=top_opa_id, points=pts, params=gs_param, opacity=opa"),
    ("# Convert the features and depths into Gaussians.", "稠密高斯参数化", "compact_gaussians=gaussians"),
    ("# Dump visualizations if needed.", "场景组装", "full_gaussians=ori_gaussians"),
)
PAPER_BACKBONE_BOUNDARIES = (
    ("dec_feat = self._decoder(", "跨视图融合", "feat=feat, pos=pos"),
)
FIXED_PAPER_SCENES = {
    "000c3ab189999a83": (25, 161, 79, 107, 52, 134),
    "000db54a47bd43fe": (151, 206, 173, 184, 162, 195),
    "0017ce4c6a39d122": (39, 89, 59, 69, 49, 79),
}


def milliseconds(start):
    return (perf_counter_ns() - start) / 1e6


def tensor_snapshot(value):
    """遍历指定检查点，统计逻辑张量字节与去重 storage 字节。

    Args:
        value (Any): 检查点处的张量或张量容器。

    Returns:
        dict: 张量元数据和该检查点的逻辑与物理负载。
    """
    rows = []
    storages = {}

    def walk(obj, name):
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            byte_count = obj.numel() * obj.element_size()
            storages[(str(obj.device), storage.data_ptr())] = storage.nbytes()
            rows.append(
                dict(name=name, shape=list(obj.shape), dtype=str(obj.dtype),
                     device=str(obj.device), stride=list(obj.stride()), bytes=byte_count)
            )
        elif isinstance(obj, Mapping):
            for key, item in obj.items():
                walk(item, f"{name}.{key}" if name else str(key))
        elif isinstance(obj, (tuple, list)):
            for index, item in enumerate(obj):
                walk(item, f"{name}[{index}]")
        elif is_dataclass(obj):
            for field in fields(obj):
                walk(getattr(obj, field.name), f"{name}.{field.name}")

    walk(value, "")
    return dict(
        logical_bytes=sum(row["bytes"] for row in rows),
        unique_storage_bytes=sum(storages.values()),
        tensors=rows,
    )


def allocator_start():
    torch.cuda.synchronize()
    start = dict(start_alloc=torch.cuda.memory_allocated(), start_reserved=torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    return start


def allocator_end(start):
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return dict(
        **start, peak_alloc=peak, peak_delta=peak - start["start_alloc"],
        end_alloc=torch.cuda.memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        end_reserved=torch.cuda.memory_reserved(),
    )


class StageProbe:
    """在一次前向中按互斥区间采集 CUDA events 或同步耗时与显存。

    Args:
        mode (str): events 记录 CUDA 区间，memory 记录分段显存和输出元数据。
    """

    def __init__(self, mode):
        self.mode = mode
        self.rows = []
        self.active = None
        self.details = []

    def observe(self, name, **tensors):
        """记录区间内部的临时张量元数据。"""
        self.details.append(dict(name=name, **tensor_snapshot(tensors)))

    def mark(self, next_stage, **outputs):
        """结束当前阶段，按需开始下一个阶段。

        Args:
            next_stage (str | None): 后续阶段名，None 表示前向结束。
            outputs (dict): 当前阶段产生的张量与容器。
        """
        if self.active is not None:
            name, start = self.active
            if self.mode == "events":
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.rows.append((name, start, end))
            else:
                started_at, memory = start
                end_memory = allocator_end(memory)
                self.rows.append(dict(
                    stage=name, sync_wall_ms=milliseconds(started_at),
                    **end_memory, **tensor_snapshot(outputs), details=self.details,
                ))
        self.active = None
        self.details = []
        if next_stage is not None:
            if self.mode == "events":
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                memory = allocator_start()
                start = (perf_counter_ns(), memory)
            self.active = (next_stage, start)

    def finish(self):
        torch.cuda.synchronize()
        if self.mode == "events":
            return [dict(stage=name, event_ms=start.elapsed_time(end)) for name, start, end in self.rows]
        return self.rows


def compile_observed_function(original, boundaries, callback, replacements=None):
    """在指定源码边界采集元数据，执行原函数的计算。"""
    lines = dedent(getsource(original)).splitlines()
    for marker, stage, values in boundaries:
        matches = [i for i, line in enumerate(lines) if line.lstrip().startswith(marker)]
        if len(matches) != 1:
            raise ValueError(f"Expected one source marker {marker!r}, found {len(matches)}")
        index = matches[0]
        padding = lines[index][:-len(lines[index].lstrip())]
        lines.insert(index, f"{padding}_profile_mark({stage!r}, {values})")
    namespace = dict(original.__globals__, _profile_mark=callback)
    namespace.update(replacements or {})
    exec(compile("\n".join(lines), original.__code__.co_filename, "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def instrument_encoder(encoder, probe, paper_release):
    """临时添加观测边界，退出时恢复官方前向函数。

    Args:
        encoder (Encoder): 已构建的官方编码器。
        probe (StageProbe): 接收边界与中间量的对象。
        paper_release (bool): 使用论文对应的 main 分支计算路径。
    """
    encoder_boundaries = PAPER_BOUNDARIES if paper_release else SPFSPLAT_BOUNDARIES
    backbone_boundaries = PAPER_BACKBONE_BOUNDARIES if paper_release else BACKBONE_BOUNDARIES
    encoder_forward = encoder.forward.__func__
    replacements = {}
    if paper_release and probe.mode == "memory":
        globals_ = encoder_forward.__globals__
        replacements["compute_frequency_high_freq_score"] = compile_observed_function(
            globals_["compute_frequency_high_freq_score"],
            (("h, w = magnitude_spectrum.shape", "FFT 临时量",
              "gray=gray_image, spectrum=f, shifted=fshift, magnitude=magnitude_spectrum"),),
            probe.observe,
        )
        replacements["bipartite_soft_matching_random2d"] = compile_observed_function(
            globals_["bipartite_soft_matching_random2d"],
            (("node_max, node_idx = fast_similarity_chunks(", "匹配距离输入", "source=a, destination=b"),),
            probe.observe,
        )

    encoder.forward = MethodType(
        compile_observed_function(encoder_forward, encoder_boundaries, probe.mark, replacements), encoder
    )
    try:
        backbone = encoder.backbone
        backbone.forward = MethodType(
            compile_observed_function(backbone.forward.__func__, backbone_boundaries, probe.mark), backbone
        )
        try:
            yield
        finally:
            del backbone.forward
    finally:
        del encoder.forward


def assert_same_output(reference, measured):
    """核对场景高斯及相机与无观测调用逐元素一致。"""
    for field in fields(reference["gaussians"]):
        expected = getattr(reference["gaussians"], field.name)
        actual = getattr(measured["gaussians"], field.name)
        if expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise AssertionError(f"Instrumented Gaussian field changed: {field.name}")
    if not torch.equal(reference["extrinsics"]["c"], measured["extrinsics"]["c"]):
        raise AssertionError("Instrumented camera output changed")


def verified_stage2(path, digest):
    """以流式方式计算已发布权重的哈希。"""
    with path.open("rb") as file:
        actual = sha256()
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError(f"Unexpected checkpoint SHA-256: {actual.hexdigest()}")


def make_encoder(args):
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder

    experiment = "ecosplat/re10k" if args.paper_release else "spfsplat/re10k_10view"
    budget = (
        f"model.encoder.primitive_ratio={args.rho}" if args.paper_release
        else f"+model.encoder.igf.inference_rho={args.rho}"
    )
    with initialize_config_dir(config_dir=str(args.repo / "config"), version_base=None):
        config = compose(
            config_name="main",
            overrides=[f"+experiment={experiment}", "mode=test",
                       f"model.encoder.pretrained_weights={args.stage1}", budget],
        )
    set_cfg(config)
    encoder, _ = get_encoder(load_typed_root_config(config).model.encoder)
    checkpoint = torch.load(args.stage2, map_location="cpu", weights_only=False)
    state = {key.removeprefix("encoder."): value for key, value in checkpoint["state_dict"].items()
             if key.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)
    step = checkpoint["global_step"]
    return encoder.eval().cuda(), int(step)


def select_scenes(repo, root, limit, paper_release):
    index = json.loads((root / "test/index.json").read_text())
    if paper_release:
        entries = list(FIXED_PAPER_SCENES.items())[:limit]
        if len(entries) != limit:
            raise ValueError("The fixed paper-release comparison uses up to three RE10K scenes")
        return [(scene, root / "test" / index[scene], list(ids)) for scene, ids in entries]
    protocol = json.loads((repo / "assets/evaluation_index_re10k.json").read_text())
    chosen = []
    chunks = set()
    for scene, entry in sorted(protocol.items()):
        if not entry or scene not in index or index[scene] in chunks:
            continue
        start, end = sorted(entry["context"])[:2]
        if end - start < 20:
            continue
        ids = [start, end]
        ids.extend(start + round((end - start) * fraction) for fraction in (0.4, 0.6, 0.2, 0.8))
        chosen.append((scene, root / "test" / index[scene], ids))
        chunks.add(index[scene])
        if len(chosen) == limit:
            return chosen
    raise ValueError("Could not form the fixed RealEstate10K scene set")


def construct_input(chunk, scene, indices, size):
    """读取 RE10K 源帧并生成归一化内参与图像张量。

    Args:
        chunk (Path): 包含场景的 RE10K 测试分片。
        scene (str): 场景 ID。
        indices (list[int]): 保留参考顺序的源视图索引。
        size (int): 正方形输入边长。

    Returns:
        tuple: CPU context 和各段墙钟、输入字节及 shape。
    """
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from src.dataset.shims.crop_shim import rescale_and_crop

    start = perf_counter_ns()
    data = torch.load(chunk, map_location="cpu", weights_only=True)
    example = next(example for example in data if example["key"] == scene)
    encoded = [example["images"][index] for index in indices]
    camera = example["cameras"][indices]
    encoded_bytes = sum(frame.numel() * frame.element_size() for frame in encoded)
    read_ms = milliseconds(start)

    start = perf_counter_ns()
    images = torch.stack([to_tensor(Image.open(BytesIO(frame.numpy().tobytes())).convert("RGB"))
                          for frame in encoded])
    decode_ms = milliseconds(start)

    start = perf_counter_ns()
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(len(indices), 1, 1)
    intrinsics[:, 0, 0], intrinsics[:, 1, 1] = camera[:, 0], camera[:, 1]
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = camera[:, 2], camera[:, 3]
    resized, cropped = rescale_and_crop(images, intrinsics, (size, size))
    if size != 256:
        height, width = images.shape[-2:]
        scale = max(size / height, size / width)
        h_scaled, w_scaled = round(height * scale), round(width * scale)
        cropped[:, 0, 2] = (intrinsics[:, 0, 2] * w_scaled - (w_scaled - size) // 2) / size
        cropped[:, 1, 2] = (intrinsics[:, 1, 2] * h_scaled - (h_scaled - size) // 2) / size
    context = dict(image=resized.unsqueeze(0), intrinsics=cropped.unsqueeze(0),
                   index=torch.tensor(indices).unsqueeze(0))
    prep_ms = milliseconds(start)
    decoded_rgb_bytes = images.numel() * images.element_size()
    input_bytes = sum(item.numel() * item.element_size() for item in context.values())
    return context, dict(
        read_scene_ms=read_ms, decode_rgb_ms=decode_ms, crop_pack_ms=prep_ms,
        encoded_frame_bytes=encoded_bytes, chunk_file_bytes=chunk.stat().st_size,
        decoded_rgb_tensor_bytes=decoded_rgb_bytes, cpu_context_bytes=input_bytes,
    )


def to_device(cpu_context):
    return {key: value.cuda() for key, value in cpu_context.items()}


def wall_forward(encoder, context, step):
    torch.cuda.synchronize()
    start = perf_counter_ns()
    output = encoder(context, global_step=step)
    torch.cuda.synchronize()
    return milliseconds(start), output


def expected_budget(context, size, rho, paper_release):
    """在计时后复算每个视图的实际保留配额。"""
    if not paper_release:
        return [int(max(rho, 1 / 16) * size * size)] * context["image"].shape[1]
    from src.utils.geometry_torch import compute_frequency_high_freq_score

    scores = torch.stack([compute_frequency_high_freq_score(frame) for frame in context["image"][0]])
    weights = torch.softmax(scores / 0.3, dim=0)
    ratios = rho * weights / weights.mean()
    return [int(ratio * size * size) for ratio in ratios]


def profile_case(encoder, step, scene, chunk, indices, size, args):
    """执行同一输入的普通、事件与显存采样，保留逐次结果。"""
    cpu_context, source = construct_input(chunk, scene, indices, size)
    context = to_device(cpu_context)
    for _ in range(args.warmup):
        forward = encoder(context, global_step=step)
        del forward

    wall = []
    for _ in range(args.repeats):
        took, output = wall_forward(encoder, context, step)
        wall.append(took)
        del output

    peaks = []
    for _ in range(args.peak_repeats):
        memory = allocator_start()
        output = encoder(context, global_step=step)
        peaks.append(allocator_end(memory))
        del output

    initial_stage = "分视图高频打分" if args.paper_release else "图像与内参编码"
    events = []
    for _ in range(args.event_repeats):
        probe = StageProbe("events")
        with instrument_encoder(encoder, probe, args.paper_release):
            probe.mark(initial_stage)
            output = encoder(context, global_step=step)
            probe.mark(None)
        events.append(probe.finish())
        del output

    memory = StageProbe("memory")
    with instrument_encoder(encoder, memory, args.paper_release):
        memory.mark(initial_stage)
        measured = encoder(context, global_step=step)
        memory.mark(None, scene=measured)
    parts = memory.finish()
    output_size = tensor_snapshot(measured)
    output_count = measured["gaussians"].means.shape[1]
    quotas = expected_budget(context, size, args.rho, args.paper_release)
    expected_count = sum(quotas)
    if output_count != expected_count or not torch.isfinite(measured["gaussians"].means).all().item():
        raise AssertionError(f"Invalid scene {scene}: {output_count} gaussians, expected {expected_count}")
    reference = encoder(context, global_step=step)
    assert_same_output(reference, measured)
    del reference, measured, context

    end_to_end = []
    for _ in range(args.e2e_repeats):
        memory = allocator_start()
        start = perf_counter_ns()
        cpu, breakdown = construct_input(chunk, scene, indices, size)
        torch.cuda.synchronize()
        transfer_start = perf_counter_ns()
        gpu = to_device(cpu)
        torch.cuda.synchronize()
        transfer_ms = milliseconds(transfer_start)
        _, output = wall_forward(encoder, gpu, step)
        elapsed = milliseconds(start)
        end_to_end.append(dict(wall_ms=elapsed, h2d_ms=transfer_ms, **breakdown, **allocator_end(memory)))
        del output, gpu, cpu

    return dict(
        scene=scene, resolution=size, view_count=len(indices), frame_ids=indices,
        source=source, forward_wall_ms=wall, forward_peaks=peaks, event_stages=events,
        stage_memory=parts, output=output_size, candidates=len(indices) * size * size,
        output_gaussians=output_count, view_quotas=quotas, exact_equivalence=True, end_to_end=end_to_end,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "dataset", "stage1", "stage2", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--stage2-sha256", required=True)
    parser.add_argument("--scene-limit", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 256])
    parser.add_argument("--views", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--peak-repeats", type=int, default=3)
    parser.add_argument("--event-repeats", type=int, default=3)
    parser.add_argument("--e2e-repeats", type=int, default=5)
    args = parser.parse_args()
    if any(v < 2 or v > 6 for v in args.views) or not 0 < args.rho <= 1:
        parser.error("This benchmark accepts 2 to 6 views and a budget in (0, 1].")
    revision = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    if revision not in (PAPER_RELEASE, SPFSPLAT_RELEASE):
        raise ValueError(f"Expected a known EcoSplat upstream commit, got {revision}")
    args.paper_release = revision == PAPER_RELEASE
    if args.paper_release and args.rho != 0.4:
        parser.error("The original paper release hardcodes its primitive ratio to 0.4.")
    sys.path.insert(0, str(args.repo))
    if not args.paper_release:
        sys.path.insert(0, str(args.repo / "ecosplat_wrapper"))
    os.chdir(args.repo)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    print("VERIFY_STAGE2", args.stage2, flush=True)
    verified_stage2(args.stage2, args.stage2_sha256)
    print("LOAD_BASELINE", revision, flush=True)
    encoder, step = make_encoder(args)
    scenes = select_scenes(args.repo, args.dataset, args.scene_limit, args.paper_release)
    payload = dict(
        baseline_commit=revision, stage2_sha256=args.stage2_sha256, global_step=step, rho=args.rho,
        upstream_remote=subprocess.check_output(["git", "-C", args.repo, "remote", "get-url", "origin"], text=True).strip(),
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        parameters=sum(p.numel() for p in encoder.parameters()), cpu_threads=torch.get_num_threads(),
        model_alloc_bytes=torch.cuda.memory_allocated(),
        settings=vars(args).copy(), cases=[],
    )
    with torch.inference_mode():
        for size in args.sizes:
            views = args.views if size != 256 else args.views[:1]
            for count in views:
                for scene, path, ids in scenes:
                    print(f"CASE_START {scene} {size}x{size} views={count}", flush=True)
                    result = profile_case(encoder, step, scene, path, ids[:count], size, args)
                    payload["cases"].append(result)
                    temporary = args.output.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                    temporary.replace(args.output)
                    print(f"CASE_DONE {scene} {size}x{size} views={count} gaussians={result['output_gaussians']}", flush=True)
    print("PROFILE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()