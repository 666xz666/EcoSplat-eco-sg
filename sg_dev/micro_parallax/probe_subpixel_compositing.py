"""用现有 3DGS 检查像素内 alpha 相关项的量级。"""

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, rasterize_to_indices_in_range

from evaluate import load_replica
from probe_edit_transport import reduced, select_views
from probe_visibility import load_tum
from readout import Camera, FrozenGaussians, ordered_alpha_samples


def exclusive(values, starts, counts):
    """以双精度计算同一像素中位于当前高斯前方的逐项和。"""
    sums = values.cumsum(dim=0, dtype=torch.float64)
    offset = (sums - values)[starts].repeat_interleave(counts, dim=0)
    return sums - values - offset


def sum_colors(sample, colors, pixels, n_pixels):
    """累加光栅贡献到扁平 RGB 图。"""
    image = colors.new_zeros((n_pixels, 3))
    image.index_add_(0, pixels, sample.to(colors.dtype)[:, None] * colors)
    return image


def composite(alpha, starts, counts):
    """计算按光栅顺序混合时，每个高斯在对应像素上的权重。"""
    front = exclusive(-torch.log1p(-alpha), starts, counts)
    return alpha * torch.exp(-front)


def correlation(alpha, gradients, front, starts, counts):
    """一阶 alpha 空间变化在像素盒积分内产生的二阶合成交叉项。

    Args:
        alpha (Tensor, (M,)): 像素中心逐项不透明度。
        gradients (Tensor, (M, 2)): 对像素坐标 x/y 的 opacity 梯度。
        front (Tensor, (M,)): 当前项之前的透过率。
        starts (Tensor, (P,)): 每条非空射线的首项索引。
        counts (Tensor, (P,)): 每条非空射线中的高斯数。

    Returns:
        Tensor: 每个可见高斯的平均混合权重校正 (M,)。
    """
    opacity_gradient = gradients / (1 - alpha)[:, None]
    prefix = exclusive(opacity_gradient, starts, counts)
    square_sum = exclusive(opacity_gradient.square().sum(dim=-1), starts, counts)
    product_term = alpha * (prefix.square().sum(dim=-1) - square_sum)
    direct_term = -2 * (gradients * prefix).sum(dim=-1)
    # 盒积分每个轴的方差是 1/12，泰勒展开系数另乘 1/2。
    return front * (product_term + direct_term) / 24


def current_camera(camera, offset):
    """探测同一像素盒内的点，保持投影几何、筛选和分辨率一致。"""
    K = camera.intrinsic.copy()
    K[:2, 2] -= np.asarray(offset)
    return Camera(camera.c2w, K, camera.width, camera.height)


def reference_image(model, camera, colors, order):
    """用高阶 Gauss-Legendre 张量积积分完整 alpha 合成。"""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = None
    for i, j in itertools.product(range(order), repeat=2):
        offset = (.5 * nodes[i], .5 * nodes[j])
        result_i = model.render(current_camera(camera, offset), colors)[0][0]
        weight = float(.25 * weights[i] * weights[j])
        result = result_i * weight if result is None else result + result_i * weight
    return result.reshape(-1, 3).clamp(0, 1)


def expanded_samples(metadata, width, height):
    """在中心投影上保留可能进入整像素盒的贡献，透明度仍由原高斯计算。"""
    threshold = 1 / 255
    proposal = 1.005
    means, conics, depths = (metadata[key] for key in ("means2d", "conics", "depths"))
    radii = metadata["radii"]
    opacity = metadata["opacities"]
    valid = (radii > 0).all(dim=-1) & (opacity > threshold)
    a, b, c = conics.unbind(dim=-1)
    largest = .5 * (a + c + torch.sqrt((a - c).square() + 4 * b.square()))
    # 高斯阈值椭圆在像素盒内按 Q 范数膨胀。盒内最大位移不超过 sqrt(1/2)。
    max_norm = torch.sqrt(2 * (opacity / threshold).clamp_min(1).log()) + torch.sqrt(.5 * largest.clamp_min(0))
    factor = float(2 * np.log(proposal)) / max_norm.square().clamp_min(1e-6)
    proposal_conics = torch.where(valid[..., None], conics * factor[..., None], 0).contiguous()
    proposal_opacity = torch.where(valid, torch.full_like(opacity, proposal * threshold), 0).contiguous()
    radii = torch.where(valid[..., None], radii + 1, 0).contiguous()
    tile, tw, th = (metadata[key] for key in ("tile_size", "tile_width", "tile_height"))
    _, ids, flat = isect_tiles(means, radii, depths, tile, tw, th)
    offsets = isect_offset_encode(ids, 1, tw, th)
    layers, pixels, _ = rasterize_to_indices_in_range(
        0, max(1, (flat.numel() + tile * tile - 1) // (tile * tile)),
        torch.ones((1, height, width), device=means.device), means, proposal_conics,
        proposal_opacity, width, height, tile, offsets, flat)
    # 保持每条射线的同深度高斯顺序，避免在比较时改变遮挡顺序。
    count = depths.shape[-1]
    rank = torch.empty(count, dtype=torch.int64, device=means.device)
    rank[depths[0].argsort(stable=True)] = torch.arange(count, device=means.device)
    order = (pixels.long() * count + rank[layers.long()]).argsort()
    return pixels[order].long(), layers[order].long()


def analytic_mean(a, b, c, dx, dy, opacity):
    """用与像素盒方差相同的高斯核近似计算单项平均 alpha。"""
    variance = 1 / 12
    determinant = (a * c - b.square()).clamp_min(0)
    volume = 1 + variance * (a + c) + variance ** 2 * determinant
    qa = (a + variance * determinant) / volume
    qc = (c + variance * determinant) / volume
    qb = b / volume
    sigma = .5 * (qa * dx.square() + qc * dy.square()) + qb * dx * dy
    return (opacity * torch.rsqrt(volume) * torch.exp(-sigma)).clamp(0, .999)


def raster_estimates(camera, native, metadata, colors, *, padded=False):
    """保持相同中心贡献列表，分离逐项平均与整条射线平均。"""
    w, h = camera.width, camera.height
    if padded:
        pixel, i = expanded_samples(metadata, w, h)
    else:
        pixel, i, alpha, cumulative, _ = ordered_alpha_samples(metadata, w, h)
    _, counts = torch.unique_consecutive(pixel, return_counts=True)
    starts = counts.cumsum(dim=0) - counts
    xy = torch.stack((pixel % w + .5, pixel // w + .5), dim=-1)
    dx, dy = (metadata["means2d"][0, i] - xy).unbind(-1)
    a, b, c = metadata["conics"][0, i].unbind(-1)
    opacity = metadata["opacities"][0, i]
    threshold = 1 / 255
    if padded:
        sigma = .5 * (a * dx.square() + c * dy.square()) + b * dx * dy
        alpha = (opacity * torch.exp(-sigma)).clamp(0, .999)
        alpha = torch.where(alpha >= threshold, alpha, 0)
        front = torch.exp(-exclusive(-torch.log1p(-alpha), starts, counts))
    else:
        front = torch.exp(-(cumulative + torch.log1p(-alpha)))
    grads = alpha[:, None] * torch.stack((a * dx + b * dy, b * dx + c * dy), dim=-1)
    grads = torch.where((alpha < .999)[:, None], grads, 0)
    change = correlation(alpha, grads, front, starts, counts)
    change_rgb = sum_colors(change, colors[i], pixel, w * h)
    phase = float(np.sqrt(1 / 12))
    total_alpha = torch.zeros_like(alpha)
    exact = torch.zeros_like(native.reshape(-1, 3))
    for u, v in itertools.product((-phase, phase), repeat=2):
        x, y = dx - u, dy - v
        sample = (opacity * torch.exp(-.5 * (a * x * x + c * y * y) - b * x * y)).clamp(0, .999)
        if padded:
            sample = torch.where(sample >= threshold, sample, 0)
        total_alpha += sample / 4
        exact += sum_colors(composite(sample, starts, counts), colors[i], pixel, w * h) / 4
    separable = sum_colors(composite(total_alpha, starts, counts), colors[i], pixel, w * h)
    if padded:
        approximated = analytic_mean(a, b, c, dx, dy, opacity)
        approximated = torch.where(approximated >= threshold, approximated, 0)
        blur = sum_colors(composite(approximated, starts, counts), colors[i], pixel, w * h)
        return {"padded_per_splat_box": separable, "padded_second_order": separable + change_rgb,
                "padded_analytic_box": blur + change_rgb, "padded_four_phase": exact}, counts.max().item()
    return {"center": native.reshape(-1, 3), "per_splat_box": separable,
            "second_order": separable + change_rgb, "four_phase_same_samples": exact}, change_rgb


def compare(model, camera, padded):
    """以留出像素盒真值对照中心采样与不同合成近似。"""
    colors = model.colors(camera)
    rgb, _, meta = model.render(camera, colors)
    ref = reference_image(model, camera, colors, order=4)
    estimates, correction = raster_estimates(camera, rgb[0], meta, colors)
    estimates["four_phase_native"] = reference_image(model, camera, colors, order=2)
    padded_seconds, maximum_samples = None, None
    if padded:
        torch.cuda.synchronize()
        started = perf_counter()
        expanded, maximum_samples = raster_estimates(camera, rgb[0], meta, colors, padded=True)
        torch.cuda.synchronize()
        padded_seconds = perf_counter() - started
        estimates.update(expanded)
    w, h = camera.width, camera.height
    color = ref.view(h, w, 3)
    dx = torch.zeros_like(color)
    dy = torch.zeros_like(color)
    dx[:, 1:] = color[:, 1:] - color[:, :-1]
    dy[1:] = color[1:] - color[:-1]
    gradient = (dx.square() + dy.square()).sum(dim=-1).flatten()
    edge = gradient >= torch.quantile(gradient, .8)
    metrics = {}
    for name, result in estimates.items():
        error = result.clamp(0, 1) - ref
        metrics[name] = {"mse": float(error.square().mean()), "edge_mse": float(error[edge].square().mean()),
                         "mae": float(error.abs().mean())}
    return metrics, float(correction.abs().mean()), padded_seconds, maximum_samples


@torch.inference_mode()
def main():
    """用同一组固定分辨率下真实检查点进行一至两个视角的短测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replica", type=Path)
    source.add_argument("--tum", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=2, choices=(1, 2))
    parser.add_argument("--padded", action="store_true")
    args = parser.parse_args()
    loader = load_tum if args.tum else load_replica
    cameras, _, names = loader(args.tum or args.replica)
    cameras, names = select_views(cameras, names)
    model = FrozenGaussians.load(args.checkpoint)
    result = []
    for i in (1, 5)[:args.views]:
        camera = reduced(cameras[i], 108)
        metrics, delta, seconds, count = compare(model, camera, args.padded)
        result.append({"frame": names[i], "image_size": [camera.width, camera.height],
                       "metrics": metrics, "mean_abs_correlation": delta,
                       "padded_prototype_seconds": seconds, "max_padded_samples": count})
        print(f"View {names[i]}: {metrics}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
