"""用现有 3DGS 检查像素内 alpha 相关项的量级。"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

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


def raster_estimates(camera, native, metadata, colors):
    """保持相同中心贡献列表，分离逐项平均与整条射线平均。"""
    w, h = camera.width, camera.height
    pixel, i, alpha, cumulative, _ = ordered_alpha_samples(metadata, w, h)
    _, counts = torch.unique_consecutive(pixel, return_counts=True)
    starts = counts.cumsum(dim=0) - counts
    xy = torch.stack((pixel % w + .5, pixel // w + .5), dim=-1)
    dx, dy = (metadata["means2d"][0, i] - xy).unbind(-1)
    a, b, c = metadata["conics"][0, i].unbind(-1)
    opacity = metadata["opacities"][0, i]
    grads = alpha[:, None] * torch.stack((a * dx + b * dy, b * dx + c * dy), dim=-1)
    grads = torch.where((alpha < .999)[:, None], grads, 0)
    front = torch.exp(-(cumulative + torch.log1p(-alpha)))
    change = correlation(alpha, grads, front, starts, counts)
    change_rgb = sum_colors(change, colors[i], pixel, w * h)
    phase = float(np.sqrt(1 / 12))
    total_alpha = torch.zeros_like(alpha)
    exact = torch.zeros_like(native.reshape(-1, 3))
    for u, v in itertools.product((-phase, phase), repeat=2):
        x, y = dx - u, dy - v
        sample = (opacity * torch.exp(-.5 * (a * x * x + c * y * y) - b * x * y)).clamp(0, .999)
        total_alpha += sample / 4
        exact += sum_colors(composite(sample, starts, counts), colors[i], pixel, w * h) / 4
    separable = sum_colors(composite(total_alpha, starts, counts), colors[i], pixel, w * h)
    return {"center": native.reshape(-1, 3), "per_splat_box": separable,
            "second_order": separable + change_rgb, "four_phase_same_samples": exact}, change_rgb


def compare(model, camera):
    """以留出像素盒真值对照中心采样与不同合成近似。"""
    colors = model.colors(camera)
    rgb, _, meta = model.render(camera, colors)
    ref = reference_image(model, camera, colors, order=4)
    estimates, correction = raster_estimates(camera, rgb[0], meta, colors)
    estimates["four_phase_native"] = reference_image(model, camera, colors, order=2)
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
    return metrics, float(correction.abs().mean())


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
    args = parser.parse_args()
    loader = load_tum if args.tum else load_replica
    cameras, _, names = loader(args.tum or args.replica)
    cameras, names = select_views(cameras, names)
    model = FrozenGaussians.load(args.checkpoint)
    result = []
    for i in (1, 5)[:args.views]:
        camera = reduced(cameras[i], 108)
        metrics, delta = compare(model, camera)
        result.append({"frame": names[i], "image_size": [camera.width, camera.height],
                       "metrics": metrics, "mean_abs_correlation": delta})
        print(f"View {names[i]}: {metrics}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
