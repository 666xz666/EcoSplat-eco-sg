"""冻结 3DGS 的原始深度、微视差及虚拟视图匹配读出。"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import rasterize_to_indices_in_range, spherical_harmonics


@dataclass(frozen=True)
class Camera:
    """OpenCV 相机到世界坐标，内参采用整数像素中心。"""

    c2w: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int

    def translated(self, horizontal, vertical):
        """平移相机中心并保持旋转和内参固定。"""
        moved = self.c2w.copy()
        moved[:3, 3] += self.c2w[:3, :2] @ np.array([horizontal, vertical])
        return Camera(moved, self.intrinsic, self.width, self.height)


class FrozenGaussians:
    """用同一个 gsplat 后端渲染原始和受控平移视图。"""

    def __init__(self, capture):
        self.degree, xyz, dc, rest, scale, rotation, opacity = capture[:7]
        self.xyz = xyz.detach().cuda()
        self.scale = scale.detach().cuda().exp()
        self.rotation = rotation.detach().cuda()
        self.opacity = opacity.detach().cuda().sigmoid().flatten()
        self.sh = torch.cat([dc.detach().cuda(), rest.detach().cuda()], dim=1).contiguous()

    @classmethod
    def load(cls, path: Path):
        capture, iteration = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded {path.name}, step {iteration}, Gaussians {len(capture[1]):,}", flush=True)
        return cls(capture)

    def colors(self, camera):
        """返回指定相机的逐高斯 RGB (N, 3)。"""
        center = torch.as_tensor(camera.c2w[:3, 3], dtype=torch.float32, device="cuda")
        return (spherical_harmonics(self.degree, self.xyz - center, self.sh) + 0.5).clamp_min(0)

    def render(self, camera, colors, *, with_depth=False):
        """渲染一个视角，所有方法共用内参转换和裁剪面。"""
        view = torch.as_tensor(np.linalg.inv(camera.c2w), dtype=torch.float32, device="cuda")
        intrinsic = torch.as_tensor(camera.intrinsic.copy(), dtype=torch.float32, device="cuda")
        # CUDA 内核使用 (x + 0.5, y + 0.5) 作为像素采样点。
        intrinsic[:2, 2] += 0.5
        return rasterization(
            self.xyz, self.rotation, self.scale, self.opacity, colors,
            view[None], intrinsic[None], camera.width, camera.height,
            near_plane=0.1, far_plane=100.0, packed=False,
            render_mode="RGB+ED" if with_depth else "RGB",
        )


def ordered_alpha_samples(metadata, width, height):
    """返回固定光栅顺序的像素与高斯贡献。

    Args:
        metadata (dict): 单相机、未打包的 gsplat 投影结果。
        width (int): 像素宽度。
        height (int): 像素高度。

    Returns:
        tuple: 每条贡献的像素索引、高斯索引、alpha、累计光学厚度和整条射线 alpha，均为 (M,)。
    """
    means = metadata["means2d"]
    conics = metadata["conics"]
    depths = metadata["depths"][0]
    opacity = metadata["opacities"]
    offsets = metadata["isect_offsets"]
    flat = metadata["flatten_ids"]
    tile = metadata["tile_size"]
    transmittance = torch.ones((1, height, width), device=means.device)
    block_count = (flat.numel() + tile * tile - 1) // (tile * tile)
    gaussians, pixels, _ = rasterize_to_indices_in_range(
        0, max(1, block_count), transmittance, means, conics, opacity,
        width, height, tile, offsets, flat,
    )
    if not gaussians.numel():
        empty = means.new_empty(0)
        return pixels.long(), gaussians.long(), empty, empty, empty
    rank = torch.empty(len(depths), dtype=torch.int64, device=depths.device)
    rank[depths.argsort(stable=True)] = torch.arange(len(depths), dtype=torch.int64, device=depths.device)
    ordered = (pixels.long() * len(depths) + rank[gaussians.long()]).argsort()
    pixels = pixels[ordered].long()
    gaussians = gaussians[ordered].long()
    del rank, ordered
    xy = torch.stack((pixels % width + 0.5, pixels // width + 0.5), dim=1)
    dx, dy = (means[0, gaussians] - xy).unbind(-1)
    a, b, c = conics[0, gaussians].unbind(-1)
    alpha = (opacity[0, gaussians] * torch.exp(-0.5 * (a * dx.square() + c * dy.square()) - b * dx * dy)).clamp(max=0.999)
    del xy, dx, dy, a, b, c
    # 光学厚度沿每条射线相加。双精度避免大前缀相减丢失微弱贡献。
    thickness = -torch.log1p(-alpha)
    prefix = thickness.cumsum(dim=0, dtype=torch.float64)
    _, count = torch.unique_consecutive(pixels, return_counts=True)
    end = count.cumsum(dim=0) - 1
    start = end - count + 1
    offset = prefix[start] - thickness[start]
    cumulative = prefix - offset.repeat_interleave(count)
    total_alpha = (-torch.expm1(-(prefix[end] - offset))).repeat_interleave(count)
    return pixels, gaussians, alpha, cumulative, total_alpha


def opacity_quantiles(metadata, width, height, probabilities=(0.1, 0.5, 0.9)):
    """从同一次 3DGS 渲染的贡献列表计算归一化 alpha 深度分位数。

    Args:
        metadata (dict): 单相机、未打包的 gsplat 投影结果。
        width (int): 像素宽度。
        height (int): 像素高度。
        probabilities (tuple): 累积可见 alpha 的分位点。

    Returns:
        ndarray: (len(probabilities), H, W)，无贡献像素为 NaN。
    """
    pixels, gaussians, _, cumulative, total_alpha = ordered_alpha_samples(metadata, width, height)
    if not gaussians.numel():
        return np.full((len(probabilities), height, width), np.nan, dtype=np.float32)
    depths = metadata["depths"][0]
    point_depth = depths[gaussians]
    quantiles = []
    for probability in probabilities:
        threshold = -torch.log1p(-probability * total_alpha)
        selected = torch.where(cumulative >= threshold, point_depth, torch.inf)
        result = torch.full((height * width,), torch.inf, dtype=depths.dtype, device=means.device)
        result.scatter_reduce_(0, pixels, selected, reduce="amin")
        result[~result.isfinite()] = torch.nan
        quantiles.append(result.reshape(height, width).cpu().numpy())
    return np.stack(quantiles)


def box(array, window=5):
    """平滑局部最小二乘的统计量，保留每个像素的输出。"""
    return cv2.boxFilter(array, ddepth=-1, ksize=(window, window))


def inverse_depth_from_motion(center, renders, focal, distance):
    """按小平移的图像梯度估计逆深度和残差。

    Args:
        center (ndarray, (H, W, 3)): 中心视角 RGB。
        renders (dict): 键 (axes, sign, scale) 对应受控平移 RGB。
        focal (ndarray, (2,)): x/y 方向焦距，单位为像素。
        distance (float): 小尺度平移的世界单位长度。

    Returns:
        tuple: 小尺度逆深度 (H, W)、置信度 (H, W)。
    """
    gradients = [cv2.Sobel(center, cv2.CV_32F, int(axis == 0), int(axis == 1), ksize=3, scale=1 / 8)
                 * focal[axis] for axis in range(2)]
    energy = box(sum(np.sum(matrix * matrix, axis=-1) for matrix in gradients))
    estimates = []
    residuals = []
    for scale in (1, 2):
        deltas = [(renders[axis, 1, scale] - renders[axis, -1, scale]) / (2 * scale * distance)
                  for axis in range(2)]
        cross = box(sum(np.sum(g * d, axis=-1) for g, d in zip(gradients, deltas)))
        response_energy = box(sum(np.sum(d * d, axis=-1) for d in deltas))
        inverse = cross / np.maximum(energy, 1e-8)
        residual = np.sqrt(np.maximum(response_energy - inverse * cross, 0) / np.maximum(response_energy, 1e-8))
        estimates.append(inverse)
        residuals.append(residual)
    fine, coarse = estimates
    disagreement = np.abs(fine - coarse) / np.maximum(np.abs(fine) + np.abs(coarse), 1e-4)
    texture = energy / (energy + (0.002 * max(focal)) ** 2)
    confidence = texture * np.exp(-3 * residuals[0] - 3 * disagreement)
    return fine, confidence


def virtual_stereo(center, renders, focal, distance):
    """用同一组虚拟图做多方向平面扫描，比较常规匹配与微视差。"""
    height, width = center.shape[:2]
    log_inverse = np.linspace(np.log(1 / 12), np.log(1 / 0.15), 96, dtype=np.float32)
    cost = np.empty((len(log_inverse), height, width), dtype=np.float32)
    center_gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    grayscale = {key: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for key, frame in renders.items()}
    for i, inverse in enumerate(np.exp(log_inverse)):
        error = np.zeros_like(center_gray)
        for (axis, sign, scale), frame in grayscale.items():
            matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            matrix[axis, 2] = -focal[axis] * distance * inverse * sign * scale
            moved = cv2.warpAffine(frame, matrix, (width, height),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            error += np.abs(moved - center_gray)
        cost[i] = box(error / len(grayscale))
    best = np.argmin(cost, axis=0)
    vmin = np.take_along_axis(cost, best[None], axis=0)[0]
    left = np.take_along_axis(cost, np.maximum(best - 1, 0)[None], axis=0)[0]
    right = np.take_along_axis(cost, np.minimum(best + 1, len(log_inverse) - 1)[None], axis=0)[0]
    refine = np.clip(0.5 * (left - right) / np.maximum(left + right - 2 * vmin, 1e-7), -0.5, 0.5)
    refine[(best == 0) | (best == len(log_inverse) - 1)] = 0
    median_cost = np.median(cost, axis=0)
    confidence = np.clip((median_cost - vmin) / (median_cost + 1e-4), 0, 1)
    inverse = np.exp(log_inverse[best] + refine * (log_inverse[1] - log_inverse[0]))
    return inverse, confidence


def depth_prediction(inverse, confidence, coverage):
    """保留可用深度和置信度，无法观测处没有高分输出。"""
    valid = np.isfinite(inverse) & (inverse > 1 / 12) & (inverse < 1 / 0.15) & (coverage > 0.25)
    depth = np.full(inverse.shape, np.nan, dtype=np.float32)
    depth[valid] = 1 / inverse[valid]
    score = np.where(valid, confidence * coverage, -np.inf)
    score[:3] = score[-3:] = -np.inf
    score[:, :3] = score[:, -3:] = -np.inf
    return depth, score


@torch.inference_mode()
def readouts(model, camera, *, smoke=False):
    """计算所有深度读出，冻结颜色与保留 SH 的试验共用相同平移。"""
    started = perf_counter()
    fixed_colors = model.colors(camera)
    center_render, alpha, metadata = model.render(camera, fixed_colors, with_depth=True)
    center = center_render[0, :, :, :3].cpu().numpy()
    expected = center_render[0, :, :, 3].cpu().numpy()
    coverage = alpha[0, :, :, 0].cpu().numpy()
    assert np.isfinite(center).all() and np.isfinite(expected).all()
    good_reference = (coverage > 0.5) & (expected > 0.15) & (expected < 12)
    if not good_reference.any():
        raise ValueError("No valid center-view optical depth")
    focal = camera.intrinsic.diagonal()[:2].astype(np.float32)
    # 小平移使中位深度处的位移约 0.75 像素，且不读取测试真值。
    distance = float(0.75 * np.median(expected[good_reference]) / max(focal))
    timing = {"center": perf_counter() - started, "fixed_probes": 0.0, "live_probes": 0.0}
    fixed = {}
    live = {}
    for scale in (1, 2):
        for axis in range(2):
            for sign in (-1, 1):
                offset = distance * sign * scale
                shifted = camera.translated(offset if axis == 0 else 0, offset if axis == 1 else 0)
                started = perf_counter()
                fixed[axis, sign, scale] = model.render(shifted, fixed_colors)[0][0].cpu().numpy()
                timing["fixed_probes"] += perf_counter() - started
                if not smoke:
                    started = perf_counter()
                    live[axis, sign, scale] = model.render(shifted, model.colors(shifted))[0][0].cpu().numpy()
                    timing["live_probes"] += perf_counter() - started
    started = perf_counter()
    estimates = {"micro_fixed": depth_prediction(*inverse_depth_from_motion(center, fixed, focal, distance), coverage)}
    timing["fixed_solver"] = perf_counter() - started
    if not smoke:
        started = perf_counter()
        q10, q50, q90 = opacity_quantiles(metadata, camera.width, camera.height)
        timing["quantiles"] = perf_counter() - started
        spread = np.log(q90 / q10)
        conventional_score = np.nan_to_num(np.exp(-spread), nan=0.0) * coverage
        for name, depth in (("expected", expected), ("median", q50)):
            valid = np.isfinite(depth) & (depth > 0.15) & (depth < 12) & (coverage > 0.25)
            estimates[name] = (depth, np.where(valid, conventional_score, -np.inf))
        started = perf_counter()
        estimates["micro_live"] = depth_prediction(*inverse_depth_from_motion(center, live, focal, distance), coverage)
        timing["live_solver"] = perf_counter() - started
        started = perf_counter()
        estimates["virtual_stereo"] = depth_prediction(*virtual_stereo(center, fixed, focal, distance), coverage)
        timing["stereo_solver"] = perf_counter() - started
    return estimates, center, distance, timing
