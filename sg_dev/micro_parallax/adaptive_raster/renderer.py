"""冻结 3DGS 的原版、Gaussian Blending、SSAA 和自适应 CUDA 渲染。"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles, spherical_harmonics
from torch.utils.cpp_extension import load


TILE_SIZE = 16
NEAR = 0.2
FAR = 100.0
RAW_EPSILON = 1e-6


@lru_cache(maxsize=1)
def adaptive_extension():
    """加载实验目录中的 RTX 3090 自适应光栅扩展。"""
    root = Path(__file__).resolve().parent
    return load(
        name="ecosplat_adaptive_subpixel_cuda",
        sources=[str(root / "bindings.cpp"), str(root / "kernels.cu")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


def resized_intrinsic(intrinsic, original_width, original_height, width, height, offset=(0, 0)):
    """按像素中心缩放内参，支持同一像素内的采样相位。"""
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    resize = np.asarray([width / original_width, height / original_height], dtype=np.float32)
    scaled[:2, :2] *= resize[:, None]
    scaled[:2, 2] = (scaled[:2, 2] + .5) * resize - .5 - np.asarray(offset, dtype=np.float32)
    return scaled


def clip_matrix(K, width, height):
    """使 Gaussian Blending 与 gsplat 更新后的针孔投影对齐。"""
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2 * K[0, 0] / width
    projection[1, 1] = 2 * K[1, 1] / height
    projection[0, 2] = 2 * (K[0, 2] + .5) / width - 1
    projection[1, 2] = 2 * (K[1, 2] + .5) / height - 1
    projection[2, 2] = FAR / (FAR - NEAR)
    projection[2, 3] = -(FAR * NEAR) / (FAR - NEAR)
    projection[3, 2] = 1
    return projection


@dataclass
class View:
    """常驻 GPU 的相机与各采样倍率内参，构造开销位于渲染计时区间外。"""

    world_to_camera: torch.Tensor
    center: torch.Tensor
    original_intrinsic: np.ndarray
    original_width: int
    original_height: int
    width: int
    height: int
    torch_device: torch.device

    @classmethod
    def from_camera(cls, camera, width, height, device):
        return cls(
            torch.as_tensor(np.linalg.inv(camera.c2w), dtype=torch.float32, device=device),
            torch.as_tensor(camera.c2w[:3, 3].copy(), dtype=torch.float32, device=device),
            camera.intrinsic, camera.width, camera.height, width, height, torch.device(device),
        )

    def K(self, width, height, offset=(0, 0)):
        return resized_intrinsic(
            self.original_intrinsic, self.original_width, self.original_height, width, height, offset,
        )

    def gsplat_intrinsic(self, width, height, offset=(0, 0)):
        K = self.K(width, height, offset)
        K[:2, 2] += .5
        return torch.as_tensor(K, device=self.torch_device)[None]


class Renderer:
    """统一检查点、黑色背景与 SH 颜色，比较从场景参数到 RGB 输出的帧延迟。"""

    def __init__(self, model, view, epsilon=RAW_EPSILON):
        self.model = model
        self.view = view
        self.epsilon = epsilon
        self.w, self.h = view.width, view.height
        self.viewmat = view.world_to_camera[None]
        self.tile_width = (self.w + TILE_SIZE - 1) // TILE_SIZE
        self.tile_height = (self.h + TILE_SIZE - 1) // TILE_SIZE
        self.tiles = self.tile_width * self.tile_height
        self.all_tiles = torch.arange(self.tiles, device=model.xyz.device)
        self.K = view.gsplat_intrinsic(self.w, self.h)
        self._gb = None
        self._ssaa = {}
        self._ext = None

    def colors(self):
        """每次渲染都执行同一个视角相关 SH 颜色计算。"""
        m = self.model
        return (spherical_harmonics(m.degree, m.xyz - self.view.center, m.sh) + .5).clamp_min(0)

    def gsplat(self, colors, K, width, height, epsilon):
        """以同一 pinhole 模型与裁剪平面渲染。"""
        m = self.model
        return rasterization(
            m.xyz, m.rotation, m.scale, m.opacity, colors,
            self.viewmat, K, width, height, near_plane=NEAR, far_plane=FAR,
            eps2d=epsilon, packed=False, render_mode="RGB",
        )[0][0]

    def reference(self, order):
        """直接积分完整 Alpha 合成。每张参考图共用相同 SH 颜色。"""
        colors = self.colors()
        nodes, weights = np.polynomial.legendre.leggauss(order)
        result = torch.zeros((self.h, self.w, 3), dtype=colors.dtype, device=colors.device)
        for i in range(order):
            for j in range(order):
                K = self.view.gsplat_intrinsic(self.w, self.h, (.5 * nodes[i], .5 * nodes[j]))
                image = self.gsplat(colors, K, self.w, self.h, self.epsilon)
                result.add_(image, alpha=float(.25 * weights[i] * weights[j]))
        return result

    def supersample(self, scale):
        """高分辨率执行同一个 rasterizer，并把完整 RGB 下采样到目标尺寸。"""
        if scale not in self._ssaa:
            width, height = round(self.w * scale), round(self.h * scale)
            K = self.view.gsplat_intrinsic(width, height)
            factor = (width * height) / (self.w * self.h)
            self._ssaa[scale] = (width, height, K, self.epsilon * factor)
        width, height, K, eps = self._ssaa[scale]
        out = self.gsplat(self.colors(), K, width, height, eps)
        if (width, height) == (self.w, self.h):
            return out
        pooled = F.adaptive_avg_pool2d(out.permute(2, 0, 1)[None], (self.h, self.w))
        return pooled[0].permute(1, 2, 0).contiguous()

    def gaussian_blending(self):
        """直接调用官方推理扩展，给定与其他渲染器相同的每视角颜色。"""
        from gaussian_blending import GaussianRasterizationSettings, GaussianRasterizer

        if self._gb is None:
            v = self.view
            K = v.K(self.w, self.h)
            projection = torch.as_tensor(clip_matrix(K, self.w, self.h), device=v.torch_device)
            view = v.world_to_camera.T.contiguous()
            full = (view @ projection.T).contiguous()
            settings = GaussianRasterizationSettings(
                image_height=self.h, image_width=self.w,
                tanfovx=float(.5 * self.w / K[0, 0]), tanfovy=float(.5 * self.h / K[1, 1]),
                bg=self.model.xyz.new_zeros(3), scale_modifier=1.,
                viewmatrix=view, projmatrix=full, sh_degree=self.model.degree, campos=v.center,
                prefiltered=False, debug=False,
            )
            self._gb = (GaussianRasterizer(settings), self.model.xyz.new_zeros(self.model.xyz.shape))
        raster, screen_means = self._gb
        m = self.model
        rgb, _ = raster(
            means3D=m.xyz, means2D=screen_means, colors_precomp=self.colors(), opacities=m.opacity[:, None],
            scales=m.scale, rotations=m.rotation,
        )
        return rgb.permute(1, 2, 0).contiguous()

    def project(self):
        """一次投影与排序供解析底图及选中图块复用。"""
        ext = self._ext
        m = self.model
        radii, means, depths, conics, _ = fully_fused_projection(
            m.xyz, None, m.rotation, m.scale, self.viewmat, self.K,
            self.w, self.h, eps2d=self.epsilon, near_plane=NEAR, far_plane=FAR,
            opacities=m.opacity, packed=False,
        )
        footprints, padded = ext.prepare_footprints(conics, m.opacity, radii)
        _, keys, ids = isect_tiles(means, padded, depths, TILE_SIZE, self.tile_width, self.tile_height)
        offsets = isect_offset_encode(keys, 1, self.tile_width, self.tile_height)
        return means, self.colors(), footprints, offsets, ids

    def adaptive(self, fraction, *, corrected=False, selector="correlation"):
        """按 Alpha 交叉项分配四节点采样，比例固定避免同步 CPU。"""
        if self._ext is None:
            self._ext = adaptive_extension()
        frame = self.project()
        ext = self._ext
        if fraction == 1.0:
            image = self.model.xyz.new_empty((self.h, self.w, 3))
            ext.render_selected(*frame, self.all_tiles, image, self.w, self.h)
            return image
        image, score = ext.render_base(*frame, self.w, self.h, corrected)
        count = round(self.tiles * fraction)
        if not count:
            return image
        if selector == "gradient":
            score = self.gradient_score(image, frame[3], frame[4].numel())
        indices = torch.topk(score, count, sorted=False).indices
        ext.render_selected(*frame, indices, image, self.w, self.h)
        return image

    def gradient_score(self, image, offsets, entries):
        """与交叉项选区共用底图、候选列表与四节点采样，对照图像梯度。"""
        gx = F.pad((image[:, 1:] - image[:, :-1]).square().sum(-1), (1, 0))
        gy = F.pad((image[1:] - image[:-1]).square().sum(-1), (0, 0, 1, 0))
        delta = gx + gy
        padded = F.pad(delta, (0, self.tile_width * TILE_SIZE - self.w,
                               0, self.tile_height * TILE_SIZE - self.h))
        energy = F.avg_pool2d(padded[None, None], TILE_SIZE).flatten()
        bounds = torch.cat((offsets.flatten(), offsets.new_tensor([entries])))
        return energy / (bounds[1:] - bounds[:-1]).clamp_min(1)
