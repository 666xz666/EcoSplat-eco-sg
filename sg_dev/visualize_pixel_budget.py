"""按像素统计预算去向并与渲染误差对照。

读取 ``sg_wrapper.pixel_dump`` 落盘的单个视角，投影每个基元、按椭圆足迹累加
「覆盖点数」「覆盖 lobe 数」「每像素 float 成本」，再与渲染误差做联合作图。

用法::

    python sg_dev/visualize_pixel_budget.py --dump-dir DIR --out DIR [--only SCENE]

每个 dump 出一张六联图与一行统计；统计里的 ``floats/px`` 用于和全局账本对账。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

FLOATS_PER_POINT = 14
FLOATS_PER_LOBE = 7
OPACITY_TAU = 0.01


def _load(path: str) -> dict:
    """读取单个 dump。

    Args:
        path: dump 路径。

    Returns:
        含张量的字典。
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _lobe_counts(sg_rgb: torch.Tensor | None, n_points: int) -> torch.Tensor:
    """统计每个点实际保留的 lobe 数。

    Args:
        sg_rgb: (N, L, 3) lobe 振幅；None 表示 SH 头或未记录。
        n_points: 点数。

    Returns:
        (N,) long，取值范围 0..L。
    """
    # 步骤1：预算模式把被剪 lobe 的振幅置零，所以「非零列数」就是保留数。
    if sg_rgb is None:
        return torch.ones(n_points, dtype=torch.long)
    return (sg_rgb.abs().sum(dim=-1) > 1e-8).sum(dim=-1).long()


def _project(means: torch.Tensor, covs: torch.Tensor, w2c: torch.Tensor, k_mat: torch.Tensor,
             height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把 3D 高斯投影到像素平面，返回中心与 2D 协方差。

    Args:
        means: (N, 3) 世界坐标中心。
        covs: (N, 3, 3) 世界坐标协方差。
        w2c: (4, 4) 世界到相机。
        k_mat: (3, 3) 归一化内参（f 与 c 都在 [0, 1]）。
        height: 图像高。
        width: 图像宽。

    Returns:
        (u, v, s2)：像素中心 (N, 2) 与 2D 协方差 (N, 2, 2)。
    """
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    cam = means @ rot.T + trans
    z = cam[:, 2].clamp_min(1e-6)

    # 步骤1：归一化内参换成像素单位，再做透视投影。
    fx, fy = k_mat[0, 0] * width, k_mat[1, 1] * height
    cx, cy = k_mat[0, 2] * width, k_mat[1, 2] * height
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy

    # 步骤2：透视雅可比 J 与旋转把 3D 协方差压成 2D（EWA）。
    jac = torch.zeros(means.shape[0], 2, 3, dtype=means.dtype)
    jac[:, 0, 0] = fx / z
    jac[:, 0, 2] = -fx * cam[:, 0] / z.pow(2)
    jac[:, 1, 1] = fy / z
    jac[:, 1, 2] = -fy * cam[:, 1] / z.pow(2)
    rotated = rot.unsqueeze(0) @ covs @ rot.T.unsqueeze(0)
    s2 = jac @ rotated @ jac.transpose(1, 2)
    # 步骤3：加一点对角噪声，避免退化协方差求逆爆炸。
    eye = torch.eye(2, dtype=means.dtype).unsqueeze(0) * 1e-6
    return u, v, s2 + eye


def _accumulate(u: torch.Tensor, v: torch.Tensor, s2: torch.Tensor, weight: torch.Tensor,
                height: int, width: int, radius: float) -> torch.Tensor:
    """按椭圆足迹把权重累加到像素上。

    Args:
        u: (N,) 像素中心 x。
        v: (N,) 像素中心 y。
        s2: (N, 2, 2) 2D 协方差。
        weight: (N,) 每个基元贡献的量（1 或 lobe 数或 float 成本）。
        height: 图像高。
        width: 图像宽。
        radius: 马氏距离阈值（像素半径按 3σ 取）。

    Returns:
        (H*W,) 累加结果。
    """
    inv = torch.linalg.inv(s2)
    cu = u.round().long()
    cv = v.round().long()
    half = int(np.ceil(torch.linalg.eigvalsh(s2).max().sqrt().item() * radius)) + 1
    half = max(1, min(half, 31))
    out = torch.zeros(height * width, dtype=weight.dtype)
    r2 = radius * radius
    # 步骤1：固定窗口逐偏移广播，全部是向量化算子，65k 基元 × 窗口 < 1 s。
    for dy in range(-half, half + 1):
        yy = cv + dy
        inside_y = (yy >= 0) & (yy < height)
        if not bool(inside_y.any()):
            continue
        for dx in range(-half, half + 1):
            xx = cu + dx
            ok = inside_y & (xx >= 0) & (xx < width)
            if not bool(ok.any()):
                continue
            du = (xx - u).to(weight.dtype)
            dv = (yy - v).to(weight.dtype)
            m2 = (
                inv[:, 0, 0] * du * du
                + 2 * inv[:, 0, 1] * du * dv
                + inv[:, 1, 1] * dv * dv
            )
            hit = ok & (m2 < r2)
            if bool(hit.any()):
                out.index_add_(0, (yy[hit] * width + xx[hit]), weight[hit])
    return out.reshape(height, width)


def _panel_stats(pts: torch.Tensor, lobes: torch.Tensor, floats: torch.Tensor,
                 err: torch.Tensor) -> dict:
    """汇总一个场景的像素级统计。

    Args:
        pts: (H, W) 每像素点数。
        lobes: (H, W) 每像素 lobe 数。
        floats: (H, W) 每像素 float 成本。
        err: (H, W) 逐像素误差（通道均值）。

    Returns:
        统计字典。
    """
    empty = (pts == 0).float().mean().item()
    total_err = err.sum().item()
    share_empty = err[pts == 0].sum().item() / max(total_err, 1e-9)
    return {
        "empty_frac": empty,
        "pts_mean": pts.mean().item(),
        "pts_p50": pts.median().item(),
        "pts_p95": torch.quantile(pts.flatten().float(), 0.95).item(),
        "pts_max": pts.max().item(),
        "lobes_mean": lobes.mean().item(),
        "floats_mean": floats.mean().item(),
        "floats_p95": torch.quantile(floats.flatten().float(), 0.95).item(),
        "err_mean": err.mean().item(),
        "err_share_empty": share_empty,
    }


def _plot(dump: dict, target: torch.Tensor, out_png: str) -> dict:
    """画六联图并返回统计。

    Args:
        dump: pixel_dump 落盘的字典。
        target: (3, H, W) 目标视图。
        out_png: 输出路径。

    Returns:
        统计字典。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = dump["image_shape"]
    means = dump["means"]
    covs = dump["covariances"]
    opac = dump["opacities"]
    u, v, s2 = _project(means, covs, dump["extrinsics"], dump["intrinsics"], height, width)
    valid = (u >= -1) & (u <= width) & (v >= -1) & (v <= height)

    m = _lobe_counts(dump.get("sg_rgb"), means.shape[0])
    keep = valid & (opac > OPACITY_TAU)
    ones = torch.ones(int(keep.sum()), dtype=torch.float32)
    pts = _accumulate(u[keep], v[keep], s2[keep], ones, height, width, radius=3.0)
    lobes = _accumulate(u[keep], v[keep], s2[keep], m[keep].float(), height, width, radius=3.0)
    floats = FLOATS_PER_POINT * pts + FLOATS_PER_LOBE * lobes

    rgb = dump["color"].permute(1, 2, 0).clamp(0, 1).numpy()
    gt = target.permute(1, 2, 0).clamp(0, 1).numpy()
    err = (dump["color"] - target).abs().mean(dim=0)

    stats = _panel_stats(pts, lobes, floats, err)
    # 对账：像素级口径的总 float 应等于「点数 × 14 + lobe 数 × 7」的全局账本。
    stats["floats_global"] = float((FLOATS_PER_POINT * keep.sum() + FLOATS_PER_LOBE * m[keep].sum()).item())
    stats["floats_pixel"] = float(floats.sum().item())

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("render")
    axes[0, 1].imshow(gt)
    axes[0, 1].set_title("target")
    im2 = axes[0, 2].imshow(err.numpy(), cmap="magma")
    axes[0, 2].set_title("|error| (per pixel)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    for ax, field, name in (
        (axes[1, 0], pts, "points / pixel"),
        (axes[1, 1], lobes, "lobes / pixel"),
        (axes[1, 2], floats, "floats / pixel (14N+7L)"),
    ):
        im = ax.imshow(field.numpy(), cmap="turbo", vmin=0, vmax=max(1.0, float(field.max())))
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{dump['scene']}  pts={stats['pts_mean']:.2f}/px  lobes={stats['lobes_mean']:.2f}/px  "
        f"floats={stats['floats_mean']:.1f}/px  empty={100 * stats['empty_frac']:.1f}%  "
        f"err={stats['err_mean']:.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return stats


def main() -> None:
    """遍历 dump 目录出图。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None, help="只处理指定场景 hash")
    args = parser.parse_args()

    scenes = sorted({f.split("_t")[0] for f in os.listdir(args.dump_dir) if "_t" in f})
    if args.only:
        scenes = [s for s in scenes if s == args.only]

    for scene in scenes:
        target_path = os.path.join(args.dump_dir, f"{scene}_target.pt")
        dumps = sorted(f for f in os.listdir(args.dump_dir) if f.startswith(scene + "_t"))
        if not dumps or not os.path.exists(target_path):
            print(f"[skip] {scene}: dump or target missing")
            continue
        target = _load(target_path)["target"][0]
        dump = _load(os.path.join(args.dump_dir, dumps[0]))
        out_png = os.path.join(args.out, f"{scene}.png")
        stats = _plot(dump, target, out_png)
        print(
            f"{scene}  pts={stats['pts_mean']:.2f} lobes={stats['lobes_mean']:.2f} "
            f"floats={stats['floats_mean']:.1f} (p95 {stats['floats_p95']:.0f}) "
            f"empty={100 * stats['empty_frac']:.1f}% err={stats['err_mean']:.4f} "
            f"err@empty={100 * stats['err_share_empty']:.1f}% "
            f"ledger={stats['floats_pixel']:.0f}/{stats['floats_global']:.0f}"
        )


if __name__ == "__main__":
    main()
