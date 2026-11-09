"""诊断 PSNR 与 SSIM/LPIPS 分化的三个来源。

三件事：

1. RGB 平均残差：区分整幅偏色与局部残差，并看「扣掉全图平均偏差」后 MSE 能降多少；
2. 平坦区与边缘区的误差占比：判断误差是被大面积平坦区（颜色/亮度/合成）主导，
   还是被边缘（错位、重影）主导；
3. 逐场景配对分布：确认「PSNR 低但 LPIPS 好」是多数场景的普遍现象，还是少数
   大误差场景把均值拉下来。

前两件用 ``pixel_dump`` 落盘的渲染与目标图，第三件用两侧的 ``scores_all.json``。

用法::

    python sg_dev/diagnose_psnr_split.py --dump-root DIR --scores-root outputs/test
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

# 配对：同字节的 (我方 run, Eco run)
PAIRS = [
    ("budget50k_rho0p12282", "eco200_rho0p05", 4.30),
    ("budget50k_rho0p24569", "eco200_rho0p10", 8.60),
]


def _load(path: str) -> dict:
    """读一个 dump。

    Args:
        path: .pt 路径。

    Returns:
        含张量的字典。
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def _edge_mask(target: np.ndarray, thresh: float = 0.06) -> np.ndarray:
    """用目标图的梯度模长划分边缘与平坦区。

    Args:
        target: (3, H, W) 目标图，[0, 1]。
        thresh: 梯度模长阈值。

    Returns:
        (H, W) 布尔的边缘掩码。
    """
    lum = target.mean(axis=0)
    gy, gx = np.gradient(lum)
    return np.sqrt(gx ** 2 + gy ** 2) > thresh


def bias_report(dump_dir: str) -> list[dict]:
    """逐场景算偏色、去偏后的 MSE、平坦/边缘分区误差。

    Args:
        dump_dir: 一个预算档的 dump 目录。

    Returns:
        每场景一行统计。
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "*_t*.pt"))):
        if path.endswith("_target.pt"):
            continue
        dump = _load(path)
        scene = dump["scene"]
        target_path = os.path.join(dump_dir, f"{scene}_target.pt")
        if not os.path.exists(target_path):
            continue
        target = _load(target_path)["target"][0].numpy()
        # 取与该 dump 同帧的目标图：dump 的 target 是整组，按顺序取第一个
        render = dump["color"].numpy()
        if target.shape != render.shape:
            continue

        resid = render - target
        # 步骤1：整幅（每通道）平均偏差，等价于把渲染整体平移后重算误差。
        bias = resid.reshape(3, -1).mean(axis=1)
        mse_raw = float((resid ** 2).mean())
        mse_debiased = float(((resid - bias[:, None, None]) ** 2).mean())
        # 步骤2：单一亮度偏移（三通道共用一个数）能解释多少。
        bias_scalar = float(resid.mean())
        mse_scalar = float(((resid - bias_scalar) ** 2).mean())

        # 步骤3：平坦 / 边缘分区，按目标图梯度划分。
        edge = _edge_mask(target)
        sq = (resid ** 2).mean(axis=0)
        flat_sum = float(sq[~edge].sum())
        edge_sum = float(sq[edge].sum())
        rows.append(
            {
                "scene": scene,
                "bias_r": float(bias[0]),
                "bias_g": float(bias[1]),
                "bias_b": float(bias[2]),
                "mse_raw": mse_raw,
                "mse_debiased": mse_debiased,
                "mse_scalar": mse_scalar,
                "edge_frac": float(edge.mean()),
                "mse_flat": float(sq[~edge].mean()),
                "mse_edge": float(sq[edge].mean()),
                "sse_share_flat": flat_sum / max(flat_sum + edge_sum, 1e-12),
            }
        )
    return rows


def paired_report(scores_root: str) -> None:
    """打印同字节配对分布：均值、中位、符号一致比例、离群场景。

    Args:
        scores_root: ``outputs/test`` 根目录。

    Returns:
        None
    """
    for ours, eco, fpp in PAIRS:
        ours_path = os.path.join(scores_root, ours, ours, "scores_all.json")
        eco_path = os.path.join(scores_root, eco, eco, "scores_all.json")
        if not (os.path.exists(ours_path) and os.path.exists(eco_path)):
            print(f"[paired] missing json for {ours} / {eco}")
            continue
        a = json.load(open(ours_path))
        b = json.load(open(eco_path))
        common = sorted(set(a) & set(b))
        d_psnr = np.array([a[s][1] - b[s][1] for s in common])
        d_ssim = np.array([a[s][2] - b[s][2] for s in common])
        d_lpips = np.array([a[s][3] - b[s][3] for s in common])
        order = np.argsort(d_psnr)
        worst = [(common[i], round(float(d_psnr[i]), 2)) for i in order[:5]]
        best = [(common[i], round(float(d_psnr[i]), 2)) for i in order[-5:]]
        # 去掉最差 5% 后的均值，判断均值是否被少数场景拖低
        k = max(1, int(0.05 * len(d_psnr)))
        trimmed = d_psnr[order[k:]].mean()
        print(
            f"[paired] {fpp:.2f} f/px  n={len(common)}  "
            f"dPSNR mean={d_psnr.mean():+.3f} median={np.median(d_psnr):+.3f} "
            f"trim5%={trimmed:+.3f}  dSSIM={d_ssim.mean():+.4f} dLPIPS={d_lpips.mean():+.4f}"
        )
        win_psnr = float((d_psnr > 0).mean())
        win_lpips = float((d_lpips < 0).mean())
        both = float(((d_psnr < 0) & (d_lpips < 0)).mean())
        print(
            f"           psnr win {100 * win_psnr:.1f}%  lpips win {100 * win_lpips:.1f}%  "
            f"psnr lose but lpips win {100 * both:.1f}%"
        )
        print(f"           worst: {worst}")
        print(f"           best : {best}")


def paired_bias(ours_dir: str, eco_dir: str, label: str) -> None:
    """同一场景、同一 target、同一预算下，比对两侧的偏置与去偏误差。

    Args:
        ours_dir: 我方 dump 目录。
        eco_dir: Eco dump 目录。
        label: 该档的字节标签，仅用于打印。

    Returns:
        None
    """
    a = {r["scene"]: r for r in bias_report(ours_dir)}
    b = {r["scene"]: r for r in bias_report(eco_dir)}
    common = sorted(set(a) & set(b))
    if not common:
        print(f"[pair-bias] {label}: no common scenes")
        return
    print(f"\n== {label}  ({len(common)} scenes) ==")
    print(
        f"{'scene':18s} {'ours mse':>9s} {'eco mse':>9s} {'ours bias':>9s} {'eco bias':>9s} "
        f"{'ours de-b':>9s} {'eco de-b':>9s} {'flat%':>6s} {'flat mse o/e':>14s}"
    )
    d_mse, d_deb, d_bias = [], [], []
    for s in common:
        ra, rb = a[s], b[s]
        bias_a = float(np.mean([ra["bias_r"], ra["bias_g"], ra["bias_b"]]))
        bias_b = float(np.mean([rb["bias_r"], rb["bias_g"], rb["bias_b"]]))
        d_mse.append(ra["mse_raw"] - rb["mse_raw"])
        d_deb.append(ra["mse_debiased"] - rb["mse_debiased"])
        d_bias.append(abs(bias_a) - abs(bias_b))
        print(
            f"{s:18s} {ra['mse_raw']:9.4f} {rb['mse_raw']:9.4f} "
            f"{bias_a:+9.3f} {bias_b:+9.3f} {ra['mse_debiased']:9.4f} {rb['mse_debiased']:9.4f} "
            f"{100 * (1 - ra['edge_frac']):6.1f} "
            f"{ra['mse_flat']:6.4f}/{rb['mse_flat']:.4f}"
        )
    print(
        f"mean Δmse(raw)={np.mean(d_mse):+.4f}  Δmse(debiased)={np.mean(d_deb):+.4f}  "
        f"Δ|bias|={np.mean(d_bias):+.4f}"
    )


def main() -> None:
    """跑三项诊断。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-root", default="/data14T/xz/figs/pixel_dump")
    parser.add_argument("--scores-root", default="outputs/test")
    parser.add_argument("--skip-dumps", action="store_true")
    args = parser.parse_args()

    if not args.skip_dumps:
        for dump_dir in sorted(glob.glob(os.path.join(args.dump_root, "rho_*"))):
            rows = bias_report(dump_dir)
            if not rows:
                continue
            print(f"\n== {os.path.basename(dump_dir)}  ({len(rows)} scenes) ==")
            for r in rows:
                print(
                    f"{r['scene']}  bias rgb=({r['bias_r']:+.3f},{r['bias_g']:+.3f},{r['bias_b']:+.3f})  "
                    f"mse raw={r['mse_raw']:.4f} debiased={r['mse_debiased']:.4f} "
                    f"scalar={r['mse_scalar']:.4f}  edge={100 * r['edge_frac']:.1f}%  "
                    f"mse flat={r['mse_flat']:.4f} edge={r['mse_edge']:.4f}  "
                    f"sse_share_flat={100 * r['sse_share_flat']:.1f}%"
                )
            mse_raw = np.mean([r["mse_raw"] for r in rows])
            mse_deb = np.mean([r["mse_debiased"] for r in rows])
            print(
                f"mean mse {mse_raw:.4f} -> debiased {mse_deb:.4f} "
                f"({100 * (1 - mse_deb / mse_raw):.1f}% of MSE is a per-channel offset)"
            )

    paired_report(args.scores_root)


if __name__ == "__main__":
    main()
