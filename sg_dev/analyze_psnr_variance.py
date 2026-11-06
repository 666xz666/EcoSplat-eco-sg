"""Decompose the per-sample PSNR spread of an IGF run into its sources.

The five-interval panels show a wide band inside every protect-rate interval.
This script answers where that band comes from: the scene being trained on, the
residual budget variation inside the interval, or the training step.  It reads
the same log as the plotting scripts, so no extra bookkeeping is needed.

The decomposition is deliberately simple: an interval's total variance is split
by (1) the share explained by the scene identity, estimated on scenes that were
visited at least twice, and (2) the slope on log budget, estimated after
removing the scene means.  Both are reported next to the raw spread so the
numbers can be checked by hand.

Args:
    log: training log written by ``train_pgt_bg.sh``.
    edges: interval boundaries, defaults to the five reference intervals.

Returns:
    None; prints one block per interval plus the repeat-visit and run-length
    diagnostics.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

LINE = re.compile(
    r"train step (\d+).*?loss = ([\d.eE+-]+); psnr = ([\d.eE+-]+); protect_rate = ([\d.eE+-]+)"
)
SCENE = re.compile(r"scene = \['([0-9a-fA-F]+)'\]")
LEVELS = ["极低预算", "低预算", "中预算", "高预算", "接近满配"]


def parse(path: Path) -> dict[str, np.ndarray]:
    """Parse step, psnr, protect rate and scene id out of one log.

    Args:
        path: log file.

    Returns:
        Dict of arrays ``step`` (int), ``psnr``, ``rho`` (float) and ``scene``
        (object, hashed scene id); one entry per logged micro-batch.
    """
    steps, psnrs, rhos, scenes = [], [], [], []
    for line in path.read_text(errors="ignore").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        scene = SCENE.search(line)
        if not scene:
            continue
        steps.append(int(match.group(1)))
        psnrs.append(float(match.group(3)))
        rhos.append(float(match.group(4)))
        scenes.append(scene.group(1))
    if not steps:
        raise SystemExit(f"no usable lines in {path}")
    return {
        "step": np.asarray(steps, dtype=np.int64),
        "psnr": np.asarray(psnrs),
        "rho": np.asarray(rhos),
        "scene": np.asarray(scenes),
    }


def split_scene_variance(psnr: np.ndarray, scene: np.ndarray, min_count: int = 2) -> dict:
    """Estimate how much of the spread is between scenes rather than inside them.

    Args:
        psnr: PSNR per sample.
        scene: scene id per sample.
        min_count: minimum visits for a scene to enter the estimate; one-visit
            scenes have no within-scene spread to compare against.

    Returns:
        Dict with ``n`` (samples used), ``n_scenes``, ``sd_total``,
        ``sd_within`` (root mean within-scene variance), ``eta2`` (between /
        total) and ``repeat_pairs`` (same-scene pairs available).
    """
    counts = Counter(scene.tolist())
    keep = np.array([counts[s] >= min_count for s in scene])
    values, ids = psnr[keep], scene[keep]
    if values.size < 10:
        return {"n": 0, "n_scenes": 0, "sd_total": float("nan"), "sd_within": float("nan"),
                "eta2": float("nan"), "repeat_pairs": 0}
    total = float(np.var(values))
    ss_within = 0.0
    pairs = 0
    for key in np.unique(ids):
        part = values[ids == key]
        ss_within += float(part.size) * float(np.var(part))
        pairs += part.size * (part.size - 1) // 2
    sd_within = float(np.sqrt(ss_within / values.size))
    return {
        "n": int(values.size),
        "n_scenes": int(np.unique(ids).size),
        "sd_total": float(np.sqrt(total)),
        "sd_within": sd_within,
        "eta2": float(1.0 - (ss_within / values.size) / total) if total > 0 else float("nan"),
        "repeat_pairs": int(pairs),
    }


def slope_on_log_budget(psnr: np.ndarray, rho: np.ndarray, scene: np.ndarray) -> tuple[float, float]:
    """Fit PSNR against log10(budget) after removing the scene means.

    Args:
        psnr: PSNR per sample.
        rho: protect rate per sample, proportional to the budget.
        scene: scene id per sample.

    Returns:
        ``(slope_db_per_decade, r2)`` of the within-scene regression.
    """
    log_rho = np.log10(rho)
    uniq, inverse = np.unique(scene, return_inverse=True)
    mean_x = np.array([log_rho[inverse == i].mean() for i in range(uniq.size)])
    mean_y = np.array([psnr[inverse == i].mean() for i in range(uniq.size)])
    x = log_rho - mean_x[inverse]
    y = psnr - mean_y[inverse]
    denom = float(x @ x)
    if denom <= 0:
        return float("nan"), float("nan")
    slope = float(x @ y) / denom
    resid = y - slope * x
    ss_tot = float(y @ y)
    return slope, float(1.0 - float(resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")


def main() -> None:
    """Run the decomposition and print one block per protect-rate interval."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--edges", default="0.05,0.20,0.40,0.60,0.80,1.00")
    args = parser.parse_args()

    edges = np.array([float(x) for x in args.edges.split(",")])
    data = parse(args.log)
    bucket = np.clip(np.searchsorted(edges, data["rho"], side="right") - 1, 0, 4)

    # Consecutive micro-batches come from the same shard, and a shard holds
    # 10-17 scenes of one recording, so neighbouring samples share the scene.
    same_next = np.mean(data["scene"][1:] == data["scene"][:-1])
    print(f"samples={data['step'].size}  scenes={np.unique(data['scene']).size}  "
          f"consecutive_same_scene={same_next:.1%}")

    for index in range(5):
        mask = bucket == index
        if not mask.any():
            print(f"\n[{edges[index]:.2f},{edges[index + 1]:.2f}) 无样本")
            continue
        part = {key: values[mask] for key, values in data.items()}
        info = split_scene_variance(part["psnr"], part["scene"])
        print(f"\n[{edges[index]:.2f},{edges[index + 1]:.2f}) {LEVELS[index]}  n={part['psnr'].size}")
        print(f"  psnr: mean={part['psnr'].mean():.2f}  sd={part['psnr'].std():.2f}  "
              f"p5-p95={np.percentile(part['psnr'], 5):.1f}..{np.percentile(part['psnr'], 95):.1f}")
        if info["n"]:
            slope, r2 = slope_on_log_budget(part["psnr"], part["rho"], part["scene"])
            print(f"  重复场景 {info['n_scenes']} 个 / {info['n']} 样本："
                  f"sd_total={info['sd_total']:.2f}  sd_within_scene={info['sd_within']:.2f}  "
                  f"口径内场景解释 {info['eta2']:.0%}")
            print(f"  桶内预算斜率 {slope:.2f} dB/decade（去场景均值后 R²={r2:.2f}），"
                  f"桶内 rho 跨度 {part['rho'].min():.3f}..{part['rho'].max():.3f}")
        else:
            print("  重复场景样本不足，跳过方差分解")


if __name__ == "__main__":
    main()
