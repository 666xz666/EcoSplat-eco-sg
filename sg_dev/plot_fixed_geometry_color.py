"""汇总场景内的重复计时，并绘制高斯数和 SG 瓣数的分段耗时曲线。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PANELS = {
    "wall_ms": "Full frame (synchronized wall time)",
    "color": "Color evaluation",
    "raster_input": "Rasterizer input preparation",
    "project": "Gaussian projection",
    "tile_intersect_sort": "Tile intersection and sort",
    "tile_index": "Tile index construction",
    "composite": "Pixel alpha compositing",
    "raster_output": "Rasterizer output preparation",
}
STAGES = [stage for stage in PANELS if stage != "wall_ms"]


def summarize(profile: dict) -> list[dict]:
    """先取每个场景的重复中位数，再等权汇总不同场景。

    Args:
        profile (dict): 计时脚本保存的原始 JSON。

    Returns:
        list[dict]: 每个点数和颜色路径的中位耗时、可见数及 tile 相交数。
    """
    groups = defaultdict(list)
    for case in profile["cases"]:
        count = case["count"]
        for name, method in case["methods"].items():
            lobes = method["lobes_per_gaussian"]
            stages = {
                stage: median(event[stage] for event in method["events"])
                for stage in STAGES if stage != "color" or lobes != 0
            }
            if lobes == 0:
                stages["color"] = 0.0  # RGB/DC 的颜色已预存，渲染时没有求值调用。
            groups[count, name, lobes].append({
                "wall_ms": median(method["wall_ms"]),
                "gpu_ms": median(method["gpu_ms"]),
                "visible": case["visible"],
                "tile_intersections": case["tile_intersections"],
                **stages,
            })

    rows = []
    for (count, name, lobes), scenes in sorted(groups.items()):
        rows.append({
            "gaussians": count,
            "method": name,
            "lobes_per_gaussian": lobes,
            "total_sg_lobes": None if lobes is None else lobes * count,
            "scenes": len(scenes),
            **{key: median(scene[key] for scene in scenes) for key in scenes[0]},
        })
    return rows


def draw(rows: list[dict], x_key: str, series_key: str, filename: Path, label: str) -> None:
    """每个子图使用同一组实验配置，保留各阶段自己的毫秒尺度。

    Args:
        rows (list[dict]): 汇总后的曲线点。
        x_key (str): 横轴字段。
        series_key (str): 区分曲线的字段。
        filename (Path): PNG 保存路径。
        label (str): 横轴名称。
    """
    series = defaultdict(list)
    for row in rows:
        if row[x_key] is not None:
            series[row[series_key]].append(row)
    if not series:
        raise ValueError(f"no measurements for {x_key}")

    x_values = sorted({row[x_key] for items in series.values() for row in items})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    for ax, (key, title) in zip(axes.flat, PANELS.items()):
        for name, items in series.items():
            ordered = sorted(items, key=lambda row: row[x_key])
            series_name = f"{name:,} Gaussians" if series_key == "gaussians" else name
            ax.plot(
                [row[x_key] for row in ordered], [row[key] for row in ordered],
                label=series_name, marker="o", markersize=4, linewidth=1.7,
                linestyle="--" if name == "SH4" else "-",
            )
        ax.set_title(title)
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel(label)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x_values)
        ax.set_xticklabels([f"{x / 1024:g}K" if x_key == "gaussians" else str(x) for x in x_values])
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main() -> None:
    """读取扫描 JSON，保存可追溯的汇总点和两张多阶段曲线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(json.loads(args.profile.read_text(encoding="utf-8")))
    if not rows:
        parser.error("the profile has no measured cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "latency_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, "gaussians", "method", args.output_dir / "latency_by_gaussians.png", "Gaussians")
    draw(rows, "lobes_per_gaussian", "gaussians", args.output_dir / "latency_by_sg_lobes.png",
         "SG lobes per Gaussian")
    print("PLOTS_COMPLETE", args.output_dir)


if __name__ == "__main__":
    main()
