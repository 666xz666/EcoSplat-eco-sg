"""Render IGF training loss split by protect rate as a five-panel figure.

Each protect-rate interval gets its own panel with the single micro-batch
losses as scatter and a rolling median on top, and the sixth panel summarises
the per-interval median as horizontal bars.  The layout is meant to answer one
question at a glance: for a fixed budget, how far has the run come, and how
much loss does a narrower budget cost.

Args:
    log: training log written by ``train_pgt_bg.sh``; lines must carry
        ``train step N ... loss = X; psnr = Y; protect_rate = R``.
    out: destination png, a csv with the parsed samples lands next to it.
    edges: interval boundaries, defaults to the five reference intervals.
    floor_step: step at which the protect-rate floor is reached; drawn as a
        grey dashed line, pass 0 to hide it.
    metric: ``loss`` or ``psnr``; the two share the layout so a run can be read
        either as "how much error" or as "how many dB", never both in one file.

Returns:
    None; writes ``out``, ``out.with_suffix('.csv')`` and prints the summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

LINE = re.compile(
    r"train step (\d+).*?loss = ([\d.eE+-]+); psnr = ([\d.eE+-]+); protect_rate = ([\d.eE+-]+)"
)

# The reference figure labels the intervals by budget level, not by number, so
# keep the wording stable and only let the edges move.
LEVELS = ["极低预算", "低预算", "中预算", "高预算", "接近满配"]
BAR_COLOURS = ["#c7dcef", "#9dc3e6", "#6aa9d6", "#3f8fc4", "#1f6fae"]
SCATTER = "#9ecae1"
MEDIAN = "#d62728"
WINDOW = 9


def setup_font() -> None:
    """Pick the first installed CJK font so the labels render.

    Returns:
        None.
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_log(path: Path) -> dict[str, np.ndarray]:
    """Parse one training log into step, loss, psnr and protect rate arrays.

    Args:
        path: log file.

    Returns:
        Dict with float arrays ``step, loss, psnr, rho`` sorted by step; one
        entry per logged micro-batch (four per optimizer step).
    """
    steps, losses, psnrs, rhos = [], [], [], []
    for line in path.read_text(errors="ignore").splitlines():
        match = LINE.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            psnrs.append(float(match.group(3)))
            rhos.append(float(match.group(4)))
    if not steps:
        raise SystemExit(f"no training lines found in {path}")
    step = np.asarray(steps, dtype=np.int64)
    order = np.argsort(step, kind="stable")
    return {
        "step": step[order],
        "loss": np.asarray(losses)[order],
        "psnr": np.asarray(psnrs)[order],
        "rho": np.asarray(rhos)[order],
    }


def collapse_steps(step: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the micro-batches of one optimizer step into one median value.

    Args:
        step: step index per logged micro-batch, sorted.
        loss: loss per logged micro-batch.

    Returns:
        ``(unique_steps, median_loss)`` with one entry per step.
    """
    unique, inverse = np.unique(step, return_inverse=True)
    medians = np.array([np.median(loss[inverse == i]) for i in range(unique.size)])
    return unique, medians


def rolling_median(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Median filter with a centred window that shrinks at the two ends.

    Args:
        values: series ordered by step.
        window: number of steps in the window, 9 by default.

    Returns:
        Filtered series of the same length.
    """
    half = window // 2
    out = np.empty_like(values)
    for i in range(values.size):
        out[i] = np.median(values[max(0, i - half) : i + half + 1])
    return out


def third_medians(series: np.ndarray) -> tuple[float, float]:
    """Return the median of the first and of the last third of a series.

    Args:
        series: losses ordered by training step.

    Returns:
        ``(head, tail)`` medians; both equal the overall median for short runs.
    """
    cut = max(1, series.size // 3)
    return float(np.median(series[:cut])), float(np.median(series[-cut:]))


def draw_interval(
    ax,
    data: dict,
    low: float,
    high: float,
    level: str,
    floor_step: int,
    xmax: float,
    metric: str = "loss",
    ylim: tuple[float, float] | None = None,
) -> dict:
    """Draw one protect-rate interval: scatter, rolling median and two medians.

    Args:
        ax: target axes.
        data: rows inside this interval (dict of arrays).
        low: lower edge of the interval, shown in the title.
        high: upper edge of the interval, shown in the title.
        level: budget wording shown next to the interval.
        floor_step: step of the grey dashed line, 0 to hide it.
        xmax: shared right end of the step axis, so panels line up vertically.
        metric: ``loss`` or ``psnr``, decides the y values and the number format.
        ylim: shared y range; the five panels must be readable against each
            other, so the caller computes it once over all intervals.

    Returns:
        Summary dict with ``n``, ``head``, ``tail`` and ``median``.
    """
    steps, values = collapse_steps(data["step"], data[metric])
    smooth = rolling_median(values)
    digits = 4 if metric == "loss" else 2
    unit = "" if metric == "loss" else " dB"

    # Step 1/3: the scatter keeps the spread visible — a run still fighting one
    # hard scene looks very different from one that is simply flat.
    ax.scatter(data["step"], data[metric], s=5, c=SCATTER, alpha=0.55, linewidths=0)

    # Step 2/3: the rolling median is what the eye should follow; it is computed
    # per step first so that four micro-batches do not outvote a single step.
    ax.plot(steps, smooth, color=MEDIAN, lw=1.3, zorder=3)

    head, tail = third_medians(values)
    ax.text(0.03, 0.93, f"首段中位 {head:.{digits}f}{unit}", transform=ax.transAxes, color=MEDIAN, fontsize=9)
    ax.text(0.97, 0.93, f"末段中位 {tail:.{digits}f}{unit}", transform=ax.transAxes, color=MEDIAN, fontsize=9, ha="right")

    if floor_step:
        ax.axvline(floor_step, color="0.6", ls="--", lw=1.0, zorder=1)

    ax.set_title(f"ρ∈[{low:.2f},{high:.2f})　{level}　n={data['step'].size}", fontsize=10)
    if ylim is None:
        ax.set_ylim(0, 0.06)
        ax.set_yticks(np.arange(0, 0.061, 0.01))
    else:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, xmax)
    return {
        "n": int(data["step"].size),
        "head": head,
        "tail": tail,
        "median": float(np.median(data[metric])),
    }


def draw_summary(ax, stats: list[dict], edges: np.ndarray, metric: str = "loss") -> None:
    """Draw the horizontal bars with the median value of every interval.

    Args:
        ax: target axes.
        stats: one summary dict per interval.
        edges: bucket edges, used for the y tick labels.
        metric: ``loss`` or ``psnr``; only changes the wording and the number
            format, the ordering stays "high budget on top".

    Returns:
        None.
    """
    order = list(range(len(stats)))[::-1]  # highest budget on top
    values = [stats[i]["median"] for i in order]
    labels = [f"{LEVELS[i]} [{edges[i]:.2f},{edges[i + 1]:.2f})" for i in order]
    colours = [BAR_COLOURS[i] for i in order]
    digits = 4 if metric == "loss" else 2
    unit = "" if metric == "loss" else " dB"

    ax.barh(np.arange(len(order)), values, color=colours, height=0.62)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=9)
    span = max(values) * 1.55 if max(values) > 0 else 1.0
    for pos, i in enumerate(order):
        value = stats[i]["median"]
        if stats[i]["n"] == 0:
            ax.text(span * 0.01, pos, "无样本", va="center", fontsize=8.5, color="0.5")
            continue
        ax.text(value + span * 0.01, pos, f"{value:.{digits}f}{unit} (n={stats[i]['n']})", va="center", fontsize=8.5)
    # A loss bar is meaningless away from zero, but a PSNR bar is meaningless at
    # zero (25 vs 32 dB would look identical), so the dB axis starts just under
    # the smallest bar and says so in the label.
    if metric == "loss":
        ax.set_xlim(0, span)
        ax.set_xlabel("中位 loss（单微批）")
    else:
        low = min(values) - 1.0 if values else 0.0
        ax.set_xlim(low, max(values) + (max(values) - low) * 0.55)
        ax.set_xlabel(f"中位 PSNR（dB，轴从 {low:.1f} dB 起）")
    ax.set_title(f"各区间 {metric} 中位数（全程）", fontsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    """Build the figure described by the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default=None, help="run name used in the title")
    parser.add_argument(
        "--edges",
        default="0.05,0.20,0.40,0.60,0.80,1.00",
        help="five interval boundaries, low to high",
    )
    parser.add_argument("--floor-step", type=int, default=16000)
    parser.add_argument("--metric", choices=("loss", "psnr"), default="loss")
    args = parser.parse_args()

    edges = np.array([float(x) for x in args.edges.split(",")])
    if edges.size != 6:
        raise SystemExit("--edges needs six boundaries for five intervals")

    setup_font()
    data = parse_log(args.log)
    metric = args.metric
    # One shared y range for the five panels: with per-panel autoscaling a
    # converged bucket and a fresh bucket look equally busy, which hides exactly
    # the difference the figure exists to show.
    if metric == "loss":
        ylim = None
    else:
        low = float(np.percentile(data["psnr"], 1)) - 1.0
        high = float(np.percentile(data["psnr"], 99)) + 1.0
        ylim = (np.floor(low), np.ceil(high))
    bucket = np.clip(np.searchsorted(edges, data["rho"], side="right") - 1, 0, 4)
    # All five panels share the step axis so a vertical slice stays comparable;
    # the floor line is always inside the frame even before training reaches it.
    xmax = max(args.floor_step, int(data["step"].max())) * 1.02

    fig = plt.figure(figsize=(20, 9.8), dpi=100)
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.22)
    stats = []
    for index in range(5):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(grid[row, col])
        mask = bucket == index
        part = {key: values[mask] for key, values in data.items()}
        if not mask.any():
            ax.set_title(f"ρ∈[{edges[index]:.2f},{edges[index + 1]:.2f})　{LEVELS[index]}　n=0", fontsize=10)
            if ylim is None:
                ax.set_ylim(0, 0.06)
                ax.set_yticks(np.arange(0, 0.061, 0.01))
            else:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_xlim(0, xmax)
            ax.text(0.5, 0.5, "尚无样本", transform=ax.transAxes, ha="center", va="center", color="0.5")
            stats.append({"n": 0, "head": float("nan"), "tail": float("nan"), "median": 0.0})
        else:
            stats.append(
                draw_interval(
                    ax, part, edges[index], edges[index + 1], LEVELS[index], args.floor_step, xmax, metric, ylim
                )
            )
        if index >= 3:
            ax.set_xlabel("训练步数")
        if index % 3 == 0:
            ax.set_ylabel("loss（单微批）" if metric == "loss" else "PSNR（dB）")

    draw_summary(fig.add_subplot(grid[1, 2]), stats, edges, metric)

    last_step = int(data["step"].max())
    stamp = dt.datetime.fromtimestamp(args.log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    tag = args.tag or args.log.stem
    floor_note = "；灰虚线 = step 16K（此后全区间随机）" if args.floor_step else ""
    what = "损失" if metric == "loss" else "PSNR"
    fig.suptitle(
        f"{tag} {what}按保护率 ρ 分区间；截至 step {last_step:,}（{stamp}）{floor_note}",
        fontsize=14,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w") as handle:
        handle.write("step,loss,psnr,rho,bucket\n")
        for i in range(data["step"].size):
            handle.write(
                f"{data['step'][i]},{data['loss'][i]:.6g},{data['psnr'][i]:.6g},{data['rho'][i]:.6g},{bucket[i]}\n"
            )
    print(f"wrote {args.out} and {csv_path}")
    digits = 5 if metric == "loss" else 2
    for index in range(5):
        item = stats[index]
        print(
            f"[{edges[index]:.2f},{edges[index + 1]:.2f}) n={item['n']:5d} "
            f"median={item['median']:.{digits}f} head={item['head']:.{digits}f} tail={item['tail']:.{digits}f}"
        )


if __name__ == "__main__":
    main()
