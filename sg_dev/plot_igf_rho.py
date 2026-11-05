#!/usr/bin/env python3
"""Bin an IGF training log by ``protect_rate`` and plot the loss per bin.

The PLGC sampler draws a fresh protect rate every step, so a raw loss curve mixes
two effects: the model getting better and the retained fraction changing.  Binning
by ``protect_rate`` separates them — each bin is a narrow slice of the rho axis,
and the trend inside a bin is training progress at (roughly) fixed budget.

Usage::

    python sg_dev/plot_igf_rho.py logs/igf_sg3_10v.latest.log \
        --other logs/igf_sh4_10v.latest.log:SH4 \
        --out figs/igf_rho/sg3_loss_by_rho.png

Outputs the figure plus a CSV next to it (one row per logged step).
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINE = re.compile(
    r"train step (\d+).*?loss = ([-\d.eE+]+); psnr = ([-\d.eE+]+); "
    r"protect_rate = ([-\d.eE+]+)"
)


def load(path: str) -> dict:
    """Parse one log into step/loss/psnr/rho arrays.

    Args:
        path: training log written by ``train_pgt_bg.sh``.

    Returns:
        Dict of numpy arrays; empty arrays when the log has no step lines yet.
    """
    steps, losses, psnrs, rhos = [], [], [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = LINE.search(line)
        if match is None:
            continue
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        psnrs.append(float(match.group(3)))
        rhos.append(float(match.group(4)))
    order = np.argsort(steps)
    return {
        "step": np.asarray(steps, dtype=np.int64)[order],
        "loss": np.asarray(losses)[order],
        "psnr": np.asarray(psnrs)[order],
        "rho": np.asarray(rhos)[order],
    }


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, so a single bad step does not define the curve."""
    if window <= 1 or values.size < window:
        return values.copy()
    half = window // 2
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def bin_index(rho: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map rho to its bucket, clipping the top bucket instead of dropping it."""
    idx = np.digitize(rho, edges) - 1
    return np.clip(idx, 0, edges.size - 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--other", default="", help="PATH[:LABEL] for a second arm")
    parser.add_argument("--width", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument(
        "--no-common",
        action="store_true",
        help="compare arms over their own full logs instead of the shared step window",
    )
    parser.add_argument("--out", default="figs/igf_rho/loss_by_rho.png")
    args = parser.parse_args()

    # The PLGC sampler lives in [plgc_floor, plgc_max] = [0.05, 0.95]; anything
    # above the last edge is clipped into it rather than dropped.
    edges = np.arange(0.05, 0.95 + 1e-9, args.width)
    data = load(args.log)
    if data["step"].size == 0:
        raise SystemExit(f"no step lines in {args.log} yet")

    other = None
    if args.other:
        path, _, label = args.other.partition(":")
        other = load(path)
        other["label"] = label or Path(path).stem
        if other["step"].size == 0:
            other = None

    # A rho-binned loss is still partly a step effect: a longer run has simply
    # seen more data. When two arms are compared, cut both to the shared step
    # window so the only difference left is the head.
    common = None
    if other is not None and not args.no_common:
        common = int(min(data["step"].max(), other["step"].max()))
        data = {k: v[data["step"] <= common] for k, v in data.items()}
        other = {**{k: v[other["step"] <= common] for k, v in other.items() if k != "label"},
                 "label": other["label"]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Step 1/3: loss and PSNR over training, each bucket its own colour, so a
    # vertical slice reads as "what the model did at this budget".  Colours are
    # spread over the populated buckets only — with 18 edges but four live bins
    # a plain colormap would paint them all the same yellow.
    idx = bin_index(data["rho"], edges)
    n_bins = edges.size - 1
    populated = [b for b in range(n_bins) if (idx == b).any()]
    rank = {b: k for k, b in enumerate(populated)}
    cmap = plt.get_cmap("turbo")
    summary = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        colour = cmap(0.08 + 0.84 * rank[b] / max(1, len(populated) - 1))
        for ax, key in zip(axes[:2], ("loss", "psnr")):
            ax.plot(
                data["step"][mask],
                rolling_median(data[key][mask], args.window),
                color=colour,
                lw=1.2,
                label=f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
            )
        loss_b = data["loss"][mask]
        psnr_b = data["psnr"][mask]
        third = max(1, loss_b.size // 3)
        summary.append(
            {
                "rho_bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
                "n": int(mask.sum()),
                "step_min": int(data["step"][mask].min()),
                "step_max": int(data["step"][mask].max()),
                "loss_median": float(np.median(loss_b)),
                "loss_first_third": float(np.median(loss_b[:third])),
                "loss_last_third": float(np.median(loss_b[-third:])),
                "psnr_median": float(np.median(psnr_b)),
            }
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("training loss (rolling median)")
    axes[0].set_title(f"loss by protect-rate bin — {Path(args.log).stem}")
    axes[0].legend(fontsize=6, ncol=2, title="rho", title_fontsize=6)
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("PSNR (dB, rolling median)")
    axes[1].set_title("PSNR by protect-rate bin")

    # Step 2/3: the same information collapsed onto the rho axis, so the two
    # arms can be compared at equal budget without any step-axis confound.
    centres = (edges[:-1] + edges[1:]) / 2
    med = [np.nan] * n_bins
    for row in summary:
        lo, hi = row["rho_bin"].split("-")
        b = int(np.argmin(np.abs(centres - (float(lo) + float(hi)) / 2)))
        med[b] = row["loss_median"]
    axes[2].plot(centres, med, "o-", color="tab:blue", label=Path(args.log).stem)
    if other is not None:
        o_idx = bin_index(other["rho"], edges)
        o_med = [np.nan] * n_bins
        for b in range(n_bins):
            mask = o_idx == b
            if mask.any():
                o_med[b] = float(np.median(other["loss"][mask]))
        axes[2].plot(centres, o_med, "s--", color="tab:orange", label=other["label"])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("protect rate rho")
    axes[2].set_ylabel("median training loss")
    window_note = f"steps <= {common}" if common is not None else "full logs"
    axes[2].set_title(f"loss vs rho (median per bin, {window_note})")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"figure: {out}")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "psnr", "rho", "rho_bin"])
        for i in range(data["step"].size):
            b = idx[i]
            writer.writerow(
                [
                    int(data["step"][i]),
                    f"{data['loss'][i]:.6f}",
                    f"{data['psnr'][i]:.4f}",
                    f"{data['rho'][i]:.6f}",
                    f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
                ]
            )
    print(f"csv   : {csv_path}")
    print(json.dumps({"log": args.log, "bins": summary}, indent=1))


if __name__ == "__main__":
    main()
