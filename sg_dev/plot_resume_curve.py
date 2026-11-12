"""Plot a training curve stitched from several logs of the same run family.

The continuation of a run writes to a new log file while the step counter keeps
increasing, so a single-run curve has to be assembled from the pieces. This
script concatenates them, drops duplicate steps, and marks the seams.

    python sg_dev/plot_resume_curve.py --out figs/resume.png --seam 100000 \
        logs/run_a.log logs/run_b.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(r"train step (\d+).*?loss = ([^\s;]+).*?psnr = ([^\s;]+)")


def parse(paths):
    steps, loss, psnr = [], [], []
    for p in paths:
        for line in Path(p).read_text(errors="ignore").splitlines():
            m = LINE.search(line)
            if not m:
                continue
            try:
                steps.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                psnr.append(float(m.group(3)))
            except ValueError:  # nan / inf rows carry no signal
                continue
    order = np.argsort(steps, kind="stable")
    return (
        np.asarray(steps)[order],
        np.asarray(loss)[order],
        np.asarray(psnr)[order],
    )


def rolling(x, k):
    if k <= 1 or x.size < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, np.nan)
    return np.concatenate([pad, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="figs/resume_curve.png")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seam", type=int, action="append", default=None)
    ap.add_argument("--smooth", type=int, default=200)
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    step, loss, psnr = parse(args.logs)
    if step.size == 0:
        raise SystemExit("no 'train step' lines parsed")

    keep = np.ones(step.shape, bool)
    if args.xmin is not None:
        keep &= step >= args.xmin
    step, loss, psnr = step[keep], loss[keep], psnr[keep]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, y, label, color in (
        (axes[0], loss, "loss", "tab:blue"),
        (axes[1], psnr, "PSNR", "tab:green"),
    ):
        ax.plot(step, y, ".", ms=1.2, alpha=0.25, color=color)
        ax.plot(step, rolling(y, args.smooth), lw=1.6, color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        for s in args.seam or []:
            ax.axvline(s, color="tab:red", ls="--", lw=1.0)
    axes[1].set_xlabel("train step")
    lo, hi = int(step.min()), int(step.max())
    axes[0].set_title(args.title or f"steps {lo}–{hi}, {step.size} points")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)

    if args.csv:
        csv = Path(args.csv)
        csv.parent.mkdir(parents=True, exist_ok=True)
        with csv.open("w") as fh:
            fh.write("step,loss,psnr,loss_roll,psnr_roll\n")
            lr, pr = rolling(loss, args.smooth), rolling(psnr, args.smooth)
            for i in range(step.size):
                fh.write(f"{step[i]},{loss[i]:.6f},{psnr[i]:.6f},{lr[i]:.6f},{pr[i]:.6f}\n")
    print(f"wrote {out} ({step.size} points, {lo}–{hi})")


if __name__ == "__main__":
    main()
