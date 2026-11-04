"""Export loss / PSNR curves from a stage-1 (PGT) training log.

The logger prints one line per micro-batch, so every optimizer step appears
``accumulate_grad_batches`` times; lines are grouped by step and averaged before
smoothing.  Parsing is tolerant: only ``train step``, ``loss`` and ``psnr`` are
required, the guard / memory fields are picked up when present.

Args:
    log: path to the training log.
    --out: output PNG path; a sibling ``.csv`` with the per-step aggregates is
        written next to it.
    --smooth: rolling window in optimizer steps, default 25.
    --max-step: ignore steps beyond this value (default: all).

Returns:
    None.  Prints a one-line JSON summary to stdout.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_RE = re.compile(r"train step (\d+)")
KV_RE = re.compile(r"(\w+) = (-?[\d.eE+]+|None)")
TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_log(path, max_step=None):
    """Collect per-line records as a dict of numpy arrays keyed by step."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            kv = dict(KV_RE.findall(line))
            if "loss" not in kv or "psnr" not in kv:
                continue
            t = TIME_RE.match(line)
            rows.append(
                {
                    "step": step,
                    "loss": float(kv["loss"]),
                    "psnr": float(kv["psnr"]),
                    "grad_norm": float(kv["grad_norm"]) if "grad_norm" in kv else np.nan,
                    "skipped": float(kv["skipped"]) if "skipped" in kv else np.nan,
                    "peak_mem": float(kv["peak_mem"]) if "peak_mem" in kv else np.nan,
                    "sec": _to_seconds(t.group(1)) if t else np.nan,
                }
            )
    if not rows:
        raise SystemExit(f"no train lines found in {path}")
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def _to_seconds(ts):
    """Seconds since the first log line, used only for the throughput readout."""
    from datetime import datetime

    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()


def aggregate(raw):
    """Average the micro-batch lines that share one optimizer step."""
    steps = np.unique(raw["step"]).astype(int)
    out = {"step": steps}
    for key in ("loss", "psnr", "grad_norm", "peak_mem", "sec"):
        vals = np.full(len(steps), np.nan)
        for i, s in enumerate(steps):
            m = raw["step"] == s
            vals[i] = np.nanmean(raw[key][m])
        out[key] = vals
    sk = np.full(len(steps), np.nan)
    for i, s in enumerate(steps):
        m = raw["step"] == s
        sk[i] = np.nanmax(raw["skipped"][m])
    out["skipped"] = sk
    return out


def rolling(x, w, how="mean"):
    """Centred rolling statistic that keeps the array length."""
    if w <= 1:
        return x
    if how == "median":
        return np.array([np.nanmedian(x[max(0, i - w + 1) : i + 1]) for i in range(len(x))])
    c = np.convolve(np.nan_to_num(x, nan=0.0), np.ones(w) / w, mode="full")
    valid = np.convolve(~np.isnan(x), np.ones(w), mode="full")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = c / valid
    return out[: len(x)]


def plot(agg, out_png, smooth):
    """Draw loss, PSNR and guard panels; the third panel is skipped when empty."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.6]})
    step = agg["step"]

    ax = axes[0]
    ax.plot(step, agg["loss"], color="tab:blue", lw=0.5, alpha=0.18)
    ax.plot(step, rolling(agg["loss"], smooth), color="tab:blue", lw=1.8, label=f"mean({smooth})")
    ax.set_yscale("log")
    ax.set_ylabel("loss (mse + 0.05*lpips + 0.001*reproj)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(step, agg["psnr"], color="tab:green", lw=0.5, alpha=0.18)
    ax.plot(step, rolling(agg["psnr"], smooth, "median"), color="tab:green", lw=1.8, label=f"median({smooth})")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    if np.isfinite(agg["grad_norm"]).any():
        ax.plot(step, agg["grad_norm"], color="tab:orange", lw=0.5, alpha=0.25)
        ax.plot(step, rolling(agg["grad_norm"], smooth), color="tab:orange", lw=1.6, label="grad_norm")
        ax.set_ylabel("grad norm")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        if np.isfinite(agg["skipped"]).any():
            ax2 = ax.twinx()
            ax2.step(step, agg["skipped"], color="tab:red", lw=1.2, where="post", label="skipped (cum)")
            ax2.set_ylabel("skipped updates", color="tab:red")
            ax2.legend(loc="upper left", fontsize=8)
    else:
        ax.axis("off")
    ax.set_xlabel("optimizer step")

    n = len(step)
    fig.suptitle(f"PGT stage-1 (SG-3, 10 views, full parameters) — {n} steps, {step[-1]} logged", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--out", default="figs/pgt_curves/curve.png")
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--max-step", type=int, default=None)
    args = ap.parse_args()

    raw = parse_log(args.log, args.max_step)
    agg = aggregate(raw)
    out_png = Path(args.out)
    plot(agg, out_png, args.smooth)
    out_csv = out_png.with_suffix(".csv")
    np.savetxt(
        out_csv,
        np.column_stack([agg[k] for k in ("step", "loss", "psnr", "grad_norm", "skipped", "peak_mem")]),
        delimiter=",",
        header="step,loss,psnr,grad_norm,skipped,peak_mem",
        comments="",
        fmt="%.6g",
    )

    step = agg["step"]
    head = step < step[0] + 50
    tail = step > step[-1] - 50
    sec = agg["sec"]
    rate = (sec[-1] - sec[0]) / max(1, step[-1] - step[0])
    summary = {
        "steps": int(step[-1]),
        "lines": int(len(raw["step"])),
        "loss_first50": round(float(np.nanmean(agg["loss"][head])), 4),
        "loss_last50": round(float(np.nanmean(agg["loss"][tail])), 4),
        "psnr_first50": round(float(np.nanmean(agg["psnr"][head])), 2),
        "psnr_last50": round(float(np.nanmean(agg["psnr"][tail])), 2),
        "psnr_median_last200": round(float(np.nanmedian(agg["psnr"][step > step[-1] - 200])), 2),
        "skipped_total": int(np.nanmax(agg["skipped"])) if np.isfinite(agg["skipped"]).any() else None,
        "peak_mem_gib": round(float(np.nanmax(agg["peak_mem"])), 2) if np.isfinite(agg["peak_mem"]).any() else None,
        "sec_per_step": round(float(rate), 2),
        "png": str(out_png),
        "csv": str(out_csv),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
