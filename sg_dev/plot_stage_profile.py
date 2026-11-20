#!/usr/bin/env python3
"""FlashGS-style stage profile for the feed-forward pipeline.

Two panels:

  left   per-stage share of one forward pass, one stacked bar per run
  right  bytes of the representation itself (floats per view, payload per view)

The left panel must only compare runs made on the SAME device: the profile is
wall clock, so numbers from different cards are not comparable.

Per-frame cost uses sum_s * 1000 / n_frames (milliseconds), not mean_ms: raster and sg_color are
called once per view (100 calls per frame with 16 context + 4 target views),
so their mean is per call while backbone's is per frame.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGES = [
    ("backbone", "Backbone (ViT)", "#4C72B0"),
    ("depth_head", "Depth head", "#55A868"),
    ("gs_head", "GS/SG param head", "#C44E52"),
    ("alloc", "Budget allocation", "#8172B2"),
    ("sg_color", "SG colour eval", "#CCB974"),
    ("raster", "Rasterisation", "#64B5CD"),
]
FLOATS_PER_POINT = 14
FLOATS_PER_LOBE = 7


def load(path: str) -> dict | None:
    with open(path) as fh:
        payload = json.load(fh)
    n = payload.get("count", {}).get("backbone", 0)
    if not n:
        print(f"[skip] {path}: no samples")
        return None
    return payload


def per_frame(payload: dict) -> dict[str, float]:
    n = payload["count"]["backbone"]
    return {name: payload["sum_s"].get(name, 0.0) * 1000.0 / n for name, _, _ in STAGES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data14T/xz/figs/prof")
    ap.add_argument("--order", default="sg3_4,sg3_34,eco_40")
    ap.add_argument("--labels", default="", help="tag=label,tag=label")
    ap.add_argument("--floats", default="", help="tag=floats_per_view overrides for runs without marks")
    ap.add_argument("--out", default="figs/stage_profile.png")
    args = ap.parse_args()

    labels = {}
    for item in filter(None, args.labels.split(",")):
        tag, _, text = item.partition("=")
        labels[tag] = text

    float_overrides = {}
    for item in filter(None, args.floats.split(",")):
        tag, _, text = item.partition("=")
        float_overrides[tag] = float(text)

    runs = []
    for tag in args.order.split(","):
        path = os.path.join(args.dir, f"{tag}.json")
        if not os.path.exists(path):
            print(f"[skip] {path}: missing")
            continue
        payload = load(path)
        if payload is None:
            continue
        runs.append((tag, labels.get(tag, tag), payload))

    if not runs:
        raise SystemExit("no profile files with samples")

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    # ---- left: stage shares -------------------------------------------------
    y = np.arange(len(runs))[::-1]
    for yi, (tag, label, payload) in zip(y, runs):
        cost = per_frame(payload)
        total = sum(cost.values())
        left = 0.0
        for name, pretty, colour in STAGES:
            frac = 100.0 * cost[name] / total
            if frac < 0.4:
                left += frac
                continue
            ax_l.barh(yi, frac, left=left, height=0.58, color=colour,
                      edgecolor="white", linewidth=0.8)
            if frac > 5.0:
                ax_l.text(left + frac / 2, yi, f"{frac:.0f}%", ha="center",
                          va="center", fontsize=8.5, color="white", weight="bold")
            left += frac
        ax_l.text(101.5, yi, f"{total:.0f} ms", va="center", fontsize=8.5, color="#333")

    ax_l.set_yticks(y)
    ax_l.set_yticklabels([label for _, label, _ in runs], fontsize=9)
    ax_l.set_xlim(0, 112)
    ax_l.set_xlabel("share of one forward pass (%)", fontsize=9)
    ax_l.set_title("Stage breakdown (same device)", fontsize=10, loc="left")
    ax_l.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax_l.set_axisbelow(True)
    for side in ("top", "right"):
        ax_l.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STAGES]
    ax_l.legend(handles, [p for _, p, _ in STAGES], fontsize=7.8, ncol=3,
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))

    # ---- right: representation bytes ---------------------------------------
    widths = 0.34
    xs = np.arange(len(runs))
    floats = []
    for tag, _, payload in runs:
        marks = payload.get("marks", {})
        n_points = marks.get("points_per_view", 0.0)
        n_lobes = marks.get("lobes_per_view", 0.0)
        value = FLOATS_PER_POINT * n_points + FLOATS_PER_LOBE * n_lobes
        floats.append(float_overrides.get(tag, value))
    if any(f > 0 for f in floats):
        mb = [f * 4 / 1e6 for f in floats]
        ax_r.bar(xs - widths / 2, mb, width=widths, color="#4C72B0", label="payload (MB)")
        ax_r.set_ylabel("payload per view (MB)", fontsize=9)
        for x, v in zip(xs, mb):
            ax_r.text(x - widths / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([label for _, label, _ in runs], fontsize=9)
    ax_r.set_title("Representation cost", fontsize=10, loc="left")
    ax_r.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax_r.set_axisbelow(True)
    for side in ("top", "right"):
        ax_r.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print("wrote", args.out)
    for tag, _, payload in runs:
        cost = per_frame(payload)
        total = sum(cost.values())
        line = ", ".join(f"{k}={v:.1f}({100*v/total:.1f}%)" for k, v in cost.items())
        print(f"{tag}: total {total:.1f} ms | {line}")


if __name__ == "__main__":
    main()
