"""Measure the run-to-run noise floor of a test rung from two identical runs.

The same checkpoint evaluated twice with the same settings is not guaranteed to
return the same numbers: the rasteriser accumulates with atomics and the
allocator breaks ties in ``topk``, both of which depend on scheduling.  Any
difference below that floor cannot be attributed to the change under test, so
the floor has to be measured before a single-switch A/B is read.

Both runs are written by ``eval_budget_sg3_rho.sh`` under
``outputs/test/<run>/<run>/scores_all.json``, a dict from scene key to
``[?, psnr, ssim, lpips]``.

    python sg_dev/noise_floor.py --a matched_sg3_50k_rho0p12282 --b rep5_rho0p12282
    python sg_dev/noise_floor.py --a X --b Y --root outputs/test --metric psnr
"""

import argparse
import json
import os

import numpy as np

# Index of each metric inside the per-scene list written by model_wrapper.
METRICS = {"psnr": 1, "ssim": 2, "lpips": 3}


def load_scores(root, run):
    """Load one run's per-scene scores.

    Args:
        root: Directory holding every run (usually ``outputs/test``).
        run: Run directory name, used twice because Lightning nests output_path.

    Returns:
        Dict from scene key to the per-scene list.
    """
    path = os.path.join(root, run, run, "scores_all.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as handle:
        return json.load(handle)


def report(a, b, metric="psnr", root="outputs/test"):
    """Print the paired distribution of a metric between two identical runs.

    Args:
        a: First run name.
        b: Second run name.
        metric: One of ``psnr``, ``ssim``, ``lpips``.
        root: Directory holding every run.

    Returns:
        None
    """
    idx = METRICS[metric]
    sa = load_scores(root, a)
    sb = load_scores(root, b)
    common = sorted(set(sa) & set(sb))
    if not common:
        print(f"[noise] no shared scenes between {a} and {b}")
        return
    delta = np.array([sa[s][idx] - sb[s][idx] for s in common], dtype=np.float64)
    n = len(delta)
    mean = float(delta.mean())
    std = float(delta.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else 0.0
    # A paired t interval is the quantity a single-switch A/B is compared with.
    half = 1.96 * se
    order = np.argsort(delta)
    worst = [(common[i], round(float(delta[i]), 4)) for i in order[:3]]
    best = [(common[i], round(float(delta[i]), 4)) for i in order[-3:]]
    exact = int((np.abs(delta) < 1e-9).sum())
    print(
        f"[noise] {metric} {a} - {b}  n={n}  "
        f"mean={mean:+.4f} sd={std:.4f} se={se:.4f} t={t:+.2f} "
        f"ci95=[{mean - half:+.4f}, {mean + half:+.4f}]"
    )
    print(
        f"        identical {exact}/{n} ({100.0 * exact / n:.1f}%)  "
        f"|d|>0.05 in {100.0 * float((np.abs(delta) > 0.05).mean()):.1f}%"
    )
    print(f"        min {worst}  max {best}")


def main():
    """Parse arguments and print one report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="first run directory name")
    parser.add_argument("--b", required=True, help="second run directory name")
    parser.add_argument("--root", default="outputs/test", help="parent of both runs")
    parser.add_argument(
        "--metric", default="psnr", choices=sorted(METRICS), help="metric to compare"
    )
    args = parser.parse_args()
    report(args.a, args.b, metric=args.metric, root=args.root)


if __name__ == "__main__":
    main()
