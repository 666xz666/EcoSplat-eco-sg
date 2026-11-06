"""Summarise the per-scene budget census written during a budget-mode eval.

The census has one row per scene and rate, written by ``model_wrapper.test_step``
when ``BUDGET_CENSUS_CSV`` is set:

    scene,rate,n_points,n_lobes,achieved,target,beta,m_hist

``m_hist`` is the number of points that ended up with 0, 1, ... L lobes, so the
report answers two questions per rate: how much of the budget went to the point
axis versus the colour axis, and whether that split is stable across scenes.

Usage:
    python sg_dev/summarize_budget_census.py logs/budget_census_budget_sg3_30k.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np


def _quantiles(values: np.ndarray) -> str:
    """Format the median with a 10-90 band."""
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return f"{p50:,.0f} [{p10:,.0f}, {p90:,.0f}]"


def load(path: str) -> dict[float, list[dict]]:
    """Group the census rows by rate.

    Args:
        path: CSV path written by the eval loop.
    Returns:
        rate -> list of parsed rows.
    """
    groups: dict[float, list[dict]] = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            hist = [int(x) for x in row["m_hist"].split("|") if x != ""]
            groups[float(row["rate"])].append({
                "n_points": int(row["n_points"]),
                "n_lobes": int(row["n_lobes"]),
                "achieved": int(row["achieved"]),
                "target": int(row["target"]),
                "beta": float(row["beta"]),
                "hist": hist,
            })
    return groups


def report(groups: dict[float, list[dict]]) -> None:
    """Print one block per rate."""
    for rate in sorted(groups):
        rows = groups[rate]
        n_points = np.array([r["n_points"] for r in rows], dtype=float)
        n_lobes = np.array([r["n_lobes"] for r in rows], dtype=float)
        achieved = np.array([r["achieved"] for r in rows], dtype=float)
        target = np.array([r["target"] for r in rows], dtype=float)
        beta = np.array([r["beta"] for r in rows], dtype=float)
        hist = np.array([r["hist"] for r in rows], dtype=float)
        shares = hist / hist.sum(axis=1, keepdims=True)
        floats_per_point = np.where(n_points > 0, achieved / np.maximum(n_points, 1), 0)

        print(f"rate={rate:g}  scenes={len(rows)}  "
              f"payload={35 * rate:.2f} float/px  target={target[0]:,.0f}")
        print(f"  points      {_quantiles(n_points)}")
        print(f"  lobes       {_quantiles(n_lobes)}")
        print(f"  lobes/point {np.median(n_lobes / np.maximum(n_points, 1)):.2f} "
              f"[{np.percentile(n_lobes / np.maximum(n_points, 1), 10):.2f}, "
              f"{np.percentile(n_lobes / np.maximum(n_points, 1), 90):.2f}]")
        print(f"  float/point {np.median(floats_per_point):.2f}  "
              f"point-axis share {np.median(14 * n_points / achieved):.1%}")
        print(f"  spend       {np.median(achieved / target):.6f} of target "
              f"(min {np.min(achieved / target):.6f})")
        print(f"  beta        {np.median(beta):.3g} "
              f"[{np.percentile(beta, 10):.3g}, {np.percentile(beta, 90):.3g}]")
        mix = "  ".join(f"m={m}:{shares[:, m].mean():.1%}" for m in range(shares.shape[1]))
        print(f"  lobe mix    {mix}")
        print()


def main() -> None:
    """Entry point."""
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report(load(sys.argv[1]))


if __name__ == "__main__":
    main()
