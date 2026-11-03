"""Small-instance optimality check for the price allocator (CPU, seconds).

The allocator is "LP greedy at a fixed price plus one integer repair", not an
exact discrete optimiser.  This script measures how far that lands from the
integer optimum on instances small enough for dynamic programming, so the
claim in the write-up can stay honest.

Run from the repo root:

    python sg_wrapper/tests/bench_price_oracle.py
"""
import torch

from sg_wrapper.budget import (
    BASE_FLOATS,
    LOBE_FLOATS,
    allocate,
    allocate_by_price,
)

NEG = float("-inf")


def _toy(P: int, L: int = 3, seed: int = 0):
    """Random instance: opacity in [0, 1), lobe values decaying with index.

    Args:
        P: number of candidate points.
        L: lobes per point.
        seed: generator seed.

    Returns:
        ``(opacity [P], D [P, L])`` in the same scale the allocator consumes.
    """
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(P, generator=g)
    d = torch.rand(P, L, generator=g) * torch.linspace(4.0, 1.0, L)
    return o, d


def _dp_optimum(o: torch.Tensor, d: torch.Tensor, cap_units: int) -> float:
    """Integer optimum by dynamic programming over the 7-float grid.

    Each point is a multiple-choice item with levels ``m = 0..L`` costing
    ``2 + m`` grid units (14 and 21..35 floats) and worth ``o + sum of top-m D``.

    Args:
        o: (P,) normalised opacity.
        d: (P, L) normalised lobe values, already sorted descending per point.
        cap_units: capacity in units of 7 floats (``F // 7``).

    Returns:
        Best attainable total value.
    """
    P, L = int(o.shape[0]), int(d.shape[1])
    prev = [NEG] * (cap_units + 1)
    prev[0] = 0.0
    for i in range(P):
        cum, run = [0.0], 0.0
        for l in range(L):
            run += float(d[i, l])
            cum.append(run)
        vals = [float(o[i]) + c for c in cum]
        cur = prev[:]                       # skip this point
        for c in range(cap_units + 1):
            base = prev[c]
            if base == NEG:
                continue
            for m in range(L + 1):
                cc = c + 2 + m
                if cc <= cap_units and base + vals[m] > cur[cc]:
                    cur[cc] = base + vals[m]
        prev = cur
    return max(prev)


def _value_of(allocation, o_hat: torch.Tensor, d_sorted: torch.Tensor) -> float:
    """Total value the allocation actually bought, under the same units."""
    m = allocation.keep_lobes.sum(dim=1)
    padded = torch.cat([torch.zeros(o_hat.numel(), 1), d_sorted.cumsum(dim=1)], 1)
    picked = o_hat + padded.gather(1, m[:, None]).squeeze(1)
    return float(picked[allocation.keep_points].sum())


def main() -> float:
    """Print the gap table and return the worst value ratio."""
    worst = 1.0
    all_r, all_f = [], []
    print(f"{'P':>4} {'frac':>5} {'F*':>7} {'price/opt':>10} {'fused/opt':>10} "
          f"{'util':>6}")
    for P in (16, 32, 64):
        for frac in (0.1, 0.2, 0.4, 0.6):
            ratios, ratios_f, utils = [], [], []
            for seed in (0, 1, 2):
                o, d = _toy(P, 3, seed)
                o_hat = o / o.max()
                d_sorted = (d / d.max()).sort(dim=1, descending=True)[0]
                full_units = P * (2 + 3)
                cap = max(2, int(round(full_units * frac)))
                F = cap * 7
                a = allocate_by_price(F, o, d)
                b = allocate(F, o, d)
                opt = _dp_optimum(o_hat, d_sorted, cap)
                vp = _value_of(a, o_hat, d_sorted)
                vf = _value_of(b, o_hat, d_sorted)
                ratios.append(vp / opt if opt > 0 else 1.0)
                ratios_f.append(vf / opt if opt > 0 else 1.0)
                utils.append(a.achieved / a.target)
            mean = sum(ratios) / len(ratios)
            mean_f = sum(ratios_f) / len(ratios_f)
            worst = min(worst, mean)
            all_r.append(mean)
            all_f.append(mean_f)
            print(f"{P:>4} {frac:>5.1f} {F:>7} {mean:>10.4f} {mean_f:>10.4f} "
                  f"{sum(utils) / len(utils):>6.3f}")
    print(f"worst price/oracle value ratio: {worst:.4f}")
    print(f"mean  price/oracle value ratio: {sum(all_r) / len(all_r):.4f}")
    print(f"mean  fused/oracle value ratio: {sum(all_f) / len(all_f):.4f}")
    print(f"per point: {BASE_FLOATS} floats, per lobe: {LOBE_FLOATS}")
    return worst


if __name__ == "__main__":
    main()
