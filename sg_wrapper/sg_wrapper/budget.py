"""Total-budget allocation: one compression-ratio knob, one ranking, exact hit.

Budget model (float32 payload per scene)

    kappa = 14 * N + 7 * L

``N`` is the number of kept Gaussians, ``L`` the number of kept spherical
lobes.  14 = position 3 + opacity 1 + scale 3 + rotation 4 + c0 3;
7 = lobe axis 3 + sharpness 1 + amplitude 3.  Every cost is a multiple of 7,
so a requested ``kappa`` is snapped to ``F = round(kappa / 7) * 7`` (residual
<= 14 bytes) and every reachable payload is also a multiple of 7.

Allocation strategy
-------------------
Each point is a multiple-choice bundle: taking it at lobe level ``m`` costs
``14 + 7m`` and yields the value

    v_i(m) = opacity_hat_i + sum_{l<=m} D_hat_il

with ``opacity_hat`` and ``D_hat`` normalised by their global maxima so that
the two axes are comparable - the normalisation is what lets one price
transfer between scenes.  Bundles are ranked by value density ``v / cost`` and
taken greedily, which respects both constraints for free: one level per point
(multiple choice) and lobes only on points that were taken (precedence).  The
remaining budget then buys next lobes of already-taken points at their
marginal density, and finally extra points at level 0; the payload therefore
lands on ``F`` or ``F - 7`` floats and the exact count is asserted at runtime.

Compared with ``allocate_two_stage`` this removes the external split ratio: the
trade-off between points and lobes is resolved by the shared ranking, and the
density at the cut is reported as ``beta`` (value per float, i.e. the shadow
price of the constraint).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .sg_prune import BASE_FLOATS, LOBE_FLOATS

GRID = LOBE_FLOATS  # both costs are multiples of 7 (base 14 = 2 * 7)


@dataclass
class Allocation:
    """Result of :func:`allocate` (all counts in float32 parameters)."""

    keep_points: Tensor      # (P,) bool
    keep_lobes: Tensor       # (P, Lmax) bool
    n_points: int
    n_lobes: int
    achieved: int            # 14 * n_points + 7 * n_lobes
    target: int              # snapped budget F
    beta: float              # value per float at the cut


def dense_floats(num_views: int, height: int, width: int,
                 lobes_per_point: int = 3) -> int:
    """Full-payload budget of the dense model (the compression-ratio baseline).

    ``lobes_per_point`` is the level the model predicts at full budget; our SG
    head uses 3, i.e. 14 + 7 * 3 = 35 float32 per point.
    """
    return (BASE_FLOATS + LOBE_FLOATS * int(lobes_per_point)) *         int(num_views) * int(height) * int(width)


def ratio_to_kappa(ratio: float, num_views: int, height: int, width: int,
                   lobes_per_point: int = 3) -> int:
    """Turn a user-facing compression ratio (e.g. 16 -> 1/16 payload) into kappa."""
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return int(round(dense_floats(num_views, height, width, lobes_per_point) / float(ratio)))


def snap_budget(kappa: int) -> int:
    """Round a byte budget down/up to the reachable 7-float grid (min one point)."""
    return max(BASE_FLOATS, int(round(int(kappa) / GRID)) * GRID)


def _normalise(x: Tensor) -> Tensor:
    x = x.clamp_min(0.0)
    return x / x.max().clamp_min(1e-12)


def allocate(kappa: int, opacity: Tensor, D: Tensor,
             lobe_max: int | None = None) -> Allocation:
    """Allocate a float budget between Gaussians and their lobes.

    Args:
        kappa: target payload in float32 parameters (per scene).
        opacity: (P,) per-point opacity score; higher keeps the point.
        D: (P, L) per-lobe dynamic range ``|a| (1 - exp(-2s))``.
        lobe_max: optional cap on lobes per point (defaults to ``D.shape[1]``).

    Returns:
        :class:`Allocation` with boolean masks, counts, achieved payload and the
        marginal density at the cut.
    """
    P = int(opacity.shape[0])
    if P == 0:
        return Allocation(opacity.new_zeros(0, dtype=torch.bool),
                          opacity.new_zeros(0, 0, dtype=torch.bool),
                          0, 0, 0, snap_budget(kappa), 0.0)

    dev = opacity.device
    L = int(D.shape[1]) if D.dim() == 2 else 0
    Lmax = L if lobe_max is None else max(0, min(int(lobe_max), L))
    F = snap_budget(kappa)

    o = _normalise(opacity.reshape(P).float())
    if Lmax > 0:
        d = D[:, :Lmax].float()
        d = _normalise(d)
        d, _ = d.sort(dim=1, descending=True)      # marginal lobes at the front
        cum = d.cumsum(dim=1)                      # (P, Lmax)
    else:
        d = o.new_zeros((P, 0))
        cum = o.new_zeros((P, 0))

    # Bundle table: level m costs 14 + 7m, value = opacity_hat + sum of top-m D.
    lv = torch.arange(Lmax + 1, device=dev)
    costs = (BASE_FLOATS + LOBE_FLOATS * lv).to(o.dtype)          # (M,)
    val = o[:, None].expand(P, Lmax + 1)
    if Lmax > 0:
        val = torch.cat([o[:, None], o[:, None] + cum], dim=1)    # (P, M)
    dens = val / costs                                            # (P, M)

    best_dens, best_m = dens.max(dim=1)
    best_cost = costs[best_m]

    # Greedy prefix over points ordered by their best bundle density.
    order = best_dens.argsort(descending=True)
    cum_cost = best_cost[order].cumsum(0)
    n_pts = int((cum_cost <= F).sum().item())
    n_pts = max(n_pts, 1)                       # F >= 14 guarantees room for one point
    sel = order[:n_pts]
    m = best_m[sel].clone()
    leftover = F - int(costs[m].sum().item())
    beta = float(best_dens[sel[-1]].item())

    # Spend the remainder on the next lobe of already-taken points.
    while leftover >= LOBE_FLOATS:
        room = (m < Lmax).nonzero(as_tuple=False).squeeze(1)
        if room.numel() == 0:
            break
        value = d[sel[room], m[room]]
        room = room[value > 0]                     # never spend on a zero-value lobe
        if room.numel() == 0:
            break
        marginal = d[sel[room], m[room]] / LOBE_FLOATS
        take = room[marginal.argsort(descending=True)[: int(leftover // LOBE_FLOATS)]]
        m[take] += 1
        leftover -= LOBE_FLOATS * int(take.numel())

    # Spend whatever is still left on extra points (level 0 bundles).
    while leftover >= BASE_FLOATS:
        taken = torch.zeros(P, dtype=torch.bool, device=dev)
        taken[sel] = True
        cand = (~taken).nonzero(as_tuple=False).squeeze(1)
        cand = cand[o[cand] > 0]                   # never spend on a zero-value point
        if cand.numel() == 0:
            break
        room = int(leftover // BASE_FLOATS)
        pick = cand[(o[cand]).argsort(descending=True)[:room]]
        sel = torch.cat([sel, pick])
        m = torch.cat([m, torch.zeros_like(pick)])
        leftover -= BASE_FLOATS * int(pick.numel())

    n_pts = int(sel.numel())
    n_lobes = int(m.sum().item())
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes
    if achieved > F or (F - achieved) % GRID != 0:
        raise AssertionError(
            f"budget mismatch: achieved={achieved} target={F} "
            f"(N={n_pts}, L={n_lobes})")

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    keep_lobes = torch.zeros((P, Lmax), dtype=torch.bool, device=dev)
    if Lmax > 0:
        keep_lobes[sel] = (torch.arange(Lmax, device=dev)[None, :] < m[:, None])

    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F, beta)


def allocate_two_stage(kappa: int, opacity: Tensor, D: Tensor,
                       split: float = 0.6) -> Allocation:
    """Two-stage baseline: split the budget, top-k points, then top-L lobes.

    ``split`` is the share of ``F`` handed to the geometry axis; the point
    budget is ``N = split * F / 14`` clamped to the feasible window, and every
    remaining float goes to lobes taken by ``D`` over the survivors.  Kept for
    the A/B comparison against :func:`allocate`.
    """
    P = int(opacity.shape[0])
    L = int(D.shape[1]) if D.dim() == 2 else 0
    F = snap_budget(kappa)
    if P == 0:
        return Allocation(opacity.new_zeros(0, dtype=torch.bool),
                          opacity.new_zeros(0, 0, dtype=torch.bool),
                          0, 0, 0, F, 0.0)

    dev = opacity.device
    lo = -(-F // (BASE_FLOATS + LOBE_FLOATS * max(L, 0))) if L > 0 else -(-F // BASE_FLOATS)
    lo = max(1, lo)
    hi = max(1, F // BASE_FLOATS)
    n_pts = int(round(split * F / BASE_FLOATS))
    n_pts = max(lo, min(hi, n_pts))
    n_lobes = (F - BASE_FLOATS * n_pts) // LOBE_FLOATS
    n_lobes = max(0, min(n_lobes, L * n_pts))

    o = _normalise(opacity.reshape(P).float())
    sel = o.argsort(descending=True)[:n_pts]

    keep_lobes = torch.zeros((P, L), dtype=torch.bool, device=dev)
    if n_lobes > 0 and L > 0:
        d = D[sel].float()
        flat = d.reshape(-1)
        take = flat.argsort(descending=True)[:n_lobes]
        rows = torch.div(take, L, rounding_mode="floor")
        cols = take % L
        keep_lobes[sel[rows], cols] = True
        n_lobes = int(keep_lobes.sum().item())

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes
    beta = float(o[sel[-1]].item()) if n_pts else 0.0
    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F, beta)
